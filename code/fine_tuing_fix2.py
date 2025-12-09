#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')

get_ipython().system('pip install evaluate rouge_score -q')

import os
import torch
import numpy as np
import json
import evaluate
import matplotlib.pyplot as plt
import re

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
)

# matplotlib 백엔드 설정 (노트북 환경이 아닌 경우 오류 방지)
plt.switch_backend('Agg')
rouge = evaluate.load("rouge")

# --------------------------------------------------------------------------------
# 1. 경로 및 환경 설정 (유지)
# --------------------------------------------------------------------------------
BASE_PATH = "/content/drive/MyDrive/textanl"
TRAIN_DATA_FILE = os.path.join(BASE_PATH, "dataset/train_instruction_corpus.jsonl")
VALID_DATA_FILE = os.path.join(BASE_PATH, "dataset/valid_instruction_corpus.jsonl")
OUTPUT_DIR = os.path.join(BASE_PATH, "qa")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.path.join(BASE_PATH, "dapt_params/qwen2.5_1.5b_dapt_adapter_bf16")

MAX_LENGTH = 512
PROMPT_SEPARATOR_WITH_NEWLINE = "### Assistant:\n" # 마스킹 기준 구분자

# --------------------------------------------------------------------------------
# 2. DAPT된 베이스 모델 + 기존 LoRA 로드
# --------------------------------------------------------------------------------

# ✅ 토크나이저는 베이스 모델에서
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    fix_mistral_regex=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 1) 순정 Qwen2.5 베이스 로드
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,   # bf16 쓰고 싶으면 여기랑 TrainingArguments 맞추면 됨
    trust_remote_code=True,
)

# 2) 기존에 학습한 LoRA 어댑터 로드 (이걸 그대로 이어서 학습)
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=True,
)

model.config.use_cache = False
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# --------------------------------------------------------------------------------
# 4. 데이터셋 로드 및 토크나이징 (📌 마스킹 및 클리닝 강화)
# --------------------------------------------------------------------------------
raw_datasets = load_dataset(
    "json",
    data_files={"train": TRAIN_DATA_FILE, "validation": VALID_DATA_FILE}
)

def clean_input_text(text):
    """
    Input 텍스트 내의 모든 개행 문자(\n)와 연속된 공백을 단일 공백으로 치환합니다.
    """
    if not isinstance(text, str):
        return ""
    # 📌 \n, \t, 연속 공백을 모두 단일 공백으로 치환
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_mask_function(examples):
    full_text = examples["text"]

    # 1. 텍스트 클리닝 적용: \n 문자를 제거하여 프롬프트를 정돈
    full_text = clean_input_text(full_text)

    # 2. 시퀀스 토큰화
    tokenized = tokenizer(
        full_text, truncation=True, max_length=MAX_LENGTH, padding=False,
    )
    labels = tokenized["input_ids"].copy()

    # --- Masking Logic: 프롬프트 부분의 labels를 -100으로 설정 ---
    # 구분자도 클리닝되었을 수 있으므로, 클리닝된 텍스트에서 구분자를 찾습니다.
    # PROMPT_SEPARATOR_WITH_NEWLINE ("### Assistant:\n")의 \n은 클리닝에 의해 공백으로 바뀝니다.
    CLEANED_SEPARATOR = clean_input_text(PROMPT_SEPARATOR_WITH_NEWLINE) # "### Assistant:"

    parts = full_text.split(CLEANED_SEPARATOR, 1)

    if len(parts) == 2:
        prompt_plus_header = parts[0] + CLEANED_SEPARATOR

        # 프롬프트 + 헤더를 토큰화하여 길이를 정확히 계산
        # add_special_tokens=False를 사용하여 토큰 개수 계산
        prompt_tokens = tokenizer(
            prompt_plus_header,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # 프롬프트 길이만큼 -100으로 마스킹
        if prompt_len < len(labels):
            labels[:prompt_len] = [-100] * prompt_len
        else:
            labels[:] = [-100] * len(labels)
    else:
        labels[:] = [-100] * len(labels)

    tokenized["labels"] = labels
    return tokenized

print(f"[-] Tokenizing dataset and applying label masking (Input Cleaned)...")
tokenized_datasets = raw_datasets.map(
    tokenize_and_mask_function, batched=False, remove_columns=["text"],
)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]


# --------------------------------------------------------------------------------
# 5. 학습 설정 (TrainingArguments)
# --------------------------------------------------------------------------------

# ROUGE 필터링 함수 (평가 시 순수 답변 추출용)
def filter_generated_text(text, separator):
    # separator는 클리닝된 "### Assistant:" 입니다.
    CLEANED_SEPARATOR = re.sub(r'\s+', ' ', separator).strip()

    if CLEANED_SEPARATOR in text:
        filtered_text = text.split(CLEANED_SEPARATOR, 1)[1].strip()
        filtered_text = filtered_text.replace("### Human:", "").strip()
        filtered_text = filtered_text.replace("### Assistant:", "").strip()
        return filtered_text
    return ""

# compute_metrics 함수 정의 (Loss와 ROUGE-L 계산)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    global tokenizer
    global rouge

    predictions = np.array(predictions)

    # 🔹 logits(T,B,V)로 들어오면 argmax, 이미 ids(B,T)이면 그대로 사용
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds_filtered = [filter_generated_text(pred, PROMPT_SEPARATOR_WITH_NEWLINE) for pred in decoded_preds]
    decoded_labels_filtered = [filter_generated_text(label, PROMPT_SEPARATOR_WITH_NEWLINE) for label in decoded_labels]

    rouge_results = rouge.compute(
        predictions=decoded_preds_filtered,
        references=decoded_labels_filtered
    )
    return {"rougeL": rouge_results["rougeL"]}



training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "qwen_r32_ift_checkpoints"),

    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    num_train_epochs=3,

    logging_strategy="steps",
    logging_steps=1000,             # 🔹 여기서 train loss 찍힘

    fp16=True,
    bf16=False,

    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,

    do_eval=True,
    eval_strategy="steps",          # 🔹 스텝 단위로 val loss 계산
    eval_steps=1000,                # 🔹 1000 step마다 eval_loss 찍힘

    save_strategy="steps",
    save_steps=1000,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 🔹 이제 기준은 eval_loss
    greater_is_better=False,

    logging_dir=os.path.join(OUTPUT_DIR, "qwen_r32_ift_checkpoints", "logs"),
    report_to="none",
    gradient_checkpointing=False,
)

from dataclasses import dataclass
from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForCausalLMWithMaskedLabels:
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) labels 따로 빼두기
        labels = [f["labels"] for f in features]

        # 2) tokenizer.pad에 넘길 때는 labels 빼고 넘기기
        features_no_labels = []
        for f in features:
            f = dict(f)         # shallow copy
            f.pop("labels")     # labels 제거
            features_no_labels.append(f)

        # 3) input_ids / attention_mask 패딩
        batch = self.tokenizer.pad(
            features_no_labels,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 4) labels도 같은 길이로 -100 패딩
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full(
            (len(labels), max_len),
            -100,
            dtype=torch.long,
        )

        for i, l in enumerate(labels):
            l = l[:max_len]
            padded_labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)

        batch["labels"] = padded_labels
        return batch

data_collator = DataCollatorForCausalLMWithMaskedLabels(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=None,
)

# --------------------------------------------------------------------------------
# 6. 학습 실행 및 저장
# --------------------------------------------------------------------------------
print("---파인튜닝 시작--- (r=32, Input Cleaned, Logging ROUGE-L)")
trainer.train()

save_path = os.path.join(OUTPUT_DIR, "qa_params")
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"[-] IFT Model saved to: {save_path}")

# --------------------------------------------------------------------------------
# 7. 학습 로그 저장 (시각화 데이터를 위한 추출)
# --------------------------------------------------------------------------------
print("\n[--- Saving Training and Evaluation Logs 저장중---]")

log_history = trainer.state.log_history
log_output_path = os.path.join(OUTPUT_DIR, "qwen_r32_ift_metrics_log.json")

with open(log_output_path, 'w', encoding='utf-8') as f:
    json.dump(log_history, f, ensure_ascii=False, indent=4)

print(f"✅ 학습/평가 로그가 성공적으로 저장완료: {log_output_path}")

print("\n로그 활용 가이드]")
print(f"저장된 {os.path.basename(log_output_path)} 파일을 이용하여 'loss', 'eval_loss', 'eval_rougeL' 값을 추출하여 시각화 가능.")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from torch.utils.data import DataLoader
import numpy as np

print("\n[--- 최종 검증: validation 셋 ROUGE-L 한 번 계산 ---]")

model.eval()
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,           # 안전하게 1
    shuffle=False,
    collate_fn=data_collator,
)

all_preds_text = []
all_labels_text = []

for batch in eval_loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits          # (1, L, V)

    # (1, L) → (L,)
    pred_ids  = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    label_ids = batch["labels"].cpu().numpy()[0]

    # 🔹 답변 토큰 위치만 사용 (프롬프트 / 헤더는 이미 -100)
    mask = label_ids != -100
    if not np.any(mask):
        continue  # 혹시 전부 -100인 이상한 샘플 있으면 스킵

    pred_ans_ids  = pred_ids[mask]
    label_ans_ids = label_ids[mask]

    # 🔹 label의 -100은 이미 마스크로 걸렀으니 추가 치환 필요 없음
    #    (혹시 안전하게 하고 싶으면 아래처럼 한 번 더)
    # label_ans_ids = np.where(label_ans_ids != -100, label_ans_ids, tokenizer.pad_token_id)

    pred_text  = tokenizer.decode(pred_ans_ids,  skip_special_tokens=True)
    label_text = tokenizer.decode(label_ans_ids, skip_special_tokens=True)

    # 🔹 여기서는 더 이상 filter_generated_text 쓰지 말 것
    all_preds_text.append(pred_text.strip())
    all_labels_text.append(label_text.strip())

# 🔹 ROUGE 계산
final_metrics = rouge.compute(
    predictions=all_preds_text,
    references=all_labels_text,
)

print(f"✅ 최종 ROUGE-L: {final_metrics['rougeL']:.4f}")


# In[ ]:




