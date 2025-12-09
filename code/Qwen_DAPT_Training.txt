from google.colab import drive
drive.mount('/content/drive')

import os
import torch
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
    LoraConfig,
    get_peft_model,
)

# --------------------------------------------------------------------------------
# 1. 경로 및 환경 설정
# --------------------------------------------------------------------------------
BASE_PATH = "/content/drive/MyDrive/textanl"
DATA_PATH = os.path.join(BASE_PATH, "dataset/df.jsonl")
OUTPUT_DIR = BASE_PATH

# FP8 말고 일반 인스트럭트 버전 사용 (양자화 안 쓰는 세팅에 더 적합)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"[-] Model ID: {MODEL_ID}")
print(f"[-] Mode: No quantization (pure BF16 full model) + LoRA")

# 약간의 성능 튜닝 (선택)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --------------------------------------------------------------------------------
# 2. Config 로드 (혹시 남아 있을 수 있는 quantization_config 정리)
# --------------------------------------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

# 안전용 방어 로직 (일반 인스트럭트 모델이면 보통 필요 없지만, 있어도 무해)
if hasattr(config, "quantization_config"):
    del config.quantization_config
    print("[-] Removed existing quantization_config from config.")

# --------------------------------------------------------------------------------
# 3. 모델 및 토크나이저 로드 (BF16, 비양자화)
# --------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    device_map="auto",                # 단일 GPU면 자동으로 그쪽에 실림
    torch_dtype=torch.bfloat16,       # A100 BF16 사용
    trust_remote_code=True,
)

# DDP/GC와 호환 위해 캐시 비활성화
model.config.use_cache = False
model.gradient_checkpointing_enable()

# --------------------------------------------------------------------------------
# 4. LoRA 어댑터 설정
# --------------------------------------------------------------------------------
# 양자화는 안 쓰지만, 4B 전체 full-finetune은 굳이 안 해도 되므로 LoRA는 유지
# 조금 더 보수적으로 가려면 r 값을 32 정도로 줄여도 됨.
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --------------------------------------------------------------------------------
# 5. 데이터셋 로드 및 토크나이징
# --------------------------------------------------------------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )

print("[-] Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# --------------------------------------------------------------------------------
# 6. 학습 설정 (양자화 없는 BF16 기준)
# --------------------------------------------------------------------------------
# 양자화 안 쓰면 VRAM 부담이 커지니까 배치는 보수적으로 시작하는 게 안전함.
# A100 80GB 기준이면 아래 값 정도는 무난한 편 (seq_len=2048 기준):
#   - per_device_train_batch_size = 4
#   - gradient_accumulation_steps = 4  -> effective batch = 16
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "qwen3_checkpoints_bf16_lora"),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,                  # DAPT + LoRA에는 이 정도도 가능
    num_train_epochs=1,
    logging_steps=1000,
    fp16=False,
    bf16=True,                           # BF16 사용
    optim="adamw_torch",                 # bnb 의존성 없는 기본 AdamW
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_strategy="epoch",
    report_to="none",
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets,
    args=training_args,
    data_collator=data_collator,
)

# --------------------------------------------------------------------------------
# 7. 학습 실행 및 저장
# --------------------------------------------------------------------------------
print("[-] Starting Training (no quantization, BF16 + LoRA)...")
trainer.train()

save_path = os.path.join(OUTPUT_DIR, "qwen2.5_1.5b_dapt_adapter_bf16")
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"[-] Model saved to: {save_path}")


