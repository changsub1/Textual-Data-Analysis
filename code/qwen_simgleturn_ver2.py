#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import re
import numpy as np
from peft import PeftModel 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

#  Qwen 모델 전용 토크나이저 클래스를 직접 임포트 
try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as AutoTokenizer
except ImportError:
    from transformers import AutoTokenizer

# ----------------- 1. 경로 및 환경 설정 -----------------
BASE_PATH = r"C:\Users\USER\Study\2025_2\2025_2_DS_Text"  # 경로 변경 -->
ADAPTER_PATH = os.path.join(BASE_PATH, "local_adapters", "qwen2.5_1.5b_dapt_adapter_bf16") 

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 384 # 넉넉한 답변 길이 확보
PROMPT_SEPARATOR_HUMAN = "### Human:\n"
PROMPT_SEPARATOR_ASSISTANT = "### Assistant:\n"
KOREAN_SENTENCE_TERMINATORS = ['.', '?', '!', '다', '요']


# ----------------- 2. 모델 로드 및 어댑터 병합 -----------------
print("[-] Loading base model and merging adapter (FP16)...")

# 1. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# 2. 베이스 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16, 
    trust_remote_code=True,
)

# 3. LoRA 어댑터 로드 및 병합
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload() 

model.eval()
device = model.device
print("Model loaded and ready for inference.")

# ----------------- 3. Stopping Criteria 정의 -----------------
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        self.stops = stops if stops is not None else []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if input_ids.shape[1] >= len(stop) and torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False

# ### Human: 태그를 토큰화하여 stop word로 정의
stop_word_ids = [tokenizer(PROMPT_SEPARATOR_HUMAN, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze()]
stop_word_ids.append(torch.tensor([tokenizer.eos_token_id], dtype=torch.long))

stop_word_ids = [stop.to(device) for stop in stop_word_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_word_ids)])


# ----------------- 4. generate_response 함수 정의 -----------------

def generate_response(prompt):
    """토큰화, 추론 및 디코딩을 수행하고 문장 완결성을 확보하는 함수"""

    cleaned_prompt = re.sub(r'\s+', ' ', prompt).strip()

    inputs = tokenizer(cleaned_prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    REPETITION_PENALTY = 1.2 # 반복 페널티

    with torch.no_grad():
        output_ids = model.generate( 
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,             
            repetition_penalty=REPETITION_PENALTY,
            stopping_criteria=stopping_criteria, 
        )

    generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    # 1차 필터링: ### Human: 이전에 생성된 모든 내용을 제거
    generated_text = generated_text.split(PROMPT_SEPARATOR_HUMAN, 1)[0].strip()

    # 2. '1. 삭제', '2. 삭제' 등의 패턴 제거
    generated_text = re.sub(r'\s*삭제\s*', ' ', generated_text, flags=re.IGNORECASE).strip()

    generated_text = re.sub(r'\s*\d+\.\s*', ' ', generated_text).strip()

    # 2차 노이즈 제거 및 문장 완결성 확보 로직
    generated_text = re.sub(r'### [A-Za-z]+:', '', generated_text).strip()
    generated_text = re.sub(r'\s+', ' ', generated_text).strip()

    last_stop_index = -1
    for terminator in KOREAN_SENTENCE_TERMINATORS:
        idx = generated_text.rfind(terminator)
        if idx != -1 and idx > last_stop_index:
            last_stop_index = idx

    if last_stop_index != -1:
        generated_text = generated_text[:last_stop_index + 1].strip()

    return generated_text


# ----------------- 5. 싱글턴 테스트 실행 및 결과 출력 -----------------

test_questions = [
    {"domain": "행정작용", "q": "허가와 인가의 차이점은 무엇인지 설명해 주세요."},
    {"domain": "의무이행확보", "q": "즉시강제가 허용되는 경우는 언제인지 설명해 주세요."},
    {"domain": "공무원법", "q": "공무원 징계처분에 대한 재량 통제 기준을 설명해 주세요."},
    {"domain": "정보공개", "q": "정보공개법상 비공개 사유를 1~2개 제시해 주세요."},
    {"domain": "손실보상", "q": "공공필요에 의한 재산권 보상의 헌법적 근거란 무엇인가요?"},
    {"domain": "행정심판", "q": "행정심판에서 기속력과 재결취지준수의무의 차이를 설명해 주세요."},
    {"domain": "조세법", "q": "조세부과처분의 하자가 중대·명백하여 무효가 되기 위한 기준을 설명해 주세요."},
    {"domain": "인허가·제재", "q": "과징금 부과가 재량행위인지 기속행위인지 판단하는 기준을 서술해 주세요."},
    {"domain": "행정작용", "q": "행정지도의 법적 구속력은 어떠한지 설명해 주세요."},
    {"domain": "국가배상", "q": "영조물 설치·관리 하자의 판단 기준은 무엇인가요?"},
    {"domain": "결정례","q": "대법원 판결에서 행정처분의 위법 여부를 판단할 때 고려한 주요 법리는 무엇인가요?"},
    {"domain": "법령","q": "행정절차법 제21조의 사전통지 의무에 관한 요건과 예외 사유를 설명해 주세요."},
    {"domain": "해석례","q": "법제처 해석례 기준으로 조문이 적용되지 않는 사유는 무엇인가요?"}
]


print("\n" + "="*70)
print("             행정법 핵심 도메인 안정성 검증 시작")
print("="*70)

results = []
for i, item in enumerate(test_questions):
    question = item['q']

    # 프롬프트 구성 
    prompt = f"{PROMPT_SEPARATOR_HUMAN}{question}\n\n{PROMPT_SEPARATOR_ASSISTANT}"

    # 답변 생성
    response = generate_response(prompt)

    print(f"\n[테스트 {i+1}/{len(test_questions)}]  도메인: {item['domain']}")
    print(f"  질문: {question}")
    print(f"  답변: {response}")
    print("-" * 70)

    results.append({
        'id': i + 1,
        'domain': item['domain'],
        'question': question,
        'response': response
    })

print("\n" + "="*70)
print("              최종 싱글턴 QA 검증 완료")
print("================================================================")


# In[ ]:




