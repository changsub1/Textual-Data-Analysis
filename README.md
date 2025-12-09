# Textual-Data-Analysis (행정법 LLM DAPT + QA 파인튜닝)

## 프로젝트 개요
- 행정법 문서 요약·질의응답을 위해 Qwen 2.5-1.5B Instruct를 도메인 적응 사전학습(DAPT) 후 QA Instruction Fine-Tuning(IFT)으로 이어 학습한 프로젝트입니다. 모든 파인튜닝은 LoRA로 수행했습니다.
- 학습·추론 코드는 Colab A100 환경을 기준으로 작성되었으며, 경로는 로컬 환경에 맞게 수정해야 합니다.

## 저장소 구성
- `models/qwen2.5_1.5b_dapt_adapter_bf16/` (Git LFS): DAPT(법령/해석례/결정례)로 학습한 LoRA 어댑터.
- `qa_params/` (Git LFS): 위 DAPT 어댑터를 기반으로 QA Instruction Fine-Tuning을 추가한 LoRA 어댑터.
- `code/Qwen_DAPT_Training.py`: 도메인 적응 사전학습(DAPT) 코드(Colab 스크립트).
- `code/fine_tuing_fix2.py`: QA Instruction Fine-Tuning 코드(ROUGE 평가 포함). `BASE_PATH`, 데이터/출력 경로, `LORA_PATH`를 환경에 맞게 수정 필요.
- `code/qwen_simgleturn_ver2.py`: 싱글턴 QA 테스트 스크립트. `BASE_PATH`, `ADAPTER_PATH`를 로컬 어댑터 경로(예: `qa_params/` 또는 `models/...dapt_adapter_bf16/`)로 바꿔 사용.
- `dataset/`: DAPT·QA 학습/검증 코퍼스(`train_law_corpus_dapt_fully_cleaned.jsonl`, `train_instruction_corpus.jsonl`, `valid_law_corpus_dapt_fully_cleaned.jsonl`, `valid_instruction_corpus.jsonl`).
- 기타: `.gitattributes`에 LFS 규칙이 포함되어 있으니 clone 전에 `git lfs install`을 실행하세요.

## 학습 파이프라인
1) **DAPT (Domin-Adaptive Pre-Training)**  
   - 데이터: AI Hub 법률 데이터셋(법령, 해석례, 결정례).  
   - 베이스: `Qwen/Qwen2.5-1.5B-Instruct` (BF16, LoRA).  
   - 출력: `models/qwen2.5_1.5b_dapt_adapter_bf16/`.

2) **QA Instruction Fine-Tuning (IFT)**  
   - 데이터: 행정법 QA 세트(`dataset/train_instruction_corpus.jsonl`, `valid_instruction_corpus.jsonl` 등; Colab 경로 기준).  
   - 입력 어댑터: DAPT 결과(`LORA_PATH`).  
   - 출력: `qa_params/` 어댑터.  
   - 평가지표: ROUGE-L 0.3893(목표 0.4000 대비 97.3% 달성).

3) **싱글턴 추론 테스트**  
   - 스크립트: `qwen_simgleturn_ver2.py`.  
   - 하드코딩된 경로를 로컬 저장소 위치로 변경 후 실행.  
   - 테스트 질의 목록이 포함되어 있어 환각 여부를 빠르게 점검할 수 있습니다.

## 재현/사용 가이드
1. 의존성: `pip install transformers peft datasets evaluate rouge_score matplotlib` (Colab에서는 추가로 `google.colab` 드라이브 마운트).  
2. Git LFS: `git lfs install` 후 clone/pull.  
3. 경로 설정: 각 스크립트 상단의 `BASE_PATH`, `LORA_PATH`, `ADAPTER_PATH` 등을 현재 저장소 경로에 맞게 수정.  
4. DAPT 실행: `Qwen_DAPT_Training.py`(또는 노트북)에서 데이터 경로(`dataset/train_law_corpus_dapt_fully_cleaned.jsonl`)와 출력 경로를 맞춘 뒤 학습.
5. QA IFT 실행: `fine_tuing_fix2.py`에서 학습/검증 데이터 경로와 `LORA_PATH`(DAPT 어댑터) 설정 후 학습.  
6. 추론 테스트: `qwen_simgleturn_ver2.py`에서 어댑터 경로를 `qa_params/`로 지정한 뒤 실행하면 내장된 질의 리스트로 응답을 확인할 수 있습니다.



