# LLM Quantization 실험 레포

## 본 브런치 코드 요약

- **`exaone_final_quant_factory.ipynb`**: EXAONE 4.0용 **AWQ W4A16** 양자화 파이프라인 노트북
  - **환경**: Google Colab + `llm-compressor`, Drive 마운트 및 Hugging Face 로그인
  - **모델**: EXAONE-4.0-1.2B (로컬 `base_model` 또는 Hub)
  - **Calibration 데이터**: Ko-LongRAG, MANTA-1M(샘플링), KMMLU-Pro, KMMLU-Redux 병합 후 텍스트 포맷으로 전처리

- **AWQ W4A16 설정**
  - `modifier_type`: `"AWQ"`, `scheme`: `"W4A16"` (가중치 4bit, 활성화 16bit)
  - `llmcompressor`의 `AWQModifier` 사용, 전략은 `STRATEGIES`에 정의 (예: `10_AWQ_W4A16`)

- **COMMON_CONFIG** (AWQ 포함 모든 전략 공통)
  - `targets`: `["Linear"]` — 양자화 적용 레이어
  - `ignore`: `["lm_head", "norm", "rotary_emb"]` — EXAONE 4.0 QK-Reorder-LN 등 민감 레이어 제외
  - `num_calib`: `min(len(calib_ds), 512)` — calibration 샘플 수
  - `max_seq`: `2048` — calibration 시 최대 시퀀스 길이

- **제출물**: 양자화 완료 시 `submit_{전략명}.zip` 자동 생성

---

## 목표
- 원본(FP16/BF16) 모델의 **베이스라인 성능** 측정
- 양자화 적용
- 양자화 모델의 **성능 저하 폭**과 **효율 개선**(VRAM 사용량, 추론 속도) 비교
- 실험 설정(버전, 파라미터, 결과)을 재현 가능하게 기록

## 평가 기준(예시)
- 정확도/정답률: ARC-Easy, GSM8K 등
- 효율: GPU VRAM 사용량, 추론 시간(latency), 처리량(throughput)

> 각 실험은 사용한 런타임/버전 정보를 ReadMe에 함께 기록.

## 실험 흐름
1. **환경 확인**: GPU/파이썬/torch 버전 및 메모리 확인  
2. **Baseline 측정**: 양자화 전 모델을 벤치마크로 평가  
3. **양자화 수행**: 선택한 방식으로 모델 양자화  
4. **양자화 모델 평가**: 동일 벤치마크로 재평가하여 성능 비교  
5. **결과 기록**: 설정값(비트수, 그룹 사이즈, calibration 등)과 결과 테이블 저장

## 기록할 항목(예시)
- 모델명 / 체크포인트
- 양자화 방식 및 설정 (bit, group size, calibration 샘플 수, 제외 레이어 등)
- 벤치마크 결과(전/후)
- VRAM 사용량 / 속도 변화
- 실행 환경(플랫폼, GPU, Python/torch/transformers 버전)
