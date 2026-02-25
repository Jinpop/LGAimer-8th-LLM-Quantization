# LG Aimer – EXAONE 양자화 실험 요약

---

## 공통 설정(실험 베이스라인)

- **모델**: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- **양자화 방식**: GPTQ 계열(`llmcompressor` + `compressed-tensors`) 기반 **가중치 4bit/활성 16bit (W4A16)**
- **저장 포맷**: `save_compressed=True`로 **pack-quantized** 형태로 저장(제출 용량 절감)
- **대상 레이어(정규식)**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **제외 레이어**: `embed_tokens`, `lm_head`
- **캘리브레이션**: 여러 데이터셋을 섞어서 “평가 분포”에 가깝게 맞추는 방식으로 반복 실험
- **캘리브레이션 시퀀스 길이**: `MAX_SEQ_LEN=2048`
- **GPTQ 설정**: `block_size=64`, `dampening_frac=0.03`

### 점수 산식(리더보드)

- `Score = max(0.5 × PerfNorm_model + 0.5 × SpeedNorm_model, 0)`
- `PerfNorm_model = Perf_model / Perf_base_model`
- `SpeedNorm_model = 1 − (Time_model / Tokens_model) ÷ (Time_base_model / Tokens_base_model)`

**동일 제출물이라도 서버 부하(토큰당 시간) 변화로 점수가 변동**할 수 있습니다.

---

## 캘리브레이션 데이터 비율 실험

아래 비율은 “캘리브레이션 샘플 구성” 비율입니다(= GPTQ 스케일/통계 추정을 위한 입력 분포).

### 1) `baseline` (= `mix_no_gsm8k`)

총 4096 샘플 기준:

- `MANTA-1M`: 50% (2048)
- `KMMLU-Pro`: 25% (1024)
- `KMMLU-Redux`: 12.5% (512)
- `Ko-LongRAG`: 12.5% (512)

### 2) `mixed` (혼합 + GSM8K 포함)

총 4096 샘플 기준:

- `MANTA-1M`: 50% (2048)
- `KMMLU-Pro`: 18.75% (768)
- `KMMLU-Redux`: 12.5% (512)
- `Ko-LongRAG`: 12.5% (512)
- `GSM8K`: 6.25% (256)

### 3) `ko_reinforce` (한국어 지식 강화형)

총 4096 샘플 기준:

- `KMMLU-Pro`: 40% (1640)
- `KMMLU-Redux`: 20% (820)
- `MANTA-1M`: 30% (1228)
- `Ko-LongRAG`: 10% (408)

### 4) `new_mix` (KoMT-Bench 추가, oversample 없음)

총 4176 샘플 기준:

- `MANTA-1M`: 2048
- `KMMLU-Pro`: 1024
- `KMMLU-Redux`: 512
- `Ko-LongRAG`: 512
- `KoMT-Bench`: 80

> KoMT-Bench는 현재 환경에서 접근 가능한 split의 샘플 수가 80개였고, **중복(oversample) 없이 “있는 만큼만”** 사용.

### 5) `double_baseline` (baseline 2배 시도)

목표(총 8192 샘플):

- `MANTA-1M`: 4096
- `KMMLU-Pro`: 2048
- `KMMLU-Redux`: 1024
- `Ko-LongRAG`: 1024

실제 구성은 `Ko-LongRAG`에서 샘플을 전부 확보하지 못해(예: 600 row) **최종 calib 크기가 8192보다 작게 생성됨**(예: 7768).

---

## LoRA(SFT) 실험 기록 (`lora_1024.ipynb`)

### 파이프라인 요약

1. **SFT 전처리(Risk-hardened)**  
   - `NO truncation`, 길이 초과 샘플 drop, `prompt` 구간 `labels=-100`, `answer` 구간만 학습
2. **LoRA 학습(1 epoch)**  
   - `r=16`, `lora_alpha=32`, `lora_dropout=0.05`  
   - target: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
3. **저장 이슈 대응**  
   - shared tensor 저장 이슈 방지를 위해 `save_safetensors=False`로 adapter 저장
4. **Merge 모델 생성**  
   - adapter merge 결과: `/workspace/sft_merged_2048`
5. **Merge 모델 GPTQ 양자화**  
   - `NUM_CALIB=1024`, `MAX_SEQ_LEN=2048`, `block_size=64`, `dampening_frac=0.03`  
   - 출력: `/workspace/sft_gptq_1024/model`

### 전처리 필터링 결과(노트북 로그)

- `KMMLU-Pro`: `1000/1000` keep (`100%`)
- `KMMLU-Redux`: `600/600` keep (`100%`)
- `MANTA-1M`: `2000/2000` keep (`100%`)
- `Ko-LongRAG`: `3/400` keep (`0.8%`)
- 최종 SFT train size: `3603`, `max_len: 1929`

### LoRA 실험 메모

- LoRA 학습 자체는 완료되었고(`adapter saved`), merge도 완료됨(`merged saved`)
- 이후 merge 모델에 GPTQ를 적용한 단일 파이프라인까지 연결됨
- `lora_1024.ipynb` 내 GPTQ 셀 코멘트에는 `NUM_CALIB=256`이 적혀 있지만 실제 변수는 `1024`로 설정되어 있었음(실행값 기준 확인 필요)
- 실행 중 tokenizer regex warning, 일부 run에서 `Failed to compress 7 modules` 경고가 관측됨(재현 시 해당 런 결과는 별도 검증 필요)

---

## 제출 결과 요약(사용자 제공 값)

> 주의: 아래 “소요시간”은 리더보드 표기 시간(전체 실행시간)이며, 점수의 속도 항은 “토큰당 추론 시간(Time/Tokens)”을 기반으로 하므로 1:1로 동일하지 않을 수 있습니다.

### 제출 로그(시간순)

| 제출 시각 | 파일명 | 실험 태그 | 점수 | 소요시간 | 비고 |
| --- | --- | --- | ---: | ---: | --- |
| 2026-02-06 01:39:38 | `mixed.zip` | `mixed` | 0.5680022838 | 10분 24초 | GSM8K 포함 혼합 |
| 2026-02-06 02:27:02 | `baseline.zip` | `baseline` | 0.4883825713 | 13분 13초 | baseline(no_gsm8k) |
| 2026-02-06 02:29:57 | `ko_reinforce.zip` | `ko_reinforce` | 0.5913649503 | 10분 47초 | 한국어 지식 강화형 |
| 2026-02-12 13:24:00 | `submit.zip` | `new_mix` | 0.4730730365 | 13분 27초 | KoMT-Bench 포함 버전(1차) |
| 2026-02-13 01:37:12 | `skip_think.zip` | `skip_think` | 제출 오류 | 20분 | 시간 제한 초과 |
| 2026-02-13 05:24:40 | `new_mix.zip` | `new_mix` | 0.4700985547 | 13분 33초 | KoMT-Bench 포함 버전(2차) |
| 2026-02-13 06:09:15 | `ko_reinforce.zip` | `ko_reinforce` | 0.5968131516 | 10분 39초 | 동일 zip 재제출 |
| 2026-02-14 06:55:51 | `double_baseline.zip` | `double_baseline` | 0.4889922267 | 12분 43초 | baseline 2배 시도 |
| 2026-02-14 07:53:19 | `double_baseline.zip` | `double_baseline` | 0.4873390169 | 12분 46초 | baseline 2배 시도 |
| 2026-02-14 13:14:33 | `awq_gptq.zip` | `awq_gptq` | 0.4861138378 | 13분 23초 | 2단계 PTQ(AWQ+GPTQ) |
| 2026-02-15 23:37:28 | `no_kolongrag.zip` | `no_ko_long_rag` | 0.4959403046 | 13분 30초 | baseline에서 Ko-LongRAG 제외 |
| 일시 미기록 | `baseline.zip` | `baseline` | 0.6052628482 | 9분 57초 | 동일 파일 재제출(서버 변동) |
| 일시 미기록 | `baseline.zip` | `baseline` | 0.47대 | 미기록 | 최고점 baseline 재제출 하락 사례 |
| 2026-02-25 04:53:45 | `requant_test_submit.zip` | `ko_reinforce_requant` | 0.5764580498 | 9분 11초 | 재양자화 모델 |

> 메모:
> - `ko_reinforce` 2차 제출은 **이전과 동일 zip**(캘리브레이션/모델 동일)이며, 점수 차이는 서버 부하/측정 변동 가능성이 큼
> - `new_mix`는 **KoMT-Bench(80) 추가** 캘리브레이션으로 2회 제출 (0.47대)
> - `baseline`은 최고점(0.60대) 재현이 항상 되지 않았고, 재제출에서 0.47대 하락 사례 확인
> - `ko_reinforce_requant`는 0.5764 / 9분 11초로 속도는 우수했지만 기존 `ko_reinforce` 최고점(0.5968)에는 미달

### 파일명 양식 정리

| 파일명 | 생성 노트북/설정 | 설명 |
| --- | --- | --- |
| `baseline.zip` | `base_line.ipynb` | baseline(no_gsm8k) |
| `mixed.zip` | `mix_best.ipynb` | GSM8K 포함 혼합 |
| `ko_reinforce.zip` | `ko_reinforce.ipynb` | KMMLU 비중 강화 |
| `new_mix.zip` / `submit.zip` | KoMT-Bench 추가 실험 | KoMT-Bench 포함(oversample 없음) |
| `double_baseline.zip` | baseline 2배 샘플 | 8192 목표, 실제는 Ko-LongRAG 부족 가능 |
| `awq_gptq.zip` | AWQ + GPTQ recipe | 2단계 PTQ 실험 |
| `no_kolongrag.zip` | baseline 변형 | Ko-LongRAG 제외 버전 |
| `no_truncate` | 최신 제출 별칭(사용자 메모) | 점수/시간 별도 기록 필요 |

### 추론 템플릿/프롬프트 변경 제출(사용자 제공 값)

> 아래는 **양자화 가중치 자체는 동일**하고 `chat_template.jinja`만 바꿔서 제출한 결과입니다.

| 태그(제출물)    |         점수 |  소요시간 |
| --------------- | -----------: | --------: |
| `no_think`      | 0.4707728995 | 13분 58초 |
| `safe_prompt`   | 0.1200848258 |  7분 21초 |
| `skip_think`    | 제출 오류(시간 제한 초과) | 20분 |

### 서버 부하로 인한 변동 예시(동일 파일)

- 동일 파일 제출에서 `0.4883825713 (13분 13초)` → `0.6052628482 (9분 57초)`로 상승 사례 확인
  - 성능(PerfNorm)이 동일하다고 가정하면, 점수 차이의 대부분은 속도 항(SpeedNorm) 변동으로 해석 가능

---

## 결론(현재까지)

- `ko_reinforce`처럼 **KMMLU(Pro/Redux) 비중을 올린 “한국어 지식 강화형”이 점수가 높게** 관측됨.
- `mixed`처럼 **GSM8K를 포함한 혼합 구성은 상대적으로 낮게** 관측됨.
- `new_mix`처럼 **KoMT-Bench(80) 추가는 점수 개선에 기여하지 못함** (0.47대)
- `double_baseline`(baseline 2배 시도)는 **점수 개선이 거의 없거나 미미**하게 관측됨 (0.487~0.489, 12분대)
- `skip_think`(<think> 태그 제거)는 **실행 시간 제한 초과로 제출 실패** → `chat_template.jinja` 변경은 리스크가 큼
- `ko_reinforce` 재양자화(`ko_reinforce_requant`)는 **추론 시간 단축(9분대)** 효과는 있었지만, 점수는 기존 최고 대비 낮게 관측됨
- 동일 제출물도 서버 부하에 따라 점수가 출렁일 수 있어, **비교는 2~3회 제출의 평균/최댓값 기반**으로 판단하는 것이 안전함.

---

## 사용자 결론 메모

- `no_ko_long_rag`(`0.4959403046`, 13분 30초)와 `no_truncate`(최신 제출) 파일 자체는 크게 다르지 않음.
- 두 제출의 소요 시간 차이는, 전날 제출이 몰리는 시간대 영향으로 해석함.
- 짧은 calib 데이터를 유지하는 방식은 토큰 생성량과 생성 속도를 줄이는 데 크게 기여한다고 판단함.

---

## 추론 템플릿 실험(제출 + 로컬 vLLM 벤치)

아래 2개 노트북은 **양자화 가중치 자체는 동일**하고, `chat_template.jinja` 추론 동작만 다르게 적용한 실험입니다.

- `no_think.ipynb`: `enable_thinking=false`를 강제해 `<think>` 출력을 비활성화
- `prompted.ipynb`: `<think>` 강제 비활성화는 제거하고, 기본 `system prompt`로 답변 형식을 짧게 유도

| 방식                          | Latency speedup (BASE/QUANT) | Gen throughput (QUANT/BASE) | Total throughput (QUANT/BASE) |
| ----------------------------- | ---------------------------: | --------------------------: | ----------------------------: |
| `no_think`                    |                       1.5545 |                      1.8407 |                        1.7432 |
| `prompted`                    |                       1.5721 |                      1.7327 |                        2.2790 |
| 차이(`prompted` - `no_think`) |                       +1.13% |                      -5.87% |                       +30.74% |

### 로컬 벤치 기준 요약

- `prompted`가 지연시간(latency)과 전체 처리량(total throughput)에서 우세 (성능 체크 x)
- `no_think`가 생성 토큰 처리량(gen throughput)에서는 우세
- 실제 제출 결과 기준으로는 `safe_prompt`(system prompt 주입)가 **점수/성능 측면에서 불리**하게 관측됨
  - (가설) 시스템 롤(`[|system|]`) 추가로 프롬프트 분포가 바뀌어, 비공개 벤치에서 정답률이 크게 하락했을 가능성
- `skip_think`(<think> 태그 자체 제거)는 **시간 제한(20분) 초과로 제출 오류**가 발생해, 템플릿 대수정은 신중하게 접근 필요

---

## 모델 파일 크기 차이(동일 코드인데 다르게 저장되는 이유)

동일한 양자화/저장 코드여도 `model.safetensors` 크기가 달라질 수 있습니다.

- 대표 케이스: `lm_head.weight`가 `embed_tokens.weight`와 **weight tying(공유)** 되어 있으면 저장 시 중복이 제거되어 파일이 작아짐
- 반대로 weight tying이 풀리면 `lm_head.weight`가 별도 텐서로 저장되어 **약 400MB 정도 커질 수 있음** (EXAONE-4.0-1.2B 기준)
