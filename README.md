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

---

## 제출 결과 요약(사용자 제공 값)

> 주의: 아래 “소요시간”은 리더보드 표기 시간(전체 실행시간)이며, 점수의 속도 항은 “토큰당 추론 시간(Time/Tokens)”을 기반으로 하므로 1:1로 동일하지 않을 수 있습니다.

| 태그(제출물)   |         점수 |  소요시간 |
| -------------- | -----------: | --------: |
| `baseline`     | 0.4883825713 | 13분 13초 |
| `baseline`     | 0.6052628482 |  9분 57초 |
| `mixed`        | 0.5680022838 | 10분 24초 |
| `ko_reinforce` | 0.5913649503 | 10분 47초 |
| `ko_reinforce` | 0.5968131516 | 10분 39초 |
| `new_mix`      | 0.4730730365 | 13분 27초 |
| `new_mix`      | 0.4700985547 | 13분 33초 |
| `skip_think`   | 제출 오류(시간 제한 초과) | 20분 |

> 메모:
> - `ko_reinforce` 2차 제출은 **이전과 동일 zip**(캘리브레이션/모델 동일)이며, 점수 차이는 서버 부하/측정 변동 가능성이 큼
> - `new_mix`는 **KoMT-Bench(80) 추가** 캘리브레이션으로 2회 제출 (0.47대)

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
- `skip_think`(<think> 태그 제거)는 **실행 시간 제한 초과로 제출 실패** → `chat_template.jinja` 변경은 리스크가 큼
- 동일 제출물도 서버 부하에 따라 점수가 출렁일 수 있어, **비교는 2~3회 제출의 평균/최댓값 기반**으로 판단하는 것이 안전함.

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
