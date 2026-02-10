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

---

## 제출 결과 요약(사용자 제공 값)

> 주의: 아래 “소요시간”은 리더보드 표기 시간(전체 실행시간)이며, 점수의 속도 항은 “토큰당 추론 시간(Time/Tokens)”을 기반으로 하므로 1:1로 동일하지 않을 수 있습니다.

| 태그(제출물)   |         점수 |  소요시간 |
| -------------- | -----------: | --------: |
| `baseline`     | 0.4883825713 | 13분 13초 |
| `baseline`     | 0.6052628482 |  9분 57초 |
| `mixed`        | 0.5680022838 | 10분 24초 |
| `ko_reinforce` | 0.5913649503 | 10분 47초 |

### 서버 부하로 인한 변동 예시(동일 파일)

- 동일 파일 제출에서 `0.4883825713 (13분 13초)` → `0.6052628482 (9분 57초)`로 상승 사례 확인
  - 성능(PerfNorm)이 동일하다고 가정하면, 점수 차이의 대부분은 속도 항(SpeedNorm) 변동으로 해석 가능

---

## 결론(현재까지)

- `ko_reinforce`처럼 **KMMLU(Pro/Redux) 비중을 올린 “한국어 지식 강화형”이 점수가 높게** 관측됨.
- `mixed`처럼 **GSM8K를 포함한 혼합 구성은 상대적으로 낮게** 관측됨.
- 동일 제출물도 서버 부하에 따라 점수가 출렁일 수 있어, **비교는 2~3회 제출의 평균/최댓값 기반**으로 판단하는 것이 안전함.

---

## 추론 템플릿 실험(미제출, 로컬 vLLM 벤치)

아래 2개 노트북은 **양자화 가중치 자체는 동일**하고, `chat_template.jinja` 추론 동작만 다르게 적용한 실험입니다.

- `no_think.ipynb`: `enable_thinking=false`를 강제해 `<think>` 출력을 비활성화
- `prompted.ipynb`: `<think>` 강제 비활성화는 제거하고, 기본 `system prompt`로 답변 형식을 짧게 유도

> 아직 리더보드 제출 전이므로, 아래 수치는 채점 점수가 아니라 로컬 vLLM 상대 속도 비교입니다.

| 방식                          | Latency speedup (BASE/QUANT) | Gen throughput (QUANT/BASE) | Total throughput (QUANT/BASE) |
| ----------------------------- | ---------------------------: | --------------------------: | ----------------------------: |
| `no_think`                    |                       1.5545 |                      1.8407 |                        1.7432 |
| `prompted`                    |                       1.5721 |                      1.7327 |                        2.2790 |
| 차이(`prompted` - `no_think`) |                       +1.13% |                      -5.87% |                       +30.74% |

### 로컬 벤치 기준 요약

- `prompted`가 지연시간(latency)과 전체 처리량(total throughput)에서 우세 (성능 체크 x)
- `no_think`가 생성 토큰 처리량(gen throughput)에서는 우세
- 채점 서버의 토큰 집계 기준(`Tokens_model`)에 따라 최종 점수 우위가 달라질 수 있으므로, 두 방식 모두 실제 제출 비교가 필요

---
