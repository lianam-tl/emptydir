# 최종 결과 — chat_template_kwargs 동작 확정

## 결정적 증거 (PR snippet 재현, 2026-05-28)

`lia-test-27b` kserve worker 의 이미지를 **lia 빌드 (`c51c8df`, server.py 에 chat_template_kwargs parsing 포함)** 로 교체 후 PR snippet 3회 재실행. PR doc 의 verification table 패턴 정확히 재현.

| RUN | chat_template_kwargs | 결과 (output text length) |
|---|---|---|
| 1 | `None` | 80 chars — "The segment of the foul involving Zlatan Ibrahimovic runs from 00:00 to 00:15." |
| 2 | `{"enable_thinking": true}` | 110 chars — model ignores instruction, short answer |
| 3 | `{"enable_thinking": true, "supports_thinking": true}` | **12141 chars** (max_tokens=4096 token cutoff) — **step-by-step deliberative reasoning** |

RUN 3 의 prompt_tokens: **27551** = baseline 27521 + **30** (PR doc 의 `+30` 패턴과 일치)

RUN 3 의 text 시작:
> "Let's look at the timestamps 01:41 - 01:47. The most prominent 'foul involving Zlatan' is the one at 03:22 where he commits it. But it's a replay. Let's look at the sequence 03:22 - 03:34..."

→ PR doc 의 RUN 3 ("1. Identify Zlatan... 2. Scan the video... Wait, looking closer...") 와 동일 패턴의 deliberative reasoning.

## 왜 그동안 안 됐나

처음 lia-test-27b 띄울 때 user 가 사용한 spec 의 image (`219219941196.dkr.ecr.us-west-2.amazonaws.com/pegasus-vllm-video:575a27f`) 는 **PR 변경분이 들어있지 않은 dev ECR 의 옛 빌드**. 그 worker 의 server.py 에 `chat_template_kwargs` parsing 0건. 그래서 body 에 chat_template_kwargs 가 와도 apply_chat_template 으로 안 forward.

이 단계가 우리 풀 파이프라인 디버그 동안 **이해의 사각지대**:
- wf-engine 쪽 plumbing 다 검증 완료 (4-stage CTTKW debug log 정상)
- worker 쪽 PR snippet 호출 시 worker server.py 의 chat_template_kwargs 받는다고 가정
- 실제로는 worker 이미지가 옛 빌드라 parsing 없음 → 무시
- PR snippet 자체로도 3 runs 동일 결과 (PR doc 의 verification 재현 X)

## 정확한 동작 조건

`chat_template_kwargs` 가 worker `processor.apply_chat_template` 까지 도달하려면:

1. **client side (eval-service compiler 또는 직접 POST 둘 다)**: params 에 `chat_template_kwargs` 포함
2. **speccenter active spec**: 1.1.3-cttkw — `params_schema.chat_template_kwargs: {type: dict}` + `init_args.chat_template_kwargs: "$job.params.chat_template_kwargs"` + `init_args.use_chat_completions: true` (하드코딩)
3. **wf-engine 이미지**: PR 변경분 포함 — `vllm_infer_direct.py` 의 signature 에 `chat_template_kwargs` param, `vllm_common.py` 의 `call_vllm` / `_iter_chat_completions_body_bytes` 에 chat_template_kwargs 처리
4. **chat completions path 강제**: spec 의 `use_chat_completions: true` 또는 env `VLLM_USE_CHAT_COMPLETIONS=1` (둘 다 만족)
5. **worker_type 이 kserve InferenceService**: istio gateway 가 `/kserve-models/{worker_type}/v1/chat/completions` 로 라우팅 가능
6. **worker 이미지**: PR 변경분 포함 — `server.py` 가 body 에서 `chat_template_kwargs` 파싱 + `processor.apply_chat_template(messages, ..., **chat_template_kwargs)`

빠진 게 하나라도 있으면 `enable_thinking` 효과 없음. 우리 디버그 동안:
- (6) 빠짐 — lia-test-27b 의 575a27f 이미지에 parsing 없음
- 풀 파이프라인 (eval-service → batch worker) 에선 (5) 빠짐 — batch worker 는 kserve 아닌 raw Deployment

## 본 적용 시나리오

| 경로 | (1)-(6) 만족 | 동작? |
|---|---|---|
| PR snippet 직접 → kserve worker (lia c51c8df) | (1)(3)(5)(6) 만족 | ✓ |
| orchestrator POST /jobs → wf-engine (lia 이미지) → kserve worker (lia c51c8df) | 전부 만족 | ✓ |
| eval-service → batch-request → orchestrator → wf-engine → batch worker | (5) 빠짐 (batch worker = raw Deployment, istio route 없음) | ✗ |

→ **eval-service 풀 파이프라인 동작시키려면 PR scope 확장 필요**:
- 옵션 B: `vllm_common.py:_build_vllm_transport_request` 에 chat_template_kwargs tensor 추가 + `server.py:_build_generate_request_from_v2` 에서 파싱
- 옵션 C: batch worker 를 kserve InferenceService 로 마이그레이션

## 다음

PR 머지 시 follow-up issue 로:
- v2 path plumbing 또는 batch worker 마이그레이션 → eval-service 풀 파이프라인에서 동작
- spec 의 `use_chat_completions default=False` → `True` 로 변경 (또는 caller 가 명시) — 이건 wider impact 라 신중
- worker image rollout — `575a27f` 같은 옛 이미지 쓰는 곳들 업데이트
