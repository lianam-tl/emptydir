# chat_template_kwargs E2E 검증 작업일지

PR `twelvelabs-io/xplatform#80` 의 `chat_template_kwargs` plumbing 을 production eval-service 풀 파이프라인으로 검증한 과정. 2026-05-27 ~ 2026-05-28 (KST).

## 목표

PR 본문이 약속한 `chat_template_kwargs` (특히 `enable_thinking=true`, `supports_thinking=true`) 가 worker `processor.apply_chat_template` 까지 도달하는지 end-to-end 검증.

## 결론 (TL;DR)

- ✅ **PR 코드 자체엔 bug 없음** — 4-stage debug log (`CTTKW_SPEC` / `CTTKW_VALIDATED` / `CTTKW_RESOLVED` / `CTTKW_DEBUG`) 모두 정상 값
- ✅ **orchestrator → wf-engine → kserve worker** 경로로는 chat_template_kwargs 가 worker 에 정상 도달 (v17 잡 `b41e9bb3` 으로 확인)
- ❌ **eval-service 풀 파이프라인 (batch-request → orchestrator → wf-engine → batch worker)** 로는 chat_template_kwargs 가 worker 에 절대 도달하지 못함. 이유는 코드 bug 아니라 **인프라 정렬 문제**:
  - batch worker (orchestrator 가 띄우는 model-batch pod) 는 KServe InferenceService 가 아니라 raw k8s Deployment
  - istio gateway 의 `/kserve-models/{worker_type}/v1/chat/completions` URL 라우팅 안 됨 → HTTP 404
  - 따라서 wf-engine 의 `call_vllm` 이 chat completions path 진입해도 endpoint 닿지 못함
  - chat_completions path 가 아닌 v2 binary infer path 만 가능 → PR 의 chat_template_kwargs plumbing 은 v1 chat completions path 에만 있으므로 drop

## 시도별 결과 (v1 ~ v17)

| 시도 | 변경 | 결과 / 막힌 지점 |
|---|---|---|
| v1 | eval-service 첫 submit, `chatTemplateKwargs` 포함 | imageDigest env override 로 worker 가 옛 main 빌드 (`56a27aa`) 사용 — server.py 에 chat_template_kwargs parsing 없음 → drop |
| v2 | 동일 (eval-service) | speccenter spec 미등록. orchestrator 가 옛 1.1.0 spec embed → init_args.chat_template_kwargs 매핑 없음 → drop |
| v3 | request body 에 `imageDigest` 명시 + speccenter 1.1.1-cttkw 활성 + isolated `workflow-engine-lia` deploy | shared 136 replicas vs lia 1 → 거의 모든 잡 shared 가 잡음. shared wf-engine 옛 task 코드 (`chat_template_kwargs` param 시그니처에 없음) → `_filter_fn_init_args` 가 silently drop |
| v4 | shared wf-engine image swap (lia tag) | KEDA 가 ScaledObject 로 replicas 강제, 새 image 의 코드는 정상이지만 spec `use_chat_completions default=False` → v2 path → batch worker 의 binary infer → chat_template_kwargs slot 없음 |
| v5 | maxSamples=1 smoke | 동일 |
| v6 | spec `use_chat_completions default=True` (1.1.2-cttkw) | 동일 — orchestrator/wf-engine 캐싱 의심 또는 path 라우팅 |
| v7~v8 | larger maxSamples 로 worker alive window 확보 | 동일 |
| v9~v10 | spec re-activate + 새 batch | 여전히 v2 path. wf-engine 로그 `call_mode=loadbalancer` |
| v11 | spec init_args `use_chat_completions: true` 하드코딩 (`1.1.3-cttkw`) | wf-engine chat_completions path 진입 ✓ 하지만 batch worker URL **HTTP 404** |
| v12, v13 | cache bust 새 prompt | 동일 — batch worker 는 chat completions endpoint 못 받음 |
| v14 | **orchestrator 직접 POST /jobs + worker_type=lia-test-27b kserve worker** | 풀 파이프라인 우회. 그러나 cache hit 으로 with/without kwargs 동일 output 보임 (lia 이미지의 c51c8df 에는 cache_key signature 없음) |
| v15 | debug log 추가 (`CTTKW_SPEC`, `CTTKW_VALIDATED`) | `validated.chat_template_kwargs={enable_thinking:T, supports_thinking:T}` 확인 |
| v16 | `CTTKW_RESOLVED` 추가 | `init_args.chat_template_kwargs={enable_thinking:T, supports_thinking:T}` 확인 |
| v17 | `AUTO_BATCH_WORKER_CTTKW_KWARGS` + `INJECT_INIT_ARGS_CTTKW` print 추가 | **`fn(**kwargs)` 진입 시 chat_template_kwargs 정상 dict** ✓ |

## 핵심 deliverable

### 발견된 PR scope gap

PR #80 doc:
> Pass **both** `enable_thinking` and `supports_thinking` for Qwen3 reasoning.
> ...
> Worker (server.py): `processor.apply_chat_template(messages, ..., **(chat_template_kwargs or {}))`.

→ chat completions path (`/v1/chat/completions`) 에만 plumbing.

문제: 우리 production wf-engine 파이프라인 (batch-request → orchestrator → wf-engine → batch worker) 에선:
- batch worker (orchestrator 가 띄우는 `model-batch-*` pod) 는 KServe InferenceService 아님
- istio gateway 가 `/kserve-models/{worker_type}/v1/chat/completions` URL 을 batch worker 로 routing 못 함
- wf-engine 이 chat completions path 호출해도 404
- 그래서 v2 binary infer path (loadbalancer queue 통해 redis queue 로 전달) 만 가능
- 하지만 PR 이 v2 path 에 chat_template_kwargs slot 안 만들었음 → drop

### 가능한 후속 작업 (옵션)

**A. call_vllm 자동 라우팅** (PR scope 내 최소 변경):
```python
# vllm_common.py:call_vllm 진입점
if chat_template_kwargs:
    use_chat_completions = True  # 강제 v1 chat completions path
```
→ batch worker 에선 여전히 404 (URL 라우팅 자체 안 됨). 하지만 PR 의도 (chat completions 에만) 보존.

**B. v2 path 에도 chat_template_kwargs plumbing** (PR scope 확장):
- `vllm_common.py:_build_vllm_transport_request` 에 chat_template_kwargs BYTES tensor input 추가
- `server.py:_build_generate_request_from_v2` 에서 파싱 → `GenerateRequest.chat_template_kwargs` 채움 → 그 다음 path 동일
- 별도 PR 로 분리 권장
- 결과: batch worker (v2 path 만 가능) 에서도 chat_template_kwargs 동작

**C. 기존 PR 머지 + production 정렬**:
- PR 본문에 "batch worker 에선 동작 안 함, kserve InferenceService 만 가능" 명시
- 추후 batch worker 도 kserve InferenceService 로 마이그레이션 (별도 작업)

### 현재 상태

- `lia-test-27b` kserve worker 활성 (infracontroller 통해 다시 띄움)
- speccenter active spec: `1.1.3-cttkw` (`use_chat_completions: true` 하드코딩, `chat_template_kwargs` route)
- wf-engine image: `lia-vllm-chat-template-kwargs-39c3e4a` (debug log 4종 포함)
- shared `workflow-workflow-engine-v1-pegasus` 배포 이미지 lia tag 로 swap
- lia 격리 `workflow-engine-lia` 배포 동일 image

## 파일

- `01-infracontroller-llm-worker.json` — lia-test-27b kserve worker spec (`POST /model-deployments`)
- `02-orchestrator-job-with-kwargs.json` — 1-job orchestrator submit (with chat_template_kwargs)
- `03-orchestrator-job-control.json` — baseline (no kwargs)
- `04-eval-service-request.json` — 풀 파이프라인 (batch worker, 현재 동작 X)
- `05-speccenter-active-spec.json` — 현재 활성 spec snapshot
- `README.md` — 각 파일 용도 + port-forward + curl 예
- `JOURNEY.md` — 이 파일 (전체 작업 일지)

## 참고 디버그 로그 (v17)

```
JobActor pid=1236   CTTKW_SPEC      job=b41e9bb3-... params_schema.chat_template_kwargs={'default': {}, 'type': 'dict'}
                                    init_args.chat_template_kwargs='$job.params.chat_template_kwargs'
                                    job.params.chat_template_kwargs={'enable_thinking': True, 'supports_thinking': True}
JobActor pid=1236   CTTKW_VALIDATED validated.chat_template_kwargs={'enable_thinking': True, 'supports_thinking': True}
JobActor pid=1236   CTTKW_RESOLVED  node=1 init_args.chat_template_kwargs={'enable_thinking': True, 'supports_thinking': True}
                                    init_args.use_chat_completions=True
_auto_batched_remote AUTO_BATCH_WORKER_CTTKW_KWARGS chat_template_kwargs={'enable_thinking': True, 'supports_thinking': True}
_auto_batched_remote CTTKW_DEBUG vllm_infer_direct ENTRY chat_template_kwargs={'enable_thinking': True, 'supports_thinking': True}
```

## PR snippet 검증 (PR doc 의 verification table)

PR 본문은 kserve worker `lia-test-27b` 에 `/v1/chat/completions` 직접 POST 로 검증:

| Run | chat_template_kwargs | prompt_tokens | completion_tokens |
|---|---|---|---|
| 1 | (omitted) | 27521 | 31 |
| 2 | `{"enable_thinking": true}` | 27549 (+28) | 34 |
| 3 | `{"enable_thinking": true, "supports_thinking": true}` | 27551 (+30) | 1363 (44× ↑) |

→ PR snippet 은 wf-engine 우회 직접 HTTP 호출 — 즉 production 파이프라인 검증 X. 우리 작업은 production 파이프라인 통합까지 확인.
