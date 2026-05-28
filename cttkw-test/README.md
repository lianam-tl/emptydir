# chat_template_kwargs end-to-end 검증용 payloads

## 파일

| 파일 | endpoint | 용도 |
|---|---|---|
| `01-infracontroller-llm-worker.json` | `POST infracontroller:8080/model-deployments` | lia-test-27b kserve worker 재기동 (Qwen3-VL-32B-Instruct + RL ckpt) |
| `02-orchestrator-job-with-kwargs.json` | `POST orchestrator:8080/jobs` | 1 sample direct job, chat_template_kwargs={enable_thinking:T, supports_thinking:T} 포함 |
| `03-orchestrator-job-control.json` | `POST orchestrator:8080/jobs` | 02 와 동일 prompt, chat_template_kwargs 없이 (baseline) |
| `04-eval-service-request.json` | `POST eval-service-long8v:8090/api/eval/runs` | 풀 파이프라인 eval-service 통한 1 sample 평가 |
| `05-speccenter-active-spec.json` | `GET speccenter:8080/pipelines/vllm-direct` 응답 캡처 | 활성 spec 검증용 (`1.1.3-cttkw`, init_args.use_chat_completions=true 하드코딩) |

## port-forward (cluster `tl-data-training`)

```bash
kubectl -n pegasus-platform port-forward svc/infracontroller   18092:8080 &
kubectl -n pegasus-platform port-forward svc/orchestrator      18091:8080 &
kubectl -n pegasus-platform port-forward svc/speccenter        18080:8080 &
kubectl -n pegasus-platform port-forward svc/eval-service-long8v 18090:8090 &
kubectl -n pegasus-platform port-forward svc/batch-request     18096:8096 &
```

## 사용 예

```bash
# 1) kserve worker 재기동 (이미 있으면 skip)
curl -X POST http://localhost:18092/model-deployments \
  -H 'Content-Type: application/json' \
  -d @01-infracontroller-llm-worker.json

# 2) 직접 orchestrator 에 1 job — kwargs 있음
curl -X POST http://localhost:18091/jobs \
  -H 'Content-Type: application/json' \
  -d @02-orchestrator-job-with-kwargs.json
# 응답: {"jobId":"<uuid>"}

# 3) 직접 orchestrator 에 1 job — kwargs 없음 (baseline)
curl -X POST http://localhost:18091/jobs \
  -H 'Content-Type: application/json' \
  -d @03-orchestrator-job-control.json

# 4) eval-service 풀 파이프라인 (참고)
curl -X POST http://localhost:18090/api/eval/runs \
  -H 'Content-Type: application/json' \
  -d @04-eval-service-request.json
```

## 핵심 변수

| 변수 | 값 |
|---|---|
| pipelineId | `vllm-direct` |
| active spec version | `1.1.3-cttkw` (init_args.use_chat_completions=true 하드코딩) |
| worker_type | `lia-test-27b` (kserve InferenceService — istio gateway 라우팅 가능) |
| modelPath / MODEL env | `s3://tl-data-training-pegasus-us-west-2/checkpoints/dylan/soce_rl_0414_r0330_a05160_reward50_w553510/` |
| vllm-video image (worker) | `tl-data-training-pegasus-vllm-video:575a27f` (kserve) 또는 `lia-vllm-chat-template-kwargs-c51c8df` (lia 빌드) |
| wf-engine image | `tl-data-training-pegasus-workflow-engine:lia-vllm-chat-template-kwargs-063e019` (debug log 포함) |

## 현재 검증 상태 (2026-05-28 기준)

- ✅ orchestrator 의 job.params 에 chat_template_kwargs 잘 들어감 (CTTKW_SPEC 로그 확인)
- ✅ speccenter 1.1.3-cttkw 활성 (params_schema + init_args 다 OK)
- ✅ wf-engine validation 통과 (CTTKW_VALIDATED 로그)
- ❌ vllm_infer_direct 함수 진입 시점에 `chat_template_kwargs={}` (CTTKW_DEBUG)
- ❓ 두 stage 사이 (`_resolve_args` → `vllm_infer_direct`) 어딘가에서 drop
- ❓ control job 과 with-kwargs job 의 output 완전 동일 (input/output tokens, text 다 same)

## 다음 단계 후보

1. `CTTKW_RESOLVED` 로그 (`pipeline_spec._resolve_args` 직후) 추가 후 재테스트 — 진행 중
2. `_filter_fn_init_args` 의 input/output 도 로그
3. spec resolver 가 일찍 진행되는지 (`_init_args_cache` 가 잘못 caching) 검증
