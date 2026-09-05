# Token Factory default migration verification

Verified on 2026-09-04/05 UTC against the public Token Factory API with one
configured account and synthetic inputs. The deprecation claim is true, with
an observed exception for Llama 3.3.

## Documented retirement and observed behavior

The [August 2026 notice](https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice)
announces removal on **August 31, 2026** from Serverless API and Playground,
without automatic rerouting. Dedicated endpoints are unaffected. This is a
published effective date, not a measurement of when every deployment stopped.

| Previous repository default | Role | Authenticated observation | New public default |
| --- | --- | --- | --- |
| `meta-llama/Llama-3.3-70B-Instruct` | Workbench text and agent standard tier | Still listed; real text completion succeeded | `nvidia/Nemotron-3_5-Lightning` |
| `Qwen/Qwen3-32B` | Agent cheap tier | HTTP 404, model does not exist | `nvidia/Nemotron-3_5-Lightning` |
| `Qwen/Qwen2.5-VL-72B-Instruct` | Captioning and hosted visual evaluation | HTTP 404, model does not exist | `MiniMaxAI/MiniMax-M3` |
| `nvidia/Cosmos3-Super-Reasoner` | Scene reasoning and Sim2Real evaluation | HTTP 409, model is stopped | `MiniMaxAI/MiniMax-M3` |

These replacements match the notice. Llama's observed success does not promise
continued availability after its documented retirement. Key/account access can
differ; this verification does not establish availability in every account or
region. No dedicated endpoint was tested.

`openai/gpt-oss-120b` remains the separate batch default. Neither that ID nor
embeddings appears in the August notice. Batch processing and embeddings were
not changed or claimed validated by this migration.

## API and integration differences

The endpoint remains the OpenAI-compatible
[`POST /v1/chat/completions`](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion).
The [vision request format](https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities)
still uses `image_url` content parts, including base64 data URLs. Explicit model,
endpoint, SDK arguments, and agent model allowlists remain authoritative; an
explicit legacy model is not silently migrated.
Persisted configurations containing retired IDs, including deployed agent model
allowlists, must be updated explicitly.

Two differences required more than replacing model strings:

- **Thinking controls differ by model.** A Lightning request without its
  control spent 506 of 512 output tokens on reasoning and returned truncated
  JSON. Its documented `chat_template_kwargs.enable_thinking=false` produced a
  complete answer and zero reasoning tokens. MiniMax requires
  `chat_template_kwargs.thinking_mode="disabled"`; `thinking=false` and
  `thinking="disabled"` did not disable its reasoning. The shared client applies
  the verified controls for direct-output workloads; explicit `extra` settings
  win. Agent reasoning turns retain thinking. Sources: the
  [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
  [MiniMax template](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/chat_template.jinja),
  and [vLLM recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3).
- **Hosted MiniMax constrained JSON was malformed.** Both `json_object` and
  `json_schema` responses contained invalid JSON prefixes in independent live
  requests. The VLM judge therefore requests JSON through its existing prompt
  for this model and retains strict parsing, required fields, and score
  validation. Other models retain constrained JSON requests; the general client
  continues honoring explicit `response_format`. No malformed score is repaired
  or promoted into a successful result.

The hosted VLM judge requires a full JSON object, boolean `success`, a finite
numeric score in `[0, 1]`, a nonempty rationale, and `finish_reason="stop"`.
Duplicate keys and fenced or prefixed JSON fail validation. Its `model` result
field preserves the requested model or explicit alias; `served_model` records
the provider's actual response identity. The two public replacement model IDs
must match exactly. Custom aliases may resolve to another nonempty served model;
self-hosted parsing and score overrides retain their existing behavior, with
`served_model=null` when no provider identity was verified.

Model-list membership and upstream model capability are insufficient proof:
`Qwen/Qwen3.5-397B-A17B` was listed and its publisher describes a multimodal
model, but this hosted deployment rejected an image request with HTTP 400,
"This model does not support image input". It was not selected as a replacement.

Sim2Real's stable `cosmos3` lane, artifact names, and configuration keys remain
for compatibility. They now carry the actual selected model and family;
MiniMax responses are never represented as NVIDIA Cosmos responses. Stage 9
checks configured model, family, component provenance, request accounting, and
exact Stage 7 coverage before using evaluations. Named self-hosted Cosmos
Reason1/2 paths remain separate. The hosted temporal adapter accepts the exact
MiniMax and legacy Cosmos3 IDs listed above; unrecognized IDs are rejected.
Start a new run after changing evaluator model
or upgrading to this evaluator contract, even if retaining the same Cosmos3
model: old Stage 8 envelope and component records lack the family fields now
required by Stage 9. Old immutable evaluation artifacts are not relabeled.

The hosted temporal path requires a complete JSON completion, finite scores and
confidence in range, and exactly one model-local event for every input action.
It rejects truncated responses and invalid scores without recovery or clamping.
Stage 9 additionally verifies the component content hash, matching immutable
image/source/workflow provenance, nonempty request identities, positive token
accounting, and exact latency/retry/cost aggregates. An unavailable provider cost
remains explicit `null`. Its migration error occurs before PPO or checkpoint
selection and names the required new run ID/output root. Preserve the old run;
rerun Stage 8 under the current contract before resuming the new run at Stage 9.

## Initial local verification (2026-09-04/05)

Local live validation passed **11 workbench/evaluator tests** and **one agent
HTTP test covering three model tiers**. The committed live tests exercise real
provider calls through CLI/SDK paths,
saved artifacts, the rendered agent HTTP backend, and the hosted temporal
scorer. They use synthetic inventory prompts and geometric image sequences;
no customer data, simulation result, or robot policy quality is implied.
With a configured key, an unavailable default fails instead of falling back
to the first listed model or skipping the affected capability.
The local test environment uses the repository's `dev,adapter` extras plus
`uvicorn` for the agent HTTP process.

The shipped generation and caption reference workflows also completed through
standard NPA/SkyPilot submission on existing CPU capacity, using a byte-verified
source snapshot of this change. The downloaded S3 artifacts contained six
distinct, correct inventory calculations and three correct image/color/shape
captions, with the expected model selections. Both runs reached independently
verified `SUCCEEDED` status. The runs used normal task output storage; optional
SkyPilot durable-storage mounts were not enabled in this runtime.

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_token_factory_e2e.py \
  npa/tests/e2e/test_hosted_rollout_e2e.py \
  npa/tests/e2e/test_agent_token_factory_e2e.py -q \
  --basetemp=<private-artifact-directory>
```

Authentication preflight and `models` are useful prerequisites, not proof of
inference or artifact correctness. Private validation retains synthetic inputs,
real outputs, model selection, available usage, and sanitized validation logs.
This initial proof did not cover full GPU simulation/training or a deployed
agent UI; agent behavior used the rendered backend on isolated loopback.
The follow-up adds a separate [protected provider contract job](../testing/token-factory-live-contracts.md),
whose receipts distinguish hosted GitHub Actions from local invocations.
The concrete daily schedule activates when the workflow reaches `main`.

The follow-up fixes the previously unrelated Cosmos Transfer unit-test failure
in `test_run_cosmos_transfer_names_gated_access_denial_without_leaking_prompt`.
It mocks guardrail-data preparation at the call site and explicitly makes the
optional Hub module unavailable, so the test reaches its intended subprocess
denial/redaction assertion in every environment. The production guardrail and
its own dedicated tests remain strict; a red required CI check is not accepted
as a merge-readiness exception.

## Model terms

NPA ships model identifiers and API integration code; no replacement weights,
vendor runtime, or vendor dataset is bundled. Hosted API authentication does not
prove an operator's commercial entitlement.

Built with MiniMax M3. The
[MiniMax Community License](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE)
expressly includes commercial API use and specifies attribution plus notification
or prior written authorization depending on product/service annual revenue.
Operators must satisfy those terms for their use; this synthetic evaluation
makes no claim to vendor authorization for a commercial deployment. Review
[NVIDIA's Lightning model card and license](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
and Token Factory service terms for the text model as well.
