# Default vision-model capability regression

This proof pack supports one narrow claim: the observed hosted
`MiniMaxAI/MiniMax-M3` deployment accepted the included synthetic image and
correctly returned `left=red, right=blue`. It also anchors regression tests so
the repository default cannot move to another model while its test oracle
moves in lockstep.

## Objective result and controls

`probe.png` is an exact 112-byte, 64x32 RGB PNG. Its left 32 columns contain
only `(255, 0, 0)` and its right 32 columns contain only `(0, 0, 255)`.
The prompt does not name either target colour.

| Role | Model | Result |
| --- | --- | --- |
| Reviewed default | `MiniMaxAI/MiniMax-M3` | HTTP 200, exact returned model, `finish_reason=stop`, correct answer |
| Positive control | `google/gemma-3-27b-it` | HTTP 200, exact returned model, `finish_reason=stop`, same correct answer |
| Negative control | `Qwen/Qwen3-235B-A22B-Instruct-2507` | HTTP 400, image input rejected |
| Negative control | `Qwen/Qwen3-30B-A3B-Instruct-2507` | HTTP 400, image input rejected |

The exact sanitized facts and hashes are in
[`evidence.json`](evidence.json). No provider request was repeated to create
this pack.

## Hardware and applicability

| Evidence | Execution | Hardware | Applies to |
| --- | --- | --- | --- |
| PNG bytes and decoded pixels | Deterministic local inspection | CPU; host identity not retained | Exact submitted input |
| Reviewed-default response | Hosted inference | Provider hardware unknown | This model, deployment, prompt, and image observation |
| Positive and negative controls | Hosted inference | Provider hardware unknown | Image-dependence control for this observation |
| Protected live regression | Scheduled repository tests | Hosted runner plus provider hardware unknown | Current catalog membership and real caption, reason, visual-judge, and attribute image paths |

The input-to-response association is retained by the collector; it is not a
server-attested provenance link. The public pack intentionally excludes
request identifiers, endpoints, infrastructure identifiers, credentials,
provider fingerprints, raw reasoning, and raw error bodies.

## Limitations

This is an illustrative synthetic capability smoke, not a model-quality
benchmark or a real robotics result. It does not establish safety, policy
success, geometry correctness, temporal understanding, global availability,
or suitability for a customer workload. Catalog membership alone is not
image-capability evidence; the protected live tests require both membership
and successful image-bearing artifacts.
