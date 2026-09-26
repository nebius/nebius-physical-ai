# CUDA regression workflow proof

This validation payload performs real CUDA forward/backward/SGD training,
preserves model and momentum tensors, and independently verifies every update on
CPU. It tests workflow execution and durable replay. It does not measure robot
policy quality or prove Ray, distributed training, or optimizer checkpoint resume.

`cuda-regression.yaml` is an ordinary two-state `npa.workflow/v0.0.1`. `train.py`
requires the exact pinned Torch 2.13.0 CUDA 13 image and a real CUDA device; there
is no CPU fallback. `verify.py` downloads the artifacts from S3, verifies exact
hashes and lengths before weights-only checkpoint decoding, reconstructs the
model and optimizer, then replays SGD using independent scalar arithmetic.

The frozen recipe uses eight features, 1,024 generated training examples, 1,024
separate held-out examples, seed 7, 32 steps, learning rate 0.1 and momentum 0.8.
Double precision allows tight comparison between CUDA tensors and scalar CPU
recomputation. The expected final training loss is approximately 0.00165440 and
held-out loss 0.00155750. The acceptance gate requires at least a 50-fold loss
reduction and agreement of every gradient norm, parameter change, final weight,
and momentum value within an absolute/relative tolerance of 1e-8. Throughput and
step timings come from synchronized real CUDA execution.

Build the deterministic archive with the repository interpreter:

```bash
npa/.venv/bin/python /path/to/gpu-proof-spec/pack_payload.py
```

Stage `payload.pyz` to one exact operator-approved S3 key, verify its bytes, then
supply its URI and the SHA-256 from `payload-manifest.json`. Both workers download
and verify the complete archive before executing it. Source is imported directly
from the zip; no archive extraction occurs. The `train.py` and `verify.py` scripts
can also be invoked directly with their adjacent modules. Their `--help` commands
do not import Torch or access the network.

Required configuration overrides:

- `bucket` and `prefix`: the exact preflighted workload storage destination.
- `candidate_source_sha`: the final host/runtime NPA candidate's full Git SHA.
- `payload_uri` and `payload_sha256`: the staged archive's exact location and hash.
- `post_output_verification`: `none` by default, or `until-release` for an explicit
  cancellation experiment.
- `release_uri`: defaults to this run's `control/release.json`; consumed only in
  `until-release` mode.

Pass `--stage-src` explicitly on every submit and resume: this upstream vendor
image does not bake NPA, and explicit vendor-image selection does not trigger
automatic image-less source staging. Assert `NPA_SRC_S3_URI` is the final
candidate's content-addressed source prefix and retain it for replay.
SkyPilot's generic Python may be its management runtime,
so both stages create a fresh private venv from the image's `/usr/bin/python3.12`
using `--system-site-packages --without-pip`, then install `boto3==1.43.89` through
that image interpreter's pip. This preserves the image's Torch 2.13.0+cu130.
The dependency installation requires package-index reachability and is distinct
from the immutable image's bytes; no fully locked transitive runtime is claimed.
Measured boto3 and botocore versions are included with the training metrics.
AWS credentials enter through `--secret-env`;
`AWS_ENDPOINT_URL` must match the endpoint verified by normal NPA submission.
The pinned image runs as root inside its unprivileged container, so the selected
namespace policy must permit that ordinary image account. No privileged pod or
host modification is required.

Normal submission uses `--runtime`, a unique run ID, and owned isolated SkyPilot
state. Disable the default observation deadlines with `--max-wait-seconds 0`
and `--image-bootstrap-timeout-seconds 0`; do not introduce an operational spend
or job-count cap. Preserve the same state root, spec and effective configuration
for resume. `NPA_TASK_IMAGE` must equal the pinned image reference; changing image
selection is a negative test, not permission to bypass the runtime pin.

The training stage publishes `checkpoint.pt` and `metrics.json`, readbacks both,
then publishes `manifest.json` last. The verification stage independently reads
those exact objects and publishes `validation.json`. No local files are shared
between stages. The root executor must also capture actual provider IDs, pod
image IDs, device allocation, durable attempts and queue evidence; self-reported
runtime metadata alone is not proof of physical execution.

For a controlled cancellation experiment, `until-release` keeps the completed
model fixed and evaluates fresh generated CUDA batches after publishing verified
training outputs. It writes measured `post-output-progress.json` outside the
immutable artifact manifest. It performs no further optimizer updates. To finish
naturally, upload the following exact object to `release_uri`, using the run's
actual values:

```json
{"action":"release","run_id":"<run-id>","source_sha":"<candidate-sha>","payload_sha":"<payload-sha256>"}
```

Only a missing-key response keeps the phase running; authorization/network errors
fail immediately. The root may cancel the exact owned provider job after seeing
`OUTPUTS_VERIFIED` and measured post-output progress. Label resulting evidence
as an injected cancellation/replay experiment after output publication, never
as mid-optimizer checkpoint resume.

`test_artifact_validation.py` uses explicitly synthetic CPU fixtures. It tests
corrupt checkpoint/metrics bytes and hashes, invalid optimizer state and journal,
cross-run bindings, and an invalid CPU-device claim. These tests are not GPU
evidence; Torch loading and actual training remain live-run prerequisites.

No cloud operation is performed by `pack_payload.py` or the CPU tests. See the parent evidence report for measured execution status.
