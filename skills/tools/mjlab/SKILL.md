---
name: mjlab
description: Train, resume, evaluate, export, deploy or compose native MJLab policies through Workbench CLI, SDK, API and GPU workflows.
---

# MJLab

Read `docs/workbench/mjlab.md` for the supported MJLab 1.6.0 contract.
Behavior lives in `npa/src/npa/workbench/mjlab/`; CLI, SDK and HTTP clients share
that implementation. Keep simulator imports inside the isolated worker.

## Interfaces

- `npa workbench mjlab train`: upstream task defaults, optional iterations,
  environment count, learning rate, seed, native checkpoint resume and one-node
  multi-GPU torchrunx. The current process already owns its GPU allocation.
- `eval`: real complete episodes with measured returns, lengths and survival;
  `--video` publishes the first-episode MP4 and self-contained `rollout.html`
  with measured results and checkpoint/video hashes. `--device cpu` is supported
  for local evaluation.
- `export`: native ONNX export and checker validation, with task metadata.
- `list`: query the installed upstream task registry; never hardcode fake suites.
- `status`, `system-info`: dependency versions; absence is not availability.
- `deploy`: existing Kubernetes cluster, explicit built image and GPU product
  label, existing token/storage Secrets, private ClusterIP, S3 scope.
- `workflow`: locate the training and evaluation specs.

`npa.sdk.workbench.mjlab` exposes typed TrainRequest, EvalRequest, ExportRequest
and DeployRequest calls. HTTP capabilities require a bearer token and use the
same shared runtime. Only health is public. Set `MJLAB_TOKEN` and
`MJLAB_ALLOWED_S3_ROOTS`; reject storage outside the scope before any I/O.

## Data and evaluation

All public artifact paths are S3 URIs. Training/evaluation artifacts are native
MJLab/RSL-RL files. Tracking requires an operator-supplied MJLab motion NPZ via
`--input-path`. Do not feed raw SONIC pickles or smoke checkpoint JSON into MJLab.
No SONIC policy adapter is implied. The SONIC locomotion workflow now stays on
SONIC's own export/evaluation path.

Never synthesize scores or allow a `--score` override. Dry runs return only
non-executed plans. The reported score is survival fraction at the finite task
horizon, not task success. Preserve the measured return and per-episode records.
Fixed per-environment quotas prevent completion-order bias; reset recurrent
state on done. Force weights-only checkpoint decoding in the worker and never
fall back to unsafe pickle.

## Workflow and image routing

- `workflows/testing/mjlab-eval.yaml`: existing native checkpoint → measured report.
- `workflows/testing/mjlab-train-eval.yaml`: native train → eval → ONNX export →
  independent-seed eval. All stages route to `npa-mjlab`, including repeated eval.
- `workflows/testing/mjlab-render.yaml`: trained checkpoint → measured evaluation,
  MP4 and HTML on RTX PRO 6000 through `workbench.mjlab.render` (`eval --video`).
- `workbench.mjlab.train`, `.eval`, `.export` use real CLI flags and declared S3
  outputs. H100 is the state-based workflow default; camera/video workloads need
  a compatible rendering GPU/runtime.
- The dedicated Dockerfile has a hash-locked Linux Python 3.12 CUDA wheel closure.
  The current recipe uses Torch 2.13.0/CUDA 13.0 and setuptools >=83 to replace
  vulnerable dependency pins. Historical CUDA 12.8 evidence covers B200 and
  RTX PRO 6000, including eight-GPU training; do not transfer that evidence to
  different image bytes. See the guide for exact artifacts and measured scope.
  The image remains publication-quarantined.
  Require an explicit operator-built image override until exact-image security,
  license and bootstrap gates pass. Do not route to SONIC's image.
- Keep orchestration in declarative workflow YAML, not a SONIC Python runner.

## Validation

Use the repository Python and run MJLab CLI/workbench/workflow tests, toolRef
argv and image-routing guards, packaging tests, skill checks and docs drift.
The golden eval `npa.smoke.test_mjlab_functional` performs real CUDA training,
checkpoint reload, measured episodes and ONNX validation. The live test
`npa/tests/e2e/test_mjlab_live.py` adds S3 hash readback; configure
`NPA_MJLAB_E2E_OUTPUT_PATH` on an authorized GPU runtime. Do not report mocked
or CPU-only evaluation as GPU training qualification.

`npa/scripts/qualify_mjlab_gpu.py` runs Cartpole, G1, Go1 and YAM training,
checkpoint evaluation, ONNX validation, S3 hash readback, authenticated service
resume, and rejection of unauthorized, out-of-scope and concurrent requests.
Use `--family b200 --gpu-count 8` for native one-node torchrunx acceptance or
`--family rtx6000 --gpu-count 1` to include a fully decoded G1 rollout video.
Both require the built image and an authorized fresh S3 output prefix. Preserve
full reports privately; its `acceptance.json` omits infrastructure identifiers.

When asked for a trained-policy video, run actual training and independent-seed
evaluation on the authorized GPUs; do not substitute the acceptance harness or
golden smoke for that workload. Hand off the `rollout.html` artifact from the
measured run. It embeds the exact MP4, so a signed HTML GET with `text/html` and
`inline` response headers opens without a separate media URL. Keep the signed
link private, verify its delivered bytes and browser playback, and state expiry.
The Workbench artifact browser plays the separately declared MP4 and downloads
HTML; do not imply that arbitrary HTML executes inside the agent UI.
