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
  optional first-episode MP4. `--device cpu` is supported for local evaluation.
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
- `workbench.mjlab.train`, `.eval`, `.export` use real CLI flags and declared S3
  outputs. H100 is the state-based workflow default; camera/video workloads need
  a compatible rendering GPU/runtime.
- The dedicated Dockerfile has a hash-locked Linux Python 3.12 CUDA wheel closure.
  It is **unbuilt and publication-quarantined**, not a public supported image.
  Require an explicit operator-built image override until exact-image security,
  license, bootstrap and GPU capability gates pass. Do not route to SONIC's image.
- Keep orchestration in declarative workflow YAML, not a SONIC Python runner.

## Validation

Use the repository Python and run MJLab CLI/workbench/workflow tests, toolRef
argv and image-routing guards, packaging tests, skill checks and docs drift.
The golden eval `npa.smoke.test_mjlab_functional` performs real CUDA training,
checkpoint reload, measured episodes and ONNX validation. The live test
`npa/tests/e2e/test_mjlab_live.py` adds S3 hash readback; configure
`NPA_MJLAB_E2E_OUTPUT_PATH` on an authorized GPU runtime. Do not report mocked
or CPU-only evaluation as GPU training qualification.
