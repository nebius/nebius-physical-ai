# MJLab

MJLab combines MuJoCo Warp simulation with RSL-RL policy training. NPA wraps
**MJLab 1.6.0** with one implementation reached through the CLI, Python SDK,
authenticated HTTP service, and declarative workflows.

Use a Nebius GPU workflow for training. The image recipe is
`npa/docker/workbench/mjlab/Dockerfile`; its pinned dependency closure includes
PyTorch's CUDA 12.8 wheels. **The MJLab image is not a published release.** Build
an operator image and supply its complete reference in the workflow GPU resource
profile (`image_id: docker:<your-image>`). Public promotion remains quarantined
until exact-image scans and real GPU qualification pass.

```bash
npa workbench workflow validate-spec workflows/testing/mjlab-train-eval.yaml
npa workbench workflow plan-spec workflows/testing/mjlab-train-eval.yaml
```

Customize the workflow's bucket, task and training settings, and override the
GPU image before submission. Its four stages train a native policy, evaluate
complete episodes, export ONNX, and evaluate the same checkpoint with seed 43.
Every stage exchanges actual artifacts through S3. The evaluation-only template
is `workflows/testing/mjlab-eval.yaml` and requires an existing native checkpoint.
Neither template is advertised as a completed GPU qualification run.

## Capabilities

Inside the GPU image, or after `pip install -e 'npa[mjlab]'` on a compatible
Python 3.10–3.13 environment:

```bash
npa workbench mjlab list
npa workbench mjlab system-info
npa workbench mjlab train --task Mjlab-Velocity-Flat-Unitree-G1 \
  --num-envs 4096 --output-path "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/train/"
npa workbench mjlab eval --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/train/checkpoint.pt" \
  --episodes 8 --output-path "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/eval/"
npa workbench mjlab export --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/train/checkpoint.pt" \
  --output-path "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/export/"
```

`train` preserves upstream task defaults when `--iterations`, `--num-envs` and
`--learning-rate` are omitted. `--checkpoint` resumes native RSL-RL training,
including optimizer state. `--gpu-count` uses upstream torchrunx on the GPUs
already assigned to one node; it does not provision nested resources. Multi-node
training is not exposed. Set the workflow resource GPU count accordingly.

`list` reads the installed upstream registry, including Cartpole, G1/Go1 velocity,
G1 tracking and YAM manipulation tasks. It fails if MJLab is missing; `status`
and `system-info` can report missing dependencies without importing the simulator.
Camera tasks require a compatible rendering runtime. Headless state-based
workflows default to H100; use a graphics-capable GPU for `eval --video`.

Tracking tasks require `--input-path s3://.../motion.npz` in **MJLab's native
motion format**. Raw SONIC retargeting pickles and arbitrary checkpoint JSON are
not interchangeable with MJLab artifacts. The generic toolRefs cover native
state-based tasks; tracking is available through CLI/API/SDK with an explicit
motion object. Custom executable task modules and arbitrary Python overrides
are not accepted through the service.

## Results and measurement

- Training publishes `checkpoint.pt`, upstream checkpoints, parameters and
  TensorBoard events, followed by `mjlab_train.json`.
- Evaluation publishes `episodes.json`, optionally `rollout.mp4`, and
  `mjlab_eval.json`. Each completed episode records return, length and survival.
- Export publishes checked `policy.onnx` with task/version metadata and, for
  compatible joint-position robot policies, upstream robot metadata, followed
  by `mjlab_export.json`.

Manifests include dependency versions, task, seed, input hashes and output
hashes. The manifest is uploaded last. Use a fresh run-scoped output prefix.
`score` is the measured fraction reaching the task horizon without a terminal
failure; `passed` compares it with `--success-threshold`. **Survival is not task
success**: Cartpole can reach its horizon with a low return. Compare the measured
return and your task-specific objective before judging policy quality. Workflows
continue when evaluation completes with `needs_iteration`; consumers may apply
their own decision gate to the report.

Evaluation restores the training episode horizon and terminations because
upstream interactive play can use infinite episodes. Fixed per-environment
quotas avoid favoring fast failures when vector environments finish at different
times. Recurrent policy state resets on episode completion. `--video` records
only the first environment's first episode. `--device cpu` supports evaluation
and export on a workstation; training requires CUDA.

`--dry-run` and `NPA_DRY_RUN=1` produce a plan with `executed: false`, with no
scores, simulation or storage writes. The former `--score`, `--suite` and
`--embodiment` placeholder options are removed. Select the real task with
`--task`; the old SONIC workflow now uses SONIC's native export/evaluation path.
Public handoffs require S3 paths. JSON output supports `--output-format json`
and the compatibility spelling `--output json`.

## Service and SDK

The image starts `uvicorn npa.workbench.mjlab.service:create_app --factory` on
port 8080. `/health` is public; `/status`, `/system-info`, `/list`, `/train`,
`/eval` (`/run` alias), and `/export` require `Authorization: Bearer ...`.
Set `MJLAB_TOKEN` and the comma-separated `MJLAB_ALLOWED_S3_ROOTS`. An empty
scope authorizes no data access. GPU operations are synchronous and serialized;
a concurrent request receives HTTP 409. There is no durable HTTP job queue;
SkyPilot workflows own durable orchestration and cancellation.

`npa workbench mjlab deploy` applies a private ClusterIP Deployment/Service in
an existing namespace (default `workbench`). It requires `--image`,
`--accelerator` (the exact `nvidia.com/gpu.product` label), `--token-secret`,
`--storage-secret`, and `--allowed-s3-roots`. The token Secret must contain
`MJLAB_TOKEN`; the storage Secret supplies `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and `AWS_ENDPOINT_URL`. `--dry-run` returns the concrete
manifest. The command provisions no nodes and creates no Secrets. Use a
loopback port forward or HTTPS to call the service with `--endpoint` and,
optionally, `--token-env` (default `MJLAB_TOKEN`). Redirects are not followed.

```python
from npa.sdk.workbench import mjlab

request = mjlab.EvalRequest(
    task="Mjlab-Velocity-Flat-Unitree-G1",
    checkpoint="s3://your-bucket/run/train/checkpoint.pt",
    output_path="s3://your-bucket/run/eval/",
)
report = mjlab.eval(request, endpoint="http://127.0.0.1:8080")
```

Checkpoint loads enforce PyTorch weights-only decoding, including upstream
loaders that request unrestricted pickle. Unsupported checkpoints fail rather
than retrying unsafe deserialization. Task/model mismatches fail during native
state-dict loading; there is no assumed SONIC-to-MJLab adapter.

## Validation

The offline tests cover shared CLI/SDK/API behavior, storage authorization,
checkpoint isolation, measured episode accounting and workflow routing.
`python -m npa.smoke.test_mjlab_functional` executes a real one-update GPU
Cartpole train → checkpoint reload → two episodes → ONNX export smoke.
For real S3 readback and hash verification, run
`npa/tests/e2e/test_mjlab_live.py` with `NPA_INTEGRATION_E2E=1` and
`NPA_MJLAB_E2E_OUTPUT_PATH` set to an authorized disposable prefix on a GPU
runtime. The training workflow is registered in the live matrix but excluded
from unattended rotation while the image is quarantined.

Upstream references: [MJLab source](https://github.com/mujocolab/mjlab),
[training API](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html),
[installation](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html).
