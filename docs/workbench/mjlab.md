# MJLab

MJLab combines MuJoCo Warp simulation with RSL-RL policy training. NPA wraps
**MJLab 1.6.0** with one implementation reached through the CLI, Python SDK,
authenticated HTTP service, and declarative workflows.

Use a Nebius GPU workflow for training. The image recipe is
`npa/docker/workbench/mjlab/Dockerfile`; its pinned dependency closure includes
PyTorch's CUDA 12.8 wheels. **The MJLab image is not a published release.** Build
an operator image and supply its complete reference in the workflow GPU resource
profile (`image_id: docker:<your-image>`). A private operator image has passed
native B200 and RTX PRO 6000 GPU qualification. Public promotion remains quarantined until the
exact-image security, licensing and bootstrap gates pass.

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
- Evaluation publishes `episodes.json`, optionally `rollout.mp4` and the
  self-contained browser page `rollout.html`, and
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

## Rendered video in Workbench

Run evaluation on a rendering-capable GPU with `--video`:

```bash
npa workbench mjlab eval --task Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/train/checkpoint.pt" \
  --episodes 32 --num-envs 4 --seed 43 --video \
  --output-path "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/eval/"
```

This uses the same implementation as `EvalRequest(video=True)` in the SDK and
`POST /eval` with `"video": true` in the authenticated service. Native MuJoCo
renders the evaluated policy's first complete episode, and Workbench verifies
that the MP4 decodes before publishing it. The result manifest lists both
`artifacts["rollout.mp4"]` and `artifacts["rollout.html"]` with S3 URI, SHA-256
and byte count.

`rollout.html` embeds the unchanged MP4 bytes, measured evaluation results and
the input checkpoint hash. Download and open it directly, or create an
authenticated S3 GET URL with response content type `text/html` and content
disposition `inline`. The page needs no separate viewer server, JavaScript,
external media request or bucket CORS change. Keep signed links out of Git and
PR text; share them only with the intended reviewer and state their expiry.

For an NPA workflow, declare `rollout.mp4` and `rollout.html` as stage outputs.
The supplied `workflows/testing/mjlab-render.yaml` does this with the
`workbench.mjlab.render` toolRef and an RTX PRO 6000 resource profile. Supply your
trained checkpoint and explicit operator image before submission.
The Workbench artifact browser classifies the MP4 as video and provides normal
playback; HTML remains a downloadable report. This does not require an MJLab
specific UI route or executing arbitrary HTML inside the agent application.

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

### Measured GPU acceptance

The 2026-09-24 operator qualification uses the immutable development image built
from source `7ccb0e8cfbf97f414916b0f026b8f204f1277781`. Full reports and artifacts
are retained in private project storage; the
[committed qualification record](validation/mjlab-gpu-20260924.json)
contains hardware, artifact hashes and measured results without live
infrastructure identifiers. A Docker image configuration ID identifies the
tested local image; it is not an OCI registry manifest digest or a public tag.

| GPU | Native cases | Complete evaluation episodes | Verified artifact hashes |
|---|---|---:|---:|
| 8 × B200 (`sm_100`) | Cartpole, G1, Go1, YAM; Cartpole service resume; eight-GPU Cartpole | 48 | 77 |
| 1 × RTX PRO 6000 (`sm_120`) | Cartpole, G1 with MP4, Go1, YAM; Cartpole service resume | 40 | 58 |

Each task trains for two PPO updates with 64 environments, reloads its actual
checkpoint, evaluates eight complete episodes, exports ONNX, checks the model,
and downloads its S3 outputs to verify hashes and byte counts. G1 training runs
through the CLI; the other task cycles use the SDK. Service acceptance verifies
HTTP 401 without a token, HTTP 400 for an out-of-scope input, HTTP 409 during an
active training request, and successful authenticated train/resume, evaluation
and export. Resume advances optimizer state and changes actor weights. B200
telemetry confirms eight simultaneous worker processes on eight GPUs, in
addition to the native launcher's eight-worker record.

The separate golden capability smoke also passes on both GPU families and adds
two measured Cartpole episodes per family. The RTX G1 MP4 contains 68 frames, all
decoded successfully using system FFmpeg. These are functional integration tests.
Two PPO updates do not establish policy convergence: the G1 acceptance policy
has zero survival at the evaluation horizon, and that measured result is preserved.

The B200 host had one NVIDIA Container Toolkit 1.20.0 `create-symlinks` startup
failure before the application started. The unchanged image and configuration
started on retry; the record preserves that failure. Its error matches the
[upstream toolkit fix](https://github.com/NVIDIA/nvidia-container-toolkit/commit/9f4201be9e351fa560e5fc6623cdfbd31d3ec0bc).

Reproduce acceptance inside the built image with the matching source's
`npa/scripts/qualify_mjlab_gpu.py` mounted at `/opt/qualify_mjlab_gpu.py`, runtime
storage credentials, and a writable evidence mount:

```bash
python /opt/qualify_mjlab_gpu.py --family b200 --gpu-count 8 \
  --evidence-dir /evidence/b200 \
  --output-path "s3://${NPA_S3_BUCKET}/runs/${RUN_ID}/b200"
```

The evidence directory and output prefix must be fresh. `--family rtx6000
--gpu-count 1` additionally requires a fully decoded first-episode G1 MP4. Each
GPU family needs its own successful acceptance report for the same image bytes.
Full reports stay private; `acceptance.json` is suitable for a sanitized handoff.

GPU qualification here covers native state-based tasks and the HTTP service in
Docker on Nebius VMs. Kubernetes deployment, a submitted SkyPilot workflow,
motion-tracking inputs, camera-policy training, B300 and multi-node training are
outside this measured scope. The declarative workflows have offline routing and
argv coverage; they have not been submitted as part of this acceptance run.

Upstream references: [MJLab source](https://github.com/mujocolab/mjlab),
[training API](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html),
[installation](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html).
