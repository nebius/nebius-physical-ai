# XR1 robot learning with Antioch and Nebius

This workflow fine-tunes **Xiaomi-Robotics-1 5B** on simulated dual-Franka
pick-and-place demonstrations. Three RGB cameras, joint positions, gripper
apertures, and the Cartesian commands actually issued to the robot are recorded
together. The trained artifact is a robot action policy. This experiment does
not establish real-robot transfer or general manipulation competence.

The data route is:

1. The operator stages pinned source, a sealed episode split, the base model,
   and its processor in **Nebius S3**.
2. **Antioch** downloads approved inputs using short-lived signed GETs and checks
   their complete SHA-256 hashes. Isaac Sim generates physical demonstrations.
3. Antioch sends recordings directly to **Nebius S3** using signed PUTs. The
   operator reads every object back and checks hashes, camera frames, and clocks.
4. The **Workbench workflow on Nebius GPUs** reads those recordings, computes
   normalization from successful training episodes, and updates XR1 with its
   native flow-matching and action-choice objectives.
5. Checkpoints, optimizer state, validation metrics, and provenance go to
   **Nebius S3**. Antioch fetches the selected checkpoint and normalization.
6. The base and trained policies act in the same untouched test scenarios.
   Their videos, measured robot outcomes, and paired comparison return to S3.

Static S3 credentials stay on the operator and the Nebius training worker.
They are not copied into the Antioch project, source bundle, or simulator.
The training YAML covers step 4 and checkpoint publication; the operator module
provides the Antioch connection and artifact transfers around it.

## Antioch authentication and Workbench integration

Workbench invokes the installed Antioch CLI from your own Antioch project. Store
an Antioch **personal access token** in the standard private NPA credential file,
`~/.npa/credentials.yaml` (or `$NPA_CONFIG_DIR/credentials.yaml`):

```yaml
tokens:
  ANTIOCH_TOKEN: "<your-antioch-personal-access-token>"
```

Protect the file with `chmod 600 ~/.npa/credentials.yaml`. If your secret manager
already supplies `ANTIOCH_TOKEN`, `npa configure --save-env-credentials` persists
supported environment credentials with an atomic 0600 write and reports names
only. It can also save other supported credentials present in that environment.
Do not put the token value in shell arguments or history.

The Workbench operator resolves `ANTIOCH_TOKEN` from the environment first, then
`tokens.ANTIOCH_TOKEN` from the NPA file. It injects the result only into local
Antioch CLI/SDK client processes; the saved token is excluded from generic
Workbench credential exports and training workflow parameters. If neither is
configured, the CLI can reuse an existing `antioch auth login` browser session.
Browser access/refresh credentials stay in Antioch's own store; do not copy
them into the personal-access-token field. Managed Antioch workspaces retain
the SDK's managed-credential precedence.

Use the operator wrapper when a native CLI command needs the NPA-stored token:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator antioch \
  --antioch-project "$XR1_ANTIOCH_PROJECT" -- auth whoami --json
```

Run identity checks privately because they print user and organization details.
A bare `antioch` invocation does not read NPA's credential file. The `exec`
operator command below supplies the same credentials to the pinned SDK adapter
for evaluation.

An Antioch token grants access according to its permissions. You also need an
owned Antioch project, the compatible SDK/engine, simulator capacity, and a
separately configured NPA project with Nebius and S3 access. Use SDK 0.4.236 for
this recipe, including the attached evaluation adapter. Never place token
values in YAML, command arguments, source archives, recordings, or Git.

The operator creates short-lived signed S3 GET/PUT URLs and passes them over
stdin to Antioch. Antioch downloads inputs or uploads native outputs directly;
complete readbacks verify hashes. Signed URLs also grant access and must stay
private. Static S3 keys stay with the operator and Nebius training worker.

Use the [Antioch Workbench skill](../../../skills/workflows/antioch-workbench/SKILL.md)
for authentication, runtime boundaries, operating steps, and evidence checks.

## Task and learning contract

Two Franka arms must grasp their respective blocks, lift them above 12 cm,
place them within 3.5 cm of their green targets, release them, and settle.
Object motion comes from finger contact. There are no welded objects, pose
teleports, or expert actions during policy evaluation.

Physics runs at 60 Hz; camera/state/action samples run at 20 Hz. XR1 predicts
30 Cartesian action targets. Evaluation executes six targets before observing
again, with a 25-second simulation horizon. Simulation waits for inference;
the recordings do not imply real-time inference at 20 Hz. The controller rejects
out-of-workspace targets and saturates grippers at their mechanical limits.
Those events remain in the outcome trace.

The reference split uses 64 training seeds, 16 validation seeds, and 32 paired
test seeds. Whole episodes and seeds are disjoint. Failed expert demonstrations
remain evidence and are excluded from behavior cloning. Checkpoint selection
uses disjoint validation loss. **Task improvement requires the closed-loop
comparison**, including every failed test episode. The report gives success
rates, Wilson intervals, and the exact paired McNemar test.

The recipe uses upstream's 10,000 optimizer steps, bf16, DeepSpeed ZeRO-2, eight
GPUs, two samples per GPU, and its learning-rate schedule. It disables XR1's
asynchronous-action training mode to match this synchronous simulator. It
fine-tunes the native model, retaining upstream's frozen input embeddings.

## Measured RTX PRO 6000 run

The recorded run completed 10,000 native optimizer steps on eight RTX PRO 6000
GPUs. Of 64 training demonstrations, 60 passed the physical success checks and
were used for behavior cloning; all 16 disjoint validation demonstrations
passed. The minimum held-out loss selected step 9,000 (0.220131), compared with
6.915094 for the base model and 0.227057 at step 10,000. Test outcomes did not
choose the checkpoint.

| Policy | Complete task successes | Success rate | Wilson 95% interval |
| --- | --- | --- | --- |
| Pinned XR1 base | 0/32 | 0% | 0–10.72% |
| Selected step-9,000 XR1 | 8/32 | 25% | 13.25–42.11% |

The paired comparison has eight wins and zero losses (exact two-sided McNemar
p = 0.0078125). Fifteen candidate episodes request an out-of-workspace target;
nine further episodes miss the physical task criteria by the simulation
horizon. All 24 failures remain in the report. This supports improvement on
these held-out scenes, while the absolute success rate remains low.

Antioch's default command deadline interrupted evaluation after 25 completed
trials. Their native artifacts were retained and hash checked unchanged; only
the seven unfinished trials were executed through the attached SDK adapter.
The interrupted partial attempt is retained separately, and prediction noise
still uses each rollout's original request seed.

The complete measured record, including every paired test outcome, checkpoint
identity, validation history, and video hashes, is in
[the execution evidence](../evidence/xr1-antioch-rtxpro.json).
All 192 native policy camera videos were downloaded from S3, hash checked, and
fully decoded; their frame counts and 20 Hz simulation timestamps match the
rollout reports. The selected checkpoint and native optimizer states were also
published to S3 and completely read back.

This is one training run on a narrow simulated task. The policy is not ready
for robot deployment, and no physical robot was tested. Inference pauses the
simulator. The retained operational recording shows real S3 reads, the original
verified Antioch checkpoint receipt, the complete paired results, and actual camera
playback. The displayed rollout is selected as the first successful sealed
seed; it does not replace the full-cohort result.

The original Antioch session lost its backing machine after evaluation and S3
publication completed. The final recording reads the durable results and
original transfer receipt from S3; it does not claim a fresh live simulator
probe. Earlier connection recordings preserve the actual transfers and execution.

The execution evidence records the exact training source archive. Subsequent
checkpoint-loader and HTTPS restrictions were verified separately against the
actual artifacts, including identical predictions with restricted decoding;
the full optimizer execution used the recorded archive.

## Prerequisites and immutable inputs

Use your own Linux operator checkout and configured NPA project. Complete
[health preflight](../../../skills/atomic/health-preflight/SKILL.md) and provision
an eight-GPU **RTX PRO 6000** node through the normal
[cluster procedure](../../../skills/tools/gpu-cluster-provisioning/SKILL.md).
The SM120 FlashAttention wheel is specific to this GPU/runtime combination;
B200 requires a separately built and verified runtime.

The implementation pins:

| Input | Revision |
| --- | --- |
| [XR1 source](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| [XR1 5B checkpoint](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) | `ee21d524b5c52ac961d941e1bc7d6d92836c3d5e` |
| [Qwen3-VL processor](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| PyTorch training image | `pytorch/pytorch@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6` |
| Policy runtime | Python 3.11, PyTorch 2.8.0+cu128, FlashAttention 2.8.3 |
| Antioch engine | `antioch-engine/isaac-sim-6.0.1:0.4.236` |

XR1 source and weights and the Qwen processor declare Apache-2.0 licensing.
Isaac Sim remains subject to its applicable vendor terms. Source and model
payloads are fetched at runtime into operator-controlled storage; this change
does not publish a derivative vendor image or model.

Set private operator variables, using a new run prefix:

```bash
export XR1_PROJECT='<configured-npa-alias>'
export XR1_BUCKET='<your-bucket>'
export XR1_PREFIX="s3://$XR1_BUCKET/<new-run-prefix>"
export XR1_RUN_DIR='<absolute-private-directory>'
export XR1_ANTIOCH_PROJECT='<absolute-own-antioch-project>'
export XR1_ANTIOCH_PYTHON='<python-in-the-antioch-sim-0.4.236-environment>'
export XR1_WORKBENCH='<absolute-own-workbench-checkout>'
export XR1_KUBE_CONTEXT='<owned-cluster-context>'
mkdir -p "$XR1_RUN_DIR"
chmod 700 "$XR1_RUN_DIR"
```

Keep concrete infrastructure values, credentials, logs, and transfer receipts
outside Git. The commands below run from the Workbench checkout unless noted.

## Seal the cohort and stage the model

Create the split **before collecting or inspecting outcomes**:

```bash
npa/.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
splits = {name: [{"episode_id": f"{name}-{seed}", "seed": seed} for seed in seeds]
          for name, seeds in [("train", range(1000, 1064)),
                              ("validation", range(2000, 2016)),
                              ("test", range(3000, 3032))]}
with (Path(os.environ["XR1_RUN_DIR"]) / "split-manifest.json").open("x") as out:
    json.dump(splits, out, indent=2)
PY
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" assets \
  --s3-uri "$XR1_PREFIX/assets" --output-path "$XR1_RUN_DIR/assets"
```

The model download verifies the pinned 10.2 GB checkpoint before deserialization.
`assets.json`, `checksums.json`, and `manifest.json` retain identities. Publication
performs a complete S3 readback.

## Prepare Antioch and collect

Create an operator-owned Antioch project using SDK 0.4.236 and the Isaac Sim
6.0.1 engine. Its simulator service needs `resources: {gpu: rtx-pro-6000}`.
Preserve the ID produced by `antioch init`; do not copy an existing project's ID.
The project Dockerfile should include the compiler required by Torch/Triton:

```dockerfile
FROM antioch-engine/isaac-sim-6.0.1:0.4.236
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libaio-dev
ENV ANTIOCH_PROJECT_DIR=/workspace/project
WORKDIR /workspace/project
COPY . .
```

Build/start the project using the installed Antioch CLI. Source transfer uses
S3 and the verified bootstrap worker, so it does not depend on interactive
rsync/watch support:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" source \
  --antioch-project "$XR1_ANTIOCH_PROJECT" \
  --s3-uri "$XR1_PREFIX/source" --output-path "$XR1_RUN_DIR/source" \
  --remote-root /workspace/project/xr1-source \
  --split-manifest "$XR1_RUN_DIR/split-manifest.json"
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" collect \
  --antioch-project "$XR1_ANTIOCH_PROJECT" \
  --source-root /workspace/project/xr1-source/source \
  --remote-root /workspace/project/outputs/dataset \
  --s3-uri "$XR1_PREFIX/dataset" --output-path "$XR1_RUN_DIR/dataset" \
  --split-manifest "$XR1_RUN_DIR/split-manifest.json"
```

`collect` seals the split in S3, runs both demonstration cohorts, preserves failed
episodes, and verifies all three videos and action/state arrays. It resumes
completed recordings and interrupted transfers without replacing different S3
bytes. Test seeds are reserved for policy evaluation, never collected as expert
training targets.

## Prove the GPU runtime, then fine-tune

Generate and submit the one-GPU runtime recipe through Workbench:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" runtime \
  --s3-uri "$XR1_PREFIX/runtime" --output-path "$XR1_RUN_DIR/runtime-recipe"
npa/.venv/bin/npa workbench workflow submit "$XR1_RUN_DIR/runtime-recipe/runtime.yaml" \
  --project "$XR1_PROJECT" --infra "k8s/$XR1_KUBE_CONTEXT" \
  --run-id '<new-runtime-run-id>' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

The private recipe stages only `runtime.py`, uses the pinned PyTorch image,
requests one RTX PRO 6000 with 32 CPUs and 256 GiB memory for compilation, and
declares the S3 output directory. It contains the storage endpoint and run
prefix, without storage keys. The worker installs Git, a C/C++ compiler, and
libaio; builds FlashAttention for SM120; runs real forward/backward and
FusedAdam checks; and publishes and reads back the wheel, package inventory,
and runtime proof. Training verifies these artifacts again before use.

Validate and submit the checked-in training graph:

```bash
npa/.venv/bin/npa workbench workflow validate-spec workflows/partners/antioch/xr1-antioch-finetune.yaml
npa/.venv/bin/npa workbench workflow submit workflows/partners/antioch/xr1-antioch-finetune.yaml \
  --project "$XR1_PROJECT" --infra "k8s/$XR1_KUBE_CONTEXT" \
  --run-id '<new-training-run-id>' --runtime --stage-src \
  --var "bucket=$XR1_BUCKET" --var "dataset_uri=$XR1_PREFIX/dataset" \
  --var "assets_uri=$XR1_PREFIX/assets" --var "runtime_uri=$XR1_PREFIX/runtime" \
  --var "training_uri=$XR1_PREFIX/training" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

For a previously verified source upload, supply `NPA_SRC_S3_URI` and omit
`--stage-src`. Preserve the isolated SkyPilot configuration used by the owning
controller. An interrupted launch must use the exact run's `--resume-run`;
that flag replaces `--run-id`.

Before fitting, the worker publishes `training/preparation/normalization.json`,
`dataset-report.json`, and the native configuration. This allows baseline
inference to use the exact training statistics. During training it records the
base validation loss and validates every 1,000 optimizer steps. On completion,
it selects the minimum validation-loss checkpoint, verifies that parameters
changed from the exact base weights, and publishes the candidate, native
optimizer checkpoints, CSV metrics, and validation JSONL. A failed worker does
not promote a checkpoint.

## Return the policy to Antioch and evaluate

Use `operator fetch` for the base assets, verified runtime wheel, and training
preparation artifacts. For example:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" fetch \
  --antioch-project "$XR1_ANTIOCH_PROJECT" --s3-uri "$XR1_PREFIX/assets" \
  --remote-root /workspace/project/xr1-assets --output-path "$XR1_RUN_DIR/assets-input-receipt.json"
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" fetch \
  --antioch-project "$XR1_ANTIOCH_PROJECT" --s3-uri "$XR1_PREFIX/runtime" \
  --remote-root /workspace/project/xr1-wheel --output-path "$XR1_RUN_DIR/wheel-input-receipt.json"
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator \
  --project "$XR1_PROJECT" fetch \
  --antioch-project "$XR1_ANTIOCH_PROJECT" --s3-uri "$XR1_PREFIX/training/preparation" \
  --include normalization.json --remote-root /workspace/project/xr1-statistics \
  --output-path "$XR1_RUN_DIR/statistics-input-receipt.json"
```

Inside the Antioch service, create a separate Python 3.11.13 environment with
`uv venv --python 3.11.13 --seed /workspace/project/xr1-policy-venv`. Install
`torch==2.8.0 torchvision==0.23.0` using the official
`https://download.pytorch.org/whl/cu128` index. Then run, in that interpreter:

```bash
PYTHONPATH=/workspace/project/xr1-source/source \
  /workspace/project/xr1-policy-venv/bin/python -m npa.workflows.xr1_antioch.policy_setup \
  --work-path /workspace/project/xr1-policy-runtime --wheel-root /workspace/project/xr1-wheel
```

The simulator keeps its engine interpreter. The policy runs in its own
interpreter over a private Unix socket. The installed Antioch CLI defaults to
a 900-second command deadline, which can interrupt a complete evaluation.
Use the pinned SDK attachment adapter below to wait until the cohort finishes.
It retains the client's managed process cleanup and operator interruption;
it does not detach the simulator or impose a workload deadline.

Run the complete base cohort through the Workbench operator, which selects the
owning Antioch project and resolves its client credential:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator exec \
  --antioch-project "$XR1_ANTIOCH_PROJECT" \
  --antioch-python "$XR1_ANTIOCH_PYTHON" -- \
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=/workspace/project/xr1-source/source \
  python -u -m npa.workflows.xr1_antioch.evaluate \
  --output-path /workspace/project/outputs/baseline \
  --policy-python /workspace/project/xr1-policy-venv/bin/python \
  --assets-path /workspace/project/xr1-assets \
  --checkpoint /workspace/project/xr1-assets/base/model_states.pt \
  --checkpoint-sha256 94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7 \
  --statistics /workspace/project/xr1-statistics/normalization.json \
  --split-manifest /workspace/project/xr1-source/source/recipe/split-manifest.json
```

Fetch `candidate/model_states.pt` from `$XR1_PREFIX/training` with
`operator fetch --include candidate/model_states.pt`. Repeat the identical
cohort with its receipt's checkpoint hash and a fresh candidate output directory.
Keep the source bundle and normalization unchanged. Infrastructure failures stop
the evaluation; they do not count as policy failures.

Publish each evaluation directory with `operator publish`, supplying the
Antioch project, remote output root, fresh S3 prefix, and local receipt path.
Retrieve both `evaluation.json` files from S3 and compare them:

```bash
npa/.venv/bin/python -m npa.workflows.xr1_antioch.operator compare \
  --baseline "$XR1_RUN_DIR/baseline/evaluation.json" \
  --candidate "$XR1_RUN_DIR/candidate/evaluation.json" \
  --split-manifest "$XR1_RUN_DIR/split-manifest.json" \
  --output-path "$XR1_RUN_DIR/comparison.json"
```

The comparison rejects missing or duplicated seeds, different normalization or
simulation hashes, expert fallback, changed control rates, and changed task
horizons. Preserve the full report even when there is no improvement.

## Output locations and recording

| Prefix beneath the private run | Contents |
| --- | --- |
| `assets/` | Pinned base weights, processor, checksums, model/source identity |
| `source/` | Source bundle and sealed cohort recipe |
| `dataset/episodes/<split-seed>/` | Three native MP4s, first-frame PNGs, synchronized trajectory JSON |
| `dataset/receipts/` | S3 readback hashes, decoded video metadata, physical success/failure |
| `runtime/` | SM120 wheel, native CUDA proof, package inventory |
| `training/preparation/` | Train-only normalization, data qualification, native configuration |
| `training/candidate/` | Selected XR1 weights and parameter-change/training report |
| `training/checkpoints/` | Native DeepSpeed model and optimizer state |
| `baseline/`, `candidate/` | Every held-out rollout, actual camera videos, and cohort results |

The simulator initially writes under the explicit `/workspace/project/outputs`
path. Those files are temporary until publication succeeds. `ego.mp4`,
`wrist_left.mp4`, and `wrist_right.mp4` are actual rendered robot recordings;
`rollout.json` links frames to observations, issued predictions, and outcomes.
Download those exact objects for a raw video handoff. A screen recording of the
live CLI can additionally show transfers and job monitoring; it does not replace
the robot videos or measured evaluation.

After preserving artifacts, cancel any unfinished owned jobs before destroying
owned infrastructure, and release the Antioch session. Follow the standard
[teardown procedure](../../teardown.md); do not tear down a shared operator VM.
