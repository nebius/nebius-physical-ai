# Post-train a warehouse vision model from Antioch data

The [post-training workflow](../../workflows/testing/antioch-posttrain.yaml)
reads an authentic Antioch warehouse recording from Nebius S3, fine-tunes
TorchVision's ImageNet-pretrained ResNet-18, and writes checkpoints and an
independent held-out evaluation back to S3. It recognizes conveying, pickup,
transfer, placement, and retract from RGB frames. It does not train robot actions.

The source is the [warehouse simulation](antioch-warehouse.md). The recording
must include every frame's measured simulation time, controller phase, and
placement count. Three final PNGs alone are insufficient training data.
Recorded input, execution, and test evidence is in the
[validation report](../testing/antioch-posttrain.md).
The verified Nebius run reached 89.7% held-out accuracy and 0.811 macro F1;
the report includes the weaker retract class and independent checkpoint readback.

```text
Antioch native recording → Nebius S3 source bundle
  → prepare (CPU) → S3 dataset
  → post-train (GPU) → S3 checkpoint
  → evaluate (GPU) → S3 metrics and predictions
```

## Source contract

Provide a sealed directory containing these files:

```text
warehouse-native-raw.mp4  # 1280×720, 30 fps native rendered recording
video-frames.json        # ordered frame, sim_s, phase, placed records
evidence/               # actual warehouse JSON measurements and three PNGs
checksums.json          # SHA-256 for every other file, using relative paths
```

`npa.workflows.antioch_posttrain.artifacts.publish(directory, destination)` seals
and publishes a local recording bundle using `StorageClient`, reads every upload
back, and writes the manifest last. `destination` is a fresh operator-selected
S3 prefix. It uses `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY`; configure them privately for the selected Nebius project.
The publisher does not create storage or change IAM. Verify the selected project's
bucket ownership and exact output-prefix access before publishing.

The native recorder is an external observer of the simulation, not a browser
screen recording. Each `video-frames.json` entry has zero-based `frame`, finite
increasing `sim_s`, a supported warehouse `phase`, and the actual `placed` count.
Its last frame can be `BATCH_COMPLETE` after six placements. Reuse measured state
labels; never infer labels from the walkthrough's text overlays.

## Run on Nebius

Run submission, monitoring, and cleanup on the same
[Linux operator host](../orchestration/skypilot-setup.md). Select a project with
writable S3 storage and a compatible GPU worker. The spec reuses the existing
LeRobot 0.6.0 container's PyTorch/TorchVision runtime at the immutable digest
recorded in `lerobot_version_manifest.json`; it runs TorchVision directly.
This variant includes the Kubernetes startup prerequisites. The default CUDA 13
LeRobot 0.5.1 image failed before stage execution and is not this workflow's pin.
No new image or LeRobot policy is introduced. Its RTX PRO 6000 default can reuse
the simulation cluster; an H100 is also compatible with this non-rendering stage.

```bash
npa workbench health preflight --project '<project>' --checks s3,nebius
npa workbench workflow validate-spec workflows/testing/antioch-posttrain.yaml
npa workbench workflow preflight-images workflows/testing/antioch-posttrain.yaml
npa workbench workflow submit workflows/testing/antioch-posttrain.yaml \
  --project '<project>' --infra 'k8s/<context>' \
  --run-id '<new-run-id>' --s3-bucket '<bucket>' \
  --var 'source_uri=s3://<bucket>/<recording-prefix>/' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

`source_overlay: true` stages the current NPA source inside the existing runtime.
The input URI must be worker-readable and end in `/`. Source is immutable for
one experiment. Use a new run ID and prefix for each training experiment.
The default output root is `s3://<bucket>/antioch-posttrain/<run-id>/`.

| Prefix | Real outputs |
| --- | --- |
| `dataset/` | `frames.npz`, `records.json`, `dataset.json`, `checksums.json` |
| `checkpoint/` | `model.pt`, `baseline.pt`, `training.json`, `selection.json`, `checksums.json` |
| `reports/` | `evaluation.json`, `checksums.json` |

The worker interfaces are `python -m npa.workflows.antioch_posttrain prepare`,
`train`, and `evaluate`. Each accepts `--input-path` and `--output-path` with
local scratch paths or S3 prefixes. Evaluation additionally takes
`--checkpoint-path`. Public workflow handoffs use S3.

## Training and evaluation contract

Preparation first recomputes all 21 physics/image acceptance checks. It rejects
corrupt manifests, invalid or unordered telemetry, video/frame-count mismatch,
and any split missing an operation class. The full image is resized to 224×224;
labels remain tied to actual controller state.

Carton cycles 0–3 train the model, cycle 4 selects the checkpoint, and cycle 5
is evaluated afterward. Retract belongs to the carton just placed. Frames are
never randomly split across these boundaries. This is a within-scene evaluation
from one run, not evidence of transfer to new scenes, cameras, or real robots.

The default `frame_stride=3` samples at 10 Hz. `epochs=20`, `batch_size=32`, and
`seed=42` are configurable through `--var`. AdamW updates the entire backbone at
0.0001 and the five-class head at 0.001, with class-balanced cross entropy and
training-only brightness augmentation. Validation macro F1 selects `model.pt`;
the test split never selects a checkpoint. Backbone hashes must change.

`baseline.pt` retains ImageNet weights and the same seeded **untrained** five-class
head. It is an unadapted baseline, not a separately optimized linear probe.
The report compares it with the selected model, includes class confusion and
every test prediction, and requires the fine-tuned model to beat both baseline
macro F1 and majority-class accuracy. A failed quality gate still publishes its
report and exits nonzero. All outputs carry content hashes and dataset lineage.

## Upstream artifacts

The implementation uses the native
[TorchVision ResNet-18 API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
and standard [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
TorchVision source is [BSD licensed](https://github.com/pytorch/vision/blob/main/LICENSE).
The official `ResNet18_Weights.IMAGENET1K_V1` checkpoint is fetched at runtime
from PyTorch; it is not baked into an image or committed to Git. Its verified
SHA-256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Source licensing is separate from training-data and weight provenance. The
source recording and fine-tuned weights remain in the operator's private storage;
this change does not publish them or redistribute NVIDIA warehouse assets.

## Live regression input

The GPU submit matrix requires `NPA_E2E_ANTIOCH_SOURCE_DIR` pointing to a sealed
real local recording bundle. It stages those exact bytes beneath the test run's
source prefix. A missing bundle fails before training; synthetic frames are not
substituted for the production input.
