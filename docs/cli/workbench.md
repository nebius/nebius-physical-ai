# `npa workbench`

## Command Tree

```text
Usage: npa workbench [OPTIONS] COMMAND [ARGS]...

Physical AI workbench tools.

Options
--help  Show this message and exit.
Commands
lerobot  LeRobot policy training, evaluation, serving, and inference.
cosmos  NVIDIA Cosmos world model serving and inference endpoints.
cosmos2  Cosmos2 transfer workflow contracts.
cosmos3  Cosmos3 omni-model generation and reasoning workflow contracts.
cosmos-curate  NVIDIA Cosmos Curator: split, transcode, motion-score, and catalog video clips.
cosmos-evaluator  Cosmos Evaluator checks plus NPA source-relative temporal and protected-appearance diagnostics.
fiftyone  Voxel51 FiftyOne dataset curation and visualization workbench.
foxglove  Foxglove embedded viewer: MCAP conversion, inspection, and SDK assets.
genesis  Genesis simulation: teacher training, demo generation, evaluation.
groot  NVIDIA Isaac GR00T humanoid foundation-model workbench.
isaac-lab  Isaac Lab simulation workbench deployment, training, and evaluation.
leisaac  LeIsaac SO101 browser teleoperation on the RTX PRO 6000 Kubernetes pool.
nurec  NVIDIA Omniverse NuRec / Neural Reconstruction Engine: sensor recordings -> 3DGUT Gaussian reconstruction -> renderable USDZ -> novel-view renders. Requires an RT-core GPU
    (L40S or RTX PRO 6000 Blackwell); never route the render path at H100/H200.
sonic  NVIDIA GEAR-SONIC whole-body-control workbench.
mjlab  MJLab locomotion policy evaluation for SONIC workflows.
lichtblick  Lichtblick (MPL-2.0) - an open-source, Foxglove-compatible MCAP / ROS-bag log viewer.
ltx2  LTX-2.5 licence surface: print the LTX-2.x Community License terms, the pinned upstream source, and the gated weights repository the operator's own Hugging Face entitlement
    unlocks.
lancedb  Deploy and query LanceDB vector-search workbenches.
detection-training  Train Faster R-CNN detectors from LanceDB materialized views.
scenario-gen  Adversarial scenario generation: mine hard scenarios that fail a policy-under-test (pluggable Isaac Lab RL backend; deterministic default).
dataset  Dataset-of-record: ingest, validate, curate, and query production sensor data.
insights  Insights: lineage graph + common metrics store over workflow-run artifacts.
vlm-eval  VLM evaluation for sim-to-real pipeline gating.
token-factory  Nebius Token Factory hosted inference (zero-GPU, OpenAI-compatible).
byof  Onboard an OSS repo as a BYOF container (Tier 0 of the OSS ladder).
workflow  Multi-stage training workflow orchestration.
health  Preflight health checks for workbench workflows.
golden-eval  Per-container golden-eval / hello-world reruns.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `lerobot` | LeRobot policy training, evaluation, serving, and inference. |
| `cosmos` | NVIDIA Cosmos world model serving and inference endpoints. |
| `cosmos2` | Cosmos2 transfer workflow contracts. |
| `cosmos3` | Cosmos3 omni-model generation and reasoning workflow contracts. |
| `cosmos-curate` | NVIDIA Cosmos Curator: split, transcode, motion-score, and catalog video clips. |
| `cosmos-evaluator` | Cosmos Evaluator checks plus NPA source-relative temporal and protected-appearance diagnostics. |
| `fiftyone` | Voxel51 FiftyOne dataset curation and visualization workbench. |
| `foxglove` | Foxglove embedded viewer: MCAP conversion, inspection, and SDK assets. |
| `genesis` | Genesis simulation: teacher training, demo generation, evaluation. |
| `groot` | NVIDIA Isaac GR00T humanoid foundation-model workbench. |
| `isaac-lab` | Isaac Lab simulation workbench deployment, training, and evaluation. |
| `leisaac` | LeIsaac SO101 browser teleoperation on the RTX PRO 6000 Kubernetes pool. |
| `nurec` | NVIDIA Omniverse NuRec / Neural Reconstruction Engine: sensor recordings -> 3DGUT Gaussian reconstruction -> renderable USDZ -> novel-view renders. Requires an RT-core GPU (L40S or RTX PRO 6000 Blackwell); never route the render path at H100/H200. |
| `sonic` | NVIDIA GEAR-SONIC whole-body-control workbench. |
| `mjlab` | MJLab locomotion policy evaluation for SONIC workflows. |
| `lichtblick` | Lichtblick (MPL-2.0) - an open-source, Foxglove-compatible MCAP / ROS-bag log viewer. |
| `ltx2` | LTX-2.5 licence surface: print the LTX-2.x Community License terms, the pinned upstream source, and the gated weights repository the operator's own Hugging Face entitlement unlocks. |
| `lancedb` | Deploy and query LanceDB vector-search workbenches. |
| `detection-training` | Train Faster R-CNN detectors from LanceDB materialized views. |
| `scenario-gen` | Adversarial scenario generation: mine hard scenarios that fail a policy-under-test (pluggable Isaac Lab RL backend; deterministic default). |
| `dataset` | Dataset-of-record: ingest, validate, curate, and query production sensor data. |
| `insights` | Insights: lineage graph + common metrics store over workflow-run artifacts. |
| `vlm-eval` | VLM evaluation for sim-to-real pipeline gating. |
| `token-factory` | Nebius Token Factory hosted inference (zero-GPU, OpenAI-compatible). |
| `byof` | Onboard an OSS repo as a BYOF container (Tier 0 of the OSS ladder). |
| `workflow` | Multi-stage training workflow orchestration. |
| `health` | Preflight health checks for workbench workflows. |
| `golden-eval` | Per-container golden-eval / hello-world reruns. |

## Examples

```bash
npa workbench --help
npa workbench lerobot --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workbench`.
