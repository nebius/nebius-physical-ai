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
cosmos3  Cosmos3 reasoning workflow contracts.
fiftyone  Voxel51 FiftyOne dataset curation and visualization workbench.
genesis  Genesis simulation: teacher training, demo generation, evaluation.
groot  NVIDIA Isaac GR00T humanoid foundation-model workbench.
isaac-lab  Isaac Lab simulation workbench deployment, training, and evaluation.
sonic  NVIDIA GEAR-SONIC whole-body-control workbench.
mjlab  MJLab locomotion policy evaluation for SONIC workflows.
lichtblick  Lichtblick (MPL-2.0) - an open-source, Foxglove-compatible MCAP / ROS-bag log viewer.
lancedb  Deploy and query LanceDB vector-search workbenches.
detection-training  Train Faster R-CNN detectors from LanceDB materialized views.
scenario-gen  Adversarial scenario generation: mine hard scenarios that fail a policy-under-test (pluggable
Isaac Lab RL backend; deterministic default).
dataset  Dataset-of-record: ingest, validate, curate, and query production sensor data.
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
| `cosmos3` | Cosmos3 reasoning workflow contracts. |
| `fiftyone` | Voxel51 FiftyOne dataset curation and visualization workbench. |
| `genesis` | Genesis simulation: teacher training, demo generation, evaluation. |
| `groot` | NVIDIA Isaac GR00T humanoid foundation-model workbench. |
| `isaac-lab` | Isaac Lab simulation workbench deployment, training, and evaluation. |
| `sonic` | NVIDIA GEAR-SONIC whole-body-control workbench. |
| `mjlab` | MJLab locomotion policy evaluation for SONIC workflows. |
| `lichtblick` | Lichtblick (MPL-2.0) - an open-source, Foxglove-compatible MCAP / ROS-bag log viewer. |
| `lancedb` | Deploy and query LanceDB vector-search workbenches. |
| `detection-training` | Train Faster R-CNN detectors from LanceDB materialized views. |
| `scenario-gen` | Adversarial scenario generation: mine hard scenarios that fail a policy-under-test (pluggable |
| `dataset` | Dataset-of-record: ingest, validate, curate, and query production sensor data. |
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
