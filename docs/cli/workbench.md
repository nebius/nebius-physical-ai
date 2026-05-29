# `npa workbench`

## Command Tree

```text
Usage: npa workbench [OPTIONS] COMMAND [ARGS]...

Physical AI workbench tools.

Options
--help  Show this message and exit.
Commands
lerobot  LeRobot policy training, evaluation, serving, and inference.
data  S3 data import bridge for Workbench pipelines.
cosmos  NVIDIA Cosmos world model serving and inference endpoints.
fiftyone  Voxel51 FiftyOne dataset curation and visualization workbench.
genesis  Genesis simulation: teacher training, demo generation, evaluation.
groot  NVIDIA Isaac GR00T humanoid foundation-model workbench.
isaac-lab  Isaac Lab simulation workbench deployment, training, and evaluation.
sonic  NVIDIA GEAR-SONIC whole-body-control workbench.
mjlab  MJLab locomotion policy evaluation for SONIC workflows.
retargeting  Motion retargeting for SONIC locomotion workflows.
lancedb  Deploy and query LanceDB vector-search workbenches.
detection-training  Train Faster R-CNN detectors from LanceDB materialized views.
vlm-eval  VLM evaluation for sim-to-real pipeline gating.
workflow  Multi-stage training workflow orchestration.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `lerobot` | LeRobot policy training, evaluation, serving, and inference. |
| `data` | S3 data import bridge for Workbench pipelines. |
| `cosmos` | NVIDIA Cosmos world model serving and inference endpoints. |
| `fiftyone` | Voxel51 FiftyOne dataset curation and visualization workbench. |
| `genesis` | Genesis simulation: teacher training, demo generation, evaluation. |
| `groot` | NVIDIA Isaac GR00T humanoid foundation-model workbench. |
| `isaac-lab` | Isaac Lab simulation workbench deployment, training, and evaluation. |
| `sonic` | NVIDIA GEAR-SONIC whole-body-control workbench. |
| `mjlab` | MJLab locomotion policy evaluation for SONIC workflows. |
| `retargeting` | Motion retargeting for SONIC locomotion workflows. |
| `lancedb` | Deploy and query LanceDB vector-search workbenches. |
| `detection-training` | Train Faster R-CNN detectors from LanceDB materialized views. |
| `vlm-eval` | VLM evaluation for sim-to-real pipeline gating. |
| `workflow` | Multi-stage training workflow orchestration. |

## Examples

```bash
npa workbench --help
npa workbench lerobot --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workbench`.
