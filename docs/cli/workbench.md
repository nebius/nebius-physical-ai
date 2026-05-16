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
fiftyone  Voxel51 FiftyOne dataset curation and visualization workbench.
genesis  Genesis simulation: teacher training, demo generation, evaluation.
groot  NVIDIA Isaac GR00T humanoid foundation-model workbench.
isaac-lab  Isaac Lab simulation workbench deployment, training, and evaluation.
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
| `fiftyone` | Voxel51 FiftyOne dataset curation and visualization workbench. |
| `genesis` | Genesis simulation: teacher training, demo generation, evaluation. |
| `groot` | NVIDIA Isaac GR00T humanoid foundation-model workbench. |
| `isaac-lab` | Isaac Lab simulation workbench deployment, training, and evaluation. |

## Examples

```bash
npa workbench --help
npa workbench lerobot --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workbench`.
