# `npa workbench genesis`

## Command Tree

```text
Usage: npa workbench genesis [OPTIONS] COMMAND [ARGS]...

Genesis simulation: teacher training, demo generation, evaluation.

Options
--project  -p  <str>  Project alias from ~/.npa/config.yaml. When set, the command runs on the workbench VM via
SSH instead of locally.
--name  -n  <str>  Workbench instance name within the project.
--help  Show this message and exit.
Commands
train-teacher  Train an RL teacher policy with PPO using privileged state in Genesis.
generate-demos  Generate camera-only demonstrations using a trained teacher policy.
simulate  Generate camera-only demonstrations using a trained teacher policy.
eval-teacher  Evaluate the teacher under held-out conditions (no cameras, privileged state).
eval-student  Evaluate a student vision policy in Genesis simulation.
diagnose  Diagnose teacher policy failures: run rollouts, classify failure phases, suggest fixes.
tune  Auto-tune loop: diagnose -> adjust config -> retrain -> re-diagnose.
list  List configured Genesis workbenches (excludes LeRobot VMs).
deploy  Deploy or destroy a Genesis simulation VM.
status  Check Genesis VM status via SSH (processes, GPU, conda env).
system-info  Collect and display system hardware information from the Genesis VM.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  <str>  Project alias from ~/.npa/config.yaml. When set, the command runs on the workbench VM via |
| `--name` | -n  <str>  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `train-teacher` | Train an RL teacher policy with PPO using privileged state in Genesis. |
| `generate-demos` | Generate camera-only demonstrations using a trained teacher policy. |
| `simulate` | Generate camera-only demonstrations using a trained teacher policy. |
| `eval-teacher` | Evaluate the teacher under held-out conditions (no cameras, privileged state). |
| `eval-student` | Evaluate a student vision policy in Genesis simulation. |
| `diagnose` | Diagnose teacher policy failures: run rollouts, classify failure phases, suggest fixes. |
| `tune` | Auto-tune loop: diagnose -> adjust config -> retrain -> re-diagnose. |
| `list` | List configured Genesis workbenches (excludes LeRobot VMs). |
| `deploy` | Deploy or destroy a Genesis simulation VM. |
| `status` | Check Genesis VM status via SSH (processes, GPU, conda env). |
| `system-info` | Collect and display system hardware information from the Genesis VM. |

## Examples

```bash
npa workbench genesis --help
npa workbench genesis train-teacher --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `genesis`.
