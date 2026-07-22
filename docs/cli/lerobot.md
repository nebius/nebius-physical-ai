# `npa workbench lerobot`

## Command Tree

```text
Usage: npa workbench lerobot [OPTIONS] COMMAND [ARGS]...

LeRobot policy training, evaluation, serving, and inference.

Options
--project  -p  <str>  Project alias (as configured in ~/.npa/config.yaml).
--name  -n  <str>  Workbench instance name within the project.
--help  Show this message and exit.
Commands
list  List configured LeRobot workbenches (excludes Genesis VMs).
status  Check what's running on the VM.
train  Run lerobot-train on the VM via SSH, stream logs.
eval  Run lerobot-eval on the VM, return metrics.
serve  Start or restart the PolicyServer with a given checkpoint.
infer  POST an observation to the running PolicyServer, return predicted actions.
list-checkpoints  List available checkpoints on the VM and in object storage.
deploy  Deploy or update LeRobot infrastructure and application.
system-info  Collect and display system hardware information from the VM.
benchmark  Run a benchmark suite: collect system info, train each model at each num_workers value, upload
results to S3.
profile-train  Profile training. Modes: wallclock (throughput), profiler (torch.profiler), or inference.
train-student  Train a vision-only student policy via LeRobot imitation learning.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  <str>  Project alias (as configured in ~/.npa/config.yaml). |
| `--name` | -n  <str>  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list` | List configured LeRobot workbenches (excludes Genesis VMs). |
| `status` | Check what's running on the VM. |
| `train` | Run lerobot-train on the VM via SSH, stream logs. |
| `eval` | Run lerobot-eval on the VM, return metrics. |
| `serve` | Start or restart the PolicyServer with a given checkpoint. |
| `infer` | POST an observation to the running PolicyServer, return predicted actions. |
| `list-checkpoints` | List available checkpoints on the VM and in object storage. |
| `deploy` | Deploy or update LeRobot infrastructure and application. |
| `system-info` | Collect and display system hardware information from the VM. |
| `benchmark` | Run a benchmark suite: collect system info, train each model at each num_workers value, upload |
| `profile-train` | Profile training. Modes: wallclock (throughput), profiler (torch.profiler), or inference. |
| `train-student` | Train a vision-only student policy via LeRobot imitation learning. |

## Examples

```bash
npa workbench lerobot --help
npa workbench lerobot list --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `lerobot`.
