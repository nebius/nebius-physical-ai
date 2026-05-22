# `npa workbench detection-training`

## Command Tree

```text
Usage: npa workbench detection-training [OPTIONS] COMMAND [ARGS]...

Train Faster R-CNN detectors from LanceDB materialized views.

Options
--help  Show this message and exit.
Commands
deploy  Deploy the detection-training service to an NPA Workbench Kubernetes cluster.
train  Start a detection-training run.
eval  Evaluate a detection-training checkpoint.
status  Fetch training run status.
system-info  Show detection-training runtime information.
list  List service-managed runs or Kubernetes resources.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Deploy the detection-training service to an NPA Workbench Kubernetes cluster. |
| `train` | Start a detection-training run. |
| `eval` | Evaluate a detection-training checkpoint. |
| `status` | Fetch training run status. |
| `system-info` | Show detection-training runtime information. |
| `list` | List service-managed runs or Kubernetes resources. |

## Examples

```bash
npa workbench detection-training --help
npa workbench detection-training deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `detection-training`.
