# `npa workbench mjlab`

## Command Tree

```text
Usage: npa workbench mjlab [OPTIONS] COMMAND [ARGS]...

MJLab GPU robot learning, evaluation and ONNX export.

Options
--help  Show this message and exit.
Commands
train  Train or resume a native MJLab policy on the current GPU node.
eval  Measure complete episodes from a native MJLab/RSL-RL checkpoint.
export  Export a policy to checked ONNX with robot metadata.
status  Report installation or remote status.
system-info  Inspect dependency versions.
list  List tasks from the installed MJLab registry.
workflow  Locate the MJLab workflow templates.
deploy  Deploy a private service on an existing GPU cluster using existing Secrets.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `train` | Train or resume a native MJLab policy on the current GPU node. |
| `eval` | Measure complete episodes from a native MJLab/RSL-RL checkpoint. |
| `export` | Export a policy to checked ONNX with robot metadata. |
| `status` | Report installation or remote status. |
| `system-info` | Inspect dependency versions. |
| `list` | List tasks from the installed MJLab registry. |
| `workflow` | Locate the MJLab workflow templates. |
| `deploy` | Deploy a private service on an existing GPU cluster using existing Secrets. |

## Examples

```bash
npa workbench mjlab --help
npa workbench mjlab train --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `mjlab`.
