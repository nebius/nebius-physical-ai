# `npa workbench sonic retargeting`

## Command Tree

```text
Usage: npa workbench sonic retargeting [OPTIONS] COMMAND [ARGS]...

Motion retargeting for SONIC locomotion workflows.

Options
--help  Show this message and exit.
Commands
run  Retarget source motion artifacts into the SONIC embodiment schema.
workflow  Show the SkyPilot YAML template for retargeting.
status  Show retargeting tool status.
list  List supported retargeting source formats.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Retarget source motion artifacts into the SONIC embodiment schema. |
| `workflow` | Show the SkyPilot YAML template for retargeting. |
| `status` | Show retargeting tool status. |
| `list` | List supported retargeting source formats. |

## Examples

```bash
npa workbench sonic retargeting --help
npa workbench sonic retargeting run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `retargeting`.
