# `npa workbench ros2`

## Command Tree

```text
Usage: npa workbench ros2 [OPTIONS] COMMAND [ARGS]...

ROS 2 Jazzy prerequisite detection and deployment planning (bridge/bag-conversion/fleet execution not implemented).

Options
--help  Show this message and exit.
Commands
preflight  Check ROS 2 Jazzy prerequisites; fail fast with remediation if unusable.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `preflight` | Check ROS 2 Jazzy prerequisites; fail fast with remediation if unusable. |

## Examples

```bash
npa workbench ros2 --help
npa workbench ros2 preflight --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `ros2`.
