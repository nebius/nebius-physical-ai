# `npa workbench mujoco`

## Command Tree

```text
Usage: npa workbench mujoco [OPTIONS] COMMAND [ARGS]...

MuJoCo contact-rich manipulation: scripted-policy rollouts with real contact solving.

Options
--help  Show this message and exit.
Commands
run  Run N episodes of a scripted policy; write trajectories + metrics.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Run N episodes of a scripted policy; write trajectories + metrics. |

## Examples

```bash
npa workbench mujoco --help
npa workbench mujoco run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `mujoco`.
