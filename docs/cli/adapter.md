# `npa adapter`

## Command Tree

```text
Usage: npa adapter [OPTIONS] COMMAND [ARGS]...

Convert simulation data to training dataset formats.

Options
--help  Show this message and exit.
Commands
convert  Convert Genesis/sim demo numpy arrays to LeRobotDataset v3 format.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `convert` | Convert Genesis/sim demo numpy arrays to LeRobotDataset v3 format. |

## Examples

```bash
npa adapter --help
npa adapter convert --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `adapter`.
