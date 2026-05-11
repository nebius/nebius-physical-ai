# `npa convert`

## Command Tree

```text
Usage: npa convert [OPTIONS] COMMAND [ARGS]...

Convert datasets and prediction artifacts between standalone formats.

Options
--help  Show this message and exit.
Commands
lerobot-to-rrd  Convert a LeRobotDataset to a Rerun .rrd recording.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `lerobot-to-rrd` | Convert a LeRobotDataset to a Rerun .rrd recording. |

## Examples

```bash
npa convert --help
npa convert lerobot-to-rrd --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `convert`.
