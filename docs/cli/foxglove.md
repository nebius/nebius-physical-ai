# `npa workbench foxglove`

## Command Tree

```text
Usage: npa workbench foxglove [OPTIONS] COMMAND [ARGS]...

Foxglove embedded viewer: MCAP conversion, inspection, and SDK assets.

Options
--help  Show this message and exit.
Commands
convert-run  Convert a run's artifacts into an MCAP recording the Foxglove viewer can open.
inspect  Report the channels, schemas, and message counts inside an MCAP recording.
install-sdk  Install the pinned, sha512-verified Foxglove embed SDK assets.
config  Show the resolved Foxglove embed settings for this environment.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `convert-run` | Convert a run's artifacts into an MCAP recording the Foxglove viewer can open. |
| `inspect` | Report the channels, schemas, and message counts inside an MCAP recording. |
| `install-sdk` | Install the pinned, sha512-verified Foxglove embed SDK assets. |
| `config` | Show the resolved Foxglove embed settings for this environment. |

## Examples

```bash
npa workbench foxglove --help
npa workbench foxglove convert-run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `foxglove`.
