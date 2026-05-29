# `npa workbench data`

## Command Tree

```text
Usage: npa workbench data [OPTIONS] COMMAND [ARGS]...

S3 data import bridge for Workbench pipelines.

Options
--project  -p  TEXT  Default project alias for S3 credentials.
--help  Show this message and exit.
Commands
sync  Copy S3 objects between pipeline prefixes.
status  Show object count and bytes for an S3 prefix.
list  List S3 objects under a Workbench data prefix.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  TEXT  Default project alias for S3 credentials. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `sync` | Copy S3 objects between pipeline prefixes. |
| `status` | Show object count and bytes for an S3 prefix. |
| `list` | List S3 objects under a Workbench data prefix. |

## Examples

```bash
npa workbench data --help
npa workbench data sync --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `data`.
