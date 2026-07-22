# `npa workbench health`

## Command Tree

```text
Usage: npa workbench health [OPTIONS] COMMAND [ARGS]...

Preflight health checks for workbench workflows.

Options
--help  Show this message and exit.
Commands
preflight  Validate HF, NGC, S3, and Token Factory credentials before a deploy or GPU job.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `preflight` | Validate HF, NGC, S3, and Token Factory credentials before a deploy or GPU job. |

## Examples

```bash
npa workbench health --help
npa workbench health preflight --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `health`.
