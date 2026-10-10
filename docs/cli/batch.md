# `npa workbench workflow batch`

## Command Tree

```text
Usage: npa workbench workflow batch [OPTIONS] COMMAND [ARGS]...

Submit dataset episodes as independent durable workflow runs.

Options
--help  Show this message and exit.
Commands
plan  Expand and validate a batch without submitting jobs.
submit  Run a batch in the foreground through standard workflow submission.
status  Read local observations; use workflow status for live reconciliation.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `plan` | Expand and validate a batch without submitting jobs. |
| `submit` | Run a batch in the foreground through standard workflow submission. |
| `status` | Read local observations; use workflow status for live reconciliation. |

## Examples

```bash
npa workbench workflow batch --help
npa workbench workflow batch plan --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `batch`.
