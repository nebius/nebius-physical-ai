# `npa workbench eval-harness`

## Command Tree

```text
Usage: npa workbench eval-harness [OPTIONS] COMMAND [ARGS]...

Standardized policy evaluation: single-policy runs and paired A/B comparisons.

Options
--help  Show this message and exit.
Commands
run  Evaluate one policy on a task and write JSON + Markdown reports.
compare  A/B compare two policies with paired episode seeds.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Evaluate one policy on a task and write JSON + Markdown reports. |
| `compare` | A/B compare two policies with paired episode seeds. |

## Examples

```bash
npa workbench eval-harness --help
npa workbench eval-harness run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `eval-harness`.
