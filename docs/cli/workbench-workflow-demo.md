# `npa workbench workflow demo`

## Command Tree

```text
Usage: npa workbench workflow demo [OPTIONS] COMMAND [ARGS]...

Run complete public sample workflows and view their results.

Options
--help  Show this message and exit.
Commands
list  Show the four public sample demos; no credentials or GPUs are needed.
run  Fetch public samples and run a complete demo through the standard workflow engine.
view  Open a compact offline HTML report containing the actual run's evidence.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list` | Show the four public sample demos; no credentials or GPUs are needed. |
| `run` | Fetch public samples and run a complete demo through the standard workflow engine. |
| `view` | Open a compact offline HTML report containing the actual run's evidence. |

## Examples

```bash
npa workbench workflow demo --help
npa workbench workflow demo list --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workbench-workflow-demo`.
