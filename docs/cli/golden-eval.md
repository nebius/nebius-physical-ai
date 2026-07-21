# `npa workbench golden-eval`

## Command Tree

```text
Usage: npa workbench golden-eval [OPTIONS] COMMAND [ARGS]...

Per-container golden-eval / hello-world reruns.

Options
--help  Show this message and exit.
Commands
list  List every container and its golden eval.
show  Show the full safety + Physical AI + golden-eval record for a container.
validate  Validate manifest completeness and consistency (offline; nightly CI gate).
run  Print, execute locally, or run on serverless a container's golden eval.
run-all  Run golden evals for every container (optionally in parallel).
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list` | List every container and its golden eval. |
| `show` | Show the full safety + Physical AI + golden-eval record for a container. |
| `validate` | Validate manifest completeness and consistency (offline; nightly CI gate). |
| `run` | Print, execute locally, or run on serverless a container's golden eval. |
| `run-all` | Run golden evals for every container (optionally in parallel). |

## Examples

```bash
npa workbench golden-eval --help
npa workbench golden-eval list --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `golden-eval`.
