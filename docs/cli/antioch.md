# `npa workbench antioch`

## Command Tree

```text
Usage: npa workbench antioch [OPTIONS] COMMAND [ARGS]...

Run Antioch simulations and collect policy-compatible data.

Options
--help  Show this message and exit.
Commands
health
package-project  Build a deterministic, credential-free immutable project package.
system-info
submit
run
status
reconcile
cancel
resume
collect
list
deploy
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `package-project` | Build a deterministic, credential-free immutable project package. system-info submit run status reconcile cancel resume collect list deploy |

## Examples

```bash
npa workbench antioch --help
npa workbench antioch package-project --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `antioch`.
