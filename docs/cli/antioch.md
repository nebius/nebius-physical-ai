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
openpi-stack  Render or apply the private RTX/Isaac-to-B200/OpenPI stack.
openpi-contract-smoke  Validate the camera/state/action wire contract without a GPU or credentials.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `package-project` | Build a deterministic, credential-free immutable project package. system-info submit run status reconcile cancel resume collect list deploy |
| `openpi-stack` | Render or apply the private RTX/Isaac-to-B200/OpenPI stack. |
| `openpi-contract-smoke` | Validate the camera/state/action wire contract without a GPU or credentials. |

## Examples

```bash
npa workbench antioch --help
npa workbench antioch package-project --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `antioch`.
