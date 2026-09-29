# `npa workbench antioch`

## Command Tree

```text
Usage: npa workbench antioch [OPTIONS] COMMAND [ARGS]...

Run Antioch simulations and collect policy-compatible data.

Options
--help  Show this message and exit.
Commands
health
terms-preflight  Verify explicit, scoped Antioch terms acceptance before runtime use.
package-project  Build a deterministic, credential-free immutable project package.
system-info
live-start  Start a continuing streamed OpenPI scenario under tmux supervision.
live-status  Inspect exact local tmux supervisor state without reading auth storage.
live-stop  Cancel the exact scenario, then stop its exact sim service.
live-k8s-deploy  Reconcile the same-pod Antioch tunnel and cluster-local policy path.
live-k8s-status  Return sanitized adapter and retained-policy readiness.
live-k8s-stop  Stop the exact scenario before its supported service tunnel.
live-k8s-finalize-cutover  Disable the exact owned public rollback Service after acceptance.
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
| `health` | - |
| `terms-preflight` | Verify explicit, scoped Antioch terms acceptance before runtime use. |
| `package-project` | Build a deterministic, credential-free immutable project package. |
| `system-info` | - |
| `live-start` | Start a continuing streamed OpenPI scenario under tmux supervision. |
| `live-status` | Inspect exact local tmux supervisor state without reading auth storage. |
| `live-stop` | Cancel the exact scenario, then stop its exact sim service. |
| `live-k8s-deploy` | Reconcile the same-pod Antioch tunnel and cluster-local policy path. |
| `live-k8s-status` | Return sanitized adapter and retained-policy readiness. |
| `live-k8s-stop` | Stop the exact scenario before its supported service tunnel. |
| `live-k8s-finalize-cutover` | Disable the exact owned public rollback Service after acceptance. |
| `submit` | - |
| `run` | - |
| `status` | - |
| `reconcile` | - |
| `cancel` | - |
| `resume` | - |
| `collect` | - |
| `list` | - |
| `deploy` | - |

## Examples

```bash
npa workbench antioch --help
npa workbench antioch health --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `antioch`.
