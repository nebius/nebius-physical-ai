# `npa workbench access`

## Command Tree

```text
Usage: npa workbench access [OPTIONS] COMMAND [ARGS]...

Configure Workbench accounts, allocations, connections, and service access.

Options
--help  Show this message and exit.
Commands
export-connections  Prepare private connections to existing Kubernetes clusters on any cloud.
check-clusters  Verify existing cluster access and boundaries from the server environment.
setup  Deploy the shared HTTPS control plane and retain its Nebius LB address.
stop-run  Cancel an exact run as the local server operator, including after offboarding.
render  Render cluster boundaries and private SkyPilot configuration for review.
enroll  Apply and verify one administrator-selected personal cluster allocation.
export-kubeconfig  Export selected cluster credentials to a new private server-only file.
serve  Run one private Workbench supervisor behind the administrator's HTTPS ingress.
storage-create  Create a personal Nebius bucket and narrowly scoped storage principal.
render-service  Render a CPU-only gateway and private SkyPilot sidecar into a new YAML file.
account  Operator-managed users, personal keys, and optional SSO links.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `export-connections` | Prepare private connections to existing Kubernetes clusters on any cloud. |
| `check-clusters` | Verify existing cluster access and boundaries from the server environment. |
| `setup` | Deploy the shared HTTPS control plane and retain its Nebius LB address. |
| `stop-run` | Cancel an exact run as the local server operator, including after offboarding. |
| `render` | Render cluster boundaries and private SkyPilot configuration for review. |
| `enroll` | Apply and verify one administrator-selected personal cluster allocation. |
| `export-kubeconfig` | Export selected cluster credentials to a new private server-only file. |
| `serve` | Run one private Workbench supervisor behind the administrator's HTTPS ingress. |
| `storage-create` | Create a personal Nebius bucket and narrowly scoped storage principal. |
| `render-service` | Render a CPU-only gateway and private SkyPilot sidecar into a new YAML file. |
| `account` | Operator-managed users, personal keys, and optional SSO links. |

## Examples

```bash
npa workbench access --help
npa workbench access export-connections --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `access`.
