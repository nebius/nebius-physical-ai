# `npa workbench team`

## Command Tree

```text
Usage: npa workbench team [OPTIONS] COMMAND [ARGS]...

Optional team access and authenticated workflow execution.

Options
--help  Show this message and exit.
Commands
setup  Deploy the shared HTTPS control plane and retain its Nebius LB address.
stop-run  Cancel an exact run as the local server operator, including after offboarding.
render  Render cluster boundaries and private SkyPilot configuration for review.
enroll  Apply and verify one administrator-selected personal cluster allocation.
export-kubeconfig  Export selected cluster credentials to a new private server-only file.
serve  Run one private team supervisor behind the administrator's HTTPS ingress.
whoami  Show verified identity and workspace access without a personal cloud account.
submit  Submit an NPA workflow through the authenticated team execution boundary.
run  Inspect, cancel, or resume an owned run using the authenticated team API.
list  List the authenticated person's runs in one workspace.
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
| `setup` | Deploy the shared HTTPS control plane and retain its Nebius LB address. |
| `stop-run` | Cancel an exact run as the local server operator, including after offboarding. |
| `render` | Render cluster boundaries and private SkyPilot configuration for review. |
| `enroll` | Apply and verify one administrator-selected personal cluster allocation. |
| `export-kubeconfig` | Export selected cluster credentials to a new private server-only file. |
| `serve` | Run one private team supervisor behind the administrator's HTTPS ingress. |
| `whoami` | Show verified identity and workspace access without a personal cloud account. |
| `submit` | Submit an NPA workflow through the authenticated team execution boundary. |
| `run` | Inspect, cancel, or resume an owned run using the authenticated team API. |
| `list` | List the authenticated person's runs in one workspace. |
| `storage-create` | Create a personal Nebius bucket and narrowly scoped storage principal. |
| `render-service` | Render a CPU-only gateway and private SkyPilot sidecar into a new YAML file. |
| `account` | Operator-managed users, personal keys, and optional SSO links. |

## Examples

```bash
npa workbench team --help
npa workbench team setup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `team`.
