# `npa cluster`

## Command Tree

```text
Usage: npa cluster [OPTIONS] COMMAND [ARGS]...

Manage NPA Workbench cluster targets and profiles.

Options
--help  Show this message and exit.
Commands
deploy  Bootstrap an NPA Workbench cluster target with local state and a cached kubeconfig.
destroy  Delete a Managed Kubernetes cluster through the API and drop its local state.
up  Create or update the Terraform-managed NPA Kubernetes cluster.
down  Destroy the Terraform-managed NPA cluster: cloud resources and local state.
status  Show NPA cluster target state from Nebius and the local cache.
list  List NPA Workbench cluster targets known locally or in the configured project.
kubeconfig  Write a kubeconfig for a Managed Kubernetes cluster that already exists.
node-group  Manage GPU node groups attached to NPA Workbench cluster targets.

`npa cluster` manages NPA Workbench cluster targets and profiles. For raw MK8s administration (edit, update, upgrade, operation inspection, version listing, compatibility matrix), use `nebius mk8s`
directly.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Bootstrap an NPA Workbench cluster target with local state and a cached kubeconfig. |
| `destroy` | Delete a Managed Kubernetes cluster through the API and drop its local state. |
| `up` | Create or update the Terraform-managed NPA Kubernetes cluster. |
| `down` | Destroy the Terraform-managed NPA cluster: cloud resources and local state. |
| `status` | Show NPA cluster target state from Nebius and the local cache. |
| `list` | List NPA Workbench cluster targets known locally or in the configured project. |
| `kubeconfig` | Write a kubeconfig for a Managed Kubernetes cluster that already exists. |
| `node-group` | Manage GPU node groups attached to NPA Workbench cluster targets. `npa cluster` manages NPA Workbench cluster targets and profiles. For raw MK8s administration (edit, update, upgrade, operation inspection, version listing, compatibility matrix), use `nebius mk8s` directly. |

## Examples

```bash
npa cluster --help
npa cluster deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cluster`.
