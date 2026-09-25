# Kubernetes namespaces

Workbench uses the namespace in the selected kubeconfig context for Kubernetes
workflows. **If that context has no namespace, the default is `default`.**
Selecting a namespace is native behavior; no feature flag or new identity system
is required. Existing Kubernetes credentials, permissions, and SkyPilot workload
identity settings continue to apply.

## Create or reuse a namespace

```bash
npa workbench namespace apply "research" --context "$KUBE_CONTEXT" --dry-run
npa workbench namespace apply "research" --context "$KUBE_CONTEXT"
```

`apply` creates only a Namespace. If it already exists, the command returns
`status: existing` without changing its labels, service accounts, roles, or
bindings. Namespace creation requires your existing identity to have that
permission. Selecting an existing namespace does not require creating one.

## Select a namespace

Prepare a private copy of your authenticated context:

```bash
npa workbench namespace context "research" --context "$KUBE_CONTEXT" \
  --output-dir "$HOME/.npa/research-client"
export KUBECONFIG="$HOME/.npa/research-client/kubeconfig"
export SKYPILOT_GLOBAL_CONFIG="$HOME/.npa/research-client/sky.yaml"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/research-client/runtime"
npa workbench workflow submit "$WORKFLOW_PATH" \
  --project "$NPA_PROJECT_ALIAS" --infra "k8s/$KUBE_CONTEXT"
```

The source kubeconfig remains unchanged. The output retains its context name,
cluster, and credentials, and sets the requested namespace. Configuration paths
are printed as JSON; credentials are never printed. The new directory is `0700`
and files are `0600`. Existing destinations are refused.

Existing SkyPilot settings are copied from `--sky-config`, otherwise
`SKYPILOT_GLOBAL_CONFIG`, otherwise `~/.sky/config.yaml` if present. Only the
allowed cloud/context are narrowed to Kubernetes and the selected context.
Settings such as `kubernetes.remote_identity` are preserved. This command does
not choose a new service account or grant permissions. With no existing config,
SkyPilot's normal workload identity behavior applies.

Use the same environment for submit, status, logs, and cancellation. The separate
runtime directory keeps the local API and job state tied to this context.
Isolated SkyPilot execution requires a Linux operator host. A separately operated
remote API must have the corresponding namespace and credentials configured on
its server; a client context cannot replace that server's credentials.

## Defaults and resource placement

| Selection | Workflow namespace |
| --- | --- |
| Context has no namespace | `default` |
| Context sets a namespace | That namespace |
| `namespace context "default"` | Explicitly `default` |

SkyPilot derives pod placement from the context namespace. Setting
`pod_config.metadata.namespace` alone is insufficient: SkyPilot replaces it with
the provider namespace. Registry pull-secret checks, model-cache PVC discovery,
controller health probes, and job diagnostics follow that same context.
An unreadable context never silently selects another namespace's Secret or PVC.
Explicit model-cache namespace overrides remain available.

Kubernetes service deployment commands retain their existing `--namespace` flags
and historical defaults (`workbench` or `default`, depending on the service).
Pass the selected namespace there as well and use the namespace-qualified service
DNS name when connecting from a workflow.

Namespace selection does not change RBAC, network policy, object-storage access,
GPU capacity permissions, or controller administration. Existing permission
requirements for those operations still apply. See the pinned
[SkyPilot Kubernetes permission contract](https://github.com/skypilot-org/skypilot/blob/v0.12.2/docs/source/cloud-setup/cloud-permissions/kubernetes.rst).

## Python and live validation

```python
from pathlib import Path
from npa.sdk.workbench.namespace import apply_namespace, write_namespace_context

apply_namespace("research", context="cluster-context")
settings = write_namespace_context("research", context="cluster-context", output_dir=Path("research-client"))
```

The opt-in live suite creates two unique namespaces on an explicitly selected
disposable cluster. It verifies independent same-name resources, unchanged
existing access, actual CPU pod placement, and the `default` fallback:

```bash
NPA_INTEGRATION_E2E=1 NPA_NAMESPACE_LIVE_E2E=1 \
  NPA_NAMESPACE_LIVE_CONTEXT="$KUBE_CONTEXT" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_namespaces_live.py -q
```

The suite deletes only its own namespaces and temporary default-namespace
resource. For application cleanup, cancel managed jobs and verify they have
stopped before removing their controller or namespace. Preserve any required
artifacts and PVC data first; namespace selection never deletes storage.
