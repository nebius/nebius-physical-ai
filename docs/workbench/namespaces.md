# Team namespaces

Use `npa workbench namespace` to give a research team its own Kubernetes
resource names, Secrets, services, and access bindings on an existing cluster.
The administrator applies access; each researcher prepares a private context
using their own authenticated kubeconfig.

## Administrator setup

Choose a dedicated namespace and use the exact administrator context. Preview
the manifests without contacting the cluster:

```bash
npa workbench namespace apply "research" --context "$KUBE_CONTEXT" \
  --user "$RESEARCHER_USERNAME" --group "$RESEARCHER_GROUP" --dry-run
npa workbench namespace apply "research" --context "$KUBE_CONTEXT" \
  --user "$RESEARCHER_USERNAME" --group "$RESEARCHER_GROUP"
```

`--user` and `--group` are repeatable **complete membership lists**. Reapply
with the desired list to remove previous grants; omit both to remove all
researcher grants managed by this command. Other administrators' bindings
remain authoritative. These are Kubernetes authentication usernames and groups,
not display names. The command does not create identity-provider accounts.

The command creates a namespace, an `npa-workbench` service account, and two
namespace-scoped bindings to Kubernetes' built-in `edit` role. Researchers and
the service account also receive read-only node, runtime-class, storage-class,
and exact namespace discovery. No access to another namespace's Pods or Secrets
is granted. Existing resources must carry this command's ownership labels;
foreign resources and server-side apply conflicts fail without forced adoption.
An interrupted apply may have created some objects; reapply the same desired
membership to converge them.

Members of a namespace share trust: `edit` permits reading its Secrets and
running Pods as its service accounts. Keep administrator credentials and
privileged service accounts outside team namespaces. Kubernetes RBAC grants
are additive, so existing cluster-wide roles can still give a principal wider
access. Namespaces do not themselves isolate network traffic, host access, S3,
or the SkyPilot API. Configure network/admission policies and independent
object-storage permissions to match your environment. No quotas or job limits
are added.

## Researcher setup and submission

Authenticate with your **own** Kubernetes identity, then prepare a new private
directory. The source kubeconfig and its current context are unchanged:

```bash
npa workbench namespace context "research" --context "$KUBE_CONTEXT" \
  --output-dir "$HOME/.npa/research-client"
export KUBECONFIG="$HOME/.npa/research-client/kubeconfig"
export SKYPILOT_GLOBAL_CONFIG="$HOME/.npa/research-client/sky.yaml"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/research-client/runtime"
kubectl --context "$KUBE_CONTEXT" get pods
npa workbench workflow submit "$WORKFLOW_PATH" \
  --project "$NPA_PROJECT_ALIAS" --infra "k8s/$KUBE_CONTEXT"
```

The context command prints paths as JSON, never credentials. Its directory is
mode `0700` and files are `0600`; existing destinations are refused. The exported
kubeconfig retains the source identity's permissions. Do not distribute an
administrator's exported context as researcher credentials.

SkyPilot uses the namespace in this context as its provider namespace. Its
generated config selects the precreated `npa-workbench` service account through
`kubernetes.remote_identity`, avoiding automatic broad workload RBAC creation.
The separate runtime directory keeps the local API and job state scoped to this
client; use the same environment for status, logs and cancellation.
Controller teardown remains an administrator operation with provider ownership
verification and cluster-wide controller inventory.
Isolated SkyPilot execution requires a Linux operator host. An independently
operated remote API must be configured with the corresponding namespace and
identity on the server; changing a client context alone cannot change that
server's backend credentials. Do not combine this local runtime with another
API endpoint.

Registry pull-secret checks and automatic model-cache PVC discovery follow the
selected context namespace. Controller health probes and job diagnostics use
that namespace too. Put referenced Secrets and PVCs there. Explicit
model-cache namespace overrides remain available. Kubernetes service deploy
commands still take their own `--namespace research` flag; use the service's
namespace-qualified DNS endpoint in a workflow.

GPU capacity checks that require a cluster-wide Pod inventory need an
operator with that additional read permission. This namespace role deliberately
does not grant it. SkyPilot object-store FUSE mounts and ingress also require
separate administrator setup; ordinary application S3 reads/writes can use
namespace-scoped credentials. See the pinned
[SkyPilot Kubernetes permission contract](https://github.com/skypilot-org/skypilot/blob/v0.12.2/docs/source/cloud-setup/cloud-permissions/kubernetes.rst).

## Python and validation

```python
from pathlib import Path
from npa.sdk.workbench.namespace import apply_namespace, write_namespace_context

plan = apply_namespace("research", context="cluster-context", users=("researcher",), dry_run=True)
# Each caller exports their own credentials after administrator setup.
settings = write_namespace_context("research", context="cluster-context", output_dir=Path("research-client"))
```

The opt-in live suite uses an explicitly selected disposable cluster, creates
two unique namespaces, and verifies actual API denials and CPU pod placement:

```bash
NPA_INTEGRATION_E2E=1 NPA_NAMESPACE_LIVE_E2E=1 \
  NPA_NAMESPACE_LIVE_CONTEXT="$KUBE_CONTEXT" \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_namespaces_live.py -q
```

It cleans up only its generated namespaces and discovery bindings. Application
cleanup remains explicit: cancel managed jobs, verify they have stopped, remove
services/PVCs after preserving data, and only then let the administrator delete
the team's namespace and its labeled discovery ClusterRole/ClusterRoleBinding.
Namespace membership updates never delete workloads or storage.
