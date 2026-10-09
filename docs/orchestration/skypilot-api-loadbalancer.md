# Persistent team access to the SkyPilot API

[SkyPilot setup](skypilot-setup.md) · [Docs](../README.md)

Run the team API on CPU nodes in the existing Workbench Kubernetes cluster,
in an operator-owned namespace. Use the upstream SkyPilot Helm chart, persistent
storage, and a maintained HTTPS ingress or Gateway backed by a Nebius
LoadBalancer. A DNS name provides the endpoint developers use across VDI and
API pod restarts. The API does not need a GPU.

This recipe uses **SkyPilot 0.12.2**, matching NPA's isolated client pin. It is
a deployment recipe, not a live deployment or a change to NPA's run-scoped
workflow APIs. Connect to the team API explicitly through upstream SkyPilot.

## Select the existing platform

Identify the exact kubeconfig, context, API namespace/release, workload namespace,
and endpoint owner. Inspect an existing API before creating another one.
Preserve its state, authentication, credentials, and chart/image version.
Exposing that API is usually sufficient; a second empty API will not inherit
its jobs or cluster identities. Moving a host-based API into Kubernetes needs
a separately planned state migration.

For a fresh API, choose a dedicated namespace such as `skypilot`, a CPU node
pool, storage class, maintained ingress class, and operator-controlled DNS name
and certificate. Reuse the Workbench ingress/Gateway when suitable. The bundled
community ingress-nginx controller was
[retired in March 2026](https://kubernetes.io/blog/2026/01/29/ingress-nginx-statement/);
disable it for new deployments. Replacing an existing ingress-nginx deployment
requires its own migration plan.

The LoadBalancer belongs to the HTTPS ingress/Gateway. The API Service stays
`ClusterIP`, with port 46580 internal. Nebius LoadBalancer Services have public
IPs by default. A private endpoint requires the Service annotation
`nebius.com/load-balancer-type: internal` and verified team connectivity. Retain
an IP across Service recreation with a reusable allocation through
`nebius.com/load-balancer-allocation-id`. Apply these annotations to the
ingress/Gateway's Service, preserving its existing allocation and DNS. See
[Nebius LoadBalancer configuration](https://docs.nebius.com/kubernetes/clusters/load-balancer).

## Fresh API using a maintained ingress

Keep values, password hashes, certificate keys, and rendered manifests in an
owner-only directory outside the checkout. Every Kubernetes or Helm mutation
must name the verified kubeconfig and context. Replace the placeholders before
using the recipe.

```bash
export API_KUBECONFIG='<selected-kubeconfig>'
export API_CONTEXT='<selected-context>'
export API_NAMESPACE=skypilot
export API_RELEASE=skypilot
export API_PRIVATE_DIR="$HOME/.config/skypilot-api"
umask 077
mkdir -p "$API_PRIVATE_DIR"
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" get nodes
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" get ingressclass
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" get storageclass
helm --kubeconfig "$API_KUBECONFIG" --kube-context "$API_CONTEXT" list --all-namespaces
```

Save this as `$API_PRIVATE_DIR/values.yaml`. The CPU node selector must match
verified pool labels: `workbench-pool: cpu` is an example, not a label NPA
installs. The upstream API defaults request 4 CPUs and 8 GiB.

```yaml
apiService:
  image: berkeleyskypilot/skypilot:0.12.2
  replicas: 1
  upgradeStrategy: Recreate
  enableUserManagement: true
  initialBasicAuthCredentials: null
  initialBasicAuthSecret: skypilot-initial-auth
  nodeSelector:
    workbench-pool: cpu
auth:
  serviceAccount:
    enabled: false
storage:
  enabled: true
  storageClassName: <selected-storage-class>
  accessMode: ReadWriteOnce
  size: 10Gi
ingress-nginx:
  enabled: false
ingress:
  enabled: true
  ingressClassName: <maintained-ingress-class>
  host: <api-dns-name>
  path: /
  authSecret: null
  authCredentials: null
  tls:
    enabled: true
```

Authentication and user management run in SkyPilot itself. Basic-auth nginx
annotations do not protect a different ingress controller. The initial secret
creates one admin; create individual team accounts with appropriate roles in
the dashboard. Preserve existing SSO on upgrades. When the team already has an
identity provider, prefer upstream
[SSO configuration](https://docs.skypilot.ai/en/stable/reference/auth.html)
for a fresh deployment.

After confirming this is a fresh installation, create the namespace and initial
secret. `htpasswd` prompts interactively; the password never appears in command
arguments. The secret key must be `auth`.

```bash
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" \
  create namespace "$API_NAMESPACE"
htpasswd -cB "$API_PRIVATE_DIR/initial-auth" '<admin-username>'
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" \
  -n "$API_NAMESPACE" create secret generic skypilot-initial-auth \
  --from-file="auth=$API_PRIVATE_DIR/initial-auth"
```

Use the platform's certificate management. The 0.12.2 ingress expects Secret
`<chart-fullname>-tls-secrets` in the API namespace; for release `skypilot`
without name overrides this is `skypilot-tls-secrets`. For a manually supplied
certificate, then a fresh Helm installation:

```bash
kubectl --kubeconfig "$API_KUBECONFIG" --context "$API_CONTEXT" \
  -n "$API_NAMESPACE" create secret tls "${API_RELEASE}-tls-secrets" \
  --cert="$API_PRIVATE_DIR/tls.crt" --key="$API_PRIVATE_DIR/tls.key"
helm repo add skypilot https://helm.skypilot.co
helm repo update skypilot
helm template "$API_RELEASE" skypilot/skypilot --version 0.12.2 \
  --namespace "$API_NAMESPACE" -f "$API_PRIVATE_DIR/values.yaml" \
  > "$API_PRIVATE_DIR/rendered.yaml"
# Review the rendered manifest before installation.
helm --kubeconfig "$API_KUBECONFIG" --kube-context "$API_CONTEXT" \
  install "$API_RELEASE" skypilot/skypilot --version 0.12.2 \
  --namespace "$API_NAMESPACE" -f "$API_PRIVATE_DIR/values.yaml" --wait
```

Require `--enable-basic-auth`, the initial secret reference, a persistent
`<chart-fullname>-state` claim, and a `ClusterIP` API Service in the rendering.
There should be no bundled ingress-nginx resources. Verify the maintained
controller's LoadBalancer Service, TLS behavior, and DNS separately; rendering
this chart does not create or prove that external endpoint. Configure that
controller to serve the API only over HTTPS, or redirect HTTP to HTTPS before
forwarding requests; an Ingress TLS entry alone does not enforce this behavior
for every controller.

For an existing Gateway API deployment, set `ingress.enabled: false` too and
attach an `HTTPRoute` to its HTTPS listener with backend
`<chart-fullname>-api-service`, port 80. Retain the API-native auth configuration.
See upstream [HTTP routing](https://gateway-api.sigs.k8s.io/guides/user-guides/http-routing/)
and [TLS configuration](https://gateway-api.sigs.k8s.io/guides/user-guides/tls/).

The fresh API uses its in-cluster service account and defaults to workloads in
its namespace. Configure upstream namespace/RBAC options deliberately for a
separate workload namespace, then verify actual pod placement. A client-side
kubeconfig does not select the remote API's backend. This recipe does not
configure direct Nebius VM provisioning or other cloud credentials.

## Existing API and availability

Do not apply fresh-install values wholesale to an existing release. Export its
values and manifest privately, record the chart/image version and PVC identity,
and back up persistent state using the storage provider's supported mechanism.
Prepare only the endpoint changes required by its existing auth and controller.
Use `helm upgrade` with `--reuse-values` and the installed chart version, or an
independently reviewed version upgrade. Preserve state, credentials, namespace,
database settings, and auth secrets.

One replica with SQLite and a ReadWriteOnce PVC survives pod replacement but
has an interruption during `Recreate` upgrades. Persistence is not high
availability or a backup. Rolling upgrades or multiple replicas require
PostgreSQL and storage compatible with the upstream chart's requirements;
do not increase replicas on the baseline PVC. See the upstream
[deployment guide](https://docs.skypilot.ai/en/stable/reference/api-server/api-server-admin-deploy.html).

## Prove the endpoint

From a developer machine and the VDI, verify the DNS name, trusted TLS
certificate, individual-user login, and rejection of unauthenticated access to
a protected API operation. Verify HTTP is refused or redirects to HTTPS without
sending credentials to it. `/api/health` may be publicly readable in native-auth
mode; health alone does not prove authentication or execution readiness.

Use the pinned isolated client from [SkyPilot setup](skypilot-setup.md) and
upstream [connection instructions](https://docs.skypilot.ai/en/stable/reference/api-server.html).
Keep credentials in private client configuration; do not paste credential-bearing
URLs into shell history or shared logs. Require Kubernetes compute access,
expected GPU discovery, and a completed CPU job before GPU submission. A zero
exit code from `sky check` can still mean no infrastructure was enabled.
Verify streamed logs and uploads through the controller: nginx annotations for
timeouts, buffering, or upload sizes do not configure another controller.

During an operator-owned maintenance check, replace only the API pod and
reconnect at the same DNS name. Require the same PVC, users, jobs, cluster
records, and intended workload namespace after restart. Keep the team API and
its storage when cleaning up benchmark jobs. Teardown is separate, after every
workload the API owns has been accounted for.

This recipe remains unqualified for a live endpoint until those checks pass on
the selected cluster, controller, certificate, and storage class.
