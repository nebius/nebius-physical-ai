# Opt-in CPU RayCluster on Fleet

[Docs](README.md)

Use `npa fleet` to provision a fixed CPU RayCluster alongside an existing
Managed Kubernetes declaration. Ray's native Jobs and Core APIs run application
code. Fleet owns the Kubernetes infrastructure and the vendored KubeRay
application; `npa.workflow` remains the durable composition path.

Start from [the reference fleet spec](../npa/examples/fleet/kuberay/cpu-raycluster.yaml).
Copy it outside the repository and select your authorized existing project,
tenant, region and Nebius profile. Validate credentials before provisioning:

```bash
npa workbench health preflight --checks nebius --json
npa fleet plan --spec /path/to/private-fleet.yaml
npa fleet deploy --spec /path/to/private-fleet.yaml --no-create-projects --yes
```

`kuberay` is accepted in the legacy cluster profile or an explicit `mk8s`
envelope. The SDK uses the same policy:

```python
from npa.sdk.fleet import ClusterSpec, KubeRaySpec, NodePoolSpec

cluster = ClusterSpec(
    name="cpu-ray",
    cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="16vcpu-64gb"),
    kuberay=KubeRaySpec(enabled=True, worker_replicas=2,
                       worker_cpus=2, worker_memory_gib=4),
)
cluster.validate()
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Explicitly deploy the CPU RayCluster. |
| `worker_replicas` | `1` | Fixed number of Ray worker pods. |
| `worker_cpus` | `2` | Integer CPU request, limit and Ray CPU capacity per worker. |
| `worker_memory_gib` | `4` | Integer memory request and limit per worker, in GiB. |

Counts and resources must be positive integers. A nonempty, explicit CPU pool
is required. Size it for the head (1 CPU, 4 GiB), workers and Kubernetes system
pods; Fleet rejects a head or worker larger than one declared CPU node and
rejects fixed pod requests above the pool's nominal aggregate CPU or memory.
Leave additional capacity for Kubernetes system pods because this validation is
only an impossibility check, not an allocatable-resource readiness result. Fleet
does not scale infrastructure to satisfy Ray tasks. A per-cluster
`kuberay: {enabled: false}` replaces an enabled defaults block completely.
Unknown fields, RayService, GPU workers, autoscaling and arbitrary image/chart
values are rejected. A GPU pool may coexist, but these Ray pods select only the
CPU platform. The legacy standalone Terraform wrapper stays disabled; use a
one-entry fleet for this opt-in policy.

Enabled KubeRay deployments use the default Terraform workspace and the local
`.terraform` data directory. Unset `TF_CLI_ARGS`, command-specific
`TF_CLI_ARGS_*` and `TF_DATA_DIR`; inherited argument overrides and nondefault
workspaces are rejected before provisioning. Retained custom-backend metadata
is unsupported; deployment requires managed implicit local state. Reapply preserves exact state and
backup files plus the provider cache, and rebuilds Terraform's module manifest
from the reviewed recipe. Additional effective Terraform inputs,
unsupported state paths and destination symlinks are rejected before refreshing
the installation. Keep custom overrides outside the managed installation.

The runtime is Ray 2.58.0 / Python 3.12, pinned to the upstream amd64 image
`docker.io/rayproject/ray@sha256:507464fe56b3d24cec2e812a25850db91b97d52752d63119fecf6914f7b0a37a`.
Head and worker use identical bytes. This consumes an upstream public image
without building or republishing an NPA container. Ray source is
[Apache-2.0](https://github.com/ray-project/ray/blob/ray-2.58.0/LICENSE);
no models, weights or datasets are required. The public NPA image catalog is
unchanged. This path makes no GPU or Ray Train capability claim.

The opt-in template disables autoscaling and bundled monitoring, uses non-root
pods without privilege escalation or service-account tokens, and creates an
Ingress NetworkPolicy before the application. Only pods in its namespace may
connect to Ray; Jobs is a ClusterIP service. Use authenticated Kubernetes access
and loopback forwarding. Anyone authorized to submit a Ray job can execute
arbitrary code with the runtime's permissions. This follows Ray's
[trusted-code security boundary](https://github.com/ray-project/ray/blob/ray-2.58.0/doc/source/ray-security/index.md);
a Ray namespace does not isolate mutually untrusted users. Do not expose Jobs,
GCS or the dashboard through a public load balancer.

## Execute and inspect worker work

Select the exact Fleet-generated kubeconfig for this cluster. In one terminal:

```bash
export KUBECONFIG=/path/to/owned-kubeconfig
kubectl -n ray-cluster get rayclusters,pods,services
kubectl -n ray-cluster port-forward --address 127.0.0.1 service/ray-cluster-head-svc 8265:8265
```

In another terminal, activate a client environment with `ray[default]==2.58.0`
and run from the repository root. Keep the first terminal's port-forward running.
Choose a fresh submission ID and use the explicit local Jobs address:

```bash
unset RAY_ADDRESS RAY_API_SERVER_ADDRESS
export RAY_API=http://127.0.0.1:8265
export RAY_JOB=cpu-worker-proof
ray job submit --address "$RAY_API" --submission-id "$RAY_JOB" \
  --working-dir npa/examples/fleet/kuberay -- python verify_workers.py
ray job status --address "$RAY_API" "$RAY_JOB"
ray job logs --address "$RAY_API" "$RAY_JOB"
```

The result checks a different deterministic square-sum and SHA-256 on each
worker and reports its Ray node identity. Head CPU capacity is zero; application
tasks explicitly target every worker and reject execution on the head. The
output is text-only verification, with no training or visualization artifact.
Save logs and any application outputs outside the cluster before teardown:
Ray's object store and `/tmp` are ephemeral.

Require native status `SUCCEEDED` and a `KUBERAY_RESULT=` log record with the
expected worker count. The template's two workers share one physical CPU node.

Stop any remaining application jobs using
`ray job stop --address "$RAY_API" "$RAY_JOB"`, confirm a terminal state, then
stop the exact port-forward process and destroy the owned fleet target:

```bash
npa fleet status --spec /path/to/private-fleet.yaml
npa fleet destroy --spec /path/to/private-fleet.yaml --yes
```

Verify provider absence of the exact owned cluster, node groups and instances.
Retain the project and unrelated infrastructure. The native live regression is
`npa/tests/e2e/test_fleet_kuberay_live.py`; it requires a private spec and exact
kubeconfig, checks actual deployed resources, submits worker work and observes
native status/logs. It creates no cloud infrastructure itself.

Once an installation has opted in, NPA retains that provenance through later
disabled or omitted KubeRay settings. Teardown rejects inherited Terraform
overrides and requires the materialized recipe and generated variables to match
the recorded deployment digest. It rebuilds cached module mappings before
initialization and verifies that canonical local state has no managed resources
before removing recovery files. Refusal, unreadable state or incomplete teardown
retains the installation for recovery. An installation without recorded deployment
provenance needs a successful reapply of the reviewed recipe before teardown;
do not discard or edit its state to bypass this check.

## Recipe compatibility

Deploy checks the selected recipe before quota checks or project/network
mutation. `kuberay_recipe_contract.json` binds the complete pristine
`k8s-training/` and `modules/` source inventory by SHA-256, including placement,
application wiring and templates. Added Terraform overrides, auto-loaded values,
symlinks, special files, unrecorded directories (including empty ones) and changed
source files fail closed. Directory names must be parents of the recorded files. Both source roots
must be real directories. Use a pristine source directory;
Fleet generates state, variables and region adjustments in its private copy.
An older pinned recipe or upstream `main` without that exact inventory is
unsupported. Installations that have never opted in retain the existing default
behavior. Updating the
contract requires reviewing the effective resources and repeating live proof;
merely declaring Terraform variables is insufficient.
