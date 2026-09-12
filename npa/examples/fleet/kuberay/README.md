# Run native Ray work on a fixed CPU cluster

[Fleet guide](../../../../docs/fleet-kuberay.md) · [Repository](../../../../README.md)

Deploy a CPU RayCluster on Nebius Managed Kubernetes, then verify work on every
Ray worker. Fleet owns the infrastructure; native Ray Jobs runs the application.
The example needs no GPU, model, or dataset.

## Before you start

- Install NPA, Terraform, `kubectl`, and the authenticated Nebius CLI; follow
  [first-run setup](../../../../docs/workbench/getting-started.md).
- Select an authorized existing project with quota for Kubernetes, one CPU node,
  and its boot disk. The template uses one `16vcpu-64gb` CPU node and two Ray
  worker pods. These are two Ray workers on one host, not two physical hosts.
- Keep your fleet YAML, kubeconfig, and validation evidence outside the checkout.

## Deploy and check readiness

From the repository root, copy [cpu-raycluster.yaml](cpu-raycluster.yaml) to a
private path. Set its `tenant_id`, `region`, `profile`, and existing project's
`name` and `project_id`; leave `profile` empty to use the authenticated default.
Use that same path throughout the lifecycle:

```bash
export FLEET_SPEC="/absolute/private/cpu-raycluster.yaml"
npa workbench health preflight --checks nebius --json
npa fleet plan --spec "$FLEET_SPEC"
npa fleet deploy --spec "$FLEET_SPEC" --no-create-projects --yes
npa fleet status --spec "$FLEET_SPEC"
```

Select this cluster's exact Fleet-generated kubeconfig, then require the head
and both workers to be ready:

```bash
export KUBECONFIG="/absolute/private/fleet-generated-kubeconfig"
kubectl -n ray-cluster get rayclusters,pods,services
```

A successful deploy is followed by **three ready Ray pods**, a fixed two-worker
RayCluster, and a ClusterIP head service. Use the
[native submit guide](../../../../docs/fleet-kuberay.md#execute-and-inspect-worker-work)
for authenticated loopback access and `ray job submit/status/logs/stop`.

## Verify actual worker execution

[verify_workers.py](verify_workers.py) computes 10,000 square values per worker
and verifies distinct Ray worker placement, sums, and SHA-256 digests. Successful
logs include `KUBERAY_RESULT=` with `worker_count: 2`. The output is a text
verification record; data inside the cluster remains ephemeral.

The committed live test also verifies deployed image digests, pod resources,
network settings, native job status, and removal of its own staged source. It
uses an already deployed cluster. Create a mode-0600 JSON file outside the
checkout with these keys (the evidence directory must be mode 0700):

```json
{
  "spec": "/absolute/private/cpu-raycluster.yaml",
  "kubeconfig": "/absolute/private/fleet-generated-kubeconfig",
  "evidence_dir": "/absolute/private/cpu-ray-evidence"
}
```

From the repository root:

```bash
export NPA_FLEET_KUBERAY_LIVE_CONFIG="/absolute/private/cpu-ray-live.json"
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_fleet_kuberay_live.py -q
```

Expected: `1 passed`. Without the private configuration and live-test gate,
ordinary test runs skip this cloud access.

## Change worker capacity

| Configuration key | Default | Meaning |
| --- | --- | --- |
| `kuberay.enabled` | `false` | Explicitly enable the CPU RayCluster. |
| `kuberay.worker_replicas` | `1` | Fixed positive number of worker pods. |
| `kuberay.worker_cpus` | `2` | CPU request and limit per worker. |
| `kuberay.worker_memory_gib` | `4` | Memory request and limit per worker, in GiB. |

Worker settings require `enabled: true`. Size the CPU pool for the head, workers,
and Kubernetes system pods. `kuberay: {enabled: false}` fully disables an inherited
policy. This path uses the pinned public upstream Ray CPU image and supports
neither autoscaling nor GPU Ray workers.

## Finish

Save needed logs and outputs, stop remaining exact Ray submission IDs, and close
any port-forward process. Then destroy the same owned fleet target:

```bash
npa fleet destroy --spec "$FLEET_SPEC" --yes
```

Verify that its cluster, node groups, and instances are absent. Fleet retains the
project; preserve unrelated infrastructure and private ownership receipts. See
[the teardown contract](../../../../docs/fleet-kuberay.md#execute-and-inspect-worker-work)
if cleanup refuses changed Terraform inputs or incomplete state.
