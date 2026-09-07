# Fixed CPU RayCluster example

This example opts a Fleet Managed Kubernetes cluster into KubeRay. Fleet owns
the infrastructure; native Ray Jobs and Core APIs execute application code on a
fixed CPU worker group. KubeRay is disabled when its configuration is omitted.

Copy [cpu-raycluster.yaml](cpu-raycluster.yaml) outside the repository and set
your authorized tenant, project, region and profile. Choose a nonempty CPU pool
large enough for the head, workers and Kubernetes system pods.

| Configuration key | Default | Meaning |
| --- | --- | --- |
| `kuberay.enabled` | `false` | Explicitly enable the CPU RayCluster. |
| `kuberay.worker_replicas` | `1` | Fixed positive number of worker pods. |
| `kuberay.worker_cpus` | `2` | Positive integer CPU request and limit per worker. |
| `kuberay.worker_memory_gib` | `4` | Positive integer memory request and limit in GiB. |

Worker settings require `enabled: true`. The example selects two workers;
`kuberay: {enabled: false}` fully disables an inherited default policy.

```bash
npa workbench health preflight --checks nebius --json
npa fleet plan --spec /path/to/private-fleet.yaml
npa fleet deploy --spec /path/to/private-fleet.yaml --no-create-projects --yes
```

Use the exact Fleet-generated kubeconfig and authenticated loopback forwarding
to access Ray Jobs. [The fleet KubeRay guide](../../../../docs/fleet-kuberay.md)
describes access, native submit/status/log/stop commands and teardown.
[verify_workers.py](verify_workers.py) computes and verifies 10,000 square values
per worker, with distinct worker placement, sums and SHA-256 digests. Its output
is a text verification record; application data in the cluster is ephemeral.

The optional live regression reads `NPA_FLEET_KUBERAY_LIVE_CONFIG`. It is unset by
default, so ordinary test runs skip live access. To run it, set the variable to
an owner-private JSON file containing `spec` (the one-target fleet YAML path),
`kubeconfig` (the exact cluster credential path), and `evidence_dir` (a private
directory with mode 0700). After deploying the cluster and observing all Ray pods
ready, run from the repository root:

```bash
npa/.venv/bin/python -m pytest npa/tests/e2e/test_fleet_kuberay_live.py -q
```

Stop remaining jobs and save needed output before `npa fleet destroy`. Verify
the exact owned resources are absent and retain the project and unrelated
infrastructure. This example uses the pinned public upstream Ray CPU image;
it does not build an NPA image or exercise GPU training.
