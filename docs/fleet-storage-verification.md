# Verify shared storage across a Fleet

`npa fleet verify-storage` proves that the filesystem declared by an existing
Fleet works on every CPU and GPU worker. It checks the host mount first, then
uses a temporary ReadWriteMany PVC and pods pinned to each exact worker to
prove shared reads and writes across the cluster.

```bash
npa fleet verify-storage --spec <private-fleet.yaml> --output json
```

Use the same owner-private spec used to deploy the Fleet. The command resolves
each target from NPA's registered cluster identity and verifies its project,
tenant, region, and cluster against fresh provider responses. It also compares
the registered kubeconfig's endpoint, certificate authority, and exec
authentication with newly obtained provider connection metadata. The registered
kubeconfig and machine's active profile are preserved.

## Prerequisites and selection

Install the checkout with `pip install -e npa`, and make the Nebius CLI
available. The operator must already have access to the selected
clusters and permission to inspect nodes and CSI resources, create temporary
pods and PVCs, execute probes, and delete those owned resources. The command
does not create IAM permissions, provision clusters, resize workers, or change
the filesystem configuration.

Filesystem-enabled targets require the supported filesystem CSI driver and its
default StorageClass, `csi-mounted-fs-path-sc`. The probe uses the immutable
image pinned by the installed NPA implementation. It needs no customer data,
model weights, or GPU allocation.

The existing Fleet selectors restrict verification before any probe is created:

```bash
npa fleet verify-storage --spec <private-fleet.yaml> \
  --only-projects <project-key> --only-clusters <cluster-name> \
  --profile <operator-profile> --output json
```

`--only-projects` accepts comma-separated project keys or display names.
`--only-clusters` accepts comma-separated cluster names within those projects.
`--project-prefix` overrides the prefix used to resolve project display names.
`--profile` overrides the spec's authentication profile. Unknown selectors,
missing registration, or mismatched identity fail before a probe runs for that
target. A filesystem explicitly disabled in the spec is reported as skipped.

## What a pass proves

Every filesystem-enabled target must pass all of these checks:

1. Every declared worker is present with the expected CPU or GPU pool identity.
2. The spec's exact host path is actively mounted read-write as `virtiofs`, with
   the declared source tag and a matching `/etc/fstab` entry containing `nofail`.
3. The mount reports a known capacity compatible with the requested size in
   binary gibibytes: one GiB is exactly `1,073,741,824` bytes. Decimal GB and
   unknown capacity cannot satisfy this check.
4. A unique host probe is written, checksummed, read, and deleted on every worker.
5. The expected default StorageClass and filesystem CSI components are healthy.
6. One temporary ReadWriteMany PVC mounts on every exact worker. Each pod writes
   its own payload, and every pod reads and checksums all workers' payloads.
7. Probe paths and all owned temporary Kubernetes resources are absent after
   cleanup. Worker replacement or stale or partial evidence cannot produce a pass.

The verifier never lists or reads pre-existing filesystem entries. It creates
unique non-sensitive paths under a dedicated NPA probe directory. Concurrent
invocations have independent ownership identities. Cleanup uses captured
resource UIDs and ownership labels, continues after individual cleanup errors,
and affects only the current invocation's resources and files. Cleanup failure
fails verification even when the reads and writes succeeded.

Host cleanup and absence audits recheck the declared mount source, device,
capacity, and `nofail` configuration. An unmounted filesystem cannot pass by
exposing an empty local directory beneath its former mount point.

A cleanup pass also requires all owned writers to stop. Each node then requests
server-synchronized attributes for the exact probe paths using Linux
[`statx` with `AT_STATX_FORCE_SYNC`](https://man7.org/linux/man-pages/man2/statx.2.html).
This identifies deleted inodes even while a worker retains an old directory
cache entry. Missing paths or synchronized zero link counts prove removal;
remaining linked entries and unsupported synchronization fail verification.

## Output and private evidence

Text and JSON output contain only target and worker counts, requested capacity,
per-target and per-node indexes, pass/fail categories, evidence hashes, and
cleanup counts. They do not contain infrastructure names, endpoints, probe
payloads, credentials, or raw provider receipts. `--output-format` is an alias
for `--output`; JSON emits one document and verification failure exits nonzero.

Provide an owner-private directory outside the checkout to retain exact evidence:

```bash
npa fleet verify-storage --spec <private-fleet.yaml> \
  --evidence-dir <owner-private-directory> --output json
```

Keep that directory outside Git and public collaboration surfaces. Share only
the sanitized report and its cryptographic evidence hashes. The report proves
the observed mount and sharing behavior; it does not simulate a reboot, change
filesystem durability policy, or benchmark storage throughput.

## Python SDK

The SDK invokes the same implementation as the CLI:

```python
from pathlib import Path
from npa.sdk import fleet

spec = fleet.load_spec(Path("<private-fleet.yaml>"))
report = fleet.verify_storage(
    spec,
    only_projects=["<project-key>"],
    evidence_dir=Path("<owner-private-directory>"),
)
if not report["passed"]:
    raise RuntimeError("Fleet storage verification failed")
```

## Live regression

The committed E2E runs the installed implementation against every target and
worker in an owner-selected spec. Expected counts and requested capacity totals
come from that declaration:

```bash
NPA_INTEGRATION_E2E=1 NPA_FLEET_STORAGE_VERIFY=1 \
  NPA_FLEET_STORAGE_VERIFY_SPEC=<private-fleet.yaml> \
  NPA_FLEET_STORAGE_EVIDENCE_DIR=<owner-private-directory> \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_fleet_storage_verification_live.py -q
```

The daily runner's `e2e-daily` tier includes this regression only when
`NPA_FLEET_STORAGE_VERIFY=1` and the private spec and evidence configuration are
provided. Ordinary repository tests mock provider and Kubernetes operations and
do not contact live infrastructure.
