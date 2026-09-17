# SkyPilot Isolated Venv Setup

[Docs](../README.md)

SkyPilot is an external CLI dependency for NPA orchestration. NPA calls the
`sky` CLI through subprocess and does not install or import SkyPilot in NPA's
Python environment.

## Install SkyPilot

Create or reuse the dedicated virtualenv with the validated SkyPilot pin:

```bash
npa skypilot bootstrap
```

By default this installs `skypilot[nebius,kubernetes]==0.12.2` into
`~/.npa/skypilot-venv`. Use `--path` or `NPA_SKYPILOT_VENV_PATH` when an
operator-managed location is required.

## Point NPA At It

Set `NPA_SKYPILOT_BIN` to the venv's `sky` executable:

```bash
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
```

Python callers can also pass `sky_bin=` directly to
`npa.orchestration.skypilot` wrapper functions. If neither is set, NPA exits
with setup guidance rather than importing SkyPilot into NPA's Python
environment.

## PATH Alternative

You can put the isolated venv on `PATH` instead of setting
`NPA_SKYPILOT_BIN`:

```bash
export PATH="$(dirname "$(npa skypilot status --bin-path)"):$PATH"
```

## Verify

```bash
test -x "$NPA_SKYPILOT_BIN"
npa skypilot verify --cluster "<npa-cluster-context>"
```

Passing the NPA cluster context is important on workstations that already use
SkyPilot with another Kubernetes cluster. NPA pins the check to that exact
context and requires SkyPilot to report `Kubernetes: enabled`; a zero exit code
with Kubernetes disabled is a failed verification. A bare `npa skypilot verify`
remains a legacy runtime/dependency check and does not require Kubernetes merely
because Kubernetes is the default controller backend; `--cluster`,
`--kubeconfig`, or an explicit `--controller-backend kubernetes` opts into the
strict Kubernetes gate.

To prove GPU execution without creating a managed-jobs controller, use NPA's
built-in smoke task:

```bash
npa provision-if-absent \
  --project "<project-alias>" \
  --cluster-name "<npa-cluster-context>" \
  --context "<npa-cluster-context>" \
  --kubeconfig "<kubeconfig>" \
  --skip-s3 \
  --sky-smoke \
  --sky-bin "$NPA_SKYPILOT_BIN"
```

Omitting `--accelerator` makes the smoke discover a compatible GPU from the
selected cluster; pass (for example) `--accelerator RTXPRO6000:1` to require a
specific family. The smoke also runs when the cluster kubeconfig is already
cached. Before either cached or fresh smoke, NPA explicitly reports and performs
an idempotent node-label repair only for its reviewed GPU aliases, then waits for
SkyPilot discovery. This mutation requires Kubernetes `get/list` on nodes plus
`patch/update` when a label is absent; an RBAC failure is reported immediately.
Every SkyPilot check, GPU discovery, launch, status poll, and cleanup remains
scoped to the selected context, and the command succeeds only after the GPU task
completes and its ephemeral SkyPilot cluster is removed.

## Managed-Jobs Controller

NPA defaults SkyPilot managed jobs to a Kubernetes controller:

```yaml
jobs:
  controller:
    resources:
      cloud: kubernetes
      cpus: 4
      memory: 16
      autostop: false
```

`autostop: false` is deliberate. Workbench and SONIC submissions generate a
per-submit `SKYPILOT_GLOBAL_CONFIG`, so the generated config must carry the
jobs-controller resource autostop policy instead of relying on the runner's
default `~/.sky/config.yaml`. This keeps fresh managed-job submits from racing a
shared controller that is already entering `AUTOSTOPPING`.

Do not set `disk_size` for this controller mode. SkyPilot 0.12.2's Kubernetes
backend does not apply custom controller disk sizing; it uses the cluster's
pod storage behavior.

For declarative Kubernetes `npa.workflow` resource profiles, `disk_size` has one
explicit contract: the NPA renderer translates it to SkyPilot
`ephemeral_storage`. This requests pod `ephemeral-storage` capacity instead of
silently emitting the Kubernetes-ignored boot-disk field. Non-Kubernetes
profiles preserve the renderer's prior behavior and do not forward this field.
Raw SkyPilot YAMLs, including the controller configuration above, are not
translated and should use `ephemeral_storage` directly when they need a
Kubernetes storage request.

The Kubernetes controller requires an MK8s node that can fit a 4 vCPU, 16 GiB
pod. The validated `npa-workbench-eu-north1` pattern uses a dedicated CPU node
group such as `cpu-e2/8vcpu-32gb` so the controller does not compete with GPU
workloads.

The previous Nebius VM controller remains available as a fallback from Python
callers:

```python
submit_workflow(yaml_path, run_id, controller_backend="nebius")
```

Use VM controller mode only if the Kubernetes cluster cannot host the
controller pod. It is not the default and should not be required for
properly-sized clusters.

## Upgrade

The validated version is SkyPilot `0.12.2` with the `nebius` and `kubernetes`
extras. To upgrade, create a new venv at a separate path, install the candidate
SkyPilot version, run `sky check`, and replay the NPA SkyPilot e2e before
switching `NPA_SKYPILOT_BIN`.

### Upgrade checklist

The `0.12.2` pin is coupled to every site below. When bumping the version,
update all of them together and re-run the SkyPilot guardrail tests:

1. `npa/src/npa/orchestration/skypilot/_bin.py` — `REQUIRED_SKYPILOT_VERSION`
   (canonical pin; `npa skypilot bootstrap` and `_bin` fail closed on mismatch).
   `npa/src/npa/cli/skypilot/__init__.py` aliases it as `SKYPILOT_VERSION`.
2. `npa/src/npa/cli/skypilot/constraints-0.12.2.txt` — rename to the new
   version and re-resolve the pinned dependency set.
3. `npa/src/npa/orchestration/skypilot/local_api.py` — health-check version
   assertion against the isolated API daemon (`health.get("version")`).
4. `npa/src/npa/orchestration/skypilot/native_preflight.py` —
   `sky.__version__` assertion in the native preflight.
5. `npa/src/npa/orchestration/skypilot/controller_clone.py` —
   `sky.__version__` assertion when cloning controller state.
6. `npa/src/npa/orchestration/skypilot/image_bootstrap_contract.py` —
   `CONTRACT_VERSION` (e.g. `skypilot-0.12.2-v1`); bump the `-vN` suffix and
   re-attest images.
7. `npa/src/npa/deploy/images.py` — attestation label values referencing the
   contract version (e.g. `org.nebius.npa.skypilot-bootstrap-contract=
   skypilot-0.12.2-v1`); rebuilt images must be re-attested.
8. Behavior mirrors of SkyPilot 0.12.2 internals — re-verify each against the
   new version and update the versioned comments:
   - `orchestration/skypilot/resource_quantities.py` (memory-suffix normalization)
   - `orchestration/skypilot/k8s_gpu_catalog.py` (GFD label handling)
   - `orchestration/skypilot/launch_transaction.py` (`jobs launch --name` idempotency)
   - `orchestration/skypilot/workflow_state.py` (S3 mount env vars)
9. `docs/orchestration/skypilot-setup.md` — this section (validated version).

Controller cleanup uses the originating isolated runtime's user and controller
identity. It snapshots controller metadata into a temporary owned API, without
replaying the original API's pending requests, and verifies remote absence before
removing the original local metadata. The temporary API and queue must stop
before their directory is deleted. If process cleanup cannot be verified, keep
`<isolated-config-dir>/controller-transactions/` and the original runtime for
recovery; do not remove their ownership records or use a shared API as a fallback.
