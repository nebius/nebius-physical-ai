# SkyPilot Isolated Venv Setup

[Docs](../README.md)

SkyPilot is an external CLI dependency for NPA orchestration. NPA calls the
`sky` CLI through subprocess and does not install or import SkyPilot in NPA's
Python environment.

Isolated workflow execution requires a **Linux operator host with `/proc`**.
NPA verifies the local API's process lifetime, environment, session, and socket
ownership through Linux procfs. macOS can install NPA and validate or plan a
workflow, but cannot run this isolated API. Use a Linux workstation or VM for
setup, submission, monitoring, recovery, and cleanup, keeping its credentials
and run state there. The GPU workload still runs on the selected Nebius cluster.
An unsupported host fails before the isolated API creates state or processes;
removing isolation does not provide an equivalent supported workflow path.

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

Cluster validation uses one owned local API session for the credential check,
GPU discovery, launch, and cleanup. It selects the requested SkyPilot interpreter
and a durable working directory even when another SkyPilot API is already
running on the host. Concurrent validation using the same state is serialized.
After confirming removal of the smoke workload, NPA stops the session's API.
When removal cannot be verified, keep the reported validation state and the
original environment for recovery. Rerun the original validation command with
that environment: NPA removes its recorded smoke before launching another.
NPA preserves the existing shared API.

Validation state lives under `cluster-validation/` in the selected NPA
configuration directory, or under the configured SkyPilot isolated directory
when one is selected. Keep `current-session.json` and its referenced
`session-*/` directory together. Completed sessions retain their evidence;
the next invocation creates a new session so it can use newly selected
credentials or an updated interpreter without changing an active session.

The owned API follows Nebius CLI `0.12.254` authentication selection: `--config`
selects the profile file, `--profile` overrides `NEBIUS_PROFILE` and the saved
default, and the renewable token cache remains `HOME/.nebius/credentials.yaml`.
`NEBIUS_CONFIG_DIR` does not redirect either CLI file. Isolated SkyPilot homes
retain the incoming CLI home configuration. For a verified RSA service-account
profile, normal cache refresh, creation, and pruning preserve API identity;
the effective profile file, account, key, and explicit credential sources remain
bound. Unsupported or mixed authentication formats remain byte-strict.
Existing API ownership records are never silently rebound to a new file model.
Preserve the original environment and exact controller records when an older
session reports a credential mismatch; do not edit its ownership record.

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

Controller cleanup uses the originating isolated runtime's user and controller
identity. It snapshots controller metadata into a temporary owned API, without
replaying the original API's pending requests, and verifies remote absence before
removing the original local metadata. The temporary API and queue must stop
before their directory is deleted. If process cleanup cannot be verified, keep
`<isolated-config-dir>/controller-transactions/` and the original runtime for
recovery; do not remove their ownership records or use a shared API as a fallback.
