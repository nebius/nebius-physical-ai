# SkyPilot Isolated Venv Setup

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
npa skypilot verify
```

## Managed-Jobs Controller

NPA defaults SkyPilot managed jobs to a Kubernetes controller:

```yaml
jobs:
  controller:
    resources:
      cloud: kubernetes
      cpus: 4
      memory: 16
```

Do not set `disk_size` for this controller mode. SkyPilot 0.12.2's Kubernetes
backend does not apply custom controller disk sizing; it uses the cluster's
pod storage behavior.

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
