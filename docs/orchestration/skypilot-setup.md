# SkyPilot Isolated Venv Setup

SkyPilot is an external CLI dependency for NPA orchestration. NPA calls the
`sky` CLI through subprocess and does not install or import SkyPilot in NPA's
Python environment.

## Install SkyPilot

Create a dedicated virtualenv and install the validated SkyPilot pin:

```bash
python -m venv /opt/npa/skypilot
/opt/npa/skypilot/bin/pip install 'skypilot[nebius,kubernetes]==0.12.2'
```

## Point NPA At It

Set `NPA_SKYPILOT_BIN` to the venv's `sky` executable:

```bash
export NPA_SKYPILOT_BIN=/opt/npa/skypilot/bin/sky
```

Python callers can also pass `sky_bin=` directly to
`npa.orchestration.skypilot` wrapper functions. If neither is set, NPA falls
back to discovering `sky` on `PATH`.

## PATH Alternative

You can put the isolated venv on `PATH` instead of setting
`NPA_SKYPILOT_BIN`:

```bash
export PATH=/opt/npa/skypilot/bin:$PATH
```

## Verify

The CLI helper `npa workflow check-skypilot` is reserved for the follow-up
CLI/SDK surface work. Until it lands, verify the same contract directly:

```bash
test -x "$NPA_SKYPILOT_BIN"
"$NPA_SKYPILOT_BIN" check nebius kubernetes
```

## Upgrade

The validated version is SkyPilot `0.12.2` with the `nebius` and `kubernetes`
extras. To upgrade, create a new venv, install the candidate SkyPilot version,
run `sky check nebius kubernetes`, and replay the NPA SkyPilot e2e before
switching `NPA_SKYPILOT_BIN`.
