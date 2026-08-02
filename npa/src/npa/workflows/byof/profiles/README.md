# BYOF SkyPilot resource profiles

These are **resource profiles**, not workflow templates. Each one describes the pod a
BYOF workload runs in — accelerator, CPU/memory floors, the image placeholder the runner
substitutes, and the smoke command for that workload — and nothing else.

The workflow surface is the `npa.workflow` spec
[`npa/workflows/workbench/npa-workflows/byof.yaml`](../../../../../workflows/workbench/npa-workflows/byof.yaml).
Its `workbench.byof.repo` toolRef runs `npa workbench byof run`, which passes one of
these files through `--yaml {{config.resource_profile_yaml}}`. Authoring a BYOF pipeline
means editing the spec; picking a *pod shape* means picking a profile here.

| Profile | Workload | Shape |
| --- | --- | --- |
| `isaac-lab-rl-train.yaml` | `rl-train` (default) | Kubernetes `L40S:1` |
| `isaac-lab-rl-train-rtxpro.yaml` | `rl-train` on RTX PRO | `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` |
| `isaac-lab-rl-train-rtxpro-smoke.yaml` | `rl-train` smoke | RTX PRO, `num_envs=4`, `iterations=1` |
| `byof-datagen-rtxpro-smoke.yaml` | `datagen` smoke | RTX PRO, scripted LeIsaac datagen |
| `byof-container-smoke-rtxpro.yaml` | `container-verify` / `solution-smoke` | CPU only |
| `byof-solution-smoke-rtxpro-gpu.yaml` | `solution-smoke` needing CUDA/EGL/Vulkan | RTX PRO |
| `skypilot-kubernetes-rtxpro.yaml` | *not a task* — SkyPilot **global config** (`--config`) setting `imagePullSecrets` | — |

## Why they are here and not in the SkyPilot catalog

`npa/src/npa/workflows/skypilot/` is being retired so that `npa.workflow` specs are the
only workflow authoring surface (see the repo-root `DESIGN.md`, "Retiring the raw
SkyPilot task catalog"). These profiles are reached *through* that surface, so they moved
out of the catalog rather than being deleted with it — the same move
`byof-solution-smoke-rtxpro-gpu.yaml` and `skypilot-kubernetes-rtxpro.yaml` already made.

Selection lives in `npa/src/npa/workflows/byof/live.py::resolve_byof_resource_yaml`
(env override → project config → profile default), and the runners
(`npa/scripts/run_isaac_lab_rl.py`, `run_byof_datagen.py`,
`run_byof_container_verify.py`) take `--yaml` so a customer can supply their own.

## Do not add a multi-stage pipeline here

If you find yourself chaining stages, that is a workflow: author an
`npa.workflow/v0.0.1` spec under `npa/workflows/workbench/npa-workflows/` instead.
`npa/tests/guardrails/test_byof_profiles.py` keeps these files single-task.
