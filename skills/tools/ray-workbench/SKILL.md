---
name: ray-workbench
description: "Choose and operate the supported Ray paths in Nebius Physical AI: native Jobs/Core development, guarded Train V2, Cosmos3 native Serve, or Fleet KubeRay."
---

# Ray in Workbench

Start with the [consolidated guide](../../../docs/workbench/ray.md). Use this
skill to route a Ray request to behavior that current code actually supports.

## Route the request

- Native Jobs or Ray Core GPU development: use
  `docs/testing/fast-source-iteration.md` and the guarded CLIP application under
  `npa/workflows/workbench/ray-clip-development/`.
- Ray Train: load `skills/tools/ray-train-synthetic/SKILL.md`. The shipped path
  is a synthetic Ray Train V2 application submitted through native Jobs, not an
  NPA Train service or workflow.
- Ray Serve: load `skills/tools/cosmos3-ray-serve/SKILL.md`. Only Cosmos3-Nano
  has a first-class native Ray Serve path.
- KubeRay: load `skills/tools/fleet/SKILL.md`. Fleet supports only the reviewed,
  fixed CPU RayCluster policy documented in `docs/fleet-kuberay.md`.
- Ray Data or Ray Tune: state that no first-class Workbench recipe, CLI, service,
  workflow, or artifact contract exists. A trusted user may submit their own Ray
  application through a supported native Jobs environment; do not imply that
  this makes Data or Tune a supported Workbench capability.

There is no `npa ray` or NPA Jobs controller. Native `ray job` owns application
submission, list, status, logs, and stop. SkyPilot owns the CLIP and Train hosts
and long-running service task. Fleet owns KubeRay infrastructure. The Cosmos
client owns verified S3 batch publication. Keep these lifecycle identities
separate.

## Operating rules

1. Before provisioning, use `npa workbench health preflight --checks nebius
   --json`; add the selected capability's storage and model-access checks.
2. For SkyPilot references, run `npa skypilot bootstrap` and resolve
   `NPA_SKYPILOT_BIN` from `npa skypilot status --bin-path`. Never use ambient
   management Ray discovery or `sky` from `PATH`.
3. Keep Jobs and GCS on private networking behind an authenticated loopback
   tunnel. Ray accepts trusted code and a namespace is not a tenant boundary.
4. Use unique native submission IDs. `ray job stop` must reach terminal status
   before hosting teardown.
5. Preserve the path-specific artifact boundary before cleanup: checksummed
   CLIP output, Train checkpoints/exports, verified Cosmos S3 publications, or
   explicitly copied KubeRay application output.
6. Cancel exact application jobs before the exact SkyPilot service task or owned
   Fleet target. Preserve shared clusters, APIs, controllers, projects, and
   storage unless their ownership is separately established.

Do not add Ray Data/Tune examples, GPU KubeRay, RayService, a Jobs wrapper, or a
new `npa.workflow` surface without implementation and live evidence. Do not
describe multiple GPUs on one node as multi-node proof; require actual Ray node
and physical-host evidence.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
npa/.venv/bin/python -m pytest --collect-only -q
```
