# Burst task examples

Single-task SkyPilot YAMLs that are inputs to **`npa burst submit-yaml`**, not workflow
templates.

`npa.burst.core.submit_yaml()` is deliberately scoped to *one* executable SkyPilot task: it
loads the document, substitutes `${VAR}` placeholders from `--var`, refuses to submit while any
placeholder is unresolved, and forwards explicit exact-host registry credentials when
the operator supplies them. That is a different capability from the workflow surface — there is no plan, no
stage graph, no decision artifact, and nothing for a `toolRef` to describe.

These files live here rather than in the retired raw SkyPilot workflow catalog so that burst
can keep its single-task path without reopening a workflow authoring surface. The same reasoning
relocated the BYOF resource profiles to `npa/src/npa/workflows/byof/profiles/` (see
`DESIGN.md` §R10).

| File | What it exercises |
| --- | --- |
| `isaac-lab-cosmos-sdg-burst-smoke.yaml` | Isaac Lab headless Cartpole training on a burst GPU, then a Cosmos SDG transfer-contract manifest. One task on purpose. |

## Rules

- **One task per file.** `submit_yaml` rejects anything else, and a multi-stage pipeline is a
  workflow: author an `npa.workflow/v0.0.1` spec under
  `npa/workflows/workbench/npa-workflows/` instead.
- **Keep `${VAR}` placeholders.** They are the burst substitution surface; concrete registry
  ids, bucket names and run ids must never be committed here.
- `npa/tests/guardrails/test_burst_examples.py` pins both rules.

## Submitting

```bash
npa burst submit-yaml npa/src/npa/burst/examples/isaac-lab-cosmos-sdg-burst-smoke.yaml \
  --name <run-id> \
  --var NPA_RUN_ID=<run-id> \
  --var ISAAC_LAB_IMAGE=<registry>/npa-isaac-lab:3.0.0b2.post1 \
  --var NPA_OUTPUT_URI=s3://<your-bucket>/<prefix>/<run-id>/
```
