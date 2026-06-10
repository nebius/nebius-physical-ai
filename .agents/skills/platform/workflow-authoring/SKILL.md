---
name: workflow-authoring
description: Use when authoring, refactoring, or reviewing NPA SkyPilot workflow YAMLs for readability, DRY-ness, agent-editability, and many-variant scale.
---

# Workflow Authoring (thin YAML, fat tool)

How to keep NPA SkyPilot workflows pretty, agent-editable, and scalable across
many workflows and customer variants. Pairs with `skypilot-workflows` (runtime)
and `workflows` (reference YAMLs).

## Core principle: thin YAML, fat tool

The YAML declares the DAG, resources, and env contract. **Workflow logic lives
in the versioned, tested `npa` CLI/SDK — never in inline bash or a custom image.**
SkyPilot's own guidance is that `run:` should invoke your program (e.g.
`npa workbench ...`), and `setup:` is only for one-time dependency install.

- Good `run:` → `npa workbench lancedb import-bdd100k --service --endpoint "$LANCEDB_ENDPOINT" ...`
- Bad `run:` → 80 lines of `curl`/`jq` with copy-pasted helper functions.

If a step's logic is not yet a CLI/SDK call, add it to the tool, don't encode it
in YAML bash. (Tracked gaps for BDD100K: `detection-training train` needs
`--label-map` + `--wait/--poll-seconds/--timeout`; `eval` needs
`--from-view-latest` + `--write-canonical-metrics`. The `train` schema already
accepts `label_map`.)

## Vanilla images

Task pods use a **stock, digest-pinned base image** (e.g.
`python:3.11-slim@sha256:...`) plus `pip install npa==<ver>` in `setup` (or a
`workdir`/`file_mounts` repo sync). Do **not** bake workflow tooling into custom
images. GPU work lives in the in-cluster **services** (LanceDB,
detection-training), so SkyPilot task pods that only call those services are
**CPU-only** — do not request `accelerators` on a pod that just makes HTTP calls.

## DRY via generation, not copy-paste

SkyPilot 0.12.2 has **no native Jinja/anchors across multi-doc YAML** and `envs`
cannot self-reference. So:

- Define a workflow from a **compact spec** (lists of stages and views) and emit
  the SkyPilot YAML. Collapse repeated tasks (e.g. train×N / eval×N) with a loop
  over the view list — one template, not N copies.
- Keep the **generated YAML a committed, inspectable artifact** so SkyPilot
  stays the contract and customers can read/hand-edit it. The generator emits
  SkyPilot; it never replaces it.
- Reference generator: `npa/scripts/generate_bdd100k_pipeline.py` (emits `--runner
  cli` target and `--runner curl` behaviour-equivalent variants); reference
  renderer/submitter: `render_pipeline()` in `npa/scripts/run_bdd100k_pipeline.py`
  + `npa.orchestration.skypilot.submit_workflow`.

## Many variants without forking

Per-run and per-customer differences (endpoints, image digests, GPU types,
thresholds, label maps) are **env overrides**, not new YAML files:

- SDK: `sky.Task.from_yaml(...).update_envs({...})` then `sky.jobs.launch(...)`.
- CLI: `sky jobs launch ... --env-file <customer>.env` (dotenv per variant).
- Mark required envs by setting them to `null` in the YAML so SkyPilot errors if
  unset.

A variant is an overlay file, never a 1,000-line copy. Concrete example:
`npa/workflows/workbench/skypilot/examples/customer-variant.env` overlays the
generated `bdd100k-pipeline.generated.yaml` (run id, endpoints, real-data label
map, etc.) — apply with `sky jobs launch --env-file customer-variant.env ...`.

## Checklist for a "pretty" workflow

- [ ] `run:` calls `npa`/program, not inline `curl`/`jq` plumbing.
- [ ] Stock digest-pinned base image; `setup:` is just dependency install.
- [ ] No duplicated tasks — repetition comes from a loop/spec.
- [ ] Task pods request GPU only if they do GPU work in-pod.
- [ ] Variants via env/`--env-file`, not forked YAML.
- [ ] Validated with `run_bdd100k_pipeline.py --mock-endpoints` before live submit.
- [ ] If a YAML snapshot-hash test pins the file, update the pin (or test the
      generator + a golden output instead).
