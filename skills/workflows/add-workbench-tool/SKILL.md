---
name: add-workbench-tool
description: Use when adding a new workbench tool to npa end to end — implementation, CLI, SDK, toolRef catalog, container, tests, docs, and skill — in the order that keeps every CI gate green.
---

# Add A Workbench Tool

Adding a tool is not one edit. A minimal tool touches ~18 files across six
phases, and ~12 separate gates fail if you skip one. Work the phases in order:
each phase's gate can only pass once the previous phase exists.

`CONTRIBUTING.md` is the prose rationale for this layer and
`docs/architecture/contributor-context.md` the architecture. This skill is the
ordered execution path.

## Decide The Archetype First

The archetype decides which phases apply. Getting this wrong means building a
container nothing routes to, or a GPU stage that lands on the default pod.

| Archetype | Runs in | Phases | Examples |
|---|---|---|---|
| In-process / CPU | default npa workflow pod | A, B, D, E | `insights`, `dataset`, `scenario_gen` |
| Containerized service | own image, deployed to `workbench` namespace | A–E | `detection_training`, `lancedb` |
| Containerized GPU stage | own image, SkyPilot task pod | A–E + image routing | `cosmos3`, `nurec` |

In-process tools need no Dockerfile and no image-routing entry. Do not add a
container "for symmetry" — a container that no `toolRef` routes to is dead
weight the packaging and security gates still police.

## The One Architectural Rule

One implementation, three thin clients. Put behavior in
`npa/src/npa/workbench/<tool_snake>/`; the FastAPI service, the CLI, and the SDK
all call into it. Never implement training, inference, ingest, or status logic
twice.

The cleanest reference to copy is detection-training:

- `npa/src/npa/workbench/detection_training/service.py`
- `npa/src/npa/workbench/detection_training/schemas.py`
- `npa/src/npa/cli/workbench/detection_training.py`
- `npa/src/npa/sdk/workbench/detection_training.py`

LeRobot (`npa/src/npa/cli/workbench/lerobot.py`) is the richest tool but keeps
much of its logic as CLI orchestration. Read it for CLI surface breadth, not as
the layering model to copy.

## Naming

Pick the name once; it derives everything else. `<tool>` is kebab-case,
`<tool_snake>` is snake_case.

| Surface | Form | Example |
|---|---|---|
| CLI group | `npa workbench <tool>` | `npa workbench scenario-gen` |
| Python package | `npa.workbench.<tool_snake>` | `npa.workbench.scenario_gen` |
| SDK module | `npa.sdk.workbench.<tool_snake>` | `npa.sdk.workbench.scenario_gen` |
| toolRef key | `workbench.<tool_snake>.<verb>` | `workbench.scenario_gen.generate` |
| Image name | `npa-<tool>` | `npa-cosmos3` |
| Skill | `skills/tools/<tool>/SKILL.md` | `skills/tools/scenario-gen/SKILL.md` |

The toolRef key is snake_case but the argv inside it is the kebab-case CLI path.
Mixing these up is the most common catalog error.

## Phase A — Implementation And Clients

1. Create `npa/src/npa/workbench/<tool_snake>/` with `__init__.py`,
   `schemas.py` (Pydantic request/response models), and the domain modules. Add
   `service.py` exposing `create_app()` when the tool owns a service.
2. Create `npa/src/npa/cli/workbench/<tool_snake>.py` with a Typer `app`. Use a
   package directory instead when the tool has many verbs
   (`npa/src/npa/cli/workbench/lancedb/cli.py` is the modular reference).
3. Register the CLI group in `npa/src/npa/cli/workbench/__init__.py`:

```python
from npa.cli.workbench.<tool_snake> import app as <tool_snake>_app

app.add_typer(<tool_snake>_app, name="<tool>")
```

4. Create `npa/src/npa/sdk/workbench/<tool_snake>.py` and export it from
   `npa/src/npa/sdk/workbench/__init__.py` (import plus `__all__` entry).
5. Add `<tool_snake>` to the lazy submodule list in
   `npa/src/npa/workbench/__init__.py`.

Register both the CLI and the Python-callable surface. SONIC is a known
deviation that registers only the CLI; do not copy it.

For option naming, output format, error handling, and the config/credentials
accessors, follow `skills/atomic/npa-cli-conventions/SKILL.md`.

## Phase B — Workflow Surface

6. Add one `ToolEntry` per invocable verb to
   `npa/src/npa/orchestration/npa_workflow/catalog.py`. The argv must name real
   CLI flags — see `skills/atomic/toolref-argv-contract/SKILL.md` before writing
   it, because this is where tools most often ship a stage that renders cleanly
   and then dies in the pod.
7. Add a row per toolRef to `docs/workbench/npa-workflow-tool-catalog.md`.
8. Create at least one reference spec under
   `npa/workflows/workbench/npa-workflows/`. Every catalog entry must be
   reachable from a shipped spec unless it is deliberately listed in
   `PUBLIC_REUSABLE_TOOLREFS` in `catalog.py`.
9. GPU or containerized stages only: map the toolRef prefix to its image in
   `npa/src/npa/orchestration/npa_workflow/skypilot_render.py`. Without this the
   stage silently runs on the default image.

## Phase C — Container (containerized archetypes only)

10. Create `npa/docker/workbench/<tool>/Dockerfile` plus a `build.sh` following
    the `--registry` / `--push` shape.
11. Add the image to `npa/docker/workbench/packaging-contract.yaml` and classify
    redistribution with `skills/atomic/solution-licensing/SKILL.md`. Verify the
    claim against the built image, not the Dockerfile.
12. Register the image name and pinned tag in `npa/src/npa/deploy/images.py` and
    `npa/pyproject.toml`. Keep the tag family (`cuda12`, or `cuda13-b300` for
    Blackwell) — do not invent a third; `npa/docker/workbench/tags.yaml` and
    `npa/docker/workbench/check_tag_consistency.py` enforce it.
13. Add a golden eval to `npa/src/npa/smoke/golden_evals.yaml` and its capability
    to `npa/src/npa/smoke/capabilities.py` so the image is provably functional.

## Phase D — Tests

14. `npa/tests/cli/test_<tool_snake>_cli.py` — `CliRunner` against
    `npa.cli.main:app`, covering registration, help, every management verb, and
    every capability verb.
15. `npa/tests/workbench/test_<tool_snake>.py` — the shared implementation and
    each service endpoint, with at least one failure path per endpoint.
16. `npa/tests/workflows/test_<tool_snake>_workflow.py` — spec validation and
    argv construction.
17. Add the tool to `npa/tests/guardrails/test_three_tier_contract.py`, either as
    a full `CapabilityContract` or in the explicit seam set. A new CLI group with
    neither fails immediately.

Mock infrastructure at the call site and keep GPU-heavy imports out of module
scope. Live coverage is not optional for workflow-facing changes — see
`skills/atomic/testing-conventions/SKILL.md`.

## Phase E — Docs And Skill

18. Write `docs/cli/<tool>.md` and add it to `docs/cli/README.md`, then
    regenerate the CLI reference with `bash scripts/build_docs.sh`.
19. Write `skills/tools/<tool>/SKILL.md` and register it in `skills/index.yaml`
    with at least one smoke. Update `AGENTS.md` and `CLAUDE.md` only if the root
    index lists the skill.
20. For a containerized tool, load
    `skills/atomic/audit-container-docs/SKILL.md` and reconcile
    `docs/workbench/container-image-catalog.md` after the image pin and
    redistribution/publication classification are final. Public-plan images
    must have their exact resolved pin in the table; restricted, internal, or
    unpublished images must not be represented as publicly available.

## Phase F — Optional Integrations

Agent UI routing, the daily GPU coverage rotation
(`npa/tests/orchestration/npa_workflow/test_daily_coverage.py` requires a new
workflow image to appear in a workflow of four or more steps), and public
registry publication each have their own gates. Skip them deliberately, not by
omission.

## What Catches Each Omission

Run these before pushing rather than discovering them in CI. The order is
cheapest-first; `skills/atomic/pre-pr-validation/SKILL.md` has the full ladder.

| Skipped step | Gate that fails |
|---|---|
| CLI group not in seam or contracts | `npa/tests/guardrails/test_three_tier_contract.py` |
| Catalog argv names a flag the CLI lacks | `npa/tests/guardrails/test_tool_catalog_argv.py` |
| `python -m` toolRef argv drifts | `npa/tests/guardrails/test_module_toolref_argv.py` |
| toolRef missing from the catalog doc | `npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py` |
| Catalog entry no spec reaches | `npa/tests/guardrails/test_shown_workflow_catalog.py` |
| Spec `outputs:` disagree with write paths | `npa/tests/guardrails/test_spec_declared_outputs.py` |
| Dockerfile not in the packaging contract | `npa/tests/docker/test_packaging_contract.py` |
| Image missing k8s prerequisites | `npa/tests/guardrails/test_workbench_image_k8s_prereqs.py` |
| Skill file absent from the index | `npa/tests/guardrails/test_skills_index.py` |
| Docs reference a spec that does not exist | `npa/tests/guardrails/test_no_dangling_workflow_references.py` |
| CLI options changed, docs not regenerated | `Lint / docs-drift` |
| New workflow image not in a 4+ step workflow | `npa/tests/orchestration/npa_workflow/test_daily_coverage.py` |

To decode a failure message into its fix, use
`skills/atomic/guardrail-failures/SKILL.md`.

## Footguns

- **A stage that renders and then dies.** `validate-spec` and `plan-spec` do not
  check argv against the CLI signature. Only the argv guardrail does.
- **A format word passed to a path option.** `--output json` on an option that
  takes a path makes the stage succeed and write the artifact nowhere.
- **Nesting infrastructure choices.** Image and accelerator belong in
  `resources.<profile>`, not in the stage argv.
- **Inventing a top-level CLI group.** New commands belong under
  `npa workbench`; the top-level utilities in `npa/src/npa/cli/main.py` are
  legacy and marked as such.
- **Copying a legacy layout.** `nurec`, `genesis`, and `groot` predate
  `cli/workbench/`. They work; they are not the pattern for new tools.
