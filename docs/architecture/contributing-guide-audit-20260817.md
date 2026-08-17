# Contributor Guide Audit — 2026-08-17

Audit of `CONTRIBUTING.md` against repo state at `9c84d4a0`. Every claim below was
checked against code, CI config, or a real command run in this checkout; the
verification command is named so a reviewer can re-run it.

`CONTRIBUTING.md` was last edited on 2026-08-03 (`b70779b4`). 41 commits have
landed since, including `npa fleet`, `npa soperator`, the Foxglove embed, the
compositional 14-stage Sim2Real workflow, the Wan 2.2 BYOF solution, and the
public container catalog.

## Verdict

The guide is not broken — all 91 file paths it cites still resolve, and its S3
data-bus, secrets, tag-family, and headless-training rules are all still correct.
The problem is narrower and worse: **its central architectural claim now
contradicts a guardrail that fails the PR gate**, and its "Known Deviations"
appendix has decayed into a mix of resolved items and understated ones.

Findings are grouped by what happens to a contributor who follows the guide
literally.

## P0 — Following the guide fails CI

### 1. Adding a workbench tool breaks the test suite, and the guide never says so

`npa/tests/guardrails/test_three_tier_contract.py::test_new_workbench_tools_require_contract_or_explicit_seam`
asserts that the set of registered `npa workbench` sub-apps equals the set of
declared `CapabilityContract`s plus an explicit `seam` allowlist. Registering a
new tool in `npa/src/npa/cli/workbench/__init__.py` — the exact step the guide's
"Registration Steps" section instructs — fails that assertion until the
contributor also adds a contract or a seam entry with a stated reason.

This is the single most consequential omission: it is a hard failure on the very
task the document exists to describe.

### 2. The three-access pattern the guide teaches is not the one that is enforced

The guide's "Workbench Tool Architecture" section, its mermaid diagram, and
`docs/architecture/contributor-context.md` all define the three access modes as
**HTTP API (source of truth) + CLI + SDK**.

The enforced contract is **CLI + SDK + `npa.workflow` spec/`toolRef`**. From the
guardrail's own module docstring:

> The third tier used to be a raw SkyPilot task YAML [...] As that catalog is
> retired, the third tier moves onto the surface that survives: the shipped
> `npa.workflow` spec plus the `toolRef` argv template the engine expands.

The code agrees with the guardrail, not the guide. Only six modules construct a
`FastAPI(` app under `npa/src/npa/workbench/` (`dataset`, `detection_training`,
`insights`, `lancedb`, `scenario_gen`, `lerobot/policy_container.py`), against 29
registered tool namespaces. An HTTP service is now the exception.

To be clear about what is *not* wrong here: the guide's required minimum endpoint
set is accurate and is being followed. The three services added since the guide
was written — `dataset`, `insights`, `scenario_gen` — each expose exactly
`/health`, `/system-info`, `/list`, and `/status` plus capability verbs. The
error is the framing that every tool must have a service and that the service is
the source of truth, not the endpoint list itself.

### 3. Four blocking CI gates are missing from "Configuration And Secrets"

The guide lists `test.yml`, `image-security-scan.yml`, and `gitleaks.yml`. The
repository runs twelve workflows. These four also block a PR and are unmentioned:

| Workflow | Gate | Why a contributor hits it |
| --- | --- | --- |
| `lint.yml` (`ruff` job) | `ruff check src tests` | Any Python change |
| `lint.yml` (`docs-drift` job) | `scripts/build_docs.sh --check` | Any CLI flag or command change — `docs/cli/` must be regenerated and committed |
| `harness-guardrails.yml` | `pytest npa/tests/guardrails -q` | Any tool, spec, skill, or catalog change |
| `confidentiality-scan.yml` | `npa.guardrails.confidentiality` over the diff and tree | Any change touching infra-shaped strings |

The `docs-drift` job is the easiest to trip and the least discoverable: nothing
in the guide mentions that `docs/cli/` is generated, let alone that it is checked.

### 4. The stated PR command is not the PR gate

The guide says a PR should pass `make test`. `test.yml` runs something
materially different:

```
cd npa && python -m pytest tests/ -v --tb=short --cov=npa --cov-report=term-missing --cov-fail-under=60
```

Differences that matter: CI enforces a **60% coverage floor** (unmentioned), does
**not** deselect the live/GPU markers, installs `.[dev,adapter]` rather than
`[dev]`, additionally runs `tests/integration/test_cli_install.sh` and
`scripts/check-source-drift.sh`, and runs a Python matrix of 3.10/3.12/3.14 on
`main`. The guide never states the supported Python range at all
(`requires-python = ">=3.10"`).

### 5. `make test` is not hermetic, and the venv location is load-bearing

Running `make test PYTHON=/tmp/npa-venv/bin/python` on a clean checkout of this
commit produced **9 failed, 10358 passed, 39 skipped, 1 xpassed** in 617s. Both
failure classes are environmental, and both contradict the guide:

- Six `tests/test_provisioning.py` failures are
  `BadParameter: Required executable not found: kubectl`. The guide describes the
  default suite as needing no live infrastructure; it needs `kubectl` on `PATH`.
- `tests/smoke/test_golden_eval_tmux.py` fails with `ModuleNotFoundError: No
  module named 'npa'` because `npa/scripts/start_golden_evals_tmux.sh` resolves
  the interpreter from `npa/.venv`, not from `PYTHON`.

That second one is a real convention the guide gets wrong. `AGENTS.md`,
`docs/workbench/contributing-a-containerized-solution.md`,
`harness-guardrails.yml`, and `confidentiality-scan.yml` all standardize on
`npa/.venv`. `docs/quickstart.md` uses a root `.venv`. `CONTRIBUTING.md` says only
"use your venv's interpreter" and advertises `make test PYTHON=…` as a general
escape hatch. `npa/tests/guardrails/test_docs_green_path.py` exists precisely
because this disagreement previously ran `npa` from the wrong interpreter.

## P1 — Stale facts

### 6. The tool inventory is roughly a quarter of reality

The guide is built around 8 validated tools plus detection training as an
unlisted extra. `registered_workbench_tools()` returns **29**:

```
byof, cosmos, cosmos-curate, cosmos-evaluator, cosmos2, cosmos3, data, dataset,
detection-training, fiftyone, foxglove, genesis, golden-eval, groot, health,
insights, isaac-lab, lancedb, lerobot, lichtblick, mjlab, nurec, scenario-gen,
sim2real, sim2real-envgen, sonic, token-factory, vlm-eval, workflow
```

Same drift elsewhere: 59 specs under `npa/workflows/workbench/npa-workflows/`,
39 Dockerfiles under `npa/docker/workbench/`, and 90 `TOOL_CATALOG` entries. The
8-tool claim also needs correcting at its two sources,
`skills/atomic/architecture/SKILL.md:19` and
`docs/architecture/contributor-context.md:9`.

### 7. Every per-tool command list in the CLI section is incomplete

Verified by running `npa workbench <tool> --help` for each:

| Tool | Guide omits |
| --- | --- |
| LeRobot | `benchmark`, `profile-train`, `train-student` |
| FiftyOne | `curate-augmented`, `datasets`, `restart`, `ensure-ingress`, `register-byovm`, `cleanup-partial` |
| Genesis | `simulate`, `eval-student` |
| Isaac Lab | `export-onnx`, `list-tasks` |
| Cosmos | `check`, `fetch`, `teardown`, `autoscale`, `reload-env` |
| GR00T | `reload-env` |
| LanceDB | `create-table`, `query`, `import-lerobot`, `refresh-mv` |
| SONIC | `export`, `eval`, `retargeting` |

### 8. Two "Known Deviations" are resolved and should be deleted

- *"SONIC is not registered in the workbench Python namespace"* — false since
  `npa/src/npa/workbench/sonic/` was added; `sonic` is in
  `npa/src/npa/workbench/__init__.py:20`.
- *"SDK namespace coverage is mixed [...] exports only detection training and
  LanceDB"* — `npa/src/npa/sdk/workbench/__init__.py` now exports 23 modules.

A different, real gap has replaced the second one and should be recorded in its
place: six tools with CLI surfaces are absent from
`npa/src/npa/workbench/__init__.py` (`cosmos_curate`, `cosmos_evaluator`,
`foxglove`, `lichtblick`, `nurec`, `token_factory`).

### 9. Two more "Known Deviations" understate the problem

- *"LanceDB and SONIC do not expose CLI `system-info`"* — 19 of 29 tools do not.
  Only `lerobot`, `fiftyone`, `genesis`, `isaac-lab`, `cosmos`, `groot`,
  `dataset`, `insights`, `scenario-gen`, and `detection-training` have it. Either
  the guide should stop presenting `system-info` as near-universal, or the
  requirement needs a guardrail — as written it is advice that most of the
  codebase ignores.
- *"Top-level CLI registrations mix namespaces and platform utilities"* — the
  named list (`adapter`, `cluster`, `convert`, `demo`, `network`, `rerun`,
  `skypilot`, `viz`) is missing `agent`, `burst`, `fleet`, `provision-if-absent`,
  `registry`, `storage`, `soperator`, `cleanup`, `uninstall`, and `destroy`.

### 10. The SONIC GPU-routing conflict is described with the wrong resolution

The guide records a code-vs-skill conflict between H100 (skills) and L40S (CLI
default). `skills/tools/sonic/SKILL.md:54` now says the opposite of both: render
validation requires RTX PRO 6000 Blackwell, and the L40S/H100/H200 images are
**quarantined**. The CLI still defaults `--gpu-type` to `l40s`
(`npa/src/npa/cli/workbench/sonic/train.py:364`) while its own help text warns
that the active variant requires RTX PRO 6000. The conflict is real but has moved.

### 11. The path contract is cited as a rule that new code does not follow

The guide says to validate `--input-path`/`--output-path` with
`npa/src/npa/cli/path_contract.py`. Five CLI modules import it — `groot`,
`fiftyone`, `cosmos`, `lerobot`, `isaac_lab`, all pre-dating the guide. None of
the newer tools do. Separately, eight modules use `--input-uri`/`--output-uri`
instead of the flags the guide calls "the public cross-tool handoff flags";
`skills/tools/workbench-tool/SKILL.md:41` already documents this exception and
the guide does not. This needs a decision recorded either way, not silence.

## P2 — Subsystems with no contributor guidance

Each of these is a place a contributor can now land a change with nothing in
`CONTRIBUTING.md` to guide them.

- **The OSS onboarding ladder.** `docs/architecture/oss-onboarding-ladder.md`
  defines Tier 0 (BYOF container) → Tier 1 (solution workflow) → Tier 2
  (first-class tool). The guide only describes Tier 2, so it reads as though
  every contribution must be a full tool. The ladder, not the tool pattern, is
  the current front door.
- **`toolRef` catalog registration.** Adding a workflow-callable capability means
  editing `npa/src/npa/orchestration/npa_workflow/catalog.py` and syncing
  `docs/workbench/npa-workflow-tool-catalog.md`, enforced by
  `npa/tests/orchestration/npa_workflow/test_catalog_doc_sync.py`. Unmentioned.
- **Licensing and redistribution.** `npa/docker/workbench/packaging-contract.yaml`
  classifies every image `public` or `restricted` and is enforced by
  `npa/tests/docker/test_packaging_contract.py`. The guide's Containerization
  section covers base images and tags and never mentions the contract file,
  `docs/workbench/container-packaging.md`, or
  `skills/atomic/solution-licensing/SKILL.md`.
- **`npa fleet` and `npa soperator`.** Two spec-driven infra subsystems
  (`npa.fleet/v0.0.1`, `npa.soperator/v0.0.1`) with SDK modules, CLI surfaces,
  and skills. Outside the guide's stated tool-layer scope, but the scope note
  should say so explicitly rather than leaving them undiscoverable.
- **The NPA agent.** ~45 `npa/src/npa/cli/agent_*.py` modules plus
  `agent_ui.html` and vendored Foxglove assets. Same treatment needed.
- **Sim2Real engine.** The canonical 14-stage graph, stateless adapters, and
  durable resume documented in `skills/workbench/sim2real-engine/SKILL.md`.

## P3 — Promises the guide does not keep

`README.md:1059` tells contributors to read `CONTRIBUTING.md` for "the review
checklist, skill-maintenance requirements, and repo hygiene rules". The guide has
none of the three as named sections.

The skill-maintenance gap is concrete and checkable.
`.github/PULL_REQUEST_TEMPLATE.md` requires a `skills/index.yaml` entry and a
matching smoke in `npa/tests/guardrails/test_skills_index.py`; the guide's "Agent
Skill Files" section says only to add a file under `skills/tools/` and to touch
`AGENTS.md` when the root index changes. `test_skills_index.py` verifies both
that every skill is covered by the index and that each declared smoke actually
runs, so a contributor following the guide alone fails `harness-guardrails.yml`.

## Still accurate — do not churn

Re-verified as correct, to bound the edit:

- All 91 cited file paths resolve.
- The S3-only composition contract and `s3://` handoff shapes.
- `storage.eu-north1.nebius.cloud` as primary, with `uk-south1` as the wrong
  legacy default (`npa/src/npa/clients/credentials.py:50-56`).
- Placeholder and environment-variable conventions, and the
  no-committed-secrets rule.
- The `cuda12` / `cuda13-b300` two-family strategy and the instruction not to
  invent a third (`npa/docker/workbench/tags.yaml`).
- The `npa.workflow/v0.0.1` authoring rules, the retirement of the raw SkyPilot
  catalog, and the headless-training requirement.
- `NPA_SKYPILOT_BIN` resolution rather than `sky` from `PATH`.
- The required minimum HTTP endpoint set, for tools that do have a service.
- The `/tmp/npa-commit-lock/` and three-commit-review conventions
  (`skills/atomic/super-prompt-patterns/SKILL.md:31,41`).

## Suggested edit plan

Ordered by contributor impact, not by document order.

1. Rewrite "Workbench Tool Architecture" and "Required Interfaces" around the
   enforced CLI + SDK + spec/`toolRef` contract, keeping the service pattern as
   the Tier 2 option it now is. Add the `CapabilityContract`-or-seam step to
   "Registration Steps".
2. Add an "OSS onboarding ladder" section ahead of the tool pattern so Tier 0 and
   Tier 1 contributions have a home.
3. Replace the CI list in "Configuration And Secrets" with all blocking gates,
   and add `scripts/build_docs.sh` regeneration to "Testing Requirements".
4. Correct "Testing Requirements": real baseline (10358 passed / 39 skipped /
   1 xpassed, ~10 min), the CI command and its 60% coverage floor, the supported
   Python range, the `kubectl` prerequisite, and `npa/.venv` as the convention.
5. Refresh the tool inventory and per-tool command lists; fix the same 8-tool
   claim in `skills/atomic/architecture/SKILL.md` and
   `docs/architecture/contributor-context.md`.
6. Rebuild "Known Deviations": delete items 8, restate items 9 and 10 with
   current numbers, add the six tools missing from `npa.workbench.__init__`.
7. Add short sections for the packaging contract, `toolRef` catalog sync, and the
   `skills/index.yaml` + smoke requirement; add a scope note pointing at fleet,
   soperator, the agent, and the Sim2Real engine.
8. Decide and record the `--input-path` vs `--input-uri` and `path_contract.py`
   question rather than leaving code and guide in disagreement.

`docs/architecture/contributor-context.md` carries a header calling itself
authoritative for rationale and marking concrete claims as hypotheses to verify.
Several of those hypotheses are now false. It should either be re-dated with the
stale claims corrected or explicitly marked historical, so it stops being cited
as current architecture.
