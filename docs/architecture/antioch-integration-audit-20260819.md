# Antioch + Workbench Integration Audit

> Audit date: 2026-08-19. Engineering assessment against the repo as of
> `9a508f68`. Not a commitment, not legal advice, and **not** an approval to
> land a skill: per `docs/architecture/partner-skills-roadmap.md`, a skill lands
> only alongside its implementation and tests.

## Recommendation

Integrate Antioch as a **scenario-evaluation stage inside the Workbench learning
loop**, and integrate it **at the artifact and decision plane first** — not at
the compute plane, and not inside the `npa` Python package.

Three findings drive that:

1. **Antioch already runs on Nebius.** Nebius publishes Antioch as a customer
   story, Antioch published a Nebius partnership post (2026-08-14), and the two
   companies have shared stage time on closed-loop learning with Workbench
   named explicitly. So this is a partner composition question, not a
   cross-cloud bridge question. The compute is already on the right substrate.
2. **But Antioch's control plane is vendor-owned and not addressable through
   Nebius APIs.** Antioch allocates its own warm GPU machines, pushes project
   images into an Antioch-side organization registry, and stores scenario runs,
   artifacts and telemetry in its own store, reachable through the `antioch`
   CLI. SkyPilot cannot schedule Antioch work, and `npa cluster` / `npa fleet`
   cannot see an Antioch machine. Any claim that Workbench "runs" an Antioch
   simulation would be false.
3. **Therefore the only honest and cheap seam is data plus decisions.** The
   `npa.workflow` engine already has exactly one shape for "the expensive work
   happens somewhere we do not schedule": the Token Factory shape — a CPU stage
   that reads S3, calls a hosted service, and normalizes the result back into
   S3 (`docs/workbench/composing-cloud-and-token-factory.md`). Antioch fits
   that shape, with one unresolved prerequisite (non-interactive auth).

The single highest-leverage thing to ask Antioch for is a **documented
server-side HTTP API and a non-interactive service credential**. With those,
the integration becomes Token-Factory-shaped: an `httpx` client in
`npa/src/npa/clients/`, no proprietary wheel, no vendor container, no Python
3.12 island. Without them, every integration tier above "import the evidence"
carries a proprietary dependency that our own packaging guards are built to
reject. See [Open questions](#open-questions-for-antioch) Q1 and Q6.

## What Antioch actually is

Assessed from public sources only: the `antioch-sim` PyPI page (0.3.32 and
0.3.55), the Apache-2.0 `antioch-robotics/antioch-agent-plugin` repo (tag
v0.2.37), `antioch.com`, Nebius customer stories and event pages, and press
coverage from 2026-04 through 2026-08.

| Dimension | What it is |
| --- | --- |
| Product | Cloud simulation platform layered **on top of** Isaac Sim 6.0.1 / Isaac Lab 3.0.0b2 — not a new engine |
| Client | `antioch-sim`: a typed Python SDK + `antioch` CLI. **Proprietary license**, Python `>=3.12,<3.13` only |
| Compute | Vendor-allocated warm GPU machines; no self-hosted or BYO-cloud deployment offered publicly |
| Project model | `antioch.yaml` declares `services.sim` (an engine image coordinate) plus auxiliary Docker services; Docker runs underneath through the Engine API |
| Authoring unit | `@antioch.scenario` functions with declared `cases` (singleton / grid / correlated combinations), grouped into immutable `suites` |
| Outcome model | `run.check(criterion, passed, detail)` — every check recorded; a run failing any check finishes `FAILED`. Plus `run.add_result(name, value)` and `run.add_artifact(path)` |
| Telemetry | A Rerun recording per scenario run (`rerun-sdk==0.36.0`), including a platform-sampled viewport entity at `/antioch/viewport` |
| Retrieval | `antioch scenario show|download|list`, `antioch suite show --follow`; every finite history command supports `--json` with cursor paging and microsecond timestamps |
| Reproducibility | `--queue` freezes a digest-pinned environment and bakes the submitted project tree; `scenario rerun` / `suite rerun` replay it |
| Assets | An organization-scoped immutable versioned asset library: `antioch assets push/pull`, `save_asset()` / `load_asset()` |
| Agent surface | Apache-2.0 plugin for Claude Code and Codex: skills for the Antioch platform, scenario design, Isaac Sim 6, Isaac Lab 3, plus an `antioch-research-mcp` MCP server for versioned doc/source grounding |

Two properties matter more than the feature list.

**The vendor surface is moving fast.** Between 0.3.32 (2026-08-13) and 0.3.55
the engine image coordinate was renamed from `antioch-sim/<engine>:<version>`
to `antioch-engine/<engine>`, and `services.sim` went from required to
optional. That is a breaking manifest change inside one patch series. Our
`test_tool_catalog_argv.py` guardrail deliberately binds catalog argv templates
to real CLI flags, so binding toolRefs to this surface without a stated
deprecation policy means our CI breaks when the vendor ships.

**The agent plugin is a position, not just a convenience.** Antioch ships
agent-facing skills for the same job the NPA agent does. It is Apache-2.0 and
its Isaac content is adapted from NVIDIA's Apache-2.0 Isaac Sim skills with a
`NOTICE` recording the upstream skill list — so copying any of it into
`skills/` would be a licensing action requiring attribution, and would put us
in the business of maintaining a fork of a vendor's docs. Do not vendor it.

## Where Antioch overlaps Workbench, and where it does not

An integration that ignores the overlap will produce two systems claiming the
same job.

| Capability | Workbench today | Antioch | Assessment |
| --- | --- | --- | --- |
| Isaac Sim / Isaac Lab execution | `npa workbench isaac-lab`, the `isaac` backend of the Sim2Real held-out eval (`docs/architecture/sim-backend-selection.md`) | Native, and the whole product | **Direct overlap.** Antioch's interactive loop (warm machines, kernel reuse, streamed viewport) is better for authoring; ours is better for batch stages already wired into a durable workflow |
| Scenario definition + ranking | `npa workbench scenario-gen generate|rank` (RL adversary mining policy failures) | `@antioch.scenario` + cases + suites (human/agent-authored regression sets) | **Complementary.** Ours generates hard scenarios; theirs is the repeatable regression harness. Mined scenarios are a natural input to an Antioch suite |
| Per-run evaluation gate | `npa workbench vlm-eval`, Sim2Real Stage-11 threshold decision | `run.check(...)` named checks; run FAILS on any failed check | **Complementary and directly adaptable.** Their checks are a cheaper, more deterministic gate signal than a VLM score |
| Multi-tool pipeline composition | `npa.workflow/v0.0.1`, durable S3 resume, loops, gates, 14-stage Sim2Real | Suites and queued fan-out inside Antioch only | **Ours.** Antioch has no Cosmos/GR00T/LeRobot/FiftyOne/LanceDB composition story |
| Dataset of record + lineage | `npa workbench dataset`, `npa workbench insights` (lineage graph + metrics store) | Scenario run history, results, artifacts | **Ours.** Antioch is a run store, not a dataset-of-record or a cross-tool lineage graph |
| Provisioning + cost control | `npa cluster`, `npa fleet`, `npa soperator`, preemptible routing, `npa cleanup` | Vendor-managed machine quota, `machine checkout/release` | **Disjoint.** Neither can see the other's resources |
| Visualization | Rerun (`reports/*.rrd`), Foxglove, Lichtblick, agent viewer | Rerun recording + Mission Control webapp + live viewport stream | **Overlap with a version conflict** — see [Rerun](#the-rerun-version-conflict-is-real) |
| Agentic dev experience | NPA agent (hosted chat on a VM, grounded intent routing, artifact viewers) | Claude Code / Codex plugin + `antioch-research-mcp` in the customer's own repo | **Overlap in position, complementary in placement.** Theirs authors sim code in the customer repo; ours operates Nebius workloads |

### The division of labor to design toward

- **Antioch owns the inner authoring loop**: writing Isaac scenarios against a
  warm machine, watching the viewport, iterating on a single scenario, and
  running suites as simulation regression tests.
- **Workbench owns the outer learning loop**: dataset of record, synthetic data
  generation, policy training, the promote/loop-back gate, lineage and metrics,
  publication, and all Nebius provisioning and teardown.
- **The seam is bidirectional and narrow**:
  Workbench artifact (policy checkpoint, USD scene, mesh) → `antioch assets
  push` → Antioch suite run → scenario report + telemetry → Workbench S3 →
  `insights ingest-run` + workflow gate.

That closed loop is credible today and does not require Antioch to change
anything except how we authenticate.

## Integration tiers

`docs/architecture/oss-onboarding-ladder.md` has no rung for a hosted
third-party control plane — its three tiers all assume an OSS repo we
containerize and run on our GPUs. Token Factory sidestepped the question by
being first-party. Antioch is the first case that needs the missing rung, so
this audit names the tiers explicitly.

### H0 — Evidence import (recommended now)

Antioch runs; a thin adapter maps its run record into schemas Workbench already
consumes; Workbench gates, ingests, and visualizes.

```
[ antioch scenario/suite run ]  --download-->  [ adapter ]  --writes-->  [ S3 ]
    vendor GPU machines           JSON + rrd     pure mapping    reports/ + gate/
                                                                       |
                                          insights ingest-run  <-------+
                                          npa.workflow gate    <-------+
                                          agent artifact viewer <------+
```

Why this tier first: **it can be proven with zero changes to `npa`.** The
insights ingester has a structural fallback that recognizes any JSON carrying
`score`, `success_threshold` and `passed` as an evaluation report
(`npa/src/npa/workbench/insights/store.py`), and gate transitions read a
`{"decision": "promote_checkpoint"|"loop_back"}` object written by
`npa/src/npa/orchestration/npa_workflow/decisions.py`. Artifact discovery is
extension-based and never checks whether npa executed the run
(`npa/src/npa/workflows/artifacts.py`). So an adapter that writes those two
files under an S3 run prefix is immediately a first-class Workbench run.

Concrete landing layout (placeholders only — never commit live bucket names):

```
s3://<artifact-bucket>/antioch/<run-id>/
  reports/antioch_scenario_report.json   # score / success_threshold / passed / checks[]
  reports/antioch.rrd                    # vendor recording (see Rerun caveat)
  gate/decision.json                     # {"decision": "promote_checkpoint"|"loop_back"}
  logs/run.log
  results/scenarios.json                 # raw vendor JSON, retained verbatim
```

Constraints this layout must respect:

- `<run-id>` must match `[A-Za-z0-9][A-Za-z0-9._-]*` and must contain at least
  one subdirectory, or `list_runs` will not treat it as a run.
- Keep the raw vendor JSON verbatim alongside the mapping. When the vendor
  surface moves, the raw record is what lets us re-map without re-running GPU
  work.
- Follow the repo's split: the mapping is pure and unit-tested (the
  `npa/src/npa/workflows/token_factory_combos.py` convention); all I/O lives in
  the runner.

Optional small follow-up once the shape is proven: register a real schema id in
`REPORT_PROFILES` (`npa/src/npa/workbench/insights/store.py`) so Antioch runs
are attributed to a named tool and stage rather than riding the structural
`vlm_eval` fallback, and extract `failed_check_count` the way
`_extract_validation_report` already does for dataset validation.

### H1 — Workflow stage (recommended next, gated)

A `workbench.antioch.*` toolRef so an Antioch suite is a state inside an
`npa.workflow` spec, with the gate and downstream training stages reacting to
its result automatically.

Shape, following Token Factory precisely: a CPU stage on our Kubernetes
(`resources:` with no `accelerators`), reading and writing `s3://` through the
`--input-path` / `--output-path` contract, with the vendor credential injected
as a submit-time secret via `SECRET_ENV_HINTS` in
`npa/src/npa/orchestration/npa_workflow/skypilot_render.py` and a fail-fast
check in the rendered setup script.

**H1 is blocked on non-interactive authentication.** `antioch auth login` is a
browser device-code flow writing a session under `~/.config/antioch`. A
SkyPilot pod cannot complete that, and mounting a human's session into a pod is
not an acceptable credential design. Do not start H1 until Q1 is answered.

If the answer to Q6 is "there is an HTTP API", H1 is a
`npa/src/npa/clients/antioch.py` + `npa/src/npa/workbench/antioch/` +
`npa/src/npa/cli/workbench/antioch.py` build, with no container work at all —
the cheapest possible version. If the answer is "the CLI is the only client",
H1 needs a dedicated Python 3.12 container that runtime-fetches the proprietary
wheel under operator credentials, plus a `packaging-contract.yaml` entry, a
`REDISTRIBUTION.md`, and exclusion from public publication. That is a
materially larger and licence-encumbered build for the same functionality.

Full file-by-file checklist if H1 proceeds: `skills/workflows/add-workbench-tool/SKILL.md`.
The gates that will fail if a step is skipped are enumerated there and in
`skills/atomic/guardrail-failures/SKILL.md`; the ones most likely to bite this
integration are `test_tool_catalog_argv.py` (argv bound to real vendor flags),
`test_spec_declared_outputs.py` (declared outputs must match the real write
path), and `test_packaging_contract.py` (any vendor container must be
classified).

### H2 — Agent-native experience (do not start yet)

The "integrated experience" a user would imagine: ask the NPA agent to design a
scenario, have it grounded in versioned Isaac documentation, run it, and
explain the failures from recorded evidence.

Every piece of this is a new subsystem, not an extension:

- **The NPA agent has no MCP client.** There is no MCP server or client
  implementation anywhere in `npa/`. Routing is a grounded regex intent matcher
  plus a bounded tool allowlist (`npa/src/npa/cli/agent_chat.py`,
  `npa/src/npa/agent_backend/actions.py`). Consuming `antioch-research-mcp`
  means building MCP client support, extending the allowlist, and bridging auth
  — and the server itself ships inside the proprietary wheel.
- **The research capability partially duplicates ours.** The agent already
  indexes docs and skills for retrieval (`npa/src/npa/agent_backend/retrieval.py`).
- **The authoring surface is the customer's own repo**, which is where Antioch's
  plugin already runs well. Pointing users at the vendor plugin for authoring,
  while the NPA agent owns Nebius orchestration and evidence, is a better
  product answer than reimplementing it.

Revisit H2 only if MCP client support is being built for independent reasons.

## Hard constraints any integration must respect

These are properties of this repo, verified in code, not preferences.

| Constraint | Evidence | Consequence |
| --- | --- | --- |
| `antioch-sim` is **proprietary** | PyPI metadata: `License: Proprietary` | Cannot be baked into a `public` image. Baking it forces `redistribution: restricted` in `npa/docker/workbench/packaging-contract.yaml`, plus a `REDISTRIBUTION.md` and removal from `publicly_publishable_tools()` in `npa/src/npa/deploy/images.py`. Runtime fetch under operator credentials is the compliant pattern (`skills/atomic/solution-licensing/SKILL.md`) |
| Python `>=3.12,<3.13` vs `npa` `requires-python = ">=3.10"` (ruff `target-version = "py310"`) | `npa/pyproject.toml:9`, `:230` | `antioch-sim` cannot join the `npa` dependency closure. It must live in a separate venv, container, or subprocess. Never import it at module scope |
| `rerun-sdk==0.36.0` vs `npa`'s pinned `rerun-sdk==0.31.4` | `npa/pyproject.toml:60`; `RERUN_VERSION = "0.31.4"` in `npa/src/npa/cli/rerun/__init__.py`; `DEFAULT_RERUN_SERVE_SDK_VERSION = "0.32.0"` in `npa/src/npa/workflows/rerun_serve.py` | See below — this is the one user-visible breakage |
| Device-code auth into `~/.config/antioch` | vendor README | Outside our credential model (`npa/src/npa/clients/credentials.py`). Do not persist a vendor session in `~/.npa/credentials.yaml`; do not reuse `ACCEPT_EULA`, which is Isaac-scoped (`skills/atomic/third-party-eula-preflight/SKILL.md`). A scoped, run-only secret is the correct pattern — the `NPA_OPENPI_ACCEPT_GEMMA_TERMS` precedent in `npa/src/npa/workflows/byof/openpi.py` |
| Artifact paths are `s3://`-only in the public CLI | `npa/src/npa/cli/path_contract.py` | Vendor output must be copied into a Nebius bucket we control before any Workbench stage reads it, unless Q3 resolves to a shared bucket |
| No hardcoded infra, and a confidentiality guard on committed text | `npa/src/npa/guardrails/confidentiality.py` | This doc and any follow-up use placeholders. Live project/bucket/registry identifiers stay in access-controlled evidence (`skills/atomic/protect-nebius-infra-details/SKILL.md`) |

### The Rerun version conflict is real

Antioch records at `rerun-sdk==0.36.0`. Workbench writes recordings at
`0.31.4`, `npa rerun host` advertises `0.31.4` to `app.rerun.io`, the
`npa-rerun-viewer` container is `0.31.4`, and in-cluster serve pods pin
`0.32.0`. Rerun requires reasonably aligned writer and viewer versions, so an
Antioch `.rrd` should be expected **not** to render in the NPA agent today.
Three options, in increasing cost:

1. **Do not promise vendor `.rrd` playback.** Link out to Antioch's scenario-run
   page for telemetry and keep Workbench's viewers for Workbench artifacts.
   Correct, cheap, and slightly disappointing.
2. **Re-log in the adapter.** The vendor documents
   `rerun.experimental.RrdReader(...)` with `blueprints()` and `stream(store=...)`
   for reading a downloaded recording, so an isolated 3.12/0.36 environment can
   extract scalars and viewport frames and the adapter can re-emit them as a
   `0.31.4` recording under `reports/`. Costs an extra hop, but yields one
   consistent viewer story.
3. **Move npa's Rerun pin.** Blast radius: `npa/pyproject.toml`, the `npa rerun`
   CLI constant, the viewer container, `rerun_serve.py`, agent VM bootstrap, and
   every existing recording produced at the old version. Worth doing on its own
   merits eventually; not worth doing *for* this integration.

Recommend option 1 for H0 and option 2 if H1 lands.

## What to build first

Ordered, each step independently useful, each with a stop condition.

1. **Answer Q1 and Q6 with Antioch.** Everything above H0 depends on them. If
   Q6 is "yes, there is an HTTP API", most of this audit's cost estimates
   collapse.
2. **Write the pure adapter and prove H0 on one real suite run.** A mapping from
   a vendor scenario-run JSON record to `reports/antioch_scenario_report.json` +
   `gate/decision.json`, unit-tested against a captured fixture, with no vendor
   import. Stop condition: `npa workbench insights ingest-run` ingests the run
   and the agent lists it as a run with viewable artifacts.
3. **Wire one npa.workflow that gates on it.** An existing loop spec whose
   `quality-gate` state reads the adapter's decision, proving Antioch checks can
   promote or loop back a Workbench training run. Stop condition: `validate-spec`
   and `plan-spec` pass and a live submit reaches the gate.
4. **Then, and only then, decide on H1** using the Q6 answer to choose between
   the client build and the container build.
5. **Do not add `skills/tools/antioch/SKILL.md` or a `skills/index.yaml` entry
   until step 3 is green with tests.** A skill that describes a capability we do
   not have routes agents at flows that do not exist — the explicit rule in
   `docs/architecture/partner-skills-roadmap.md`.

## Open questions for Antioch

Ordered by how much they change the design.

1. **Non-interactive authentication.** Is there a service account, API key, or
   OIDC client-credentials path that a CI job or a Kubernetes pod can use
   without a browser? *Blocks H1 entirely.*
2. **Account topology.** Does an Antioch organization map onto the customer's
   own Nebius project or tenant, so GPU spend, data residency, and egress stay
   inside the customer's account and appear on one bill? Or is compute in
   Antioch's account? *Determines whether "integrated" can mean one project and
   one bill, or only one UI.*
3. **Artifact storage location.** Is the Antioch artifact store a Nebius bucket
   that a customer can point at their own bucket? *If yes, the copy step in H0
   disappears and Workbench reads Antioch output in place — by far the largest
   single simplification available.*
4. **Contract stability.** Is there a supported machine-readable run contract
   with a deprecation policy? The `antioch-sim/<engine>:<version>` →
   `antioch-engine/<engine>` rename and `services.sim` becoming optional
   happened inside one patch series. *Determines whether we can bind toolRef
   argv to the CLI at all.*
5. **Rerun alignment.** Will recordings track a version we can read, or can
   scalars and frames be exported as plain JSON/Parquet alongside the `.rrd`?
   *Decides which Rerun option above we take.*
6. **Server-side API.** Is there a documented HTTP API, or is the proprietary
   wheel the only client? *Converts H1 from a licence-encumbered vendor
   container into a Token-Factory-shaped client.*
7. **Redistribution terms** for a Workbench-side integration image, if one is
   needed, and whether the engine images may be pulled by anything other than
   the Antioch platform.
8. **Positioning.** How do we present Antioch's agent plugin and the NPA agent
   together, so a customer is not asked to choose between two agentic sim
   experiences?

## Sources

Public only, dated, and re-verifiable:

- `antioch-sim` on PyPI — versions 0.3.32 (2026-08-13) and 0.3.55, including the
  full SDK/CLI README: license, Python requirement, dependency pins, scenario
  and suite model, `--queue` semantics, asset library, engine image coordinates.
- `antioch-robotics/antioch-agent-plugin` at tag v0.2.37 (Apache-2.0):
  `README.md`, `NOTICE`, `.mcp.json`, `.claude-plugin/plugin.json`,
  `.codex-plugin/plugin.json`, and the shipped skill tree.
- `antioch.com`; Antioch's Nebius partnership post (2026-08-14); Nebius customer
  stories; Nebius Actuate 2026 and MACHINA 2026 event pages.
- TechCrunch (2026-04-16), The Robot Report, and Robotics & Automation News
  (2026-05-03) on funding, sensor-modality coverage, and the platform-layer
  positioning.

Supersedes the `Antioch | Insufficient credible public evidence in this pass`
row in `docs/architecture/workflow-engine-recommendation-20260514T233740Z.md`.
