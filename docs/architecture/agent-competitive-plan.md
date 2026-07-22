# Agent Competitive Plan — grounded assistant → Physical-AI agent

Status: living design doc for the workstream that evolves the NPA workbench chat
agent from a grounded assistant into an agent that can *drive* Physical-AI
workflows. Companion skills: `skills/atomic/agent-development/SKILL.md`,
`skills/tools/npa-agent/SKILL.md`, `skills/workbench/sim2real-engine/SKILL.md`.

## Non-negotiable invariants

1. **Grounded-first stays the default.** The zero-token path
   (`match_chat_intent` → `build_grounded_reply` in `agent_chat.py`) answers
   high-frequency turns with `"grounded": true` and no model call. Everything
   added here is a *fallthrough*, reached only after grounded intent misses.
2. **Cost discipline.** New model calls default to the cheap tier
   (`agent_routing.classify_tier`/`build_model_ladder`); escalation is
   deliberate. Structured calls stay small.
3. **Confirmation gates.** Any GPU-spending or destructive action requires an
   explicit confirmation-gate token in the request. A model turn can *propose*
   such an action but never auto-executes it.
4. **Embedded-backend mechanism.** Testable logic lives in real modules under
   `npa/src/npa/cli/` and is embedded into the VM `backend.py` via
   `_embedded_agent_<name>_source()` + `_AGENT_<NAME>_EMBED` placeholder in
   `_bootstrap_agent_stack` (`agent.py`). No new logic inside the f-string.
   Every template edit is followed by the rendered-backend compile check.
5. **No fabricated run data, no hardcoded infra/secrets.** Stages are only
   marked complete when the real status/runs APIs confirm them; no stock Franka
   `.rrd` masquerading as run data.

## Phase 0 — current agent surface (state check)

### `_INTENT_RULES` intents (`agent_chat.py`)

`start_sim2real`, `watch_sim`, `load_franka`, `create_vlm_rl_workflow`,
`create_gate_workflow`, `create_loop_gate_workflow`, `create_rl_policy_workflow`,
`create_workflow`, `workflow_execute_guidance`, `find_artifacts`,
`onboard_solution`, `list_recordings`, `sim2real_status`, `sim_assets`,
`cameras`, `infra_backends`, `mk8s_provision`, `live_infra_loop`,
`cosmos_capabilities`, `lancedb_capabilities`, `sonic_capabilities`,
`lerobot_capabilities`, `groot_capabilities`, `genesis_capabilities`,
`mjlab_capabilities`, `isaac_lab_capabilities`, `component_capabilities`,
`tools_catalog`, `configure_s3`, `cosmos3`, `soperator`.

`build_grounded_reply` has a dedicated branch per intent (with the multi-line
`watch_sim` block being the largest), falling back to `format_sim2real_status`.
`match_chat_intent` also has pre-`_INTENT_RULES` special cases: soperator,
non-stock artifact discovery, success-gated `watch_sim`, and a franka↔watch
precedence rule. The `watch_sim` regex is the ~40-alternation "monster" Phase D
targets.

### Cost tiers/ladders (`agent_routing.py`)

Tiers: `cheap` (default), `standard` (long/compound), `reasoning` (analytical),
`vision` (image content). `TIER_MODELS` orders concrete models per tier
(cheapest-capable first); `build_model_ladder` applies user override → tier prefs
→ configured; `flavor_variants`/`filter_available` handle `-fast`;
`chat_extra`/`thinking_enabled` disable billed traces off the reasoning tier;
`enforce_input_budget` caps pastes; `usage_summary` surfaces token usage.

### Embedded modules wired in `_bootstrap_agent_stack`

`_AGENT_ROUTING_EMBED` (agent_routing), `_AGENT_VISUAL_FEEDBACK_EMBED`
(agent_visual_feedback), `_AGENT_RRD_PROXY_EMBED` (agent_rrd_proxy),
`_AGENT_STAGES_EMBED` (agent_stages), `_AGENT_CHAT_EMBED` (agent_chat),
`_AGENT_WORKFLOW_EMBED` (agent_workflow), `_AGENT_ARTIFACTS_EMBED`
(workflows/artifacts), plus `_AGENT_UI_HTML__`.

### `/api/*` routes (read-only vs state-changing)

- **Read-only:** `GET /health`, `/models`, `/session`, `/chat/sessions`,
  `/chat/sessions/{id}`, `/tools`, `/tools/{ref}`, `/sim-viz/status`,
  `/sim-viz/runs`, `/sim-viz/recordings`, `/sim-viz/rrd`, `/sim-viz/rrd-blob`,
  `/artifacts/runs`, `/artifacts/run/{id}`, `/sim-assets`, `/sim-assets/catalog`,
  `/sim-assets/cameras`, `/sim-assets/selection`, `/workflows/sim2real/status`,
  `/workflows/sim2real/runs/{id}`, `/workbench/actions`, `/workflows/draft`,
  `/infra/k8s`, `/infra/backends`, `/infra/mk8s`, `/infra/soperator/status/{name}`.
- **State-changing / model-spending:** `POST /chat`, `/chat/sessions`,
  `/chat/sessions/{id}/select`, `/sim-viz/select-run`, `/sim-viz/load-run`,
  `/sim-viz/load-artifact`, `/sim-viz/load-franka-demo`, `/sim-viz/camera-preview`,
  `PUT /sim-assets/cameras/selection`, `POST /sim-assets/selection`,
  `/infra/provision`, `/infra/k8s/provision`, `/infra/mk8s/provision`,
  `/infra/soperator/validate`, `/infra/soperator/deploy`, `/workflows/draft`,
  `/workflows/validate`, `/workflows/plan`, `/workflows/submit`,
  **`/workflows/sim2real/submit`** (GPU/destructive — the one action that spends).

Baseline suite (146 tests) green before any change.

## Target architecture

New real modules under `npa/src/npa/cli/` (embedded like the existing ones):

| Module | Phase | Role |
| --- | --- | --- |
| `agent_actions.py` | B | Tool allowlist + confirmation-gate contract + bounded classify→plan→call→observe→decide loop. |
| `agent_sim2real_loop.py` | C | Autonomous Sim2Real outer-loop orchestration built on the action loop + engine/status APIs. |
| `agent_semantic_router.py` | D | `classify_intent_semantic()` fallthrough behind the grounded regex layer. |
| `agent_memory.py` | F | Persistent per-agent run/experiment memory (storage-backed, no hardcoded bucket). |
| `agent_visual_feedback.py` (extend) | F | Quantitative viewer signals (success-rate, collapse detection, run-to-run compare). |

Test-only: `npa/tests/agent_eval/` (Phase E) — task-completion harness +
scorecard, not unit tests.

### Dependency-injection contract (why this is testable at 0 tokens)

Every module that would spend tokens or touch infra takes its side-effecting
collaborators as **injected callables**:

- `model_call(messages, *, tier, ...) -> dict` — wraps `_chat_with_resilience`
  on the VM; a deterministic fake in tests.
- `tools: Mapping[str, ToolHandler]` — the allowlisted tool implementations;
  in the backend these wrap existing route handlers, in tests they are fakes.
- `store` (Phase F) — an object with `read`/`write`; a real S3/JSON-backed store
  on the VM, an in-memory fake in tests.

This keeps the pure logic in the real module and lets Tier-0/1 tests assert
behavior with zero network/model calls.

## Phase B — bounded agentic tool-calling loop

`agent_actions.py`:

- `TOOL_ALLOWLIST: dict[str, ToolSpec]` where `ToolSpec` records
  `read_only: bool`, `requires_confirmation: bool`, `summary`, and `params`.
  Read-only tools: `workflow_validate_spec`, `workflow_plan_spec`,
  `sim_viz_status`, `artifacts_runs`, `artifacts_run`, `health`. Confirmation
  tools: `sim2real_submit` (and any future GPU/destructive tool).
- `run_action_loop(goal, *, tools, model_call, confirm_token="", max_steps=6,
  classify_tier=...)` → returns `{ok, reply, steps:[...], tools_used:[...],
  stopped_reason, needs_confirmation, proposed_action}`.
- Loop: **classify** the goal (cheap tier by default; escalate only when
  `classify_tier` says so) → **plan** a next tool call via `model_call` returning
  a structured `{tool, args, done, final}` → **enforce allowlist** (unknown tool
  → error step, no execution) → **enforce confirmation gate** (a tool with
  `requires_confirmation` and no valid `confirm_token` stops the loop with
  `needs_confirmation`, returning `proposed_action`) → **call tool** and record
  observation → **decide** next / stop → **hard max-steps guard**.
- Wired via `POST /api/agent/act`; `/api/chat` stays grounded-first (the loop is
  only entered when explicitly requested or via Phase D fallthrough).

Confirmation-gate contract: the request carries `confirm_token`; the backend
issues/echoes a per-session token, and only a matching token unlocks
`requires_confirmation` tools. A model turn can populate `proposed_action` but
the operator must resubmit with the token to execute.

Rollback: `agent_actions.py` is additive; remove the `_AGENT_ACTIONS_EMBED`
wiring and the `/api/agent/act` route to revert. `/api/chat` is untouched.

## Phase C — autonomous Sim2Real outer loop

`agent_sim2real_loop.py`:

- `drive_sim2real_loop(goal, *, launch, run_eval, read_gate, diagnose, adjust,
  status, confirm_token, max_iterations)` composes: launch sim → run eval →
  read gate metrics → diagnose failure mode → adjust config → re-run.
- Every GPU-spending step routes through the Phase-B confirmation gate.
- Each iteration surfaces a grounded decision (`promote_checkpoint` /
  `loop_back`) with the reason, mirroring engine `threshold_decision`.
- A stage is only marked complete when the real
  `workflows/sim2real/status` / `runs/{run_id}` confirms it. Terminal on
  `promote_checkpoint`, exhausted iterations, or error.
- Exposed via a chat intent (`drive_sim2real`) + `POST /api/agent/sim2real/drive`.

Rollback: additive module + route; remove wiring to revert. Does not modify the
engine (`workflows/sim2real/engine.py`).

## Phase D — semantic router fallthrough

`agent_semantic_router.py`:

- `classify_intent_semantic(user_text, *, known_intents, model_call, cache=None)`
  → returns `{intent | None, mode: "intent"|"action"|"none", confidence,
  tokens}`. Called **only** when `match_chat_intent` returns `None`.
- Obvious cases short-circuit to 0 tokens via a keyword pre-filter and an
  optional cache; genuine paraphrases use one cheap-tier structured call.
- The grounded regex path is unchanged, so
  `test_agent_capability_parity.py` keeps matching at 0 tokens. New tests add
  paraphrases that regex misses and assert the semantic layer routes them
  (mocked `model_call`, 0 real tokens).

Rollback: the fallthrough is gated; if `classify_intent_semantic` errors or is
disabled, chat falls through to the existing cheap-LLM path exactly as today.

## Phase E — agent task-eval harness

`npa/tests/agent_eval/`:

- `scenarios.py` — operator-goal cases: `goal`, `expected_end_state`, `kind`
  (grounded / workflow_draft / action_loop / sim2real_loop).
- `harness.py` — runs each case against the pure modules with fakes, records
  `success`, `steps`, `tokens` (via `usage_summary`).
- `test_agent_eval_scorecard.py` — asserts a competitive bar and emits a
  scorecard artifact (`success_rate`, `avg_steps`, `avg_tokens`) to
  `npa/tests/agent_eval/_artifacts/scorecard.json`.
- Fully mocked by default (0 tokens, CI-safe); a live variant is gated behind
  `NPA_AGENT_CHAT_LIVE=1` with the cheapest pinned model (Tier-2 convention).

Rollback: test-only; delete the directory.

## Phase F — quantitative viewer eval + cross-run memory

- `agent_visual_feedback.py` (extend): `extract_quantitative_signals(metrics)`
  and `compare_rollouts(run_a, run_b)` returning success-rate readouts,
  policy-collapse/degenerate-rollout flags, and run-to-run deltas. Vision-tier
  gating and the non-blank-frame quality wait are preserved; these functions
  operate on report/metric JSON, feeding the Phase-C diagnose step.
- `agent_memory.py`: `RunMemory` over an injected `store` — `record_run`,
  `get_run`, `list_runs`, `compare_runs`, `explain_regression`. Storage-backed
  (S3/JSON), no hardcoded bucket; answers "why did run B regress vs run A" from
  stored metadata, not model recall.

Rollback: signal helpers are additive pure functions; memory is a new module +
route, removable without touching existing paths.

## Phase G — backend extraction (pay down f-string debt)

Introduce an importable package `npa/src/npa/agent_backend/` that is *shipped* to
the VM (its files uploaded next to `backend.py`, imported via `sys.path`) instead
of string-substituted. Migrate the new B–F modules there incrementally, keeping
the embed mechanism working for everything not yet migrated. Behavior stays
byte-for-byte; the rendered-backend compile check and the full agent suite stay
green at every commit; migration proceeds in small reversible commits.

**Implemented (pilot):** `agent_memory` is the first module migrated. Its logic
lives in `npa/src/npa/agent_backend/memory.py`; `npa/src/npa/cli/agent_memory.py`
is a thin re-export shim so existing import paths/tests are unchanged. The
bootstrap ships the package files via a `cat <<'PY' … agent_backend/memory.py`
heredoc (placeholder `_AGENT_MEMORY_SHIP` substituted with the full source), and
`backend.py` does `sys.path.insert(0, "/opt/npa-agent")` + `from
agent_backend.memory import RunMemory, JsonFileStore, InMemoryStore` instead of
inlining the class. A dedicated test compiles the shipped module and asserts
`backend.py` no longer inlines it.

**Mechanism for the remaining modules (actions, sim2real_loop, semantic_router):**
each follows the same pilot pattern — `git mv` into `agent_backend/`, add a
`cli/agent_*.py` shim, swap the embed placeholder for a ship heredoc +
`from agent_backend.<mod> import …`, and add a shipped-module compile check.
This is preferable to `import *` because the module keeps its own globals (no
backend-namespace symbol collisions such as the shared `STOP_ERROR`).

**Rollback:** re-embedding a shipped module is mechanical — restore its
`_embedded_*` reader + `_AGENT_*_EMBED` placeholder and drop the ship heredoc.
Because the shim keeps the `npa.cli.*` import path stable, no caller changes.

## Consistency with the three-layer cost model

Layer 1 (grounded, 0 tokens) is untouched and remains the default. Layer 2
(cheap routing) is reused by B/C/D for their model calls. Layer 3 (Token Factory
client) is reached only through injected `model_call`, so the new agentic
surface never bypasses cost-tier routing or the resilience ladder.
