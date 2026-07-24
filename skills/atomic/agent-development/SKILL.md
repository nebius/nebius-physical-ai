---
name: agent-development
description: Use when building, enhancing, or testing the NPA chat agent backend — grounded-first routing, cost-aware Token Factory model selection, the embedded-backend mechanism, and cheap-token test tiers.
---

# Agent Development (cheap-token chat backend)

How to *develop* the NPA agent chat backend. For *operating* a deployed agent
(deploy/bootstrap/verify, chat UX, API grounding, Rerun) use
`skills/tools/npa-agent/SKILL.md`; for fresh deploy/teardown loops use
`skills/workflows/agent-fresh-operate/SKILL.md`.

Guiding principle: **the cheapest token is the one you never spend.** Protect and
widen the zero-token grounded path; make the unavoidable model calls small,
structured, and cheap-model-first.

## Architecture (three layers)

1. **Grounded intent router (zero tokens)** — `npa/src/npa/cli/agent_chat.py`.
   `match_chat_intent()` classifies a turn; `build_grounded_reply()` answers from
   live session state with **no model call** (`"grounded": true`). Most operator
   turns end here.
2. **Cost-tier routing (cheap model calls)** — `npa/src/npa/cli/agent_routing.py`.
   Only turns that fall through the router reach Token Factory, and this layer
   keeps them cheap.
3. **Token Factory client** — `npa/src/npa/clients/token_factory.py` (operator/SDK
   path) and the embedded `_provider_chat` / `_chat_with_resilience` in
   `npa/src/npa/cli/agent.py` (agent-VM path).

## The embedded-backend mechanism (critical)

The agent VM runs `backend.py`, which is built as one big f-string inside
`_bootstrap_agent_stack` in `npa/src/npa/cli/agent.py`. Pure helper modules are
inlined into it via placeholder substitution — the pattern to reuse when adding
logic:

- Real module (normal Python, **no** brace escaping): `agent_chat.py`,
  `agent_workflow.py`, `agent_routing.py`.
- Embed helper `_embedded_agent_<name>_source()` strips the docstring +
  `from __future__` line.
- Placeholder constant `_AGENT_<NAME>_EMBED` appears in the template and is
  replaced in the `.replace(...)` chain near the end of `_bootstrap_agent_stack`.

Rules:

- Put testable logic in a **real module** and embed it; do **not** write new
  logic directly inside the f-string unless it must touch template variables.
- Code written *inside* the f-string must escape literal braces as `{{` / `}}`
  and newlines in strings as `\\n`; substitutions use single `{var}`.
- After editing the template, **validate the rendered backend compiles** (see
  Testing). A stray brace is a `SyntaxError` at import of `agent.py`.

## Cost-tier routing (`agent_routing.py`)

Pure, side-effect-free functions (no network) so they unit-test cheaply:

- `classify_tier(text, intent, messages)` → `cheap` (default) / `standard`
  (long/compound) / `reasoning` (analytical language) / `vision` (image content).
- `build_model_ladder(tier, configured, interactive, requested_model,
  allow_tier_defaults)` → cheapest-capable first; explicit user model wins;
  operator allowlist (`NPA_AGENT_LLM_MODELS`) respected when set.
- `flavor_variants` + `filter_available` — prefer the Token Factory `-fast`
  flavor for interactive turns, but drop variants the key cannot serve (never
  strand a turn).
- `chat_extra` / `thinking_enabled` — disable hidden reasoning traces off the
  reasoning tier (don't pay for discarded tokens).
- `enforce_input_budget` — cap oversized pastes (head+tail preserved).
- `usage_summary` — surface per-turn token usage.

The `/chat` handler classifies the tier, enforces the input guardrail, honors an
explicit model override, and returns `tier` + `usage` + `input_budget_ok`.

## Adding a capability cheaply (decision order)

1. **Grounded intent** — can a regex intent + grounded state reply answer it?
   (0 tokens) Add to `_INTENT_RULES` / `build_grounded_reply` in `agent_chat.py`
   and cover it in `test_agent_chat.py`. Prefer this for anything high-frequency.
2. **Cheap model** — if it needs generation, let routing pick the cheap tier;
   only add reasoning/vision signals to `classify_tier` when the turn truly
   needs them.
3. **Escalate deliberately** — reserve `nvidia/Cosmos3-Super-Reasoner` for
   analytical/physical-AI/vision turns; it is overkill for routine chat.
4. **Visual feedback** — UI **Describe this** captures the active viewer frame
   and posts multimodal `/api/chat` with `visual_context`. Helpers live in
   `agent_visual_feedback.py` (embedded). Never ground these turns; use vision
   when an image is attached. See `skills/atomic/agent-visual-feedback/SKILL.md`.

## Token Factory notes

- OpenAI-compatible: base `https://api.tokenfactory.nebius.com/v1/`, key
  `NEBIUS_TOKEN_FACTORY_KEY`. Same `chat/completions` shape everywhere.
- Model **flavors**: `-fast` (low latency, pricier) vs base; identical output.
  Only append `-fast` when the key exposes it (`filter_available`).
- Cost-ordered default ladder lives in `DEFAULT_LLM_MODELS` (cheap first). A bare
  `npa agent deploy` seeds this whole ladder, so per-turn routing reaches every
  tier without `--llm-models`. Explicit `--llm-models` is a governance allowlist;
  `/api/models` still surfaces every model the key can serve for per-request
  selection.
- Reasoning-trace handling: `split_reasoning()` normalizes Cosmos3 inline
  `<think>` and Kimi/GLM `reasoning` fields.

## Testing tiers (keep tokens out of CI)

Follow `skills/atomic/testing-conventions/SKILL.md`; use `npa/.venv/bin/python`.

- **Tier 0 — pure logic (0 tokens):** `npa/tests/cli/test_agent_routing.py`
  (tiers, ladder, flavor, availability filter, budget, usage) and
  `test_agent_chat.py` (intent router, grounded replies). Highest-value coverage.
- **Tier 1 — mocked LLM (0 tokens):** patch `_provider_chat` /
  `_chat_with_resilience`; assert prompt assembly, resilience fallthrough, and
  tier/usage in the response.
- **Rendered-backend check:** confirm the embedded backend compiles with all
  wiring inlined — render `setup_script` with mocked SSH, extract the
  `backend.py` heredoc body, and `ast.parse` + `compile` it. Guards the f-string.
- **Tier 2 — live e2e (bounded tokens):** gate behind `NPA_AGENT_CHAT_LIVE=1` /
  `NPA_INTEGRATION_E2E=1`; pin the cheapest model; assert `grounded: true` where
  possible so most turns cost 0 tokens.

```bash
npa/.venv/bin/python -m pytest npa/tests/cli/test_agent_routing.py \
  npa/tests/cli/test_agent.py npa/tests/cli/test_agent_chat.py \
  npa/tests/smoke/test_agent_smoke.py npa/tests/smoke/test_agent_chat_smoke.py \
  npa/tests/guardrails/test_agent_secret_guard.py -q
```

## Guardrails

- Never leak credentials/auth env/secrets into chat, logs, or workflow YAML.
- Do not hardcode project IDs, tenant IDs, bucket names, registry IDs, usernames,
  or public IPs in code or examples.
- Preserve the chat contract: grounded-first, then a cheap LLM fallback. Do not
  regress the agent into a chat-only (always-LLM) design.

## Agentic surface (fallthrough beyond grounded)

Everything below runs **only after** the grounded intent router misses; the
zero-token path stays the default. Design doc:
`docs/architecture/agent-competitive-plan.md`.

- **Bounded tool-calling loop** — `npa/src/npa/cli/agent_actions.py`
  (`run_action_loop`): classify → plan → call → observe → decide → stop with a
  hard `max_steps` guard, an explicit `TOOL_ALLOWLIST`, and a confirmation-gate
  contract. GPU/destructive tools need a token **bound to the action digest**
  (`action_digest`); tokens are single-use. Route: `POST /api/agent/act`.
- **Autonomous Sim2Real drive** — `npa/src/npa/cli/agent_sim2real_loop.py`
  (`drive_sim2real_loop`): launch→status→gate→diagnose→adjust→re-run, mirroring
  the engine `promote_checkpoint`/`loop_back` gate. Every launch is
  confirmation-gated; stages complete only when live status confirms the run
  (no fabrication); stops on insufficient signal / no-adjustment to avoid
  runaway GPU. Route: `POST /api/agent/sim2real/drive`; chat intent
  `drive_sim2real` returns grounded guidance.
- **Semantic fallthrough** — `npa/src/npa/cli/agent_semantic_router.py`
  (`classify_intent_semantic`): keyword + cache (0 tokens) then one cheap
  structured call to map regex-missed paraphrases to a known intent/action.
  Wired into the `/chat` fallthrough; degrades to `none` on failure. Parity
  intents still match in `match_chat_intent` and never reach it.
- **Quantitative viewer eval + memory** — `agent_visual_feedback.py`
  (`extract_quantitative_signals`, `compare_rollouts`) and the shipped package
  `npa/src/npa/agent_backend/memory.py` (`RunMemory`, storage-injected, no
  hardcoded bucket). Routes: `GET/POST /api/agent/memory/*`.
- **Task-eval harness** — `npa/tests/agent_eval/` (mocked, 0 tokens): scenarios
  + scorecard (`success_rate`/`avg_steps`/`avg_tokens`); live variant gated on
  `NPA_AGENT_CHAT_LIVE=1`.

**Shipped vs embedded (Phase G):** new logic still uses the embed mechanism by
default; `agent_backend/` is the shipped-package migration target (uploaded to
`/opt/npa-agent/agent_backend/`, imported via `sys.path`). `agent_memory` is the
migrated pilot; `cli/agent_memory.py` is a re-export shim. Rendered-backend
compile check: `npa/tests/cli/test_agent_backend_render.py`.

## Source Layout

- CLI + bootstrap + embedded backend: `npa/src/npa/cli/agent.py`
- Grounded intent router (testable): `npa/src/npa/cli/agent_chat.py`
- Cost-tier routing (testable): `npa/src/npa/cli/agent_routing.py`
- Agentic tool loop / sim2real drive / semantic router:
  `npa/src/npa/cli/agent_actions.py`, `agent_sim2real_loop.py`,
  `agent_semantic_router.py`
- Shipped backend package: `npa/src/npa/agent_backend/` (memory pilot)
- Token Factory client: `npa/src/npa/clients/token_factory.py`
- Routing tests: `npa/tests/cli/test_agent_routing.py`; agentic tests:
  `test_agent_actions.py`, `test_agent_sim2real_loop.py`,
  `test_agent_semantic_router.py`, `test_agent_memory.py`,
  `test_agent_backend_render.py`; eval harness: `npa/tests/agent_eval/`
