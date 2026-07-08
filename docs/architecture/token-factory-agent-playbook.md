# Building and Testing the NPA Agent on Token Factory (Cheap-Token Playbook)

Research note on how to build an *intelligent* NPA chat agent that stays
*cheap on tokens*, and how to test it without burning inference spend. It maps
Nebius Token Factory capabilities onto the agent that already ships in
`npa/src/npa/cli/agent.py`, and records the patterns we should keep, extend, and
test.

Audience: engineers working on the agent VM (`npa agent`), the grounded chat
router (`npa/src/npa/cli/agent_chat.py`), and the Token Factory client
(`npa/src/npa/clients/token_factory.py`).

---

## 1. What Token Factory gives us

Nebius Token Factory (formerly Nebius AI Studio) is a serverless,
**OpenAI-compatible** inference API for hosted open models. The single source of
truth for endpoint/auth/request shaping in this repo is
`npa/src/npa/clients/token_factory.py`; workbench tools and the agent VM both
route through it (or through the embedded copy baked into the agent server).

Key facts that shape a cheap-token design:

| Capability | Detail | Why it matters for cost |
| --- | --- | --- |
| OpenAI-compatible API | Base URL `https://api.tokenfactory.nebius.com/v1/`, `NEBIUS_TOKEN_FACTORY_KEY`. Only base URL + key change vs. OpenAI. | We reuse the same `chat/completions` shape everywhere; no bespoke SDK. |
| Model flavors | Every model has a **Base** and a **Fast** flavor (append `-fast`). Identical outputs; Fast trades price for latency via speculative decoding. | Use Fast only for interactive turns; Base for everything else. |
| Batch API | Async jobs at **50% off** base pricing, up to 5M requests / 10 GB, ~24h window, and **does not consume per-model rate limits**. | Offline evals, regression suites, and bulk reasoning belong here. |
| Wide model spread | 60+ models from ~8B (Llama 3.1-8B) to large MoE (Qwen3-235B, DeepSeek, Kimi, GLM, GPT-OSS) plus vision (Qwen2.5-VL) and the `nvidia/Cosmos3-Super-Reasoner`. | Route by task difficulty; small models are 5-20x cheaper. |
| Native function calling | `tools` + `tool_choice` (`auto`/`required`/`none`). | Lets the model pick an action without us paying for long free-text plans. |
| Structured outputs | `response_format` with a JSON schema (already plumbed through `TokenFactoryClient.chat_completion`). | Short, parseable replies = fewer output tokens + no retry parsing loops. |
| Reasoning trace handling | `split_reasoning()` normalizes Cosmos3 inline `<think>` and Kimi/GLM `reasoning` fields. | Avoids paying for and then displaying raw thinking; can disable thinking. |

Indicative self-service pricing (per 1M tokens, subject to change — always
confirm with `npa workbench token-factory models` and the pricing page):

| Model | Input $ | Output $ | Good for |
| --- | --- | --- | --- |
| Qwen 3 32B | ~0.10 | ~0.30 | Default cheap workhorse / routing / drafting |
| Llama 3.3 70B | ~0.13 | ~0.40 | Balanced quality, current agent fallback |
| GPT-OSS 120B | ~0.15 | ~0.60 | Harder reasoning at moderate cost |
| Qwen 3 235B | ~0.20 | ~0.60 | High quality when needed |
| DeepSeek V3 / Kimi K2 | ~0.50 | ~1.5-2.4 | Escalation ceiling only |

The cheapest capable model is roughly **an order of magnitude** below the
escalation tier. The whole game is: **answer without an LLM when possible, use a
small model when you must, and reserve big models for the rare hard turn.**

---

## 2. How the NPA agent already saves tokens

The agent VM backend (`npa/src/npa/cli/agent.py`, `POST /chat`) is *not* a naive
"forward everything to the LLM" loop. It is a **grounded-first, LLM-fallback**
design that already embodies most cheap-token principles. Preserve and extend
this; do not regress it into a chat-only agent.

### 2.1 Deterministic intent routing (zero-token path)

`match_chat_intent()` in `agent_chat.py` is a large regex router that classifies
a user turn into ~25 intents (`watch_sim`, `find_artifacts`, `create_workflow`,
`infra_backends`, `soperator`, `cosmos3`, …). When an intent matches,
`build_grounded_reply()` / `_maybe_toolground_chat_reply()` produce a **grounded,
canned reply from live session state** — no model call at all. The response is
tagged `"grounded": true` (see `_agent_chat_with_tools`).

```470:498:npa/src/npa/cli/agent_chat.py
def match_chat_intent(user_text: str) -> str | None:
    text = str(user_text or "").strip()
    if not text:
        return None
    lowered = _normalize_intent_text(text)
    ...
    for intent, pattern in _INTENT_RULES:
        if pattern.search(text) or pattern.search(lowered):
            return intent
    return None
```

This is the single biggest cost lever: **the most common operator questions
(status, "watch the sim", "list recordings", "what tools exist", "onboard a
repo") never touch Token Factory.** Every new high-frequency capability should
first be considered for a grounded intent before it is considered for an LLM
prompt.

### 2.2 LLM only as fallback, with a compact grounded prompt

Only when no intent/tool matches does `chat()` build a real prompt and call
`_chat_with_resilience()`. Even then it keeps the payload lean:

- System prompt is assembled once by `_agent_system_prompt()` (API map + tool
  catalog), plus a compact JSON **live session snapshot** from
  `format_live_context_block()` — not the full state blob.
- Skill context is injected selectively via `_resolve_skill_context()` based on
  the detected intent, not dumped wholesale.
- History is capped (`chat_history[-80:]`) so context does not grow unbounded.

### 2.3 Provider/model resilience ladder

`_chat_with_resilience()` iterates configured providers × models and returns the
first success. Defaults (see `agent.py`):

```54:58:npa/src/npa/cli/agent.py
DEFAULT_LLM_MODELS = (
    DEFAULT_LLM_MODEL,
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
)
```

Today the list is ordered by *capability/availability*, not *cost*. That is the
main place the current design leaves money on the table (see §3.2).

### 2.4 Model list caching

`_available_llm_models()` caches the Token Factory `/models` result for 5
minutes, so the model picker does not re-hit the API on every UI load.

---

## 3. Recommendations: an intelligent agent on cheap tokens

Ordered by cost impact. Items marked **(have)** already exist and should be
protected by tests; items marked **(gap)** are proposed improvements.

### 3.1 Deterministic-first, LLM-last (have — extend)

Keep the grounded router as the front door. When adding a capability, ask in
order:
1. Can a regex intent + grounded state reply answer it? (0 tokens)
2. Can a small model with a **tool/function schema** pick a known action? (tiny
   output)
3. Only then, free-form generation on a small model, escalating on failure.

Add a lightweight **"grounded coverage" metric** (share of turns answered with
`grounded: true`) to the agent so we can watch it and defend it in review.

### 3.2 Cost-ordered model ladder (gap)

Reorder the default ladder so the **cheapest adequate model is tried first** and
escalation is explicit, not accidental:

```
route/classify + short replies : Qwen/Qwen3-32B  (or -fast for interactive)
default drafting               : meta-llama/Llama-3.3-70B-Instruct
hard reasoning / vision        : nvidia/Cosmos3-Super-Reasoner, Qwen2.5-VL-72B
escalation ceiling             : Qwen3-235B / DeepSeek V3 (rare)
```

Make the ladder configurable via `NPA_AGENT_LLM_MODELS` (already supported) and
document a **cheap default** rather than a Cosmos3-first default for text-only
chat. Reserve Cosmos3-Super-Reasoner for physical-AI/vision reasoning where it
earns its cost; it is overkill for "how do I configure S3".

### 3.3 Use `-fast` only where a human is waiting (gap)

Interactive chat turns → `-fast` flavor for sub-2s latency. Background/agentic
steps (workflow drafting, evals, summarization) → Base flavor. Same output,
lower price. Wire a `flavor` hint into `_provider_chat()` payload construction.

### 3.4 Structured outputs + function calling to shrink output tokens (have client / gap in agent)

`TokenFactoryClient.chat_completion` already accepts `response_format` and
`extra` (so `tools`/`tool_choice` pass through). The **agent VM** path
(`_provider_chat`) does not yet send these. For any turn that ends in a
concrete action (submit workflow, load artifact, choose a preset), send a JSON
schema or tool schema so the model returns a compact structured object instead
of prose. Fewer output tokens, no brittle text parsing, fewer retries.

### 3.5 Disable thinking unless it changes the answer (have — apply)

Reasoning models bill for the hidden trace. `split_reasoning()` already handles
Cosmos3/`<think>` and Kimi/GLM `reasoning`. For routing, classification, and
short factual replies, pass `chat_template_kwargs={"thinking": false}` (via
`extra`) to avoid paying for a trace we will discard. Keep thinking only for
genuine multi-step reasoning turns.

### 3.6 Context minimization and memory (partial)

- Keep the **compact JSON snapshot** pattern (`format_live_context_block`) — do
  not inline full state.
- Cap and summarize history. Consider a cheap "rolling summary" turn (small
  model, Base flavor) instead of resending 80 messages once sessions get long.
- Selective skill injection is already intent-gated; keep it that way. Do not
  concatenate all `SKILL.md` files into the prompt.
- System prompt and tool catalog are stable across turns — good candidates for
  prompt caching if/when Token Factory exposes it; today, keep them short.

### 3.7 Batch API for anything not real-time (gap)

Nightly eval suites, bulk artifact captioning, dataset labeling, and regression
prompt banks should use the **Batch API (50% off, no rate-limit consumption)**
rather than synchronous calls in a loop. This is the cheapest way to run the
large "does the agent still answer these 200 prompts correctly" suite.

### 3.8 Guardrails to cap runaway input (gap)

Add input validation before the LLM path (as the Nebius "Agent 101" guidance
recommends): reject/trim oversized pastes, and short-circuit obvious
out-of-scope requests to a grounded "I can help with NPA workbench tasks" reply.
One 10k-word paste can dwarf a day of normal chat cost.

---

## 4. Testing the agent cheaply

The point of a cheap-token *build* is undermined by an expensive *test* loop.
The repo already separates test tiers; keep LLM spend out of the default suite.

### 4.1 Tier 0 — pure-logic unit tests (no network, no tokens)

The router and formatters are pure functions and must have exhaustive tests:

- `match_chat_intent()` — table-driven cases per intent, including the tricky
  `watch_sim` success-gating and precedence rules. This is the cheapest, highest
  value coverage: it guards the **zero-token** path.
- `build_grounded_reply()` / `format_*` — assert grounded replies contain the
  required fields (see `agent_live_helpers.assert_grounded_onboard_solution_reply`
  as the shape to mirror) and never emit raw `GET /api/...` stubs or
  `<your-registry-id>` placeholders.
- Model-ladder / provider selection logic — order, dedupe
  (`_normalize_llm_models`), and cost-first ordering once §3.2 lands.

Convention (per `CLAUDE.md`): use `npa/.venv/bin/python -m pytest`, mock every
network boundary at the call site, never import GPU packages at module level.

### 4.2 Tier 1 — mocked LLM (no tokens)

For the fallback path, mock `_chat_with_resilience` / `_provider_chat` (or patch
`httpx.post`) and assert:
- prompt assembly (system prompt present, compact snapshot injected, history
  capped),
- resilience ladder falls through providers/models on transient errors and
  retries only on 408/409/425/429/5xx (already coded in `_provider_chat`),
- structured-output / tool payloads are constructed when §3.4 lands,
- `grounded:false` only when no intent matched.

Mocking here lets us assert *that we would send a cheap request*, without
sending one.

### 4.3 Tier 2 — live e2e on the cheapest model (bounded tokens)

`npa/tests/e2e/test_agent_live.py` + `agent_live_helpers.py` hit a real deployed
agent. To keep this cheap:
- Pin the live test model to the **cheapest capable** model (e.g. Qwen3-32B),
  not Cosmos3, via `NPA_AGENT_LLM_MODELS` in the test env.
- Prefer prompts that exercise **grounded** intents (assert `grounded: true`) so
  most live assertions cost 0 tokens and only a couple of cases exercise the LLM
  fallback.
- Keep `max_tokens` small and temperature low for determinism.
- Gate behind `NPA_INTEGRATION_E2E=1` so it never runs in the default suite
  (mirror the serverless e2e pattern in `docs/testing/e2e-serverless.md`).

### 4.4 Tier 3 — offline eval via Batch (50% off)

For agent-quality regression ("golden prompt bank"), build a JSONL of prompts +
expected properties and run it through the **Batch API** overnight. This is the
right home for the large, slow, LLM-in-the-loop evaluation — half price and
outside per-model rate limits. Score outputs with cheap assertions (schema
validity, required substrings) and, where needed, a small **LLM-as-judge**
model rather than a frontier judge.

### 4.5 A token budget per turn

Add a simple budget assertion to tests and (optionally) a runtime cap:
`prompt_tokens + completion_tokens` per chat turn should stay under a documented
ceiling for the fallback path. The chat-completion response already returns a
`usage` block — surface and log it so we can track cost per intent over time.

---

## 5. Concrete next steps (proposed, not yet implemented)

1. Reorder `DEFAULT_LLM_MODELS` to a cost-first ladder; document the cheap
   default and keep Cosmos3 for vision/physical reasoning intents only. (§3.2)
2. Thread `flavor` (`-fast` vs base), `response_format`/`tools`, and
   `chat_template_kwargs.thinking=false` through the agent-VM `_provider_chat`.
   (§3.3–3.5)
3. Add input guardrails + a per-turn token budget log using the `usage` field.
   (§3.8, §4.5)
4. Expand `test_agent_chat.py` intent-router coverage and add mocked-LLM
   fallback tests; pin live e2e to the cheapest model. (§4.1–4.3)
5. Stand up a Batch-API golden-prompt eval as the nightly quality gate. (§4.4)

The overarching principle: **the cheapest token is the one you never spend.**
The NPA agent's grounded-first router is the asset that makes an intelligent
agent affordable — every design and test decision should protect and widen that
zero-token path, and make the unavoidable model calls small, structured, and
cheap-model-first.

---

## References

- `npa/src/npa/clients/token_factory.py` — Token Factory client (base URL, auth,
  `chat_completion`, `split_reasoning`, `list_models`).
- `npa/src/npa/cli/agent.py` — agent VM backend: `POST /chat`,
  `_maybe_toolground_chat_reply`, `_chat_with_resilience`, `_provider_chat`,
  `_agent_system_prompt`, model ladder + caching.
- `npa/src/npa/cli/agent_chat.py` — grounded intent router and reply formatters.
- `docs/hackathon-cosmos3-reasoner.md`, `docs/hackathon-isaac-token-factory.md`
  — serverless Token Factory reasoning paths (no GPU).
- `docs/testing/e2e-serverless.md`, `npa/tests/e2e/test_agent_live.py` — e2e
  conventions to gate live token spend.
- Nebius Token Factory docs: model flavors (Fast/Base), Batch inference (50%
  off), function calling, and structured `response_format` outputs.
