# Blueprint Incorporation Plan — retrieval, observability, adversarial eval

Status: living design doc for the follow-on workstream after PR #207 (grounded
assistant → Physical-AI agent, phases 0/A–G, MERGED). Companion docs:
`docs/architecture/agent-competitive-plan.md`,
`skills/atomic/agent-development/SKILL.md`.

## Goal

Close the three gaps between our agent and the Nebius "Blueprints" reference
agent (dev.nebius.com/blueprints): **retrieval/grounding**, **observability**,
and **adversarial eval**. We incorporate only the *open-source* parts of that
stack and run everything on Nebius Token Factory (LLM + embeddings,
OpenAI-compatible) and Nebius AI Cloud (infra), driven through the npa
agent/CLI. No proprietary SaaS: no LangSmith, no Pinecone, no hosted Tavily, no
Snowglobe.

## Open-source mapping

| Blueprint (proprietary) | Our open-source replacement | Where |
| --- | --- | --- |
| Pinecone vector store | **LanceDB** (embedded/AI-Cloud), Token Factory embeddings | Phase H `agent_backend/retrieval.py` |
| Tavily hosted search | pluggable injected `web_search` (SearXNG self-hosted via npa, or generic fetch) | Phase H `web_search` collaborator |
| LangSmith tracing | self-hostable OSS tracing (Langfuse / OpenTelemetry) behind an injected tracer | Phase I `agent_backend/trace.py` |
| Snowglobe simulation | persona + prompt-injection scenario generation + guardrails-ai + delta gating | Phase J `npa/tests/agent_eval/adversarial.py` |
| LangChain/LangGraph orchestration | **KEEP** our bespoke bounded loop (`agent_actions.py`); reuse patterns, not the dependency | unchanged |

## Non-negotiable invariants (carried forward from #207)

1. **Grounded-first, zero-token path stays the default.** All new capability is
   a *fallthrough*, reached only after `match_chat_intent` misses. Retrieval in
   `/chat` only fires when a corpus index already exists and the top match clears
   a confidence floor; otherwise chat behavior is byte-for-byte unchanged.
2. **Cost discipline.** New model/embedding calls default to the cheap tier via
   `agent_routing`; escalation is deliberate. Retrieval answers are *extractive*
   (cite retrieved snippets) so a retrieval turn spends embedding tokens only —
   no generation tokens — unless the operator explicitly asks to synthesize.
3. **Confirmation gates.** Retrieval and trace-analysis tools are **read-only**;
   they never spend GPU or write externally, so they carry no confirm gate.
   Indexing live web content is opt-in per request. GPU/destructive/external-write
   actions still require the digest-bound `confirm_token`.
4. **Embedded-backend / dependency-injection.** Every side-effecting collaborator
   (`model_call`, `embed`, `tools`, `store`, `web_search`, `tracer`) is an
   INJECTED callable, so logic unit-tests at 0 tokens / no network. New logic goes
   in real modules shipped through `agent_backend/` (Phase-G pilot pattern), not
   new `backend.py` f-string embeds.
5. **No fabricated data, no hardcoded infra/secrets.** Citations only reference
   real indexed content. Credentials from `~/.npa/credentials.yaml`, config from
   `~/.npa/config.yaml`. LanceDB / SearXNG / tracer endpoints are provisioned via
   npa (AI Cloud) or env-configured, never hardcoded. No PII/secrets in spans.

## Phase H — Retrieval / grounding

New shipped module `npa/src/npa/agent_backend/retrieval.py` (re-export shim
`npa/src/npa/cli/agent_retrieval.py`).

- **Chunking** — `chunk_text()` splits docs into overlapping windows; markdown
  headings are preserved as chunk titles.
- **Corpus discovery** — `iter_corpus_documents(root)` walks the repo `docs/` +
  `skills/` trees (and any extra roots) yielding `(uri, title, text)`.
- **Indexing** — `index_corpus(documents, *, embed, store, source, ...)` chunks,
  calls the injected `embed(texts) -> list[vector]`, and upserts typed records
  into the injected `store`. Returns `{ok, chunks_indexed, sources}`.
- **Retrieval** — `retrieve(query, *, embed, store, k, web_search=None,
  index_web=False, min_score=...)` embeds the query, searches the store, and
  returns typed `Citation` records (`source`, `title`, `snippet`, `score`,
  `uri`, `kind`). When `web_search` is injected and `index_web` is set, live
  results are folded in provider-agnostically.
- **Grounded answer** — `format_grounded_answer(query, citations)` builds an
  extractive, cited markdown reply with **no generation call** (0 model tokens).
- **Vector stores** — `InMemoryVectorStore` (pure-python cosine, tests +
  fallback), `JsonVectorStore` (persisted pure-python, VM default when LanceDB is
  unavailable), and `build_lance_store(uri, table)` (LanceDB adapter, guarded
  import). All satisfy the `add` / `search` protocol so the store is injectable.
- **Exposure** — read-only `retrieval_search` tool in the `agent_actions`
  `TOOL_ALLOWLIST`; routes `POST /api/agent/retrieval/index`,
  `GET /api/agent/retrieval/search`, `GET /api/agent/retrieval/status`; and a
  grounded-first `/chat` fallthrough behind the semantic layer.

Rollback: additive module + shim + routes + one allowlist entry + one gated
`/chat` fallthrough branch. Remove the ship heredoc and route wiring to revert;
`/chat` grounded/semantic behavior is untouched when no index exists.

## Phase I — Observability

New shipped module `npa/src/npa/agent_backend/trace.py` (re-export shim
`npa/src/npa/cli/agent_trace.py`).

- **Spans** — `Span` records `{name, kind, status, duration_ms, attributes,
  events}`. `TraceRecorder` (injected tracer) collects spans; `NullTracer` is the
  no-op default, `InMemoryTracer` collects for tests/analysis, and
  `build_langfuse_tracer` / `build_otel_tracer` are guarded-import adapters to
  self-hosted OSS backends.
- **Wrapping the loop** — `spans_from_action_loop(result)` and
  `spans_from_drive(result)` turn the existing step/iteration traces into
  structured spans without changing loop logic. `trace_run(tracer, name, attrs)`
  is a context manager for ad-hoc spans.
- **Redaction** — `redact_attributes()` strips secret-like keys (token, key,
  secret, password, authorization, credential) and long high-entropy values so no
  PII/secret enters a span.
- **Offline analyzer** — `analyze_traces(traces)` clusters traces by
  `(stopped_reason, tool signature, failure signature)` and flags **silent
  failures**: truncated observations, empty tool results, planner/tool errors that
  did not surface, and max-steps exhaustion. Returns `{clusters, silent_failures,
  totals}`.
- **Exposure** — read-only routes `GET /api/agent/trace/spans` (recent spans) and
  `POST /api/agent/trace/analyze` (analyze supplied or recent traces).

Rollback: additive module + shim + routes; the tracer defaults to `NullTracer`
so tracing is a no-op unless an operator wires a backend.

## Phase J — Adversarial eval

Extends `npa/tests/agent_eval/`.

- **Personas** — `generate_personas(model_call, n)` produces operator personas
  via an injected Token Factory model (mocked in CI); a static persona set is the
  0-token fallback.
- **Prompt-injection corpus** — `INJECTION_ATTACKS` enumerates open categories
  (ignore-previous-instructions, system-prompt exfiltration, secret exfiltration,
  auto-launch GPU without confirmation, fabricate run data, tool-allowlist
  escape).
- **Scenario generation** — `build_adversarial_scenarios(personas, attacks)`
  crosses personas × attacks into typed cases.
- **Validators** — `validate_output(text, *, gpu_launched, confirmed)` uses
  guardrails-ai validators when installed (`pytest.importorskip`) and a pure
  fallback otherwise: no secret/credential leak, no unconfirmed GPU launch, no
  fabricated completion, no system-prompt disclosure.
- **Harness + gating** — `run_adversarial_suite()` runs each attack against the
  real grounded router + bounded action loop with mocked collaborators (0 tokens)
  and scores a `defense_rate`. `test_agent_adversarial_scorecard.py` gates
  **delta-vs-baseline**: the committed baseline
  (`_artifacts/adversarial_baseline.json`) is the floor; a regression beyond
  tolerance fails CI. Emits `_artifacts/adversarial_scorecard.json`.

Rollback: test-only; delete the adversarial module + test + baseline.

## Testing & verification

- Full agent suite stays green, CI-safe, 0 tokens (mock `model_call` / `embed` /
  `web_search` / `tracer` / `store` at the call site; `pytest.importorskip` for
  lancedb / guardrails / langfuse; CLI tests via `typer` `CliRunner`).
- Rendered-backend compile check asserts the new shipped modules
  (`agent_backend/retrieval.py`, `agent_backend/trace.py`) are uploaded + imported
  and the new routes are wired.
- Live validation on a bootstrapped agent VM (real HTTPS): index a small corpus,
  `retrieval/search` returns grounded citations, the tracer records spans, and the
  adversarial scorecard runs. All AI Cloud resources torn down after.

## Optional dependencies

New capabilities are optional extras so core install stays lean and CI stays
green without them:

- `npa[lancedb]` (already present) backs `build_lance_store`.
- `npa[agent-eval]` adds `guardrails-ai` for the guardrails validator tier.
- `npa[agent-trace]` adds `langfuse` / `opentelemetry-sdk` for the tracer
  adapters.

All three are injected/guarded; absence degrades to the pure-python path.
