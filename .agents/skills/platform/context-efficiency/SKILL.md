---
name: context-efficiency
description: Single source of truth for behavioral constraints. Apply on every turn to protect prompt-cache prefix stability, minimize context ingestion, keep multi-turn chat memory lean, avoid full-workspace scans, and route work to the right model. Use when deciding how much to read, how to track state, how to structure code for small reads, and which model tier to use.
---

# Context & Token Efficiency

Highest priority: protect the cached prompt prefix, minimize context ingestion,
avoid full-workspace scans, and keep multi-turn chat memory lean. Apply these
rules on every turn.

## 0. Prompt Caching & Prefix Stability

- Treat this file, the system instructions, and the repo index files
  (`AGENTS.md`, `CLAUDE.md`) as a permanent, static prefix. Do not restate,
  paraphrase, or re-emit them — doing so invalidates the prompt cache and taxes
  every subsequent turn.
- Never emit repetitive explanations, long greetings, restated task summaries,
  or structural boilerplate that shifts prompt layout between turns. Stable
  layout keeps the cache warm.
- Keep responses compact and direct. Lead with the action or answer; omit
  preamble and filler. Brevity preserves context-window longevity.

## 1. Context Minimization & Boundaries

- NEVER read unrequested files or entire directories. Read only what the current
  task requires. Cross-module changes are the only justification for widening
  scope, and even then read the specific touched symbols, not whole trees.
- Enforce targeted symbol isolation. Prefer a named class, type, function, or
  constant over reading full files; request specific file lines or exact symbol
  names via `@`-references (e.g. `@path/to/file.py` then a symbol/line range).
  Use grep/symbol search to jump to the relevant lines, then read a tight range.
- If a file exceeds 500 lines, refuse to read it whole. Request the specific
  snippet, symbol, function, or module chunk you need, or use a search to locate
  the exact region first.
- Do not re-read a file you already have in context. Reuse what you've seen
  unless it changed.
- Avoid speculative exploration. If you're unsure a file is relevant, ask before
  reading it.

## 2. State & Memory Management

- Maintain an ultra-lean structural log (e.g. `todo.md` or `agents.md`) that
  tracks ONLY: active architectural state, current task blocks, and breaking
  changes.
- The memory file must stay high-level. Strictly forbidden: code dumps, raw
  logs, full stack traces, command output, or copied file contents. Summarize in
  one line instead (e.g. "auth flow refactored to use middleware; token refresh
  pending").
- When a localized sub-task is completed, advise clearing the chat history or
  opening a fresh thread before starting the next unrelated task. A clean thread
  is cheaper and more accurate than a long one.

## 3. DRY / Modular Code Principles

- Enforce high modularity and strong separation of concerns. One responsibility
  per file/module.
- Keep files small and single-purpose. Smaller, isolated files keep automatic
  context ingestion small and make targeted reads possible.
- Favor shallow nesting and flat, predictable structure over deep hierarchies.
- Eliminate duplication: extract shared logic into reusable functions/modules
  rather than copy-pasting.

## 4. Tool & Model Routing

Use a fast/medium model for:

- Minor syntax fixes and formatting
- Boilerplate generation and scaffolding
- Basic unit tests and simple, single-file edits

Escalate to a deep-reasoning model for:

- Complex orchestration or control-flow logic
- Heavy multi-file debugging and refactors
- Architectural decisions and cross-module design

When in doubt, start with the cheaper model and escalate only if the task proves
to need deeper reasoning.
