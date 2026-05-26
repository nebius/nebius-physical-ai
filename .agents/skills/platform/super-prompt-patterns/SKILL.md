---
name: super-prompt-patterns
description: Use when drafting, executing, or reviewing Codex super-prompts for this repository.
last_verified: 2026-05-26
owner: platform
version: 1.1.0
---

# Super-Prompt Patterns

## Dirty Tree

Use `RELAXED_DIRTY_TREE_MODE`: halt only when dirty files overlap with the run's target paths. Pre-existing dirty files in unrelated paths never halt the run.

## Budgets

Do not add time, cost, or job-count limits unless the operator explicitly specifies them. Strip per-phase time budgets, wall-clock caps, and GPU spend limits from prompts.

## Phase Shape

Standard phase flow:

1. Phase 0: state check, read-only.
2. Phase A: design.
3. Phase B-G: implementation.
4. Phase L: final report.

## NOVEL_ISSUE Protocol

During a run, log novel issues to `/tmp/<run-id>/novel-issues.md`. Use log/skip/continue for non-blocking issues; halt only for true blockers.

Each entry MUST use this structured template:

```
- skill: <target SKILL.md path, or `unknown`>
  trigger: <what the agent was doing>
  observation: <what surprised the agent>
  resolution: <how the agent handled it in-run>
  propose: <concrete edit suggestion, or `investigate`>
  severity: info | gotcha | blocker
```

When the proposal is concrete and high-confidence (you would commit the edit yourself), additionally append it to `/tmp/<run-id>/skill-deltas.md`. That second log feeds the curation loop in `skill-curation` directly; entries left only in `novel-issues.md` get triaged but require more thought.
 The same review pass performs skill curation per `skill-curation`.

Root `AGENTS.md` is a lightweight index; details belong in skill files.

## Changelog

- 2026-05-26: Restructured `NOVEL_ISSUE` into a six-field template; added `/tmp/<run-id>/skill-deltas.md` log; added explicit capture triggers; linked to `skill-curation`.

Log a NOVEL_ISSUE whenever any of these happens during a run:

- Validation, test, or CLI command fails for a reason not covered by an existing skill.
- GPU routing or cluster placement behaves differently than the skill claims.
- Documentation, CLI help, or code drifts from the relevant skill.
- The same fix is applied more than once in the same run (extract a convention).
- A missing convention forces an ad-hoc decision the next agent will face again.
- Two skills contradict each other.

## Commit Lock

Acquire `/tmp/npa-commit-lock/<scope>` during commit phases for parallel runs that share files.

## Broken CLI Workflow

If the `npa` CLI is broken during a run, document the breakage as a structured prompt, run Codex to fix it, then use the fixed CLI.

## Autonomous Run Rhythm

Use separate agents by phase: Build with Codex, verify with Codex in read-only mode, review with Claude Code for architecture, then cleanup with Codex in a scoped pass.

Trigger a Claude Code review after 3 or more commits land. The CC agent must differ from the building agent.

Root `AGENTS.md` is a lightweight index; details belong in skill files.
