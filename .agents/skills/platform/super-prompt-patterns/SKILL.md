---
name: super-prompt-patterns
description: Use when drafting, executing, or reviewing Codex super-prompts for this repository.
last_verified: 2026-05-26
owner: platform
version: 1.2.0
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

## Run ID

Every run MUST have a `run-id`. Format: `YYYYMMDDThhmmssZ-<short-slug>` (UTC timestamp + a short kebab-case slug describing the run, e.g. `20260526T154512Z-self-improvement`). The orchestrator or operator sets it at the start of Phase 0 and exports it as `NPA_RUN_ID`. Every subsequent phase MUST reuse the same value. Parallel runs use distinct run-ids so their logs and commit-locks do not collide.

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

### Capture Triggers

Log a NOVEL_ISSUE whenever any of these happens during a run:

- Validation, test, or CLI command fails for a reason not covered by an existing skill.
- GPU routing or cluster placement behaves differently than the skill claims.
- Documentation, CLI help, or code drifts from the relevant skill.
- The same fix is applied more than once in the same run (extract a convention).
- A missing convention forces an ad-hoc decision the next agent will face again.
- Two skills contradict each other.

### Durable Handoff to the Reviewer

`/tmp/<run-id>/` is local and ephemeral; the reviewer cannot see it. At the end of Phase L (final report), the building agent MUST persist the run's logs into the repo so the next agent can pick them up:

1. Create `.agents/runs/<run-id>/` (gitignore does not exclude this path).
2. Copy `novel-issues.md` and `skill-deltas.md` from `/tmp/<run-id>/` into that directory. If either file is empty or absent, write the file with a single line `none` so absence is intentional rather than ambiguous.
3. Stage and commit the run directory in the same final commit as the rest of Phase L's output. The commit message MUST reference `run-id: <run-id>`.
4. If the run produced no commit (read-only verification), open a tiny housekeeping commit containing only `.agents/runs/<run-id>/` on the same branch.

Reviewers and curators read from `.agents/runs/<run-id>/`, never from `/tmp`. Once a run's deltas have been triaged per `skill-curation`, the curator may delete `.agents/runs/<run-id>/` in a follow-up commit; the curation decisions live on in `.agents/curation-log.md` and the skills' `## Changelog` sections.

## Commit Lock

Acquire `/tmp/npa-commit-lock/<scope>` during commit phases for parallel runs that share files.

## Broken CLI Workflow

If the `npa` CLI is broken during a run, document the breakage as a structured prompt, run Codex to fix it, then use the fixed CLI.

## Autonomous Run Rhythm

Use separate agents by phase: Build with Codex, verify with Codex in read-only mode, review with Claude Code for architecture, then cleanup with Codex in a scoped pass.

Trigger a Claude Code review after 3 or more commits land. The CC agent must differ from the building agent. The same review pass performs skill curation per `skill-curation`.

Root `AGENTS.md` is a lightweight index; details belong in skill files.

## Changelog

- 2026-05-26: Defined `run-id` format + `NPA_RUN_ID` env var; added "Durable Handoff to the Reviewer" requiring logs to be persisted at `.agents/runs/<run-id>/` so the reviewer can read them; reordered Capture Triggers back under the NOVEL_ISSUE Protocol section.
- 2026-05-26: Restructured `NOVEL_ISSUE` into a six-field template; added `/tmp/<run-id>/skill-deltas.md` log; added explicit capture triggers; linked to `skill-curation`.

