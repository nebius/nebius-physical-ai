---
name: super-prompt-patterns
description: Use when drafting, executing, or reviewing Codex super-prompts for this repository.
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

Log novel issues to `/tmp/<run-id>/novel-issues.md`. Use log/skip/continue for non-blocking issues and halt only for true blockers.

## Commit Lock

Acquire `/tmp/npa-commit-lock/<scope>` during commit phases for parallel runs that share files.

## Broken CLI Workflow

If the `npa` CLI is broken during a run, document the breakage as a structured prompt, run Codex to fix it, then use the fixed CLI.

## Autonomous Run Rhythm

Use separate agents by phase: Build with Codex, verify with Codex in read-only mode, review with Claude Code for architecture, then cleanup with Codex in a scoped pass.

Trigger a Claude Code review after 3 or more commits land. The CC agent must differ from the building agent.

Root `AGENTS.md` is a lightweight index; details belong in skill files.
