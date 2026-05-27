---
name: skill-curation
description: Use when triaging novel-issue logs, promoting skill deltas, or running periodic drift checks on SKILL.md files.
last_verified: 2026-05-26
owner: platform
version: 1.1.0
applies_to:
  - .agents/skills/**
  - .claude/skills/**
  - .agents/runs/**
  - .agents/curation-log.md
---

# Skill Curation

Captures the loop that turns ephemeral lessons logged during runs into durable updates in `SKILL.md` files. This is the back half of the self-improvement loop; the front half lives in `super-prompt-patterns` (capture during a run).

## When to Triage

Trigger a curation pass when any of the following holds:

- Three or more commits have landed since the last pass.
- Any NOVEL_ISSUE in the last run window has `severity: blocker`.
- Any skill's `last_verified` is older than 30 days.

The triage agent MUST differ from the agent that built the changes (mirrors the builder/verifier/reviewer rhythm in `super-prompt-patterns`). Codex emits deltas during runs; Claude Code triages during its review pass.

## Inputs

The curator reads only from repo-visible locations; `/tmp/<run-id>/` is local to the building agent and is NOT a valid source for triage.

- `.agents/runs/<run-id>/novel-issues.md` — the building agent's raw observations, persisted at end of Phase L (see `super-prompt-patterns` "Durable Handoff to the Reviewer").
- `.agents/runs/<run-id>/skill-deltas.md` — high-confidence proposed edits with a target skill identified, also persisted by the builder.
- Recent commit diffs touching paths covered by any skill's `applies_to`.

If the expected log files are missing for a run that produced commits, that is itself a curation finding: open an Open Question on `super-prompt-patterns` and remind the builder to persist next time.

## Loop: Triage → Promote → Drop → Escalate

For each candidate delta:

1. **Triage**: dedupe across runs. Group by target skill. Identify duplicates that already appear in the skill body or `Changelog`.
2. **Promote**: edit the target `SKILL.md` body, add a `## Changelog` entry (newest first, dated), bump `version` (PATCH/MINOR/MAJOR per `skill-authoring`), and update `last_verified` to today.
3. **Drop**: if rejected, append a one-line entry to `.agents/curation-log.md` with date, brief summary, and reason. This prevents re-proposal.
4. **Escalate**: if ambiguous, add a bullet to the target skill's `## Open Questions` section (create the section if missing) and leave it for the next pass or a human.

Every candidate MUST land in exactly one of Promote / Drop / Escalate. Nothing is silently dropped.

After all candidates from a `.agents/runs/<run-id>/` directory are resolved, the curator MAY delete the directory in a follow-up commit. The audit trail survives in `.agents/curation-log.md` and the relevant skills' `## Changelog` sections.

## Self-Review Before PR

Before opening a PR that touches code under any skill's `applies_to` paths, the building agent MUST:

1. Read the relevant skill.
2. Confirm reality still matches; if not, emit a `skill-deltas.md` entry and persist it under `.agents/runs/<run-id>/` per `super-prompt-patterns`.
3. Note the self-review outcome in the PR description ("skill X reviewed, no drift" or "skill X delta filed under .agents/runs/<run-id>/").

A missing self-review is itself a `review-checklist` finding.

## Drift Checks (5-minute pass per skill)

Run the following on every skill touched by recent commits, and on any skill whose `last_verified` is older than 30 days:

1. **CLI surface**: every `npa <cmd>` referenced still resolves under `npa --help`.
2. **File paths**: every relative path mentioned still exists in the tree.
3. **Version pins**: any pinned package versions match `npa/requirements-lock.txt`.
4. **Infra facts**: any cluster, region, namespace, or storage-endpoint mention matches `nebius-infra` (the single source of truth).
5. **Mirror parity**: if the skill exists in both `.agents/` and `.claude/`, the two files agree.

On any drift, either fix inline (bump version, dated changelog entry) or file an Open Question.

## Curation Log Ledger

`.agents/curation-log.md` is append-only. One line per decision: date, target skill, action (`promoted` / `dropped` / `escalated`), short reason. Used to prevent re-proposing rejected deltas and to give the next curator an audit trail.

## Changelog

- 2026-05-26: Inputs read from `.agents/runs/<run-id>/` (repo-visible) instead of `/tmp/<run-id>/`; documented missing-logs as a curation finding; allowed the curator to delete the run directory after triage.
- 2026-05-26: Initial version. Defines triage/promote/drop/escalate loop, drift checklist, and curation log conventions.
