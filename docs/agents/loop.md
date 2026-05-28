# Self-Improvement Loop

Skills under `.agents/skills/` and `.claude/skills/` are living documents.
Every agent run feeds the next through a small capture → triage → promote
cycle. This page is the single source of truth; `AGENTS.md` and `CLAUDE.md`
link here instead of restating it.

## End-to-end Flow

1. **Phase 0 (builder)** — set a `run-id` (`YYYYMMDDThhmmssZ-<slug>`) and
   export it as `NPA_RUN_ID`. Every subsequent phase reuses the same value.
   See [.agents/skills/platform/super-prompt-patterns/SKILL.md](../../.agents/skills/platform/super-prompt-patterns/SKILL.md).
2. **During the run (builder)** — log surprises to
   `/tmp/<run-id>/novel-issues.md` using the six-field NOVEL_ISSUE template
   (`skill`, `trigger`, `observation`, `resolution`, `propose`, `severity`).
   When the proposed edit is concrete and high-confidence, also append it to
   `/tmp/<run-id>/skill-deltas.md`.
3. **End of Phase L (builder)** — persist both files into
   `.agents/runs/<run-id>/` and commit them in the same final commit. The
   reviewer cannot see `/tmp`; the repo-visible copy is the durable handoff.
   If a file is empty, write a single line `none` so absence is intentional.
4. **Before opening a PR (builder)** — if the diff touches code under a
   skill's `applies_to` paths, read the skill, confirm reality still matches,
   and note the self-review outcome in the PR description.
5. **Review pass (reviewer, different agent from builder)** — read
   `.agents/runs/<run-id>/` and triage each candidate per
   [.agents/skills/platform/skill-curation/SKILL.md](../../.agents/skills/platform/skill-curation/SKILL.md):
   - **Promote** — edit the target `SKILL.md`, add a `## Changelog` entry,
     bump `version`, update `last_verified`.
   - **Drop** — append a one-line reason to
     [.agents/curation-log.md](../../.agents/curation-log.md) so the same
     idea is not re-proposed.
   - **Escalate** — add a bullet to the skill's `## Open Questions`.
6. **Cadence** — trigger curation after any of: 3+ commits since the last
   pass, a NOVEL_ISSUE with `severity: blocker`, or any skill whose
   `last_verified` is older than 30 days.

## Roles

- **Codex** is the default builder. It captures NOVEL_ISSUEs and emits
  skill-deltas during runs.
- **Claude Code** is the default reviewer/curator. It triages
  `.agents/runs/<run-id>/` during its review pass.
- The reviewer MUST differ from the builder for any given run.

## File Map

| Path | Purpose |
| --- | --- |
| [AGENTS.md](../../AGENTS.md) | Codex-loaded index. Lightweight; points at skills. |
| [CLAUDE.md](../../CLAUDE.md) | Claude-Code-loaded index. Lightweight; points at skills. |
| [.agents/skills/](../../.agents/skills/) | Canonical skill files. Edit here. |
| [.claude/skills/](../../.claude/skills/) | Claude-side skills. The `skill-authoring` and `skill-curation` entries are stubs that point back at `.agents/`. |
| [.agents/runs/](../../.agents/runs/) | Per-run durable handoff: `novel-issues.md` and `skill-deltas.md`. |
| [.agents/curation-log.md](../../.agents/curation-log.md) | Append-only ledger of curation decisions (promoted / dropped / escalated). |

## When You Are A Contributor (Not An Agent)

If you are a human editing code under a skill's `applies_to` paths:

1. Read the relevant skill before changing the code.
2. If your change makes the skill wrong, update the skill in the same PR.
   Bump `last_verified` and `version`, add a `## Changelog` line.
3. Note the self-review outcome in the PR description, the same way an agent
   would.

This keeps the skills usable for the next agent (or contributor) who needs
them.
