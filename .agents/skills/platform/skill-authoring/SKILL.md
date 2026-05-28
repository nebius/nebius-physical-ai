---
name: skill-authoring
description: Use when creating or editing any SKILL.md in this repo; defines the frontmatter contract, required sections, and mirroring rules between .agents/ and .claude/.
last_verified: 2026-05-26
owner: platform
version: 1.0.0
applies_to:
  - .agents/skills/**
  - .claude/skills/**
---

# Skill Authoring

This skill is the canonical template for every other `SKILL.md` in the repo. Treat it as the source of truth; if another skill drifts from this contract, fix the drift as part of normal curation.

## Frontmatter Contract

Every `SKILL.md` MUST have YAML frontmatter with these keys:

- `name`: short kebab-case identifier matching the directory name.
- `description`: one sentence starting with "Use when ..." so the agent knows when to load it.
- `last_verified`: ISO date (`YYYY-MM-DD`) the skill was last reviewed against reality. Bump on any curated change.
- `owner`: `platform`, `workbench`, or a specific human/role. Identifies who to ping on contradiction.
- `version`: semver-ish (`MAJOR.MINOR.PATCH`). Bump PATCH on typo/clarification, MINOR on new content, MAJOR on contract change.

Optional:

- `applies_to`: glob list of repo paths the skill governs. Used by curators to flag drift when those paths change without a skill update.

## Required Sections

- `# <Title>` — H1 matching the topic.
- Body — facts, conventions, examples. Short bullets over prose.
- `## Changelog` — append-only, newest first, one dated line per curated change. Required on every skill.
- `## Open Questions` — optional, included only when there are unresolved ambiguities. Remove the section when empty.

## Style

- Imperative voice. "Use X" not "You should use X".
- One fact per bullet. No paragraphs longer than four lines.
- Link to code or docs by relative repo path; do not paste large code blocks.
- Never hardcode tenant IDs, project IDs, bucket names, or secrets.

## Mirroring Between Surfaces

- `.agents/` is the Codex surface; `.claude/` is the Claude surface. Some skills live on only one surface; the two `skill-authoring` and `skill-curation` skills are mirrored on both.
- When a mirrored skill changes on one side, mirror the change on the other side in the same commit. Drift between mirrors is a curation finding.

## Changelog

- 2026-05-26: Initial version. Defines frontmatter contract, required sections, and mirroring rules.
