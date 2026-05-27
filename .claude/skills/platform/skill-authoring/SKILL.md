---
name: skill-authoring
description: Use when creating or editing any SKILL.md in this repo; defines the frontmatter contract, required sections, and mirroring rules between .agents/ and .claude/.
last_verified: 2026-05-27
owner: platform
version: 2.0.0
applies_to:
  - .agents/skills/**
  - .claude/skills/**
---

# Skill Authoring (stub)

The canonical content lives at
[.agents/skills/platform/skill-authoring/SKILL.md](../../../../.agents/skills/platform/skill-authoring/SKILL.md).

Both Codex and Claude Code load from the canonical file. Edit it there; do not
fork the content into this stub. The stub exists only so Claude Code can
discover the skill from its surface.

## Changelog

- 2026-05-27: Replaced mirrored copy with a stub pointing at the canonical
  `.agents/` file to eliminate manual mirror drift.
