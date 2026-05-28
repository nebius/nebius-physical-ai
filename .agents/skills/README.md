# Skills

This directory holds the canonical agent skills for the repo. Each
`SKILL.md` is a focused, versioned reference for one topic that Codex (and,
via stubs, Claude Code) loads on demand.

For the end-to-end self-improvement loop (capture, triage, promote, drop,
escalate), see [docs/agents/loop.md](../../docs/agents/loop.md).

## Layout

```
.agents/skills/
  platform/    # cross-cutting concerns (auth, curation, testing, infra, ...)
  workbench/   # one skill per workbench tool
```

A skill lives under `platform/` if it applies across tools or describes
agent process. Otherwise it lives under `workbench/<tool>/`.

## Lifecycle

Every skill is in exactly one of three states, indicated by the
`description` line and by the `## Changelog`:

- **draft** — newly written, not yet verified against reality. Description
  should begin "Draft: use when ...". Treat its claims with care.
- **active** — verified within the last 30 days. The default state.
  `last_verified` is the source of truth.
- **deprecated** — content kept for history but no longer authoritative.
  Description should begin "Deprecated: see <other skill>". Curators do not
  bump `last_verified` on deprecated skills; they archive them.

A skill becomes **stale** (not a separate state, but a curation finding)
when `last_verified` is older than 30 days. Stale skills get the drift
checklist on the next curation pass.

## Authoring A New Skill

1. Read [platform/skill-authoring/SKILL.md](platform/skill-authoring/SKILL.md)
   for the frontmatter contract and required sections.
2. Pick the smallest scope that is still useful. A skill that tries to
   cover too much will not be read.
3. Add the new bullet to [AGENTS.md](../../AGENTS.md). If Claude Code also
   needs it, add a stub under `.claude/skills/<category>/<name>/SKILL.md`
   that points back here (see the existing `skill-authoring` and
   `skill-curation` stubs).

## Editing An Existing Skill

1. Make the content change.
2. Bump `version` (`PATCH` for typo, `MINOR` for new content, `MAJOR` for
   contract change).
3. Update `last_verified` to today.
4. Add a dated line at the top of `## Changelog`.

## Asking For A New Skill (Humans)

Open a GitHub issue titled `skill: <topic>` describing what conventions
are currently undocumented and where they came from (which runs, which
files). The next curation pass will either create the skill or extend an
existing one.

## Why The Stubs Under `.claude/`

`skill-authoring` and `skill-curation` apply to both Codex and Claude
Code. To avoid maintaining two copies that drift, the canonical files live
here and the `.claude/` copies are short stubs that point back. Edit only
the canonical files.
