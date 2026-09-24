---
name: contributions
description: Write or review NPA contributions, including workbench agent changes, using the repository's readability, documentation, and anti-pattern rules.
---

# Contributions

Apply these rules to new and changed code when implementing or reviewing an NPA
contribution. Keep the change scoped to the requested behavior; existing
violations elsewhere are not a reason for a repository-wide rewrite.

Read `AGENTS.md`, applicable nested instructions, and `CONTRIBUTING.md` before
editing. Use `skills/index.yaml` to load the relevant tool or workflow skill.
This skill defines contribution quality; it does not authorize unrelated
operations or publication.

## Readability and code style

- Keep functions under 40 lines. Extract named helpers when a function would
  reach 40 lines or more.
- Name things after domain concepts, not types: `retryBudget`, not `intValue2`.
  Follow the language's existing casing convention, such as `retry_budget` in
  Python.
- Avoid clever one-liners. Prefer three obvious statements over one dense
  expression.
- Do not abbreviate names except for widely known terms such as `id`, `url`,
  and `http`.
- Comments explain why, never what. Delete comments that restate the code.
- Prefer early returns over nested conditionals. Keep nesting depth at most 3.

## Documentation

- Give every exported symbol a docstring with a one-line summary and `Args`,
  `Returns`, and `Raises` sections. Use the language's equivalent documentation
  comment where it has no docstring syntax. State `None` for inapplicable
  sections rather than inventing arguments, results, or exceptions.
- Start every new module with a header comment stating its responsibility in
  one sentence. A Python module docstring serves as that header.
- Update the relevant README whenever adding a command, environment variable,
  or configuration key. Explain what it does, how to use it, and any default
  or required value; link to detailed reference documentation when needed.
- Write for someone who joined last week and has not read the rest of the
  repository. Introduce domain concepts and show the context needed to use the
  changed behavior.

## Anti-patterns

- Do not add a `utils.py` or `helpers.ts` dumping ground. Put shared behavior in
  a module named for its domain responsibility.
- Do not generate defensive `try`/`except` around code that cannot fail. Handle
  actual failure boundaries with the exceptions the operation can raise.
- Do not leave TODO comments. Implement the work or open an issue within the
  task's existing authorization; a TODO is not a substitute for either.

## Before handing off

Review the full scoped diff against every rule above, including exported
symbols, new modules, and README coverage. Check that extracted helpers clarify
the domain rather than merely moving dense code elsewhere.

Use `skills/atomic/testing-conventions/SKILL.md` and
`skills/atomic/pre-pr-validation/SKILL.md` to select and run the required checks
with `npa/.venv/bin/python`. Prove the changed behavior and relevant failure
paths with appropriate tests or real execution evidence. Do not weaken checks
to obtain a pass or claim that style review proves security.

Before publishing code, documentation, or review evidence, follow
`skills/atomic/protect-nebius-infra-details/SKILL.md`. Report what changed,
validation outcomes, and concrete limitations so another contributor can assess
the result without reading the original conversation.
