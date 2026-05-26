---
name: review-checklist
description: Use during Claude Code reviews to classify API, IAM, cleanup, exception, concurrency, config, temp-file, and version-pin risks.
last_verified: 2026-05-26
owner: platform
version: 1.1.0
---

# Review Checklist

Prioritize findings that can break users, expand permissions, hide failures, or make parallel agent runs unsafe.

- API contract safety: are new endpoints backward compatible? Does removing a field break existing callers?
- Silent IAM expansion: any change that quietly adds new IAM permissions is a HIGH finding. It must be explicit and operator-approved.
- Exception handling: bare `except:` or overly broad `except Exception:` that discards traceback context is a MEDIUM finding. Prefer typed exceptions with `raise ... from e`.
- Cleanup safety: cleanup code must be best-effort. Use `try/finally`; never let cleanup raise and abort the cleanup sequence. `also_teardown_controller=False` is the established safe default for SkyPilot.
- Narrow exception paths: test coverage for failure paths such as submit failure, auth failure, and cleanup failure is required. Absence is a MEDIUM finding.
- Concurrent run safety: does the code handle parallel Codex runs safely? Check for file ownership assumptions and missing commit-lock patterns.
- Config injection consistency: config should resolve through one precedence-ordered path: explicit arg, then env var, then config file. Inconsistent resolution is a MEDIUM finding.
- Temp file leaks: `tempfile.mkdtemp` without cleanup on exception paths is a MEDIUM finding. Prefer `TemporaryDirectory` context manager.
- Version pins: required dependency versions should be asserted at runtime, not just documented.
- Missing skill delta: when a PR touches code under a skill's `applies_to` paths but the PR description does not record a self-review outcome (per `skill-curation`), this is a MEDIUM finding. When the diff visibly contradicts the skill body and no delta was filed, it is a HIGH finding.
- Stale skill: when a touched skill's `last_verified` is older than 30 days and the PR did not update it, this is a LOW finding; remind the author to run the drift checklist.

## Curation Duty

Reviews after 3+ commits or any blocker NOVEL_ISSUE also perform skill curation per `skill-curation`. Walk `/tmp/<run-id>/skill-deltas.md` and `novel-issues.md`, then promote / drop / escalate each entry.

## Changelog

- 2026-05-26: Added missing-skill-delta and stale-skill findings; documented curation duty during review passes.

