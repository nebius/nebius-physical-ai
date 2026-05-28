# Run Logs

Each subdirectory is a `run-id` (format `YYYYMMDDThhmmssZ-<slug>`, set as `NPA_RUN_ID` at Phase 0). The building agent persists the run's `novel-issues.md` and `skill-deltas.md` here at end of Phase L so the reviewer/curator can read them.

See `.agents/skills/platform/super-prompt-patterns/SKILL.md` ("Durable Handoff to the Reviewer") and `.agents/skills/platform/skill-curation/SKILL.md` ("Inputs") for the full protocol.

After a curator triages a run, the directory MAY be deleted in a follow-up commit. The audit trail survives in `.agents/curation-log.md` and the affected skills' `## Changelog` sections.
