---
name: agent-run-data-collection
description: Use when collecting NPA agent work for fine-tuning or evaluation so every goal-level episode emits a sanitized, outcome-linked trajectory to an append-only S3 dataset in an operator-specified Nebius tenant and bucket.
---

# Agent Run Data Collection

Collect complete, replayable agent trajectories without making telemetry a new
source of fabricated success or lost operator work.

## Required runtime configuration

Resolve these from owner-only runtime configuration; never commit their values:

- `NPA_AGENT_DATASET_TENANT_ID`: exact tenant that owns the destination.
- `NPA_AGENT_DATASET_URI`: exact `s3://<bucket>/<prefix>` dataset root.
- Existing NPA S3 endpoint and credentials for that bucket. Do not copy secrets
  into a trajectory, command line, workflow spec, log, or skill file.

Before enabling collection, verify the active agent deployment reports the same
tenant and that the destination bucket is a writable resource in that tenant.
Tenant-wide discovery alone does not authorize cross-project writes. Do not
create a bucket, grant IAM, or change tenant/project configuration unless the
operator separately requests it.

## Collection hierarchy and episode boundary

Use three levels:

- **Session:** the whole conversation or operator work period.
- **Episode:** one accepted goal pursued until objective success, failure,
  refusal, cancellation, or an explicit handoff. Follow-up messages that refine
  the same unfinished goal stay in the episode; a materially new goal starts a
  new episode.
- **Event:** each prompt, plan, model decision, tool call, observation,
  confirmation, retry, correction, or final response inside an episode.

The episode is the dataset row. Assign its id when the goal is accepted and
finalize exactly one `npa.agent.trajectory.v1` record at the terminal boundary.
Link it to its parent session and include all nested events; do not create a
separate dataset row for every chat turn or tool call.

## Preamble integration

Link the skill explicitly from the agent preamble. Use this canonical line:

> For every goal-level episode, load and follow `$agent-run-data-collection` at `skills/atomic/agent-run-data-collection/SKILL.md`; record the episode from goal acceptance through success, failure, refusal, cancellation, or handoff as one sanitized trajectory containing all nested events, linked to its parent session and stored using `NPA_AGENT_DATASET_TENANT_ID` and `NPA_AGENT_DATASET_URI`.

The preamble selects the skill; the runtime emitter still must be implemented
and configured. Do not claim collection is active merely because this sentence
is present.

## Write contract

Write a new immutable object for every episode:

```text
<dataset-root>/episodes/<yyyy>/<mm>/<dd>/<episode-id>-<content-sha256>.json
```

Never append by rewriting one shared JSONL object: concurrent S3 writers lose
rows. Retrying the same finalized payload must resolve to the same object key;
different payloads for one episode id remain visible and must be treated as a data
quality conflict rather than overwritten.

If the S3 write fails, persist the same finalized record to an owner-only local
outbox and mark collection `pending`. Never report the record as collected until
a read-after-write check confirms the object hash. Do not fail or repeat an
otherwise completed GPU or destructive operation merely because telemetry
failed. Retry pending records at the start of the next episode or through an
explicit flush.

## Record contents

Read [references/trajectory-schema.md](references/trajectory-schema.md) before
implementing or changing the emitter. Preserve:

- sanitized request and relevant initial state;
- ordered plans, tool calls, arguments, observations, retries, and confirmations;
- objective outcome evidence, produced artifact URIs, and operator corrections;
- grounded/model routing, model name, token usage, latency, and agent/tool versions;
- tenant scope, dataset URI role, timestamps, redaction status, and collection status.

Use live workflow status, exit codes, validation reports, tests, and artifact
existence as outcome evidence. The agent's own final sentence is not proof of
success. Store operator corrections as explicit preference pairs when both the
rejected and corrected actions are available.

## Privacy and integrity

Redact before serialization and again before upload. Remove credentials,
authorization headers, environment dumps, private keys, signed URLs, raw image
data, and secret-shaped values. Replace concrete infrastructure identifiers in
free text with typed references; retain the configured tenant id only in the
access-controlled scope field needed to prove routing. Keep raw and curated
datasets separate, and never train on evaluation or held-out runs.

At completion, surface only `collected`, `pending`, or `disabled` plus the episode
id. Do not print the tenant id, bucket name, credentials, or full destination
URI in ordinary handoffs.

## Verification

For emitter changes, test at least:

1. success, tool failure, refusal, cancellation, and grounded zero-token runs;
2. redaction before either S3 or outbox writes;
3. deterministic keys and idempotent retries;
4. concurrent runs producing distinct immutable objects;
5. tenant/bucket mismatch failing before collection is enabled;
6. S3 failure entering the outbox and later flushing with read-after-write proof.

Use `npa/.venv/bin/python` for repository validation. Load
`skills/atomic/testing-conventions/SKILL.md` when implementing the emitter and
`skills/atomic/protect-nebius-infra-details/SKILL.md` before publishing evidence.
