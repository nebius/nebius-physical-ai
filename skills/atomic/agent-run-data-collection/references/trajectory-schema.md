# `npa.agent.trajectory.v1`

Emit one JSON object per terminal goal-level episode. A session contains one or
more episodes, and an episode contains ordered events. This is a semantic
contract; an implementation may use stricter typed models.

## Required shape

```json
{
  "schema_version": "npa.agent.trajectory.v1",
  "episode_id": "stable-episode-id",
  "session_id": "parent-session-id",
  "scope": {
    "tenant_id": "runtime-resolved",
    "dataset_role": "agent-finetuning-raw"
  },
  "timing": {
    "started_at": "RFC3339",
    "ended_at": "RFC3339",
    "latency_ms": 0
  },
  "request": {
    "content": "sanitized user request",
    "intent": "resolved intent"
  },
  "initial_state": {},
  "trajectory": [
    {
      "sequence": 0,
      "phase": "plan|tool|observation|confirm|final",
      "tool": "",
      "arguments": {},
      "observation": {},
      "status": "ok|error|rejected|cancelled"
    }
  ],
  "outcome": {
    "status": "succeeded|failed|refused|cancelled",
    "verified": false,
    "verified_by": [],
    "artifact_uris": [],
    "operator_interventions": [],
    "preference_pairs": []
  },
  "routing": {
    "grounded": false,
    "tier": "",
    "model": "",
    "input_tokens": 0,
    "output_tokens": 0
  },
  "versions": {
    "agent": "",
    "tools": {}
  },
  "redaction": {
    "applied": true,
    "fields_removed": []
  },
  "collection": {
    "status": "collected|pending|disabled",
    "content_sha256": ""
  }
}
```

## Invariants

- `episode_id` is assigned when the goal is accepted, before its first plan or tool call.
- `session_id` is stable across episodes in the same conversation or operator work period.
- `trajectory.sequence` is strictly increasing and preserves retries.
- `outcome.verified=true` requires at least one objective `verified_by` entry.
- A grounded reply records zero model tokens; missing usage is not fabricated as zero.
  Unknown `routing.input_tokens` and `routing.output_tokens` may be omitted or
  explicitly `null`. Observed counts are integers; booleans, strings and floats
  are invalid token counts.
- Store large artifacts by sanitized URI and digest, never inline their bytes.
- `content_sha256` is computed over canonical JSON with that field empty.
- Collection metadata must not claim `collected` until S3 read-after-write verifies
  the uploaded bytes. The immutable raw record is frozen with
  `collection.status=pending`; its status and content hash do not change on retry.
  A separate immutable delivery receipt establishes collection after read-back.
- A curated training example references its raw source episode and dataset version;
  it does not replace or mutate this raw record.

## Immutable delivery metadata

The runtime writes raw episodes at the prescribed
`episodes/<yyyy>/<mm>/<dd>/<episode-id>-<content-sha256>.json` key. It conditionally
creates an episode claim under `episode-index/<sha256-of-episode-id>.json`, bound
to the episode content hash. Another digest for the same episode leaves both raw
objects visible and reports a data-quality conflict; it cannot overwrite the
claim or receive a successful delivery receipt.

After verifying raw bytes and the episode claim, the runtime conditionally writes
and reads back a receipt at
`receipts/<sha256-of-episode-id>/<content-sha256>.json`:

```json
{
  "schema_version": "npa.agent.trajectory-receipt.v1",
  "episode_id": "stable-episode-id",
  "content_sha256": "canonical-content-hash",
  "payload_sha256": "sha256-of-exact-uploaded-bytes",
  "status": "collected"
}
```

`content_sha256` retains the existing rule of hashing canonical JSON with that
field empty. `payload_sha256` hashes the exact final serialized bytes, including
the filled content hash. The emitter returns `collected` only after both raw
payload and receipt read-back succeed. Failures preserve a pending private
outbox envelope containing the unchanged payload, its exact-byte digest, and a
digest binding the original tenant/bucket/prefix. Destination or digest mismatch
must cause retention without writes or scope rewriting. No dataset URI or bucket
name is serialized into the outbox envelope.
