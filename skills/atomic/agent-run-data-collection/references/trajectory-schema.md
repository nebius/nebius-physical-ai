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
- Store large artifacts by sanitized URI and digest, never inline their bytes.
- `content_sha256` is computed over canonical JSON with that field empty.
- Collection metadata must not claim `collected` until S3 read-after-write verifies
  the uploaded bytes.
- A curated training example references its raw source episode and dataset version;
  it does not replace or mutate this raw record.
