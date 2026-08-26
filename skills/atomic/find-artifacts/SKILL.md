---
name: find-artifacts
description: Use when discovering or loading run artifacts in npa agent without workflow/type/path allowlists.
---

# Find Artifacts (artifact-first)

Use this skill when the operator asks:

- "what can I view?"
- "show artifacts for this run"
- "load this recording/video/report"
- "browse outputs from storage"

## Model

Artifact-first means:

1. Enumerate what exists in storage.
2. Group by discovered run prefix.
3. Classify each object with a render hint.
4. Never hide unknown formats; degrade to download.

No workflow registry, path allowlist, or known-type gate is required.

## Agent API flow

All calls are same-origin (`/api/...`) on the authenticated agent VM.

1. Discover runs:

```http
GET /api/artifacts/runs?prefix=&limit=100
```

2. List artifacts for one run:

```http
GET /api/artifacts/run/{run_id}
```

3. Load one explicit artifact:

```http
POST /api/sim-viz/load-artifact
{
  "run_id": "run-prefix",
  "run_ref": "<server-issued opaque run reference>",
  "project_id": "<selected project>",
  "resource_bucket": "<selected bucket>",
  "resolved_prefix": "<selected prefix, possibly empty>",
  "source_selected": true,
  "key": "path/to/run/object"
}
```

Preserve this exact source tuple from list response to card, thumbnail, preview,
download, and load action. `s3_uri` is provenance only: URI-only or otherwise
unscoped media requests are rejected with a stable 400 response because bucket
membership alone is not run-level authorization. Follow every artifact-page
cursor before computing preference; the shipped UI does this automatically.

4. Confirm loaded state:

```http
GET /api/sim-viz/status
```

Check `artifact_key`, `artifact_render`, `artifact_uri`, and `rerun_ready`.

## Render hints

- `rerun` (`.rrd`)
- `video` (`.mp4`, `.webm`, `.mov`)
- `image` (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`)
- `json`
- `text`
- `download` (fallback for unknown/new types)

Unknown types must stay visible/selectable.

## Safety

- Validate run ids (`validate_run_id`) before listing/loading.
- Reject traversal keys (`..`, empty segments).
- Surface S3 failures directly (`ok: false` or error detail); do not claim
  success when load/list fails.
- Agent artifact discovery is S3-first and S3-only for run artifacts. The agent
  must stage the configured artifact bucket credentials, upload completed run
  artifacts there, and use `/api/artifacts/runs`, `/api/artifacts/run/{run_id}`,
  and `/api/sim-viz/load-artifact` against that S3 data. Do not silently fall
  back to `/opt/npa-agent/runs` for user-visible discovery.
