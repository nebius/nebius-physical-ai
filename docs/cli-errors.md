# NPA CLI Error Reference

The `npa` CLI formats known serverless failures as actionable errors instead
of raw Python exceptions. Typed serverless errors exit with code `1`.
Unexpected errors exit with code `2` and hide stack traces unless
`NPA_DEBUG=1` is set.

## Capacity and quota errors

Capacity failures surface as `NotEnoughResources`; quota failures surface as
`Quota`. Text output includes the project, platform, preset, GPU count when
known, and suggested alternatives.

```text
Error: Not enough resources to schedule this request.

  Project: project-1
  Platform: gpu-h200-sxm
  Preset: 8gpu-128vcpu-1600gb
  GPU count: 8

  Cause: capacity (capacity blocked)

  Try one of:
    - Retry in a few minutes
    - Reduce gpu-count
    - Try a different gpu-type (e.g., l40s)
    - Try a different project

  See: docs/cli-errors.md
```

Quota errors use the same shape, with `error_class: quota` in JSON and quota
specific alternatives such as requesting a quota increase or using a different
project.

## Authentication errors

Authentication or authorization failures are not capacity failures. They keep
their own `Auth` error type and include a recovery hint.

```text
Error: Nebius authentication failed.

  Cause: permission denied
  Hint: Run `nebius profile create` or refresh Nebius credentials.
```

## Not-found errors

Missing serverless resources surface as `EndpointNotFound`. When available, the
error includes `project_id`, `endpoint_name`, and `endpoint_id` so operators can
check local aliases against Nebius resources.

## Status queue states

Serverless Job status distinguishes ordinary queueing from likely capacity
blocking:

- `scheduled`: the Job is accepted and waiting to start.
- `waiting_for_capacity`: the Job is queued beyond the threshold or Nebius
  reports a capacity/resource pending reason.
- `running`: the Job is actively executing.
- terminal states such as `succeeded`, `failed`, and `cancelled` are unchanged.

Example JSON status for a capacity wait:

```json
{
  "job_id": "job-1",
  "job_name": "train-1",
  "status": "waiting_for_capacity",
  "raw_status": "queued",
  "output_uris": [],
  "queue_state_classification": "capacity",
  "queued_for_seconds": 492,
  "platform": "gpu-h200-sxm",
  "gpu_count": 8,
  "hint": "Platform may be at capacity. Retry status in a few minutes.",
  "runtime": "serverless",
  "workbench": "lerobot"
}
```

## JSON error mode

Set `NPA_ERROR_FORMAT=json` for top-level errors, or use a command's JSON output
flag where supported, such as `--output json` or `--output-format json`.

```json
{
  "error": "NotEnoughResources",
  "error_class": "capacity",
  "message": "capacity blocked",
  "project_id": "project-1",
  "platform": "gpu-h200-sxm",
  "preset": "8gpu-128vcpu-1600gb",
  "gpu_count": 8,
  "suggested_alternatives": ["Retry in a few minutes"]
}
```

## Unexpected errors

Unexpected errors are formatted without a stack trace by default:

```text
Error: Unexpected error: boom
  Run with NPA_DEBUG=1 for full traceback.
```

Set `NPA_DEBUG=1` when filing an issue or debugging locally.

## Exit codes

- `0`: success
- `1`: typed, user-actionable serverless error
- `2`: unexpected error
- `130`: interrupted by Ctrl-C
