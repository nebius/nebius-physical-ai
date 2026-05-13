# SDK Error Reference

The serverless SDK exposes typed exceptions from
`npa.clients.serverless`. Existing exception class names are stable and
`str(exc)` returns only the message for backward compatibility.

## Exception hierarchy

```text
ServerlessClientError
+-- NotEnoughResourcesError
|   +-- QuotaError
+-- AuthError
+-- EndpointNotFoundError
```

## Fields

| Exception | Stable fields |
| --- | --- |
| `ServerlessClientError` | `message` |
| `NotEnoughResourcesError` | `message`, `project_id`, `platform`, `preset`, `gpu_count`, `suggested_alternatives`, `raw_stderr`, `error_class` |
| `QuotaError` | same as `NotEnoughResourcesError`, with `error_class="quota"` |
| `AuthError` | `message`, `hint` |
| `EndpointNotFoundError` | `message`, `project_id`, `endpoint_name`, `endpoint_id` |

`error_class` is one of `capacity`, `quota`, or `scheduling`. New fields may be
added in future releases without removing the fields above.

## Capacity handling example

```python
from npa.clients.serverless import NotEnoughResourcesError, ServerlessClient

client = ServerlessClient()

try:
    client.create_job(
        project_id="project-1",
        name="train-1",
        image="registry.example/npa-lerobot:latest",
        command="python train.py",
        gpu_type="gpu-h200-sxm",
        gpu_count=8,
        output_path="s3://bucket/out/",
    )
except NotEnoughResourcesError as exc:
    if exc.error_class == "quota":
        route_to_quota_workflow(exc.project_id)
    elif exc.platform:
        retry_later_or_try_platform(exc.platform, exc.suggested_alternatives)
    else:
        retry_later(exc.suggested_alternatives)
```

## Agent integration

Agents and orchestrators should branch on fields instead of parsing error text:

- Use `isinstance(exc, NotEnoughResourcesError)` for capacity-family handling.
- Use `exc.error_class == "quota"` for quota workflows.
- Use `exc.platform`, `exc.preset`, and `exc.gpu_count` to select a smaller or
  alternate request.
- Use `exc.project_id` to rotate to another eligible project.
- Use `exc.suggested_alternatives` as user-facing remediation text.

For status polling, call `ServerlessClient.classify_queue_state(job)` after
`get_job(...)`. A queued Job returns:

- `scheduled` when it was recently accepted or Nebius reports an accepted queue
  state.
- `waiting_for_capacity` when Nebius reports capacity/resource pressure or the
  queued duration exceeds the threshold.

## Backward compatibility

Existing code that catches `ServerlessClientError`,
`NotEnoughResourcesError`, `QuotaError`, `AuthError`, or
`EndpointNotFoundError` keeps working. The exception names are unchanged and
`str(exc)` remains the plain message string.
