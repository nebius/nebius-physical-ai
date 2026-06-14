# Cosmos serverless job via the SDK

This is the Python SDK counterpart to:

```bash
npa workbench cosmos train --runtime serverless --smoke
```

Cosmos serverless jobs run as Nebius Serverless AI Jobs. They are submitted
through `npa.clients.serverless.ServerlessClient.create_job(...)`, with the
job environment assembled by the `npa.serverless_common` helpers. (The
`npa.sdk.workbench.cosmos` namespace currently exposes only `check`/`fetch`; the
job submission lives in the client below.)

The script reads non-secret coordinates from the environment
(`NEBIUS_PROJECT_ID`, `NPA_REGISTRY_ID`, `NPA_S3_BUCKET`, `AWS_ENDPOINT_URL`) and
the credentials your shell already has (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`). Nothing is hardcoded to an account.

```python
import json
import os
import subprocess
import time

from npa.clients.serverless import ServerlessClient
from npa.cli.cosmos import _cosmos_train_smoke_command  # same hardened smoke the CLI submits
from npa.serverless_common import build_serverless_job_env, split_serverless_env

project_id = os.environ["NEBIUS_PROJECT_ID"]
registry_id = os.environ["NPA_REGISTRY_ID"]
bucket = os.environ["NPA_S3_BUCKET"].rstrip("/")
run_id = "cosmos-sdk-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
output_path = f"{bucket}/cosmos-verify/{run_id}/"
image = f"cr.<your-region>.nebius.cloud/{registry_id}/npa-cosmos:1.0.9"

# Multi-subnet projects require an explicit READY subnet.
subnets = json.loads(
    subprocess.check_output(
        ["nebius", "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"]
    )
)
subnet_id = next(s["metadata"]["id"] for s in subnets["items"] if s["status"]["state"] == "READY")

# Build the job env (S3 creds + HF token), then split secret-like vars out.
full_env = build_serverless_job_env(
    output_path=output_path,
    hf_token=os.environ.get("HF_TOKEN"),
    s3_credentials={
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "endpoint_url": os.environ["AWS_ENDPOINT_URL"],
    },
    extra_env={"NPA_JOB_NAME": run_id},
)
full_env.update({"COSMOS_TRAIN_SMOKE": "1", "NPA_JOB_NAME": run_id})
env, secret_env = split_serverless_env(full_env)

client = ServerlessClient()
info = client.create_job(
    project_id=project_id,
    name=run_id,
    image=image,
    command=_cosmos_train_smoke_command(5),
    gpu_type="gpu-h100-sxm",   # or gpu-h200-sxm, gpu-b300-sxm, gpu-l40s
    gpu_count=1,
    preset="1gpu-16vcpu-200gb",
    subnet_id=subnet_id,
    output_path=output_path,
    env=env,
    extra_env=secret_env,
)
info = client.poll_job(info.id, project_id, interval_s=15, ceiling_s=900)
print(json.dumps({"status": info.status, "job_name": info.name, "output_path": output_path}))
```

Expected output (validated live on `gpu-h100-sxm`, eu-north1):

```json
{"status": "succeeded", "job_name": "cosmos-sdk-<run-id>", "output_path": "s3://<your-bucket>/cosmos-verify/cosmos-sdk-<run-id>/"}
```

and `checkpoint.json` lands at `<output_path>/checkpoint.json` containing
`{"status": "succeeded", "job": "<run-id>", "smoke": true}`.

Notes:

- The underlying `nebius ai job create` call can take several minutes; the client
  logs `create_job CLI call timed out after 300s; recovering by lookup-by-name`
  and recovers automatically by job name. This is expected, not an error.
- The principal behind your `nebius` profile must have the AI Jobs role on the
  project, or submission fails with `PermissionDenied` (see the quickstart
  troubleshooting section).
