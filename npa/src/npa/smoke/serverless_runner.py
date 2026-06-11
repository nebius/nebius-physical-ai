"""Submit container golden evals to Nebius Serverless AI Jobs.

This wires the golden-eval manifest to the existing serverless Job path
(`npa.clients.serverless.ServerlessClient`), so a golden eval can be executed in
its real container image on a Nebius GPU without any bespoke infrastructure.

It is import-safe (no GPU/framework deps) and is used by both
`npa workbench golden-eval run --serverless` and the nightly CI driver.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from npa.clients.credentials import load_credentials
from npa.clients.serverless import ServerlessClient
from npa.deploy.images import container_image_for_tool
from npa.serverless_common import (
    build_serverless_job_env,
    resolve_gpu_platform,
    resolve_subnet,
    split_serverless_env,
)

from npa.smoke.manifest import container

# Nebius AI Jobs always require a GPU preset, even for CPU-only workloads, so a
# small default is used when a golden eval does not pin its own serverless GPU.
DEFAULT_SERVERLESS_GPU = "l40s"
_TERMINAL_OK = {"completed", "succeeded", "success"}


def _project_id(explicit: str | None) -> str:
    import os

    if explicit:
        return explicit
    for key in ("NEBIUS_PROJECT_ID", "NPA_PROJECT_ID"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    creds_path = Path("~/.npa/credentials.yaml").expanduser()
    if creds_path.is_file():
        raw = yaml.safe_load(creds_path.read_text()) or {}
        value = str((raw.get("nebius") or {}).get("project_id", "")).strip()
        if value:
            return value
    raise RuntimeError(
        "No Nebius project id; set NEBIUS_PROJECT_ID or nebius.project_id in "
        "~/.npa/credentials.yaml, or pass project_id explicitly."
    )


def submit_golden_eval(
    tool: str,
    *,
    gpu_type: str | None = None,
    project_id: str | None = None,
    registry: str | None = None,
    tag: str | None = None,
    timeout: str = "40m",
    poll_ceiling_s: float = 2700.0,
    wait: bool = True,
    on_state_change: Any = None,
) -> dict[str, Any]:
    """Submit ``tool``'s golden eval as a Nebius Serverless Job.

    Returns a dict with ``tool``, ``job_id``, ``status``, ``image`` and ``ok``.
    """

    spec = container(tool)
    command = spec.golden_eval.command
    gpu = gpu_type or spec.golden_eval.serverless_gpu or DEFAULT_SERVERLESS_GPU

    resolved_project = _project_id(project_id)
    image = container_image_for_tool(tool, registry=registry, tag=tag)
    cfg = load_credentials(export_to_environment=True)
    bucket = (cfg.s3_bucket or "").rstrip("/")
    if not bucket:
        raise RuntimeError("No S3 bucket configured (credentials.storage.bucket)")

    run_id = f"golden-{tool}-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = f"{bucket}/golden-evals/{run_id}/"
    platform, preset, gpu_count = resolve_gpu_platform(gpu, 1)
    subnet_id = resolve_subnet(resolved_project)

    full_env = build_serverless_job_env(
        output_path=output_path,
        hf_token=cfg.hf_token,
        s3_credentials={
            "aws_access_key_id": cfg.s3_access_key_id,
            "aws_secret_access_key": cfg.s3_secret_access_key,
            "endpoint_url": cfg.s3_endpoint,
        },
        extra_env={"NPA_GOLDEN_EVAL": tool},
    )
    env, secret_env = split_serverless_env(full_env)

    client = ServerlessClient()
    info = client.create_job(
        project_id=resolved_project,
        name=run_id,
        image=image,
        command=command,
        gpu_type=platform,
        gpu_count=gpu_count,
        preset=preset,
        subnet_id=subnet_id,
        output_path=output_path,
        env=env,
        extra_env=secret_env,
        timeout=timeout,
    )
    result = {
        "tool": tool,
        "job_id": info.id,
        "job_name": run_id,
        "image": image,
        "platform": platform,
        "status": info.status,
        "ok": False,
    }
    if not wait:
        return result

    info = client.poll_job(
        info.id,
        resolved_project,
        interval_s=15,
        ceiling_s=poll_ceiling_s,
        on_state_change=on_state_change,
    )
    result["status"] = info.status
    result["ok"] = str(info.status).lower() in _TERMINAL_OK
    return result
