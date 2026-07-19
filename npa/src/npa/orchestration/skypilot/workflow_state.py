"""Durable S3 state for SkyPilot workbench workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import boto3
import yaml
from botocore.config import Config as BotoConfig

from npa.clients.config import resolve_project_storage
from npa.clients.credentials import load_credentials, storage_endpoint_url

UTC = timezone.utc


DEFAULT_WORKFLOW_MOUNT_PATH = "/mnt/npa-workflow-state"
DEFAULT_WORKFLOW_STATE_PREFIX = ""
WORKFLOW_SCHEMA_VERSION = 1
SECRET_ENV_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
_SENSITIVE_ENV_NAMES = (
    *SECRET_ENV_NAMES,
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "NGC_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "SKYPILOT_DOCKER_PASSWORD",
)


class WorkflowStateError(RuntimeError):
    """Raised when durable workflow state cannot be read or written."""


@dataclass(frozen=True)
class WorkflowS3Config:
    """Resolved S3 location and credentials for a workflow run prefix."""

    bucket: str
    prefix: str
    endpoint_url: str
    aws_access_key_id: str = field(default="", repr=False)
    aws_secret_access_key: str = field(default="", repr=False)
    project: str | None = None

    @property
    def uri(self) -> str:
        return _join_s3_uri(self.bucket, self.prefix)

    def client(self):
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url or None,
            aws_access_key_id=self.aws_access_key_id or None,
            aws_secret_access_key=self.aws_secret_access_key or None,
            config=BotoConfig(signature_version="s3v4"),
        )

    def secret_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.endpoint_url:
            env["AWS_ENDPOINT_URL"] = self.endpoint_url
            env["NEBIUS_S3_ENDPOINT"] = self.endpoint_url
            # SkyPilot 0.12.2 S3 mounts use rclone/goofys. These env vars let
            # rclone-backed S3-compatible mounts pick up a non-AWS endpoint
            # while the normal AWS_* variables carry the access keys.
            env["RCLONE_CONFIG_S3_TYPE"] = "s3"
            env["RCLONE_CONFIG_S3_PROVIDER"] = "Other"
            env["RCLONE_CONFIG_S3_ENV_AUTH"] = "true"
            env["RCLONE_CONFIG_S3_ENDPOINT"] = self.endpoint_url
            env["RCLONE_S3_ENDPOINT"] = self.endpoint_url
        return env

    @property
    def sky_mount_source(self) -> str:
        scheme = "nebius" if _is_nebius_endpoint(self.endpoint_url) else "s3"
        return f"{scheme}://{self.bucket}"

    @property
    def sky_mount_store(self) -> str:
        return "NEBIUS" if _is_nebius_endpoint(self.endpoint_url) else "S3"


@dataclass(frozen=True)
class InstrumentedWorkflow:
    """A workflow YAML with S3 state/log instrumentation applied."""

    yaml_text: str
    manifest: dict[str, Any]
    stages: tuple[str, ...]
    mount_path: str = DEFAULT_WORKFLOW_MOUNT_PATH


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_workflow_s3_config(
    *,
    run_id: str,
    project: str | None = None,
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = DEFAULT_WORKFLOW_STATE_PREFIX,
    s3_bucket: str = "",
    s3_endpoint: str = "",
) -> WorkflowS3Config:
    """Resolve an exact S3 run prefix from CLI args, env, and NPA config."""

    if not run_id and not workflow_s3_uri:
        raise WorkflowStateError("run_id or workflow_s3_uri is required")

    storage = resolve_project_storage(project)
    credentials = load_credentials()
    endpoint = storage_endpoint_url(
        s3_endpoint
        or storage.endpoint_url
        or credentials.s3_endpoint
        or os.environ.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
    )
    access_key = (
        storage.aws_access_key_id
        or credentials.s3_access_key_id
        or os.environ.get("AWS_ACCESS_KEY_ID", "")
    )
    secret_key = (
        storage.aws_secret_access_key
        or credentials.s3_secret_access_key
        or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    )

    if workflow_s3_uri:
        bucket, prefix = parse_s3_uri(workflow_s3_uri)
    else:
        bucket_source = s3_bucket or storage.checkpoint_bucket or credentials.s3_bucket
        if not bucket_source:
            raise WorkflowStateError(
                "S3 bucket is not configured. Pass --s3-bucket, --workflow-s3-uri, "
                "or configure project storage."
            )
        bucket, base_prefix = parse_s3_uri(bucket_source)
        parent_prefix = workflow_s3_prefix.strip("/")
        if not parent_prefix:
            parent_prefix = base_prefix
        prefix = "/".join(part for part in (parent_prefix, run_id) if part).strip("/")

    if not bucket or not prefix:
        raise WorkflowStateError("workflow S3 state requires a bucket and run prefix")
    if not endpoint:
        raise WorkflowStateError(
            "S3 endpoint is not configured. Pass --s3-endpoint or configure project storage."
        )
    if not access_key or not secret_key:
        raise WorkflowStateError(
            "S3 access key and secret are not configured for workflow state."
        )

    return WorkflowS3Config(
        bucket=bucket,
        prefix=prefix.strip("/"),
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        project=project or None,
    )


def parse_s3_uri(value: str) -> tuple[str, str]:
    """Parse either a bucket name or an s3://bucket/prefix URI."""

    raw = value.strip().rstrip("/")
    if not raw:
        return "", ""
    parsed = urlparse(raw)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if "://" in raw:
        raise WorkflowStateError(f"Expected an s3:// URI, got: {value}")
    return raw, ""


def instrument_workflow_yaml(
    yaml_path: Path,
    *,
    run_id: str,
    state: WorkflowS3Config,
    job_id: str = "",
    mount_path: str = DEFAULT_WORKFLOW_MOUNT_PATH,
) -> InstrumentedWorkflow:
    """Add writable S3 mounts, redacted tee logging, and status writes."""

    docs = _load_yaml_documents(yaml_path)
    if not docs:
        raise WorkflowStateError("SkyPilot YAML is empty")

    stages = _workflow_stage_names(docs)
    manifest = build_manifest(
        run_id=run_id,
        state=state,
        stages=stages,
        workflow_name=_workflow_name(docs, yaml_path),
        job_id=job_id,
    )
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    for index, doc in enumerate(docs):
        if index not in _stage_doc_indexes(docs):
            continue
        stage = str(doc.get("name") or f"stage-{index}")
        _instrument_stage_doc(
            doc,
            run_id=run_id,
            stage=stage,
            state=state,
            manifest_json=manifest_json,
            mount_path=mount_path,
        )
    return InstrumentedWorkflow(
        yaml_text=yaml.safe_dump_all(docs, sort_keys=False),
        manifest=manifest,
        stages=tuple(stages),
        mount_path=mount_path,
    )


def build_manifest(
    *,
    run_id: str,
    state: WorkflowS3Config,
    stages: Sequence[str],
    workflow_name: str,
    job_id: str = "",
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "run_id": run_id,
        "workflow_name": workflow_name,
        "run_prefix_uri": state.uri,
        "submitted_at": now,
        "updated_at": now,
        "sky_job_id": str(job_id or ""),
        "stages": {
            stage: {
                "name": stage,
                "sky_job_id": str(job_id or ""),
                "sky_task_id": "",
                "log_uri": _join_s3_uri(state.bucket, state.prefix, "logs", stage, "run.log"),
                "status_uri": _join_s3_uri(state.bucket, state.prefix, "logs", stage, "status.json"),
                "artifact_uri": _join_s3_uri(state.bucket, state.prefix, "artifacts", stage) + "/",
            }
            for stage in stages
        },
    }


def write_manifest(
    manifest: Mapping[str, Any],
    state: WorkflowS3Config,
    *,
    job_id: str = "",
) -> dict[str, Any]:
    payload = dict(manifest)
    payload["updated_at"] = utc_now()
    if job_id:
        payload["sky_job_id"] = str(job_id)
    stages = dict(payload.get("stages", {}))
    for stage, info in list(stages.items()):
        if isinstance(info, dict) and job_id:
            info = dict(info)
            info["sky_job_id"] = str(job_id)
            stages[stage] = info
    payload["stages"] = stages
    put_json(state, "manifest.json", payload=payload)
    return payload


def read_manifest(state: WorkflowS3Config) -> dict[str, Any]:
    return get_json(state, "manifest.json")


def read_stage_status(state: WorkflowS3Config, stage: str) -> dict[str, Any] | None:
    try:
        return get_json(state, "logs", stage, "status.json")
    except WorkflowStateError as exc:
        if "not found" in str(exc).lower():
            return None
        raise


def read_stage_log(state: WorkflowS3Config, stage: str) -> str:
    return get_text(state, "logs", stage, "run.log")


def list_artifacts(state: WorkflowS3Config, stage: str | None = None) -> list[str]:
    prefix = "/".join(part for part in (state.prefix, "artifacts", stage or "") if part).strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    objects: list[str] = []
    paginator = state.client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=state.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key and not key.endswith("/"):
                objects.append(_join_s3_uri(state.bucket, key))
    return sorted(objects)


def list_runs(
    *,
    state_parent: WorkflowS3Config,
    limit: int = 50,
) -> list[dict[str, Any]]:
    prefix = state_parent.prefix.strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    paginator = state_parent.client().get_paginator("list_objects_v2")
    runs: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=state_parent.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key.endswith("/manifest.json"):
                continue
            run_prefix = key.removesuffix("/manifest.json")
            run_state = WorkflowS3Config(
                bucket=state_parent.bucket,
                prefix=run_prefix,
                endpoint_url=state_parent.endpoint_url,
                aws_access_key_id=state_parent.aws_access_key_id,
                aws_secret_access_key=state_parent.aws_secret_access_key,
                project=state_parent.project,
            )
            try:
                manifest = read_manifest(run_state)
            except WorkflowStateError:
                continue
            runs.append(
                {
                    "run_id": manifest.get("run_id", run_prefix.rsplit("/", 1)[-1]),
                    "workflow_name": manifest.get("workflow_name", ""),
                    "run_prefix_uri": run_state.uri,
                    "updated_at": manifest.get("updated_at", ""),
                    "sky_job_id": manifest.get("sky_job_id", ""),
                }
            )
            if len(runs) >= limit:
                return sorted(runs, key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return sorted(runs, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def put_json(state: WorkflowS3Config, *parts: str, payload: Mapping[str, Any]) -> None:
    key = _key(state.prefix, *parts)
    state.client().put_object(
        Bucket=state.bucket,
        Key=key,
        Body=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def get_json(state: WorkflowS3Config, *parts: str) -> dict[str, Any]:
    text = get_text(state, *parts)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WorkflowStateError(f"Invalid JSON at {_join_s3_uri(state.bucket, _key(state.prefix, *parts))}") from exc
    if not isinstance(payload, dict):
        raise WorkflowStateError("Workflow state JSON must be an object")
    return payload


def get_text(state: WorkflowS3Config, *parts: str) -> str:
    key = _key(state.prefix, *parts)
    try:
        response = state.client().get_object(Bucket=state.bucket, Key=key)
    except Exception as exc:  # boto3 exposes provider-specific ClientError payloads.
        raise WorkflowStateError(f"S3 object not found or unreadable: {_join_s3_uri(state.bucket, key)}") from exc
    return response["Body"].read().decode("utf-8", errors="replace")


def tail_live_job_logs(
    *,
    sky_bin: str,
    job_id: str,
    stage: str = "",
    follow: bool = False,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    cmd = [sky_bin, "jobs", "logs", str(job_id)]
    if stage:
        cmd.append(stage)
    cmd.append("--follow" if follow else "--no-follow")
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def cancel_workflow_job(
    *,
    sky_bin: str,
    job_id: str,
    run_id: str,
    cluster: str = "",
    timeout: int = 900,
    poll_seconds: float = 10.0,
) -> dict[str, Any]:
    cancel = subprocess.run(
        [sky_bin, "jobs", "cancel", "--yes", str(job_id)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    cluster_name = cluster or run_id
    down = subprocess.run(
        [sky_bin, "down", "--yes", cluster_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        status = subprocess.run(
            [sky_bin, "status", "--refresh"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(timeout, 300),
            check=False,
        )
        last_status = status.stdout + status.stderr
        if cluster_name not in status.stdout:
            break
        time.sleep(poll_seconds)
    return {
        "job_id": str(job_id),
        "cluster": cluster_name,
        "cancel_returncode": cancel.returncode,
        "cancel_stdout": redact_text(cancel.stdout),
        "cancel_stderr": redact_text(cancel.stderr),
        "down_returncode": down.returncode,
        "down_stdout": redact_text(down.stdout),
        "down_stderr": redact_text(down.stderr),
        "status_after_down": redact_text(last_status),
    }


def redact_text(text: str, secrets: Sequence[str] | None = None) -> str:
    redacted = text
    for secret in secrets or ():
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    for name in _SENSITIVE_ENV_NAMES:
        redacted = re.sub(
            rf"({re.escape(name)}\s*[:=]\s*)[^\s,;'\"]+",
            r"\1<redacted>",
            redacted,
            flags=re.IGNORECASE,
        )
    patterns = (
        r"hf_[A-Za-z0-9_=-]{8,}",
        r"nvapi-[A-Za-z0-9_=-]{8,}",
        r"gh[pousr]_[A-Za-z0-9_=-]{20,}",
        r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    )
    for pattern in patterns:
        redacted = re.sub(pattern, "<redacted>", redacted)
    return redacted


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise WorkflowStateError("SkyPilot YAML documents must be mappings")
    return docs


def _workflow_name(docs: Sequence[Mapping[str, Any]], yaml_path: Path) -> str:
    first = docs[0] if docs else {}
    return str(first.get("name") or Path(yaml_path).stem)


def _stage_doc_indexes(docs: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    indexes: list[int] = []
    for index, doc in enumerate(docs):
        if index == 0 and "execution" in doc and "run" not in doc:
            continue
        if "run" in doc or "resources" in doc:
            indexes.append(index)
    return tuple(indexes)


def _workflow_stage_names(docs: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for index in _stage_doc_indexes(docs):
        names.append(str(docs[index].get("name") or f"stage-{index}"))
    return names


def _instrument_stage_doc(
    doc: dict[str, Any],
    *,
    run_id: str,
    stage: str,
    state: WorkflowS3Config,
    manifest_json: str,
    mount_path: str,
) -> None:
    envs = doc.setdefault("envs", {})
    if not isinstance(envs, dict):
        raise WorkflowStateError(f"envs for stage {stage!r} must be a mapping")
    envs.update(
        {
            "NPA_WORKFLOW_RUN_ID": run_id,
            "NPA_WORKFLOW_STAGE": stage,
            "NPA_WORKFLOW_S3_BUCKET": state.bucket,
            "NPA_WORKFLOW_S3_PREFIX": state.prefix,
            "NPA_WORKFLOW_RUN_PREFIX_URI": state.uri,
            "NPA_WORKFLOW_MOUNT_ROOT": mount_path,
            "NPA_WORKFLOW_MANIFEST_JSON": manifest_json,
            "AWS_ENDPOINT_URL": state.endpoint_url,
            "NEBIUS_S3_ENDPOINT": state.endpoint_url,
            "RCLONE_CONFIG_S3_TYPE": "s3",
            "RCLONE_CONFIG_S3_PROVIDER": "Other",
            "RCLONE_CONFIG_S3_ENV_AUTH": "true",
            "RCLONE_CONFIG_S3_ENDPOINT": state.endpoint_url,
            "RCLONE_S3_ENDPOINT": state.endpoint_url,
        }
    )
    file_mounts = doc.setdefault("file_mounts", {})
    if not isinstance(file_mounts, dict):
        raise WorkflowStateError(f"file_mounts for stage {stage!r} must be a mapping")
    file_mounts.setdefault(
        mount_path,
        {
            "source": state.sky_mount_source,
            "store": state.sky_mount_store,
            "mode": "MOUNT",
            "persistent": True,
        },
    )
    original_run = str(doc.get("run", ""))
    doc["run"] = _instrumented_run_script(original_run)


def _instrumented_run_script(original_run: str) -> str:
    prelude = r'''set -euo pipefail
npa_workflow_python="$(command -v python3 || command -v python || true)"
npa_workflow_mount_root="${NPA_WORKFLOW_MOUNT_ROOT:-/mnt/npa-workflow-state}"
npa_workflow_prefix="${NPA_WORKFLOW_S3_PREFIX:?NPA_WORKFLOW_S3_PREFIX is required}"
npa_workflow_stage="${NPA_WORKFLOW_STAGE:?NPA_WORKFLOW_STAGE is required}"
npa_workflow_log_dir="${npa_workflow_mount_root}/${npa_workflow_prefix}/logs/${npa_workflow_stage}"
npa_workflow_artifact_dir="${npa_workflow_mount_root}/${npa_workflow_prefix}/artifacts/${npa_workflow_stage}"
mkdir -p "${npa_workflow_log_dir}" "${npa_workflow_artifact_dir}" "${npa_workflow_mount_root}/${npa_workflow_prefix}"
npa_workflow_stage_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export NPA_WORKFLOW_STAGE_START="${npa_workflow_stage_start}"
export NPA_WORKFLOW_MOUNT_ROOT_RESOLVED="${npa_workflow_mount_root}"
export NPA_WORKFLOW_S3_PREFIX_RESOLVED="${npa_workflow_prefix}"
export NPA_WORKFLOW_STAGE_RESOLVED="${npa_workflow_stage}"
npa_workflow_redact_stream() {
  sed -E \
    -e 's/(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_TOKEN|HUGGINGFACE_HUB_TOKEN|NGC_API_KEY|GITHUB_TOKEN|GH_TOKEN|SKYPILOT_DOCKER_PASSWORD)([=:])[A-Za-z0-9_+=\/.,:@-]+/\1\2<redacted>/Ig' \
    -e 's/hf_[A-Za-z0-9_=-]{8,}/<redacted>/g' \
    -e 's/nvapi-[A-Za-z0-9_=-]{8,}/<redacted>/g' \
    -e 's/gh[pousr]_[A-Za-z0-9_=-]{20,}/<redacted>/g' \
    -e 's/(AKIA|ASIA)[A-Z0-9]{16}/<redacted>/g'
}
npa_workflow_write_manifest() {
  [ -n "${npa_workflow_python}" ] || return 0
  "${npa_workflow_python}" -c '
import json, os
from pathlib import Path
raw = os.environ.get("NPA_WORKFLOW_MANIFEST_JSON", "")
if not raw:
    raise SystemExit(0)
manifest = json.loads(raw)
task_id = os.environ.get("SKYPILOT_TASK_ID", "")
stage = os.environ.get("NPA_WORKFLOW_STAGE", "")
target = Path(os.environ["NPA_WORKFLOW_MOUNT_ROOT_RESOLVED"]) / os.environ["NPA_WORKFLOW_S3_PREFIX_RESOLVED"] / "manifest.json"
existing = {}
if target.exists():
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
job_id = (
    os.environ.get("SKYPILOT_MANAGED_JOB_ID", "")
    or manifest.get("sky_job_id", "")
    or existing.get("sky_job_id", "")
)
manifest["sky_job_id"] = str(job_id or "")
manifest["updated_at"] = os.environ.get("NPA_WORKFLOW_STAGE_START", "")
manifest["last_writer"] = "pod"
manifest["pod_written_at"] = os.environ.get("NPA_WORKFLOW_STAGE_START", "")
stages = manifest.setdefault("stages", {})
for info in stages.values():
    if isinstance(info, dict) and job_id:
        info["sky_job_id"] = str(job_id)
if stage in stages and isinstance(stages[stage], dict):
    stages[stage]["sky_task_id"] = str(task_id or "")
target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'
}
npa_workflow_write_status() {
  local state="$1"
  local tier="$2"
  local error_summary="${3:-}"
  local end_time="${4:-}"
  [ -n "${npa_workflow_python}" ] || return 0
  NPA_WORKFLOW_STATUS_STATE="${state}" \
  NPA_WORKFLOW_STATUS_TIER="${tier}" \
  NPA_WORKFLOW_STATUS_ERROR="${error_summary}" \
  NPA_WORKFLOW_STATUS_END="${end_time}" \
  "${npa_workflow_python}" -c '
import json, os
from pathlib import Path
mount = Path(os.environ["NPA_WORKFLOW_MOUNT_ROOT_RESOLVED"])
prefix = os.environ["NPA_WORKFLOW_S3_PREFIX_RESOLVED"]
stage = os.environ["NPA_WORKFLOW_STAGE_RESOLVED"]
manifest_path = mount / prefix / "manifest.json"
manifest = {}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
job_id = os.environ.get("SKYPILOT_MANAGED_JOB_ID", "") or manifest.get("sky_job_id", "")
bucket = os.environ.get("NPA_WORKFLOW_S3_BUCKET", "")
payload = {
    "schema_version": 1,
    "run_id": os.environ.get("NPA_WORKFLOW_RUN_ID", ""),
    "stage": stage,
    "state": os.environ["NPA_WORKFLOW_STATUS_STATE"],
    "tier": os.environ["NPA_WORKFLOW_STATUS_TIER"],
    "start": os.environ.get("NPA_WORKFLOW_STAGE_START", ""),
    "end": os.environ.get("NPA_WORKFLOW_STATUS_END", ""),
    "start_time": os.environ.get("NPA_WORKFLOW_STAGE_START", ""),
    "end_time": os.environ.get("NPA_WORKFLOW_STATUS_END", ""),
    "sky_job_id": str(job_id or ""),
    "sky_task_id": os.environ.get("SKYPILOT_TASK_ID", ""),
    "artifact_uri": "s3://{}/{}/artifacts/{}/".format(bucket, prefix, stage),
    "log_uri": "s3://{}/{}/logs/{}/run.log".format(bucket, prefix, stage),
    "error_summary": os.environ.get("NPA_WORKFLOW_STATUS_ERROR", ""),
}
target = mount / prefix / "logs" / stage / "status.json"
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
'
}
npa_workflow_finalize() {
  local rc="$?"
  trap - EXIT
  set +e
  local end_time
  end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "${rc}" -eq 0 ]; then
    npa_workflow_write_status "SUCCEEDED" "WORKS" "" "${end_time}"
  else
    npa_workflow_write_status "FAILED" "PARTIAL" "exit ${rc}" "${end_time}"
  fi
  exit "${rc}"
}
npa_workflow_write_manifest
npa_workflow_write_status "RUNNING" "SEAM" "" ""
exec > >(npa_workflow_redact_stream | tee -a "${npa_workflow_log_dir}/run.log") 2>&1
trap npa_workflow_finalize EXIT
'''
    if original_run.strip():
        return prelude + "\n" + original_run.rstrip() + "\n"
    return prelude + "\n"


def _join_s3_uri(bucket: str, *parts: str) -> str:
    key = "/".join(part.strip("/") for part in parts if part is not None and part.strip("/") != "")
    if parts and str(parts[-1]).endswith("/"):
        key = key.rstrip("/") + "/"
    return f"s3://{bucket}/{key}" if key else f"s3://{bucket}/"


def _key(prefix: str, *parts: str) -> str:
    return "/".join(part.strip("/") for part in (prefix, *parts) if part and part.strip("/"))


def _is_nebius_endpoint(endpoint_url: str) -> bool:
    return "nebius.cloud" in endpoint_url.lower()
