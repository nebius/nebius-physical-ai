"""Live status for Sim2Real staged runs (K8s direct submit + S3 artifacts)."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.config import artifact_uris, build_config_from_env
from npa.workflows.sim2real.constants import DEFAULT_PREFIX, DEFAULT_S3_ENDPOINT

_STAGE_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("stage_01_trigger", "stage_01_trigger/trigger.json", "file"),
    ("stage_02_assets", "stage_02_assets/assets_manifest.json", "file"),
    ("stage_03_augment", "augment/cosmos2-transfer-result.json", "file"),
    ("stage_04_envs_raw", "envs/raw/", "prefix"),
    ("stage_05_envs_train", "envs/train/", "prefix"),
    ("stage_06_tokens", "tokens/manifest.json", "file"),
    ("stage_07_actions_train", "actions/train/", "prefix"),
    ("stage_08_vlm_eval_train", "vlm_eval/train/", "prefix"),
    ("stage_09_training_signal", "training_signal/train/", "prefix"),
    ("stage_10_eval_heldout", "eval/heldout/report.json", "file"),
    ("stage_11_outer_loop", "outer_loop/decision.json", "file"),
    ("stage_12_external_validation_stub", "stage_12_external_validation/external_stub.json", "file"),
    ("stage_13_retrigger", "stage_13_retrigger/retrigger.json", "file"),
    ("report", "reports/sim2real-report.json", "file"),
)


@dataclass(frozen=True)
class OperatorConfig:
    bucket: str
    endpoint_url: str
    registry: str
    k8s_context: str


def load_operator_config() -> OperatorConfig:
    """Read non-secret operator settings from ``~/.npa/config.yaml``."""

    path = Path.home() / ".npa" / "config.yaml"
    if not path.exists():
        raise ValueError("missing ~/.npa/config.yaml — run: npa configure")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    storage = cfg.get("storage") or {}
    bucket = str(storage.get("bucket", "")).replace("s3://", "").split("/", 1)[0]
    endpoint = str(storage.get("endpoint_url") or DEFAULT_S3_ENDPOINT)
    registry = str(storage.get("registry", cfg.get("registry", ""))).rstrip("/")
    k8s_context = str(storage.get("k8s_context") or "")
    if not k8s_context:
        for proj in (cfg.get("projects") or {}).values():
            if isinstance(proj, dict) and proj.get("k8s_context"):
                k8s_context = str(proj["k8s_context"])
                break
    if not bucket:
        raise ValueError("storage.bucket is not set in ~/.npa/config.yaml")
    if not k8s_context:
        raise ValueError("storage.k8s_context is not set in ~/.npa/config.yaml")
    return OperatorConfig(
        bucket=bucket,
        endpoint_url=endpoint,
        registry=registry,
        k8s_context=k8s_context,
    )


def resolve_kubeconfig(context: str) -> Path:
    explicit = Path(str(os.environ.get("KUBECONFIG", "")))
    if explicit.is_file():
        return explicit
    path = Path.home() / ".npa" / "clusters" / context / "kubeconfig"
    if path.is_file():
        return path
    resolved = Path.home() / ".npa" / "clusters" / context / "kubeconfig.resolved"
    if resolved.is_file():
        return resolved
    raise ValueError(
        f"kubeconfig not found for context {context!r} "
        f"(expected {path} or KUBECONFIG)"
    )


def orchestrator_job_name(run_id: str) -> str:
    return f"sim2real-{run_id}"


def run_prefix_uri(*, bucket: str, prefix: str, run_id: str, endpoint: str) -> str:
    del endpoint
    return f"s3://{bucket}/{prefix.rstrip('/')}/{run_id}/"


def _s3_object_exists(client: StorageClient, bucket: str, key: str) -> bool:
    import botocore.exceptions

    try:
        client._s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False
    except Exception:
        return False


def _s3_prefix_nonempty(client: StorageClient, bucket: str, prefix: str) -> bool:
    response = client._s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return int(response.get("KeyCount") or 0) > 0


def _stage_states(
    *,
    bucket: str,
    run_id: str,
    s3_prefix: str,
    endpoint: str,
) -> dict[str, dict[str, Any]]:
    config = build_config_from_env(
        run_id=run_id,
        s3_bucket=bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=endpoint,
    )
    uris = artifact_uris(config)
    client = StorageClient.from_environment(endpoint_url=endpoint)
    stages: dict[str, dict[str, Any]] = {}
    workflow_state: dict[str, Any] | None = None
    state_key = f"{s3_prefix.rstrip('/')}/{run_id}/state/workflow_state.json"
    if _s3_object_exists(client, bucket, state_key):
        body = client._s3.get_object(Bucket=bucket, Key=state_key)["Body"].read()
        workflow_state = json.loads(body.decode("utf-8"))

    record_by_stage = {}
    if workflow_state:
        for record in workflow_state.get("stage_records") or []:
            name = str(record.get("stage") or record.get("name") or "")
            if name:
                record_by_stage[name] = record

    for stage_name, rel_path, kind in _STAGE_CHECKS:
        uri = uris.get(stage_name, f"s3://{bucket}/{s3_prefix.rstrip('/')}/{run_id}/{rel_path}")
        parsed = urlparse(uri)
        key = parsed.path.lstrip("/")
        if kind == "prefix" and not key.endswith("/"):
            key = f"{key}/"
        if kind == "file":
            present = _s3_object_exists(client, bucket, key)
        else:
            present = _s3_prefix_nonempty(client, bucket, key)
        tier = ""
        if stage_name in record_by_stage:
            tier = str(record_by_stage[stage_name].get("tier") or "")
        stages[stage_name] = {
            "name": stage_name,
            "state": "SUCCEEDED" if present else "PENDING",
            "tier": tier,
            "artifact_uri": uri,
        }
    return stages


def _kubectl_json(args: list[str], *, kubeconfig: Path) -> dict[str, Any]:
    cmd = [
        "kubectl",
        *args,
        "-o",
        "json",
    ]
    env = dict(os.environ)
    env["KUBECONFIG"] = str(kubeconfig)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _k8s_orchestrator_status(
    *,
    run_id: str,
    context: str,
    kubeconfig: Path,
    namespace: str = "default",
) -> dict[str, Any]:
    job_name = orchestrator_job_name(run_id)
    job = _kubectl_json(
        ["--context", context, "-n", namespace, "get", "job", job_name],
        kubeconfig=kubeconfig,
    )
    if not job:
        return {
            "job_name": job_name,
            "found": False,
            "phase": "MISSING",
            "active": 0,
            "succeeded": 0,
            "failed": 0,
            "pod_phase": "",
            "pod_reason": "",
        }

    status = job.get("status") or {}
    pod = _kubectl_json(
        [
            "--context",
            context,
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            f"job-name={job_name}",
        ],
        kubeconfig=kubeconfig,
    )
    pod_items = list((pod.get("items") or []))
    pod_phase = ""
    pod_reason = ""
    if pod_items:
        pod_status = pod_items[0].get("status") or {}
        pod_phase = str(pod_status.get("phase") or "")
        for state_key in ("waiting", "terminated"):
            for container in pod_status.get("containerStatuses") or []:
                state = (container.get("state") or {}).get(state_key) or {}
                if state.get("reason"):
                    pod_reason = str(state["reason"])
                    break
            if pod_reason:
                break

    active = int(status.get("active") or 0)
    succeeded = int(status.get("succeeded") or 0)
    failed = int(status.get("failed") or 0)
    if succeeded >= 1:
        phase = "SUCCEEDED"
    elif failed >= 1:
        phase = "FAILED"
    elif active >= 1:
        phase = "RUNNING"
    else:
        phase = "UNKNOWN"
    return {
        "job_name": job_name,
        "found": True,
        "phase": phase,
        "active": active,
        "succeeded": succeeded,
        "failed": failed,
        "pod_phase": pod_phase,
        "pod_reason": pod_reason,
    }


def _k8s_sibling_summary(
    *,
    run_id: str,
    context: str,
    kubeconfig: Path,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    jobs = _kubectl_json(
        ["--context", context, "-n", namespace, "get", "jobs"],
        kubeconfig=kubeconfig,
    )
    rows: list[dict[str, Any]] = []
    needle = run_id.lower()
    for item in jobs.get("items") or []:
        name = str((item.get("metadata") or {}).get("name") or "")
        if not name.startswith("s2r-") or needle not in name.lower():
            continue
        status = item.get("status") or {}
        rows.append(
            {
                "name": name,
                "active": int(status.get("active") or 0),
                "succeeded": int(status.get("succeeded") or 0),
                "failed": int(status.get("failed") or 0),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _aggregate_status(
    stages: dict[str, dict[str, Any]],
    k8s: dict[str, Any],
) -> str:
    if k8s.get("phase") == "SUCCEEDED" or stages.get("report", {}).get("state") == "SUCCEEDED":
        return "SUCCEEDED"
    if k8s.get("phase") == "FAILED" or int(k8s.get("failed") or 0) > 0:
        return "FAILED"
    if k8s.get("pod_reason") in {"ImagePullBackOff", "ErrImagePull", "CrashLoopBackOff"}:
        return "FAILED"
    if k8s.get("phase") == "RUNNING":
        return "RUNNING"
    if any(info.get("state") == "SUCCEEDED" for info in stages.values()):
        return "RUNNING"
    if not k8s.get("found"):
        return "FAILED" if not any(info.get("state") == "SUCCEEDED" for info in stages.values()) else "UNKNOWN"
    return "UNKNOWN"


def _current_stage(stages: dict[str, dict[str, Any]]) -> str:
    last_done = ""
    for stage_name, _, _ in _STAGE_CHECKS:
        if stages.get(stage_name, {}).get("state") == "SUCCEEDED":
            last_done = stage_name
    if not last_done:
        return "stage_01_trigger"
    names = [name for name, _, _ in _STAGE_CHECKS]
    if last_done == names[-1]:
        return last_done
    idx = names.index(last_done)
    return names[min(idx + 1, len(names) - 1)]


def get_sim2real_workflow_status(
    run_id: str,
    *,
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
    k8s_context: str = "",
    k8s_namespace: str = "default",
    kubeconfig: str | Path = "",
) -> dict[str, Any]:
    """Return workflow-style status for a Sim2Real staged K8s run."""

    operator = load_operator_config()
    bucket = s3_bucket or operator.bucket
    endpoint = s3_endpoint or operator.endpoint_url
    context = k8s_context or operator.k8s_context
    kcfg = Path(kubeconfig) if kubeconfig else resolve_kubeconfig(context)

    stages = _stage_states(
        bucket=bucket,
        run_id=run_id,
        s3_prefix=s3_prefix,
        endpoint=endpoint,
    )
    k8s = _k8s_orchestrator_status(
        run_id=run_id,
        context=context,
        kubeconfig=kcfg,
        namespace=k8s_namespace,
    )
    siblings = _k8s_sibling_summary(
        run_id=run_id,
        context=context,
        kubeconfig=kcfg,
        namespace=k8s_namespace,
    )
    status = _aggregate_status(stages, k8s)
    if status == "RUNNING":
        current = _current_stage(stages)
        if isinstance(stages.get(current), dict):
            stages[current]["state"] = "RUNNING"

    return {
        "run_id": run_id,
        "workflow_name": "sim2real-staged-loop",
        "status": status,
        "live_status": str(k8s.get("phase") or ""),
        "current_stage": _current_stage(stages),
        "run_prefix_uri": run_prefix_uri(
            bucket=bucket,
            prefix=s3_prefix,
            run_id=run_id,
            endpoint=endpoint,
        ),
        "k8s_job": k8s.get("job_name"),
        "k8s_context": context,
        "pod_phase": k8s.get("pod_phase"),
        "pod_reason": k8s.get("pod_reason"),
        "sibling_jobs": siblings,
        "stages": stages,
    }
