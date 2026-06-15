"""Live status for Sim2Real staged runs (K8s direct submit + S3 artifacts)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.config import artifact_uris, build_config_from_env
from npa.workflows.sim2real.constants import DEFAULT_PREFIX, DEFAULT_S3_ENDPOINT

@dataclass(frozen=True)
class _ArtifactRule:
    """Relative S3 keys under the run prefix; ``any`` or ``all`` must exist."""

    paths: tuple[str, ...]
    kind: str  # file | prefix
    match: str = "any"  # any | all


@dataclass(frozen=True)
class _StageMonitorSpec:
    name: str
    rules: tuple[_ArtifactRule, ...]
    component_names: tuple[str, ...] = ()
    stage_numbers: tuple[int, ...] = ()
    infer_from_later: bool = False


_STAGE_SPECS: tuple[_StageMonitorSpec, ...] = (
    _StageMonitorSpec(
        "stage_01_trigger",
        (_ArtifactRule(("stage_01_trigger/trigger.json",), "file"),),
        component_names=("stage_01_trigger",),
        stage_numbers=(1,),
        infer_from_later=True,
    ),
    _StageMonitorSpec(
        "stage_02_assets",
        (
            _ArtifactRule(
                (
                    "stage_02_assets/consumed_scene_spec.json",
                    "stage_02_assets/consumed_robot_spec.json",
                ),
                "file",
                match="all",
            ),
            _ArtifactRule(("stage_02_assets/assets_manifest.json",), "file"),
        ),
        component_names=("stage_02_assets",),
        stage_numbers=(2,),
    ),
    _StageMonitorSpec(
        "stage_03_augment",
        (
            _ArtifactRule(("augment/manifest.json",), "file"),
            _ArtifactRule(("augment/cosmos2-transfer-result.json",), "file"),
            _ArtifactRule(("augment/frames/index.json",), "file"),
            _ArtifactRule(("augment/frames/",), "prefix"),
        ),
        component_names=("stage_03_augment",),
        stage_numbers=(3,),
    ),
    _StageMonitorSpec(
        "stage_04_envs_raw",
        (
            _ArtifactRule(("envs/raw/",), "prefix"),
            _ArtifactRule(("envs/raw/raw-shard-00-of-01-summary.json",), "file"),
            _ArtifactRule(("envs/manifest/scene-spec.json",), "file"),
        ),
        component_names=("stage_04_06_env_gen_split_tokens",),
        stage_numbers=(4,),
    ),
    _StageMonitorSpec(
        "stage_05_envs_train",
        (
            _ArtifactRule(("envs/train/envs.jsonl",), "file"),
            _ArtifactRule(("envs/train/",), "prefix"),
            _ArtifactRule(("envs/train/manifest.json",), "file"),
        ),
        component_names=("stage_04_06_env_gen_split_tokens",),
        stage_numbers=(5,),
    ),
    _StageMonitorSpec(
        "stage_06_tokens",
        (
            _ArtifactRule(("envs/train/envs.jsonl",), "file"),
            _ArtifactRule(("envs/heldout/envs.jsonl",), "file"),
            _ArtifactRule(("tokens/manifest.json",), "file"),
            _ArtifactRule(("envs/manifest/split-manifest.json",), "file"),
            _ArtifactRule(("envs/split-manifest.json",), "file"),
        ),
        component_names=("stage_04_06_env_gen_split_tokens",),
        stage_numbers=(6,),
    ),
    _StageMonitorSpec(
        "stage_07_actions_train",
        (_ArtifactRule(("actions/train/",), "prefix"),),
        component_names=("stage_07_actions_train",),
        stage_numbers=(7,),
    ),
    _StageMonitorSpec(
        "stage_08_vlm_eval_train",
        (_ArtifactRule(("vlm_eval/train/",), "prefix"),),
        component_names=("stage_08_vlm_eval_train",),
        stage_numbers=(8,),
    ),
    _StageMonitorSpec(
        "stage_09_training_signal",
        (_ArtifactRule(("training_signal/train/",), "prefix"),),
        component_names=("stage_09_training_signal",),
        stage_numbers=(9,),
    ),
    _StageMonitorSpec(
        "stage_10_eval_heldout",
        (_ArtifactRule(("eval/heldout/report.json",), "file"),),
        component_names=("stage_10_eval_heldout",),
        stage_numbers=(10,),
    ),
    _StageMonitorSpec(
        "stage_11_outer_loop",
        (_ArtifactRule(("outer_loop/decision.json",), "file"),),
        component_names=("stage_11_outer_loop",),
        stage_numbers=(11,),
    ),
    _StageMonitorSpec(
        "stage_12_external_validation_stub",
        (_ArtifactRule(("stage_12_external_validation/external_stub.json",), "file"),),
        component_names=("stage_12_external_validation",),
        stage_numbers=(12,),
    ),
    _StageMonitorSpec(
        "stage_13_retrigger",
        (_ArtifactRule(("stage_13_retrigger/retrigger.json",), "file"),),
        component_names=("stage_13_retrigger",),
        stage_numbers=(13,),
    ),
    _StageMonitorSpec(
        "report",
        (_ArtifactRule(("reports/sim2real-report.json",), "file"),),
    ),
)

_STAGE_ORDER: tuple[str, ...] = tuple(spec.name for spec in _STAGE_SPECS)
_STAGE_NUMBER_TO_NAME: dict[int, str] = {
    number: spec.name for spec in _STAGE_SPECS for number in spec.stage_numbers
}
_PREAMBLE_STAGE_NAMES: frozenset[str] = frozenset(
    {
        "stage_01_trigger",
        "stage_02_assets",
        "stage_03_augment",
        "stage_04_envs_raw",
        "stage_05_envs_train",
        "stage_06_tokens",
    }
)
_OUTER_LOOP_STAGE_NAMES: frozenset[str] = frozenset(
    {
        "stage_07_actions_train",
        "stage_08_vlm_eval_train",
        "stage_09_training_signal",
        "stage_10_eval_heldout",
        "stage_11_outer_loop",
    }
)
_STATUS_COMPLETED_STAGES: dict[str, frozenset[str]] = {
    "preamble_completed": _PREAMBLE_STAGE_NAMES,
    "outer_iteration_completed": _PREAMBLE_STAGE_NAMES | _OUTER_LOOP_STAGE_NAMES,
    "finalize_completed": _PREAMBLE_STAGE_NAMES | _OUTER_LOOP_STAGE_NAMES,
    "completed": frozenset(_STAGE_ORDER),
}
_ENV_SPLIT_COMPONENT = "stage_04_06_env_gen_split_tokens"


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


_STAGED_RUN_ID_RE = re.compile(r"sim2real-staged-\d{8}t\d{6}z", re.IGNORECASE)


def normalize_staged_run_id(run_id: str) -> str:
    """Canonicalize staged run ids and strip polluted submit-log suffixes."""

    rid = (run_id or "").strip()
    if not rid:
        return rid
    first = rid.split()[0]
    match = _STAGED_RUN_ID_RE.search(first)
    if match:
        return match.group(0).lower()
    lowered = first.lower()
    if lowered.startswith("sim2real-staged-"):
        return lowered
    if lowered.startswith("sim2real-"):
        rest = lowered[len("sim2real-") :]
        if rest.startswith("staged-"):
            return f"sim2real-{rest}"
    if lowered.startswith("staged-"):
        return f"sim2real-{lowered}"
    return first


def parse_submit_run_id(output: str) -> str:
    """Parse ``run_id=`` lines from operator submit script output."""

    parsed = ""
    for line in output.splitlines():
        if line.startswith("run_id=") or line.startswith("run_id:"):
            raw = line.split("=", 1)[-1].split(":", 1)[-1].strip()
            parsed = normalize_staged_run_id(raw)
    if not parsed:
        raise ValueError("submit script did not return run_id")
    return parsed


def parse_submit_job(output: str, run_id: str = "") -> str:
    """Parse orchestrator job name from submit script output."""

    job = ""
    for line in output.splitlines():
        if line.startswith("job=") or line.startswith("job_id:"):
            job = line.split("=", 1)[-1].split(":", 1)[-1].strip().split()[0]
    if not job and run_id:
        return orchestrator_job_name(run_id)
    return job


def orchestrator_job_name(run_id: str) -> str:
    return f"sim2real-{normalize_staged_run_id(run_id)}"


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


def _artifact_rule_matches(
    client: StorageClient,
    bucket: str,
    *,
    run_prefix: str,
    rule: _ArtifactRule,
) -> bool:
    checks: list[bool] = []
    for rel_path in rule.paths:
        key = f"{run_prefix.rstrip('/')}/{rel_path.lstrip('/')}"
        if rule.kind == "prefix":
            if not key.endswith("/"):
                key = f"{key}/"
            checks.append(_s3_prefix_nonempty(client, bucket, key))
        else:
            checks.append(_s3_object_exists(client, bucket, key))
    if rule.match == "all":
        return bool(checks) and all(checks)
    return any(checks)


def _stage_artifact_present(
    client: StorageClient,
    bucket: str,
    *,
    run_prefix: str,
    spec: _StageMonitorSpec,
) -> bool:
    return any(
        _artifact_rule_matches(client, bucket, run_prefix=run_prefix, rule=rule)
        for rule in spec.rules
    )


def _record_completed_at(entry: dict[str, Any], fallback: str) -> str:
    record = entry.get("record") or {}
    payload = record.get("payload") or {}
    for candidate in (
        payload.get("created_at"),
        payload.get("updated_at"),
        record.get("created_at"),
        record.get("updated_at"),
        fallback,
    ):
        if candidate:
            return str(candidate)
    return fallback


def _workflow_completion_index(
    workflow_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for component in workflow_state.get("components") or []:
        if not isinstance(component, dict):
            continue
        name = str(component.get("name") or "")
        if not name:
            continue
        index[name] = {
            "source": "component",
            "tier": str(component.get("tier") or ""),
            "record": component,
        }
    for record in workflow_state.get("stage_records") or []:
        if not isinstance(record, dict):
            continue
        payload = record.get("payload") or {}
        stage_number = payload.get("stage")
        if not isinstance(stage_number, int):
            continue
        name = _STAGE_NUMBER_TO_NAME.get(stage_number)
        if not name or name in index:
            continue
        index[name] = {
            "source": "stage_record",
            "tier": "",
            "record": record,
        }
    env_split = index.get(_ENV_SPLIT_COMPONENT)
    if env_split:
        for stage_name in ("stage_04_envs_raw", "stage_05_envs_train", "stage_06_tokens"):
            if stage_name not in index:
                index[stage_name] = env_split
    return index


def _workflow_stage_succeeded(
    stage_name: str,
    *,
    workflow_state: dict[str, Any] | None,
    completion_index: dict[str, dict[str, Any]],
    spec: _StageMonitorSpec,
) -> dict[str, Any] | None:
    if not workflow_state:
        return None
    updated_at = str(workflow_state.get("updated_at") or "")
    status = str(workflow_state.get("status") or "")
    for milestone, stages in _STATUS_COMPLETED_STAGES.items():
        if status == milestone and stage_name in stages:
            return {
                "state": "SUCCEEDED",
                "source": "workflow_state_status",
                "tier": "",
                "completed_at": updated_at,
            }
    for component_name in spec.component_names:
        entry = completion_index.get(component_name)
        if entry:
            return {
                "state": "SUCCEEDED",
                "source": "workflow_state",
                "tier": str(entry.get("tier") or ""),
                "completed_at": _record_completed_at(entry, updated_at),
            }
    entry = completion_index.get(stage_name)
    if entry:
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state",
            "tier": str(entry.get("tier") or ""),
            "completed_at": _record_completed_at(entry, updated_at),
        }
    if stage_name in {"stage_05_envs_train", "stage_06_tokens"} and workflow_state.get(
        "train_envs_uri"
    ):
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state_train_envs_uri",
            "tier": "",
            "completed_at": updated_at,
        }
    if stage_name == "stage_04_envs_raw" and int(workflow_state.get("env_count") or 0) > 0:
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state_env_count",
            "tier": "",
            "completed_at": updated_at,
        }
    if stage_name == "stage_10_eval_heldout" and workflow_state.get("final_eval"):
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state_final_eval",
            "tier": "",
            "completed_at": updated_at,
        }
    if stage_name == "stage_11_outer_loop" and workflow_state.get("final_decision"):
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state_final_decision",
            "tier": "",
            "completed_at": updated_at,
        }
    if stage_name == "report" and str(workflow_state.get("report_path") or "").strip():
        return {
            "state": "SUCCEEDED",
            "source": "workflow_state_report_path",
            "tier": "",
            "completed_at": updated_at,
        }
    return None


def _apply_infer_from_later(stages: dict[str, dict[str, Any]]) -> None:
    later_succeeded = False
    for spec in reversed(_STAGE_SPECS):
        info = stages.get(spec.name) or {}
        if info.get("state") == "SUCCEEDED":
            later_succeeded = True
            continue
        if later_succeeded and spec.infer_from_later:
            stages[spec.name] = {
                **info,
                "state": "SUCCEEDED",
                "source": "inferred_from_later_stage",
            }


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
    run_prefix = f"{s3_prefix.rstrip('/')}/{run_id}"
    workflow_state: dict[str, Any] | None = None
    state_key = f"{run_prefix}/state/workflow_state.json"
    if _s3_object_exists(client, bucket, state_key):
        body = client._s3.get_object(Bucket=bucket, Key=state_key)["Body"].read()
        workflow_state = json.loads(body.decode("utf-8"))

    completion_index = (
        _workflow_completion_index(workflow_state) if workflow_state else {}
    )
    stages: dict[str, dict[str, Any]] = {}
    for spec in _STAGE_SPECS:
        primary_path = spec.rules[0].paths[0] if spec.rules else ""
        uri = uris.get(
            spec.name,
            f"s3://{bucket}/{run_prefix}/{primary_path}" if primary_path else "",
        )
        resolved = _workflow_stage_succeeded(
            spec.name,
            workflow_state=workflow_state,
            completion_index=completion_index,
            spec=spec,
        )
        if resolved:
            present = True
            source = str(resolved.get("source") or "workflow_state")
            tier = str(resolved.get("tier") or "")
            completed_at = str(resolved.get("completed_at") or "")
        else:
            present = _stage_artifact_present(
                client,
                bucket,
                run_prefix=run_prefix,
                spec=spec,
            )
            source = "s3_artifact" if present else ""
            tier = ""
            completed_at = ""
        stages[spec.name] = {
            "name": spec.name,
            "state": "SUCCEEDED" if present else "PENDING",
            "tier": tier,
            "artifact_uri": uri,
            "source": source,
            "completed_at": completed_at,
        }

    _apply_infer_from_later(stages)
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
    for stage_name in _STAGE_ORDER:
        if stages.get(stage_name, {}).get("state") == "SUCCEEDED":
            last_done = stage_name
    if not last_done:
        return "stage_01_trigger"
    if last_done == _STAGE_ORDER[-1]:
        return last_done
    idx = _STAGE_ORDER.index(last_done)
    return _STAGE_ORDER[min(idx + 1, len(_STAGE_ORDER) - 1)]


def status_is_terminal(status: str) -> bool:
    normalized = status.upper()
    return normalized == "SUCCEEDED" or normalized.startswith("FAILED")


def emit_sim2real_status(result: dict[str, Any], *, json_output: bool = False) -> None:
    """Print status in the same shape as durable workflow monitors."""

    if json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"run_id: {result.get('run_id')}")
    print(f"status: {result.get('status')}")
    if result.get("current_stage"):
        print(f"current_stage: {result.get('current_stage')}")
    if result.get("k8s_job"):
        print(f"k8s_job: {result.get('k8s_job')}")
    if result.get("pod_reason"):
        print(f"pod_reason: {result.get('pod_reason')}")
    print(f"run_prefix_uri: {result.get('run_prefix_uri')}")
    stages = result.get("stages", {})
    if isinstance(stages, dict):
        for stage, info in stages.items():
            state = info.get("state", "UNKNOWN") if isinstance(info, dict) else "UNKNOWN"
            tier = info.get("tier", "") if isinstance(info, dict) else ""
            suffix = f" ({tier})" if tier else ""
            print(f"{stage}: {state}{suffix}")
    siblings = result.get("sibling_jobs")
    if isinstance(siblings, list) and siblings:
        print("sibling_jobs:")
        for row in siblings:
            if isinstance(row, dict):
                print(
                    f"  {row.get('name')}: "
                    f"active={row.get('active', 0)} "
                    f"succeeded={row.get('succeeded', 0)} "
                    f"failed={row.get('failed', 0)}"
                )


def watch_sim2real_status(
    run_id: str,
    *,
    watch: bool = False,
    interval: float = 10.0,
    json_output: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Poll staged run progress until terminal or one-shot."""

    import time

    while True:
        result = get_sim2real_workflow_status(run_id, **kwargs)
        emit_sim2real_status(result, json_output=json_output)
        if not watch or status_is_terminal(str(result.get("status", ""))):
            return result
        time.sleep(interval)


def looks_like_sim2real_run(run_id: str) -> bool:
    normalized = run_id.strip().lower()
    return normalized.startswith("sim2real") or "sim2real-" in normalized


def sim2real_run_exists(
    run_id: str,
    *,
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
    k8s_context: str = "",
    kubeconfig: str | Path = "",
) -> bool:
    """Best-effort detection for ``npa workbench workflow status`` routing."""

    if looks_like_sim2real_run(run_id):
        return True
    try:
        operator = load_operator_config()
    except ValueError:
        operator = None
    bucket = s3_bucket or (operator.bucket if operator else "")
    endpoint = s3_endpoint or (operator.endpoint_url if operator else DEFAULT_S3_ENDPOINT)
    context = k8s_context or (operator.k8s_context if operator else "")
    if context:
        kcfg = Path(kubeconfig) if kubeconfig else resolve_kubeconfig(context)
        k8s = _k8s_orchestrator_status(
            run_id=run_id,
            context=context,
            kubeconfig=kcfg,
        )
        if k8s.get("found"):
            return True
    if bucket:
        client = StorageClient.from_environment(endpoint_url=endpoint)
        prefix = f"{s3_prefix.rstrip('/')}/{run_id}/"
        if _s3_prefix_nonempty(client, bucket, prefix):
            return True
    return False


def _missing_k8s_status(run_id: str) -> dict[str, Any]:
    return {
        "job_name": orchestrator_job_name(run_id),
        "found": False,
        "phase": "MISSING",
        "active": 0,
        "succeeded": 0,
        "failed": 0,
        "pod_phase": "",
        "pod_reason": "",
    }


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
    run_id = normalize_staged_run_id(run_id)

    try:
        operator = load_operator_config()
    except ValueError:
        operator = None

    bucket = s3_bucket or (operator.bucket if operator else "")
    endpoint = s3_endpoint or (
        operator.endpoint_url if operator else DEFAULT_S3_ENDPOINT
    )
    context = k8s_context or (operator.k8s_context if operator else "")
    if not bucket:
        raise ValueError(
            "S3 bucket required (--s3-bucket or storage.bucket in ~/.npa/config.yaml)"
        )

    stages = _stage_states(
        bucket=bucket,
        run_id=run_id,
        s3_prefix=s3_prefix,
        endpoint=endpoint,
    )
    k8s = _missing_k8s_status(run_id)
    siblings: list[dict[str, Any]] = []
    if context:
        try:
            kcfg = Path(kubeconfig) if kubeconfig else resolve_kubeconfig(context)
        except ValueError:
            kcfg = None
        if kcfg is not None:
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
