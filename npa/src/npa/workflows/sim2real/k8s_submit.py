"""Submit Sim2Real staged runs via the operator K8s Job manifest."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from npa.workflows.sim2real.constants import DEFAULT_PREFIX
from npa.workflows.sim2real.monitor import load_operator_config, orchestrator_job_name


@dataclass(frozen=True)
class Sim2RealSubmitResult:
    run_id: str
    job_name: str
    k8s_context: str
    run_prefix_uri: str
    status: str = "submitted"
    log_path: str = ""
    manifest_path: str = ""


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "npa" / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (npa/pyproject.toml)")


def submit_sim2real_staged_job(
    *,
    run_id: str = "",
    trigger_dataset_uri: str = "",
    trigger_dataset_id: str = "lerobot/pusht",
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
    k8s_context: str = "",
    registry: str = "",
    inner_iterations: int | None = None,
    outer_iterations: int | None = None,
    env_count: int | None = None,
    launch_monitor: bool = False,
) -> Sim2RealSubmitResult:
    """Apply the direct-K8s sim2real Job used by the RTX operator pack."""

    operator = load_operator_config()
    root = _repo_root()
    script = root / "ops" / "private" / "sim2real-rtxpro" / "submit-k8s-staged-job.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing operator submit script: {script}")

    bucket = s3_bucket or operator.bucket
    endpoint = s3_endpoint or operator.endpoint_url
    context = k8s_context or operator.k8s_context
    reg = registry or operator.registry
    if not reg:
        raise ValueError("storage.registry is not set in ~/.npa/config.yaml")

    resolved_run_id = run_id or os.environ.get("RUN_ID") or ""
    env = dict(os.environ)
    env.update(
        {
            "S3_BUCKET": bucket,
            "S3_ENDPOINT": endpoint,
            "REGISTRY": reg,
            "KUBECONTEXT": context,
            "S3_PREFIX": s3_prefix,
            "LAUNCH_MONITOR": "1" if launch_monitor else "0",
            "NPA_SIM2REAL_TRIGGER_DATASET_ID": trigger_dataset_id,
        }
    )
    if resolved_run_id:
        env["RUN_ID"] = resolved_run_id
    trigger = trigger_dataset_uri or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI") or os.environ.get(
        "TRIGGER_DATASET_URI", ""
    )
    if trigger:
        env["NPA_SIM2REAL_TRIGGER_DATASET_URI"] = trigger
        env["TRIGGER_DATASET_URI"] = trigger
    if inner_iterations is not None:
        env["INNER_ITERATIONS"] = str(inner_iterations)
    if outer_iterations is not None:
        env["OUTER_ITERATIONS"] = str(outer_iterations)
    if env_count is not None:
        env["NPA_ENV_COUNT"] = str(env_count)

    proc = subprocess.run(
        ["bash", str(script)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"sim2real K8s submit failed:\n{output}")

    parsed_run_id = resolved_run_id
    job_name = ""
    for line in output.splitlines():
        if line.startswith("run_id="):
            parsed_run_id = line.split("=", 1)[1].strip()
        if line.startswith("job="):
            job_name = line.split("=", 1)[1].strip()
    if not parsed_run_id:
        raise RuntimeError(f"submit script did not return run_id:\n{output}")
    if not job_name:
        job_name = orchestrator_job_name(parsed_run_id)

    log_path = f"/tmp/sim2real-cluster/{parsed_run_id}.log"
    manifest_path = f"/tmp/sim2real-cluster/{job_name}.yaml"
    prefix_uri = f"s3://{bucket}/{s3_prefix.rstrip('/')}/{parsed_run_id}/"
    return Sim2RealSubmitResult(
        run_id=parsed_run_id,
        job_name=job_name,
        k8s_context=context,
        run_prefix_uri=prefix_uri,
        log_path=log_path,
        manifest_path=manifest_path,
    )


def is_sim2real_runbook(yaml_path: Path) -> bool:
    """True when ``yaml_path`` is the staged Sim2Real runbook (direct K8s route)."""

    path = str(yaml_path.resolve()).lower()
    return "/sim2real/" in path and yaml_path.name.lower().startswith("runbook")


def status_monitor_command(run_id: str) -> str:
    return f"python -m npa.workflows.sim2real status {run_id} --watch"


def submit_sim2real_from_workflow_vars(
    *,
    run_id: str,
    substitutions: dict[str, str],
    s3_bucket: str = "",
    s3_prefix: str = DEFAULT_PREFIX,
    s3_endpoint: str = "",
) -> Sim2RealSubmitResult:
    """Submit a staged Sim2Real run using workflow ``--var KEY=VALUE`` substitutions."""

    trigger_uri = (
        substitutions.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or substitutions.get("TRIGGER_DATASET_URI")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_URI")
        or os.environ.get("TRIGGER_DATASET_URI")
        or ""
    )
    trigger_id = (
        substitutions.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or substitutions.get("TRIGGER_DATASET_ID")
        or os.environ.get("NPA_SIM2REAL_TRIGGER_DATASET_ID")
        or os.environ.get("TRIGGER_DATASET_ID")
        or "lerobot/pusht"
    )
    inner = substitutions.get("INNER_ITERATIONS")
    outer = substitutions.get("OUTER_ITERATIONS")
    env_count = substitutions.get("NPA_ENV_COUNT")
    return submit_sim2real_staged_job(
        run_id=run_id,
        trigger_dataset_uri=trigger_uri,
        trigger_dataset_id=trigger_id,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix or DEFAULT_PREFIX,
        s3_endpoint=s3_endpoint,
        inner_iterations=int(inner) if inner else None,
        outer_iterations=int(outer) if outer else None,
        env_count=int(env_count) if env_count else None,
        launch_monitor=False,
    )
