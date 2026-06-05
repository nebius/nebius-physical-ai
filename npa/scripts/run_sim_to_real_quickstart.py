#!/usr/bin/env python3
"""One-command H100 sim-to-real quickstart.

This script is intentionally orchestration glue. It renders and submits the
checked-in sim-to-real SkyPilot YAML, runs the existing real LeRobot loop on a
small H100 job, fetches the uploaded report from S3, and always tears down the
run-scoped SkyPilot cluster with ``sky down``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_sim_to_real_pipeline import (  # noqa: E402
    DEFAULT_YAML,
    _s3_secret_envs,
    _write_yaml_documents,
    output_paths,
    render_workflow,
)

from npa.clients.credentials import load_credentials, storage_endpoint_url  # noqa: E402
from npa.clients.storage import StorageClient  # noqa: E402
from npa.cli.skypilot import SkyPilotBootstrapError, bootstrap_skypilot  # noqa: E402
from npa.orchestration.skypilot import WorkflowResult  # noqa: E402
from npa.orchestration.skypilot._bin import (  # noqa: E402
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_config,
    resolve_sky_bin,
)
from npa.orchestration.skypilot.cleanup import CleanupResult, run_tag, sky_environment  # noqa: E402
from npa.orchestration.skypilot.signal_teardown import (  # noqa: E402
    SignalTeardown,
    install_teardown_signal_handlers,
    restore_signal_handlers,
)
from npa.workflows.sim_to_real import (  # noqa: E402
    DEFAULT_EVAL_BACKEND,
    DEFAULT_FEEDBACK_SOURCE,
    DEFAULT_FEEDBACK_TYPE,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_SIM_BACKEND,
    default_policy_image,
    default_s3_prefix,
)

DEFAULT_RUN_PREFIX = "s2r-quickstart"
DEFAULT_TRAIN_STEPS = 20
DEFAULT_TRAIN_STEP_BUDGET = 20
DEFAULT_MAX_TRAINING_ITERATIONS = 1
DEFAULT_EVAL_EPISODES = 1
DEFAULT_TRAIN_BATCH_SIZE = 4
DEFAULT_TRAIN_NUM_WORKERS = 2
DEFAULT_GPU = "H100:1"
DEFAULT_SOURCE_REPO = "https://github.com/nebius/nebius-physical-ai.git"
DEFAULT_SOURCE_REF = "main"
STORAGE_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_BUCKET",
    "NPA_S3_BUCKET",
    "S3_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
    "NEBIUS_S3_ENDPOINT",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or _default_run_id()
    started = time.perf_counter()
    try:
        storage = _resolve_storage(args, run_id)
        sky_bin = _resolve_or_bootstrap_sky_bin(args)
        sky_config = resolve_config(sky_bin=sky_bin, isolated_config_dir=args.isolated_config_dir)
    except (
        QuickstartConfigError,
        SkyPilotConfigError,
        SkyPilotNotInstalledError,
        SkyPilotVersionError,
        SkyPilotBootstrapError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    previous_env = _apply_storage_environment(storage)
    try:
        teardown = SignalTeardown(
            run_id=run_id,
            isolated_config_dir=sky_config.isolated_config_dir,
            config_path=sky_config.global_config_path,
            sky_bin=sky_bin,
            timeout=float(args.teardown_timeout),
            poll_interval=float(args.teardown_poll_interval),
        )
        previous_handlers = install_teardown_signal_handlers(teardown.teardown)
        cleanup_payload: dict[str, Any] = {}
        try:
            try:
                result = _submit_and_wait(args, run_id=run_id, storage=storage, sky_bin=str(sky_bin), teardown=teardown)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                result = WorkflowResult(status="FAILED", returncode=1, error=str(exc))
            finally:
                cleanup = teardown.teardown()
                cleanup_payload = _cleanup_payload(cleanup)
        finally:
            restore_signal_handlers(previous_handlers)

        wall_clock = round(time.perf_counter() - started, 2)
        report = (
            {}
            if args.render_only or result.status != "SUCCEEDED"
            else _download_report(storage, run_id=run_id, output_dir=args.local_output_dir)
        )
        summary = _result_summary(
            run_id=run_id,
            wall_clock_seconds=wall_clock,
            workflow=result,
            storage=storage,
            report=report,
            cleanup=cleanup_payload,
        )
        _print_summary(summary, as_json=args.output_json)
        if args.render_only:
            return 0
        return 0 if result.status == "SUCCEEDED" and not cleanup_payload.get("errors") and report else 1
    finally:
        _restore_environment(previous_env)


def _submit_and_wait(
    args: argparse.Namespace,
    *,
    run_id: str,
    storage: "StorageSettings",
    sky_bin: str,
    teardown: SignalTeardown,
) -> WorkflowResult:
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        bucket=storage.bucket,
        s3_prefix=storage.prefix,
        s3_endpoint=storage.endpoint,
        input_data_uri=args.input_data_uri,
        dataset_repo_id=args.dataset_repo_id,
        dataset_revision=args.dataset_revision,
        policy_image=args.policy_image or default_policy_image(),
        sim_backend=args.sim_backend,
        eval_backend=args.eval_backend,
        feedback_source=args.feedback_source,
        feedback_type=args.feedback_type,
        split_fraction=args.split_fraction,
        env_count=args.env_count,
        episodes=args.episodes,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        threshold=args.threshold,
        seed=args.seed,
        gpu=args.gpu,
        gpu_failover=args.gpu_failover,
        max_training_iterations=args.max_training_iterations,
        train_step_budget=args.train_step_budget,
        min_eval_improvement=args.min_eval_improvement,
        policy_type=args.policy_type,
        train_batch_size=args.train_batch_size,
        train_num_workers=args.train_num_workers,
        policy_device=args.policy_device,
        sky_bin=sky_bin,
        task_cloud=args.task_cloud,
        vlm_eval_backend=args.vlm_eval_backend,
        vlm_eval_model=args.vlm_eval_model,
        vlm_eval_endpoint_url=args.vlm_eval_endpoint_url,
        vlm_eval_frame_selection=args.vlm_eval_frame_selection,
        vlm_eval_max_frames=args.vlm_eval_max_frames,
        vlm_eval_score=args.vlm_eval_score,
        trainer_command=args.trainer_command,
        byo_feedback_endpoint_url=args.byo_feedback_endpoint_url,
        byo_feedback_command=args.byo_feedback_command,
        byo_feedback_mode=args.byo_feedback_mode,
        rerun_max_frames_per_episode=args.rerun_max_frames_per_episode,
    )
    _prepare_quickstart_docs(
        docs,
        run_id=run_id,
        source_repo=args.source_repo,
        source_ref=args.source_ref,
    )

    if args.render_only:
        render_dir = Path(tempfile.mkdtemp(prefix=f"npa-{run_id}-"))
        rendered_yaml = render_dir / "sim-to-real-quickstart.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        print(json.dumps({"run_id": run_id, "rendered_yaml": str(rendered_yaml)}, indent=2, sort_keys=True))
        return WorkflowResult(status="SUCCEEDED", job_id="render-only")

    with tempfile.TemporaryDirectory(prefix=f"npa-{run_id}-") as tmp:
        rendered_yaml = Path(tmp) / "sim-to-real-quickstart.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        cluster_name = run_tag(run_id)
        cmd = [sky_bin, "launch", "--cluster", cluster_name, "--name", cluster_name, "--yes"]
        for secret_name in _s3_secret_envs():
            if os.environ.get(secret_name):
                cmd.extend(["--secret", secret_name])
        cmd.append(str(rendered_yaml))
        teardown.mark_launched()
        try:
            result = subprocess.run(
                cmd,
                env=sky_environment(args.isolated_config_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.wait_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return WorkflowResult(
                status="TIMEOUT",
                job_id=cluster_name,
                returncode=124,
                error=f"Timed out after {args.wait_timeout}s waiting for SkyPilot launch.",
                submitted_yaml_path=str(rendered_yaml),
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else "",
            )
        status = "SUCCEEDED" if result.returncode == 0 else "FAILED"
        return WorkflowResult(
            status=status,
            job_id=cluster_name,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            submitted_yaml_path=str(rendered_yaml),
        )


def _prepare_quickstart_docs(
    docs: list[dict[str, Any]],
    *,
    run_id: str,
    source_repo: str,
    source_ref: str,
) -> None:
    task_name = run_tag(run_id)
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        doc["name"] = task_name
        envs = doc.get("envs")
        if isinstance(envs, dict):
            envs["NPA_SOURCE_REPO"] = source_repo
            envs["NPA_SOURCE_REF"] = source_ref


class QuickstartConfigError(ValueError):
    """Raised when quickstart prerequisites are missing."""


class StorageSettings:
    """Resolved S3-compatible storage settings for the quickstart."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        endpoint: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint = endpoint
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key


def _resolve_storage(args: argparse.Namespace, run_id: str) -> StorageSettings:
    credentials = load_credentials(path=args.credential_path)
    bucket_value = args.bucket or os.environ.get("S3_BUCKET") or os.environ.get("NPA_S3_BUCKET") or credentials.s3_bucket
    bucket, credential_prefix = _split_bucket_and_prefix(bucket_value)
    if not bucket:
        raise QuickstartConfigError(
            "S3 bucket is not configured. Set S3_BUCKET/NPA_S3_BUCKET, pass --bucket, "
            "or set storage.bucket in ~/.npa/credentials.yaml."
        )
    endpoint = storage_endpoint_url(
        args.s3_endpoint
        or os.environ.get("S3_ENDPOINT_URL", "")
        or os.environ.get("AWS_ENDPOINT_URL", "")
        or os.environ.get("NEBIUS_S3_ENDPOINT", "")
        or credentials.s3_endpoint
        or DEFAULT_S3_ENDPOINT
    )
    prefix = args.s3_prefix or _join_s3_prefix(credential_prefix, default_s3_prefix(run_id))
    access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or credentials.s3_access_key_id
    secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or credentials.s3_secret_access_key
    if not access_key_id or not secret_access_key:
        raise QuickstartConfigError(
            "S3 credentials are not configured. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            "or storage access keys in ~/.npa/credentials.yaml."
        )
    return StorageSettings(
        bucket=bucket,
        prefix=prefix,
        endpoint=endpoint,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )


def _resolve_or_bootstrap_sky_bin(args: argparse.Namespace) -> Path:
    requested = args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN", "")
    try:
        return resolve_sky_bin(requested or None)
    except SkyPilotNotInstalledError:
        if requested or args.no_bootstrap_skypilot:
            raise
        result = bootstrap_skypilot()
        os.environ.setdefault("NPA_SKYPILOT_BIN", str(result.sky_bin))
        return result.sky_bin


def _apply_storage_environment(storage: StorageSettings) -> dict[str, str | None]:
    previous = {key: os.environ.get(key) for key in STORAGE_ENV_KEYS}
    os.environ["AWS_ACCESS_KEY_ID"] = storage.access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = storage.secret_access_key
    os.environ["S3_BUCKET"] = storage.bucket
    os.environ["NPA_S3_BUCKET"] = storage.bucket
    os.environ["S3_ENDPOINT_URL"] = storage.endpoint
    os.environ["AWS_ENDPOINT_URL"] = storage.endpoint
    os.environ["NEBIUS_S3_ENDPOINT"] = storage.endpoint
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _download_report(storage: StorageSettings, *, run_id: str, output_dir: Path | None) -> dict[str, Any]:
    paths = output_paths(run_id, bucket=storage.bucket, s3_prefix=storage.prefix, s3_endpoint=storage.endpoint)
    report_uri = paths.get("report", "")
    if not report_uri:
        return {}
    target_dir = output_dir or Path(tempfile.mkdtemp(prefix=f"npa-{run_id}-report-"))
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "sim-to-real-report.json"
    client = StorageClient.from_environment(
        endpoint_url=storage.endpoint,
        aws_access_key_id=storage.access_key_id,
        aws_secret_access_key=storage.secret_access_key,
    )
    deadline = time.monotonic() + 120
    while True:
        try:
            client.download_path(report_uri, str(report_path))
            if report_path.exists() and report_path.stat().st_size > 0:
                return json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return {}
        time.sleep(5)


def _result_summary(
    *,
    run_id: str,
    wall_clock_seconds: float,
    workflow: WorkflowResult,
    storage: StorageSettings,
    report: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    paths = output_paths(run_id, bucket=storage.bucket, s3_prefix=storage.prefix, s3_endpoint=storage.endpoint)
    outer_loop = report.get("outer_loop", {}) if isinstance(report, dict) else {}
    feedback = report.get("feedback", {}) if isinstance(report, dict) else {}
    metric_value = outer_loop.get("score", feedback.get("score"))
    summary = {
        "run_id": run_id,
        "workflow_status": workflow.status,
        "wall_clock_seconds": wall_clock_seconds,
        "metric": {
            "name": "task_success_score",
            "value": metric_value,
            "decision": outer_loop.get("decision", ""),
            "trend": outer_loop.get("trend", []),
        },
        "artifacts": {
            "checkpoint": paths.get("checkpoint", ""),
            "report": paths.get("report", ""),
            "rrd": paths.get("rrd", ""),
        },
        "teardown": {
            "cluster_absent": not cleanup.get("errors"),
            **cleanup,
        },
    }
    if workflow.status != "SUCCEEDED":
        summary["error"] = workflow.error or _tail_text(workflow.stderr, workflow.stdout)
    return summary


def _print_summary(summary: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print("sim-to-real quickstart result")
    print(f"run_id: {summary['run_id']}")
    print(f"workflow_status: {summary['workflow_status']}")
    print(f"wall_clock_seconds: {summary['wall_clock_seconds']}")
    metric = summary["metric"]
    print(f"metric: {metric['name']}={metric['value']}")
    if metric.get("trend"):
        print(f"metric_trend: {metric['trend']}")
    print(f"checkpoint_uri: {summary['artifacts']['checkpoint']}")
    print(f"report_uri: {summary['artifacts']['report']}")
    print(f"rrd_uri: {summary['artifacts']['rrd']}")
    print(f"teardown: cluster_absent={summary['teardown']['cluster_absent']}")
    if summary.get("error"):
        print(f"error: {summary['error']}")


def _cleanup_payload(cleanup: CleanupResult | Any) -> dict[str, Any]:
    try:
        return asdict(cleanup)
    except TypeError:
        return {"resources_removed": [], "errors": [], "commands": []}


def _tail_text(*values: str, max_lines: int = 24, max_chars: int = 4000) -> str:
    text = "\n".join(value.strip() for value in values if value and value.strip())
    if not text:
        return ""
    lines = text.splitlines()[-max_lines:]
    return "\n".join(lines)[-max_chars:]


def _split_bucket_and_prefix(value: str) -> tuple[str, str]:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return "", ""
    if raw.startswith("s3://"):
        parsed = urlparse(raw)
        return parsed.netloc, parsed.path.strip("/")
    bucket, _, prefix = raw.partition("/")
    return bucket, prefix.strip("/")


def _join_s3_prefix(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part and part.strip("/"))


def _default_run_id() -> str:
    return DEFAULT_RUN_PREFIX + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", "--yaml-path", dest="yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bucket", default="")
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--s3-endpoint", default="")
    parser.add_argument("--credential-path", type=Path, default=None)
    parser.add_argument("--local-output-dir", type=Path, default=None)
    parser.add_argument("--source-repo", default=os.environ.get("NPA_SOURCE_REPO", DEFAULT_SOURCE_REPO))
    parser.add_argument("--source-ref", default=os.environ.get("NPA_SOURCE_REF", DEFAULT_SOURCE_REF))
    parser.add_argument("--input-data-uri", default="")
    parser.add_argument("--dataset-repo-id", default="lerobot/pusht")
    parser.add_argument("--dataset-revision", default="7628202a2180972f291ba1bc6723834921e72c19")
    parser.add_argument("--policy-image", default=os.environ.get("POLICY_IMAGE", ""))
    parser.add_argument("--sim-backend", default=DEFAULT_SIM_BACKEND)
    parser.add_argument("--eval-backend", default=DEFAULT_EVAL_BACKEND)
    parser.add_argument("--feedback-source", default=DEFAULT_FEEDBACK_SOURCE)
    parser.add_argument("--feedback-type", default=DEFAULT_FEEDBACK_TYPE)
    parser.add_argument("--split-fraction", type=float, default=0.8)
    parser.add_argument("--env-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default=os.environ.get("NPA_GPU_TYPE", DEFAULT_GPU))
    parser.add_argument("--gpu-failover", default=os.environ.get("NPA_GPU_FAILOVER", ""))
    parser.add_argument("--max-training-iterations", type=int, default=DEFAULT_MAX_TRAINING_ITERATIONS)
    parser.add_argument("--train-step-budget", type=int, default=DEFAULT_TRAIN_STEP_BUDGET)
    parser.add_argument("--min-eval-improvement", type=float, default=0.0)
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--train-num-workers", type=int, default=DEFAULT_TRAIN_NUM_WORKERS)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--task-cloud", choices=("kubernetes", "nebius"), default="nebius")
    parser.add_argument("--controller-backend", choices=("kubernetes", "nebius"), default="nebius")
    parser.add_argument("--vlm-eval-backend", default="stub")
    parser.add_argument("--vlm-eval-model", default="vlm-eval-stub")
    parser.add_argument("--vlm-eval-endpoint-url", default="")
    parser.add_argument("--vlm-eval-frame-selection", default="keyframes")
    parser.add_argument("--vlm-eval-max-frames", type=int, default=4)
    parser.add_argument("--vlm-eval-score", type=float, default=None)
    parser.add_argument("--trainer-command", default="")
    parser.add_argument("--byo-feedback-endpoint-url", default="")
    parser.add_argument("--byo-feedback-command", default="")
    parser.add_argument("--byo-feedback-mode", choices=("provided-rollout", "self-rollout"), default="provided-rollout")
    parser.add_argument("--rerun-max-frames-per-episode", type=int, default=8)
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--isolated-config-dir", type=Path, default=None)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    parser.add_argument("--wait-timeout", type=int, default=14400)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--teardown-timeout", type=int, default=900)
    parser.add_argument("--teardown-poll-interval", type=int, default=10)
    parser.add_argument("--no-bootstrap-skypilot", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output-json", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
