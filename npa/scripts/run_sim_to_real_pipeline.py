#!/usr/bin/env python3
"""Submit or render the tiered sim-to-real SkyPilot pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from npa.orchestration.skypilot import (
    WorkflowResult,  # noqa: F401 - kept for tests and downstream wrapper imports.
    cleanup_all_for_run,
    submit_workflow,
    workflow_status,
)
from npa.orchestration.skypilot._bin import (
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    resolve_sky_bin,
)
from npa.workflows.sim_to_real import (
    DEFAULT_EVAL_BACKEND,
    DEFAULT_FEEDBACK_SOURCE,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_SIM_BACKEND,
    DEFAULT_SPLIT_FRACTION,
    DEFAULT_THRESHOLD,
    DEFAULT_VLM_EVAL_BACKEND,
    DEFAULT_VLM_EVAL_MODEL,
    artifact_uris,
    build_config_from_env,
    default_policy_image,
    default_s3_prefix,
)


DEFAULT_YAML = (
    Path(__file__).resolve().parents[1]
    / "workflows"
    / "workbench"
    / "skypilot"
    / "sim-to-real-pipeline.yaml"
)
DEFAULT_BUCKET = os.environ.get("NPA_S3_BUCKET", "your-bucket-name")
TERMINAL_STATUSES = {
    "SUCCEEDED",
    "CANCELLED",
    "FAILED",
    "FAILED_SETUP",
    "FAILED_PRECHECKS",
    "FAILED_NO_RESOURCE",
    "FAILED_CONTROLLER",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _submit_and_wait(args)
    except (SkyPilotNotInstalledError, SkyPilotConfigError, SkyPilotVersionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("For a no-infrastructure check, rerun with --render-only.", file=sys.stderr)
        return 2


def render_workflow(
    yaml_path: Path,
    *,
    run_id: str,
    bucket: str = DEFAULT_BUCKET,
    s3_prefix: str = "",
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    input_data_uri: str = "",
    policy_image: str = "",
    sim_backend: str = DEFAULT_SIM_BACKEND,
    eval_backend: str = DEFAULT_EVAL_BACKEND,
    feedback_source: str = DEFAULT_FEEDBACK_SOURCE,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
    env_count: int = 10,
    episodes: int = 4,
    train_steps: int = 50,
    eval_episodes: int = 2,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 42,
    gpu: str = "H100:1",
    vlm_eval_backend: str = DEFAULT_VLM_EVAL_BACKEND,
    vlm_eval_model: str = DEFAULT_VLM_EVAL_MODEL,
    vlm_eval_endpoint_url: str = "",
    vlm_eval_frame_selection: str = "keyframes",
    vlm_eval_max_frames: int = 4,
    vlm_eval_score: float | None = None,
    trainer_command: str = "",
) -> list[dict[str, Any]]:
    """Return SkyPilot YAML documents with concrete run settings injected."""

    resolved_prefix = s3_prefix or default_s3_prefix(run_id)
    resolved_policy = policy_image or default_policy_image()
    config = build_config_from_env(
        run_id=run_id,
        s3_endpoint=s3_endpoint,
        s3_bucket=bucket,
        s3_prefix=resolved_prefix,
        input_data_uri=input_data_uri,
        policy_image=resolved_policy,
        sim_backend=sim_backend,
        eval_backend=eval_backend,
        feedback_source=feedback_source,
        split_fraction=split_fraction,
        env_count=env_count,
        episodes=episodes,
        train_steps=train_steps,
        eval_episodes=eval_episodes,
        threshold=threshold,
        seed=seed,
        gpu=gpu,
        vlm_eval_backend=vlm_eval_backend,
        vlm_eval_model=vlm_eval_model,
        vlm_eval_endpoint_url=vlm_eval_endpoint_url,
        vlm_eval_frame_selection=vlm_eval_frame_selection,
        vlm_eval_max_frames=vlm_eval_max_frames,
        vlm_eval_score=vlm_eval_score,
        trainer_command=trainer_command,
    )
    config.validate()
    paths = artifact_uris(config)
    docs = _load_yaml_documents(yaml_path)
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        envs = doc.get("envs")
        if isinstance(envs, dict):
            envs.update(
                {
                    "NPA_SIM_TO_REAL_RUN_ID": run_id,
                    "NEBIUS_S3_ENDPOINT": s3_endpoint,
                    "AWS_ENDPOINT_URL": s3_endpoint,
                    "NPA_S3_BUCKET": bucket,
                    "S3_PREFIX": resolved_prefix,
                    "PIPELINE_ROOT_URI": paths.get("root", ""),
                    "INPUT_DATA_URI": input_data_uri,
                    "RAW_ENVS_URI": paths.get("raw_envs", ""),
                    "TRAIN_ENVS_URI": paths.get("train_envs", ""),
                    "HELDOUT_ENVS_URI": paths.get("heldout_envs", ""),
                    "POLICY_IMAGE": resolved_policy,
                    "CHECKPOINT_URI": paths.get("checkpoint", ""),
                    "RERUN_RRD_PATH": paths.get("rrd", ""),
                    "SIM_BACKEND": sim_backend,
                    "EVAL_BACKEND": eval_backend,
                    "FEEDBACK_SOURCE": feedback_source,
                    "SPLIT_FRACTION": str(split_fraction),
                    "ENV_COUNT": str(env_count),
                    "EPISODES": str(episodes),
                    "TRAIN_STEPS": str(train_steps),
                    "EVAL_EPISODES": str(eval_episodes),
                    "SUCCESS_THRESHOLD": str(threshold),
                    "SEED": str(seed),
                    "GPU": gpu,
                    "VLM_EVAL_BACKEND": vlm_eval_backend,
                    "VLM_EVAL_MODEL": vlm_eval_model,
                    "VLM_EVAL_ENDPOINT_URL": vlm_eval_endpoint_url,
                    "VLM_EVAL_FRAME_SELECTION": vlm_eval_frame_selection,
                    "VLM_EVAL_MAX_FRAMES": str(vlm_eval_max_frames),
                    "VLM_EVAL_SCORE": "" if vlm_eval_score is None else str(vlm_eval_score),
                    "CUSTOM_LEROBOT_TRAINER_COMMAND": trainer_command,
                }
            )
        if doc.get("name") == "s2r-policy-feedback-update":
            resources = doc.setdefault("resources", {})
            if isinstance(resources, dict):
                resources["image_id"] = (
                    resolved_policy if resolved_policy.startswith("docker:") else f"docker:{resolved_policy}"
                )
    return docs


def output_paths(
    run_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    s3_prefix: str = "",
    s3_endpoint: str = DEFAULT_S3_ENDPOINT,
    policy_image: str = "",
) -> dict[str, str]:
    """Return the pipeline S3 output paths."""

    config = build_config_from_env(
        run_id=run_id,
        s3_endpoint=s3_endpoint,
        s3_bucket=bucket,
        s3_prefix=s3_prefix or default_s3_prefix(run_id),
        policy_image=policy_image or default_policy_image(),
    )
    return artifact_uris(config)


def _submit_and_wait(args: argparse.Namespace) -> int:
    run_id = args.run_id or _default_run_id()
    docs = render_workflow(
        args.yaml_path,
        run_id=run_id,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint=args.s3_endpoint,
        input_data_uri=args.input_data_uri,
        policy_image=args.policy_image,
        sim_backend=args.sim_backend,
        eval_backend=args.eval_backend,
        feedback_source=args.feedback_source,
        split_fraction=args.split_fraction,
        env_count=args.env_count,
        episodes=args.episodes,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        threshold=args.threshold,
        seed=args.seed,
        gpu=args.gpu,
        vlm_eval_backend=args.vlm_eval_backend,
        vlm_eval_model=args.vlm_eval_model,
        vlm_eval_endpoint_url=args.vlm_eval_endpoint_url,
        vlm_eval_frame_selection=args.vlm_eval_frame_selection,
        vlm_eval_max_frames=args.vlm_eval_max_frames,
        vlm_eval_score=args.vlm_eval_score,
        trainer_command=args.trainer_command,
    )
    outputs = output_paths(
        run_id,
        bucket=args.bucket,
        s3_prefix=args.s3_prefix,
        s3_endpoint=args.s3_endpoint,
        policy_image=args.policy_image,
    )

    if args.render_only:
        render_dir = Path(tempfile.mkdtemp(prefix=f"npa-sim-to-real-{run_id}-"))
        rendered_yaml = render_dir / "sim-to-real-pipeline.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        print(json.dumps({"run_id": run_id, "rendered_yaml": str(rendered_yaml), "outputs": outputs}, indent=2))
        return 0

    with tempfile.TemporaryDirectory(prefix=f"npa-sim-to-real-{run_id}-") as tmp:
        rendered_yaml = Path(tmp) / "sim-to-real-pipeline.rendered.yaml"
        _write_yaml_documents(rendered_yaml, docs)
        sky_bin = str(resolve_sky_bin(args.sky_bin or os.environ.get("NPA_SKYPILOT_BIN")))
        result = submit_workflow(
            rendered_yaml,
            run_id,
            isolated_config_dir=args.isolated_config_dir,
            sky_bin=sky_bin,
            timeout=args.submit_timeout,
        )
        summary: dict[str, Any] = {
            "run_id": run_id,
            "submit": result.__dict__,
            "outputs": outputs,
        }
        if not result.ok or result.status != "SUBMITTED":
            print(json.dumps(summary, indent=2, sort_keys=True))
            return result.returncode or 1

        deadline = time.monotonic() + args.wait_timeout
        final = result
        while time.monotonic() < deadline:
            final = workflow_status(
                result.job_id,
                isolated_config_dir=args.isolated_config_dir,
                config_path=Path(result.log_paths["config"]) if result.log_paths.get("config") else None,
                sky_bin=sky_bin,
            )
            if final.status in TERMINAL_STATUSES:
                break
            time.sleep(args.poll_interval)
        summary["final"] = final.__dict__

        if args.cleanup:
            cleanup = cleanup_all_for_run(
                run_id,
                isolated_config_dir=args.isolated_config_dir,
                config_path=Path(result.log_paths["config"]) if result.log_paths.get("config") else None,
                sky_bin=sky_bin,
            )
            summary["cleanup"] = cleanup.__dict__

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if final.status == "SUCCEEDED" else 1


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError(f"SkyPilot YAML documents must be mappings: {path}")
    return docs


def _write_yaml_documents(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text(yaml.safe_dump_all(docs, sort_keys=False), encoding="utf-8")


def _default_run_id() -> str:
    return "sim-to-real-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", "--yaml-path", dest="yaml_path", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--s3-prefix", default="")
    parser.add_argument("--s3-endpoint", default=DEFAULT_S3_ENDPOINT)
    parser.add_argument("--input-data-uri", default="")
    parser.add_argument("--policy-image", default="")
    parser.add_argument("--sim-backend", default=DEFAULT_SIM_BACKEND)
    parser.add_argument("--eval-backend", default=DEFAULT_EVAL_BACKEND)
    parser.add_argument("--feedback-source", default=DEFAULT_FEEDBACK_SOURCE)
    parser.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--env-count", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="H100:1")
    parser.add_argument("--vlm-eval-backend", default=DEFAULT_VLM_EVAL_BACKEND)
    parser.add_argument("--vlm-eval-model", default=DEFAULT_VLM_EVAL_MODEL)
    parser.add_argument("--vlm-eval-endpoint-url", default="")
    parser.add_argument("--vlm-eval-frame-selection", default="keyframes")
    parser.add_argument("--vlm-eval-max-frames", type=int, default=4)
    parser.add_argument("--vlm-eval-score", type=float, default=None)
    parser.add_argument("--trainer-command", default="")
    parser.add_argument("--sky-bin", default="")
    parser.add_argument("--isolated-config-dir", type=Path, default=None)
    parser.add_argument("--submit-timeout", type=int, default=1800)
    parser.add_argument("--wait-timeout", type=int, default=43200)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
