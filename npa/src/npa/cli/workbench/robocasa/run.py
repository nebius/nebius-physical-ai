"""Run a RoboCasa capability."""

from __future__ import annotations

import time
from typing import Any

import typer

from npa.cli.path_contract import PathContractError, validate_read_path, validate_write_path
from npa.workbench.robocasa.schemas import (
    DEFAULT_ENV_ID,
    DEFAULT_ITERATIONS,
    DEFAULT_NUM_ENVS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_TOKEN_ENV,
    RoboCasaRunRequest,
)

from npa.cli.workbench.robocasa.helpers import OutputFormat, emit, fail, request_json, resolve_endpoint

RUN_DONE = "completed"
RUN_FAILED = "failed"


def run_cmd(
    capability: str = typer.Option(..., "--capability", help="RoboCasa capability to run."),
    env_id: str = typer.Option(DEFAULT_ENV_ID, "--env-id", help="RoboCasa Gymnasium env id."),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="S3 output path for artifacts (--output-uri is a compatibility alias).",
    ),
    iterations: int = typer.Option(DEFAULT_ITERATIONS, "--iterations", help="Number of rollout iterations."),
    num_envs: int = typer.Option(DEFAULT_NUM_ENVS, "--num-envs", help="Number of parallel envs."),
    timeout_seconds: int = typer.Option(DEFAULT_TIMEOUT_SECONDS, "--timeout-seconds", help="Run timeout in seconds."),
    download_assets: bool = typer.Option(True, "--download-assets/--no-download-assets", help="Download kitchen assets before running."),
    seed: int = typer.Option(None, "--seed", help="Random seed."),
    checkpoint_uri: str = typer.Option("", "--checkpoint-uri", help="Exact trained checkpoint S3 prefix for policy evaluation."),
    train_env_ids: str = typer.Option("", "--train-env-ids", help="Comma-separated RoboCasa training env ids used for disjointness proof."),
    heldout_env_ids: str = typer.Option("", "--heldout-env-ids", help="Comma-separated held-out RoboCasa env ids to evaluate."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="RoboCasa service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    wait: bool = typer.Option(False, "--wait", help="Poll /status until the run completes."),
    poll_seconds: float = typer.Option(30.0, "--poll-seconds", help="Poll interval when --wait is set."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Run a RoboCasa capability (task registration, asset check, EGL reset, or random rollout)."""
    try:
        output_path = validate_write_path(
            output_path,
            tool="RoboCasa run",
            option="--output-path",
            required=True,
        )
        if capability == "kitchen_policy_eval":
            checkpoint_uri = validate_read_path(
                checkpoint_uri,
                tool="RoboCasa run",
                option="--checkpoint-uri",
                allow_hf=False,
            )
    except PathContractError as exc:
        fail(str(exc))
    request = RoboCasaRunRequest(
        env_id=env_id,
        capability=capability,
        output_uri=output_path,
        iterations=iterations,
        num_envs=num_envs,
        timeout_seconds=timeout_seconds,
        download_assets=download_assets,
        seed=seed,
        checkpoint_uri=checkpoint_uri,
        train_env_ids=train_env_ids,
        heldout_env_ids=heldout_env_ids,
    )
    if service:
        result = request_json(
            "POST",
            resolve_endpoint(endpoint),
            "/run",
            payload=request.model_dump(mode="json"),
            token_env=token_env,
            timeout=30.0,
        )
    else:
        from npa.sdk.workbench.robocasa import run

        local_payload = request.model_dump(mode="json")
        local_payload.pop("output_uri")
        result = run(
            output_path=output_path, **local_payload
        ).model_dump(mode="json")
    if wait:
        run_id = str(result.get("run_id") or "")
        result = _wait_for_run(
            run_id,
            endpoint=endpoint,
            token_env=token_env,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
    emit(result, output=output, text=f"run_id: {result.get('run_id')}\nstatus: {result.get('status')}")


def _wait_for_run(
    run_id: str,
    *,
    endpoint: str,
    token_env: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not run_id:
        fail("service did not return a run_id to wait for")
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        status_payload = request_json(
            "GET",
            resolve_endpoint(endpoint),
            "/status",
            params={"run_id": run_id},
            token_env=token_env,
            timeout=30.0,
        )
        status = str(status_payload.get("status") or "").strip().lower()
        if status == RUN_DONE:
            return status_payload
        if status == RUN_FAILED:
            typer.echo(str(status_payload), err=True)
            fail(f"robocasa run {run_id} failed")
        if time.monotonic() >= deadline:
            typer.echo(str(status_payload), err=True)
            fail(f"robocasa run {run_id} did not complete within {timeout_seconds:g}s")
        time.sleep(max(poll_seconds, 0.0))
