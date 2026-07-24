"""Typer CLI for `npa workbench scenario-gen`."""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

import httpx
import typer

from npa.workbench.scenario_gen.schemas import (
    DEFAULT_ADVERSARY_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_SCENARIOS,
    DEFAULT_SEED,
    DEFAULT_SEVERITY_WEIGHT,
    DEFAULT_TASK,
    DEFAULT_TOKEN_ENV,
    DEFAULT_TOP_K,
)

app = typer.Typer(
    name="scenario-gen",
    help=(
        "Adversarial scenario generation: mine hard scenarios that fail a "
        "policy-under-test (pluggable Isaac Lab RL backend; deterministic default)."
    ),
    no_args_is_help=True,
)

ENDPOINT_ENV = "NPA_SCENARIO_GEN_ENDPOINT"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text if text is not None else "\n".join(f"{key}: {value}" for key, value in payload.items()))


@app.command("generate")
def generate_cmd(
    policy_uri: str = typer.Option(..., "--policy-uri", help="S3 URI of the policy-under-test checkpoint."),
    input_path: str = typer.Option(..., "--input-path", help="S3 URI of the base task/scene config."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix for the adversarial scenario set."),
    task: str = typer.Option(DEFAULT_TASK, "--task", help="Simulator task name for the adversary."),
    num_scenarios: int = typer.Option(DEFAULT_NUM_SCENARIOS, "--num-scenarios", help="Number of adversarial scenarios to mine."),
    adversary_steps: int = typer.Option(DEFAULT_ADVERSARY_STEPS, "--adversary-steps", help="Adversary RL training steps."),
    learning_rate: float = typer.Option(DEFAULT_LEARNING_RATE, "--learning-rate", help="Adversary learning rate."),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", help="Adversary batch size."),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Adversary sampling seed."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into lineage."),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Emit a Rerun .rrd visualization next to the manifest."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Scenario-gen service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Mine ranked adversarial scenarios against a policy-under-test.

    Uses the pluggable adversary backend (Isaac Lab RL intended; deterministic
    heuristic default when no GPU backend is configured).
    """
    payload = {
        "policy_uri": policy_uri,
        "base_config_uri": input_path,
        "output_uri": output_path,
        "task_name": task,
        "num_scenarios": num_scenarios,
        "adversary_steps": adversary_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "workflow_run": workflow_run,
        "visualize": visualize,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/generate", payload=payload, token_env=token_env, timeout=120.0)
    else:
        from npa.sdk.workbench.scenario_gen import generate

        result = generate(**payload).model_dump(mode="json")
    emit(
        result,
        output=output,
        text=f"run_id: {result.get('run_id')}\nscenario_count: {result.get('scenario_count')}\nmanifest_uri: {result.get('manifest_uri')}",
    )


@app.command("rank")
def rank_cmd(
    input_path: str = typer.Option(..., "--input-path", "--input-uri", help="S3 URI of an adversarial scenario set manifest."),
    output_path: str = typer.Option(..., "--output-path", "--output-uri", help="S3 URI prefix for the ranked scenario set."),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", help="Number of ranked scenarios to retain."),
    severity_weight: float = typer.Option(DEFAULT_SEVERITY_WEIGHT, "--severity-weight", help="Weight on failure severity."),
    diversity_weight: float = typer.Option(DEFAULT_DIVERSITY_WEIGHT, "--diversity-weight", help="Weight on scenario diversity."),
    workflow_run: str = typer.Option("", "--workflow-run", help="Workflow run id threaded into lineage."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Scenario-gen service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.json, "--output", help="Output format."),
) -> None:
    """Score and rank generated adversarial scenarios."""
    payload = {
        "input_uri": input_path,
        "output_uri": output_path,
        "top_k": top_k,
        "severity_weight": severity_weight,
        "diversity_weight": diversity_weight,
        "workflow_run": workflow_run,
    }
    if service:
        result = request_json("POST", resolve_endpoint(endpoint), "/rank", payload=payload, token_env=token_env, timeout=60.0)
    else:
        from npa.sdk.workbench.scenario_gen import rank

        result = rank(**payload).model_dump(mode="json")
    emit(
        result,
        output=output,
        text=f"run_id: {result.get('run_id')}\nranked_count: {result.get('ranked_count')}\nranked_manifest_uri: {result.get('ranked_manifest_uri')}",
    )


@app.command("status")
def status_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Scenario-gen run ID."),
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Scenario-gen service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Fetch a scenario-gen run status."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/status", params={"run_id": run_id}, token_env=token_env, timeout=30.0)
    else:
        from npa.sdk.workbench.scenario_gen import status

        result = status(run_id=run_id).model_dump(mode="json")
    emit(result, output=output, text=f"status: {result.get('status')}\nscenario_count: {result.get('scenario_count')}")


@app.command("system-info")
def system_info_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Scenario-gen service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Show scenario-gen runtime information."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/system-info", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.scenario_gen.service import system_info_payload

        result = system_info_payload()
    emit(result, output=output)


@app.command("list")
def list_cmd(
    service: bool = typer.Option(False, "--service", help="Call a deployed service endpoint."),
    endpoint: str = typer.Option("", "--endpoint", help="Scenario-gen service endpoint."),
    token_env: str = typer.Option(DEFAULT_TOKEN_ENV, "--token-env", help="Environment variable containing service token."),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """List service-managed scenario-gen runs."""
    if service:
        result = request_json("GET", resolve_endpoint(endpoint), "/list", token_env=token_env, timeout=30.0)
    else:
        from npa.workbench.scenario_gen.service import RUNS

        result = {"runs": [run.model_dump(mode="json") for run in RUNS.values()]}
    emit(result, output=output, text="\n".join(run["run_id"] for run in result.get("runs", [])) or "No runs found.")


def resolve_endpoint(endpoint: str) -> str:
    resolved = endpoint.strip() or os.environ.get(ENDPOINT_ENV, "")
    if not resolved:
        fail("--endpoint is required")
    if not resolved.startswith(("http://", "https://")):
        fail("--endpoint must be an http:// or https:// URL")
    return resolved.rstrip("/")


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    token_env: str,
    payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{endpoint}{path}", headers=headers, json=payload, params=params, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        fail(f"Scenario-gen request failed ({exc.response.status_code}): {exc.response.text.strip()}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach scenario-gen endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail("Scenario-gen endpoint returned non-JSON response")
    if not isinstance(data, dict):
        fail("Scenario-gen endpoint returned an unexpected response")
    return data
