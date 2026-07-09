"""npa workbench workflow — orchestrate multi-stage training workflows."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from npa.cli.workbench.trigger import app as trigger_app

app = typer.Typer(
    name="workflow",
    help="Multi-stage training workflow orchestration.",
    no_args_is_help=True,
)

console = Console(stderr=True)
_PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class ActionSpace(str, Enum):
    cartesian = "cartesian"
    joint = "joint"


class ControllerBackendOption(str, Enum):
    kubernetes = "kubernetes"
    nebius = "nebius"


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {msg}")
    raise typer.Exit(code)


@app.command("submit")
def submit_cmd(
    yaml_path: Path = typer.Argument(
        help="Workflow YAML path (SkyPilot or npa.workflow/v0.0.1)."
    ),
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="SkyPilot managed job name. Defaults to the YAML filename stem.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.",
    ),
    isolated_config_dir: Path | None = typer.Option(
        None,
        "--isolated-config-dir",
        help="Directory for isolated SkyPilot state.",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config-path",
        help="SkyPilot global config path.",
    ),
    controller_backend: ControllerBackendOption = typer.Option(
        ControllerBackendOption.kubernetes,
        "--controller-backend",
        help="Managed-jobs controller backend.",
    ),
    infra: str = typer.Option(
        "",
        "--infra",
        help="SkyPilot infrastructure target, for example k8s/<context>.",
    ),
    submit_timeout: int = typer.Option(
        1800,
        "--submit-timeout",
        help="Submission timeout in seconds.",
    ),
    var: list[str] = typer.Option(
        [],
        "--var",
        help=(
            "Variable substitution as KEY=VALUE. For SkyPilot YAML this replaces "
            "${KEY}; for npa.workflow specs this merges into config."
        ),
    ),
    assume_decision: str = typer.Option(
        "",
        "--assume-decision",
        help=(
            "For npa.workflow specs with dynamic transitions: "
            "promote_checkpoint or loop_back."
        ),
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only/--no-plan-only",
        help=(
            "For npa.workflow specs: render the SkyPilot YAML and print it, "
            "but do not submit."
        ),
    ),
    tool: str = typer.Option(
        "",
        "--tool",
        help="Workflow-specific materializer. Currently supported: sonic.",
    ),
    registry: str = typer.Option(
        "",
        "--registry",
        help="Container registry used by workflow materializers / npa.workflow renderer.",
    ),
    image: str = typer.Option(
        "",
        "--image",
        help="First-party tool image override used by workflow materializers / npa.workflow renderer.",
    ),
    npa_image: str = typer.Option(
        "",
        "--npa-image",
        help="Generic NPA helper image override for multi-tool workflows.",
    ),
    registry_auth: bool = typer.Option(
        True,
        "--registry-auth/--no-registry-auth",
        help=(
            "For VM SONIC image pulls, materialize Docker registry auth envs. "
            "Nebius Container Registry defaults to a fresh IAM token."
        ),
    ),
    registry_username: str = typer.Option(
        "",
        "--registry-username",
        help="BYO Docker registry username for SONIC VM image pulls.",
    ),
    registry_password: str = typer.Option(
        "",
        "--registry-password",
        help="BYO Docker registry password/token for SONIC VM image pulls.",
    ),
    registry_server: str = typer.Option(
        "",
        "--registry-server",
        help="BYO Docker registry server for SONIC VM image pulls.",
    ),
    gpu_target: str = typer.Option(
        "",
        "--gpu-target",
        "--gpu-type",
        help="GPU target used by workflow materializers / npa.workflow renderer.",
    ),
    image_variant: str = typer.Option(
        "",
        "--image-variant",
        help="Manifest image variant used by workflow materializers.",
    ),
    accelerators: str = typer.Option(
        "",
        "--accelerators",
        help="SkyPilot accelerator string for materialized GPU tasks.",
    ),
    cloud: str = typer.Option(
        "",
        "--cloud",
        help="Cloud value for materialized GPU tasks.",
    ),
    region: str = typer.Option(
        "",
        "--region",
        help="Nebius region for materialized SONIC VM GPU tasks. Defaults to eu-north1; me-west1 is rejected.",
    ),
    use_spot: bool | None = typer.Option(
        None,
        "--use-spot/--no-use-spot",
        help="Optional SkyPilot spot/preemptible setting for materialized SONIC VM GPU tasks.",
    ),
    aws_profile: str = typer.Option(
        "",
        "--aws-profile",
        help="AWS profile name materialized for S3-compatible storage access.",
    ),
    require_controller_up: bool = typer.Option(
        False,
        "--require-controller-up/--skip-controller-health-guard",
        help="Before submit, require an existing SkyPilot jobs-controller with status UP.",
    ),
    s3_endpoint: str = typer.Option(
        "",
        "--s3-endpoint",
        help="S3-compatible endpoint materialized into workflow envs.",
    ),
    s3_bucket: str = typer.Option(
        "",
        "--s3-bucket",
        help="S3 bucket name materialized into workflow envs.",
    ),
    s3_prefix: str = typer.Option(
        "",
        "--s3-prefix",
        help="S3 object prefix materialized into workflow envs.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias used to resolve durable workflow S3 credentials.",
    ),
    durable_s3: bool = typer.Option(
        False,
        "--durable-s3/--no-durable-s3",
        help="Instrument the workflow with S3 manifest, status, artifacts, and redacted logs.",
    ),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact durable workflow run prefix, for example s3://bucket/run-id/.",
    ),
    workflow_s3_prefix: str = typer.Option(
        "",
        "--workflow-s3-prefix",
        help="Parent prefix for durable workflow state. The run ID is appended.",
    ),
    secret_env: list[str] = typer.Option(
        [],
        "--secret-env",
        help="Environment variable name to pass to SkyPilot as a secret.",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="Output format.",
    ),
) -> None:
    """Submit a SkyPilot or npa.workflow/v0.0.1 YAML through the NPA controller."""
    from npa.orchestration.npa_workflow.detect import is_npa_workflow_spec
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
    from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit
    from npa.orchestration.skypilot.workflow import SkyPilotSubmitError, submit_workflow
    from npa.orchestration.skypilot.workflow_state import (
        SECRET_ENV_NAMES,
        WorkflowStateError,
        instrument_workflow_yaml,
        resolve_workflow_s3_config,
        write_manifest,
    )

    if submit_timeout <= 0:
        _fail(f"--submit-timeout must be positive, got {submit_timeout}")

    substitutions = _parse_submit_vars(var)
    materializer = _resolve_materializer(tool, yaml_path)
    resolved_run_id = run_id or _default_submit_run_id(yaml_path)

    from npa.workflows.sim2real.k8s_submit import (
        is_sim2real_runbook,
        status_monitor_command,
        submit_sim2real_from_workflow_vars,
    )

    if is_sim2real_runbook(yaml_path):
        try:
            result = submit_sim2real_from_workflow_vars(
                run_id=resolved_run_id,
                substitutions=substitutions,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix or "sim2real-b",
                s3_endpoint=s3_endpoint,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            _fail(str(exc))
            return
        payload = {
            "status": result.status,
            "run_id": result.run_id,
            "job_id": result.job_name,
            "k8s_context": result.k8s_context,
            "run_prefix_uri": result.run_prefix_uri,
            "log_path": result.log_path,
            "manifest_path": result.manifest_path,
        }
        if output_format == OutputFormat.json:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"status: {result.status}")
            typer.echo(f"run_id: {result.run_id}")
            typer.echo(f"job_id: {result.job_name}")
            typer.echo(f"k8s_context: {result.k8s_context}")
            typer.echo(f"run_prefix_uri: {result.run_prefix_uri}")
            typer.echo(f"monitor: {status_monitor_command(result.run_id)}")
        return

    prepared_npa = None
    if is_npa_workflow_spec(yaml_path):
        image_overrides: dict[str, str] = {}
        if image.strip():
            image_overrides["*"] = image.strip()
        try:
            prepared_npa = prepare_npa_workflow_for_submit(
                yaml_path,
                run_id=resolved_run_id,
                assume_decision=assume_decision,
                config_overrides=substitutions,
                render_options=SkypilotRenderOptions(
                    registry=registry,
                    image_overrides=image_overrides,
                    aws_endpoint_url=s3_endpoint
                    or os.environ.get("AWS_ENDPOINT_URL")
                    or os.environ.get("NEBIUS_S3_ENDPOINT")
                    or "https://storage.eu-north1.nebius.cloud",
                    gpu_target=gpu_target,
                    image_variant=image_variant,
                ),
            )
        except NpaWorkflowError as exc:
            _fail(str(exc))
            return

        if plan_only:
            rendered = prepared_npa.skypilot_yaml_path.read_text(encoding="utf-8")
            payload = {
                "status": "PLANNED",
                "run_id": resolved_run_id,
                "workflow": prepared_npa.spec.name,
                "steps": len(prepared_npa.plan.steps),
                "secret_env_hints": list(prepared_npa.secret_env_hints),
                "skypilot_yaml": rendered,
            }
            if output_format == OutputFormat.json:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo("status: PLANNED")
                typer.echo(f"run_id: {resolved_run_id}")
                typer.echo(f"workflow: {prepared_npa.spec.name}")
                typer.echo(f"steps: {len(prepared_npa.plan.steps)}")
                if prepared_npa.secret_env_hints:
                    typer.echo(
                        "secret_env_hints: "
                        + ",".join(prepared_npa.secret_env_hints)
                    )
                typer.echo("---")
                typer.echo(rendered)
            prepared_npa.temp_dir.cleanup()
            return

        # Skip SkyPilot-path materializers; npa.workflow already planned.
        materializer = ""
        substitutions = {}
        yaml_path = prepared_npa.skypilot_yaml_path
        if prepared_npa.secret_env_hints:
            missing = [
                name
                for name in prepared_npa.secret_env_hints
                if name not in secret_env
            ]
            if missing:
                typer.echo(
                    "Hint: consider --secret-env "
                    + " --secret-env ".join(missing),
                    err=True,
                )

    submitted_yaml_path = yaml_path
    submitted_yaml_context: tempfile.TemporaryDirectory[str] | None = None
    workflow_state = None
    instrumented = None
    extra_env: dict[str, str] = {}
    if substitutions or materializer or durable_s3:
        submitted_yaml_context = tempfile.TemporaryDirectory(prefix="npa-workflow-")
        submitted_yaml_path = Path(submitted_yaml_context.name) / yaml_path.name

    try:
        source_yaml_path = yaml_path
        if substitutions:
            substituted = _substitute_workflow_vars(yaml_path, substitutions)
            source_yaml_path = Path(submitted_yaml_context.name) / f"substituted-{yaml_path.name}"
            source_yaml_path.write_text(substituted, encoding="utf-8")

        if materializer == "sonic":
            from npa.workbench.sonic.workflow import (
                materialize_sonic_workflow,
                unresolved_submit_placeholders,
            )

            try:
                plan = materialize_sonic_workflow(
                    source_yaml_path,
                    run_id=resolved_run_id,
                    registry=registry,
                    image=image,
                    npa_image=npa_image,
                    registry_auth=registry_auth,
                    registry_username=registry_username,
                    registry_password=registry_password,
                    registry_server=registry_server,
                    gpu_target=gpu_target,
                    image_variant=image_variant,
                    s3_endpoint=s3_endpoint,
                    s3_bucket=s3_bucket,
                    s3_prefix=s3_prefix,
                    accelerators=accelerators,
                    cloud=cloud,
                    region=region,
                    use_spot=use_spot,
                    aws_profile=aws_profile,
                    env_overrides=substitutions,
                )
            except ValueError as exc:
                _fail(str(exc))
                return
            unresolved = unresolved_submit_placeholders(plan.yaml_text)
            if unresolved:
                _fail(
                    "SONIC workflow still has unresolved submit placeholders: "
                    + ", ".join(unresolved)
                )
                return
            submitted_yaml_path.write_text(plan.yaml_text, encoding="utf-8")
        elif substitutions:
            substituted = source_yaml_path.read_text(encoding="utf-8")
            _warn_unresolved_placeholders(substituted)
            submitted_yaml_path.write_text(substituted, encoding="utf-8")
        else:
            _warn_unresolved_placeholders(yaml_path.read_text(encoding="utf-8"))

        if durable_s3:
            try:
                workflow_state = resolve_workflow_s3_config(
                    run_id=resolved_run_id,
                    project=project or None,
                    workflow_s3_uri=workflow_s3_uri,
                    workflow_s3_prefix=workflow_s3_prefix,
                    s3_bucket=s3_bucket,
                    s3_endpoint=s3_endpoint,
                )
                instrumented = instrument_workflow_yaml(
                    submitted_yaml_path if submitted_yaml_path.exists() else source_yaml_path,
                    run_id=resolved_run_id,
                    state=workflow_state,
                )
                submitted_yaml_path.write_text(instrumented.yaml_text, encoding="utf-8")
                write_manifest(instrumented.manifest, workflow_state)
                extra_env.update(workflow_state.secret_env())
                for name in SECRET_ENV_NAMES:
                    if name not in secret_env:
                        secret_env.append(name)
            except WorkflowStateError as exc:
                _fail(str(exc))
                return

        result = submit_workflow(
            submitted_yaml_path,
            resolved_run_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin or None,
            controller_backend=controller_backend.value,
            infra=infra,
            secret_envs=secret_env,
            require_controller_up=require_controller_up,
            extra_env=extra_env,
            timeout=submit_timeout,
        )
        if workflow_state is not None and instrumented is not None:
            instrumented_manifest = write_manifest(
                instrumented.manifest,
                workflow_state,
                job_id=result.job_id,
            )
            result.log_paths["run_prefix_uri"] = workflow_state.uri
            result.log_paths["manifest_uri"] = f"{workflow_state.uri.rstrip('/')}/manifest.json"
            result.log_paths["stages"] = ",".join(instrumented_manifest.get("stages", {}).keys())
    except OSError as exc:
        _fail(f"SkyPilot workflow submission failed: {exc}")
        return
    except SkyPilotSubmitError as exc:
        _fail(str(exc))
        return
    finally:
        if submitted_yaml_context is not None:
            submitted_yaml_context.cleanup()
        if prepared_npa is not None:
            prepared_npa.temp_dir.cleanup()

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return

    typer.echo(f"status: {result.status}")
    if result.job_id:
        typer.echo(f"job_id: {result.job_id}")
    if workflow_state is not None:
        typer.echo(f"run_prefix_uri: {workflow_state.uri}")


def _default_submit_run_id(yaml_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(yaml_path).stem).strip("-")
    return stem or "workflow"


def _parse_submit_vars(var: list[str]) -> dict[str, str]:
    substitutions: dict[str, str] = {}
    for item in var:
        if "=" not in item:
            _fail("Invalid --var format. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            _fail("Invalid --var format. Use KEY=VALUE.")
        substitutions[key] = value
    return substitutions


def _resolve_materializer(tool: str, yaml_path: Path) -> str:
    requested = tool.strip().lower()
    if requested in {"", "auto"}:
        return "sonic" if "sonic" in yaml_path.name.lower() else ""
    if requested != "sonic":
        _fail(f"Unsupported workflow materializer: {tool}")
    return requested


def _substitute_workflow_vars(yaml_path: Path, substitutions: dict[str, str]) -> str:
    content = yaml_path.read_text(encoding="utf-8")
    for key, value in substitutions.items():
        content = content.replace(f"${{{key}}}", value)
    return content


def _warn_unresolved_placeholders(content: str) -> None:
    unresolved = sorted({f"${{{match}}}" for match in _PLACEHOLDER_RE.findall(content)})
    if unresolved:
        typer.echo(
            f"Warning: unresolved placeholders remain: {', '.join(unresolved)}",
            err=True,
        )


def _uses_s3_monitor(
    run_id: str,
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
) -> bool:
    return bool(
        run_id.startswith("s3://")
        or project
        or workflow_s3_uri
        or workflow_s3_prefix
        or s3_bucket
    )


def _resolve_monitor_state(
    run_id: str,
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
):
    from npa.orchestration.skypilot.workflow_state import resolve_workflow_s3_config

    exact_uri = workflow_s3_uri or (run_id if run_id.startswith("s3://") else "")
    resolved_run_id = _display_run_id(run_id)
    return resolve_workflow_s3_config(
        run_id=resolved_run_id,
        project=project or None,
        workflow_s3_uri=exact_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )


def _resolve_monitor_parent_state(
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
):
    from npa.orchestration.skypilot.workflow_state import WorkflowS3Config, parse_s3_uri, resolve_workflow_s3_config

    if workflow_s3_uri:
        bucket, prefix = parse_s3_uri(workflow_s3_uri)
        child = resolve_workflow_s3_config(
            run_id=prefix.rsplit("/", 1)[-1] or "runs",
            project=project or None,
            workflow_s3_uri=workflow_s3_uri,
            s3_endpoint=s3_endpoint,
        )
        return WorkflowS3Config(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=child.endpoint_url,
            aws_access_key_id=child.aws_access_key_id,
            aws_secret_access_key=child.aws_secret_access_key,
            project=child.project,
        )

    sentinel = "__npa_workflow_parent__"
    child = resolve_workflow_s3_config(
        run_id=sentinel,
        project=project or None,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    prefix = child.prefix.removesuffix("/" + sentinel)
    if prefix == child.prefix and child.prefix == sentinel:
        prefix = ""
    return WorkflowS3Config(
        bucket=child.bucket,
        prefix=prefix,
        endpoint_url=child.endpoint_url,
        aws_access_key_id=child.aws_access_key_id,
        aws_secret_access_key=child.aws_secret_access_key,
        project=child.project,
    )


def _display_run_id(run_id: str) -> str:
    if not run_id.startswith("s3://"):
        return run_id
    from npa.orchestration.skypilot.workflow_state import parse_s3_uri

    _bucket, prefix = parse_s3_uri(run_id)
    return prefix.rstrip("/").rsplit("/", 1)[-1] or run_id


def _resolve_sky_bin(sky_bin: str = "") -> str:
    resolved = sky_bin or shutil.which("sky") or ""
    if resolved:
        return resolved
    env_value = str(Path.home() / ".npa" / "skypilot-venv" / "bin" / "sky")
    return env_value


def _durable_workflow_status(
    run_id: str,
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
    sky_bin: str = "",
) -> dict[str, object]:
    from npa.orchestration.skypilot.workflow import workflow_status
    from npa.orchestration.skypilot.workflow_state import read_manifest, read_stage_status

    state = _resolve_monitor_state(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    manifest = read_manifest(state)
    stages: dict[str, dict[str, object]] = {}
    for stage, info in (manifest.get("stages", {}) or {}).items():
        stage_info = dict(info) if isinstance(info, dict) else {"name": str(stage)}
        status = read_stage_status(state, str(stage))
        if status:
            stage_info.update(status)
        stages[str(stage)] = stage_info

    job_id = str(manifest.get("sky_job_id") or "")
    live_status = ""
    if job_id:
        try:
            live = workflow_status(job_id, sky_bin=_resolve_sky_bin(sky_bin))
            live_status = live.status if not live.error else ""
        except Exception:
            live_status = ""
    status = _aggregate_stage_status(stages, live_status)
    return {
        "run_id": manifest.get("run_id") or _display_run_id(run_id),
        "workflow_name": manifest.get("workflow_name", ""),
        "status": status,
        "live_status": live_status,
        "sky_job_id": job_id,
        "run_prefix_uri": manifest.get("run_prefix_uri") or state.uri,
        "manifest_uri": f"{state.uri.rstrip('/')}/manifest.json",
        "stages": stages,
    }


def _aggregate_stage_status(stages: dict[str, dict[str, object]], live_status: str) -> str:
    stage_states = [str(info.get("state") or "").upper() for info in stages.values()]
    if any(state.startswith("FAILED") or state in {"CANCELLED", "BLOCKED"} for state in stage_states):
        return "FAILED"
    if stage_states and all(state == "SUCCEEDED" for state in stage_states):
        return "SUCCEEDED"
    live = live_status.upper()
    if live:
        return live
    if any(state == "RUNNING" for state in stage_states):
        return "RUNNING"
    return "UNKNOWN"


def _workflow_status_is_terminal(status: str) -> bool:
    normalized = status.upper()
    return normalized == "SUCCEEDED" or normalized.startswith("FAILED") or normalized in {"CANCELLED", "BLOCKED"}


def _emit_workflow_status(result: dict[str, object], output_format: OutputFormat) -> None:
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    typer.echo(f"run_id: {result.get('run_id')}")
    typer.echo(f"status: {result.get('status')}")
    if result.get("current_stage"):
        typer.echo(f"current_stage: {result.get('current_stage')}")
    if result.get("k8s_job"):
        typer.echo(f"k8s_job: {result.get('k8s_job')}")
    if result.get("pod_reason"):
        typer.echo(f"pod_reason: {result.get('pod_reason')}")
    if result.get("sky_job_id"):
        typer.echo(f"sky_job_id: {result.get('sky_job_id')}")
    typer.echo(f"run_prefix_uri: {result.get('run_prefix_uri')}")
    eval_metrics = result.get("eval_metrics")
    if isinstance(eval_metrics, dict) and eval_metrics:
        if eval_metrics.get("success_rate") is not None:
            typer.echo(f"success_rate: {eval_metrics.get('success_rate')}")
        if eval_metrics.get("threshold") is not None:
            typer.echo(f"threshold: {eval_metrics.get('threshold')}")
        if eval_metrics.get("decision"):
            typer.echo(f"decision: {eval_metrics.get('decision')}")
    stages = result.get("stages", {})
    if isinstance(stages, dict):
        for stage, info in stages.items():
            state = info.get("state", "UNKNOWN") if isinstance(info, dict) else "UNKNOWN"
            tier = info.get("tier", "") if isinstance(info, dict) else ""
            suffix = f" ({tier})" if tier else ""
            typer.echo(f"{stage}: {state}{suffix}")
    siblings = result.get("sibling_jobs")
    if isinstance(siblings, list) and siblings:
        typer.echo("sibling_jobs:")
        for row in siblings:
            if isinstance(row, dict):
                typer.echo(
                    f"  {row.get('name')}: "
                    f"active={row.get('active', 0)} "
                    f"succeeded={row.get('succeeded', 0)} "
                    f"failed={row.get('failed', 0)}"
                )


def _resolve_stage_name(manifest: dict[str, object], requested: str) -> str:
    stages = manifest.get("stages", {})
    if not isinstance(stages, dict) or not stages:
        if requested:
            return requested
        raise ValueError("manifest contains no stages")
    if requested:
        if requested not in stages:
            raise ValueError(f"stage not found in manifest: {requested}")
        return requested
    if len(stages) == 1:
        return next(iter(stages.keys()))
    raise ValueError("--stage is required when a workflow has multiple stages")


@app.command("run")
def run_cmd(
    workflow: str = typer.Argument(help="Workflow name (e.g., 'distill')."),
    project: str = typer.Option(
        "", "--project", "-p", help="Project alias from ~/.npa/config.yaml."
    ),
    robot: str = typer.Option("franka_panda", "--robot", help="Robot type."),
    task: str = typer.Option("pick_place", "--task", help="Task name."),
    n_envs: int = typer.Option(4096, "--n-envs", help="Parallel environments for simulation."),
    remote: bool = typer.Option(
        False, "--remote/--local",
        help="Execute on remote VMs via SSH (requires --s3-bucket).",
    ),
    s3_bucket: str = typer.Option(
        "", "--s3-bucket", help="S3 bucket URI for artifact storage (required for --remote)."
    ),
    sim_workbench: str = typer.Option(
        "", "--sim-workbench", help="Workbench name for sim VM (Genesis stages)."
    ),
    train_workbench: str = typer.Option(
        "", "--train-workbench", help="Workbench name for training VM (LeRobot stages). Defaults to sim workbench."
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space for Genesis env: 'cartesian' (4D: delta xyz + gripper) "
             "or 'joint' (8D: delta joint positions + gripper).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Run a named workflow end-to-end."""
    if workflow != "distill":
        _fail(f"Unknown workflow: '{workflow}'. Available: distill")

    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")

    if remote and not s3_bucket:
        _fail("--remote requires --s3-bucket for artifact handoff between VMs.")

    mode = "remote" if remote else "local"
    console.print(f"[bold]Running workflow: {workflow} ({mode})[/bold]")
    console.print(f"  project={project or '(default)'}  robot={robot}  task={task}")
    console.print(f"  n_envs={n_envs}")
    if remote:
        console.print(f"  sim_workbench={sim_workbench or '(default)'}  train_workbench={train_workbench or '(same as sim)'}")

    from npa.workflows.distill import DistillationError, run_distillation

    try:
        result = run_distillation(
            project=project or None,
            robot=robot,
            task=task,
            n_envs=n_envs,
            remote=remote,
            s3_bucket=s3_bucket,
            sim_workbench=sim_workbench,
            train_workbench=train_workbench,
            action_space=action_space.value,
        )
    except DistillationError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print("[green]Workflow complete.[/green]")
        console.print(f"  run_id: {result.get('run_id')}")
        for stage, info in result.get("stages", {}).items():
            status = info.get("status", "unknown")
            tag = "[green]OK[/green]" if status == "success" else "[red]FAILED[/red]"
            console.print(f"  {stage}: {tag}")


@app.command("status")
def status_cmd(
    run_id: str = typer.Argument(help="Run ID to check status of."),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias used to resolve durable workflow S3 credentials.",
    ),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact durable workflow run prefix, for example s3://bucket/run-id/.",
    ),
    workflow_s3_prefix: str = typer.Option(
        "",
        "--workflow-s3-prefix",
        help="Parent prefix for durable workflow state. The run ID is appended.",
    ),
    s3_bucket: str = typer.Option(
        "",
        "--s3-bucket",
        help="S3 bucket name or URI for durable workflow state.",
    ),
    s3_endpoint: str = typer.Option(
        "",
        "--s3-endpoint",
        help="S3-compatible endpoint for durable workflow state.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path for live status.",
    ),
    watch: bool = typer.Option(
        False,
        "--watch/--no-watch",
        help="Refresh status until the workflow reaches a terminal state.",
    ),
    interval: float = typer.Option(
        10.0,
        "--interval",
        help="Watch refresh interval in seconds.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Shortcut for --output-format json."),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Check the status of a workflow run."""
    resolved_run_id = _display_run_id(run_id)

    from npa.workflows.sim2real.monitor import (
        emit_sim2real_status,
        get_sim2real_workflow_status,
        sim2real_run_exists,
        status_is_terminal,
    )

    prefix = workflow_s3_prefix or "sim2real-b"
    if sim2real_run_exists(
        resolved_run_id,
        s3_bucket=s3_bucket,
        s3_prefix=prefix,
        s3_endpoint=s3_endpoint,
    ):
        try:
            while True:
                result = get_sim2real_workflow_status(
                    resolved_run_id,
                    s3_bucket=s3_bucket,
                    s3_prefix=prefix,
                    s3_endpoint=s3_endpoint,
                )
                if json_output or output_format == OutputFormat.json:
                    emit_sim2real_status(result, json_output=True)
                else:
                    _emit_workflow_status(result, output_format)
                if not watch or status_is_terminal(str(result.get("status", ""))):
                    return
                time.sleep(interval)
        except Exception as exc:
            _fail(str(exc))
            return

    if _uses_s3_monitor(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
    ):
        try:
            while True:
                result = _durable_workflow_status(
                    run_id,
                    project=project,
                    workflow_s3_uri=workflow_s3_uri,
                    workflow_s3_prefix=workflow_s3_prefix,
                    s3_bucket=s3_bucket,
                    s3_endpoint=s3_endpoint,
                    sky_bin=sky_bin,
                )
                _emit_workflow_status(result, OutputFormat.json if json_output else output_format)
                if not watch or _workflow_status_is_terminal(str(result.get("status", ""))):
                    return
                time.sleep(interval)
        except Exception as exc:
            _fail(str(exc))
            return

    from npa.workflows.distill import DistillationError, get_run_status

    try:
        result = get_run_status(run_id)
    except DistillationError as exc:
        _fail(str(exc))
        return

    if json_output or output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"  run_id: {result.get('run_id')}")
        console.print(f"  status: {result.get('status')}")
        for stage, info in result.get("stages", {}).items():
            console.print(f"  {stage}: {info.get('status', 'unknown')}")


@app.command("logs")
def logs_cmd(
    run_id: str = typer.Argument(help="Run ID."),
    stage: str | None = typer.Argument(
        None,
        help="Stage name. Legacy distill runs require this positional argument.",
    ),
    stage_option: str = typer.Option(
        "",
        "--stage",
        help="Stage name for durable S3 workflow logs.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias used to resolve durable workflow S3 credentials.",
    ),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact durable workflow run prefix, for example s3://bucket/run-id/.",
    ),
    workflow_s3_prefix: str = typer.Option(
        "",
        "--workflow-s3-prefix",
        help="Parent prefix for durable workflow state. The run ID is appended.",
    ),
    s3_bucket: str = typer.Option(
        "",
        "--s3-bucket",
        help="S3 bucket name or URI for durable workflow state.",
    ),
    s3_endpoint: str = typer.Option(
        "",
        "--s3-endpoint",
        help="S3-compatible endpoint for durable workflow state.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path for live --follow logs.",
    ),
    follow: bool = typer.Option(
        False,
        "--follow/--no-follow",
        help="Tail live SkyPilot logs when the managed job is still running.",
    ),
) -> None:
    """Show logs for a specific stage of a workflow run."""
    selected_stage = stage_option or stage or ""
    if _uses_s3_monitor(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
    ):
        try:
            state = _resolve_monitor_state(
                run_id,
                project=project,
                workflow_s3_uri=workflow_s3_uri,
                workflow_s3_prefix=workflow_s3_prefix,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
            )
            from npa.orchestration.skypilot.workflow_state import (
                read_manifest,
                read_stage_log,
                tail_live_job_logs,
            )

            manifest = read_manifest(state)
            selected_stage = _resolve_stage_name(manifest, selected_stage)
            job_id = str(manifest.get("sky_job_id") or "")
            if follow and job_id:
                live = tail_live_job_logs(
                    sky_bin=_resolve_sky_bin(sky_bin),
                    job_id=job_id,
                    stage=selected_stage,
                    follow=True,
                    timeout=86400,
                )
                if live.stdout:
                    typer.echo(live.stdout, nl=False)
                if live.stderr:
                    typer.echo(live.stderr, err=True, nl=False)
                if live.returncode == 0:
                    return
            typer.echo(read_stage_log(state, selected_stage), nl=False)
            return
        except Exception as exc:
            _fail(str(exc))
            return

    if not selected_stage:
        _fail("stage is required for legacy distill logs")
        return

    from npa.workflows.distill import DistillationError, get_stage_logs

    try:
        logs = get_stage_logs(run_id, selected_stage)
    except DistillationError as exc:
        _fail(str(exc))
        return

    typer.echo(logs)


@app.command("artifacts")
def artifacts_cmd(
    run_id: str = typer.Argument(help="Durable workflow run ID or s3:// run prefix."),
    stage: str = typer.Option("", "--stage", help="Optional stage name."),
    project: str = typer.Option("", "--project", "-p", help="Project alias for S3 credentials."),
    workflow_s3_uri: str = typer.Option("", "--workflow-s3-uri", help="Exact durable workflow run prefix."),
    workflow_s3_prefix: str = typer.Option("", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List durable S3 artifact URIs for a workflow run."""
    try:
        from npa.orchestration.skypilot.workflow_state import list_artifacts

        state = _resolve_monitor_state(
            run_id,
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            workflow_s3_prefix=workflow_s3_prefix,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
        )
        artifacts = list_artifacts(state, stage or None)
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps({"run_id": _display_run_id(run_id), "artifacts": artifacts}, indent=2))
        return
    for uri in artifacts:
        typer.echo(uri)


@app.command("list")
def list_cmd(
    project: str = typer.Option("", "--project", "-p", help="Project alias for S3 credentials."),
    workflow_s3_uri: str = typer.Option("", "--workflow-s3-uri", help="Parent durable workflow prefix."),
    workflow_s3_prefix: str = typer.Option("", "--workflow-s3-prefix", help="Parent prefix for durable workflow state."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    limit: int = typer.Option(50, "--limit", help="Maximum runs to list."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List durable S3 workflow runs."""
    try:
        from npa.orchestration.skypilot.workflow_state import list_runs

        parent_state = _resolve_monitor_parent_state(
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            workflow_s3_prefix=workflow_s3_prefix,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
        )
        runs = list_runs(state_parent=parent_state, limit=limit)
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps({"runs": runs}, indent=2))
        return
    for item in runs:
        typer.echo(
            f"{item.get('run_id', '')}\t{item.get('workflow_name', '')}\t"
            f"{item.get('sky_job_id', '')}\t{item.get('run_prefix_uri', '')}"
        )


@app.command("cancel")
def cancel_cmd(
    run_id: str = typer.Argument(help="Durable workflow run ID or s3:// run prefix."),
    project: str = typer.Option("", "--project", "-p", help="Project alias for S3 credentials."),
    workflow_s3_uri: str = typer.Option("", "--workflow-s3-uri", help="Exact durable workflow run prefix."),
    workflow_s3_prefix: str = typer.Option("", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    sky_bin: str = typer.Option("", "--sky-bin", help="SkyPilot executable path."),
    cluster: str = typer.Option("", "--cluster", help="SkyPilot cluster name to tear down. Defaults to run ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Cancel a managed workflow job and explicitly tear down its cluster."""
    try:
        from npa.orchestration.skypilot.workflow_state import cancel_workflow_job, read_manifest

        state = _resolve_monitor_state(
            run_id,
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            workflow_s3_prefix=workflow_s3_prefix,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
        )
        manifest = read_manifest(state)
        job_id = str(manifest.get("sky_job_id") or "")
        if not job_id:
            _fail("manifest does not contain a sky_job_id")
            return
        result = cancel_workflow_job(
            sky_bin=_resolve_sky_bin(sky_bin),
            job_id=job_id,
            run_id=str(manifest.get("run_id") or _display_run_id(run_id)),
            cluster=cluster,
        )
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    typer.echo(f"job_id: {result['job_id']}")
    typer.echo(f"cluster: {result['cluster']}")
    typer.echo(f"cancel_returncode: {result['cancel_returncode']}")
    typer.echo(f"down_returncode: {result['down_returncode']}")


@app.command("teardown")
def teardown_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format.",
    ),
) -> None:
    """Destroy both VMs from a distill workflow run.

    Reads the sim and train VM specs from the distill module, bootstraps
    Nebius credentials, and destroys each VM via Terraform.  Also removes
    the workbench entries from ~/.npa/config.yaml.
    """
    from npa.workflows.distill_two_vm import (
        PROJECT_ALIAS, PROJECT_ID, REGION, SIM_VM, TENANT_ID,
        TRAIN_VM, TwoVMDistillError, _destroy_vm,
    )
    from npa.clients.config import (
        ConfigError, resolve_ssh_config,
        remove_workbench_config,
    )

    # Verify at least one VM is still registered.
    found_any = False
    for spec in (SIM_VM, TRAIN_VM):
        try:
            resolve_ssh_config(project=PROJECT_ALIAS, name=spec.name)
            found_any = True
        except ConfigError:
            pass

    if not found_any:
        _fail(
            f"No distill VMs found in config. Expected "
            f"'{SIM_VM.name}' and/or '{TRAIN_VM.name}' in "
            f"project '{PROJECT_ALIAS}'."
        )
        return

    # Bootstrap Nebius credentials for Terraform.
    console.print("[bold]Tearing down distill infrastructure[/bold]")
    console.print(f"  sim:   {SIM_VM.name} ({SIM_VM.gpu_platform})")
    console.print(f"  train: {TRAIN_VM.name} ({TRAIN_VM.gpu_platform})")

    from npa.clients.nebius import NebiusError, bootstrap_environment

    console.print("  Bootstrapping Nebius credentials...")
    try:
        nebius_creds = bootstrap_environment(
            PROJECT_ID, TENANT_ID, REGION,
            on_status=lambda msg: console.print(f"    {msg}"),
        )
    except NebiusError as exc:
        _fail(f"Nebius bootstrap failed: {exc}")
        return

    failed: list[str] = []
    destroyed: list[str] = []

    for spec in (SIM_VM, TRAIN_VM):
        # Skip VMs that are already gone from config.
        try:
            resolve_ssh_config(project=PROJECT_ALIAS, name=spec.name)
        except ConfigError:
            console.print(f"  {spec.name}: not in config, skipping")
            continue

        console.print(f"  Destroying {spec.name}...")
        try:
            _destroy_vm(spec, nebius_creds)
            remove_workbench_config(PROJECT_ALIAS, spec.name)
            destroyed.append(spec.name)
            console.print(f"    {spec.name}: destroyed")
        except TwoVMDistillError as exc:
            failed.append(spec.name)
            console.print(f"    [red]{spec.name}: destroy failed: {exc}[/red]")

    result = {"destroyed": destroyed, "failed": failed}

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        if destroyed:
            console.print(f"\n[green]Destroyed: {', '.join(destroyed)}[/green]")
        if failed:
            console.print(f"\n[red]Failed: {', '.join(failed)}[/red]")

    if failed:
        raise typer.Exit(1)


@app.command("distill")
def distill_cmd(
    teardown: bool = typer.Option(
        False, "--teardown/--no-teardown",
        help="Destroy both VMs after the workflow completes (even on failure).",
    ),
    skip_infra: bool = typer.Option(
        False, "--skip-infra/--provision",
        help="Skip provisioning and Nebius bootstrap; resolve VMs and S3 "
             "credentials from ~/.npa/config.yaml.",
    ),
    skip_setup: bool = typer.Option(
        False, "--skip-setup/--setup",
        help="Skip runtime setup (conda env + npa install). Use when VMs "
             "already have the correct environment.",
    ),
    n_envs: int = typer.Option(4096, "--n-envs", help="Parallel environments for simulation."),
    teacher_max_iterations: int = typer.Option(
        500, "--teacher-max-iterations",
        help="PPO training iterations for teacher.",
    ),
    student_policy: str = typer.Option(
        "act", "--student-policy",
        help="Student policy type: act, diffusion, smolvla.",
    ),
    student_epochs: int = typer.Option(
        100, "--student-epochs", help="Training epochs for student.",
    ),
    student_batch_size: int = typer.Option(
        64, "--student-batch-size", help="Batch size for student training.",
    ),
    eval_n_episodes: int = typer.Option(
        1024, "--eval-n-episodes", help="Number of eval episodes for the student.",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian, "--action-space",
        help="Action space for Genesis env: 'cartesian' (4D: delta xyz + gripper) "
             "or 'joint' (8D: delta joint positions + gripper).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format.",
    ),
) -> None:
    """Run expert distillation: L40S (Genesis) + H100 (LeRobot).

    Provisions an L40S VM for Genesis simulation (stages 1, 2, 3, 5)
    and an H100 VM for LeRobot training (stage 4), with S3 artifact
    handoff between VMs.
    """
    if n_envs <= 0:
        _fail(f"--n-envs must be positive, got {n_envs}")
    if teacher_max_iterations <= 0:
        _fail(f"--teacher-max-iterations must be positive, got {teacher_max_iterations}")
    if student_epochs <= 0:
        _fail(f"--student-epochs must be positive, got {student_epochs}")
    if student_batch_size <= 0:
        _fail(f"--student-batch-size must be positive, got {student_batch_size}")
    if eval_n_episodes <= 0:
        _fail(f"--eval-n-episodes must be positive, got {eval_n_episodes}")
    if student_policy not in ("act", "diffusion", "smolvla"):
        _fail(f"--student-policy must be act, diffusion, or smolvla, got {student_policy}")

    mode = "skip-infra" if skip_infra else "provision"
    console.print(f"[bold]Expert distillation ({mode})[/bold]")
    console.print(f"  sim:   L40S  ({mode})")
    console.print(f"  train: H100  ({mode})")
    console.print(f"  policy={student_policy}  n_envs={n_envs}  epochs={student_epochs}")

    from npa.workflows.distill_two_vm import TwoVMDistillError, distill

    try:
        result = distill(
            teardown=teardown,
            skip_infra=skip_infra,
            skip_setup=skip_setup,
            n_envs=n_envs,
            teacher_max_iterations=teacher_max_iterations,
            student_policy=student_policy,
            student_epochs=student_epochs,
            student_batch_size=student_batch_size,
            eval_n_episodes=eval_n_episodes,
            action_space=action_space.value,
        )
    except TwoVMDistillError as exc:
        _fail(str(exc))
        return

    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"\n[green]Workflow {result.get('status', 'unknown')}.[/green]")
        console.print(f"  run_id: {result.get('run_id')}")
        console.print(f"  s3:     {result.get('s3_base', '')}")
        for stage, info in result.get("stages", {}).items():
            status = info.get("status", "unknown")
            tag = "[green]OK[/green]" if status == "success" else "[red]FAILED[/red]"
            console.print(f"  {stage}: {tag}")


def _load_npa_workflow(path: Path):
    from npa.orchestration.npa_workflow import NpaWorkflowError, load_spec

    try:
        return load_spec(path)
    except NpaWorkflowError as exc:
        _fail(str(exc))


@app.command("validate-spec")
def validate_spec_cmd(
    yaml_path: Path = typer.Argument(help="NPA workflow spec (apiVersion: npa.workflow/v0.0.1)."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON result."),
) -> None:
    """Validate an NPA workflow specification file."""

    spec = _load_npa_workflow(yaml_path)
    payload = {
        "status": "valid",
        "apiVersion": spec.api_version,
        "name": spec.name,
        "states": sorted(spec.states),
        "initial": spec.initial,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"valid: {spec.name} ({spec.api_version})")
        typer.echo(f"states: {', '.join(sorted(spec.states))}")


@app.command("plan-spec")
def plan_spec_cmd(
    yaml_path: Path = typer.Argument(help="NPA workflow spec path."),
    run_id: str = typer.Option("", "--run-id", help="Run id for token expansion."),
    assume_decision: str = typer.Option(
        "",
        "--assume-decision",
        help="Plan branch after decide states (promote_checkpoint or loop_back).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON plan."),
) -> None:
    """Expand an NPA workflow spec into an execution plan (dry-run)."""

    from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan

    spec = _load_npa_workflow(yaml_path)
    resolved_run_id = run_id or f"{spec.name}-plan"
    try:
        plan = build_plan(spec, run_id=resolved_run_id, assume_decision=assume_decision)
    except NpaWorkflowError as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return
    typer.echo(f"workflow: {plan.workflow}")
    typer.echo(f"assume_decision: {plan.assume_decision}")
    for index, step in enumerate(plan.steps, start=1):
        label = step.state
        if step.iteration is not None:
            label = f"{label}#{step.iteration}"
        if step.tool_ref:
            typer.echo(f"  {index:02d}. {label} toolRef={step.tool_ref}")
        elif step.argv:
            typer.echo(f"  {index:02d}. {label} argv={' '.join(step.argv[:4])}...")
        else:
            typer.echo(f"  {index:02d}. {label} shell=<{len(step.shell)} chars>")


@app.command("run-spec")
def run_spec_cmd(
    yaml_path: Path = typer.Argument(help="NPA workflow spec path."),
    run_id: str = typer.Option("", "--run-id", help="Run identifier."),
    execute: bool = typer.Option(
        False,
        "--execute/--plan-only",
        help="Execute tool commands locally (default: plan only).",
    ),
    assume_decision: str = typer.Option("", "--assume-decision", help="Branch assumption for planning."),
    persist_state: bool = typer.Option(
        False,
        "--persist-state",
        help="Write run manifest and status to S3 (config.bucket + config.prefix).",
    ),
    require_inputs: bool = typer.Option(
        False,
        "--require-inputs",
        help="Fail before each step when declared input URIs are missing on S3.",
    ),
    scheduler_plan: bool = typer.Option(
        False,
        "--scheduler-plan",
        help="Include portable scheduler task documents in JSON output.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON report."),
) -> None:
    """Run or plan an NPA workflow spec."""

    from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan, run_workflow
    from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan

    spec = _load_npa_workflow(yaml_path)
    resolved_run_id = run_id or f"{spec.name}-{int(time.time())}"
    resolved_assume = assume_decision or str(spec.config.get("plan_assume_decision") or "")
    try:
        report = run_workflow(
            spec,
            run_id=resolved_run_id,
            execute=execute,
            assume_decision=resolved_assume,
            persist_state=persist_state,
            require_inputs=require_inputs,
        )
    except NpaWorkflowError as exc:
        _fail(str(exc))
        return
    if scheduler_plan:
        plan = build_plan(spec, run_id=resolved_run_id, assume_decision=resolved_assume)
        report["scheduler"] = build_scheduler_plan(spec, plan.steps, run_id=resolved_run_id)
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {report['status']}")
        typer.echo(f"run_id: {report['run_id']}")
        if report.get("run_prefix_uri"):
            typer.echo(f"run_prefix_uri: {report['run_prefix_uri']}")
        typer.echo(f"steps: {len(report['plan']['steps'])}")


app.add_typer(trigger_app, name="trigger")
