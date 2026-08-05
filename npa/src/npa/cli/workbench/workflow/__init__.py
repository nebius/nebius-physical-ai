"""npa workbench workflow — orchestrate multi-stage training workflows."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import re
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from npa.cli.workbench.trigger import app as trigger_app
from npa.orchestration.npa_workflow.spec import load_spec

if TYPE_CHECKING:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

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


# ``docker:cr.<region>.nebius.cloud/<registry-id>/<image>:<tag>`` in a rendered plan.
_NEBIUS_IMAGE_RE = re.compile(r"image_id:\s*docker:(cr\.[a-z0-9-]+\.nebius\.cloud)/", re.IGNORECASE)


def nebius_registry_hosts(rendered_yaml: str) -> list[str]:
    """Distinct Nebius registry hosts a rendered plan pulls images from."""

    return sorted({match.group(1).lower() for match in _NEBIUS_IMAGE_RE.finditer(rendered_yaml)})


def _refresh_kubernetes_pull_secrets(rendered_path: Path) -> None:
    """Refresh the cluster's Nebius registry pull secret before launching.

    Kubernetes pulls private images with an ``imagePullSecret``, and the Nebius
    registry only accepts short-lived IAM tokens — so a cluster whose secret was
    written days ago fails every pull with ``401 Unauthorized`` even though
    ``docker login`` on the operator box works, and SkyPilot reports it as
    ``ErrImagePull`` / resources-unavailable rather than an auth problem. Minting a
    fresh token here keeps a pinned-image submit from failing for a reason that has
    nothing to do with the workflow.

    A private-image submit is blocked if this refresh fails: preflight and the
    credentials Kubernetes actually uses must agree, rather than accepting a job
    that is guaranteed to sit in ``ImagePullBackOff``.
    """

    try:
        rendered = rendered_path.read_text(encoding="utf-8")
    except OSError:
        return
    hosts = nebius_registry_hosts(rendered)
    if not hosts:
        return

    from npa.orchestration.skypilot.registry_preflight import resolve_registry_credentials
    from npa.workflows.sim2real.registry_auth import ensure_nebius_registry_pull_secret

    joined = ", ".join(hosts)
    # One call with every host: the secret holds a single dockerconfigjson and each
    # apply replaces it, so refreshing host by host would leave only the last one.
    try:
        username, password = resolve_registry_credentials(hosts[0], mint=True)
        if not password:
            raise RuntimeError("no registry credential could be resolved")
        ensure_nebius_registry_pull_secret(
            registry_servers=hosts,
            username=username,
            token=password,
        )
    except Exception as exc:
        raise RuntimeError(
            "could not install the Kubernetes imagePullSecret for "
            f"{joined}: {exc}. Fix kubectl access/registry credentials before submit; "
            "otherwise every private-image task will fail with ImagePullBackOff."
        ) from exc
    console.print(f"Refreshed the Kubernetes pull secret for {joined}")


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
    runtime: bool = typer.Option(
        False,
        "--runtime/--no-runtime",
        help=(
            "For npa.workflow specs: drive the run with the runtime orchestrator "
            "(submit each wave, poll to terminal, read the real decision artifact "
            "from S3, then replan). Required for parallel fan-out and for real "
            "runtime early-exit; the default one-shot path renders the flattened "
            "serial plan with --assume-decision."
        ),
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help=(
            "With --runtime: replay waves already recorded as succeeded for this run id "
            "(enabled by default for restart-safe submit)."
        ),
    ),
    poll_seconds: int = typer.Option(
        30,
        "--poll-seconds",
        help="With --runtime: seconds between managed-job status polls.",
    ),
    max_wait_seconds: int = typer.Option(
        3600,
        "--max-wait-seconds",
        help="With --runtime: per-wave deadline before the job is cancelled.",
    ),
    cancel_on_timeout: bool = typer.Option(
        True,
        "--cancel-on-timeout/--no-cancel-on-timeout",
        help="With --runtime: cancel the managed job (and its cluster) on timeout.",
    ),
    retries: int = typer.Option(
        0,
        "--retries",
        help="With --runtime: retry a failed wave this many times before failing the run.",
    ),
    max_concurrency: int = typer.Option(
        0,
        "--max-concurrency",
        help=(
            "With --runtime: cap concurrent tasks per parallel group "
            "(0 keeps each group's declared maxConcurrency)."
        ),
    ),
    preflight_images: bool = typer.Option(
        True,
        "--preflight-images/--no-preflight-images",
        help=(
            "For npa.workflow specs: reproduce each step's image pull with this run's "
            "own registry credentials before submitting, so a 403 fails here instead of "
            "leaving workers in ImagePullBackOff."
        ),
    ),
    resolve_accelerators: bool = typer.Option(
        True,
        "--resolve-accelerators/--no-resolve-accelerators",
        help=(
            "For npa.workflow specs on Kubernetes: map the spec's accelerator name "
            "onto the one the target cluster actually advertises, and fail fast when "
            "the requested per-task GPU count exceeds what one node can provide."
        ),
    ),
    gpu_readiness_timeout: float = typer.Option(
        600.0,
        "--gpu-readiness-timeout",
        help="Seconds to wait for SkyPilot to discover the requested Kubernetes GPU.",
    ),
    gpu_readiness_poll_interval: float = typer.Option(
        10.0,
        "--gpu-readiness-poll-interval",
        help="Seconds between SkyPilot GPU discovery checks.",
    ),
    deploy_if_absent: bool = typer.Option(
        True,
        "--deploy-if-absent/--no-deploy-if-absent",
        help=(
            "For npa.workflow specs: honor resource `deployIfAbsent` and provision "
            "missing Kubernetes/GPU clusters via npa before submit (idempotent). "
            "With --plan-only the provisioning is dry-run only."
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
    stage_src: bool | None = typer.Option(
        None,
        "--stage-src/--no-stage-src",
        help=(
            "Content-address and persist the local npa package for image-less steps. "
            "The default is automatic when the spec needs source; --no-stage-src is "
            "an advanced opt-out."
        ),
    ),
    auto_load: bool = typer.Option(
        True,
        "--auto-load/--no-auto-load",
        help="After successful PAIDF runtime completion, load and verify the final Rerun artifact in the configured agent.",
    ),
    agent_name: str = typer.Option(
        "",
        "--agent-name",
        help="Configured agent name for final-artifact auto-load (defaults to agent or the sole agent).",
    ),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip the pre-submit prerequisite checks (SkyPilot CLI, npa source, bucket).",
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
    from npa.orchestration.skypilot.workflow import (
        SkyPilotSubmitError,
        WorkflowResult,
        find_job_ids_by_name,
        submit_workflow,
    )
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
    from npa.orchestration.npa_workflow.run_resolution import validate_run_id

    try:
        resolved_run_id = validate_run_id(resolved_run_id)
    except Exception as exc:
        _fail(str(exc))
        return
    from npa.orchestration.npa_workflow.submit_credentials import resolve_submit_credentials

    submit_credentials = resolve_submit_credentials(
        project=project,
        explicit_endpoint=s3_endpoint,
        requested=secret_env,
    )
    s3_endpoint = submit_credentials.endpoint_url
    extra_env: dict[str, str] = dict(submit_credentials.secret_values)
    if submit_credentials.missing and not plan_only:
        _fail(
            "--secret-env requested values that are not present in the process "
            "environment or supported NPA credentials for project "
            f"{project or '<default>'}: {', '.join(submit_credentials.missing)}. "
            "Set them explicitly or store them with `npa configure`."
        )
        return

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
        # One prerequisite report instead of a sequence of one-at-a-time
        # failures spread over the render, the SkyPilot resolver and the
        # controller. Everything the operator still has to do is listed once,
        # each with the command that fixes it.
        spec_config = _npa_spec_config(yaml_path, substitutions)
        # Resolve the pinned context before the preflight reports on it: npa keeps
        # its cluster kubeconfigs outside ~/.kube/config, so a context it
        # provisioned looks missing until KUBECONFIG points at it.
        infra_context = _infra_kube_context(infra)
        if infra_context and not plan_only:
            _adopt_npa_kubeconfig(infra_context)
        image_value_for_source = str(image or "").strip().lower()
        image_pins_all_tasks = bool(image_value_for_source) and image_value_for_source not in {
            "none",
            "default",
            "-",
        }
        bucket_for_source = str(s3_bucket or spec_config.get("bucket", "") or "").strip()
        existing_source_uri = _resolve_submit_src_s3_uri(project)
        if existing_source_uri:
            # The renderer has a process/config resolver without a project
            # parameter. Pin the explicitly selected project's URI for this
            # invocation so a non-default project cannot inherit another
            # project's source prefix.
            os.environ["NPA_SRC_S3_URI"] = existing_source_uri
        auto_stage_source = (
            stage_src is None
            and not plan_only
            and not image_pins_all_tasks
            and not existing_source_uri
            and not _is_placeholder_bucket(bucket_for_source)
        )
        stage_source_planned = stage_src is True or auto_stage_source
        if not skip_preflight:
            missing = _submit_prerequisites(
                spec_config,
                sky_bin=sky_bin,
                image=image,
                plan_only=plan_only,
                infra=infra,
                self_provisions=deploy_if_absent and _spec_self_provisions(yaml_path),
                requires_s3=_spec_requires_s3(yaml_path),
                s3_endpoint=submit_credentials.endpoint_url,
                s3_access_key_id=getattr(submit_credentials, "access_key_id", ""),
                s3_secret_access_key=getattr(
                    submit_credentials, "secret_access_key", ""
                ),
                source_staging_planned=stage_source_planned,
            )
            if missing:
                _fail_missing_prerequisites(yaml_path, missing)
                return
        existing_fingerprint = existing_source_uri.rstrip("/").rsplit("/", 1)[-1]
        if (
            not stage_source_planned
            and not plan_only
            and not image_pins_all_tasks
            and re.fullmatch(r"[0-9a-f]{64}", existing_fingerprint)
        ):
            from npa.orchestration.npa_workflow.src_staging import (
                _storage_client,
                verify_staged_source,
            )

            try:
                verify_staged_source(
                    existing_source_uri,
                    client=_storage_client(
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=extra_env.get("AWS_ACCESS_KEY_ID", ""),
                        aws_secret_access_key=extra_env.get("AWS_SECRET_ACCESS_KEY", ""),
                    ),
                    expected_fingerprint=existing_fingerprint,
                )
            except Exception as exc:  # noqa: BLE001
                _fail(str(exc))
                return
        if stage_source_planned:
            staged_uri = _stage_npa_src_for_submit(
                spec_config,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
                credential_values=extra_env,
                project=project,
                run_id=resolved_run_id,
            )
            if not staged_uri:
                return
            os.environ["NPA_SRC_S3_URI"] = staged_uri
        _warn_placeholder_bucket(
            spec_config, quiet=output_format == OutputFormat.json
        )
        image_overrides: dict[str, str] = {}
        # ``none`` / ``default`` clears workbench image pins so tasks use the
        # SkyPilot default image (needed when registry images fail k8s apt-ssh).
        image_value = image.strip()
        if image_value.lower() in {"none", "default", "-"}:
            image_overrides["*"] = ""
        elif image_value:
            image_overrides["*"] = image_value

        # Which images a run pulls depends only on the registry and the overrides,
        # never on the cluster -- so check them before provisioning rather than
        # after, and a registry missing the workbench images costs no GPU time.
        _preflight_submit_images(
            yaml_path,
            options=SkypilotRenderOptions(
                registry=_resolve_submit_registry(registry, project),
                image_overrides=image_overrides,
                gpu_target=gpu_target,
                image_variant=image_variant,
                materialize_registry_secrets=False,
            ),
            assume_decision=assume_decision,
            enabled=preflight_images and not plan_only,
        )

        if deploy_if_absent:
            from npa.orchestration.npa_workflow.deploy import (
                ensure_infra_present,
                parse_deploy_targets,
            )
            from npa.orchestration.npa_workflow.spec import load_spec as _load_deploy_spec

            try:
                deploy_targets = parse_deploy_targets(_load_deploy_spec(yaml_path))
                if deploy_targets:
                    for record in ensure_infra_present(
                        deploy_targets,
                        dry_run=plan_only,
                        gpu_readiness_timeout=gpu_readiness_timeout,
                        gpu_readiness_poll_interval=gpu_readiness_poll_interval,
                        sky_bin=sky_bin,
                    ):
                        typer.echo(
                            "deployIfAbsent["
                            f"{record['profile']}]: {record['status']} "
                            f"context={record['context']} "
                            f"actions={','.join(record['actions']) or 'none'}",
                            err=True,
                        )
                        # A `partial` outcome (no project_id, no bucket, ...) used to
                        # print its status with the reason dropped, and the submit
                        # carried on into a launch that could not work.
                        for warning in record.get("warnings", []) or []:
                            typer.echo(
                                f"deployIfAbsent[{record['profile']}]: warning: {warning}",
                                err=True,
                            )
            except NpaWorkflowError as exc:
                _fail(str(exc))
                return

        # Provisioning may have just created the context (or failed to). Either way
        # the launch cannot work without it, so stop here with the remedy rather
        # than deep inside `sky jobs launch`.
        if infra_context and not plan_only and not _adopt_npa_kubeconfig(infra_context):
            _fail(
                f"Kube context {infra_context!r} (from --infra {infra!r}) is not available: "
                "it is not in your kubeconfig and npa has no kubeconfig for it under "
                f"~/.npa/clusters/{infra_context}/. Provision the cluster with "
                "`npa provision-if-absent --project <alias>` (check its output for "
                "warnings), or point KUBECONFIG at the cluster you want and pass a "
                "context from `kubectl config get-contexts`."
            )
            return
        npa_render_options = SkypilotRenderOptions(
            registry=_resolve_submit_registry(registry, project),
            image_overrides=image_overrides,
            aws_endpoint_url=s3_endpoint
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("NEBIUS_S3_ENDPOINT")
            or "https://storage.eu-north1.nebius.cloud",
            gpu_target=gpu_target,
            image_variant=image_variant,
            # Never mint/print live registry tokens for --plan-only.
            materialize_registry_secrets=not plan_only,
            gpu_accelerator_overrides=_resolve_submit_accelerators(
                yaml_path,
                infra=infra,
                sky_bin=sky_bin,
                enabled=resolve_accelerators and not plan_only,
                readiness_timeout=gpu_readiness_timeout,
                readiness_poll_interval=gpu_readiness_poll_interval,
            ),
        )

        if runtime and not plan_only:
            _run_npa_workflow_runtime(
                yaml_path,
                run_id=resolved_run_id,
                assume_decision=assume_decision,
                config_overrides=substitutions,
                render_options=npa_render_options,
                secret_envs=secret_env,
                secret_env_values=extra_env,
                controller_backend=controller_backend.value,
                infra=infra,
                isolated_config_dir=isolated_config_dir,
                submit_timeout=submit_timeout,
                poll_seconds=poll_seconds,
                max_wait_seconds=max_wait_seconds,
                cancel_on_timeout=cancel_on_timeout,
                retries=retries,
                max_concurrency=max_concurrency,
                resume=resume,
                output_format=output_format,
                project=project,
                auto_load=auto_load,
                agent_name=agent_name,
                s3_endpoint=s3_endpoint,
            )
            return

        try:
            prepared_npa = prepare_npa_workflow_for_submit(
                yaml_path,
                run_id=resolved_run_id,
                assume_decision=assume_decision,
                config_overrides=substitutions,
                render_options=npa_render_options,
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

        _refresh_kubernetes_pull_secrets(prepared_npa.skypilot_yaml_path)

        # Skip SkyPilot-path materializers; npa.workflow already planned.
        materializer = ""
        substitutions = {}
        yaml_path = prepared_npa.skypilot_yaml_path
        if prepared_npa.secret_env_hints:
            missing_secret_hints = [
                name
                for name in prepared_npa.secret_env_hints
                if name not in secret_env
            ]
            if missing_secret_hints:
                typer.echo(
                    "Hint: consider --secret-env "
                    + " --secret-env ".join(missing_secret_hints),
                    err=True,
                )

    submitted_yaml_path = yaml_path
    submitted_yaml_context: tempfile.TemporaryDirectory[str] | None = None
    workflow_state = None
    instrumented = None
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

        def _launch() -> WorkflowResult:
            return submit_workflow(
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

        if prepared_npa is not None:
            from npa.orchestration.npa_workflow.submission_state import (
                load_submission_state,
                submission_lock,
                update_submission_state,
            )

            ledger_project = project or "default"
            with submission_lock(ledger_project, resolved_run_id):
                update_submission_state(
                    ledger_project,
                    resolved_run_id,
                    {"workflow": _npa_submission_receipt(prepared_npa, resolved_run_id)},
                    locked=True,
                )
                submission = load_submission_state(ledger_project, resolved_run_id)
                launch_state = submission.get("launch")
                launch_record = launch_state if isinstance(launch_state, dict) else {}
                existing_job_id = str(launch_record.get("sky_job_id") or "")
                if not existing_job_id and launch_record.get("status") == "launching":
                    matches = find_job_ids_by_name(
                        resolved_run_id,
                        isolated_config_dir=isolated_config_dir,
                        config_path=config_path,
                        sky_bin=sky_bin or None,
                    )
                    existing_job_id = matches[0] if matches else ""
                if existing_job_id:
                    result = WorkflowResult(
                        status="SUBMITTED",
                        job_id=existing_job_id,
                        log_paths={"resume": "existing managed job; no duplicate launch"},
                    )
                    update_submission_state(
                        ledger_project,
                        resolved_run_id,
                        {
                            "launch": {
                                "status": "submitted",
                                "sky_job_id": existing_job_id,
                                "resumed": True,
                            }
                        },
                        locked=True,
                    )
                    typer.echo(
                        f"submit-resume: reusing managed job {existing_job_id}; no duplicate launch",
                        err=True,
                    )
                else:
                    update_submission_state(
                        ledger_project,
                        resolved_run_id,
                        {"launch": {"status": "launching", "sky_job_id": ""}},
                        locked=True,
                    )
                    try:
                        result = _launch()
                    except Exception:
                        update_submission_state(
                            ledger_project,
                            resolved_run_id,
                            {"launch": {"status": "failed", "sky_job_id": ""}},
                            locked=True,
                        )
                        raise
                    update_submission_state(
                        ledger_project,
                        resolved_run_id,
                        {
                            "launch": {
                                "status": "submitted",
                                "sky_job_id": result.job_id,
                                "resumed": False,
                            }
                        },
                        locked=True,
                    )
        else:
            result = _launch()
        if workflow_state is not None and instrumented is not None:
            instrumented_manifest = write_manifest(
                instrumented.manifest,
                workflow_state,
                job_id=result.job_id,
            )
            result.log_paths["run_prefix_uri"] = workflow_state.uri
            result.log_paths["manifest_uri"] = f"{workflow_state.uri.rstrip('/')}/manifest.json"
            result.log_paths["stages"] = ",".join(instrumented_manifest.get("stages", {}).keys())
        if prepared_npa is not None:
            # Persist the npa.workflow run manifest for the submitted run. Only the
            # local `run-spec --persist-state` path used to write it, so a run that
            # actually reached the cluster left no `npa.workflow.run.v1` record and
            # was invisible to every manifest consumer (e.g. the insights GPU metric).
            run_prefix_uri = _persist_npa_run_manifest(
                prepared_npa,
                run_id=resolved_run_id,
                job_id=str(getattr(result, "job_id", "") or ""),
                s3_endpoint=s3_endpoint,
                credential_values=extra_env,
                accelerator_overrides=npa_render_options.gpu_accelerator_overrides,
                accelerator_override=os.environ.get(
                    "NPA_WORKFLOW_GPU_ACCELERATOR", ""
                ).strip(),
            )
            # ``result`` is union-typed across submit paths; only the workflow
            # result carries log_paths, so probe for it instead of assuming.
            log_paths = getattr(result, "log_paths", None)
            if run_prefix_uri and isinstance(log_paths, dict):
                log_paths.setdefault("npa_workflow_run_prefix_uri", run_prefix_uri)
                log_paths.setdefault(
                    "npa_workflow_manifest_uri",
                    f"{run_prefix_uri.rstrip('/')}/npa-workflow/manifest.json",
                )
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
    log_paths = getattr(result, "log_paths", {})
    if isinstance(log_paths, dict) and log_paths.get("npa_workflow_manifest_uri"):
        typer.echo(f"manifest_uri: {log_paths['npa_workflow_manifest_uri']}")


def _persist_npa_run_manifest(
    prepared,
    *,
    run_id: str,
    job_id: str = "",
    s3_endpoint: str = "",
    credential_values: dict[str, str] | None = None,
    accelerator_overrides: Mapping[str, str] | None = None,
    accelerator_override: str = "",
) -> str:
    """Write the `npa.workflow.run.v1` manifest for a submitted run (best effort).

    A failed manifest write must never turn an accepted submit into a reported
    failure, but it must be visible, so the failure is warned about rather than
    swallowed. Returns the run prefix URI (``""`` when the spec sets no config.bucket).
    """
    from npa.orchestration.npa_workflow.run_state import persist_submitted_manifest
    from npa.orchestration.npa_workflow.runtime import _resolved_config

    try:
        config = _resolved_config(prepared.spec, run_id)
        return persist_submitted_manifest(
            config,
            run_id=run_id,
            workflow=prepared.spec.name,
            api_version=prepared.spec.api_version,
            steps=prepared.plan.steps,
            sky_job_id=job_id,
            endpoint_url=s3_endpoint,
            aws_access_key_id=(credential_values or {}).get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=(credential_values or {}).get(
                "AWS_SECRET_ACCESS_KEY", ""
            ),
            accelerator_overrides=accelerator_overrides,
            accelerator_override=accelerator_override,
        )
    except Exception as exc:  # noqa: BLE001 - never fail an accepted submit
        typer.echo(f"warning: could not persist the npa.workflow run manifest: {exc}", err=True)
        return ""


def _npa_submission_receipt(prepared, run_id: str) -> dict[str, object]:
    """Record the non-secret run contract before the managed launch side effect."""

    return _workflow_submission_receipt(
        prepared.spec,
        prepared.plan.steps,
        run_id,
    )


def _workflow_submission_receipt(spec, steps, run_id: str) -> dict[str, object]:
    """Build the shared non-secret receipt for runtime and single-job submits."""

    from npa.orchestration.npa_workflow.run_state import (
        PAIDF_WORKFLOW_NAME,
        paidf_artifact_prefix,
        plan_step_records,
    )
    from npa.orchestration.npa_workflow.runtime import _resolved_config

    config = _resolved_config(spec, run_id)
    bucket = str(config.get("bucket") or "").strip()
    prefix = str(config.get("prefix") or run_id).strip("/")
    if spec.name == PAIDF_WORKFLOW_NAME:
        canonical = paidf_artifact_prefix(run_id)
        if prefix != canonical:
            raise RuntimeError(
                "PAIDF run prefix must use the canonical contract "
                f"{canonical!r}, got {prefix!r}"
            )
    run_prefix_uri = f"s3://{bucket}/{prefix}" if bucket and prefix else ""
    return {
        "name": spec.name,
        "api_version": spec.api_version,
        "run_prefix_uri": run_prefix_uri,
        "manifest_uri": (
            f"{run_prefix_uri}/npa-workflow/manifest.json" if run_prefix_uri else ""
        ),
        "steps": plan_step_records(steps),
    }


def _run_npa_workflow_runtime(
    yaml_path: Path,
    *,
    run_id: str,
    assume_decision: str,
    config_overrides: dict[str, str],
    render_options,
    secret_envs: list[str],
    secret_env_values: dict[str, str],
    controller_backend: str,
    infra: str,
    isolated_config_dir: Path | None,
    submit_timeout: int,
    poll_seconds: int,
    max_wait_seconds: int,
    cancel_on_timeout: bool,
    retries: int,
    max_concurrency: int,
    resume: bool,
    output_format: "OutputFormat",
    project: str = "",
    auto_load: bool = True,
    agent_name: str = "",
    s3_endpoint: str = "",
) -> None:
    """Drive an npa.workflow spec through the runtime orchestrator tier."""

    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.runtime import (
        RuntimeOptions,
        run_workflow_runtime,
        secret_env_names,
    )
    from npa.orchestration.npa_workflow.submit import load_spec_for_submit

    try:
        spec = load_spec_for_submit(yaml_path, config_overrides=config_overrides)
    except NpaWorkflowError as exc:
        _fail(str(exc))
        return

    from npa.orchestration.npa_workflow.runtime import plan_preview
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    try:
        receipt_plan = plan_preview(
            spec,
            run_id=run_id,
            assume_decision=(
                assume_decision or str(spec.config.get("plan_assume_decision") or "")
            ),
        )
        receipt_steps = receipt_plan.steps
    except Exception:  # noqa: BLE001 - runtime remains the authoritative planner
        receipt_steps = []
    update_submission_state(
        project or "default",
        run_id,
        {"workflow": _workflow_submission_receipt(spec, receipt_steps, run_id)},
    )

    options = RuntimeOptions(
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
        retries=max(0, retries),
        cancel_on_timeout=cancel_on_timeout,
        max_concurrency=max(0, max_concurrency),
        secret_envs=secret_env_names(secret_envs, values=secret_env_values),
        secret_env_values=secret_env_values,
        submit_timeout=submit_timeout,
        infra=infra,
        controller_backend=controller_backend,
        isolated_config_dir=isolated_config_dir,
        resume=resume,
    )
    runtime_env = dict(secret_env_values)
    endpoint = str(getattr(render_options, "aws_endpoint_url", "") or "").strip()
    if endpoint:
        runtime_env.setdefault("AWS_ENDPOINT_URL", endpoint)
        runtime_env.setdefault("NEBIUS_S3_ENDPOINT", endpoint)
    previous_env = {name: os.environ.get(name) for name in runtime_env}
    try:
        os.environ.update(runtime_env)
        try:
            report = run_workflow_runtime(
                spec,
                run_id=run_id,
                render_options=render_options,
                options=options,
                assume_decision=assume_decision,
                logger=lambda message: typer.echo(f"[runtime] {message}", err=True),
            )
        except NpaWorkflowError as exc:
            _fail(str(exc))
            return
    finally:
        for name, previous in previous_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    artifact_load: dict[str, object] | None = None
    if report.status == "succeeded" and auto_load and report.workflow == "physical-ai-data-factory":
        artifact_load = _load_paidf_artifact(
            project=project,
            run_id=run_id,
            run_prefix_uri=report.run_prefix_uri,
            s3_endpoint=s3_endpoint,
            credential_values=secret_env_values,
            agent_name=agent_name,
        )
    payload = report.to_dict()
    if artifact_load is not None:
        payload["artifact_load"] = artifact_load
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {report.status}")
        typer.echo(f"run_id: {report.run_id}")
        typer.echo(f"waves: {len(report.waves)}")
        for wave in report.waves:
            states = ",".join(wave.get("states") or [])
            typer.echo(
                f"  {wave.get('key')}: {wave.get('status')} "
                f"[{wave.get('kind')}] states={states} job_id={wave.get('job_id')}"
            )
        for decision in report.decisions:
            typer.echo(f"decision: {decision.get('decision')} <- {decision.get('uri')}")
        if report.run_prefix_uri:
            typer.echo(f"run_prefix_uri: {report.run_prefix_uri}")
        if report.error:
            typer.echo(f"error: {report.error}")
        if artifact_load is not None:
            typer.echo(f"artifact_load: {artifact_load.get('status')}")
            if artifact_load.get("artifact_uri"):
                typer.echo(f"artifact_uri: {artifact_load.get('artifact_uri')}")
            if artifact_load.get("detail"):
                typer.echo(f"artifact_load_detail: {artifact_load.get('detail')}")
            if artifact_load.get("retry_command") and not artifact_load.get("verified"):
                typer.echo(f"artifact_load_retry: {artifact_load.get('retry_command')}")
    if report.status != "succeeded":
        raise typer.Exit(1)


def _load_paidf_artifact(
    *,
    project: str,
    run_id: str,
    run_prefix_uri: str,
    s3_endpoint: str = "",
    credential_values: dict[str, str] | None = None,
    agent_name: str = "",
) -> dict[str, object]:
    """Run the optional post-success handoff without changing workflow success."""

    from npa.orchestration.npa_workflow.artifact_load import (
        load_final_artifact_into_agent,
    )
    from npa.orchestration.npa_workflow.src_staging import _storage_client
    from npa.orchestration.npa_workflow.submission_state import update_submission_state

    if not run_prefix_uri:
        result: dict[str, object] = {
            "status": "partial",
            "detail": "workflow succeeded but its exact run prefix is unavailable",
            "retry_command": (
                f"npa workbench workflow load-artifact {run_id}"
                + (f" --project {project}" if project else "")
            ),
            "verified": False,
        }
        update_submission_state(
            project or "default", run_id, {"artifact_load": result}
        )
        return result
    try:
        client = _storage_client(
            endpoint_url=s3_endpoint,
            aws_access_key_id=(credential_values or {}).get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=(credential_values or {}).get("AWS_SECRET_ACCESS_KEY", ""),
        )
        return load_final_artifact_into_agent(
            project=project,
            run_id=run_id,
            run_prefix_uri=run_prefix_uri,
            storage_client=client,
            agent_name=agent_name,
        ).to_dict()
    except Exception as exc:  # noqa: BLE001 - optional post-success operation
        result = {
            "status": "partial",
            "detail": f"workflow succeeded; artifact load is incomplete: {exc}",
            "retry_command": (
                f"npa workbench workflow load-artifact {run_id}"
                + (f" --project {project}" if project else "")
                + (f" --agent-name {agent_name}" if agent_name else "")
            ),
            "verified": False,
        }
        update_submission_state(
            project or "default", run_id, {"artifact_load": result}
        )
        return result


def _default_submit_run_id(yaml_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(yaml_path).stem).strip("-")
    return stem or "workflow"


def _is_nebius_registry_image(image: str) -> bool:
    value = str(image or "").removeprefix("docker:").strip()
    host = value.split("/", 1)[0] if "/" in value else ""
    return host.startswith("cr.") and host.endswith(".nebius.cloud")


def _resolve_submit_src_s3_uri(project: str) -> str:
    """Resolve source for the selected project, with explicit env precedence."""

    value = (
        os.environ.get("NPA_SRC_S3_URI")
        or os.environ.get("NPA_E2E_NPA_SRC_S3_URI")
        or ""
    ).strip()
    if value:
        return value
    try:
        from npa.clients.config import resolve_workflow_src_s3_uri

        return resolve_workflow_src_s3_uri(project or None)
    except Exception:  # noqa: BLE001 - submit preflight reports an unset source
        return ""


def _resolve_submit_registry(registry: str, project: str) -> str:
    """Return the registry a submit should pull from.

    An explicit --registry wins. Otherwise use the registry `npa configure` saved
    for the project: the image pins otherwise resolved against the first-party
    default even when the operator had selected (or been given) a project
    registry, so preflight checked one registry while the run pulled from another
    and the printed build command targeted the wrong place.
    """

    explicit = str(registry or "").strip()
    if explicit:
        return explicit
    configured_env = str(os.environ.get("NPA_REGISTRY") or "").strip()
    if configured_env:
        return configured_env
    try:
        from npa.clients.config import resolve_container_registry

        return str(resolve_container_registry(project or None) or "").strip()
    except Exception:  # noqa: BLE001 - fall back to the render's own default
        return ""


def _preflight_submit_images(
    yaml_path: Path,
    *,
    options: SkypilotRenderOptions,
    assume_decision: str,
    enabled: bool,
) -> None:
    """Fail before the run starts when a step's image cannot actually be pulled.

    Only Nebius registry images are treated as blocking: those are the ones the
    credentials this submit injects are authoritative for. Anything else is
    reported so the operator sees it, but a third-party registry that needs its
    own in-pod credentials must not block submit.
    """

    if not enabled:
        return

    from npa.orchestration.npa_workflow import build_plan
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.skypilot_render import plan_images
    from npa.orchestration.skypilot.registry_preflight import check_image_pulls_with_credentials

    try:
        spec = load_spec(yaml_path)
        run_id = f"{spec.name}-preflight"
        plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
        images = plan_images(spec, plan.steps, run_id=run_id, options=options)
    except NpaWorkflowError:
        # Planning problems are reported by the submit path itself with better context.
        return
    if not images:
        return

    checks = check_image_pulls_with_credentials(images, mint=True)
    blocking = []
    for check in checks:
        if check.ok:
            continue
        blocking.append(check)
    if blocking:
        detail = "\n".join(check.render() for check in blocking)
        _fail(
            "image-preflight failed; the run would sit in ImagePullBackOff rather than "
            f"fail, so it was not submitted:\n{detail}"
        )
    typer.echo(f"image-preflight: {len(checks)} image(s) pullable", err=True)


def _resolve_submit_accelerators(
    yaml_path: Path,
    *,
    infra: str,
    sky_bin: str,
    enabled: bool,
    readiness_timeout: float = 600.0,
    readiness_poll_interval: float = 10.0,
) -> dict[str, str]:
    """Map a spec's Kubernetes accelerators onto what the target cluster advertises.

    Returns a possibly empty remap keyed by the spec's own accelerator string. A
    Discovery is a readiness gate, not a best-effort rewrite: Kubernetes may
    already report allocatable GPUs while SkyPilot still has an empty catalog.
    Submission waits for the configured bounded interval and then fails without
    deleting capacity.
    """

    if not enabled:
        return {}
    if os.environ.get("NPA_WORKFLOW_GPU_ACCELERATOR", "").strip():
        # An explicit blanket override is the operator's decision; honor it as-is.
        return {}

    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.skypilot._bin import SkyPilotNotInstalledError
    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        KubernetesGpuCatalogError,
        UnsatisfiableAcceleratorError,
        context_from_infra,
        spec_accelerators,
        wait_for_kubernetes_accelerators,
    )

    try:
        requested = spec_accelerators(load_spec(yaml_path).resources)
    except NpaWorkflowError:
        return {}
    if not requested:
        return {}

    context = context_from_infra(infra) or os.environ.get("KUBECONTEXT", "").strip()
    try:
        resolutions = wait_for_kubernetes_accelerators(
            requested,
            context=context,
            sky_bin=sky_bin or None,
            timeout=readiness_timeout,
            poll_interval=readiness_poll_interval,
            on_status=lambda message: typer.echo(message, err=True),
        )
    except (
        KubernetesGpuCatalogError,
        SkyPilotNotInstalledError,
        UnsatisfiableAcceleratorError,
        ValueError,
    ) as exc:
        _fail(f"accelerator readiness failed: {exc}")
        return {}

    overrides: dict[str, str] = {}
    for accelerator, resolution in resolutions.items():
        if resolution.remapped:
            overrides[accelerator] = resolution.resolved
        typer.echo(f"accelerator-resolve: {resolution.describe()}", err=True)
    return overrides


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


#: Bucket values that are spec placeholders rather than real storage. Shipped
#: specs default to ``example-bucket`` so they validate offline; planning or
#: running against that value looks like a real plan but writes nowhere useful.
_PLACEHOLDER_BUCKETS = frozenset({"", "example-bucket", "your-bucket", "my-bucket"})


def _is_placeholder_bucket(bucket: str) -> bool:
    value = str(bucket or "").strip()
    value = value.removeprefix("s3://").strip("/")
    if not value or value in _PLACEHOLDER_BUCKETS:
        return True
    return "<" in value or ">" in value


def _warn_placeholder_bucket(config, *, quiet: bool = False) -> None:
    """Warn when a spec is being planned/run against its placeholder bucket.

    ``quiet`` suppresses the notice for machine-readable output (``--json``),
    where the caller wants a clean document and the shipped specs all default to
    ``bucket: example-bucket``.
    """
    if quiet:
        return
    bucket = str((config or {}).get("bucket", "") or "")
    if not _is_placeholder_bucket(bucket):
        return
    shown = bucket or "<unset>"
    typer.echo(
        f"Warning: config.bucket is {shown!r}, a spec placeholder rather than "
        "your configured storage. Pass `--var bucket=<your-bucket>` so artifact "
        "URIs point at a bucket you can actually read.",
        err=True,
    )


def _npa_spec_config(yaml_path: Path, substitutions: dict[str, str]) -> dict:
    """Return an npa.workflow spec's config with ``--var`` overrides applied."""
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.spec import load_spec

    try:
        spec = load_spec(yaml_path)
    except NpaWorkflowError:
        # Let the real load happen later so its error is the one reported.
        return dict(substitutions)
    config = dict(spec.config)
    config.update(substitutions)
    return config


def _stage_npa_src_for_submit(
    spec_config: dict,
    *,
    s3_bucket: str = "",
    s3_endpoint: str = "",
    credential_values: dict[str, str] | None = None,
    project: str = "",
    run_id: str = "workflow",
) -> str:
    """Upload once, then durably record the exact ``NPA_SRC_S3_URI``."""
    from npa.clients.config import ConfigError, persist_workflow_src_s3_uri
    from npa.orchestration.npa_workflow.src_staging import SrcStagingError, stage_npa_source
    from npa.orchestration.npa_workflow.submission_state import (
        submission_lock,
        update_submission_state,
    )

    bucket = str(s3_bucket or spec_config.get("bucket", "") or "").strip()
    if _is_placeholder_bucket(bucket):
        _fail(
            "--stage-src needs a real bucket. Pass --var bucket=<your-bucket> "
            "(or --s3-bucket <your-bucket>)."
        )
        return ""
    try:
        with submission_lock(project or "default", run_id):
            uri = stage_npa_source(
                bucket=bucket,
                endpoint_url=s3_endpoint,
                aws_access_key_id=(credential_values or {}).get("AWS_ACCESS_KEY_ID", ""),
                aws_secret_access_key=(credential_values or {}).get(
                    "AWS_SECRET_ACCESS_KEY", ""
                ),
                on_status=lambda message: typer.echo(f"  {message}", err=True),
            )
            persist_workflow_src_s3_uri(uri, project or None)
            fingerprint = uri.rstrip("/").rsplit("/", 1)[-1]
            update_submission_state(
                project or "default",
                run_id,
                {
                    "source": {
                        "status": "verified",
                        "uri": uri,
                        "fingerprint": fingerprint,
                    }
                },
                locked=True,
            )
            return uri
    except (ConfigError, SrcStagingError) as exc:
        _fail(str(exc))
        return ""


def _infra_kube_context(infra: str) -> str:
    """Return the kube context named by ``--infra k8s/<context>``, or "".

    Only ``k8s``/``kubernetes`` targets that pin an explicit context are
    returned; a bare ``k8s`` (SkyPilot uses the current context), a non-k8s
    target, or an empty value yield "".
    """
    value = str(infra or "").strip()
    if "/" not in value:
        return ""
    kind, _, context = value.partition("/")
    if kind.strip().lower() not in {"k8s", "kubernetes"}:
        return ""
    return context.strip()


def _available_kube_contexts() -> list[str] | None:
    """Return context names from the active kubeconfig(s), or None if unreadable.

    Reads ``KUBECONFIG`` (``:``-separated, as SkyPilot/kubectl do) or
    ``~/.kube/config``. Returns ``None`` (not ``[]``) when no kubeconfig file can
    be read, so the caller can skip the check rather than false-fail; an empty
    list means "readable, but defines no contexts".
    """
    import yaml as _yaml

    raw = os.environ.get("KUBECONFIG", "").strip()
    paths = (
        [Path(p) for p in raw.split(os.pathsep) if p.strip()]
        if raw
        else [Path.home() / ".kube" / "config"]
    )
    contexts: list[str] = []
    read_any = False
    for path in paths:
        try:
            if not path.is_file():
                continue
            data = _yaml.safe_load(path.read_text()) or {}
        except (OSError, _yaml.YAMLError):
            continue
        read_any = True
        for entry in (data.get("contexts") or []):
            name = str((entry or {}).get("name", "") or "").strip()
            if name and name not in contexts:
                contexts.append(name)
    return contexts if read_any else None


def _adopt_npa_kubeconfig(context: str) -> bool:
    """Make an npa-provisioned kube context usable by SkyPilot, and say whether it is.

    `npa cluster up` / `npa provision-if-absent` write a dedicated kubeconfig under
    ``~/.npa/clusters/<context>/kubeconfig`` instead of merging into
    ``~/.kube/config``, while `sky jobs launch` reads ``KUBECONFIG`` (or
    ``~/.kube/config``). A cluster npa had just created was therefore invisible to
    the submit that asked for it — `Context <name> not found ... Available
    contexts: []` — unless the operator knew to export KUBECONFIG by hand. Prepend
    npa's kubeconfig when the context is missing from the active one.
    """
    if not context:
        return False
    available = _available_kube_contexts()
    if available is not None and context in available:
        return True

    from npa.cluster.state import existing_kubeconfig

    path = existing_kubeconfig(context)
    if path is None:
        return False
    current = os.environ.get("KUBECONFIG", "").strip()
    entries = [str(path)] + [
        entry for entry in current.split(os.pathsep) if entry.strip() and entry != str(path)
    ]
    os.environ["KUBECONFIG"] = os.pathsep.join(entries)
    typer.echo(
        f"Using the npa kubeconfig for context {context!r}: {path} "
        "(prepended to KUBECONFIG for this run).",
        err=True,
    )
    return True


def _spec_self_provisions(yaml_path: Path) -> bool:
    """Whether the spec declares ``deployIfAbsent`` targets submit can provision.

    Those specs create their own cluster (and therefore its kube context) later in
    this same submit, so a context the kubeconfig does not have yet is not a
    missing prerequisite.
    """
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError

    try:
        from npa.orchestration.npa_workflow.deploy import parse_deploy_targets
        from npa.orchestration.npa_workflow.spec import load_spec

        return bool(parse_deploy_targets(load_spec(yaml_path)))
    except (NpaWorkflowError, OSError, ValueError):
        # A spec this cannot load fails with a real error further down; never let
        # the preflight's own bookkeeping be the thing that fails a submit.
        return False


def _spec_requires_s3(yaml_path: Path) -> bool:
    """Whether a workflow declares an S3 artifact/data contract.

    This is intentionally spec-aware rather than a blanket submit requirement:
    CPU/local or image-only workflows without S3 handoffs must remain usable
    when object storage is not configured.
    """

    import yaml as _yaml

    try:
        document = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, _yaml.YAMLError):
        return False

    def _walk(value) -> bool:
        if isinstance(value, dict):
            return any(_walk(item) for item in value.values())
        if isinstance(value, list):
            return any(_walk(item) for item in value)
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower()
        return normalized.startswith("s3://") or (
            "s3://" in normalized and ("{bucket}" in normalized or "${" in normalized)
        )

    return _walk(document)


def _submit_prerequisites(
    spec_config: dict,
    *,
    sky_bin: str,
    image: str,
    plan_only: bool,
    infra: str = "",
    self_provisions: bool = False,
    requires_s3: bool = False,
    s3_endpoint: str = "",
    s3_access_key_id: str = "",
    s3_secret_access_key: str = "",
    source_staging_planned: bool = False,
) -> list[tuple[str, str]]:
    """Return ``[(missing, remedy)]`` for an npa.workflow submit.

    A first submit used to fail one prerequisite at a time — no npa source, then
    no SkyPilot CLI, then a placeholder bucket, then an unresolvable kube context
    — each as a separate run. Collect them so the operator sees the whole list
    once.
    """
    # Same resolver the renderer uses, so a prefix persisted with
    # `npa configure --src-s3-uri` satisfies the check without re-exporting it.
    from npa.orchestration.npa_workflow.skypilot_render import resolve_src_s3_uri

    missing: list[tuple[str, str]] = []

    if not plan_only:
        from npa.orchestration.skypilot._bin import (
            SkyPilotConfigError,
            SkyPilotNotInstalledError,
            resolve_sky_bin,
        )

        try:
            resolve_sky_bin(sky_bin or None)
        except (SkyPilotNotInstalledError, SkyPilotConfigError) as exc:
            missing.append(
                (
                    f"SkyPilot CLI is not usable ({exc})",
                    "run `npa skypilot bootstrap` (it now saves skypilot.sky_bin "
                    "into ~/.npa/config.yaml)",
                )
            )

    # Image-less steps install npa from S3 on the worker. `--image none` pins
    # every task to SkyPilot's default image, so it needs the source too.
    image_value = str(image or "").strip().lower()
    image_pins_tasks = bool(image_value) and image_value not in {"none", "default", "-"}
    if not image_pins_tasks and not resolve_src_s3_uri() and not source_staging_planned:
        missing.append(
            (
                "npa source for image-less steps (NPA_SRC_S3_URI is unset)",
                "pass --stage-src, or set NPA_SRC_S3_URI=s3://<bucket>/npa-src/npa, "
                "or pin --image <registry>/npa-<tool>:<tag>",
            )
        )

    bucket = str((spec_config or {}).get("bucket", "") or "")
    if not plan_only and requires_s3 and _is_placeholder_bucket(bucket):
        missing.append(
            (
                f"config.bucket is the spec placeholder {bucket or '<unset>'!r}",
                "pass --var bucket=<your-bucket>",
            )
        )

    if not plan_only and requires_s3 and not _is_placeholder_bucket(bucket):
        from npa.clients.storage_validation import probe_storage_write

        probe = probe_storage_write(
            bucket=bucket,
            endpoint_url=s3_endpoint,
            access_key_id=s3_access_key_id,
            secret_access_key=s3_secret_access_key,
        )
        if not probe.ok:
            missing.append(
                (
                    f"writable S3 for this workflow ({probe.summary})",
                    "run `npa provision-if-absent --project <alias> --skip-k8s`, "
                    "then retry; the probe object is deleted before a successful preflight",
                )
            )

    # Catch an `--infra k8s/<context>` that names a context the kubeconfig does
    # not define, up front. Otherwise `sky jobs launch` fails late with a long
    # SkyPilot stack ("Context <name> not found ... Available contexts: []") —
    # e.g. after a stale controller is purged but no real cluster was provisioned.
    # Skipped for specs that provision that context themselves later in this
    # submit (`--deploy-if-absent`, on by default), where its absence is the
    # normal starting state rather than a missing prerequisite.
    context = _infra_kube_context(infra)
    if not plan_only and context and not self_provisions:
        available = _available_kube_contexts()
        if available is not None and context not in available:
            shown = ", ".join(available) if available else "none"
            missing.append(
                (
                    f"kube context {context!r} (from --infra {infra!r}) is not in "
                    f"your kubeconfig (available: {shown})",
                    "provision a cluster (`npa provision-if-absent --project "
                    "<alias>`) or point KUBECONFIG at the target cluster, then pass "
                    "`--infra k8s/<context>` with a context from "
                    "`kubectl config get-contexts`",
                )
            )
    return missing


def _fail_missing_prerequisites(
    yaml_path: Path, missing: list[tuple[str, str]]
) -> None:
    lines = [f"Cannot submit {yaml_path.name}: missing prerequisites:"]
    for item, remedy in missing:
        lines.append(f"  - {item}")
        lines.append(f"      fix: {remedy}")
    lines.append("  (bypass these checks with --skip-preflight)")
    _fail("\n".join(lines))


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
    from npa.orchestration.npa_workflow.run_resolution import (
        resolution_diagnostics,
        resolve_run,
    )

    resolution = resolve_run(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
    )
    if resolution.state is not None:
        return resolution.state
    raise RuntimeError("\n".join(resolution_diagnostics(resolution)))


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
    from npa.orchestration.npa_workflow.run_resolution import run_id_from_locator

    return run_id_from_locator(run_id)


def _resolve_sky_bin(sky_bin: str = "") -> str:
    from npa.orchestration.skypilot._bin import ensure_skypilot_version, resolve_config

    config = resolve_config(sky_bin=sky_bin or None)
    return str(ensure_skypilot_version(config.sky_bin))


def _durable_workflow_status(
    run_id: str,
    *,
    project: str = "",
    workflow_s3_uri: str = "",
    workflow_s3_prefix: str = "",
    s3_bucket: str = "",
    s3_endpoint: str = "",
    sky_bin: str = "",
    startup_failure_threshold: int = 3,
) -> dict[str, object]:
    from npa.orchestration.skypilot.workflow import (
        workflow_controller_logs,
        workflow_status,
        workflow_task_statuses,
    )
    from npa.orchestration.npa_workflow.run_resolution import (
        resolution_diagnostics,
        resolve_run,
    )
    from npa.orchestration.skypilot.workflow_state import read_stage_status

    resolution = resolve_run(
        run_id,
        project=project,
        workflow_s3_uri=workflow_s3_uri,
        workflow_s3_prefix=workflow_s3_prefix,
        s3_bucket=s3_bucket,
        s3_endpoint=s3_endpoint,
        sky_bin=sky_bin,
    )
    if not resolution.found:
        return {
            "run_id": resolution.run_id,
            "workflow_name": resolution.workflow_name,
            "status": (
                "NOT_FOUND"
                if resolution.conclusively_absent
                else "VERIFICATION_UNAVAILABLE"
            ),
            "verification": (
                "conclusively_absent"
                if resolution.conclusively_absent
                else "unavailable"
            ),
            "manifest_state": "absent" if resolution.conclusively_absent else "unknown",
            "manifest_pending": False,
            "resolution_source": "",
            "resolution_checks": resolution.checks_payload(),
            "live_status": "",
            "sky_job_id": "",
            "run_prefix_uri": resolution.run_prefix_uri,
            "manifest_uri": resolution.manifest_uri,
            "stages": {},
            "diagnostics": resolution_diagnostics(resolution),
        }

    state = resolution.state
    manifest = resolution.manifest
    if manifest is None:
        return _manifest_pending_status(
            resolution,
            project=project,
            sky_bin=sky_bin,
            startup_failure_threshold=startup_failure_threshold,
        )
    assert state is not None
    if manifest.get("schema_version") == "npa.workflow.run.v1":
        from npa.orchestration.npa_workflow.run_state import (
            RunManifest,
            build_actionable_run_status,
            reconcile_submitted_manifest,
        )

        run_manifest = RunManifest.from_dict(manifest)
        if not run_manifest.sky_job_id and resolution.job_id:
            run_manifest.sky_job_id = resolution.job_id
        live_status = ""
        task_rows: list[dict[str, object]] = []
        controller_output = ""
        diagnostics: list[str] = []
        if run_manifest.sky_job_id:
            try:
                live = workflow_status(
                    run_manifest.sky_job_id, sky_bin=sky_bin or None
                )
                if live.error:
                    diagnostics.append(
                        "SkyPilot controller status is unavailable; showing the last "
                        f"persisted manifest state ({live.error})."
                    )
                else:
                    live_status = live.status
                task_rows = workflow_task_statuses(
                    run_manifest.sky_job_id, sky_bin=sky_bin or None
                )
                if str(live_status or run_manifest.status).upper() not in {
                    "SUCCEEDED",
                    "CANCELLED",
                } and not str(live_status or run_manifest.status).upper().startswith("FAILED"):
                    controller_logs = workflow_controller_logs(
                        run_manifest.sky_job_id,
                        sky_bin=sky_bin or None,
                    )
                    controller_output = "\n".join(
                        item for item in (controller_logs.stdout, controller_logs.stderr) if item
                    )
                    if controller_logs.returncode != 0:
                        diagnostics.append(
                            "SkyPilot controller logs are unavailable; startup failure "
                            f"classification is incomplete ({controller_output.strip() or 'no detail'})."
                        )
            except Exception as exc:  # noqa: BLE001 - persisted status remains useful
                diagnostics.append(
                    "SkyPilot controller status is unavailable; showing the last "
                    f"persisted manifest state. Diagnostic: {exc}"
                )
        reconcile_submitted_manifest(
            run_manifest, live_status=live_status, task_rows=task_rows
        )
        run_payload = build_actionable_run_status(
            run_manifest,
            live_status=live_status,
            task_rows=task_rows,
            controller_output=controller_output,
            project=project or state.project,
            failure_threshold=startup_failure_threshold,
        )
        run_payload.update(
            {
                "run_id": run_manifest.run_id or _display_run_id(run_id),
                "run_prefix_uri": run_manifest.run_prefix_uri
                or state.uri.removesuffix("/npa-workflow"),
                "manifest_uri": f"{state.uri.rstrip('/')}/manifest.json",
                "manifest_state": "available",
                "manifest_pending": False,
                "resolution_source": resolution.source,
                "resolution_checks": resolution.checks_payload(),
                "verification": "found",
            }
        )
        if diagnostics:
            run_payload["diagnostics"] = diagnostics
        blockers = _stalled_job_blockers(
            run_manifest.sky_job_id, live_status, sky_bin=sky_bin
        )
        if blockers:
            run_payload["blockers"] = blockers
        return run_payload
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
            live = workflow_status(job_id, sky_bin=sky_bin or None)
            live_status = live.status if not live.error else ""
        except Exception:
            live_status = ""
    status = _aggregate_stage_status(stages, live_status)
    payload: dict[str, object] = {
        "run_id": manifest.get("run_id") or _display_run_id(run_id),
        "workflow_name": manifest.get("workflow_name", ""),
        "status": status,
        "live_status": live_status,
        "sky_job_id": job_id,
        "run_prefix_uri": manifest.get("run_prefix_uri") or state.uri,
        "manifest_uri": f"{state.uri.rstrip('/')}/manifest.json",
        "manifest_state": "available",
        "manifest_pending": False,
        "resolution_source": resolution.source,
        "resolution_checks": resolution.checks_payload(),
        "verification": "found",
        "stages": stages,
    }
    blockers = _stalled_job_blockers(job_id, live_status, sky_bin=sky_bin)
    if blockers:
        payload["blockers"] = blockers
    return payload


def _manifest_pending_status(
    resolution,
    *,
    project: str,
    sky_bin: str,
    startup_failure_threshold: int,
) -> dict[str, object]:
    """Project receipt/S3/Sky evidence through the shared actionable model."""

    from npa.orchestration.npa_workflow.run_resolution import resolution_diagnostics
    from npa.orchestration.npa_workflow.run_state import (
        RunManifest,
        build_actionable_run_status,
        reconcile_submitted_manifest,
    )
    from npa.orchestration.skypilot.workflow import (
        workflow_controller_logs,
        workflow_status,
        workflow_task_statuses,
    )

    receipt_workflow = resolution.receipt.get("workflow")
    workflow_record = receipt_workflow if isinstance(receipt_workflow, dict) else {}
    steps = [
        dict(item)
        for item in workflow_record.get("steps") or []
        if isinstance(item, dict)
    ]
    runtime_waves = [
        dict(item)
        for item in resolution.runtime_state.get("waves") or []
        if isinstance(item, dict)
    ]
    active_wave = next(
        (
            item
            for item in reversed(runtime_waves)
            if str(item.get("status") or "").lower()
            not in {"succeeded", "failed", "cancelled"}
        ),
        runtime_waves[-1] if runtime_waves else {},
    )
    if not steps and runtime_waves:
        for wave in runtime_waves:
            wave_status = str(wave.get("status") or "").lower()
            step_status = {
                "succeeded": "SUCCEEDED",
                "failed": "FAILED",
                "cancelled": "CANCELLED",
            }.get(wave_status, "submitted")
            for state_name in wave.get("states") or []:
                steps.append(
                    {
                        "state": str(state_name),
                        "status": step_status,
                        "resources_profile": {},
                    }
                )
    job_id = resolution.job_id
    live_status = ""
    task_rows: list[dict[str, object]] = []
    controller_output = ""
    diagnostics = resolution_diagnostics(resolution)
    if resolution.managed_job is not None and resolution.managed_job.outcome == "found":
        live_status = resolution.managed_job.status
        task_rows = [dict(item) for item in resolution.managed_job.task_rows]
    elif job_id:
        try:
            live = workflow_status(job_id, sky_bin=sky_bin or None)
            if live.error:
                diagnostics.append(
                    "SkyPilot status is unavailable; the run remains receipt-proven "
                    f"({live.error})."
                )
            else:
                live_status = live.status
            task_rows = workflow_task_statuses(job_id, sky_bin=sky_bin or None)
        except Exception as exc:  # noqa: BLE001 - durable evidence still proves the run
            diagnostics.append(
                "SkyPilot status is unavailable; the run remains receipt-proven. "
                f"Diagnostic: {exc}"
            )
    if not task_rows and active_wave:
        task_rows = [
            dict(item) for item in active_wave.get("tasks") or [] if isinstance(item, dict)
        ]
    if not steps and task_rows:
        rows = sorted(task_rows, key=lambda item: int(item.get("task_id") or 0))
        steps = [
            {
                "state": str(row.get("task_name") or f"step-{index}"),
                "status": "submitted",
                "resources_profile": {},
            }
            for index, row in enumerate(rows)
        ]
    elif steps and task_rows and active_wave:
        indexes_by_name = {
            str(step.get("state") or ""): index for index, step in enumerate(steps)
        }
        for row in task_rows:
            task_name = str(row.get("task_name") or "")
            if task_name in indexes_by_name:
                row["task_id"] = indexes_by_name[task_name]
    pending_manifest = RunManifest(
        workflow=resolution.workflow_name or "physical-ai-data-factory",
        run_id=resolution.run_id,
        api_version=str(workflow_record.get("api_version") or "npa.workflow/v0.0.1"),
        run_prefix_uri=resolution.run_prefix_uri,
        status="submitted",
        sky_job_id=job_id,
        steps=steps,
        updated_at=str(resolution.receipt.get("updated_at") or ""),
    )
    if job_id and str(live_status).upper() not in {
        "SUCCEEDED",
        "CANCELLED",
    } and not str(live_status).upper().startswith("FAILED"):
        try:
            controller_logs = workflow_controller_logs(
                job_id, sky_bin=sky_bin or None
            )
            controller_output = "\n".join(
                item for item in (controller_logs.stdout, controller_logs.stderr) if item
            )
            if controller_logs.returncode != 0:
                diagnostics.append(
                    "SkyPilot controller logs are unavailable; startup failure "
                    f"classification is incomplete ({controller_output.strip() or 'no detail'})."
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(f"SkyPilot controller logs are unavailable: {exc}")
    reconcile_submitted_manifest(
        pending_manifest, live_status=live_status, task_rows=task_rows
    )
    payload = build_actionable_run_status(
        pending_manifest,
        live_status=live_status,
        task_rows=task_rows,
        controller_output=controller_output,
        project=project,
        failure_threshold=startup_failure_threshold,
    )
    if not job_id and not task_rows:
        payload["status"] = "MANIFEST_PENDING"
    payload.update(
        {
            "run_prefix_uri": resolution.run_prefix_uri,
            "manifest_uri": resolution.manifest_uri,
            "manifest_state": "pending",
            "manifest_pending": True,
            "resolution_source": resolution.source,
            "resolution_checks": resolution.checks_payload(),
            "verification": "found",
            "diagnostics": [
                "Run found; final workflow manifest is pending.",
                *diagnostics,
            ],
        }
    )
    blockers = _stalled_job_blockers(job_id, live_status, sky_bin=sky_bin)
    if blockers:
        payload["blockers"] = blockers
    return payload


def _stalled_job_blockers(
    job_id: str, live_status: str, *, sky_bin: str = ""
) -> list[dict[str, str]]:
    """Explain a managed job that is not progressing, from its own pods.

    A job whose pod cannot start never becomes FAILED -- Kubernetes retries image
    pulls and scheduling forever -- so the only way to tell a slow start from a
    dead one is to ask the pods.
    """

    if not job_id or live_status.upper() not in {"PENDING", "STARTING"}:
        return []
    from npa.orchestration.skypilot.job_blockers import inspect_job_blockers
    from npa.orchestration.skypilot.workflow import workflow_task_statuses

    try:
        rows = workflow_task_statuses(job_id, sky_bin=_resolve_sky_bin(sky_bin))
    except Exception:
        rows = []
    clusters = sorted(
        {str(row.get("cluster_name") or "").strip() for row in rows} - {""}
    )
    # `sky jobs queue` reports a null cluster for a job that never provisioned --
    # exactly the case worth diagnosing -- so fall back to the job id.
    reports = (
        [inspect_job_blockers(job_id=job_id, cluster_name=cluster) for cluster in clusters]
        if clusters
        else [inspect_job_blockers(job_id=job_id)]
    )
    reported: list[dict[str, str]] = []
    for report in reports:
        for blocker in report.blockers:
            reported.append(
                {
                    "pod": blocker.pod,
                    "reason": blocker.reason,
                    "message": blocker.message,
                    "remedy": report.remedy(),
                }
            )
        # A job whose nodes were reclaimed has no pod-level reason at all.
        for node in report.unready_nodes:
            reported.append(
                {
                    "pod": node,
                    "reason": "NodeNotReady",
                    "message": "the node this job needs is not Ready",
                    "remedy": report.remedy(),
                }
            )
    return reported


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
    return (
        normalized == "SUCCEEDED"
        or normalized.startswith("FAILED")
        or normalized
        in {"CANCELLED", "BLOCKED", "NOT_SUBMITTED", "NOT_FOUND", "VERIFICATION_UNAVAILABLE"}
    )


def _emit_workflow_status(result: dict[str, object], output_format: OutputFormat) -> None:
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    typer.echo(f"run_id: {result.get('run_id')}")
    typer.echo(f"status: {result.get('status')}")
    if result.get("resolution_source"):
        typer.echo(f"resolution_source: {result.get('resolution_source')}")
    if result.get("manifest_state"):
        typer.echo(f"manifest_state: {result.get('manifest_state')}")
    if result.get("submission_state"):
        typer.echo(f"submission_state: {result.get('submission_state')}")
    if result.get("current_stage"):
        typer.echo(f"current_stage: {result.get('current_stage')}")
    if result.get("k8s_job"):
        typer.echo(f"k8s_job: {result.get('k8s_job')}")
    if result.get("pod_reason"):
        typer.echo(f"pod_reason: {result.get('pod_reason')}")
    if result.get("sky_job_id"):
        typer.echo(f"sky_job_id: {result.get('sky_job_id')}")
    if result.get("active_stage_name"):
        typer.echo(
            f"active_stage: {result.get('active_stage_index')} "
            f"{result.get('active_stage_name')}"
        )
    if result.get("last_heartbeat_at"):
        typer.echo(f"last_heartbeat_at: {result.get('last_heartbeat_at')}")
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            typer.echo(f"diagnostic: {diagnostic}")
    checks = result.get("resolution_checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            detail = f" ({check.get('detail')})" if check.get("detail") else ""
            typer.echo(
                f"checked: {check.get('source')}: {check.get('outcome')}{detail}"
            )
    blockers = result.get("blockers")
    if isinstance(blockers, list) and blockers:
        # A managed job whose pod cannot start never becomes FAILED, so say why
        # it is stuck rather than letting PENDING look like slow progress.
        typer.echo(f"blocked: {len(blockers)} pod(s) cannot start")
        for blocker in blockers:
            if not isinstance(blocker, dict):
                continue
            detail = f"  {blocker.get('pod')}: {blocker.get('reason')}"
            if blocker.get("message"):
                detail = f"{detail} - {blocker.get('message')}"
            typer.echo(detail)
        remedy = str((blockers[0] or {}).get("remedy") or "")
        if remedy:
            typer.echo(f"  Suggested action: {remedy}")
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
            accelerator = info.get("requested_accelerators", "") if isinstance(info, dict) else ""
            suffix_parts = [part for part in (tier, f"requested={accelerator}" if accelerator else "") if part]
            suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
            typer.echo(f"{stage}: {state}{suffix}")
            if isinstance(info, dict) and "scheduler_state" in info:
                typer.echo(
                    f"  scheduler={info.get('scheduler_state') or 'UNKNOWN'} "
                    f"task_id={info.get('task_id')} retries={info.get('retry_count', 0)} "
                    f"last_progress={info.get('last_progress_at') or 'unknown'} "
                    f"staleness_seconds={info.get('staleness_seconds')}"
                )
                if info.get("last_normalized_startup_failure"):
                    typer.echo(
                        f"  startup_failure={info.get('last_normalized_startup_failure')} "
                        f"evidence={info.get('startup_failure_evidence')}"
                    )
                typer.echo(f"  logs: {info.get('log_command')}")
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
        help=(
            "Exact workflow prefix; takes precedence over receipts and discovery, "
            "for example s3://bucket/physical-ai-data-factory/RUN/npa-workflow."
        ),
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
    startup_failure_threshold: int = typer.Option(
        3,
        "--startup-failure-threshold",
        help="Identical deterministic controller failures required for FAILED_STARTUP.",
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

    # An exact S3 URI or project alias selects the durable resolver. Do not probe
    # the legacy sim2real prefix first: that probe builds a separate S3 client
    # from ambient env vars and can fail before the selected project's configured
    # credentials ever reach the durable reader.
    exact_durable_uri = bool(run_id.startswith("s3://") or workflow_s3_uri or project)
    prefix = workflow_s3_prefix or "sim2real-b"
    if not exact_durable_uri and sim2real_run_exists(
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
                    startup_failure_threshold=startup_failure_threshold,
                )
                _emit_workflow_status(result, OutputFormat.json if json_output else output_format)
                normalized = str(result.get("status") or "").upper()
                if normalized == "NOT_FOUND":
                    raise typer.Exit(code=1)
                if normalized == "VERIFICATION_UNAVAILABLE":
                    raise typer.Exit(code=2)
                if not watch or _workflow_status_is_terminal(str(result.get("status", ""))):
                    return
                time.sleep(interval)
        except typer.Exit:
            raise
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
        help=(
            "Exact workflow prefix; takes precedence over receipts and discovery, "
            "for example s3://bucket/physical-ai-data-factory/RUN/npa-workflow."
        ),
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
    try:
        from npa.orchestration.npa_workflow.run_resolution import validate_run_id

        validate_run_id(_display_run_id(run_id))
        if selected_stage:
            validate_run_id(selected_stage)
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
            from npa.orchestration.npa_workflow.run_resolution import (
                require_resolved_run,
                resolve_run,
            )

            resolution = require_resolved_run(resolve_run(
                run_id,
                project=project,
                workflow_s3_uri=workflow_s3_uri,
                workflow_s3_prefix=workflow_s3_prefix,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
                sky_bin=sky_bin,
            ))
            state = resolution.state
            from npa.orchestration.skypilot.workflow_state import (
                read_stage_log,
                tail_live_job_logs,
            )

            manifest = resolution.manifest
            if manifest is None:
                pending = _manifest_pending_status(
                    resolution,
                    project=project,
                    sky_bin=sky_bin,
                    startup_failure_threshold=3,
                )
                if not selected_stage:
                    selected_stage = str(pending.get("active_stage_name") or "")
                job_id = str(pending.get("sky_job_id") or resolution.job_id)
                if not job_id:
                    raise RuntimeError(
                        f"run {resolution.run_id!r} was found via {resolution.source}, "
                        "but its manifest is pending and no exact managed-job identity "
                        "is available for logs"
                    )
                live = tail_live_job_logs(
                    sky_bin=_resolve_sky_bin(sky_bin),
                    job_id=job_id,
                    stage=selected_stage,
                    follow=follow,
                    timeout=86400 if follow else 300,
                )
                if live.stdout:
                    typer.echo(live.stdout, nl=False)
                if live.stderr:
                    typer.echo(live.stderr, err=True, nl=False)
                if live.returncode != 0:
                    raise RuntimeError(
                        "run found with manifest pending, but SkyPilot logs are unavailable"
                    )
                return
            assert state is not None
            if manifest.get("schema_version") == "npa.workflow.run.v1":
                steps = [
                    item for item in (manifest.get("steps") or []) if isinstance(item, dict)
                ]
                available = [str(item.get("state") or "") for item in steps]
                if not selected_stage:
                    selected_stage = next(
                        (
                            str(item.get("state") or "")
                            for item in steps
                            if str(item.get("status") or "").upper().startswith("FAILED")
                        ),
                        available[0] if available else "",
                    )
                if selected_stage not in available:
                    raise RuntimeError(
                        f"stage {selected_stage!r} is not in the run manifest; "
                        f"available stages: {', '.join(available) or '<none>'}"
                    )
                job_id = str(manifest.get("sky_job_id") or "")
                if not job_id:
                    raise RuntimeError(
                        "the run manifest has no SkyPilot job id, so live logs cannot "
                        "be queried; re-submit with the current NPA CLI"
                    )
                live = tail_live_job_logs(
                    sky_bin=_resolve_sky_bin(sky_bin),
                    job_id=job_id,
                    stage=selected_stage,
                    follow=follow,
                    timeout=86400 if follow else 300,
                )
                if live.stdout:
                    typer.echo(live.stdout, nl=False)
                if live.stderr:
                    typer.echo(live.stderr, err=True, nl=False)
                if live.returncode != 0:
                    raise RuntimeError(
                        "SkyPilot logs are unavailable from the controller. Verify "
                        f"`npa skypilot status`, then retry; manifest: {state.uri}/manifest.json"
                    )
                return
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
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact workflow prefix; takes precedence over receipts and discovery.",
    ),
    workflow_s3_prefix: str = typer.Option("", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    sky_bin: str = typer.Option("", "--sky-bin", help="Pinned SkyPilot executable path."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List durable S3 artifact URIs for a workflow run."""
    resolution = None
    try:
        from npa.orchestration.npa_workflow.run_resolution import (
            list_resolved_artifacts,
            resolution_diagnostics,
            resolve_run,
        )

        resolution = resolve_run(
            run_id,
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            workflow_s3_prefix=workflow_s3_prefix,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
            sky_bin=sky_bin,
        )
        if not resolution.found:
            if json_output:
                typer.echo(
                    json.dumps(
                        {
                            "run_id": resolution.run_id,
                            "status": (
                                "NOT_FOUND"
                                if resolution.conclusively_absent
                                else "VERIFICATION_UNAVAILABLE"
                            ),
                            "verification": (
                                "conclusively_absent"
                                if resolution.conclusively_absent
                                else "unavailable"
                            ),
                            "resolution_checks": resolution.checks_payload(),
                            "artifacts": [],
                            "diagnostics": resolution_diagnostics(resolution),
                        },
                        indent=2,
                    )
                )
                raise typer.Exit(code=1 if resolution.conclusively_absent else 2)
            raise RuntimeError("\n".join(resolution_diagnostics(resolution)))
        artifacts = list_resolved_artifacts(resolution, stage=stage)
    except typer.Exit:
        raise
    except Exception as exc:
        if json_output and resolution is not None:
            typer.echo(
                json.dumps(
                    {
                        "run_id": resolution.run_id,
                        "status": "VERIFICATION_UNAVAILABLE",
                        "verification": "unavailable",
                        "resolution_source": resolution.source,
                        "manifest_pending": resolution.manifest_pending,
                        "resolution_checks": resolution.checks_payload(),
                        "artifacts": [],
                        "diagnostics": [str(exc)],
                    },
                    indent=2,
                )
            )
            raise typer.Exit(code=2)
        _fail(str(exc))
        return
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_id": resolution.run_id,
                    "resolution_source": resolution.source,
                    "manifest_pending": resolution.manifest_pending,
                    "resolution_checks": resolution.checks_payload(),
                    "artifacts": artifacts,
                },
                indent=2,
            )
        )
        return
    for uri in artifacts:
        typer.echo(uri)


@app.command("load-artifact")
def load_artifact_cmd(
    run_id: str = typer.Argument(help="Successful PAIDF run ID or exact s3:// run prefix."),
    project: str = typer.Option("", "--project", "-p", help="Configured project alias."),
    agent_name: str = typer.Option("", "--agent-name", help="Configured agent name."),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact npa-workflow prefix; takes precedence over receipts and discovery.",
    ),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Retry only the final artifact load; never relaunch workflow stages."""

    try:
        status = _durable_workflow_status(
            run_id,
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            s3_endpoint=s3_endpoint,
        )
        if str(status.get("status") or "").upper() != "SUCCEEDED":
            raise RuntimeError(
                "artifact auto-load is available after workflow success; current status is "
                f"{status.get('status') or 'UNKNOWN'}"
            )
        result = _load_paidf_artifact(
            project=project,
            run_id=str(status.get("run_id") or _display_run_id(run_id)),
            run_prefix_uri=str(status.get("run_prefix_uri") or ""),
            s3_endpoint=s3_endpoint,
            agent_name=agent_name,
        )
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {result.get('status')}")
        typer.echo(f"artifact_uri: {result.get('artifact_uri') or ''}")
        typer.echo(f"verified: {str(bool(result.get('verified'))).lower()}")
        if result.get("detail"):
            typer.echo(f"detail: {result.get('detail')}")
        if result.get("retry_command") and not result.get("verified"):
            typer.echo(f"retry: {result.get('retry_command')}")


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
        if project and not workflow_s3_uri and not workflow_s3_prefix and not s3_bucket:
            # Project checkpoint prefixes scope agent artifacts, but declarative
            # workflows such as PAIDF intentionally write at a product prefix in
            # the same bucket. A project-level list therefore scans the bucket
            # root and relies on durable-manifest validation to exclude all
            # component and source-staging manifests.
            parent_state = type(parent_state)(
                bucket=parent_state.bucket,
                prefix="",
                endpoint_url=parent_state.endpoint_url,
                aws_access_key_id=parent_state.aws_access_key_id,
                aws_secret_access_key=parent_state.aws_secret_access_key,
                project=parent_state.project,
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
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact workflow prefix; takes precedence over receipts and discovery.",
    ),
    workflow_s3_prefix: str = typer.Option("", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint."),
    sky_bin: str = typer.Option("", "--sky-bin", help="SkyPilot executable path."),
    cluster: str = typer.Option("", "--cluster", help="SkyPilot cluster name to tear down. Defaults to run ID."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Cancel a launched run; never-launched and terminal runs are repeat-safe no-ops."""
    try:
        status = _durable_workflow_status(
            run_id,
            project=project,
            workflow_s3_uri=workflow_s3_uri,
            workflow_s3_prefix=workflow_s3_prefix,
            s3_bucket=s3_bucket,
            s3_endpoint=s3_endpoint,
            sky_bin=sky_bin,
        )
        normalized = str(status.get("status") or "").upper()
        resolved_run_id = str(status.get("run_id") or _display_run_id(run_id))
        job_id = str(status.get("sky_job_id") or "")
        if normalized == "NOT_FOUND":
            result = {
                "run_id": resolved_run_id,
                "outcome": "already_absent",
                "launch_state": "not_launched",
                "sky_job_id": "",
                "cloud_calls": False,
                "verification": "conclusively_absent",
                "resolution_checks": status.get("resolution_checks", []),
                "message": "Run not found after every applicable exact source was checked; no cancellation was issued.",
            }
        elif normalized == "VERIFICATION_UNAVAILABLE":
            result = {
                "run_id": resolved_run_id,
                "outcome": "verification_failed",
                "status": normalized,
                "sky_job_id": "",
                "cloud_calls": False,
                "verification": "unavailable",
                "resolution_checks": status.get("resolution_checks", []),
                "message": "Run verification is unavailable; provider/auth failure is not absence, so cancellation was not attempted.",
            }
        elif _workflow_status_is_terminal(normalized):
            result = {
                "run_id": resolved_run_id,
                "outcome": "terminal",
                "status": normalized,
                "sky_job_id": job_id,
                "cloud_calls": False,
                "message": "The launched workflow is already terminal; no cancellation was issued.",
            }
        elif not job_id:
            result = {
                "run_id": resolved_run_id,
                "outcome": "verification_failed",
                "status": normalized or "UNKNOWN",
                "sky_job_id": "",
                "cloud_calls": False,
                "message": "Durable state does not contain a manifest-proven SkyPilot job ID.",
            }
        else:
            from npa.orchestration.skypilot.cleanup import cleanup_launched_workflow

            cleanup = cleanup_launched_workflow(
                job_id,
                resolved_run_id,
                cluster=cluster,
                sky_bin=sky_bin or None,
            )
            result = {
                "run_id": resolved_run_id,
                "outcome": "cancelled" if cleanup.ok else "verification_failed",
                "status": normalized or "SUBMITTED",
                "sky_job_id": job_id,
                "cloud_calls": True,
                "resources_removed": cleanup.resources_removed,
                "commands": cleanup.commands,
                "errors": cleanup.errors,
            }
    except Exception as exc:
        result = {
            "run_id": _display_run_id(run_id),
            "outcome": "verification_failed",
            "cloud_calls": False,
            "message": str(exc),
        }
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"run_id: {result['run_id']}")
        typer.echo(f"outcome: {result['outcome']}")
        if result.get("message"):
            typer.echo(str(result["message"]))
        for error in result.get("errors", []):
            typer.echo(f"cleanup warning: {error}", err=True)
    if result["outcome"] == "verification_failed":
        raise typer.Exit(code=2)


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


@app.command("stage-src")
def stage_src_cmd(
    bucket: str = typer.Option(
        "",
        "--bucket",
        help="Destination bucket (name or s3://name) for the npa package copy.",
    ),
    prefix: str = typer.Option(
        "",
        "--prefix",
        help="Key prefix inside the bucket. Defaults to npa-src/npa.",
    ),
    endpoint: str = typer.Option(
        "",
        "--endpoint",
        help="S3-compatible endpoint. Defaults to AWS_ENDPOINT_URL / NEBIUS_S3_ENDPOINT.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias whose durable source setting should be updated.",
    ),
    run_id: str = typer.Option(
        "staged-source",
        "--run-id",
        help="Submission ledger key used to make concurrent staging restart-safe.",
    ),
) -> None:
    """Upload the local npa package to S3 for image-less workflow steps.

    Workflow tasks that run on SkyPilot's default image (Token Factory tools and
    `run.shell` states) install `npa` by syncing `$NPA_SRC_S3_URI` on the worker.
    This publishes that copy and prints the export line to use, so a first submit
    does not dead-end on "planned step ... has no workbench image and
    NPA_SRC_S3_URI is unset".
    """
    from npa.orchestration.npa_workflow.src_staging import DEFAULT_SRC_PREFIX

    target = bucket.strip()
    if not target:
        from npa.clients.credentials import load_credentials

        target = str(load_credentials().s3_bucket or "").strip()
    if not target:
        _fail(
            "No bucket to stage into. Pass --bucket <your-bucket> or run "
            "`npa configure` so storage.bucket is set."
        )
        return

    if prefix and prefix != DEFAULT_SRC_PREFIX:
        # The shared helper intentionally uses the canonical prefix; preserve an
        # explicit custom prefix for advanced callers while still persisting it.
        from npa.clients.config import ConfigError, persist_workflow_src_s3_uri
        from npa.orchestration.npa_workflow.src_staging import (
            SrcStagingError,
            stage_npa_source,
        )
        from npa.orchestration.npa_workflow.submission_state import update_submission_state

        try:
            uri = stage_npa_source(
                bucket=target,
                prefix=prefix,
                endpoint_url=endpoint,
                on_status=lambda message: typer.echo(f"  {message}", err=True),
            )
            persist_workflow_src_s3_uri(uri, project or None)
            update_submission_state(
                project or "default",
                run_id,
                {"source": {"status": "verified", "uri": uri}},
            )
        except (ConfigError, SrcStagingError) as exc:
            _fail(str(exc))
            return
    else:
        uri = _stage_npa_src_for_submit(
            {"bucket": target},
            s3_bucket=target,
            s3_endpoint=endpoint,
            project=project,
            run_id=run_id,
        )
    if not uri:
        return

    typer.echo(f"npa_src_s3_uri: {uri}")
    typer.echo(f"export NPA_SRC_S3_URI={uri}")


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
    var: list[str] = typer.Option(
        [],
        "--var",
        help=(
            "Config override as KEY=VALUE, merged into the spec's config "
            "(same as `submit --var`). Without `--var bucket=<your-bucket>` the "
            "plan uses the spec's `example-bucket` placeholder."
        ),
    ),
    waves: bool = typer.Option(
        False,
        "--waves",
        help=(
            "Show the runtime wave shape (serial steps and parallel fan-out groups "
            "with their concurrency batches) instead of the flat step list."
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON plan."),
) -> None:
    """Expand an NPA workflow spec into an execution plan (dry-run)."""

    from npa.orchestration.npa_workflow import NpaWorkflowError, build_plan
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = _load_npa_workflow(yaml_path)
    spec = merge_config_overrides(spec, _parse_submit_vars(var))
    _warn_placeholder_bucket(spec.config, quiet=json_output)
    resolved_run_id = run_id or f"{spec.name}-plan"
    try:
        plan = build_plan(spec, run_id=resolved_run_id, assume_decision=assume_decision)
    except NpaWorkflowError as exc:
        _fail(str(exc))
        return

    if waves:
        from npa.orchestration.npa_workflow.waves import wave_plan_from_plan

        wave_plan = wave_plan_from_plan(spec, plan, run_id=resolved_run_id)
        if json_output:
            typer.echo(json.dumps(wave_plan.to_dict(), indent=2, sort_keys=True))
            return
        typer.echo(f"workflow: {wave_plan.workflow}")
        typer.echo(f"waves: {len(wave_plan.waves)}")
        for wave in wave_plan.waves:
            states = ", ".join(step.state for step in wave.steps)
            suffix = (
                f" maxConcurrency={wave.max_concurrency} batches={len(wave.batches())}"
                if wave.kind == "parallel"
                else ""
            )
            typer.echo(f"  {wave.index:02d}. [{wave.kind}] {wave.name}: {states}{suffix}")
        return

    if json_output:
        payload = plan.to_dict()
        # The human warning is suppressed under --json to keep the document clean,
        # which made a placeholder plan look valid. Say it in the document instead.
        if _is_placeholder_bucket(str(spec.config.get("bucket", "") or "")):
            payload["bucket_is_placeholder"] = True
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"workflow: {plan.workflow}")
    if plan.assume_decision:
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
    var: list[str] = typer.Option(
        [],
        "--var",
        help=(
            "Config override as KEY=VALUE, merged into the spec's config "
            "(same as `submit --var`). Without `--var bucket=<your-bucket>` the "
            "run uses the spec's `example-bucket` placeholder."
        ),
    ),
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
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = _load_npa_workflow(yaml_path)
    spec = merge_config_overrides(spec, _parse_submit_vars(var))
    _warn_placeholder_bucket(spec.config, quiet=json_output)
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
        if _is_placeholder_bucket(str(spec.config.get("bucket", "") or "")):
            report["bucket_is_placeholder"] = True
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"status: {report['status']}")
        typer.echo(f"run_id: {report['run_id']}")
        if report.get("run_prefix_uri"):
            typer.echo(f"run_prefix_uri: {report['run_prefix_uri']}")
        typer.echo(f"steps: {len(report['plan']['steps'])}")


@app.command("preflight-images")
def preflight_images_cmd(
    yaml_path: Path = typer.Argument(help="npa.workflow spec path."),
    registry: str = typer.Option("", "--registry", help="Container registry override."),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias whose configured registry to check. Defaults to the configured project.",
    ),
    image: str = typer.Option("", "--image", help="Pin every step to this image."),
    assume_decision: str = typer.Option(
        "", "--assume-decision", help="Branch assumption for planning."
    ),
    gpu_target: str = typer.Option("", "--gpu-target", help="SONIC GPU target."),
    image_variant: str = typer.Option("", "--image-variant", help="SONIC image variant."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON report."),
) -> None:
    """Prove every image this spec pulls is pullable, with the run's own credentials.

    Kubernetes retries image pulls forever, so a registry that answers 403 leaves the
    job in PENDING/ImagePullBackOff rather than failing. Being able to list a
    repository's tags is a different permission from pulling it, so this reproduces
    the actual manifest fetch a worker performs.
    """

    from npa.orchestration.npa_workflow import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        plan_images,
    )
    from npa.orchestration.skypilot.registry_preflight import check_image_pulls_with_credentials

    spec = _load_npa_workflow(yaml_path)
    image_overrides: dict[str, str] = {}
    if image.strip():
        image_overrides["*"] = image.strip()
    resolved_registry = _resolve_submit_registry(registry, project)
    options = SkypilotRenderOptions(
        registry=resolved_registry,
        image_overrides=image_overrides,
        gpu_target=gpu_target,
        image_variant=image_variant,
        materialize_registry_secrets=False,
    )
    if resolved_registry:
        typer.echo(f"registry: {resolved_registry}", err=True)
    run_id = f"{spec.name}-preflight"
    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    images = plan_images(spec, plan.steps, run_id=run_id, options=options)
    if not images:
        typer.echo("images: none pinned by this spec")
        return

    checks = check_image_pulls_with_credentials(images, mint=True)
    if json_output:
        typer.echo(
            json.dumps(
                [
                    {
                        "image": check.image,
                        "status": check.status,
                        "http_status": check.http_status,
                        "detail": check.detail,
                        "remedy": check.remedy,
                    }
                    for check in checks
                ],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in checks:
            typer.echo(check.render())
    failed = [check for check in checks if not check.ok]
    if failed:
        _fail(
            f"{len(failed)} of {len(checks)} image(s) cannot be pulled with this run's credentials"
        )


@app.command("gpus")
def gpus_cmd(
    cluster: str = typer.Option(
        "",
        "--cluster",
        "--cluster-name",
        help="NPA cluster name. Resolves ~/.npa/clusters/<name>/kubeconfig.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        help="Kubernetes context to inspect. Defaults to KUBECONTEXT or every context.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.",
    ),
    spec: Path | None = typer.Option(
        None,
        "--spec",
        help="npa.workflow spec whose accelerators should be resolved against the cluster.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON report."),
) -> None:
    """Print the accelerator names this cluster advertises to SkyPilot.

    Kubernetes clusters name GPUs after their node labels, so the string a spec
    must use is discovered rather than guessed. Run this once after `npa configure`
    and export the printed NPA_WORKFLOW_GPU_ACCELERATOR line.
    """

    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        KubernetesGpuCatalogError,
        UnsatisfiableAcceleratorError,
        discover_kubernetes_gpu_catalog,
        resolve_kubernetes_accelerator,
        spec_accelerators,
    )

    resolved_context = context.strip() or os.environ.get("KUBECONTEXT", "").strip()
    env_backup: str | None = None
    if cluster.strip():
        from npa.cluster.state import kubeconfig_file

        kubeconfig = kubeconfig_file(cluster.strip())
        if not kubeconfig.exists():
            _fail(f"Kubeconfig not found for cluster {cluster!r}: {kubeconfig}")
            return
        env_backup = os.environ.get("KUBECONFIG")
        os.environ["KUBECONFIG"] = str(kubeconfig)
    try:
        catalog = discover_kubernetes_gpu_catalog(
            context=resolved_context, sky_bin=sky_bin or None
        )
    except KubernetesGpuCatalogError as exc:
        _fail(str(exc))
        return
    finally:
        if cluster.strip():
            if env_backup is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = env_backup

    resolutions: list[dict[str, object]] = []
    if spec is not None:
        for accelerator in spec_accelerators(_load_npa_workflow(spec).resources):
            try:
                resolution = resolve_kubernetes_accelerator(accelerator, catalog=catalog)
            except UnsatisfiableAcceleratorError as exc:
                resolutions.append({"requested": accelerator, "error": str(exc)})
                continue
            resolutions.append(
                {
                    "requested": resolution.requested,
                    "resolved": resolution.resolved,
                    "remapped": resolution.remapped,
                }
            )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "context": catalog.context,
                    "accelerators": {
                        name: sorted(quantities)
                        for name, quantities in catalog.quantities_by_accelerator.items()
                    },
                    "spec_resolutions": resolutions,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if catalog.is_empty:
        typer.echo("accelerators: none advertised")
        return
    typer.echo(f"context: {catalog.context or 'all'}")
    for name in sorted(catalog.quantities_by_accelerator, key=str.casefold):
        quantities = sorted(catalog.quantities_by_accelerator[name])
        offered = ", ".join(str(value) for value in quantities)
        typer.echo(f"  {name}: requestable per node {offered}")
        typer.echo(f"    export NPA_WORKFLOW_GPU_ACCELERATOR={name}:{quantities[0]}")
    for item in resolutions:
        if item.get("error"):
            typer.echo(f"  {item['requested']}: {item['error']}")
        else:
            typer.echo(f"  {item['requested']} -> {item['resolved']}")


app.add_typer(trigger_app, name="trigger")
