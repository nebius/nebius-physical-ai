"""npa workbench workflow — orchestrate multi-stage training workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import yaml
from rich.console import Console
from rich.text import Text

from npa.cli.workbench.trigger import app as trigger_app
from npa.orchestration.npa_workflow.spec import load_spec
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

if TYPE_CHECKING:
    from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions

app = typer.Typer(
    name="workflow",
    help="Multi-stage training workflow orchestration.",
    no_args_is_help=True,
)

console = Console(stderr=True)
logger = logging.getLogger(__name__)
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
    # Operational recovery commands and status phrases must remain copyable and
    # machine-observable even when Rich detects a narrow non-interactive console.
    error = Text("Error:", style="red")
    error.append(" ")
    # Exception messages can legitimately contain bracketed values (for example,
    # a malformed URI such as ``[/foo]``).  Keep them literal so Rich does not
    # replace the original failure with a MarkupError while reporting it.
    error.append(str(msg))
    console.print(error, soft_wrap=True)
    raise typer.Exit(code)


# ``docker:cr.<region>.nebius.cloud/<registry-id>/<image>:<tag>`` in a rendered plan.
_NEBIUS_IMAGE_RE = re.compile(
    r"image_id:\s*docker:(cr\.[a-z0-9-]+\.nebius\.cloud)/", re.IGNORECASE
)


def nebius_registry_hosts(rendered_yaml: str) -> list[str]:
    """Distinct Nebius registry hosts a rendered plan pulls images from."""

    return sorted(
        {match.group(1).lower() for match in _NEBIUS_IMAGE_RE.finditer(rendered_yaml)}
    )


def _refresh_kubernetes_pull_secrets(
    rendered_path: Path, *, k8s_context: str = "", kubeconfig: str = ""
) -> None:
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

    from npa.workflows.sim2real.registry_auth import (
        ensure_nebius_registry_pull_secret,
        mint_nebius_registry_token,
    )

    joined = ", ".join(hosts)
    # One call with every host: the secret holds a single dockerconfigjson and each
    # apply replaces it, so refreshing host by host would leave only the last one.
    # Do not consult SKYPILOT_DOCKER_PASSWORD/NPA_REGISTRY_PASSWORD here. Those are
    # valid render/preflight overrides, but can be the short-lived token installed at
    # the start of a long runtime loop. "Refresh" must mint a genuinely new,
    # profile-scoped Nebius credential; callers with an independently managed secret
    # explicitly select --no-refresh-registry-secret.
    try:
        username = "iam"
        password = mint_nebius_registry_token()
        if not password:
            raise RuntimeError("no registry credential could be resolved")
        ensure_nebius_registry_pull_secret(
            registry_servers=hosts,
            username=username,
            token=password,
            kubeconfig=kubeconfig,
            k8s_context=k8s_context,
        )
    except Exception as exc:
        raise RuntimeError(
            "could not install the Kubernetes imagePullSecret for "
            f"{joined}: {exc}. Fix kubectl access/registry credentials before submit; "
            "otherwise every private-image task will fail with ImagePullBackOff."
        ) from exc
    console.print(f"Refreshed the Kubernetes pull secret for {joined}")


@app.command("prepare-run")
def prepare_run_cmd(
    yaml_path: Path = typer.Argument(help="npa.workflow/v0.0.1 YAML path."),
    project: str = typer.Option(
        "", "--project", "-p", help="Configured project alias."
    ),
    resume_run: str = typer.Option(
        "",
        "--resume-run",
        help="Explicit existing run ID to resume; omitted always generates a fresh ID.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit scoped-state metadata."
    ),
) -> None:
    """Prepare a project/workflow-scoped fresh or explicit-resume run ID."""

    from npa.orchestration.npa_workflow.first_run_state import prepare_run
    from npa.orchestration.npa_workflow.run_resolution import validate_run_id

    try:
        spec = load_spec(yaml_path)
        requested = validate_run_id(resume_run) if resume_run else ""
        prepared = prepare_run(
            project=project,
            workflow_identity=spec.name,
            resume_run=requested,
        )
        from npa.orchestration.npa_workflow.submission_state import (
            update_submission_state,
        )

        update_submission_state(
            project or "default",
            prepared.run_id,
            {
                "launch_state": "reserved",
                "workflow": {
                    "name": spec.name,
                    "kind": "npa.workflow/v0.0.1",
                },
                "planning": {
                    "state": "durable",
                    "source": "prepare-run",
                },
            },
        )
    except Exception as exc:
        _fail(str(exc))
        return
    if prepared.warning:
        typer.echo(f"Warning: {prepared.warning}", err=True)
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "run_id": prepared.run_id,
                    "generated_new": prepared.generated_new,
                    "resume_explicit": bool(resume_run),
                    "state_path": prepared.state_path,
                    "previous_run": prepared.previous_run,
                    "lifecycle_state": "PLAN_ONLY",
                    "submission_state": "NOT_SUBMITTED",
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        # Deliberately one stdout line for safe shell capture.
        typer.echo(prepared.run_id)


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
    resume_run: str = typer.Option(
        "",
        "--resume-run",
        help="Explicit existing run ID to resume. Omitted means a fresh run.",
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
    accept_eula: bool = typer.Option(
        True,
        "--accept-eula/--no-accept-eula",
        help=(
            "Run-scoped Isaac EULA routing. Resolved Isaac stages receive "
            "ACCEPT_EULA=Y by default; use --no-accept-eula to opt out."
        ),
    ),
    details: bool = typer.Option(
        False,
        "--details/--compact",
        help=(
            "With --plan-only, include the complete rendered SkyPilot YAML. "
            "The default human plan shows shared setup once and compact stage deltas; "
            "JSON always retains full details."
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
        False,
        "--resume/--no-resume",
        help=(
            "With --runtime: replay waves already recorded as succeeded for this run id "
            "(disabled by default; prefer the explicit --resume-run ID contract)."
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
        help=(
            "With --runtime: per-wave deadline before the job is cancelled; "
            "0 waits indefinitely."
        ),
    ),
    cancel_on_timeout: bool = typer.Option(
        True,
        "--cancel-on-timeout/--no-cancel-on-timeout",
        help=(
            "With --runtime: request cancellation only for the exact recorded "
            "managed-job ID on timeout and record whether it was verified."
        ),
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
    image_override: list[str] = typer.Option(
        [],
        "--image-override",
        "--tool-image",
        help=(
            "Repeatable TOOL_REF=IMAGE override for npa.workflow specs. Exact "
            "or tool-family matches take precedence over --image."
        ),
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
    refresh_registry_secret: bool = typer.Option(
        True,
        "--refresh-registry-secret/--no-refresh-registry-secret",
        help=(
            "Refresh the Kubernetes Nebius pull secret before submit. Disable only "
            "when the cluster already has a separately managed pull credential."
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
    bind_controller: bool = typer.Option(
        False,
        "--bind-controller",
        help="Explicitly bind an unowned shared controller to this NPA project/context.",
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
    input_video: Path | None = typer.Option(
        None,
        "--input-video",
        help=(
            "PAIDF only: replace the pinned starter with a local H.264 MP4. "
            "The verified source and conditioning data are staged before provisioning."
        ),
    ),
    input_uri: str = typer.Option(
        "",
        "--input-uri",
        help="PAIDF only: replace the pinned starter with one readable s3://... MP4 object.",
    ),
    seed_fixture: bool = typer.Option(
        False,
        "--seed-fixture",
        help=(
            "PAIDF only: explicitly use repository-generated synthetic geometric "
            "frames for development/tests; never selected by default."
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
    from npa.orchestration.npa_workflow.run_state import PAIDF_WORKFLOW_NAME
    from npa.orchestration.skypilot.workflow import (
        SkyPilotSubmitError,
        WorkflowResult,
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
    try:
        specific_image_overrides = _parse_image_overrides(image_override)
    except ValueError as exc:
        _fail(str(exc))
        return
    is_npa_spec = is_npa_workflow_spec(yaml_path)
    merged_npa_spec = None
    if is_npa_spec:
        from npa.orchestration.npa_workflow.submit import load_spec_for_submit

        try:
            # This is the authoritative spec for every later preflight.  It is
            # intentionally loaded before credentials, images, ledgers, input
            # staging, provisioning, or accelerator discovery.
            merged_npa_spec = load_spec_for_submit(
                yaml_path, config_overrides=substitutions
            )
        except Exception as exc:
            _fail(str(exc))
            return
    if image_override and not is_npa_spec:
        _fail(
            "--image-override/--tool-image is supported only for "
            "npa.workflow/v0.0.1 specs"
        )
        return
    is_paidf_spec = bool(
        merged_npa_spec is not None
        and merged_npa_spec.name == PAIDF_WORKFLOW_NAME
    )
    legacy_fixture = _is_truthy_submit_value(
        substitutions.get("seed_fixture")
    ) or _is_truthy_submit_value(substitutions.get("seed_default_input"))
    fixture_requested = seed_fixture or legacy_fixture
    try:
        from npa.workflows.data_factory_input import select_paidf_input

        select_paidf_input(
            input_video=input_video,
            input_uri=input_uri,
            seed_fixture=fixture_requested,
        )
    except RuntimeError as exc:
        _fail(str(exc))
        return
    if (
        input_video is not None or input_uri.strip() or fixture_requested
    ) and not is_paidf_spec:
        _fail(
            "--input-video, --input-uri, and --seed-fixture are supported only by "
            "the physical-ai-data-factory workflow"
        )
        return
    materializer = _resolve_materializer(tool, yaml_path)
    if run_id and resume_run:
        _fail("--run-id and --resume-run are mutually exclusive")
        return
    if resume and not (run_id or resume_run):
        _fail("--resume requires an explicit --resume-run ID (or legacy --run-id)")
        return
    workflow_identity = ""
    if is_npa_spec:
        assert merged_npa_spec is not None
        workflow_identity = merged_npa_spec.name
        from npa.orchestration.npa_workflow.first_run_state import prepare_run

        try:
            prepared_identity = prepare_run(
                project=project,
                workflow_identity=workflow_identity,
                resume_run=resume_run,
                new_run_id=run_id,
                persist=False,
            )
        except Exception as exc:
            _fail(str(exc))
            return
        resolved_run_id = prepared_identity.run_id
        resume = bool(resume_run or (resume and run_id))
        if prepared_identity.warning and output_format != OutputFormat.json:
            typer.echo(f"Warning: {prepared_identity.warning}", err=True)
        if output_format != OutputFormat.json and not plan_only:
            typer.echo(
                ("Explicitly resuming" if resume else "Reserved fresh run")
                + f" {resolved_run_id}; state will be created only after preflight",
                err=True,
            )
    else:
        if resume_run:
            _fail("--resume-run is supported only for npa.workflow specs")
            return
        resolved_run_id = run_id or _default_submit_run_id(yaml_path)
    from npa.orchestration.npa_workflow.run_resolution import validate_run_id

    try:
        resolved_run_id = validate_run_id(resolved_run_id)
    except Exception as exc:
        _fail(str(exc))
        return
    routes_at_isaac = False
    if not plan_only and not accept_eula:
        routes_at_isaac = (
            _plan_routes_at_isaac(
                yaml_path,
                run_id=resolved_run_id,
                assume_decision=assume_decision,
                config_overrides=substitutions,
                options=SkypilotRenderOptions(
                    registry=_resolve_submit_registry(registry, project),
                    image_overrides={
                        **(
                            {"*": image}
                            if str(image or "").strip().lower()
                            not in {"", "none", "default", "-"}
                            else {"*": ""}
                            if str(image or "").strip().lower()
                            in {"none", "default", "-"}
                            else {}
                        ),
                        **specific_image_overrides,
                    },
                    gpu_target=gpu_target,
                    image_variant=image_variant,
                    materialize_registry_secrets=False,
                    accept_eula=accept_eula,
                ),
            )
            if is_npa_spec
            else _sky_yaml_routes_at_isaac(yaml_path)
        )
    if routes_at_isaac:
        _fail(
            "Refusing before provisioning: --no-accept-eula explicitly opted this "
            "Isaac-routed workflow out of NVIDIA's documented ACCEPT_EULA setting. "
            "The applicable NVIDIA Omniverse Licence Agreement and Isaac Sim "
            "Additional Software and Materials Licence are listed at "
            "https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html. "
            "PRIVACY_CONSENT is optional and is not enabled by NPA. No expensive "
            "action has begun. Resume by omitting --no-accept-eula."
        )
        return
    from npa.orchestration.npa_workflow.submit_credentials import (
        resolve_submit_credentials,
    )

    checkpoint_access_error = ""
    checkpoint_modalities: set[str] = set()
    try:
        required_secret_env = list(secret_env)
        checkpoint_tool_refs = {
            "workbench.cosmos2.transfer_execute",
            "workbench.cosmos2.transfer_conditioned_execute",
        }
        checkpoint_access_required = bool(
            not plan_only
            and merged_npa_spec is not None
            and any(
                state.tool_ref in checkpoint_tool_refs
                for state in merged_npa_spec.states.values()
            )
        )
        if checkpoint_access_required:
            required_secret_env.append("HF_TOKEN")
        submit_credentials = resolve_submit_credentials(
            project=project,
            explicit_endpoint=s3_endpoint,
            requested=required_secret_env,
        )
        secret_env[:] = list(dict.fromkeys(required_secret_env))
    except Exception as exc:
        _fail(f"Workflow credential resolution failed: {exc}")
        return
    s3_endpoint = submit_credentials.endpoint_url
    extra_env: dict[str, str] = dict(submit_credentials.secret_values)
    missing_secrets = list(submit_credentials.missing)
    if checkpoint_access_required and "HF_TOKEN" in missing_secrets:
        missing_secrets.remove("HF_TOKEN")
        checkpoint_access_error = (
            "HF_TOKEN is required to verify the exact gated Cosmos Transfer "
            "checkpoint before provisioning or GPU work"
        )
    if missing_secrets and not plan_only:
        _fail(
            "Required secret values are not present in the process environment "
            "or supported NPA credentials for project "
            f"{project or '<default>'}: {', '.join(missing_secrets)}. "
            "Set them explicitly or store them with `npa configure`."
        )
        return
    if merged_npa_spec is not None and checkpoint_access_required:
        transfer_tool_refs = {
            "workbench.cosmos2.transfer_execute",
            "workbench.cosmos2.transfer_conditioned_execute",
        }
        transfer_states = [
            state
            for state in merged_npa_spec.states.values()
            if state.tool_ref in transfer_tool_refs
        ]
        if transfer_states and not checkpoint_access_error:
            try:
                checkpoint_modalities = _transfer_control_modalities(
                    merged_npa_spec, run_id=resolved_run_id
                )
            except Exception as exc:  # token failures must not reach GPU work
                checkpoint_access_error = (
                    "could not resolve the state-local Cosmos control modality "
                    f"for checkpoint preflight: {exc}"
                )

    if checkpoint_access_error and skip_preflight:
        # This switch may skip ordinary convenience diagnostics, but never a
        # gated-model authorization boundary.
        _fail(checkpoint_access_error)
        return

    prepared_npa = None
    # The runtime registry-auth render has its own temporary directory.  Keep
    # the cleanup sentinel in the enclosing submit scope: fully image-pinned
    # workflows with --no-stage-src never enter the source-staging branch
    # below, but their fail-fast render errors must still cleanly reach the
    # operator instead of being masked by an unbound local in ``finally``.
    registry_auth_plan = None
    deploy_targets = []
    resolved_deploy_plans: dict[str, Any] = {}
    paidf_placement_prechecked = False
    source_action = "not-required"
    planned_source_uri = ""
    if is_npa_spec:
        # One prerequisite report instead of a sequence of one-at-a-time
        # failures spread over the render, the SkyPilot resolver and the
        # controller. Everything the operator still has to do is listed once,
        # each with the command that fixes it.
        assert merged_npa_spec is not None
        spec_config = dict(merged_npa_spec.config)
        # Resolve the pinned context before the preflight reports on it: npa keeps
        # its cluster kubeconfigs outside ~/.kube/config, so a context it
        # provisioned looks missing until KUBECONFIG points at it.
        infra_context = _infra_kube_context(infra)
        from npa.orchestration.npa_workflow.deploy import (
            bind_deploy_targets_to_submit,
            parse_deploy_targets,
        )

        try:
            deploy_targets = bind_deploy_targets_to_submit(
                parse_deploy_targets(merged_npa_spec),
                project=project,
                infra=infra,
            )
        except Exception as exc:  # noqa: BLE001 - fail before any mutation
            _fail(f"deployIfAbsent target resolution failed: {exc}")
            return
        if is_paidf_spec and not infra_context:
            declared_contexts = sorted(
                {
                    target.resolved_context
                    for target in deploy_targets
                    if target.cloud.strip().lower() in {"k8s", "kubernetes"}
                    and target.resolved_context
                }
            )
            if len(declared_contexts) != 1:
                _fail(
                    "PAIDF Kubernetes target cannot be resolved safely without "
                    "--infra: expected exactly one Kubernetes deployIfAbsent context, "
                    f"found {declared_contexts or 'none'}. Pass --infra k8s/<context>; "
                    "no S3 input or source was written."
                )
                return
            infra_context = declared_contexts[0]
            infra = f"k8s/{infra_context}"
        if infra_context and not plan_only:
            _adopt_npa_kubeconfig(infra_context)
        image_value_for_source = str(image or "").strip().lower()
        image_pins_all_tasks = bool(
            image_value_for_source
        ) and image_value_for_source not in {
            "none",
            "default",
            "-",
        }
        if image_pins_all_tasks and stage_src is True:
            _fail(
                "--image and --stage-src conflict: an image override supplies npa "
                "to every task, while --stage-src explicitly forces a source overlay. "
                "Choose one source strategy."
            )
            return
        try:
            requires_npa_source = _plan_requires_npa_source(
                yaml_path,
                run_id=resolved_run_id,
                assume_decision=assume_decision,
                config_overrides=substitutions,
                options=SkypilotRenderOptions(
                    registry=_resolve_submit_registry(registry, project),
                    image_overrides={
                        **(
                            {"*": image}
                            if image_pins_all_tasks
                            else {"*": ""}
                            if image_value_for_source in {"none", "default", "-"}
                            else {}
                        ),
                        **specific_image_overrides,
                    },
                    gpu_target=gpu_target,
                    image_variant=image_variant,
                    materialize_registry_secrets=False,
                    accept_eula=accept_eula,
                ),
            )
        except Exception as exc:
            _fail(f"cannot resolve the workflow's source requirement: {exc}")
            return
        bucket_for_source = str(
            s3_bucket or spec_config.get("bucket", "") or ""
        ).strip()
        existing_source_uri, source_origin = _resolve_submit_src_s3_uri_with_origin(
            project
        )
        if (
            requires_npa_source
            and existing_source_uri
            and not _valid_npa_source_uri(existing_source_uri)
        ):
            _fail(
                "NPA_SRC_S3_URI must be an s3:// URI with a bucket and key prefix; "
                "no source or run state was written."
            )
            return
        if existing_source_uri:
            # The renderer has a process/config resolver without a project
            # parameter. Pin the explicitly selected project's URI for this
            # invocation so a non-default project cannot inherit another
            # project's source prefix.
            os.environ["NPA_SRC_S3_URI"] = existing_source_uri
        local_source_fingerprint = ""
        if requires_npa_source or stage_src is True:
            try:
                local_source_fingerprint = _local_source_fingerprint()
            except Exception as exc:
                if stage_src is not False and not existing_source_uri:
                    _fail(f"npa source staging is not feasible: {exc}")
                    return
        existing_fingerprint = existing_source_uri.rstrip("/").rsplit("/", 1)[-1]
        persisted_source_is_stale = bool(
            requires_npa_source
            and existing_source_uri
            and source_origin == "saved"
            and local_source_fingerprint
            and existing_fingerprint != local_source_fingerprint
        )
        auto_stage_source = (
            stage_src is None
            and requires_npa_source
            and (not existing_source_uri or persisted_source_is_stale)
        )
        stage_source_planned = stage_src is True or auto_stage_source
        if image_pins_all_tasks:
            source_action = "image-override"
        elif stage_source_planned:
            source_action = "planned"
            destination_bucket = (
                bucket_for_source
                if not _is_placeholder_bucket(bucket_for_source)
                else "planned-bucket"
            )
            planned_source_uri = (
                f"s3://{destination_bucket}/npa-src/npa/"
                f"{local_source_fingerprint or '<source-fingerprint>'}/"
            )
            if plan_only:
                # The renderer needs the URI it would receive on real submit,
                # but this is metadata only: no S3 or config write occurs.
                os.environ["NPA_SRC_S3_URI"] = planned_source_uri
        elif requires_npa_source and existing_source_uri:
            source_action = (
                "reuse-explicit" if source_origin == "environment" else "reuse-saved"
            )
        elif requires_npa_source:
            source_action = "disabled"
        else:
            source_action = "not-required"

        # Resolve and quota-check the exact deployIfAbsent topology before the
        # writable-storage probe below. The probe is cleaned up, but it is still
        # an S3 write; a known identity/quota blocker must prevent even that
        # temporary mutation.
        if deploy_if_absent:
            from npa.orchestration.npa_workflow.deploy import (
                plan_infra_present,
            )

            try:
                resolved_deploy_plans = plan_infra_present(
                    deploy_targets, mutation=not plan_only
                )
            except Exception as exc:  # noqa: BLE001 - normalized before all mutation
                _fail(f"deployIfAbsent preflight failed: {exc}")
                return
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
                requires_npa_source=requires_npa_source,
                source_staging_planned=stage_source_planned,
                checkpoint_access_error=checkpoint_access_error,
                probe_storage=False,
            )
            if not plan_only and is_paidf_spec:
                from npa.orchestration.npa_workflow.paidf_preflight import (
                    static_prerequisites as paidf_static_prerequisites,
                )

                missing.extend(
                    paidf_static_prerequisites(
                        requested_secret_envs=secret_env,
                    )
                )
            if not plan_only and workflow_identity == "sim2real":
                from npa.clients.huggingface import validate_hf_access
                from npa.clients.kube import run_kubectl
                from npa.orchestration.npa_workflow.sim2real_preflight import (
                    kubernetes_prerequisites,
                    static_prerequisites,
                )

                missing.extend(
                    static_prerequisites(
                        spec_config,
                        requested_secret_envs=secret_env,
                        secret_values=extra_env,
                        hf_validator=validate_hf_access,
                    )
                )
                kubeconfig = os.environ.get("KUBECONFIG", "")

                def _run_sim2real_kubectl(args: list[str]):
                    return run_kubectl(
                        args,
                        context=infra_context,
                        kubeconfig=kubeconfig,
                        timeout=30,
                    )

                missing.extend(
                    kubernetes_prerequisites(
                        spec_config,
                        runner=_run_sim2real_kubectl,
                    )
                )
            if missing:
                _fail_missing_prerequisites(yaml_path, missing)
                return

        # An existing target can prove that PAIDF has nowhere schedulable to run
        # without any provider or model call.  Preserve that cheapest failure
        # ordering, then verify the exact modality-specific checkpoint before
        # image work, provisioning, or launch.  A deployIfAbsent target cannot
        # be placement-checked until it exists, so its checkpoint fence remains
        # ahead of provisioning below.
        if (
            is_paidf_spec
            and not plan_only
            and not skip_preflight
            and not deploy_if_absent
        ):
            placement_missing = _paidf_kubernetes_prerequisites_for_submit(
                infra_context
            )
            if placement_missing:
                _fail_missing_prerequisites(yaml_path, placement_missing)
                return
            paidf_placement_prechecked = True

        if checkpoint_access_required and not plan_only:
            from npa.workbench.cosmos.checkpoint_access import (
                CosmosCheckpointAccessError,
                preflight_control_checkpoint_access,
            )

            try:
                for modality in sorted(checkpoint_modalities):
                    preflight_control_checkpoint_access(
                        modality=modality,
                        token=str(
                            submit_credentials.secret_values.get("HF_TOKEN") or ""
                        ),
                    )
            except CosmosCheckpointAccessError as exc:
                _fail(str(exc))
                return

        # Image reachability and the complete cumulative infrastructure plan are
        # both read before source/input upload or any run/journal state exists.
        image_overrides_for_preflight: dict[str, str] = {}
        image_value_for_preflight = image.strip()
        if image_value_for_preflight.lower() in {"none", "default", "-"}:
            image_overrides_for_preflight["*"] = ""
        elif image_value_for_preflight:
            image_overrides_for_preflight["*"] = image_value_for_preflight
        image_overrides_for_preflight.update(specific_image_overrides)
        image_digest_pins = _preflight_submit_images(
            yaml_path,
            options=SkypilotRenderOptions(
                registry=_resolve_submit_registry(registry, project),
                image_overrides=image_overrides_for_preflight,
                gpu_target=gpu_target,
                image_variant=image_variant,
                materialize_registry_secrets=False,
            ),
            assume_decision=assume_decision,
            enabled=preflight_images and not plan_only,
            infra=infra,
        )

        # Provision/adopt the exact submission target before any writable-S3
        # probe, PAIDF input upload, or NPA source staging. PAIDF derives
        # ``infra`` from its sole deployIfAbsent context when the caller omits it,
        # so kubectl and SkyPilot can never diverge onto an ambient context.
        if deploy_if_absent and deploy_targets:
            from npa.orchestration.npa_workflow.deploy import ensure_infra_present

            try:
                if not plan_only:
                    records = ensure_infra_present(
                        deploy_targets,
                        dry_run=False,
                        gpu_readiness_timeout=gpu_readiness_timeout,
                        gpu_readiness_poll_interval=gpu_readiness_poll_interval,
                        sky_bin=sky_bin,
                        resolved_plans=resolved_deploy_plans,
                    )
                else:
                    records = [
                        {
                            "profile": next(
                                target.profile
                                for target in deploy_targets
                                if target.resolved_context == context
                            ),
                            "status": plan.decision,
                            "context": context,
                            "actions": [],
                            "warnings": list(plan.reasons),
                            "topology": plan.topology.to_dict(),
                            "quotas": [quota.to_dict() for quota in plan.quotas],
                        }
                        for context, plan in resolved_deploy_plans.items()
                    ]
                for record in records:
                    typer.echo(
                        "deployIfAbsent["
                        f"{record['profile']}]: {record['status']} "
                        f"context={record['context']} "
                        f"actions={','.join(record['actions']) or 'none'}",
                        err=True,
                    )
                    for warning in record.get("warnings", []) or []:
                        typer.echo(
                            f"deployIfAbsent[{record['profile']}]: warning: {warning}",
                            err=True,
                        )
            except NpaWorkflowError as exc:
                _fail(str(exc))
                return

        if infra_context and not plan_only and not _adopt_npa_kubeconfig(infra_context):
            _fail(
                f"Kube context {infra_context!r} (submission target {infra!r}) is not "
                "available in KUBECONFIG or under "
                f"~/.npa/clusters/{infra_context}/. Provision it with `npa "
                "provision-if-absent --project <alias>`, or pass "
                "--infra k8s/<context> for an available context; no S3 input or source "
                "was written."
            )
            return
        if infra_context and not plan_only:
            from npa.controller_ownership import (
                ClusterOwnerIdentityMismatchError,
                bind_controller_owner,
                resolve_controller_candidate,
                verify_controller_owner,
            )

            try:
                if bind_controller is True:
                    bind_controller_owner(
                        resolve_controller_candidate(project, infra_context)
                    )
                verify_controller_owner(project, infra_context)
            except ClusterOwnerIdentityMismatchError as exc:
                _fail(str(exc))
                return

        if not skip_preflight and not plan_only:
            post_infra_missing: list[tuple[str, str]] = []
            if is_paidf_spec and not paidf_placement_prechecked:
                post_infra_missing.extend(
                    _paidf_kubernetes_prerequisites_for_submit(infra_context)
                )
            if post_infra_missing:
                _fail_missing_prerequisites(yaml_path, post_infra_missing)
                return
            storage_missing = _submit_storage_prerequisites(
                spec_config,
                requires_s3=_spec_requires_s3(yaml_path),
                s3_endpoint=submit_credentials.endpoint_url,
                s3_access_key_id=getattr(
                    submit_credentials, "access_key_id", ""
                ),
                s3_secret_access_key=getattr(
                    submit_credentials, "secret_access_key", ""
                ),
            )
            if storage_missing:
                _fail_missing_prerequisites(yaml_path, storage_missing)
                return

        if not plan_only:
            # Establish the exact run and current-schema no-launch evidence
            # before PAIDF input/source staging mutates storage. Infrastructure
            # is already resolved and placement-checked above. Status can
            # therefore prove NOT_SUBMITTED if a later staging step fails.
            try:
                persisted_identity = prepare_run(
                    project=project,
                    workflow_identity=workflow_identity,
                    resume_run=resolved_run_id if resume else "",
                    new_run_id="" if resume else resolved_run_id,
                    persist=True,
                )
                from npa.orchestration.npa_workflow.submission_state import (
                    update_submission_state,
                )

                update_submission_state(
                    project or "default",
                    resolved_run_id,
                    {
                        "launch_state": "planned",
                        "workflow": {
                            "name": workflow_identity,
                            "kind": "npa.workflow/v0.0.1",
                        },
                        "planning": {
                            "state": "durable",
                            "source_action": source_action,
                            "input_action": "planned"
                            if is_paidf_spec
                            else "not-required",
                            "infra_context": infra_context,
                        },
                    },
                )
            except Exception as exc:
                _fail(f"could not persist pre-mutation submission ledger: {exc}")
                return
            if output_format != OutputFormat.json:
                typer.echo(
                    f"Run planning ledger persisted before mutation: "
                    f"{persisted_identity.state_path}",
                    err=True,
                )
        # Validate and stage the selected PAIDF input before staging the NPA
        # source. Invalid media, inaccessible input, and provenance conflicts
        # therefore leave no source upload or run/controller state. The PAIDF
        # input prefix is itself provenance-committed and restart-safe.
        if is_paidf_spec:
            from npa.workflows.data_factory_input import (
                PaidfInputError,
                plan_paidf_input,
                prepare_paidf_input,
            )

            try:
                if plan_only:
                    prepared_input = plan_paidf_input(
                        run_id=resolved_run_id,
                        bucket=bucket_for_source,
                        input_video=input_video,
                        input_uri=input_uri,
                        seed_fixture=fixture_requested,
                    )
                else:
                    prepared_input = prepare_paidf_input(
                        run_id=resolved_run_id,
                        bucket=bucket_for_source,
                        input_video=input_video,
                        input_uri=input_uri,
                        seed_fixture=fixture_requested,
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=extra_env.get("AWS_ACCESS_KEY_ID", ""),
                        aws_secret_access_key=extra_env.get(
                            "AWS_SECRET_ACCESS_KEY", ""
                        ),
                        reporter=lambda message: typer.echo(message, err=True),
                    )
            except PaidfInputError as exc:
                _fail(str(exc))
                return
            substitutions.update(prepared_input.config_overrides())
            substitutions["seed_fixture"] = (
                "true" if prepared_input.selection == "synthetic_fixture" else "false"
            )
            substitutions["seed_default_input"] = substitutions["seed_fixture"]

        if (
            not stage_source_planned
            and not plan_only
            and requires_npa_source
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
                        aws_secret_access_key=extra_env.get(
                            "AWS_SECRET_ACCESS_KEY", ""
                        ),
                    ),
                    expected_fingerprint=existing_fingerprint,
                )
            except Exception as exc:  # noqa: BLE001
                _fail(str(exc))
                return
        if stage_source_planned and not plan_only:
            staged_uri = _stage_npa_src_for_submit(
                spec_config,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
                credential_values=extra_env,
                project=project,
                run_id=resolved_run_id,
                force=stage_src is True,
            )
            if not staged_uri:
                return
            os.environ["NPA_SRC_S3_URI"] = staged_uri
            source_action = "reused" if staged_uri == existing_source_uri else "staged"
        _warn_placeholder_bucket(spec_config, quiet=output_format == OutputFormat.json)
        image_overrides: dict[str, str] = {}
        # ``none`` / ``default`` clears workbench image pins so tasks use the
        # SkyPilot default image (needed when registry images fail k8s apt-ssh).
        image_value = image.strip()
        if image_value.lower() in {"none", "default", "-"}:
            image_overrides["*"] = ""
        elif image_value:
            image_overrides["*"] = image_value
        image_overrides.update(specific_image_overrides)

        npa_render_options = SkypilotRenderOptions(
            registry=_resolve_submit_registry(registry, project),
            image_overrides=image_overrides,
            image_digest_pins=image_digest_pins,
            aws_endpoint_url=s3_endpoint
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("NEBIUS_S3_ENDPOINT")
            or "https://storage.eu-north1.nebius.cloud",
            gpu_target=gpu_target,
            image_variant=image_variant,
            # Never mint/print live registry tokens for --plan-only.
            materialize_registry_secrets=not plan_only,
            accept_eula=accept_eula,
            gpu_accelerator_overrides=_resolve_submit_accelerators(
                yaml_path,
                spec=merged_npa_spec,
                infra=infra,
                sky_bin=sky_bin,
                enabled=resolve_accelerators and not plan_only,
                readiness_timeout=gpu_readiness_timeout,
                readiness_poll_interval=gpu_readiness_poll_interval,
            ),
        )
        if not plan_only and not skip_preflight and merged_npa_spec is not None:
            try:
                _preflight_submit_gang_capacity(
                    merged_npa_spec,
                    context=infra_context,
                    accelerator_overrides=npa_render_options.gpu_accelerator_overrides,
                    allowed_nodes=None,
                    sky_bin=sky_bin,
                    config_path=config_path,
                    isolated_config_dir=isolated_config_dir,
                )
            except Exception as exc:
                _fail(f"multi-node GPU capacity preflight failed: {exc}")
                return

        if runtime and not plan_only:
            # Runtime renders one wave at a time and historically returned before
            # the one-shot path refreshed Kubernetes pull credentials.  Validate
            # the complete selected plan once solely to install the exact private-
            # registry secret before any managed job can enter ErrImagePull.
            try:
                registry_auth_plan = prepare_npa_workflow_for_submit(
                    yaml_path,
                    run_id=resolved_run_id,
                    assume_decision=assume_decision,
                    config_overrides=substitutions,
                    render_options=npa_render_options,
                )
                if refresh_registry_secret:
                    _refresh_kubernetes_pull_secrets(
                        registry_auth_plan.skypilot_yaml_path,
                        k8s_context=infra_context,
                        kubeconfig=os.environ.get("KUBECONFIG", ""),
                    )
            except NpaWorkflowError as exc:
                _fail(str(exc))
                return
            finally:
                if registry_auth_plan is not None:
                    registry_auth_plan.temp_dir.cleanup()
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
                config_path=config_path,
                sky_bin=sky_bin or "",
                submit_timeout=submit_timeout,
                poll_seconds=poll_seconds,
                max_wait_seconds=max_wait_seconds,
                cancel_on_timeout=cancel_on_timeout,
                retries=retries,
                max_concurrency=max_concurrency,
                resume=resume,
                refresh_registry_secret=refresh_registry_secret,
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
            infrastructure = {
                context: plan.to_dict()
                for context, plan in resolved_deploy_plans.items()
            }
            plan_checks: list[dict[str, object]] = []
            if submit_credentials.missing:
                plan_checks.append(
                    {
                        "name": "credentials",
                        "status": "blocked",
                        "reason": (
                            "missing requested credential names: "
                            + ", ".join(submit_credentials.missing)
                        ),
                    }
                )
            else:
                plan_checks.append(
                    {
                        "name": "credentials",
                        "status": "ready",
                        "reason": "requested credentials are resolvable (values redacted)",
                    }
                )
            if _spec_requires_s3(yaml_path):
                plan_checks.append(
                    {
                        "name": "writable_storage",
                        "status": "unknown",
                        "reason": (
                            "plan-only is read-only; real submit will run the cleaned "
                            "append-only capability probe before staging or submission"
                        ),
                    }
                )
            plan_checks.append(
                {
                    "name": "source_staging",
                    "status": "ready" if source_action != "disabled" else "blocked",
                    "reason": source_action,
                }
            )
            decisions = [
                str(item.get("status") or "unknown") for item in plan_checks
            ] + [
                str(item.get("decision") or "unknown")
                for item in infrastructure.values()
            ]
            preflight_decision = (
                "blocked"
                if "blocked" in decisions
                else "unknown"
                if "unknown" in decisions
                else "ready"
            )
            planned_payload: dict[str, Any] = {
                "status": "PLANNED",
                "lifecycle_state": "PLAN_ONLY",
                "submission_state": "NOT_SUBMITTED",
                "submission_receipt": None,
                "run_id": resolved_run_id,
                "workflow": prepared_npa.spec.name,
                "steps": len(prepared_npa.plan.steps),
                "secret_env_hints": list(prepared_npa.secret_env_hints),
                "source": {
                    "status": source_action,
                    "uri": planned_source_uri or existing_source_uri,
                },
                "preflight": {
                    "decision": preflight_decision,
                    "checks": plan_checks,
                },
                "infrastructure": infrastructure,
                "plan": prepared_npa.plan.to_dict(),
                "skypilot_yaml": rendered,
            }
            if output_format == OutputFormat.json:
                typer.echo(json.dumps(planned_payload, indent=2, sort_keys=True))
            else:
                typer.echo("status: PLANNED")
                typer.echo("lifecycle_state: PLAN_ONLY")
                typer.echo("submission_state: NOT_SUBMITTED")
                typer.echo(f"preflight_decision: {preflight_decision}")
                typer.echo(f"run_id: {resolved_run_id}")
                typer.echo(f"workflow: {prepared_npa.spec.name}")
                typer.echo(f"steps: {len(prepared_npa.plan.steps)}")
                typer.echo(
                    "source: "
                    + source_action
                    + (
                        f" ({planned_source_uri or existing_source_uri})"
                        if planned_source_uri or existing_source_uri
                        else ""
                    )
                )
                _emit_compact_submit_plan(
                    prepared_npa.plan,
                    infrastructure=infrastructure,
                )
                if prepared_npa.secret_env_hints:
                    typer.echo(
                        "secret_env_hints: " + ",".join(prepared_npa.secret_env_hints)
                    )
                if details:
                    typer.echo("--- full rendered SkyPilot YAML ---")
                    typer.echo(rendered)
                else:
                    typer.echo("details: pass --details (or --output-format json)")
            prepared_npa.temp_dir.cleanup()
            return

        if refresh_registry_secret:
            _refresh_kubernetes_pull_secrets(
                prepared_npa.skypilot_yaml_path,
                k8s_context=infra_context,
                kubeconfig=os.environ.get("KUBECONFIG", ""),
            )

        # Skip SkyPilot-path materializers; npa.workflow already planned.
        materializer = ""
        substitutions = {}
        yaml_path = prepared_npa.skypilot_yaml_path
        if prepared_npa.secret_env_hints:
            missing_secret_hints = [
                name for name in prepared_npa.secret_env_hints if name not in secret_env
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
            assert submitted_yaml_context is not None
            substituted = _substitute_workflow_vars(yaml_path, substitutions)
            source_yaml_path = (
                Path(submitted_yaml_context.name) / f"substituted-{yaml_path.name}"
            )
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
                    accept_eula=accept_eula,
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
                    submitted_yaml_path
                    if submitted_yaml_path.exists()
                    else source_yaml_path,
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

        ledger_project = project or "default"
        from npa.orchestration.skypilot.launch_transaction import (
            logical_launch_identity,
        )

        launch_identity = logical_launch_identity(
            ledger_project,
            resolved_run_id,
            "single-wave",
            "attempt-1",
            hashlib.sha256(submitted_yaml_path.read_bytes()).hexdigest(),
        )

        def _record_transaction(payload: dict[str, object]) -> None:
            if prepared_npa is None:
                return
            from npa.orchestration.npa_workflow.submission_state import (
                update_submission_state,
            )

            update_submission_state(
                ledger_project,
                resolved_run_id,
                {"launch": dict(payload)},
                locked=True,
            )

        def _launch() -> WorkflowResult:
            from npa.clients.config import default_project_name, resolve_environment
            from npa.provisioning_journal import (
                ProvisioningOperation,
                current_operation,
                operation_context,
            )

            def submit() -> WorkflowResult:
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
                    logical_launch_id=launch_identity,
                    transaction_recorder=_record_transaction,
                )

            if current_operation() is not None:
                return submit()
            alias = str(project or default_project_name()).strip() or "default"
            environment = resolve_environment(alias)
            operation = ProvisioningOperation.prepare(
                command="npa workbench workflow submit",
                project_alias=alias,
                project_id=str(getattr(environment, "project_id", "") or ""),
                tenant_id=str(getattr(environment, "tenant_id", "") or ""),
                region=str(getattr(environment, "region", "") or ""),
                resource_type="workflow-submit",
                requested_name=resolved_run_id,
                ownership_source="workflow-submit-cli",
                resume_command="",
                resume_argv=[
                    "npa",
                    "workbench",
                    "workflow",
                    "submit",
                    str(yaml_path),
                    "--project",
                    alias,
                    "--resume-run",
                    resolved_run_id,
                ],
                destroy_argv=[
                    "npa",
                    "workbench",
                    "workflow",
                    "cancel",
                    resolved_run_id,
                    "--project",
                    alias,
                ],
            )
            with operation_context(operation):
                operation.transition("mutating")
                try:
                    submitted = submit()
                except BaseException as exc:
                    _record_workflow_submit_failure(operation, exc)
                    raise
                operation.transition("state-durable")
                operation.commit()
                return submitted

        if prepared_npa is not None:
            from npa.orchestration.npa_workflow.submission_state import (
                submission_lock,
                update_submission_state,
            )

            with submission_lock(ledger_project, resolved_run_id):
                update_submission_state(
                    ledger_project,
                    resolved_run_id,
                    {
                        "workflow": _npa_submission_receipt(
                            prepared_npa, resolved_run_id
                        )
                    },
                    locked=True,
                )
                result = _launch()
                update_submission_state(
                    ledger_project,
                    resolved_run_id,
                    {
                        "launch": {
                            **dict(getattr(result, "launch_transaction", {}) or {}),
                            "status": result.status.lower(),
                            "sky_job_id": result.job_id,
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
            result.log_paths["manifest_uri"] = (
                f"{workflow_state.uri.rstrip('/')}/manifest.json"
            )
            result.log_paths["stages"] = ",".join(
                instrumented_manifest.get("stages", {}).keys()
            )
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
        transaction = getattr(exc, "transaction", None)
        if output_format == OutputFormat.json and transaction is not None:
            typer.echo(
                json.dumps(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "launch_transaction": transaction.to_dict(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise typer.Exit(1) from exc
        _fail(str(exc))
        return
    finally:
        if submitted_yaml_context is not None:
            submitted_yaml_context.cleanup()
        if prepared_npa is not None:
            prepared_npa.temp_dir.cleanup()

    if output_format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {**result.__dict__, "run_id": resolved_run_id},
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo(f"status: {result.status}")
    typer.echo(f"run_id: {resolved_run_id}")
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
        typer.echo(
            f"warning: could not persist the npa.workflow run manifest: {exc}", err=True
        )
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
    config_path: Path | None,
    submit_timeout: int,
    poll_seconds: int,
    max_wait_seconds: int,
    cancel_on_timeout: bool,
    retries: int,
    max_concurrency: int,
    resume: bool,
    refresh_registry_secret: bool,
    output_format: "OutputFormat",
    project: str = "",
    auto_load: bool = True,
    agent_name: str = "",
    s3_endpoint: str = "",
    sky_bin: str = "",
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

    submitted_yaml = yaml_path.read_bytes()
    if config_overrides:
        import yaml

        source = yaml.safe_load(submitted_yaml)
        if not isinstance(source, dict) or not isinstance(source.get("config"), dict):
            _fail("npa.workflow source has no config mapping for --var overrides")
            return
        source["config"].update(config_overrides)
        submitted_yaml = yaml.safe_dump(source, sort_keys=False).encode("utf-8")

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

    resolved_secret_envs = secret_env_names(
        secret_envs, values=secret_env_values
    )
    pre_submit_hook = None
    if refresh_registry_secret:
        runtime_k8s_context = _infra_kube_context(infra)
        runtime_kubeconfig = os.environ.get("KUBECONFIG", "")

        def _refresh_runtime_pull_secret(rendered_path: Path) -> None:
            _refresh_kubernetes_pull_secrets(
                rendered_path,
                k8s_context=runtime_k8s_context,
                kubeconfig=runtime_kubeconfig,
            )

        pre_submit_hook = _refresh_runtime_pull_secret
    options = RuntimeOptions(
        poll_seconds=poll_seconds,
        max_wait_seconds=max_wait_seconds,
        retries=max(0, retries),
        cancel_on_timeout=cancel_on_timeout,
        max_concurrency=max(0, max_concurrency),
        secret_envs=resolved_secret_envs,
        secret_env_values=secret_env_values,
        submit_timeout=submit_timeout,
        infra=infra,
        controller_backend=controller_backend,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        resume=resume,
        project=project or "default",
        sky_bin=sky_bin,
        credential_resolver=lambda: _resolve_runtime_secret_values(
            project=project,
            requested=list(resolved_secret_envs),
        ),
        pre_submit_hook=pre_submit_hook,
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
                workflow_yaml=submitted_yaml,
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
    if (
        report.status == "succeeded"
        and auto_load
        and report.workflow == "physical-ai-data-factory"
    ):
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


def _resolve_runtime_secret_values(
    *, project: str, requested: list[str]
) -> dict[str, str]:
    """Re-resolve one wave's secrets from the selected project immediately before launch."""

    from npa.orchestration.npa_workflow.submit_credentials import (
        resolve_submit_credentials,
    )

    context = resolve_submit_credentials(project=project, requested=requested)
    return dict(context.secret_values)


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
        update_submission_state(project or "default", run_id, {"artifact_load": result})
        return result
    try:
        client = _storage_client(
            endpoint_url=s3_endpoint,
            aws_access_key_id=(credential_values or {}).get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=(credential_values or {}).get(
                "AWS_SECRET_ACCESS_KEY", ""
            ),
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
        update_submission_state(project or "default", run_id, {"artifact_load": result})
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

    return _resolve_submit_src_s3_uri_with_origin(project)[0]


def _resolve_submit_src_s3_uri_with_origin(project: str) -> tuple[str, str]:
    """Return ``(uri, environment|saved|missing)`` without exposing credentials."""

    value = (
        os.environ.get("NPA_SRC_S3_URI")
        or os.environ.get("NPA_E2E_NPA_SRC_S3_URI")
        or ""
    ).strip()
    if value:
        return value, "environment"
    try:
        from npa.clients.config import resolve_workflow_src_s3_uri

        saved = resolve_workflow_src_s3_uri(project or None)
        return (saved, "saved") if saved else ("", "missing")
    except Exception:  # noqa: BLE001 - submit preflight reports an unset source
        return "", "missing"


def _valid_npa_source_uri(value: str) -> bool:
    match = re.fullmatch(r"s3://([^/]+)/(.+)", str(value or "").strip())
    return bool(match and match.group(1).strip() and match.group(2).strip("/"))


def _local_source_fingerprint() -> str:
    """Inspect local source provenance without uploading or writing state."""

    from npa.orchestration.npa_workflow.src_staging import (
        find_npa_package_root,
        iter_source_files,
        source_fingerprint,
    )

    root = find_npa_package_root()
    files = list(iter_source_files(root))
    if not files:
        raise RuntimeError(f"no source files found under {root}")
    return source_fingerprint(root, files)


def _plan_requires_npa_source(
    yaml_path: Path,
    *,
    run_id: str,
    assume_decision: str,
    config_overrides: Mapping[str, str] | None = None,
    options: SkypilotRenderOptions,
) -> bool:
    """Return whether any fully configured planned step lacks a container image."""

    from npa.orchestration.npa_workflow import build_plan, load_spec
    from npa.orchestration.npa_workflow.skypilot_render import (
        build_scheduler_task,
        resolve_task_image,
    )
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    # Resource image fields may be config tokens populated only by submit
    # ``--var`` values. Inspect the same merged spec that render/runtime will use;
    # otherwise a fully digest-pinned workflow is incorrectly forced to stage an
    # unused source tree (and ``--no-stage-src`` cannot submit it at all).
    spec = merge_config_overrides(load_spec(yaml_path), config_overrides)
    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    for step in plan.steps:
        task = build_scheduler_task(spec, step, run_id=run_id)
        if not resolve_task_image(
            str(task.get("tool_ref") or ""),
            task.get("resources") or {},
            options=options,
        ):
            return True
    return False


def _plan_routes_at_isaac(
    yaml_path: Path,
    *,
    run_id: str,
    assume_decision: str,
    config_overrides: Mapping[str, str] | None = None,
    options,
) -> bool:
    """Return whether the selected plan acquires or runs an Isaac image."""

    from npa.orchestration.npa_workflow import build_plan, load_spec
    from npa.orchestration.npa_workflow.scheduler import build_scheduler_task
    from npa.orchestration.npa_workflow.skypilot_render import (
        resolve_task_image,
        routes_at_an_isaac_image,
    )
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = merge_config_overrides(load_spec(yaml_path), config_overrides)
    plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
    for step in plan.steps:
        task = build_scheduler_task(spec, step, run_id=run_id)
        resources = task.get("resources") or {}
        tool_ref = str(task.get("tool_ref") or "")
        resolved_image = resolve_task_image(tool_ref, resources, options=options)
        if routes_at_an_isaac_image(
            tool_ref,
            resources,
            spec.config,
            resolved_image=resolved_image,
        ):
            return True
    return False


def _sky_yaml_routes_at_isaac(yaml_path: Path) -> bool:
    """Return whether a raw SkyPilot YAML routes any task through Isaac."""

    from npa.orchestration.npa_workflow.skypilot_render import routes_at_an_isaac_image

    try:
        documents = yaml.safe_load_all(yaml_path.read_text(encoding="utf-8"))
        for document in documents:
            if not isinstance(document, dict):
                continue
            resources = dict(document.get("resources") or {})
            resources.setdefault("image", resources.get("image_id", ""))
            envs = document.get("envs") or {}
            policy_image = str(envs.get("POLICY_IMAGE") or "").lower()
            if routes_at_an_isaac_image("", resources) or "npa-sonic" in policy_image:
                return True
        return False
    except (OSError, yaml.YAMLError):
        return False


def _emit_compact_submit_plan(plan, *, infrastructure: Mapping[str, object]) -> None:
    """Render shared setup once and one workflow-specific delta per stage."""

    typer.echo("setup:")
    if infrastructure:
        for context, raw in infrastructure.items():
            item = raw if isinstance(raw, dict) else {}
            topology = item.get("topology") if isinstance(item, dict) else {}
            topology = topology if isinstance(topology, dict) else {}
            typer.echo(
                f"  deployIfAbsent {context}: {item.get('decision', 'unknown')} "
                f"cpu={topology.get('cpu_nodes', '?')}x"
                f"{topology.get('cpu_platform', '?')}/{topology.get('cpu_preset', '?')} "
                f"gpu={topology.get('gpu_nodes', '?')}x"
                f"{topology.get('gpu_platform', '?')}/{topology.get('gpu_preset', '?')} "
                f"preemptible={topology.get('gpu_preemptible', False)}"
            )
            for quota in item.get("quotas", []) if isinstance(item, dict) else []:
                if isinstance(quota, dict):
                    typer.echo(
                        f"    quota {quota.get('name')}: {quota.get('status')} "
                        f"required={quota.get('required')} available={quota.get('available')} "
                        f"shortfall={quota.get('shortfall')}"
                    )
    else:
        typer.echo("  infrastructure: existing/externally managed")
    typer.echo("stages:")
    previous = ""
    for index, step in enumerate(plan.steps, 1):
        profile = step.resources_profile or {}
        resource_class = step.resources or "default"
        image = str(profile.get("image_id") or profile.get("image") or "default")
        dependency = previous or "none"
        gate = step.loop_label or ("parallel:" + step.group if step.group else "none")
        typer.echo(
            f"  {index}. {step.state}: tool={step.tool_ref or 'shell'} "
            f"resource={resource_class} image={image} depends={dependency} gate={gate}"
        )
        previous = step.state


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
    infra: str = "",
) -> dict[str, str]:
    """Fail before the run starts when a step's image cannot actually be pulled.

    Authority is resolved per registry and execution path. An exact operator-side
    manifest fetch proves the pull directly; a Kubernetes path may instead prove a
    declared docker-config imagePullSecret for that registry. If neither authority
    is verified, every registry fails closed with a path-specific remedy.
    """

    if not enabled:
        return {}

    from npa.orchestration.npa_workflow import build_plan
    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.skypilot_render import (
        plan_image_pull_secrets,
        plan_images,
    )
    from npa.orchestration.skypilot.k8s_gpu_catalog import context_from_infra
    from npa.orchestration.skypilot.registry_preflight import (
        check_image_pulls_with_credentials,
    )

    try:
        spec = load_spec(yaml_path)
        run_id = f"{spec.name}-preflight"
        plan = build_plan(spec, run_id=run_id, assume_decision=assume_decision)
        images = plan_images(spec, plan.steps, run_id=run_id, options=options)
        pull_secrets_by_image = plan_image_pull_secrets(
            spec, plan.steps, run_id=run_id, options=options
        )
    except NpaWorkflowError:
        # Planning problems are reported by the submit path itself with better context.
        return {}
    if not images:
        return {}

    checks = check_image_pulls_with_credentials(
        images,
        mint=True,
        pull_secrets_by_image=pull_secrets_by_image,
        context=context_from_infra(infra),
    )
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
    contract_checks = _preflight_image_bootstrap_contracts(
        images=images,
        pull_checks=checks,
        context=context_from_infra(infra),
        pull_secrets_by_image=pull_secrets_by_image,
    )
    typer.echo(
        f"image-preflight: {len(checks)} image(s) pullable and bootstrap-compatible",
        err=True,
    )
    return {
        image: str(item.get("image") or "")
        for image, item in zip(dict.fromkeys(images), contract_checks, strict=True)
    }


def _preflight_image_bootstrap_contracts(
    *,
    images: list[str],
    pull_checks: Sequence[object],
    context: str,
    pull_secrets_by_image: Mapping[str, tuple[str, ...]] | None = None,
) -> list[dict[str, object]]:
    """Verify each selected digest, never a mutable tag, against one contract."""

    from npa.orchestration.skypilot.image_bootstrap_contract import (
        CONTRACT_VERSION,
        ImageBootstrapContractError,
        is_trusted_npa_image,
        load_cached_evidence,
        probe_image_capabilities,
        store_cached_evidence,
        verify_attestation,
    )
    from npa.orchestration.skypilot.registry_preflight import (
        fetch_image_config_metadata,
        parse_image_reference,
        resolve_registry_credentials,
    )
    from npa.deploy.images import requires_skypilot_bootstrap_runtime_probe

    check_by_image = {str(getattr(item, "image", "")): item for item in pull_checks}
    cache_path = Path.home() / ".npa" / "cache" / "sky-image-bootstrap.json"
    results: list[dict[str, object]] = []
    for image in dict.fromkeys(
        str(item).strip() for item in images if str(item).strip()
    ):
        try:
            host = parse_image_reference(image).registry
            username, password = resolve_registry_credentials(host, mint=True)
            digest, labels = fetch_image_config_metadata(
                image, username=username, password=password
            )
            pull_digest = str(getattr(check_by_image.get(image), "digest", "") or "")
            if pull_digest and pull_digest != digest:
                raise ImageBootstrapContractError(
                    "mutable tag resolved to a different digest between pull and contract checks"
                )
            runtime_probe_required = requires_skypilot_bootstrap_runtime_probe(image)
            cached = load_cached_evidence(cache_path, digest)
            if cached is not None and (
                not runtime_probe_required
                or cached.source == "ephemeral_capability_probe"
            ):
                evidence = cached
            else:
                attested = verify_attestation(image=image, digest=digest, labels=labels)
                if runtime_probe_required:
                    # Canonical and derived GR00T artifacts share one repository.
                    # The derived source carries a label, but the canonical source
                    # does not implement the full contract, so the label cannot
                    # establish provenance. Probe the selected immutable bytes and
                    # ignore stale label-backed cache entries for the same digest.
                    evidence = probe_image_capabilities(
                        image=image,
                        digest=digest,
                        context=context,
                        kubeconfig=str(os.environ.get("KUBECONFIG") or ""),
                    )
                elif attested.ok:
                    evidence = attested
                elif (
                    is_trusted_npa_image(image) or "version mismatch" in attested.detail
                ):
                    # NPA images are governed by the build-time attestation
                    # policy. Missing metadata is missing evidence, not a reason
                    # to substitute a runtime probe for that build contract.
                    evidence = attested
                else:
                    evidence = probe_image_capabilities(
                        image=image,
                        digest=digest,
                        context=context,
                        kubeconfig=str(os.environ.get("KUBECONFIG") or ""),
                        image_pull_secrets=tuple(
                            (pull_secrets_by_image or {}).get(image, ())
                        ),
                    )
                store_cached_evidence(cache_path, evidence)
        except (ImageBootstrapContractError, RuntimeError, OSError, ValueError) as exc:
            _fail(
                f"image bootstrap contract {CONTRACT_VERSION} could not be verified "
                f"for {image}: {exc}"
            )
        if not evidence.ok:
            _fail(
                f"image bootstrap contract {CONTRACT_VERSION} failed for "
                f"{evidence.image}: {evidence.detail or evidence.state}"
            )
        results.append(evidence.to_dict())
    return results


def _resolve_submit_accelerators(
    yaml_path: Path,
    *,
    spec=None,
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
        requested = spec_accelerators((spec or load_spec(yaml_path)).resources)
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


def _preflight_submit_gang_capacity(
    spec,
    *,
    context: str,
    accelerator_overrides: Mapping[str, str] | None = None,
    allowed_nodes: tuple[str, ...] | None = (),
    sky_bin: str = "",
    config_path: Path | None = None,
    isolated_config_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Check existing/new cluster free compatible nodes before job submission."""

    from npa.orchestration.npa_workflow.spec import (
        profile_num_nodes,
        resolve_resource_profile,
    )
    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        discover_kubernetes_gpu_inventory,
        preflight_kubernetes_gpu_gang,
    )

    checks: list[dict[str, object]] = []
    resolved_allowed_nodes = allowed_nodes
    for state in spec.states.values():
        profile = spec.resources.get(state.resources)
        if not isinstance(profile, Mapping):
            continue
        resolved = resolve_resource_profile(
            state.resources,
            profile,
            config=spec.config,
            run={"id": "capacity-preflight"},
        )
        nodes = profile_num_nodes(resolved, name=state.resources)
        accelerator = str(resolved.get("accelerators") or "").strip()
        if nodes <= 1 or not accelerator:
            continue
        if not context:
            raise RuntimeError(
                f"state {state.name!r} requests a {nodes}-node GPU gang, but no "
                "explicit Kubernetes context was selected; ambient context fallback "
                "is forbidden for gang capacity preflight"
            )
        if resolved_allowed_nodes is None:
            resolved_allowed_nodes = _skypilot_allowed_nodes(
                sky_bin=sky_bin,
                config_path=config_path,
                isolated_config_dir=isolated_config_dir,
            )
        selected = str(
            (accelerator_overrides or {}).get(accelerator) or accelerator
        )
        kubernetes = resolved.get("kubernetes")
        kubernetes = kubernetes if isinstance(kubernetes, Mapping) else {}
        pod_config = kubernetes.get("pod_config")
        pod_config = pod_config if isinstance(pod_config, Mapping) else {}
        pod_spec = pod_config.get("spec")
        pod_spec = pod_spec if isinstance(pod_spec, Mapping) else {}
        inventory = discover_kubernetes_gpu_inventory(context=context)
        result = preflight_kubernetes_gpu_gang(
            inventory,
            accelerator=selected,
            node_count=nodes,
            cpus=resolved.get("cpus", 0),
            memory=resolved.get("memory", 0),
            allowed_nodes=resolved_allowed_nodes,
            pod_spec=pod_spec,
        )
        result["state"] = state.name
        result["profile"] = state.resources
        checks.append(result)
    return checks


def _skypilot_allowed_nodes(
    *,
    sky_bin: str,
    config_path: Path | None,
    isolated_config_dir: Path | None,
) -> tuple[str, ...]:
    """Read the exact SkyPilot node-affinity allowlist used for submission."""

    import yaml

    from npa.orchestration.skypilot._bin import resolve_config

    resolved = resolve_config(
        sky_bin=sky_bin or None,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    if resolved.global_config_path is None:
        return ()
    try:
        document = yaml.safe_load(resolved.global_config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            "could not read the selected SkyPilot config for allowed_nodes preflight"
        ) from exc
    kubernetes = document.get("kubernetes") if isinstance(document, Mapping) else None
    raw = kubernetes.get("allowed_nodes") if isinstance(kubernetes, Mapping) else None
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw or not all(
        isinstance(name, str) and name.strip() for name in raw
    ):
        raise RuntimeError(
            "SkyPilot kubernetes.allowed_nodes must be a non-empty list of node names"
        )
    names = tuple(dict.fromkeys(name.strip() for name in raw))
    return names


def _transfer_control_modalities(spec, *, run_id: str) -> set[str]:
    """Resolve each transfer state's exact state-local checkpoint modality."""

    from npa.orchestration.npa_workflow.tokens import resolve_tokens

    tool_refs = {
        "workbench.cosmos2.transfer_execute",
        "workbench.cosmos2.transfer_conditioned_execute",
    }
    modalities: set[str] = set()
    for state in spec.states.values():
        if state.tool_ref not in tool_refs:
            continue
        effective = dict(spec.config)
        for key, value in state.params.items():
            effective[key] = (
                resolve_tokens(
                    value,
                    config=spec.config,
                    run={"id": run_id},
                )
                if isinstance(value, str)
                else value
            )
        modalities.add(str(effective.get("augment_control") or "edge").strip())
    return modalities


def _record_workflow_submit_failure(operation, exc: BaseException) -> None:  # noqa: ANN001
    """Keep recovery only when this transaction may have issued a launch.

    A failed initial reconciliation has ``launch_sequence == 0``: NPA never
    called ``sky jobs launch`` and therefore owns no workflow resource to
    recover.  Marking that journal recovery-required blocks unrelated safe
    project operations forever, even though no mutation occurred.
    """

    transaction = getattr(exc, "transaction", None)
    launch_sequence = getattr(transaction, "launch_sequence", None)
    if launch_sequence == 0:
        operation.record_rollback(
            attempted=False,
            completed=True,
            removed=[],
            preserved=[],
            outcomes=[],
        )
        operation.transition(
            "rolled-back",
            error=str(exc),
            details={"error_type": type(exc).__name__, "launch_attempted": False},
        )
        return
    operation.transition("recovery-required", error=str(exc))


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


def _parse_image_overrides(items: list[str]) -> dict[str, str]:
    """Parse repeatable exact toolRef image overrides without silent replacement."""

    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                "Invalid --image-override/--tool-image format. Use TOOL_REF=IMAGE."
            )
        tool_ref, image_ref = (part.strip() for part in item.split("=", 1))
        if (
            not tool_ref
            or tool_ref == "*"
            or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", tool_ref)
            or not image_ref
        ):
            raise ValueError(
                "Invalid --image-override/--tool-image format. Use TOOL_REF=IMAGE; "
                "use --image for a global fallback."
            )
        if tool_ref in overrides:
            raise ValueError(
                f"Duplicate --image-override for exact tool ref {tool_ref!r}."
            )
        overrides[tool_ref] = (
            "" if image_ref.lower() in {"none", "default", "-"} else image_ref
        )
    return overrides


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


def _is_paidf_workflow_spec(yaml_path: Path) -> bool:
    """Identify the one workflow whose submit command owns starter preparation."""

    from npa.orchestration.npa_workflow.errors import NpaWorkflowError
    from npa.orchestration.npa_workflow.run_state import PAIDF_WORKFLOW_NAME
    from npa.orchestration.npa_workflow.spec import load_spec

    try:
        return load_spec(yaml_path).name == PAIDF_WORKFLOW_NAME
    except NpaWorkflowError:
        return False


def _is_truthy_submit_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _stage_npa_src_for_submit(
    spec_config: dict,
    *,
    s3_bucket: str = "",
    s3_endpoint: str = "",
    credential_values: dict[str, str] | None = None,
    project: str = "",
    run_id: str = "workflow",
    force: bool = False,
) -> str:
    """Upload once, then durably record the exact ``NPA_SRC_S3_URI``."""
    from npa.clients.config import ConfigError, persist_workflow_src_s3_uri
    from npa.orchestration.npa_workflow.src_staging import (
        SrcStagingError,
        stage_npa_source,
    )

    bucket = str(s3_bucket or spec_config.get("bucket", "") or "").strip()
    if _is_placeholder_bucket(bucket):
        _fail(
            "--stage-src needs a real bucket. Pass --var bucket=<your-bucket> "
            "(or --s3-bucket <your-bucket>)."
        )
        return ""
    try:
        uri = stage_npa_source(
            bucket=bucket,
            endpoint_url=s3_endpoint,
            aws_access_key_id=(credential_values or {}).get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=(credential_values or {}).get(
                "AWS_SECRET_ACCESS_KEY", ""
            ),
            on_status=lambda message: typer.echo(f"  {message}", err=True),
            force=force,
        )
        # This project-level cache is content-addressed and safe to update before
        # the run exists.  Do not create a run submission ledger here: upload
        # failure must leave no evidence that a managed job was reserved.
        persist_workflow_src_s3_uri(uri, project or None)
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


def _paidf_kubernetes_prerequisites_for_submit(
    context: str,
) -> list[tuple[str, str]]:
    """Run PAIDF's placement check with the exact submit kube context."""

    from npa.clients.kube import run_kubectl
    from npa.orchestration.npa_workflow.paidf_preflight import (
        kubernetes_prerequisites,
    )

    kubeconfig = os.environ.get("KUBECONFIG", "")

    def _run(args: list[str]):
        return run_kubectl(
            args,
            context=context,
            kubeconfig=kubeconfig,
            timeout=30,
        )

    return kubernetes_prerequisites(runner=_run)


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
        for entry in data.get("contexts") or []:
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
        entry
        for entry in current.split(os.pathsep)
        if entry.strip() and entry != str(path)
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
    requires_npa_source: bool = True,
    source_staging_planned: bool = False,
    checkpoint_access_error: str = "",
    probe_storage: bool = True,
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

    if checkpoint_access_error:
        missing.append(
            (
                checkpoint_access_error,
                "provide a caller-owned HF_TOKEN that has gated access to the "
                "selected nvidia/Cosmos-Transfer2.5-2B checkpoint; token presence "
                "does not itself record license consent",
            )
        )

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
    if (
        requires_npa_source
        and not image_pins_tasks
        and not resolve_src_s3_uri()
        and not source_staging_planned
    ):
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

    if probe_storage and not plan_only:
        missing.extend(
            _submit_storage_prerequisites(
                spec_config,
                requires_s3=requires_s3,
                s3_endpoint=s3_endpoint,
                s3_access_key_id=s3_access_key_id,
                s3_secret_access_key=s3_secret_access_key,
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


def _submit_storage_prerequisites(
    spec_config: Mapping[str, Any],
    *,
    requires_s3: bool,
    s3_endpoint: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
) -> list[tuple[str, str]]:
    """Run the cleaned writable-storage probe at its explicit mutation boundary."""

    bucket = str((spec_config or {}).get("bucket", "") or "")
    if not requires_s3 or _is_placeholder_bucket(bucket):
        return []
    from npa.clients.storage_validation import (
        StorageCapabilityProfile,
        probe_storage_write,
    )

    probe = probe_storage_write(
        bucket=bucket,
        endpoint_url=s3_endpoint,
        access_key_id=s3_access_key_id,
        secret_access_key=s3_secret_access_key,
        profile=StorageCapabilityProfile.WORKFLOW_SUBMISSION,
    )
    if probe.ok:
        return []
    return [
        (
            f"writable S3 for this workflow ({probe.summary})",
            "run `npa provision-if-absent --project <alias> --skip-k8s`, then "
            "retry; this append-only preflight uses a unique object and does not "
            "require DeleteObject",
        )
    ]


def _fail_missing_prerequisites(
    yaml_path: Path, missing: list[tuple[str, str]]
) -> None:
    lines = [f"Cannot submit {yaml_path.name}: missing prerequisites:"]
    for item, remedy in missing:
        lines.append(f"  - {item}")
        lines.append(f"      fix: {remedy}")
    lines.append(
        "  (--skip-preflight skips only optional environment diagnostics; "
        "model authorization and deterministic safety contracts remain mandatory)"
    )
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
    from npa.orchestration.skypilot.workflow_state import (
        WorkflowS3Config,
        parse_s3_uri,
        resolve_workflow_s3_config,
    )

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


def _latest_runtime_wave_states(
    runtime_waves: list[dict[str, object]],
) -> set[str]:
    """Return statuses from the authoritative latest attempt of each wave.

    Runtime history is append-only. A failed transport or workload attempt remains
    in that history after an explicit retry succeeds, so treating every historical
    row as current makes a truthful terminal manifest appear contradictory. The
    stable wave key identifies one logical wave; attempt number and append order
    select its latest evidence. Legacy rows without a key remain independent so
    ambiguous failure evidence continues to fail closed.
    """

    latest: dict[str, tuple[int, int, str]] = {}
    for position, wave in enumerate(runtime_waves):
        key = str(wave.get("key") or "").strip() or f"__legacy_{position}"
        try:
            attempt = int(str(wave.get("attempt") or 1))
        except (TypeError, ValueError):
            attempt = 1
        status = str(wave.get("status") or "").upper()
        candidate = (attempt, position, status)
        previous = latest.get(key)
        if previous is None or candidate[:2] >= previous[:2]:
            latest[key] = candidate
    return {status for _attempt, _position, status in latest.values() if status}


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
    cached: bool = False,
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
        allow_local_not_submitted=True,
    )
    from npa.verification import (
        CACHED,
        VERIFIED,
        VERIFICATION_UNAVAILABLE,
        apply_verification,
        sanitize_reason,
        utc_now as verification_now,
    )

    attempted_at = verification_now()
    retry_command = f"npa workbench workflow status {_display_run_id(run_id)}" + (
        f" --project {project}" if project else ""
    )
    if resolution.not_submitted:
        payload = {
            "run_id": resolution.run_id,
            "workflow_name": resolution.workflow_name,
            "status": "NOT_SUBMITTED",
            "lifecycle_state": "PLAN_ONLY",
            "submission_state": "NOT_SUBMITTED",
            "manifest_state": "absent",
            "manifest_pending": False,
            "resolution_source": resolution.source,
            "resolution_checks": resolution.checks_payload(),
            "live_status": "",
            "sky_job_id": "",
            "run_prefix_uri": "",
            "manifest_uri": "",
            "stages": {},
            "diagnostics": [
                "Current-schema durable local evidence proves submission never began."
            ],
        }
        return apply_verification(
            payload,
            status=VERIFIED,
            target=resolution.run_id,
            last_known_state="NOT_SUBMITTED",
            last_known_at=str(resolution.receipt.get("updated_at") or ""),
            last_known_source="durable_submission_receipt",
            retry_command=retry_command,
            attempted_at=attempted_at,
        )
    if not resolution.found:
        payload = {
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
        if resolution.conclusively_absent:
            verified_absent = apply_verification(
                payload,
                status=VERIFIED,
                target=resolution.manifest_uri or resolution.run_prefix_uri,
                last_known_state="NOT_FOUND",
                last_known_source="authoritative_resolution",
                retry_command=retry_command,
                attempted_at=attempted_at,
            )
            # Preserve the established discriminator while the additive envelope
            # carries the new live-verification model.
            verified_absent["verification"] = "conclusively_absent"
            return verified_absent
        return apply_verification(
            payload,
            status=VERIFICATION_UNAVAILABLE,
            target=resolution.manifest_uri or resolution.run_prefix_uri,
            last_known_state="UNKNOWN",
            last_known_source="resolution_checks",
            reason="; ".join(resolution_diagnostics(resolution)),
            retry_command=retry_command,
            attempted_at=attempted_at,
        )

    if resolution.durable_terminal_state:
        payload = {
            "run_id": resolution.run_id,
            "workflow_name": resolution.workflow_name,
            "status": resolution.durable_terminal_state,
            "submission_state": resolution.durable_terminal_state,
            "manifest_state": "verification_unavailable",
            "manifest_pending": False,
            "artifact_verification": "unavailable",
            "resolution_source": resolution.source,
            "resolution_checks": resolution.checks_payload(),
            "live_status": "",
            "sky_job_id": resolution.job_id,
            "run_prefix_uri": resolution.run_prefix_uri,
            "manifest_uri": resolution.manifest_uri,
            "stages": {},
            "diagnostics": [
                "Durable project/run terminal evidence has precedence; storage artifact "
                "verification is unavailable and does not regress terminal lifecycle truth."
            ],
        }
        return apply_verification(
            payload,
            status=VERIFIED,
            target=resolution.run_id,
            last_known_state=resolution.durable_terminal_state,
            last_known_source="project_run_terminal_ledger",
            retry_command=retry_command,
            attempted_at=attempted_at,
        )

    state = resolution.state
    manifest = resolution.manifest
    if manifest is None:
        payload = _manifest_pending_status(
            resolution,
            project=project,
            sky_bin=sky_bin,
            startup_failure_threshold=startup_failure_threshold,
            cached=cached,
        )
        return payload
    assert state is not None
    if manifest.get("schema_version") == "npa.workflow.run.v1":
        from npa.orchestration.npa_workflow.run_state import (
            RunManifest,
            build_actionable_run_status,
            reconstruct_stage_job_attribution,
            reconcile_submitted_manifest,
        )

        run_manifest = RunManifest.from_dict(manifest)
        runtime_waves = [
            dict(item)
            for item in resolution.runtime_state.get("waves") or []
            if isinstance(item, dict)
        ]
        runtime_stages = [
            dict(item)
            for item in resolution.runtime_state.get("stages") or []
            if isinstance(item, dict)
        ]
        for step in run_manifest.steps:
            name = str(step.get("state") or "")
            candidates = [
                item for item in runtime_stages if str(item.get("stage") or "") == name
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda item: int(item.get("attempt") or 1))
            latest = candidates[-1]
            if latest.get("last_heartbeat_at"):
                step["last_heartbeat_at"] = latest["last_heartbeat_at"]
            if latest.get("last_observed_at"):
                step["last_updated_at"] = latest["last_observed_at"]
            if latest.get("pending_reason"):
                step["pending_reason"] = latest["pending_reason"]
        # A runtime ledger proves one job per wave.  Its latest discovered ID must
        # never be promoted to the root and broadcast across every logical stage.
        if not run_manifest.sky_job_id and resolution.job_id and not runtime_waves:
            run_manifest.sky_job_id = resolution.job_id
        live_status = ""
        task_rows: list[dict[str, object]] = []
        job_observations: dict[str, dict[str, object]] = {}
        controller_output = ""
        diagnostics: list[str] = []
        verification_errors: list[str] = []
        attribution = reconstruct_stage_job_attribution(
            run_manifest, runtime_waves=runtime_waves
        )
        job_ids = sorted(
            {
                str(item.get("managed_job_id") or "")
                for item in attribution.values()
                if str(item.get("managed_job_id") or "")
            },
            key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
        )
        if (
            run_manifest.sky_job_id
            and not runtime_waves
            and run_manifest.sky_job_id not in job_ids
        ):
            job_ids.append(run_manifest.sky_job_id)
        for managed_job_id in [] if cached else job_ids:
            try:
                live = workflow_status(managed_job_id, sky_bin=sky_bin or None)
                if live.error:
                    safe_error = sanitize_reason(live.error)
                    diagnostics.append(
                        f"SkyPilot status for managed job {managed_job_id} is "
                        f"unavailable; using durable stage evidence ({safe_error})."
                    )
                    observed_status = ""
                    verification_errors.append(safe_error)
                elif str(live.status or "").upper() in {"", "UNKNOWN"}:
                    observed_status = ""
                    verification_errors.append(
                        f"managed job {managed_job_id} was not present in a parseable live queue response"
                    )
                else:
                    observed_status = live.status
                observed_rows = workflow_task_statuses(
                    managed_job_id, sky_bin=sky_bin or None
                )
                job_observations[managed_job_id] = {
                    "status": observed_status,
                    "task_rows": observed_rows,
                    "error": live.error,
                    "observed_at": attempted_at,
                }
                if managed_job_id == run_manifest.sky_job_id and not runtime_waves:
                    task_rows = observed_rows
                if str(observed_status or run_manifest.status).upper() not in {
                    "SUCCEEDED",
                    "CANCELLED",
                    "ABSENT",
                } and not str(
                    observed_status or run_manifest.status
                ).upper().startswith("FAILED"):
                    controller_logs = workflow_controller_logs(
                        managed_job_id,
                        sky_bin=sky_bin or None,
                    )
                    output = "\n".join(
                        sanitize_reason(line)
                        for line in "\n".join(
                            item
                            for item in (controller_logs.stdout, controller_logs.stderr)
                            if item
                        ).splitlines()
                    )[:4000]
                    if output:
                        controller_output += (
                            "\n" if controller_output else ""
                        ) + output
                    if controller_logs.returncode != 0:
                        diagnostics.append(
                            "SkyPilot controller logs are unavailable; startup failure "
                            f"classification is incomplete for job {managed_job_id} "
                            f"({output.strip() or 'no detail'})."
                        )
            except Exception as exc:  # noqa: BLE001 - persisted status remains useful
                safe_error = sanitize_reason(f"{type(exc).__name__}: {exc}")
                verification_errors.append(safe_error)
                diagnostics.append(
                    f"SkyPilot status for managed job {managed_job_id} is unavailable; "
                    f"showing durable stage evidence. Diagnostic: {safe_error}"
                )
        observed_states = [
            str(item.get("status") or "").upper()
            for item in job_observations.values()
            if str(item.get("status") or "").strip()
        ]
        if any(item.startswith("FAILED") for item in observed_states):
            live_status = "FAILED"
        elif any(item == "CANCELLED" for item in observed_states):
            live_status = "CANCELLED"
        elif observed_states and all(item == "SUCCEEDED" for item in observed_states):
            live_status = "SUCCEEDED"
        else:
            live_status = next(
                (
                    item
                    for item in reversed(observed_states)
                    if item in {"RUNNING", "STARTING", "PENDING", "RECOVERING"}
                ),
                "",
            )
        if not runtime_waves:
            reconcile_submitted_manifest(
                run_manifest, live_status=live_status, task_rows=task_rows
            )
        run_payload = build_actionable_run_status(
            run_manifest,
            live_status=live_status,
            task_rows=task_rows,
            runtime_waves=runtime_waves,
            job_observations=job_observations,
            controller_output=controller_output,
            project=project or state.project,
            failure_threshold=startup_failure_threshold,
        )
        manifest_terminal = str(run_manifest.status or "").upper()
        runtime_terminal_states = _latest_runtime_wave_states(runtime_waves)
        if (
            manifest_terminal == "SUCCEEDED"
            and runtime_terminal_states
            and any(
                state in {"FAILED", "CANCELLED"} for state in runtime_terminal_states
            )
        ):
            run_payload["status"] = "EVIDENCE_INCONSISTENT"
            run_payload["submission_state"] = "EVIDENCE_INCONSISTENT"
            run_payload.setdefault("diagnostics", []).append(
                "Terminal manifest success conflicts with the latest authoritative "
                "attempt for at least one runtime wave."
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
        blockers = [
            blocker
            for managed_job_id, observation in job_observations.items()
            for blocker in _stalled_job_blockers(
                managed_job_id,
                str(observation.get("status") or ""),
                sky_bin=sky_bin,
            )
        ]
        if blockers:
            run_payload["blockers"] = blockers
            active_name = str(run_payload.get("active_stage_name") or "")
            stage_payload = (run_payload.get("stages") or {}).get(active_name)
            if not isinstance(stage_payload, dict):
                stage_payload = next(
                    (
                        item
                        for item in (run_payload.get("stages") or {}).values()
                        if isinstance(item, dict)
                        and str(item.get("workflow_state") or "") == active_name
                    ),
                    None,
                )
            if isinstance(stage_payload, dict):
                stage_payload["pending_reason"] = blockers[0]
        last_known = str(run_payload.get("status") or run_manifest.status or "UNKNOWN")
        last_known_at = str(
            run_payload.get("last_observed_at") or run_manifest.updated_at or ""
        )
        if cached:
            return apply_verification(
                run_payload,
                status=CACHED,
                target=", ".join(job_ids) or state.uri,
                last_known_state=last_known,
                last_known_at=last_known_at,
                last_known_source="runtime_ledger_or_manifest",
                reason="live controller query intentionally skipped (--cached)",
                retry_command=retry_command,
                attempted_at=attempted_at,
            )
        if verification_errors:
            run_payload.setdefault("diagnostics", []).append(
                "Current live verification failed; last-known stage evidence is retained."
            )
            return apply_verification(
                run_payload,
                status=VERIFICATION_UNAVAILABLE,
                target=", ".join(job_ids) or state.uri,
                last_known_state=last_known,
                last_known_at=last_known_at,
                last_known_source="runtime_ledger_or_manifest",
                reason="; ".join(verification_errors),
                retry_command=retry_command,
                attempted_at=attempted_at,
            )
        return apply_verification(
            run_payload,
            status=VERIFIED,
            target=", ".join(job_ids) or state.uri,
            last_known_state=last_known,
            last_known_at=last_known_at,
            last_known_source="live_scheduler" if job_ids else "authoritative_manifest",
            retry_command=retry_command,
            attempted_at=attempted_at,
        )
    stages: dict[str, dict[str, object]] = {}
    for stage, info in (manifest.get("stages", {}) or {}).items():
        stage_info = dict(info) if isinstance(info, dict) else {"name": str(stage)}
        stage_status = read_stage_status(state, str(stage))
        if stage_status:
            stage_info.update(stage_status)
        stages[str(stage)] = stage_info

    job_id = str(manifest.get("sky_job_id") or "")
    live_status = ""
    legacy_verification_errors: list[str] = []
    if job_id and not cached:
        try:
            live = workflow_status(job_id, sky_bin=sky_bin or None)
            if live.error:
                legacy_verification_errors.append(str(live.error))
            elif str(live.status or "").upper() in {"", "UNKNOWN"}:
                legacy_verification_errors.append(
                    f"managed job {job_id} was not present in a parseable live queue response"
                )
            else:
                live_status = live.status
        except Exception as exc:
            legacy_verification_errors.append(f"{type(exc).__name__}: {exc}")
            live_status = ""
    status = _aggregate_stage_status(stages, live_status)
    legacy_payload: dict[str, object] = {
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
        legacy_payload["blockers"] = blockers
    last_known = str(status or manifest.get("status") or "UNKNOWN")
    if cached:
        verification_status = CACHED
        reason = "live controller query intentionally skipped (--cached)"
    elif legacy_verification_errors:
        verification_status = VERIFICATION_UNAVAILABLE
        reason = "; ".join(legacy_verification_errors)
    else:
        verification_status = VERIFIED
        reason = ""
    return apply_verification(
        legacy_payload,
        status=verification_status,
        target=job_id or state.uri,
        last_known_state=last_known,
        last_known_at=str(manifest.get("updated_at") or ""),
        last_known_source="legacy_manifest",
        reason=reason,
        retry_command=retry_command,
        attempted_at=attempted_at,
    )


def _manifest_pending_status(
    resolution,
    *,
    project: str,
    sky_bin: str,
    startup_failure_threshold: int,
    cached: bool = False,
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
    from npa.verification import sanitize_reason

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
                        "job_id": str(wave.get("job_id") or ""),
                        "job_name": str(wave.get("job_name") or ""),
                        "attempt": int(wave.get("attempt") or 1),
                        "wave_key": str(wave.get("key") or ""),
                        "sky_status": str(wave.get("sky_status") or ""),
                        "resources_profile": {},
                    }
                )
    runtime_job_ids = {
        str(wave.get("job_id") or wave.get("sky_job_id") or "").strip()
        for wave in runtime_waves
        if str(wave.get("job_id") or wave.get("sky_job_id") or "").strip()
    }
    job_id = str(resolution.job_id or "").strip()
    if runtime_waves and job_id not in runtime_job_ids:
        # A resolver's legacy/latest root ID is not stage evidence. Prefer only
        # the exact active wave identity; if it is missing, keep status unknown.
        job_id = str(
            active_wave.get("job_id") or active_wave.get("sky_job_id") or ""
        ).strip()
    live_status = ""
    task_rows: list[dict[str, object]] = []
    controller_output = ""
    diagnostics = resolution_diagnostics(resolution)
    verification_errors: list[str] = []
    if (
        not cached
        and resolution.managed_job is not None
        and resolution.managed_job.outcome == "found"
        and str(resolution.job_id or "").strip() == job_id
    ):
        live_status = resolution.managed_job.status
        task_rows = [dict(item) for item in resolution.managed_job.task_rows]
    elif job_id and not cached:
        try:
            live = workflow_status(job_id, sky_bin=sky_bin or None)
            if live.error:
                safe_error = sanitize_reason(live.error)
                diagnostics.append(
                    "SkyPilot status is unavailable; the run remains receipt-proven "
                    f"({safe_error})."
                )
                verification_errors.append(safe_error)
            elif str(live.status or "").upper() in {"", "UNKNOWN"}:
                verification_errors.append(
                    f"managed job {job_id} was not present in a parseable live queue response"
                )
            else:
                live_status = live.status
            task_rows = workflow_task_statuses(job_id, sky_bin=sky_bin or None)
        except Exception as exc:  # noqa: BLE001 - durable evidence still proves the run
            safe_error = sanitize_reason(f"{type(exc).__name__}: {exc}")
            verification_errors.append(safe_error)
            diagnostics.append(
                "SkyPilot status is unavailable; the run remains receipt-proven. "
                f"Diagnostic: {safe_error}"
            )
    if not task_rows and active_wave:
        task_rows = [
            dict(item)
            for item in active_wave.get("tasks") or []
            if isinstance(item, dict)
        ]
    if not steps and task_rows:
        rows = sorted(task_rows, key=lambda item: int(str(item.get("task_id") or 0)))
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
        sky_job_id=(job_id if not runtime_waves else ""),
        steps=steps,
        updated_at=str(resolution.receipt.get("updated_at") or ""),
    )
    if (
        not cached
        and job_id
        and str(live_status).upper()
        not in {
            "SUCCEEDED",
            "CANCELLED",
        }
        and not str(live_status).upper().startswith("FAILED")
    ):
        try:
            controller_logs = workflow_controller_logs(job_id, sky_bin=sky_bin or None)
            controller_output = "\n".join(
                sanitize_reason(line)
                for line in "\n".join(
                    item
                    for item in (controller_logs.stdout, controller_logs.stderr)
                    if item
                ).splitlines()
            )[:4000]
            if controller_logs.returncode != 0:
                diagnostics.append(
                    "SkyPilot controller logs are unavailable; startup failure "
                    f"classification is incomplete ({controller_output.strip() or 'no detail'})."
                )
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                "SkyPilot controller logs are unavailable: " + sanitize_reason(exc)
            )
    if not runtime_waves:
        reconcile_submitted_manifest(
            pending_manifest, live_status=live_status, task_rows=task_rows
        )
    payload = build_actionable_run_status(
        pending_manifest,
        live_status=live_status,
        task_rows=task_rows,
        runtime_waves=runtime_waves,
        job_observations=(
            {
                job_id: {
                    "status": live_status,
                    "task_rows": task_rows,
                    "error": "",
                }
            }
            if job_id
            else {}
        ),
        controller_output=controller_output,
        project=project,
        failure_threshold=startup_failure_threshold,
    )
    launch_raw = resolution.receipt.get("launch")
    launch_record = launch_raw if isinstance(launch_raw, dict) else {}
    launch_status = str(launch_record.get("status") or "").lower()
    durable_terminal_waves = bool(runtime_waves) and all(
        str(item.get("status") or "").lower()
        in {"succeeded", "failed", "cancelled", "quality_rejected", "blocked"}
        for item in runtime_waves
    )
    current_terminal_success = (
        str(resolution.runtime_state.get("schema_version") or "")
        == "npa.workflow.runtime.v1"
        and str(resolution.runtime_state.get("status") or "").lower() == "succeeded"
        and bool(runtime_waves)
        and all(
            str(item.get("status") or "").lower() == "succeeded"
            for item in runtime_waves
        )
    )
    if not job_id and not task_rows and not durable_terminal_waves:
        if launch_status in {"launching", "submitted", "completed"}:
            payload["status"] = "PARTIAL_SUBMISSION"
            payload["submission_state"] = "SUBMISSION_UNVERIFIED"
        else:
            payload["status"] = "NOT_SUBMITTED"
            payload["lifecycle_state"] = "PLAN_ONLY"
            payload["submission_state"] = "NOT_SUBMITTED"
    payload.update(
        {
            "run_prefix_uri": resolution.run_prefix_uri,
            "manifest_uri": resolution.manifest_uri,
            "manifest_state": (
                "pending"
                if job_id or task_rows or current_terminal_success
                else "absent"
            ),
            "manifest_pending": bool(job_id or task_rows or current_terminal_success),
            "resolution_source": resolution.source,
            "resolution_checks": resolution.checks_payload(),
            "verification": "found",
            "diagnostics": [
                (
                    "Run found; final workflow manifest is pending."
                    if job_id or task_rows or current_terminal_success
                    else "Run identity exists, but no managed-job submission evidence exists."
                ),
                *diagnostics,
            ],
        }
    )
    blockers = _stalled_job_blockers(job_id, live_status, sky_bin=sky_bin)
    if blockers:
        payload["blockers"] = blockers
    from npa.verification import (
        CACHED,
        VERIFIED,
        VERIFICATION_UNAVAILABLE,
        apply_verification,
        utc_now as verification_now,
    )

    retry_command = f"npa workbench workflow status {resolution.run_id}" + (
        f" --project {project}" if project else ""
    )
    if cached:
        verification_status = CACHED
        reason = "live controller query intentionally skipped (--cached)"
    elif verification_errors:
        verification_status = VERIFICATION_UNAVAILABLE
        reason = "; ".join(verification_errors)
    elif str(payload.get("status") or "").upper() == "NOT_SUBMITTED":
        # Exact planning/reservation evidence plus the absence of a submission
        # receipt is the state being reported, not a failed live job query.
        verification_status = VERIFIED
        reason = ""
    elif job_id or resolution.managed_job is not None or current_terminal_success:
        verification_status = VERIFIED
        reason = ""
    else:
        verification_status = VERIFICATION_UNAVAILABLE
        reason = "no exact managed-job identity is available for live verification"
    result = apply_verification(
        payload,
        status=verification_status,
        target=job_id or resolution.manifest_uri,
        last_known_state=str(payload.get("status") or "NOT_SUBMITTED"),
        last_known_at=str(resolution.receipt.get("updated_at") or ""),
        last_known_source=resolution.source or "submission_receipt",
        reason=reason,
        retry_command=retry_command,
        attempted_at=verification_now(),
    )
    if str(payload.get("status") or "").upper() == "PARTIAL_SUBMISSION":
        # Preserve the distinct automation state while the verification
        # envelope explains why its managed-job identity is unavailable.
        result["status"] = "PARTIAL_SUBMISSION"
    return result


def _stalled_job_blockers(
    job_id: str, live_status: str, *, sky_bin: str = ""
) -> list[dict[str, object]]:
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
        [
            inspect_job_blockers(job_id=job_id, cluster_name=cluster)
            for cluster in clusters
        ]
        if clusters
        else [inspect_job_blockers(job_id=job_id)]
    )
    reported: list[dict[str, object]] = []
    for report in reports:
        if report.error:
            reported.append(
                {
                    "pod": "",
                    "reason": "DiagnosticsUnavailable",
                    "reason_code": report.error_code
                    or "KUBERNETES_DIAGNOSTICS_UNAVAILABLE",
                    "message": report.error,
                    "source": "kubernetes_api",
                    "observed_at": report.observed_at,
                    "live": False,
                    "remedy": (
                        f"npa workbench workflow status {job_id}; then retry "
                        f"npa workbench workflow logs {job_id} --cached"
                    ),
                }
            )
        for blocker in report.blockers:
            reported.append(
                {
                    "pod": blocker.pod,
                    "reason": blocker.reason,
                    "reason_code": blocker.reason_code,
                    "message": blocker.message,
                    "source": blocker.source,
                    "observed_at": blocker.observed_at,
                    "live": blocker.live,
                    "remedy": report.remedy(),
                }
            )
        # A job whose nodes were reclaimed has no pod-level reason at all.
        for node in report.unready_nodes:
            reported.append(
                {
                    "pod": node,
                    "reason": "NodeNotReady",
                    "reason_code": "NODE_NOT_READY",
                    "message": "the node this job needs is not Ready",
                    "source": "kubernetes_node_condition",
                    "observed_at": report.observed_at,
                    "live": True,
                    "remedy": report.remedy(),
                }
            )
    return reported


def _aggregate_stage_status(
    stages: dict[str, dict[str, object]], live_status: str
) -> str:
    stage_states = [str(info.get("state") or "").upper() for info in stages.values()]
    if any(
        state.startswith("FAILED") or state in {"CANCELLED", "BLOCKED"}
        for state in stage_states
    ):
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
        in {
            "CANCELLED",
            "BLOCKED",
            "NOT_SUBMITTED",
            "NOT_FOUND",
            "VERIFICATION_UNAVAILABLE",
        }
    )


def _emit_workflow_status(
    result: dict[str, object], output_format: OutputFormat
) -> None:
    if output_format == OutputFormat.json:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
        return
    verification_status = str(result.get("verification_status") or "")
    if verification_status in {"VERIFICATION_UNAVAILABLE", "CACHED"}:
        typer.echo(verification_status)
        last_known = result.get("last_known")
        if isinstance(last_known, dict):
            typer.echo(
                f"last-known state: {last_known.get('state') or 'UNKNOWN'} "
                f"(observed_at={last_known.get('observed_at') or 'unknown'}, "
                f"source={last_known.get('source') or 'unknown'})"
            )
        verification = result.get("live_verification")
        if isinstance(verification, dict):
            if verification.get("reason"):
                typer.echo(
                    f"verification cause [{verification.get('error_code') or 'CACHED'}]: "
                    f"{verification.get('reason')}"
                )
            if verification.get("retry_command"):
                typer.echo(f"retry: {verification.get('retry_command')}")
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
        typer.echo(
            f"heartbeat_age_seconds: {result.get('heartbeat_age_seconds')} "
            f"stale={bool(result.get('heartbeat_stale'))}"
        )
    if result.get("last_observed_at"):
        typer.echo(f"last_observed_at: {result.get('last_observed_at')}")
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
            state = (
                info.get("state", "UNKNOWN") if isinstance(info, dict) else "UNKNOWN"
            )
            tier = info.get("tier", "") if isinstance(info, dict) else ""
            accelerator = (
                info.get("requested_accelerators", "") if isinstance(info, dict) else ""
            )
            suffix_parts = [
                part
                for part in (tier, f"requested={accelerator}" if accelerator else "")
                if part
            ]
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
    n_envs: int = typer.Option(
        4096, "--n-envs", help="Parallel environments for simulation."
    ),
    remote: bool = typer.Option(
        False,
        "--remote/--local",
        help="Execute on remote VMs via SSH (requires --s3-bucket).",
    ),
    s3_bucket: str = typer.Option(
        "",
        "--s3-bucket",
        help="S3 bucket URI for artifact storage (required for --remote).",
    ),
    sim_workbench: str = typer.Option(
        "", "--sim-workbench", help="Workbench name for sim VM (Genesis stages)."
    ),
    train_workbench: str = typer.Option(
        "",
        "--train-workbench",
        help="Workbench name for training VM (LeRobot stages). Defaults to sim workbench.",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian,
        "--action-space",
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
        console.print(
            f"  sim_workbench={sim_workbench or '(default)'}  train_workbench={train_workbench or '(same as sim)'}"
        )

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
@intent_boundary(OperationIntent.OBSERVE)
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
    cached: bool = typer.Option(
        False,
        "--cached",
        help=(
            "Inspect last-known persisted state without contacting the live controller. "
            "Output is explicitly marked CACHED and is not automation-trustworthy."
        ),
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
    json_output: bool = typer.Option(
        False, "--json", help="Shortcut for --output-format json."
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="Output format."
    ),
) -> None:
    """Check the status of a workflow run."""
    resolved_run_id = _display_run_id(run_id)
    if cached and watch:
        _fail("--cached cannot be combined with --watch")

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
                    cached=cached,
                )
                try:
                    from npa.orchestration.npa_workflow.first_run_state import (
                        update_run_observation,
                    )

                    last_known = result.get("last_known")
                    update_run_observation(
                        project=project,
                        workflow_identity=str(
                            result.get("workflow_name") or "workflow"
                        ),
                        run_id=str(result.get("run_id") or resolved_run_id),
                        last_known_state=str(
                            last_known.get("state")
                            if isinstance(last_known, dict)
                            else result.get("status")
                        ),
                        verification_status=str(
                            result.get("verification_status") or "UNVERIFIED"
                        ),
                    )
                except Exception:  # noqa: BLE001 - status output must not depend on local UX metadata
                    logger.debug(
                        "Could not update scoped first-run observation metadata",
                        exc_info=True,
                    )
                _emit_workflow_status(
                    result, OutputFormat.json if json_output else output_format
                )
                normalized = str(result.get("status") or "").upper()
                if normalized == "NOT_FOUND":
                    raise typer.Exit(code=1)
                if normalized == "VERIFICATION_UNAVAILABLE":
                    raise typer.Exit(code=2)
                if normalized in {"PARTIAL", "DEGRADED", "PARTIAL_SUBMISSION"}:
                    raise typer.Exit(code=3)
                if normalized in {"NOT_SUBMITTED", "PLAN_ONLY"}:
                    raise typer.Exit(code=4)
                if normalized.startswith("FAILED") or normalized in {
                    "CANCELLED",
                    "BLOCKED",
                }:
                    raise typer.Exit(code=1)
                if not watch or _workflow_status_is_terminal(
                    str(result.get("status", ""))
                ):
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
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Read persisted/object-storage log evidence only; never query the live controller.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the log-source contract as JSON."
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

            resolution = require_resolved_run(
                resolve_run(
                    run_id,
                    project=project,
                    workflow_s3_uri=workflow_s3_uri,
                    workflow_s3_prefix=workflow_s3_prefix,
                    s3_bucket=s3_bucket,
                    s3_endpoint=s3_endpoint,
                    sky_bin=sky_bin,
                )
            )
            state = resolution.state
            from npa.orchestration.skypilot.workflow_state import (
                read_stage_log,
                redact_text,
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
                safe_stdout = redact_text(live.stdout)
                safe_stderr = redact_text(live.stderr)
                if safe_stdout and not json_output:
                    typer.echo(safe_stdout, nl=False)
                if safe_stderr and not json_output:
                    typer.echo(safe_stderr, err=True, nl=False)
                if live.returncode != 0:
                    raise RuntimeError(
                        "run found with manifest pending, but SkyPilot logs are unavailable"
                    )
                return
            assert state is not None
            if manifest.get("schema_version") == "npa.workflow.run.v1":
                steps = [
                    item
                    for item in (manifest.get("steps") or [])
                    if isinstance(item, dict)
                ]
                runtime_stages = [
                    item
                    for item in resolution.runtime_state.get("stages") or []
                    if isinstance(item, dict)
                ]
                if not runtime_stages:
                    # Conservative migration for older ledgers: wave membership is
                    # known, but any missing per-task detail stays unknown.
                    for wave in resolution.runtime_state.get("waves") or []:
                        if not isinstance(wave, dict):
                            continue
                        for state_name in wave.get("states") or []:
                            runtime_stages.append(
                                {
                                    "stage": str(state_name),
                                    "attempt": int(wave.get("attempt") or 1),
                                    "managed_job_id": str(wave.get("job_id") or ""),
                                    "logical_state": str(
                                        wave.get("status") or "unknown"
                                    ),
                                    "provenance": "legacy_runtime_wave_reconstruction",
                                }
                            )
                available = list(
                    dict.fromkeys(
                        [str(item.get("state") or "") for item in steps]
                        + [str(item.get("stage") or "") for item in runtime_stages]
                    )
                )
                available = [item for item in available if item]
                if not selected_stage:
                    selected_stage = next(
                        (
                            str(item.get("state") or "")
                            for item in steps
                            if str(item.get("status") or "")
                            .upper()
                            .startswith("FAILED")
                        ),
                        available[0] if available else "",
                    )
                if selected_stage not in available:
                    raise RuntimeError(
                        f"stage {selected_stage!r} is not in the run manifest; "
                        f"available stages: {', '.join(available) or '<none>'}"
                    )
                stage_attempts = [
                    item
                    for item in runtime_stages
                    if str(item.get("stage") or "") == selected_stage
                ]
                stage_attempts.sort(key=lambda item: int(item.get("attempt") or 1))
                selected_attempt = stage_attempts[-1] if stage_attempts else {}
                job_id = str(selected_attempt.get("managed_job_id") or "")
                if not job_id and not resolution.runtime_state.get("waves"):
                    # Root job IDs are compatible only for the historical one-job
                    # manifest contract. Never broadcast one ID across runtime waves.
                    job_id = str(manifest.get("sky_job_id") or "")
                source_payload: dict[str, object] = {
                    "run_id": resolution.run_id,
                    "stage": selected_stage,
                    "attempt": int(selected_attempt.get("attempt") or 1),
                    "managed_job_id": job_id,
                    "manifest_state": "available",
                    "persisted_stages": available,
                    "stage_ledger_state": "available"
                    if runtime_stages
                    else "manifest_only",
                    "live_log_state": "not_attempted" if cached else "pending",
                    "cached_log_state": "unknown",
                    "provenance": str(selected_attempt.get("provenance") or "manifest"),
                }
                from npa.verification import (
                    CACHED,
                    VERIFIED,
                    VERIFICATION_UNAVAILABLE,
                    apply_verification,
                )

                last_known_state = str(
                    selected_attempt.get("logical_state")
                    or next(
                        (
                            item.get("status")
                            for item in steps
                            if str(item.get("state") or "") == selected_stage
                        ),
                        manifest.get("status") or "UNKNOWN",
                    )
                )
                last_known_at = str(
                    selected_attempt.get("last_observed_at")
                    or manifest.get("updated_at")
                    or ""
                )
                log_retry = (
                    f"npa workbench workflow logs {resolution.run_id} "
                    f"--stage {selected_stage}"
                    + (f" --project {project}" if project else "")
                )
                if cached:
                    cached_text = ""
                    try:
                        cached_text = read_stage_log(state, selected_stage)
                    except Exception:  # noqa: BLE001 - reported through the source contract
                        cached_text = ""
                    source_payload = apply_verification(
                        source_payload,
                        status=CACHED,
                        target=job_id or state.uri,
                        last_known_state=last_known_state,
                        last_known_at=last_known_at,
                        last_known_source="stage_ledger_or_manifest",
                        reason="live controller logs intentionally skipped (--cached)",
                        retry_command=log_retry,
                    )
                    source_payload["cached_log_state"] = (
                        "available" if cached_text else "unavailable"
                    )
                    source_payload["log"] = cached_text
                    if json_output:
                        typer.echo(json.dumps(source_payload, indent=2, sort_keys=True))
                    else:
                        typer.echo("CACHED")
                        typer.echo("manifest_state: available")
                        typer.echo(f"persisted stages: {', '.join(available)}")
                        typer.echo(
                            "cached/object-storage logs: "
                            + ("available" if cached_text else "unavailable")
                        )
                        if cached_text:
                            typer.echo(cached_text, nl=False)
                    return
                if not job_id:
                    reason = (
                        "no exact managed-job identity is recorded for this stage/attempt; "
                        "live logs cannot be attributed safely"
                    )
                    source_payload = apply_verification(
                        source_payload,
                        status=VERIFICATION_UNAVAILABLE,
                        target=state.uri,
                        last_known_state=last_known_state,
                        last_known_at=last_known_at,
                        last_known_source="stage_ledger_or_manifest",
                        reason=reason,
                        retry_command=f"npa workbench workflow status {resolution.run_id}"
                        + (f" --project {project}" if project else ""),
                    )
                    source_payload["live_log_state"] = "unavailable"
                    source_payload["error_code"] = "STAGE_JOB_ID_UNAVAILABLE"
                    live_verification = source_payload["live_verification"]
                    assert isinstance(live_verification, dict)
                    live_verification["error_code"] = "STAGE_JOB_ID_UNAVAILABLE"
                    live_verification["category"] = "ATTRIBUTION"
                    if json_output:
                        typer.echo(json.dumps(source_payload, indent=2, sort_keys=True))
                    else:
                        typer.echo("VERIFICATION_UNAVAILABLE")
                        typer.echo("manifest_state: available")
                        typer.echo(f"persisted stages: {', '.join(available)}")
                        typer.echo(f"cause: {reason}")
                        typer.echo(f"retry: {live_verification['retry_command']}")
                    raise typer.Exit(code=2)
                live = tail_live_job_logs(
                    sky_bin=_resolve_sky_bin(sky_bin),
                    job_id=job_id,
                    stage=selected_stage,
                    follow=follow,
                    timeout=86400 if follow else 300,
                )
                safe_stdout = redact_text(live.stdout)
                safe_stderr = redact_text(live.stderr)
                if safe_stdout and not json_output:
                    typer.echo(safe_stdout, nl=False)
                if safe_stderr and not json_output:
                    typer.echo(safe_stderr, err=True, nl=False)
                if live.returncode != 0:
                    from npa.verification import (
                        classify_verification_failure,
                        sanitize_reason,
                    )

                    reason = sanitize_reason(
                        safe_stderr or safe_stdout or "controller logs unavailable"
                    )
                    error_code, category = classify_verification_failure(reason)
                    source_payload = apply_verification(
                        source_payload,
                        status=VERIFICATION_UNAVAILABLE,
                        target=job_id,
                        last_known_state=last_known_state,
                        last_known_at=last_known_at,
                        last_known_source="stage_ledger_or_manifest",
                        reason=reason,
                        retry_command=log_retry,
                    )
                    source_payload.update(
                        {
                            "live_log_state": "unavailable",
                            "error_code": error_code,
                            "error_category": category,
                            "reason": reason,
                            "retry_command": log_retry,
                        }
                    )
                    if json_output:
                        typer.echo(json.dumps(source_payload, indent=2, sort_keys=True))
                    else:
                        typer.echo("VERIFICATION_UNAVAILABLE")
                        typer.echo("manifest_state: available")
                        typer.echo(f"persisted stages: {', '.join(available)}")
                        typer.echo(f"cause [{error_code}]: {reason}")
                        typer.echo(f"retry: {source_payload['retry_command']}")
                    raise typer.Exit(code=2)
                if json_output:
                    source_payload = apply_verification(
                        source_payload,
                        status=VERIFIED,
                        target=job_id,
                        last_known_state=last_known_state,
                        last_known_at=last_known_at,
                        last_known_source="stage_ledger_or_manifest",
                        retry_command=log_retry,
                    )
                    source_payload.update(
                        {"live_log_state": "available", "log": safe_stdout}
                    )
                    typer.echo(json.dumps(source_payload, indent=2, sort_keys=True))
                elif live.stdout:
                    # stdout was emitted above for the text contract.
                    pass
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
                safe_stdout = redact_text(live.stdout)
                safe_stderr = redact_text(live.stderr)
                if safe_stdout:
                    typer.echo(safe_stdout, nl=False)
                if safe_stderr:
                    typer.echo(safe_stderr, err=True, nl=False)
                if live.returncode == 0:
                    return
            typer.echo(read_stage_log(state, selected_stage), nl=False)
            return
        except typer.Exit:
            raise
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
    project: str = typer.Option(
        "", "--project", "-p", help="Project alias for S3 credentials."
    ),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact workflow prefix; takes precedence over receipts and discovery.",
    ),
    workflow_s3_prefix: str = typer.Option(
        "", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."
    ),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option(
        "", "--s3-endpoint", help="S3-compatible endpoint."
    ),
    sky_bin: str = typer.Option(
        "", "--sky-bin", help="Pinned SkyPilot executable path."
    ),
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
    run_id: str = typer.Argument(
        help="Successful PAIDF run ID or exact s3:// run prefix."
    ),
    project: str = typer.Option(
        "", "--project", "-p", help="Configured project alias."
    ),
    agent_name: str = typer.Option("", "--agent-name", help="Configured agent name."),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact npa-workflow prefix; takes precedence over receipts and discovery.",
    ),
    s3_endpoint: str = typer.Option(
        "", "--s3-endpoint", help="S3-compatible endpoint."
    ),
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
@intent_boundary(OperationIntent.OBSERVE)
def list_cmd(
    project: str = typer.Option(
        "", "--project", "-p", help="Project alias for S3 credentials."
    ),
    workflow_s3_uri: str = typer.Option(
        "", "--workflow-s3-uri", help="Parent durable workflow prefix."
    ),
    workflow_s3_prefix: str = typer.Option(
        "", "--workflow-s3-prefix", help="Parent prefix for durable workflow state."
    ),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option(
        "", "--s3-endpoint", help="S3-compatible endpoint."
    ),
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
@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def cancel_cmd(
    run_id: str = typer.Argument(help="Durable workflow run ID or s3:// run prefix."),
    project: str = typer.Option(
        "", "--project", "-p", help="Project alias for S3 credentials."
    ),
    receipt: str = typer.Option("", "--receipt", help="Opaque teardown receipt ID."),
    project_id: str = typer.Option("", "--project-id", help="Exact Nebius project ID."),
    job_id: str = typer.Option("", "--job-id", help="Exact SkyPilot managed-job ID."),
    workflow_s3_uri: str = typer.Option(
        "",
        "--workflow-s3-uri",
        help="Exact workflow prefix; takes precedence over receipts and discovery.",
    ),
    workflow_s3_prefix: str = typer.Option(
        "", "--workflow-s3-prefix", help="Parent prefix. The run ID is appended."
    ),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket name or URI."),
    s3_endpoint: str = typer.Option(
        "", "--s3-endpoint", help="S3-compatible endpoint."
    ),
    sky_bin: str = typer.Option("", "--sky-bin", help="SkyPilot executable path."),
    cluster: str = typer.Option(
        "", "--cluster", help="SkyPilot cluster name to tear down. Defaults to run ID."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Cancel a launched run; never-launched and terminal runs are repeat-safe no-ops."""
    resolved_run_id = ""
    identity = None
    result: dict[str, Any]
    try:
        from npa.orchestration.npa_workflow.cancellation import (
            assess_run_cancellation,
            is_terminal_workflow_state,
        )
        from npa.orchestration.npa_workflow.run_resolution import resolve_run
        from npa.orchestration.skypilot.workflow_state import WorkflowStateError
        from npa.cleanup_identity import resolve_cleanup_identity
        from npa.clients.config import resolve_environment

        environment = resolve_environment(project) if project else None
        identity = resolve_cleanup_identity(
            explicit={
                "project_alias": project,
                "project_id": project_id,
                "run_id": str(run_id) if not str(run_id).startswith("s3://") else "",
                "workflow_s3_uri": workflow_s3_uri,
                "sky_job_id": job_id,
            },
            receipt_id=receipt,
            live={
                "project_alias": project,
                "project_id": str(getattr(environment, "project_id", "") or ""),
            },
            phase="workflow",
            resource=str(run_id) if not str(run_id).startswith("s3://") else "",
        )
        resolved_identity_run = str(identity.get("run_id") or run_id)
        if identity.receipt_is_terminal:
            from npa.orchestration.npa_workflow.run_resolution import RunResolution

            resolution = RunResolution(
                run_id=resolved_identity_run,
                project=str(identity.get("project_alias") or project),
                not_submitted=True,
                source=f"receipt:{receipt}",
            )
        else:
            resolution = resolve_run(
                resolved_identity_run,
                project=str(identity.get("project_alias") or project),
                workflow_s3_uri=str(identity.get("workflow_s3_uri") or workflow_s3_uri),
                workflow_s3_prefix=workflow_s3_prefix,
                s3_bucket=s3_bucket,
                s3_endpoint=s3_endpoint,
                sky_bin=sky_bin,
                exact_job_id=str(identity.get("sky_job_id") or job_id),
                allow_local_not_submitted=True,
            )
        resolved_run_id = resolution.run_id
        if resolution.not_submitted:
            detected = (
                "NOT_SUBMITTED"
                if not identity.receipt_is_terminal
                or identity.terminal_state == "not_submitted"
                else "TERMINAL"
            )
            result = {
                "run_id": resolved_run_id,
                "outcome": "not_submitted"
                if detected == "NOT_SUBMITTED"
                else "already_absent",
                "launch_state": "not_submitted"
                if detected == "NOT_SUBMITTED"
                else "terminal",
                "detected_state": detected,
                "sky_job_id": "",
                "sky_job_ids": [],
                "cloud_calls": False,
                "verification": "terminal_receipt"
                if identity.receipt_is_terminal
                else "durable_planning_ledger",
                "resolution_checks": resolution.checks_payload(),
                "message": "No cancellation was needed; durable evidence proves the run never launched or is terminal.",
            }
        elif not resolution.found and resolution.conclusively_absent:
            result = {
                "run_id": resolved_run_id,
                "outcome": "already_absent",
                "launch_state": "not_launched",
                "detected_state": "NOT_FOUND",
                "sky_job_id": "",
                "sky_job_ids": [],
                "cloud_calls": False,
                "verification": "conclusively_absent",
                "resolution_checks": resolution.checks_payload(),
                "message": (
                    "No cancellation was needed; the run is conclusively absent after "
                    "every applicable exact source was checked (detected NOT_FOUND)."
                ),
            }
        elif not resolution.found:
            result = {
                "run_id": resolved_run_id,
                "outcome": "verification_failed",
                "detected_state": "VERIFICATION_UNAVAILABLE",
                "sky_job_id": "",
                "sky_job_ids": [],
                "cloud_calls": False,
                "verification": "unavailable",
                "resolution_checks": resolution.checks_payload(),
                "message": (
                    "Run verification is unavailable; a malformed manifest, ambiguous "
                    "run, or provider/auth failure is not absence, so cancellation was "
                    "not attempted."
                ),
            }
        else:
            assessment = assess_run_cancellation(
                resolution,
                sky_bin=sky_bin,
            )
            jobs_payload = [item.to_dict() for item in assessment.jobs]
            job_ids = [item.job_id for item in assessment.jobs]
            active_ids = [item.job_id for item in assessment.active_jobs]
            if not assessment.active_jobs and not assessment.errors:
                terminal = is_terminal_workflow_state(assessment.detected_state)
                result = {
                    "run_id": resolved_run_id,
                    "outcome": "terminal" if terminal else "no_cancellation_needed",
                    "detected_state": assessment.detected_state,
                    "status": assessment.detected_state,
                    "sky_job_id": job_ids[-1] if len(job_ids) == 1 else "",
                    "sky_job_ids": job_ids,
                    "cloud_calls": False,
                    "jobs": jobs_payload,
                    "message": (
                        "No cancellation was needed; authoritative workflow/stage "
                        f"state is {assessment.detected_state}."
                        if terminal
                        else "No cancellation was needed; every exact recorded job is "
                        "terminal or authoritatively absent (detected NO_ACTIVE_JOB)."
                    ),
                }
            elif not assessment.active_jobs:
                result = {
                    "run_id": resolved_run_id,
                    "outcome": "verification_failed",
                    "detected_state": assessment.detected_state,
                    "sky_job_id": "",
                    "sky_job_ids": job_ids,
                    "cloud_calls": False,
                    "jobs": jobs_payload,
                    "errors": assessment.errors,
                    "message": (
                        "Cancellation was not attempted because one or more exact "
                        "workflow/job records could not be verified."
                    ),
                }
            else:
                from npa.orchestration.skypilot.cleanup import (
                    cleanup_launched_workflows,
                )

                cleanup = cleanup_launched_workflows(
                    [
                        (item.job_id, item.job_name or resolved_run_id)
                        for item in assessment.active_jobs
                    ],
                    resolved_run_id,
                    cluster=cluster,
                    sky_bin=sky_bin or None,
                )
                errors = [*assessment.errors, *cleanup.errors]
                result = {
                    "run_id": resolved_run_id,
                    "outcome": "cancelled" if not errors else "partial_cancellation",
                    "detected_state": assessment.detected_state,
                    "status": assessment.detected_state,
                    "sky_job_id": active_ids[0] if len(active_ids) == 1 else "",
                    "sky_job_ids": job_ids,
                    "cancelled_job_ids": active_ids,
                    "cloud_calls": True,
                    "jobs": jobs_payload,
                    "resources_removed": cleanup.resources_removed,
                    "commands": cleanup.commands,
                    "errors": errors,
                    "message": (
                        f"Cancellation converged for {len(active_ids)} active managed job(s)."
                        if not errors
                        else "Cancellation was only partial; retry after resolving the "
                        "reported exact job/provider failures."
                    ),
                }
    except (WorkflowStateError, OSError, RuntimeError, ValueError) as exc:
        result = {
            "run_id": resolved_run_id or str(run_id),
            "outcome": "verification_failed",
            "cloud_calls": False,
            "message": f"{type(exc).__name__}: {exc}",
        }
    try:
        from npa.clients.config import resolve_environment
        from npa.teardown_receipts import record_teardown_event

        environment = resolve_environment(project or None) if project else None
        receipt_state = {
            "already_absent": "verified_absent",
            "cancelled": "cancelled",
            "terminal": "terminal",
            "no_cancellation_needed": "terminal",
            "not_submitted": "not_submitted",
        }.get(str(result.get("outcome") or ""), "verification_failed")
        record_teardown_event(
            phase="workflow",
            resource=resolved_run_id or str(run_id),
            terminal_state=receipt_state,
            project_alias=str(
                identity.get("project_alias") if identity is not None else project
            ),
            project_id=str(
                identity.get("project_id")
                if identity is not None
                else getattr(environment, "project_id", "") or ""
            ),
            identity=(identity.values if identity is not None else None),
            precheck={
                "detected_state": result.get("detected_state", ""),
                "resolution_checks": result.get("resolution_checks", []),
                "job_ids": result.get("sky_job_ids", []),
            },
            action={
                "kind": "managed_job_cancel" if result.get("cloud_calls") else "none",
                "cancelled_job_ids": result.get("cancelled_job_ids", []),
            },
            verification={"outcome": result.get("outcome", "")},
            errors=[str(item) for item in result.get("errors", [])],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        if result.get("cloud_calls"):
            result.setdefault("errors", []).append(
                f"durable teardown receipt could not be written: {exc}"
            )
            result["outcome"] = "partial_cancellation"
        else:
            result.setdefault("diagnostics", []).append(
                f"teardown receipt unavailable: {exc}"
            )
    result["identity_source"] = (
        identity.source if identity is not None else "unavailable"
    )
    result["receipt_id"] = identity.receipt_id if identity is not None else ""
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"run_id: {result['run_id']}")
        typer.echo(f"identity_source: {result['identity_source']}")
        typer.echo(f"outcome: {result['outcome']}")
        if result.get("message"):
            typer.echo(str(result["message"]))
        for error in result.get("errors", []):
            typer.echo(f"cleanup warning: {error}", err=True)
    if result["outcome"] in {"verification_failed", "partial_cancellation"}:
        raise typer.Exit(code=2)


@app.command("teardown")
def teardown_cmd(
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="Output format.",
    ),
) -> None:
    """Destroy both VMs from a distill workflow run.

    Reads the sim and train VM specs from the distill module, bootstraps
    Nebius credentials, and destroys each VM via Terraform.  Also removes
    the workbench entries from ~/.npa/config.yaml.
    """
    from npa.workflows.distill_two_vm import (
        PROJECT_ALIAS,
        PROJECT_ID,
        REGION,
        SIM_VM,
        TENANT_ID,
        TRAIN_VM,
        TwoVMDistillError,
        _destroy_vm,
    )
    from npa.clients.config import (
        ConfigError,
        resolve_ssh_config,
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
            PROJECT_ID,
            TENANT_ID,
            REGION,
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
        False,
        "--teardown/--no-teardown",
        help="Destroy both VMs after the workflow completes (even on failure).",
    ),
    skip_infra: bool = typer.Option(
        False,
        "--skip-infra/--provision",
        help="Skip provisioning and Nebius bootstrap; resolve VMs and S3 "
        "credentials from ~/.npa/config.yaml.",
    ),
    skip_setup: bool = typer.Option(
        False,
        "--skip-setup/--setup",
        help="Skip runtime setup (conda env + npa install). Use when VMs "
        "already have the correct environment.",
    ),
    n_envs: int = typer.Option(
        4096, "--n-envs", help="Parallel environments for simulation."
    ),
    teacher_max_iterations: int = typer.Option(
        500,
        "--teacher-max-iterations",
        help="PPO training iterations for teacher.",
    ),
    student_policy: str = typer.Option(
        "act",
        "--student-policy",
        help="Student policy type: act, diffusion, smolvla.",
    ),
    student_epochs: int = typer.Option(
        100,
        "--student-epochs",
        help="Training epochs for student.",
    ),
    student_batch_size: int = typer.Option(
        64,
        "--student-batch-size",
        help="Batch size for student training.",
    ),
    eval_n_episodes: int = typer.Option(
        1024,
        "--eval-n-episodes",
        help="Number of eval episodes for the student.",
    ),
    action_space: ActionSpace = typer.Option(
        ActionSpace.cartesian,
        "--action-space",
        help="Action space for Genesis env: 'cartesian' (4D: delta xyz + gripper) "
        "or 'joint' (8D: delta joint positions + gripper).",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text,
        "--output-format",
        help="Output format.",
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
        _fail(
            f"--teacher-max-iterations must be positive, got {teacher_max_iterations}"
        )
    if student_epochs <= 0:
        _fail(f"--student-epochs must be positive, got {student_epochs}")
    if student_batch_size <= 0:
        _fail(f"--student-batch-size must be positive, got {student_batch_size}")
    if eval_n_episodes <= 0:
        _fail(f"--eval-n-episodes must be positive, got {eval_n_episodes}")
    if student_policy not in ("act", "diffusion", "smolvla"):
        _fail(
            f"--student-policy must be act, diffusion, or smolvla, got {student_policy}"
        )

    mode = "skip-infra" if skip_infra else "provision"
    console.print(f"[bold]Expert distillation ({mode})[/bold]")
    console.print(f"  sim:   L40S  ({mode})")
    console.print(f"  train: H100  ({mode})")
    console.print(
        f"  policy={student_policy}  n_envs={n_envs}  epochs={student_epochs}"
    )

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
        from npa.orchestration.npa_workflow.submission_state import (
            update_submission_state,
        )

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
    yaml_path: Path = typer.Argument(
        help="NPA workflow spec (apiVersion: npa.workflow/v0.0.1)."
    ),
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
    try:
        spec = merge_config_overrides(spec, _parse_submit_vars(var))
    except NpaWorkflowError as exc:
        _fail(str(exc))
        return
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
            typer.echo(
                f"  {wave.index:02d}. [{wave.kind}] {wave.name}: {states}{suffix}"
            )
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
    assume_decision: str = typer.Option(
        "", "--assume-decision", help="Branch assumption for planning."
    ),
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

    from npa.orchestration.npa_workflow import (
        NpaWorkflowError,
        build_plan,
        run_workflow,
    )
    from npa.orchestration.npa_workflow.scheduler import build_scheduler_plan
    from npa.orchestration.npa_workflow.submit import merge_config_overrides

    spec = _load_npa_workflow(yaml_path)
    spec = merge_config_overrides(spec, _parse_submit_vars(var))
    _warn_placeholder_bucket(spec.config, quiet=json_output)
    resolved_run_id = run_id or f"{spec.name}-{int(time.time())}"
    resolved_assume = assume_decision or str(
        spec.config.get("plan_assume_decision") or ""
    )
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
        report["scheduler"] = build_scheduler_plan(
            spec, plan.steps, run_id=resolved_run_id
        )
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
    var: list[str] = typer.Option(
        [],
        "--var",
        help="Workflow config override as KEY=VALUE (same as submit --var).",
    ),
    registry: str = typer.Option("", "--registry", help="Container registry override."),
    project: str = typer.Option(
        "",
        "--project",
        "-p",
        help="Project alias whose configured registry to check. Defaults to the configured project.",
    ),
    image: str = typer.Option("", "--image", help="Pin every step to this image."),
    image_override: list[str] = typer.Option(
        [],
        "--image-override",
        help="Repeatable exact TOOL_REF=IMAGE override.",
    ),
    assume_decision: str = typer.Option(
        "", "--assume-decision", help="Branch assumption for planning."
    ),
    gpu_target: str = typer.Option("", "--gpu-target", help="SONIC GPU target."),
    image_variant: str = typer.Option(
        "", "--image-variant", help="SONIC image variant."
    ),
    infra: str = typer.Option(
        "", "--infra", help="Exact k8s/<context> used for unattested image probes."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON report."),
) -> None:
    """Prove every image this spec pulls is pullable, with the run's own credentials.

    Kubernetes retries image pulls forever, so a registry that answers 403 leaves the
    job in PENDING/ImagePullBackOff rather than failing. Being able to list a
    repository's tags is a different permission from pulling it, so this reproduces
    the actual manifest fetch a worker performs.
    """

    from npa.orchestration.npa_workflow import build_plan
    from npa.orchestration.npa_workflow.submit import merge_config_overrides
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        plan_image_pull_secrets,
        plan_images,
    )
    from npa.orchestration.skypilot.registry_preflight import (
        check_image_pulls_with_credentials,
    )

    spec = merge_config_overrides(
        _load_npa_workflow(yaml_path), _parse_submit_vars(var)
    )
    image_overrides: dict[str, str] = {}
    if image.strip():
        image_overrides["*"] = image.strip()
    try:
        image_overrides.update(_parse_image_overrides(image_override))
    except ValueError as exc:
        _fail(str(exc))
        return
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
    pull_secrets_by_image = plan_image_pull_secrets(
        spec, plan.steps, run_id=run_id, options=options
    )
    if not images:
        typer.echo("images: none pinned by this spec")
        return

    checks = check_image_pulls_with_credentials(
        images,
        mint=True,
        pull_secrets_by_image=pull_secrets_by_image,
    )
    failed = [check for check in checks if not check.ok]
    contract_checks: list[dict[str, object]] = []
    if not failed:
        from npa.orchestration.skypilot.k8s_gpu_catalog import context_from_infra

        contract_checks = _preflight_image_bootstrap_contracts(
            images=images,
            pull_checks=checks,
            context=context_from_infra(infra),
            pull_secrets_by_image=pull_secrets_by_image,
        )
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
                        "operator_status": check.operator_status,
                        "target_status": check.target_status,
                        "authority": check.authority,
                        "digest": check.digest,
                        "bootstrap_contract": next(
                            (
                                item
                                for item in contract_checks
                                if item.get("digest") == check.digest
                            ),
                            {},
                        ),
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
    if failed:
        _fail(
            f"{len(failed)} of {len(checks)} image(s) cannot be pulled with this run's credentials"
        )


@app.command("gpus")
def gpus_cmd(
    project: str = typer.Option(
        "",
        "--project",
        help="Project alias used to verify shared-controller ownership.",
    ),
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
        discover_kubernetes_gpu_inventory,
        resolve_kubernetes_accelerator,
        spec_accelerators,
    )

    resolved_context = (
        context.strip() or os.environ.get("KUBECONTEXT", "").strip() or cluster.strip()
    )
    env_backup: str | None = None
    if cluster.strip():
        from npa.cluster.state import kubeconfig_file

        kubeconfig = kubeconfig_file(cluster.strip())
        if not kubeconfig.exists():
            _fail(f"Kubeconfig not found for cluster {cluster!r}: {kubeconfig}")
            return
        env_backup = os.environ.get("KUBECONFIG")
        os.environ["KUBECONFIG"] = str(kubeconfig)
    inventory = discover_kubernetes_gpu_inventory(context=resolved_context)
    sky_error = ""
    if resolved_context:
        try:
            from npa.controller_ownership import (
                verify_controller_owner,
                verify_recorded_controller_owner,
            )

            if isinstance(project, str) and project.strip():
                verify_controller_owner(project, resolved_context)
            else:
                owner = verify_recorded_controller_owner()
                if owner is not None and owner.context != resolved_context:
                    raise RuntimeError(
                        "Shared controller owner context does not match requested GPU context."
                    )
        except (OSError, RuntimeError, ValueError) as exc:
            sky_error = str(exc)
    try:
        if sky_error:
            raise KubernetesGpuCatalogError(sky_error)
        catalog = discover_kubernetes_gpu_catalog(
            context=resolved_context, sky_bin=sky_bin or None
        )
    except KubernetesGpuCatalogError as exc:
        sky_error = str(exc)
        from npa.orchestration.skypilot.k8s_gpu_catalog import KubernetesGpuCatalog

        catalog = KubernetesGpuCatalog({}, context=resolved_context)
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
                resolution = resolve_kubernetes_accelerator(
                    accelerator, catalog=catalog
                )
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
                    "kubernetes": inventory.to_dict(),
                    "skypilot_error": sky_error,
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
        if sky_error:
            raise typer.Exit(code=2)
        return

    typer.echo(
        "kubernetes: "
        f"ready_nodes={inventory.ready_nodes} "
        f"eligible_gpu_nodes={inventory.eligible_gpu_nodes} "
        f"capacity={inventory.capacity} allocatable_gpus={inventory.allocatable} "
        f"accelerator_product={','.join(inventory.products) or 'unknown/unlabeled'}"
    )
    if inventory.allocatable > 0 and not inventory.products:
        typer.echo(
            "label_readiness: blocked_missing_product_label (wait for GPU Feature Discovery/NFD)",
            err=True,
        )
    if catalog.is_empty:
        typer.echo("accelerators: none advertised")
        if sky_error:
            typer.echo(f"skypilot_error: {sky_error}", err=True)
            raise typer.Exit(code=2)
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
