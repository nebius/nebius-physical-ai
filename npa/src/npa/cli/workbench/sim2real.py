"""CLI for the Sim2Real VLM-to-RL workflow."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from npa.clients.credentials import load_credentials
from npa.workflows.sim2real_loop import (
    DEFAULT_HELDOUT_ENVS,
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_LEROBOT_DATASET_ID,
    DEFAULT_LOOP_OF_LOOPS_ITERATIONS,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_S3_ENDPOINT,
    DEFAULT_STEPS_PER_ROLLOUT,
    DEFAULT_THRESHOLD,
    build_config_from_env,
    convert_vlm_eval_to_rl_signal,
    run_full_loop,
    run_inner_loop,
)
from npa.workflows.sim2real_rerun_regen import (
    Sim2RealRerunRegenError,
    default_regen_local_dir,
    download_rrd_from_s3,
    regen_sim2real_rrd,
    rerun_heldout_eval_only,
    resolve_local_rrd_path,
)
from npa.workflows.rerun_serve import (
    DEFAULT_NAMESPACE,
    DEFAULT_PORT,
    DEFAULT_RERUN_IMAGE,
    DEFAULT_S3_PREFIX,
    Sim2RealRerunServeError,
    apply_rerun_serve,
    build_rerun_serve_config,
    build_rerun_serve_manifest,
    destroy_rerun_serve,
    redact_rerun_serve_manifest,
    resolve_cluster_name_from_config,
    require_kubeconfig,
)
from npa.workflows.sim2real import onboarding as onboarding_svc
from npa.workflows.sim2real.onboarding_spec import (
    OnboardingSpecError,
    load_onboarding_spec,
)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


app = typer.Typer(
    name="sim2real",
    help="Sim2Real VLM-to-RL loop and threshold-gated workflow.",
    no_args_is_help=True,
)


@app.command("run")
def run_command(
    run_id: str = typer.Option(
        "sim2real-cli", "--run-id", help="Run id for local and S3 artifacts."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Local artifact directory."
    ),
    s3_bucket: str = typer.Option(
        "", "--s3-bucket", help="S3 bucket for artifact upload."
    ),
    s3_prefix: str = typer.Option(
        "sim2real-b", "--s3-prefix", help="S3 prefix parent for this run."
    ),
    s3_endpoint: str = typer.Option(
        DEFAULT_S3_ENDPOINT, "--s3-endpoint", help="S3-compatible endpoint."
    ),
    trigger_dataset_uri: str = typer.Option(
        "",
        "--trigger-dataset-uri",
        help="LeRobot dataset URI under the dedicated trigger path.",
    ),
    trigger_dataset_id: str = typer.Option(
        DEFAULT_LEROBOT_DATASET_ID,
        "--trigger-dataset-id",
        help="Human-readable source dataset id recorded in Stage 1.",
    ),
    action_rollouts_uri: str = typer.Option(
        "", "--action-rollouts-uri", help="Stage A action rollout URI or local path."
    ),
    train_envs_uri: str = typer.Option("", "--train-envs-uri", help="Train env URI."),
    heldout_envs_uri: str = typer.Option(
        "", "--heldout-envs-uri", help="Held-out env URI."
    ),
    assets_uri: str = typer.Option("", "--assets-uri", help="BYO assets URI."),
    scene_spec_uri: str = typer.Option(
        "", "--scene-spec-uri", help="BYO SceneSpec URI."
    ),
    cameras_uri: str = typer.Option(
        "", "--cameras-uri", help="Standalone cameras.json URI (optional)."
    ),
    robot_spec_uri: str = typer.Option(
        "", "--robot-spec-uri", help="BYO RobotSpec JSON URI (robot embodiment)."
    ),
    robot_source: str = typer.Option(
        "",
        "--robot-source",
        help="BYO robot source: stock_franka/byo_urdf/byo_mjcf/byo_usd/genesis_builtin.",
    ),
    robot_preset: str = typer.Option(
        "",
        "--robot-preset",
        help="Built-in robot preset: franka/ur5e/ur10e/flexiv.",
    ),
    augment_image: str = typer.Option(
        "", "--augment-image", help="BYO augmentation image."
    ),
    policy_image: str = typer.Option("", "--policy-image", help="BYO policy image."),
    trainer_image: str = typer.Option(
        "", "--trainer-image", help="BYO VLM-RL trainer image."
    ),
    vlm_image: str = typer.Option("", "--vlm-image", help="BYO VLM image."),
    eval_image: str = typer.Option("", "--eval-image", help="BYO held-out eval image."),
    isaac_image: str = typer.Option(
        "", "--isaac-image", help="Isaac Lab held-out rollout image (Isaac Sim headless)."
    ),
    sim_backend: str = typer.Option(
        "isaac",
        "--sim-backend",
        help="Held-out sim backend: 'isaac' (default) or 'genesis'.",
    ),
    isaac_task: str = typer.Option(
        "Isaac-Lift-Cube-Franka-v0",
        "--isaac-task",
        help="Isaac Lab manipulation task id for the stock held-out rollout.",
    ),
    vlm_model: str = typer.Option(
        "nvidia/Cosmos-Reason2-8B",
        "--vlm-model",
        help="Legacy single-VLM model id when dual Reason eval is disabled.",
    ),
    threshold: float = typer.Option(
        DEFAULT_THRESHOLD, "--threshold", help="Held-out success threshold."
    ),
    inner_iterations: int = typer.Option(
        DEFAULT_INNER_ITERATIONS, "--inner-iterations", help="Inner-loop cap."
    ),
    outer_iterations: int = typer.Option(
        DEFAULT_OUTER_ITERATIONS, "--outer-iterations", help="Outer-loop cap."
    ),
    loop_of_loops_iterations: int = typer.Option(
        DEFAULT_LOOP_OF_LOOPS_ITERATIONS,
        "--loop-of-loops-iterations",
        help="Stage 12 to 13 to 1 cap.",
    ),
    rollout_count: int = typer.Option(
        DEFAULT_ROLLOUT_COUNT, "--rollout-count", help="Train rollout count."
    ),
    steps_per_rollout: int = typer.Option(
        DEFAULT_STEPS_PER_ROLLOUT, "--steps-per-rollout", help="Steps per rollout."
    ),
    heldout_env_count: int = typer.Option(
        DEFAULT_HELDOUT_ENVS, "--heldout-env-count", help="Held-out env count."
    ),
    seed: int = typer.Option(42, "--seed", help="Deterministic seed."),
    upload_artifacts: bool = typer.Option(
        False, "--upload-artifacts", help="Upload local artifacts to S3."
    ),
    no_guardrails: bool = typer.Option(
        False, "--no-guardrails", help="Skip optional guardrails where supported."
    ),
    signal_loss_weight: float = typer.Option(
        1.0, "--signal-loss-weight", help="VLM signal loss multiplier."
    ),
    learning_rate: float = typer.Option(
        0.05, "--learning-rate", help="Reference adapter learning rate."
    ),
    byo_signal_converter: str = typer.Option(
        "", "--byo-signal-converter", help="BYO signal converter command."
    ),
    byo_trainer_command: str = typer.Option(
        "", "--byo-trainer-command", help="BYO trainer command."
    ),
    byo_vlm_command: str = typer.Option(
        "", "--byo-vlm-command", help="BYO VLM command."
    ),
    byo_eval_command: str = typer.Option(
        "", "--byo-eval-command", help="BYO eval command."
    ),
    k8s_namespace: str = typer.Option(
        "", "--k8s-namespace", help="Namespace for sibling component Jobs."
    ),
    k8s_service_account: str = typer.Option(
        "agent-sa", "--k8s-service-account", help="Service account for sibling Jobs."
    ),
    k8s_image_pull_secrets: str = typer.Option(
        "agent-sa,ngc-nvcr-imagepullsecret,npa-nebius-registry",
        "--k8s-image-pull-secrets",
        help="Comma-separated imagePullSecrets for sibling Jobs.",
    ),
    k8s_env_secret_names: str = typer.Option(
        "hf-ngc-tokens,npa-storage-credentials",
        "--k8s-env-secret-names",
        help="Comma-separated env secrets for sibling Jobs.",
    ),
    k8s_gpu_resource: str = typer.Option(
        "nvidia.com/gpu", "--k8s-gpu-resource", help="Kubernetes GPU resource key."
    ),
    k8s_gpu_product: str = typer.Option(
        "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        "--k8s-gpu-product",
        help="GPU product node selector for sibling Jobs.",
    ),
    k8s_job_timeout_s: int = typer.Option(
        7200, "--k8s-job-timeout-s", help="Sibling Job timeout in seconds."
    ),
    source_repo: str = typer.Option(
        "", "--source-repo", help="Optional source repository cloned by sibling Jobs."
    ),
    source_ref: str = typer.Option(
        "", "--source-ref", help="Optional source ref cloned by sibling Jobs."
    ),
    heldout_eval_limit: int = typer.Option(
        0, "--heldout-eval-limit", help="Optional held-out env sample cap."
    ),
    output_json: bool = typer.Option(False, "--output-json", help="Print only JSON."),
) -> None:
    """Run the full 14-stage Sim2Real workflow."""

    config = build_config_from_env(
        run_id=run_id,
        output_dir=output_dir,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=s3_endpoint,
        trigger_dataset_uri=trigger_dataset_uri,
        trigger_dataset_id=trigger_dataset_id,
        action_rollouts_uri=action_rollouts_uri,
        train_envs_uri=train_envs_uri,
        heldout_envs_uri=heldout_envs_uri,
        assets_uri=assets_uri,
        scene_spec_uri=scene_spec_uri,
        cameras_uri=cameras_uri,
        robot_spec_uri=robot_spec_uri,
        robot_source=robot_source,
        robot_preset=robot_preset,
        augment_image=augment_image,
        policy_image=policy_image,
        trainer_image=trainer_image,
        vlm_image=vlm_image,
        eval_image=eval_image,
        isaac_image=isaac_image,
        sim_backend=sim_backend,
        isaac_task=isaac_task,
        vlm_model=vlm_model,
        threshold=threshold,
        inner_iterations=inner_iterations,
        outer_iterations=outer_iterations,
        loop_of_loops_iterations=loop_of_loops_iterations,
        rollout_count=rollout_count,
        steps_per_rollout=steps_per_rollout,
        heldout_env_count=heldout_env_count,
        seed=seed,
        upload_artifacts=upload_artifacts,
        no_guardrails=no_guardrails,
        signal_loss_weight=signal_loss_weight,
        learning_rate=learning_rate,
        byo_signal_converter=byo_signal_converter,
        byo_trainer_command=byo_trainer_command,
        byo_vlm_command=byo_vlm_command,
        byo_eval_command=byo_eval_command,
        k8s_namespace=k8s_namespace,
        k8s_service_account=k8s_service_account,
        k8s_image_pull_secrets=k8s_image_pull_secrets,
        k8s_env_secret_names=k8s_env_secret_names,
        k8s_gpu_resource=k8s_gpu_resource,
        k8s_gpu_product=k8s_gpu_product,
        k8s_job_timeout_s=k8s_job_timeout_s,
        source_repo=source_repo,
        source_ref=source_ref,
        heldout_eval_limit=heldout_eval_limit,
    )
    report = run_full_loop(config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if output_json:
        typer.echo(text)
    else:
        typer.echo(text)


@app.command("materialize")
def materialize_command(
    runbook: Optional[Path] = typer.Argument(
        None,
        help="SkyPilot runbook to render (default: the committed sim2real runbook).",
    ),
    run_id: str = typer.Option("", "--run-id", help="Run id; also sets NPA_SIM2REAL_RUN_ID."),
    image: str = typer.Option(
        "", "--image", help="Registry-qualified trainer image (required while the runbook ships a placeholder)."
    ),
    env: list[str] = typer.Option(
        [], "--env", help="KEY=VALUE override for a runbook env (repeatable)."
    ),
    namespace: str = typer.Option("", "--namespace", help="Kubernetes namespace override."),
    skip_setup: bool = typer.Option(
        False, "--skip-setup", help="Omit the runbook setup block (image already carries npa)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the Job manifest YAML here instead of stdout."
    ),
) -> None:
    """Render the sim2real runbook to a Kubernetes Job (no SkyPilot, no operator pack).

    This is the in-repo GPU-reaching path while raw `sky jobs launch` is blocked
    by the SkyPilot 0.12.2 pre-setup getcwd() bug: materialize, then
    `kubectl apply -f <manifest>` and follow with `npa workbench sim2real status`.
    """

    from npa.workflows.sim2real.materialize import (
        Sim2RealMaterializeError,
        materialize_k8s_job,
    )

    overrides: dict[str, str] = {}
    for item in env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise typer.BadParameter(f"--env expects KEY=VALUE, got {item!r}")
        overrides[key] = value

    try:
        job = materialize_k8s_job(
            runbook,
            run_id=run_id,
            image=image,
            env_overrides=overrides,
            namespace=namespace,
            include_setup=not skip_setup,
        )
    except (Sim2RealMaterializeError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    if output is not None:
        output.write_text(job.to_yaml(), encoding="utf-8")
        typer.echo(f"wrote {output}")
        typer.echo(
            f"apply with: kubectl apply -f {output} "
            f"(job {job.job_name} in namespace {job.namespace}, image {job.image})"
        )
    else:
        typer.echo(job.to_yaml())
    for warning in job.warnings:
        typer.echo(f"note: {warning}", err=True)


@app.command("status")
def status_command(
    run_id: str = typer.Option(..., "--run-id", help="Sim2Real staged run id."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket for run artifacts."),
    s3_prefix: str = typer.Option(
        DEFAULT_S3_PREFIX, "--s3-prefix", help="S3 parent prefix (default: sim2real-b)."
    ),
    s3_endpoint: str = typer.Option(
        DEFAULT_S3_ENDPOINT, "--s3-endpoint", help="S3-compatible endpoint."
    ),
    k8s_context: str = typer.Option("", "--k8s-context", help="Kubernetes context."),
    k8s_namespace: str = typer.Option("default", "--k8s-namespace", help="Job namespace."),
    watch: bool = typer.Option(
        False, "--watch/--no-watch", help="Refresh until the run reaches a terminal state."
    ),
    interval: float = typer.Option(10.0, "--interval", help="Watch refresh interval."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Check kubectl-submitted Sim2Real runs via S3 workflow_state.json and K8s jobs."""

    from npa.workflows.sim2real.monitor import watch_sim2real_status

    watch_sim2real_status(
        run_id,
        watch=watch,
        interval=interval,
        json_output=json_output,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=s3_endpoint,
        k8s_context=k8s_context,
        k8s_namespace=k8s_namespace,
    )


@app.command("inner-loop")
def inner_loop_command(
    run_id: str = typer.Option("sim2real-inner-cli", "--run-id"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, "--threshold"),
    inner_iterations: int = typer.Option(
        DEFAULT_INNER_ITERATIONS, "--inner-iterations"
    ),
    rollout_count: int = typer.Option(DEFAULT_ROLLOUT_COUNT, "--rollout-count"),
    steps_per_rollout: int = typer.Option(
        DEFAULT_STEPS_PER_ROLLOUT, "--steps-per-rollout"
    ),
    initial_quality: float = typer.Option(0.38, "--initial-quality"),
) -> None:
    """Run only Stage 7-9 and print closure evidence."""

    config = build_config_from_env(
        run_id=run_id,
        output_dir=output_dir,
        threshold=threshold,
        inner_iterations=inner_iterations,
        rollout_count=rollout_count,
        steps_per_rollout=steps_per_rollout,
    )
    evidence = run_inner_loop(
        config, local_dir=output_dir, initial_quality=initial_quality
    )
    typer.echo(json.dumps(evidence, indent=2, sort_keys=True))


@app.command("convert-signal")
def convert_signal_command(
    vlm_json: Path = typer.Option(..., "--vlm-json", help="Structured VLM eval JSON."),
    output_json: Path = typer.Option(
        ..., "--output-json", help="RL signal JSON output path."
    ),
) -> None:
    """Convert one structured VLM eval JSON into the RL signal schema."""

    payload = json.loads(vlm_json.read_text(encoding="utf-8"))
    signal = convert_vlm_eval_to_rl_signal(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(signal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(str(output_json))


def _onboard_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@app.command("onboard-robot")
def onboard_robot_command(
    spec: Path = typer.Option(
        ..., "--spec", help="Onboarding robot+task spec YAML (see robot-onboarding.template.yaml)."
    ),
    smoke: bool = typer.Option(
        False,
        "--smoke/--no-smoke",
        help="Submit a short BYO Isaac trainer job to confirm the robot loads, "
        "retargets, and trains (requires cluster creds + ISAAC_IMAGE in env).",
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="Run id for the smoke job (default: derived from robot name)."
    ),
    smoke_iterations: int = typer.Option(
        onboarding_svc.DEFAULT_SMOKE_ITERATIONS,
        "--smoke-iterations",
        help="Smoke-job training iterations (short; just confirms it trains).",
    ),
    smoke_num_envs: int = typer.Option(
        onboarding_svc.DEFAULT_SMOKE_NUM_ENVS,
        "--smoke-num-envs",
        help="Smoke-job parallel envs.",
    ),
    k8s_namespace: str = typer.Option("default", "--k8s-namespace", help="Smoke job namespace."),
    json_output: bool = typer.Option(False, "--json", help="Emit the derived plan as JSON."),
) -> None:
    """Validate an onboarding spec, print the auto-derived robot-aware task config,
    and (with --smoke) submit a short Isaac trainer job to confirm the robot trains.

    The customer supplies a minimal robot+task YAML; this command shows what was
    derived (action scale, reach, object/goal placement, init pose, gripper
    targets) vs. supplied, gates on task/embodiment compatibility, and optionally
    runs a smoke train so they see the robot load and learn before a full run.
    """

    if smoke_iterations < 1:
        _fail_onboard(f"--smoke-iterations must be >= 1, got {smoke_iterations}")
    if smoke_num_envs < 1:
        _fail_onboard(f"--smoke-num-envs must be >= 1, got {smoke_num_envs}")

    try:
        parsed = load_onboarding_spec(spec)
    except OnboardingSpecError as exc:
        _fail_onboard(f"spec validation failed: {exc}")
    except OSError as exc:
        _fail_onboard(f"could not read spec {spec}: {exc}")

    plan = onboarding_svc.build_plan(parsed)
    derived = plan.derived.to_dict()
    report = {
        "robot": parsed.robot.name,
        "robot_source": parsed.robot.robot_source,
        "skill": parsed.task.skill,
        "derived": derived,
        "robot_payload": plan.robot_payload,
        "robot_usd_uri": plan.robot_usd_uri,
        "compat": plan.compat,
    }

    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"Onboarding spec: {spec}")
        typer.echo(f"  robot       : {parsed.robot.name} ({parsed.robot.robot_source})")
        typer.echo(f"  task skill  : {parsed.task.skill}")
        typer.echo("Auto-derived robot-aware config (value [source]):")
        src = derived.get("source", {})
        for key in (
            "action_scale",
            "workspace_reach_m",
            "minimal_height_m",
            "success_distance_m",
            "gripper_open",
            "gripper_close",
        ):
            typer.echo(f"  {key:<20} {derived.get(key)} [{src.get(key, 'default')}]")
        typer.echo(f"  object_init_range    {derived.get('object_init_range')}")
        typer.echo(f"  goal_range           {derived.get('goal_range')}")
        if derived.get("goal_pos"):
            typer.echo(f"  goal_pos             {derived.get('goal_pos')}")
        if derived.get("init_joint_pos"):
            typer.echo(f"  init_joint_pos       {derived.get('init_joint_pos')}")

    compat = plan.compat
    if not compat.get("task_robot_compatible", True):
        typer.secho(
            f"\nINCOMPATIBLE: {compat.get('reason')}", fg=typer.colors.RED, err=True
        )
        for req in compat.get("requirements", []):
            typer.secho(f"  - need: {req}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not smoke:
        if not json_output:
            typer.secho(
                "\nSpec valid and embodiment compatible. Re-run with --smoke to "
                "confirm it trains on the cluster.",
                fg=typer.colors.GREEN,
            )
        return

    # --- smoke submit (real cluster) ---
    from npa.workflows.sim2real import byo_isaac_trainer as trainer

    image = _onboard_env("ISAAC_IMAGE") or _onboard_env("NPA_SIM2REAL_ISAAC_IMAGE")
    bucket = _onboard_env("NPA_SIM2REAL_BUCKET") or _onboard_env("S3_BUCKET")
    endpoint = _onboard_env("AWS_ENDPOINT_URL")
    service_account = _onboard_env("NPA_SIM2REAL_K8S_SERVICE_ACCOUNT", "agent-sa")
    if not image:
        _fail_onboard("--smoke needs ISAAC_IMAGE/NPA_SIM2REAL_ISAAC_IMAGE in the env")
    if not bucket:
        _fail_onboard("--smoke needs NPA_SIM2REAL_BUCKET/S3_BUCKET in the env")

    rid = run_id.strip() or f"onboard-{_slug(parsed.robot.name)}"

    def _kubectl_apply(manifest: dict) -> int:
        trainer._kubectl(
            ["delete", "job", f"s2r-onboard-smoke-{rid}"[:63], "-n", k8s_namespace,
             "--ignore-not-found"],
            timeout=60,
        )
        res = trainer._kubectl(
            ["apply", "-f", "-"], stdin=json.dumps(manifest), timeout=120
        )
        if res.returncode != 0:
            typer.secho(f"kubectl apply failed: {res.stderr}", fg=typer.colors.RED, err=True)
        return res.returncode

    result = onboarding_svc.submit_smoke_job(
        plan,
        run_id=rid,
        image=image,
        bucket=bucket,
        endpoint=endpoint,
        namespace=k8s_namespace,
        service_account=service_account,
        iterations=smoke_iterations,
        num_envs=smoke_num_envs,
        kubectl_apply=_kubectl_apply,
    )
    if result["apply_rc"] != 0:
        typer.secho(
            f"smoke job submit FAILED (rc={result['apply_rc']})",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)
    typer.secho(
        f"smoke job submitted: {result['job_name']} -> {result['out_uri']}",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"watch: npa workbench sim2real status --run-id {rid} --k8s-namespace {k8s_namespace}"
    )


def _slug(text: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in text.strip().lower())
    return cleaned.strip("-") or "robot"


def _fail_onboard(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


rerun_app = typer.Typer(
    name="rerun",
    help="Host Sim2Real Rerun recordings on the customer's mk8s cluster.",
    no_args_is_help=True,
)
app.add_typer(rerun_app, name="rerun")


def _emit_rerun_serve_result(payload: dict, *, output: OutputFormat) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"status: {payload['status']}")
    typer.echo(f"run_id: {payload['run_id']}")
    typer.echo(f"rrd_s3_uri: {payload['rrd_s3_uri']}")
    if payload.get("public_url"):
        typer.echo(f"public_url: {payload['public_url']}")
    else:
        service_type = str(payload.get("service_type", "")).strip().lower()
        if service_type in {"loadbalancer", "lb"}:
            deployment = payload.get("deployment_name", "npa-rerun")
            namespace = payload.get("namespace", DEFAULT_NAMESPACE)
            typer.echo(
                "public_url: pending — LoadBalancer external IP not assigned yet. "
                "Wait and re-run serve, or inspect cluster networking (for example "
                f"`kubectl describe svc {deployment} -n {namespace}` for quota or cloud-controller errors).",
                err=True,
            )
    if payload.get("local_url"):
        typer.echo(f"local_url: {payload['local_url']}")
    if payload.get("port_forward_command"):
        typer.echo(f"port_forward: {payload['port_forward_command']}")
    if payload.get("local_rrd_path"):
        typer.echo(f"local_rrd_path: {payload['local_rrd_path']}")
    if not payload.get("public_url") and not payload.get("local_url"):
        typer.echo(f"cluster_url: {payload['cluster_url']}")


def _rerun_serve_credentials() -> tuple[str, str]:
    creds = load_credentials()
    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key
    if not access_key or not secret_key:
        raise Sim2RealRerunServeError(
            "S3 credentials are required. Configure ~/.npa/credentials.yaml or export AWS_*."
        )
    return access_key, secret_key


@rerun_app.command("serve")
def rerun_serve_command(
    run_id: str = typer.Option(..., "--run-id", help="Completed Sim2Real run id."),
    project: str = typer.Option("", "--project", "-p", help="Project alias for storage resolution."),
    cluster_name: str = typer.Option(
        "", "--cluster-name", help="NPA cluster profile for cached kubeconfig (default: from ~/.npa/config.yaml)."
    ),
    kubeconfig: str = typer.Option("", "--kubeconfig", help="Kubeconfig path override."),
    namespace: str = typer.Option(DEFAULT_NAMESPACE, "--namespace", help="Kubernetes namespace."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help="Rerun web viewer port."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket override."),
    s3_prefix: str = typer.Option(DEFAULT_S3_PREFIX, "--s3-prefix", help="S3 prefix parent for runs."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint override."),
    rrd_uri: str = typer.Option(
        "", "--rrd-uri", help="Explicit s3:// URI for reports/sim2real.rrd (no local download)."
    ),
    report_uri: str = typer.Option(
        "",
        "--report-uri",
        help="s3:// URI for reports/sim2real-report.json; .rrd is derived as sibling.",
    ),
    rerun_image: str = typer.Option(
        DEFAULT_RERUN_IMAGE,
        "--rerun-image",
        help=(
            "Rerun viewer container image (default: python:3.11-slim-bookworm with pip-installed "
            "rerun-sdk; override with npa-rerun-viewer when serving from a "
            "or NPA_RERUN_VIEWER_IMAGE (legacy alias: NPA_SIM2REAL_RERUN_IMAGE)."
        ),
    ),
    service_type: str = typer.Option(
        "loadbalancer",
        "--service-type",
        help="Kubernetes Service type: loadbalancer, nodeport, or clusterip.",
    ),
    name: str = typer.Option(
        "",
        "--name",
        help="Override the shared cluster viewer Deployment/Service name.",
    ),
    destroy: bool = typer.Option(
        False,
        "--destroy",
        help="Delete the shared cluster Rerun viewer (not scoped to run_id).",
    ),
    destroy_wait: bool = typer.Option(
        False,
        "--wait",
        help="When used with --destroy, wait for Kubernetes to confirm resource deletion.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the Kubernetes manifest only."),
    local_record: bool = typer.Option(
        False,
        "--local-record",
        help="Also download reports/sim2real.rrd to disk (LOCAL_RRD_PATH or /tmp/sim2real-regen/<run-id>/reports/sim2real.rrd).",
    ),
    local_rrd_path: Optional[Path] = typer.Option(
        None,
        "--local-rrd-path",
        help="Override local .rrd destination when using --local-record.",
    ),
    auth_user: str = typer.Option(
        "", "--auth-user", help="Enable HTTP basic-auth on the hosted viewer with this username."
    ),
    auth_password: str = typer.Option(
        "", "--auth-password",
        help="Password for --auth-user (generated if --auth-user is set without one).",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Deploy a hosted Rerun viewer; pod init container pulls reports/sim2real.rrd from S3."""
    try:
        access_key, secret_key = _rerun_serve_credentials()
        cluster_context = cluster_name.strip() or resolve_cluster_name_from_config()
        # Basic-auth: gate the cloud LoadBalancer URL behind credentials.
        if auth_user.strip() and not auth_password:
            import secrets
            auth_password = secrets.token_urlsafe(12)
        config = build_rerun_serve_config(
            auth_user=auth_user,
            auth_password=auth_password,
            run_id=run_id,
            project=project or None,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
            namespace=namespace,
            port=port,
            name=name,
            cluster_context=cluster_context,
            rerun_image=rerun_image,
            service_type=service_type,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            rrd_s3_uri=rrd_uri,
            report_uri=report_uri,
        )
        if dry_run:
            manifest = build_rerun_serve_manifest(config)
            if output == OutputFormat.json:
                typer.echo(json.dumps(redact_rerun_serve_manifest(manifest), indent=2, sort_keys=True))
            else:
                typer.echo(json.dumps(redact_rerun_serve_manifest(manifest), indent=2, sort_keys=True))
            return
        resolved_kubeconfig = require_kubeconfig(
            cluster_name=cluster_context,
            kubeconfig=kubeconfig,
        )
        if destroy:
            result = destroy_rerun_serve(
                config,
                kubeconfig=resolved_kubeconfig,
                wait=destroy_wait,
            )
        else:
            if (
                service_type.strip().lower() in {"loadbalancer", "lb"}
                and not dry_run
                and not config.auth_enabled
            ):
                typer.echo(
                    "Warning: LoadBalancer exposes the Rerun web viewer without auth. "
                    "Pass --auth-user (and optionally --auth-password) to gate it, "
                    "or restrict access at the network layer.",
                    err=True,
                )
            result = apply_rerun_serve(config, kubeconfig=resolved_kubeconfig)
    except Sim2RealRerunServeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    if config.auth_enabled and not destroy:
        payload["auth_user"] = config.auth_user
        payload["auth_password"] = config.auth_password
        typer.echo(
            f"basic-auth enabled — user={config.auth_user} password={config.auth_password}",
            err=True,
        )
    if local_record and not destroy:
        loop_config = build_config_from_env(
            run_id=run_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
        )
        dest = resolve_local_rrd_path(
            run_id,
            override=str(local_rrd_path) if local_rrd_path is not None else "",
        )
        try:
            download_rrd_from_s3(loop_config, dest_path=dest)
        except Sim2RealRerunRegenError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        payload["local_rrd_path"] = str(dest)

    _emit_rerun_serve_result(payload, output=output)


@rerun_app.command("regen")
def rerun_regen_command(
    run_id: str = typer.Option(..., "--run-id", help="Completed Sim2Real run id."),
    project: str = typer.Option("", "--project", "-p", help="Project alias for storage resolution."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket override."),
    s3_prefix: str = typer.Option(DEFAULT_S3_PREFIX, "--s3-prefix", help="S3 prefix parent for runs."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint override."),
    local_dir: Optional[Path] = typer.Option(
        None,
        "--local-dir",
        help="Working directory for synced artifacts (default: /tmp/sim2real-regen/<run-id>).",
    ),
    local_rrd_path: Optional[Path] = typer.Option(
        None,
        "--local-rrd-path",
        help="Destination .rrd path (default: LOCAL_RRD_PATH or <local-dir>/reports/sim2real.rrd).",
    ),
    upload: bool = typer.Option(
        True,
        "--upload/--no-upload",
        help="Upload regenerated reports/sim2real.rrd (and held-out artifacts) to S3.",
    ),
    no_sync: bool = typer.Option(
        False,
        "--no-sync",
        help="Skip S3 download; use artifacts already under --local-dir.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Regenerate reports/sim2real.rrd locally from S3 artifacts (held-out PNG sync included)."""
    try:
        config = build_config_from_env(
            run_id=run_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
        )
        work_dir = local_dir or default_regen_local_dir(run_id)
        result = regen_sim2real_rrd(
            config,
            local_dir=work_dir,
            local_rrd_path=local_rrd_path,
            upload=upload,
            sync_inputs=not no_sync,
        )
    except Sim2RealRerunRegenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = result.to_dict()
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"local_rrd_path: {payload['local_rrd_path']}")
    typer.echo(f"heldout_frame_count: {payload['heldout_frame_count']}")
    if payload.get("upload_uri"):
        typer.echo(f"upload_uri: {payload['upload_uri']}")


@rerun_app.command("heldout-only")
def rerun_heldout_only_command(
    run_id: str = typer.Option(..., "--run-id", help="Existing Sim2Real run id."),
    project: str = typer.Option("", "--project", "-p", help="Project alias for storage resolution."),
    s3_bucket: str = typer.Option("", "--s3-bucket", help="S3 bucket override."),
    s3_prefix: str = typer.Option(DEFAULT_S3_PREFIX, "--s3-prefix", help="S3 prefix parent for runs."),
    s3_endpoint: str = typer.Option("", "--s3-endpoint", help="S3-compatible endpoint override."),
    local_dir: Optional[Path] = typer.Option(
        None,
        "--local-dir",
        help="Local working directory (default: /tmp/sim2real-regen/<run-id>).",
    ),
    outer_iteration: int = typer.Option(1, "--outer-iteration", help="Outer loop index for stage 10."),
    no_publish: bool = typer.Option(
        False,
        "--no-publish",
        help="Skip uploading held-out report/renders to the run prefix on S3.",
    ),
    output: OutputFormat = typer.Option(OutputFormat.text, "--output", help="Output format."),
) -> None:
    """Re-run Isaac held-out eval (stage 10) on cluster for an existing run (~5–15 min)."""
    try:
        config = build_config_from_env(
            run_id=run_id,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint=s3_endpoint,
        )
        work_dir = local_dir or default_regen_local_dir(run_id)
        report = rerun_heldout_eval_only(
            config,
            local_dir=work_dir,
            outer_iteration=outer_iteration,
            publish=not no_publish,
        )
    except Sim2RealRerunRegenError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    payload = {
        "run_id": run_id,
        "success_rate": report.get("success_rate"),
        "render_manifest_episodes": len((report.get("render_manifest") or {}).get("episodes") or []),
        "sim_backend": report.get("sim_backend"),
        "rollout_backend": report.get("rollout_backend"),
    }
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"run_id: {run_id}")
    typer.echo(f"success_rate: {payload['success_rate']}")
    typer.echo(f"render_manifest_episodes: {payload['render_manifest_episodes']}")
