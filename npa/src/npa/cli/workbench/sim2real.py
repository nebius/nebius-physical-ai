"""CLI for the Sim2Real VLM-to-RL workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

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
    """Run the full 13-stage Sim2Real workflow."""

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
