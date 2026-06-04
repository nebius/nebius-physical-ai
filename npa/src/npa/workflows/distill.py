"""Expert distillation workflow orchestrator.

Ties the full sim-to-real pipeline into a single sequential workflow:
    1. Train teacher (PPO with privileged state in Genesis)
    2. Generate demos (teacher rollouts with cameras + domain randomization)
    3. Convert demos (numpy arrays → LeRobotDataset v3)
    4. Train student (vision-only ACT policy via LeRobot)
    5. Eval student (camera-only eval in Genesis)

Each stage uploads artifacts to object storage under a unique run ID.
The workflow can be executed locally or orchestrated across VMs via SSH.

Execution model:
    - Stages 1-3 run on a GPU sim VM (Genesis + camera rendering)
    - Stage 4 runs on a GPU training VM (LeRobot + ACT)
    - Stage 5 runs on the sim VM (Genesis + student inference)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DistillationError(Exception):
    pass


STAGES = [
    "train_teacher",
    "generate_demos",
    "convert",
    "train_student",
    "eval_student",
]


@dataclass
class RunConfig:
    """Configuration for a distillation run."""

    run_id: str
    project: str | None
    robot: str
    task: str
    n_envs: int
    # Per-stage config
    teacher_max_iterations: int = 500
    demo_domain_randomize: bool = True
    demo_fps: int = 20
    demo_seed: int = 42
    student_policy: str = "act"
    student_epochs: int = 100
    student_batch_size: int = 64
    eval_n_episodes: int = 1024
    eval_seed: int = 7777  # Must differ from demo_seed for held-out evaluation
    # Action space
    action_space: str = "cartesian"  # "cartesian" or "joint"
    # Storage
    s3_bucket: str = ""
    s3_prefix: str = ""
    # Remote execution
    sim_workbench: str = ""    # Workbench name for sim stages (genesis env)
    train_workbench: str = ""  # Workbench name for training stages (lerobot)


def generate_run_id() -> str:
    """Generate a unique run ID from timestamp + short hash."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha256(f"{ts}-{time.time_ns()}".encode()).hexdigest()[:8]
    return f"{ts}-{h}"


def run_distillation(
    project: str | None = None,
    robot: str = "franka_panda",
    task: str = "pick_place",
    n_envs: int = 4096,
    s3_bucket: str = "",
    remote: bool = False,
    sim_workbench: str = "",
    train_workbench: str = "",
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Run the full expert distillation pipeline.

    In local mode, executes all stages sequentially on the current machine.
    In remote mode, orchestrates across VMs via SSH with S3 artifact handoff
    between the sim VM (stages 1-3, 5) and training VM (stage 4).

    Args:
        project: NPA project alias for VM lookup.
        robot: Robot type identifier.
        task: Task name.
        n_envs: Parallel environments for simulation stages.
        s3_bucket: S3 bucket for artifact storage (required for remote mode).
        remote: If True, execute on remote VMs via SSH.
        sim_workbench: Workbench name for sim stages (genesis). If empty,
            uses default workbench.
        train_workbench: Workbench name for training stages (lerobot). If
            empty, uses same workbench as sim.
        action_space: "cartesian" (4D: delta xyz + gripper) or
            "joint" (8D: delta joint positions + gripper). Passed through
            to all Genesis stages.

    Returns:
        Result dict with run_id and per-stage status.

    Raises:
        DistillationError: If any stage fails.
    """
    if remote and not s3_bucket:
        raise DistillationError(
            "Remote mode requires --s3-bucket for artifact handoff between VMs."
        )

    run_id = generate_run_id()
    base_dir = Path(f"./runs/{run_id}")
    base_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        run_id=run_id,
        project=project,
        robot=robot,
        task=task,
        n_envs=n_envs,
        s3_bucket=s3_bucket,
        s3_prefix=f"distill/{run_id}",
        sim_workbench=sim_workbench,
        train_workbench=train_workbench or sim_workbench,
        action_space=action_space,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "config": {
            "robot": robot,
            "task": task,
            "n_envs": n_envs,
            "remote": remote,
            "action_space": action_space,
        },
        "stages": {},
    }

    # Save run config
    config_path = base_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(result["config"], f, indent=2)

    if remote:
        return _run_remote(cfg, base_dir, result)
    return _run_local(cfg, base_dir, result)


def _run_local(
    cfg: RunConfig,
    base_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Execute all stages locally in sequence."""
    teacher_dir = base_dir / "teacher"
    demos_dir = base_dir / "demos"
    dataset_dir = base_dir / "dataset"
    student_dir = base_dir / "student"
    eval_dir = base_dir / "eval"
    failed = False

    # ── Stage 1: Train teacher ──────────────────────────────────────
    logger.info("[1/5] Training teacher...")
    try:
        from npa.genesis.train_teacher import train_teacher

        stage_result = train_teacher(
            n_envs=cfg.n_envs,
            max_iterations=cfg.teacher_max_iterations,
            output_dir=teacher_dir,
            log_dir=base_dir / "logs" / "teacher",
            action_space=cfg.action_space,
        )
        result["stages"]["train_teacher"] = {
            "status": "success",
            **stage_result,
        }
        teacher_checkpoint = Path(stage_result["checkpoint_path"])
    except Exception as exc:
        result["stages"]["train_teacher"] = {
            "status": "failed",
            "error": str(exc),
        }
        failed = True
        _save_result(base_dir, result)
        raise DistillationError(
            f"Stage train_teacher failed: {exc}"
        ) from exc

    # ── Stage 2: Generate demos ─────────────────────────────────────
    logger.info("[2/5] Generating demonstrations...")
    try:
        from npa.genesis.generate_demos import generate_demos

        stage_result = generate_demos(
            checkpoint_path=teacher_checkpoint,
            n_envs=cfg.n_envs,
            output_dir=demos_dir,
            domain_randomize=cfg.demo_domain_randomize,
            fps=cfg.demo_fps,
            seed=cfg.demo_seed,
            allow_failure_demos=False,
            action_space=cfg.action_space,
        )
        result["stages"]["generate_demos"] = {
            "status": "success",
            **stage_result,
        }
        if stage_result.get("includes_failures"):
            logger.warning(
                "Demo dataset includes non-successful rollouts "
                "(teacher_success_rate=%.2f%%). Student will train on failure "
                "trajectories — this may degrade distillation quality.",
                stage_result.get("teacher_success_rate", 0) * 100,
            )
    except Exception as exc:
        result["stages"]["generate_demos"] = {
            "status": "failed",
            "error": str(exc),
        }
        failed = True
        _save_result(base_dir, result)
        raise DistillationError(
            f"Stage generate_demos failed: {exc}"
        ) from exc

    # ── Stage 3: Convert to LeRobotDataset ──────────────────────────
    logger.info("[3/5] Converting demos to LeRobotDataset v3...")
    try:
        from npa.adapter.sim_to_lerobot import convert

        convert(
            demos_dir,
            dataset_dir,
            fps=cfg.demo_fps,
            robot_type=cfg.robot,
            task=_task_description(cfg.task),
        )
        result["stages"]["convert"] = {
            "status": "success",
            "dataset_path": str(dataset_dir),
        }
    except Exception as exc:
        result["stages"]["convert"] = {
            "status": "failed",
            "error": str(exc),
        }
        failed = True
        _save_result(base_dir, result)
        raise DistillationError(
            f"Stage convert failed: {exc}"
        ) from exc

    # ── Stage 4: Train student ──────────────────────────────────────
    logger.info("[4/5] Training student policy...")
    try:
        from npa.lerobot.train_student import train_student

        stage_result = train_student(
            dataset_path=dataset_dir,
            output_dir=student_dir,
            policy_type=cfg.student_policy,
            num_epochs=cfg.student_epochs,
            batch_size=cfg.student_batch_size,
        )
        result["stages"]["train_student"] = {
            "status": "success",
            **stage_result,
        }
        student_checkpoint = Path(stage_result["checkpoint_path"])
    except Exception as exc:
        result["stages"]["train_student"] = {
            "status": "failed",
            "error": str(exc),
        }
        failed = True
        _save_result(base_dir, result)
        raise DistillationError(
            f"Stage train_student failed: {exc}"
        ) from exc

    # ── Teacher eval (held-out baseline) ──────────────────────────────
    logger.info("Evaluating teacher under held-out conditions...")
    teacher_success_rate = None
    try:
        from npa.genesis.generate_demos import eval_teacher

        teacher_success_rate = eval_teacher(
            checkpoint_path=teacher_checkpoint,
            n_envs=min(cfg.n_envs, cfg.eval_n_episodes),
            seed=cfg.eval_seed,
            action_space=cfg.action_space,
        )
        result["stages"]["eval_teacher"] = {
            "status": "success",
            "teacher_success_rate": round(teacher_success_rate, 4),
        }
    except Exception as exc:
        logger.warning("Teacher eval failed: %s — distillation gap will not be computed.", exc)
        result["stages"]["eval_teacher"] = {
            "status": "failed",
            "error": str(exc),
        }

    # ── Stage 5: Eval student ───────────────────────────────────────
    logger.info("[5/5] Evaluating student policy...")
    try:
        from npa.genesis.eval_student import eval_student

        stage_result = eval_student(
            checkpoint_path=student_checkpoint,
            n_envs=min(cfg.n_envs, cfg.eval_n_episodes),
            n_episodes=cfg.eval_n_episodes,
            output_dir=eval_dir,
            seed=cfg.eval_seed,
            teacher_success_rate=teacher_success_rate,
            action_space=cfg.action_space,
        )
        result["stages"]["eval_student"] = {
            "status": "success",
            **stage_result,
        }
    except Exception as exc:
        result["stages"]["eval_student"] = {
            "status": "failed",
            "error": str(exc),
        }
        failed = True
        _save_result(base_dir, result)
        raise DistillationError(
            f"Stage eval_student failed: {exc}"
        ) from exc

    # ── Upload to S3 (if configured) ────────────────────────────────
    if cfg.s3_bucket:
        _upload_artifacts(cfg, base_dir, result)

    result["status"] = "failed" if failed else "success"
    _save_result(base_dir, result)
    return result


def _run_remote(
    cfg: RunConfig,
    base_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Execute stages on remote VMs via SSH with S3 artifact handoff.

    Stages 1-3 and 5 run on the sim VM (genesis conda env).
    Stage 4 runs on the train VM (lerobot conda env).
    Artifacts are transferred between VMs via S3.
    """
    from npa.clients.config import ConfigError, resolve_config
    from npa.clients.ssh import SSHClient, SSHError

    # Resolve SSH configs for sim and train workbenches
    try:
        sim_cfg = resolve_config(
            project=cfg.project or None,
            name=cfg.sim_workbench or None,
        )
    except ConfigError as exc:
        raise DistillationError(
            f"Cannot resolve sim VM config: {exc}"
        ) from exc

    try:
        train_cfg = resolve_config(
            project=cfg.project or None,
            name=cfg.train_workbench or None,
        )
    except ConfigError as exc:
        raise DistillationError(
            f"Cannot resolve train VM config: {exc}"
        ) from exc

    sim_ssh = SSHClient(sim_cfg.ssh)
    train_ssh = SSHClient(train_cfg.ssh)

    remote_base = f"/opt/npa/runs/{cfg.run_id}"
    s3_base = f"{cfg.s3_bucket.rstrip('/')}/{cfg.s3_prefix}"

    # Conda env activation prefixes
    genesis_activate = "eval \"$(conda shell.bash hook)\" && conda activate genesis && "
    lerobot_activate = "eval \"$(conda shell.bash hook)\" && conda activate lerobot && "

    # Map stage → (ssh_client, conda_prefix, command)
    stage_plan: dict[str, tuple[SSHClient, str, str]] = {
        "train_teacher": (
            sim_ssh,
            genesis_activate,
            f"npa workbench genesis train-teacher "
            f"--n-envs {cfg.n_envs} "
            f"--max-iterations {cfg.teacher_max_iterations} "
            f"--action-space {cfg.action_space} "
            f"--output {remote_base}/teacher/",
        ),
        "generate_demos": (
            sim_ssh,
            genesis_activate,
            f"npa workbench genesis generate-demos "
            f"--checkpoint {remote_base}/teacher/model.pt "
            f"--n-envs {cfg.n_envs} "
            f"--seed {cfg.demo_seed} "
            f"--action-space {cfg.action_space} "
            f"--output {remote_base}/demos/",
        ),
        "convert": (
            sim_ssh,
            genesis_activate,
            f"npa adapter convert "
            f"--input {remote_base}/demos/ "
            f"--output {remote_base}/dataset/ "
            f"--fps {cfg.demo_fps} "
            f"--robot {cfg.robot}",
        ),
        "train_student": (
            train_ssh,
            lerobot_activate,
            f"npa workbench lerobot train-student "
            f"--dataset {remote_base}/dataset/ "
            f"--policy {cfg.student_policy} "
            f"--epochs {cfg.student_epochs} "
            f"--batch-size {cfg.student_batch_size} "
            f"--output-dir {remote_base}/student/",
        ),
        "eval_student": (
            sim_ssh,
            genesis_activate,
            f"npa workbench genesis eval-student "
            f"--checkpoint {remote_base}/student/ "
            f"--n-envs {min(cfg.n_envs, cfg.eval_n_episodes)} "
            f"--n-episodes {cfg.eval_n_episodes} "
            f"--seed {cfg.eval_seed} "
            f"--action-space {cfg.action_space} "
            f"--output {remote_base}/eval/",
        ),
    }

    for stage_name in STAGES:
        ssh, conda_prefix, cmd = stage_plan[stage_name]

        # S3 downloads before cross-VM stages
        if stage_name == "train_student" and train_cfg.ssh.host != sim_cfg.ssh.host:
            logger.info("[%s] Downloading dataset from S3 to train VM...", stage_name)
            _s3_sync_dir(
                train_ssh, lerobot_activate,
                direction="download",
                s3_uri=f"{s3_base}/dataset/",
                local_path=f"{remote_base}/dataset/",
            )

        if stage_name == "eval_student" and train_cfg.ssh.host != sim_cfg.ssh.host:
            logger.info("[%s] Downloading student checkpoint from S3 to sim VM...", stage_name)
            _s3_sync_dir(
                sim_ssh, genesis_activate,
                direction="download",
                s3_uri=f"{s3_base}/student/",
                local_path=f"{remote_base}/student/",
            )

        logger.info("[%s] Running on %s: %s", stage_name, ssh._config.host, cmd)

        full_cmd = (
            f"mkdir -p {remote_base} && "
            f"{conda_prefix}"
            f"{cmd}"
        )

        try:
            exit_code, stdout, stderr = ssh.run(full_cmd, stream=True)
        except SSHError as exc:
            result["stages"][stage_name] = {
                "status": "failed",
                "error": f"SSH error: {exc}",
            }
            _save_result(base_dir, result)
            raise DistillationError(
                f"Stage {stage_name} failed (SSH): {exc}"
            ) from exc

        if exit_code != 0:
            result["stages"][stage_name] = {
                "status": "failed",
                "exit_code": exit_code,
                "stderr": stderr.strip()[-500:] if stderr else "",
            }
            _save_result(base_dir, result)
            raise DistillationError(
                f"Stage {stage_name} failed (exit {exit_code})"
            )

        result["stages"][stage_name] = {
            "status": "success",
            "exit_code": exit_code,
        }

        # S3 uploads after cross-VM stages
        if stage_name == "convert" and train_cfg.ssh.host != sim_cfg.ssh.host:
            logger.info("[%s] Uploading dataset to S3...", stage_name)
            _s3_sync_dir(
                sim_ssh, genesis_activate,
                direction="upload",
                s3_uri=f"{s3_base}/dataset/",
                local_path=f"{remote_base}/dataset/",
            )

        if stage_name == "train_student" and train_cfg.ssh.host != sim_cfg.ssh.host:
            logger.info("[%s] Uploading student checkpoint to S3...", stage_name)
            _s3_sync_dir(
                train_ssh, lerobot_activate,
                direction="upload",
                s3_uri=f"{s3_base}/student/",
                local_path=f"{remote_base}/student/",
            )

    result["status"] = "success"
    _save_result(base_dir, result)
    return result


def _s3_sync_dir(
    ssh: Any,
    conda_prefix: str,
    *,
    direction: str,
    s3_uri: str,
    local_path: str,
) -> None:
    """Upload or download a directory via S3 on a remote VM."""
    from npa.clients.ssh import SSHError
    from urllib.parse import urlparse

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    if direction == "upload":
        script = (
            f"import boto3, os, pathlib; "
            f"s3 = boto3.client('s3', "
            f"endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', ''), "
            f"aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''), "
            f"aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')); "
            f"base = pathlib.Path('{local_path}'); "
            f"[s3.upload_file(str(f), '{bucket}', '{prefix}' + str(f.relative_to(base))) "
            f"for f in base.rglob('*') if f.is_file()]; "
            f"print('s3_sync_upload_done')"
        )
    else:
        script = (
            f"import boto3, os, pathlib; "
            f"s3 = boto3.client('s3', "
            f"endpoint_url=os.environ.get('NEBIUS_S3_ENDPOINT', ''), "
            f"aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', ''), "
            f"aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', '')); "
            f"pag = s3.get_paginator('list_objects_v2'); "
            f"dest = pathlib.Path('{local_path}'); "
            f"[("
            f"os.makedirs(str((dest / o['Key'][len('{prefix}'):]).parent), exist_ok=True), "
            f"s3.download_file('{bucket}', o['Key'], str(dest / o['Key'][len('{prefix}'):]))) "
            f"for page in pag.paginate(Bucket='{bucket}', Prefix='{prefix}') "
            f"for o in page.get('Contents', []) "
            f"if o['Key'][len('{prefix}'):]"
            f"]; "
            f"print('s3_sync_download_done')"
        )

    cmd = (
        f"mkdir -p {local_path} && "
        f"{conda_prefix}"
        f"python3 -c \"{script}\""
    )

    try:
        exit_code, stdout, _ = ssh.run(cmd)
    except SSHError as exc:
        raise DistillationError(f"S3 sync ({direction}) failed: {exc}") from exc

    if exit_code != 0:
        raise DistillationError(
            f"S3 sync ({direction}) failed (exit {exit_code})"
        )


def _upload_artifacts(
    cfg: RunConfig,
    base_dir: Path,
    result: dict[str, Any],
) -> None:
    """Upload run artifacts to S3."""
    try:
        from npa.clients.config import resolve_config
        from npa.clients.storage import StorageClient

        wb_cfg = resolve_config(project=cfg.project or None)
        store = StorageClient(
            endpoint_url=wb_cfg.storage.endpoint_url,
            aws_access_key_id=wb_cfg.storage.aws_access_key_id,
            aws_secret_access_key=wb_cfg.storage.aws_secret_access_key,
        )
        uri = store.upload_directory(
            str(base_dir),
            cfg.s3_bucket,
            remote_prefix=cfg.s3_prefix,
        )
        result["s3_uri"] = uri
        logger.info("Artifacts uploaded to %s", uri)
    except Exception as exc:
        logger.warning("Failed to upload artifacts to S3: %s", exc)
        result["s3_upload_error"] = str(exc)


def _save_result(base_dir: Path, result: dict[str, Any]) -> None:
    """Persist the result dict to disk."""
    path = base_dir / "result.json"
    with path.open("w") as f:
        json.dump(result, f, indent=2)


def _task_description(task_name: str) -> str:
    """Map task short names to full descriptions."""
    task_map = {
        "pick_place": "Pick and place cube to target",
        "push": "Push block to target position",
        "stack": "Stack cube on top of another cube",
    }
    return task_map.get(task_name, task_name)


# ── Status and log queries ──────────────────────────────────────────────


def get_run_status(run_id: str) -> dict[str, Any]:
    """Read the status of a workflow run from disk."""
    base_dir = Path(f"./runs/{run_id}")
    result_path = base_dir / "result.json"

    if not result_path.exists():
        raise DistillationError(f"Run not found: {run_id}")

    with result_path.open() as f:
        return json.load(f)


def get_stage_logs(run_id: str, stage: str) -> str:
    """Read logs for a specific stage of a workflow run."""
    if stage not in STAGES:
        raise DistillationError(
            f"Unknown stage: '{stage}'. Available: {', '.join(STAGES)}"
        )

    base_dir = Path(f"./runs/{run_id}")

    # Try common log locations
    log_paths = [
        base_dir / "logs" / stage / "log.txt",
        base_dir / stage / "train.log",
        base_dir / stage / "log.txt",
    ]

    for p in log_paths:
        if p.exists():
            return p.read_text()

    # Fallback: return result.json stage info
    result_path = base_dir / "result.json"
    if result_path.exists():
        with result_path.open() as f:
            result = json.load(f)
        stage_info = result.get("stages", {}).get(stage)
        if stage_info:
            return json.dumps(stage_info, indent=2)

    raise DistillationError(
        f"No logs found for stage '{stage}' in run {run_id}"
    )
