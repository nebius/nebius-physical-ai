from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pyarrow.parquet as pq
import pytest

from npa.clients.serverless import EndpointNotFoundError, ServerlessClient
from npa.serverless_common import build_serverless_output_upload_cmd, resolve_subnet


PROJECT_ALIAS = "eu-north1"
PROJECT_ID = "project-test-00000000000"
BUCKET = "your-bucket-name"
ENDPOINT_URL = "https://storage.eu-north1.nebius.cloud"
GENESIS_IMAGE = "cr.eu-north1.nebius.cloud/your-registry-id/npa-genesis:0.4.6"
LEROBOT_IMAGE = "cr.eu-north1.nebius.cloud/your-registry-id/npa-lerobot:0.5.1"
GENESIS_GPU_TYPE = "gpu-h200-sxm"
GENESIS_GPU_PRESET = "1gpu-16vcpu-200gb"
LEROBOT_GPU_TYPE = "h200"
WORKBENCH_NAME = "h200"
SMOKE_NUM_DEMOS = 4
SMOKE_N_ENVS = 4
SMOKE_SEED = 42
SMOKE_TRAIN_STEPS = 50
SMOKE_EVAL_EPISODES = 2
ACTION_SPACE = "joint"
FPS = 20
JOB_PREFIX = "npa-pipes2r"
POLL_INTERVAL = float(os.environ.get("NPA_E2E_PIPELINE_POLL_INTERVAL", "30"))
MAX_WAIT = float(os.environ.get("NPA_E2E_PIPELINE_MAX_WAIT", "7200"))
STARTING_WAIT = float(os.environ.get("NPA_E2E_PIPELINE_STARTING_WAIT", "3600"))
SECRET_ENV_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
}


@dataclass(frozen=True)
class PipelineSettings:
    project_alias: str
    project_id: str
    bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    genesis_image: str = GENESIS_IMAGE
    lerobot_image: str = LEROBOT_IMAGE

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        return cls(
            project_alias=os.environ.get("NPA_E2E_PROJECT", PROJECT_ALIAS),
            project_id=os.environ.get("NPA_E2E_SERVERLESS_PROJECT", PROJECT_ID),
            bucket=os.environ.get("NPA_E2E_S3_BUCKET", BUCKET),
            endpoint_url=os.environ.get("NPA_E2E_S3_ENDPOINT", ENDPOINT_URL),
            access_key_id=os.environ.get("NPA_E2E_S3_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("NPA_E2E_S3_SECRET_ACCESS_KEY", ""),
            genesis_image=os.environ.get("NPA_E2E_GENESIS_IMAGE", GENESIS_IMAGE),
            lerobot_image=os.environ.get("NPA_E2E_LEROBOT_IMAGE", LEROBOT_IMAGE),
        )


@dataclass
class PipelineJob:
    stage: str
    job_name: str
    output_path: str
    job_id: str = ""


def test_sim_to_real_pipeline_shape() -> None:
    settings = PipelineSettings(
        project_alias=PROJECT_ALIAS,
        project_id=PROJECT_ID,
        bucket=BUCKET,
        endpoint_url=ENDPOINT_URL,
        access_key_id="access",
        secret_access_key="secret",
    )
    pipeline_id = "w7pipes2r-shape"
    s3_base = _s3_base(settings, pipeline_id)

    genesis_command = _genesis_demo_container_command()
    assert "train_teacher" in genesis_command
    assert "generate_demos" in genesis_command
    assert "domain_randomize=True" in genesis_command
    assert "allow_failure_demos=True" in genesis_command

    adapter_command = _adapter_command(
        input_path=f"{s3_base}/stage1-demos/",
        output_path=f"{s3_base}/stage2-lerobot-dataset/",
    )
    assert adapter_command[:3] == ["adapter", "convert", "--input-path"]
    assert "--output-path" in adapter_command

    lerobot_command = _lerobot_train_command(
        settings,
        job_name=f"{JOB_PREFIX}-shape-lerobot",
        input_path=f"{s3_base}/stage2-lerobot-dataset/",
        output_path=f"{s3_base}/stage3-student-checkpoint/",
    )
    assert lerobot_command[:7] == [
        "workbench",
        "lerobot",
        "-p",
        PROJECT_ALIAS,
        "-n",
        WORKBENCH_NAME,
        "train",
    ]
    assert "--runtime" in lerobot_command
    assert "serverless" in lerobot_command
    assert "--input-path" in lerobot_command
    assert "--subnet-id" not in lerobot_command

    eval_command = _genesis_eval_container_command()
    assert "eval_student" in eval_command
    assert "NPA_STAGE_INPUT_PATH" in eval_command
    assert "domain_randomize=True" in eval_command


@pytest.mark.gpu
@pytest.mark.e2e
@pytest.mark.e2e_pipeline
def test_sim_to_real_pipeline_e2e(tmp_path: Path) -> None:
    """Validate Genesis -> SimToLeRobot -> LeRobot -> Genesis composition.

    This is a real Nebius Serverless AI composition test. Genesis demo and eval
    stages submit direct Jobs because the current Genesis CLI has serverless
    support only for `train-teacher`; adapter conversion runs inline via the
    public adapter CLI; LeRobot training uses the serverless train CLI.
    """

    settings = PipelineSettings.from_env()
    _require_pipeline_e2e(settings)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pipeline_id = f"w7pipes2r-{timestamp}-{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path("/tmp") / f"pipeline-sim-to-real-{pipeline_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s3_base = _s3_base(settings, pipeline_id)
    demo_output = f"{s3_base}/stage1-demos/"
    dataset_output = f"{s3_base}/stage2-lerobot-dataset/"
    checkpoint_output = f"{s3_base}/stage3-student-checkpoint/"
    eval_output = f"{s3_base}/stage4-eval-metrics/"
    jobs: list[PipelineJob] = []

    _write_json(
        artifacts_dir / "manifest.json",
        {
            "pipeline_id": pipeline_id,
            "settings": _redacted_settings(settings),
            "s3_base": s3_base,
            "stages": {
                "stage1_genesis_demos": demo_output,
                "stage2_adapter_dataset": dataset_output,
                "stage3_lerobot_checkpoint": checkpoint_output,
                "stage4_genesis_eval": eval_output,
            },
        },
    )

    try:
        genesis_demo_job = _submit_genesis_demo_generation(
            settings=settings,
            output_path=demo_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(genesis_demo_job)
        final = _poll_job(settings.project_id, genesis_demo_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_genesis_demos_schema(settings, demo_output, artifacts_dir)

        adapter_result = _run_adapter(
            settings=settings,
            input_path=demo_output,
            output_path=dataset_output,
            artifacts_dir=artifacts_dir,
        )
        assert adapter_result.returncode == 0, _format_result(adapter_result)
        _assert_lerobot_dataset_schema(settings, dataset_output, artifacts_dir)

        lerobot_job = _submit_lerobot_train(
            settings=settings,
            input_path=dataset_output,
            output_path=checkpoint_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(lerobot_job)
        final = _poll_job(settings.project_id, lerobot_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_checkpoint_loadable(settings, checkpoint_output, artifacts_dir)

        genesis_eval_job = _submit_genesis_eval(
            settings=settings,
            checkpoint_path=checkpoint_output,
            output_path=eval_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(genesis_eval_job)
        final = _poll_job(settings.project_id, genesis_eval_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_eval_metrics_schema(settings, eval_output, artifacts_dir)
    finally:
        _write_json(
            artifacts_dir / "jobs.json",
            [asdict(job) for job in jobs],
        )
        for job in reversed(jobs):
            _cleanup_job(settings.project_id, job.job_id or job.job_name, artifacts_dir)


def _require_pipeline_e2e(settings: PipelineSettings) -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    missing = [
        key
        for key, value in {
            "NPA_E2E_SERVERLESS_PROJECT": settings.project_id,
            "NPA_E2E_S3_ACCESS_KEY_ID": settings.access_key_id,
            "NPA_E2E_S3_SECRET_ACCESS_KEY": settings.secret_access_key,
        }.items()
        if not value
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed; SimToLeRobot adapter needs it")


def _s3_base(settings: PipelineSettings, pipeline_id: str) -> str:
    return f"s3://{settings.bucket}/pipeline-sim-to-real/{pipeline_id}"


def _redacted_settings(settings: PipelineSettings) -> dict[str, str]:
    return {
        "project_alias": settings.project_alias,
        "project_id": settings.project_id,
        "bucket": settings.bucket,
        "endpoint_url": settings.endpoint_url,
        "access_key_id": "***" if settings.access_key_id else "",
        "secret_access_key": "***" if settings.secret_access_key else "",
        "genesis_image": settings.genesis_image,
        "lerobot_image": settings.lerobot_image,
    }


def _submit_genesis_demo_generation(
    *,
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage1_genesis_demos",
        job_name=_job_name("genesis-demos"),
        output_path=output_path,
    )
    client = ServerlessClient()
    info = client.create_job(
        project_id=settings.project_id,
        name=job.job_name,
        image=settings.genesis_image,
        command=_genesis_demo_container_command(),
        gpu_type=GENESIS_GPU_TYPE,
        gpu_count=1,
        preset=GENESIS_GPU_PRESET,
        subnet_id=_subnet_id(settings.project_id),
        output_path=output_path,
        env=_safe_job_env(settings, job.job_name),
        extra_env=_secret_job_env(settings),
        timeout="2h",
    )
    job.job_id = info.id
    _write_json(artifacts_dir / "stage1-submit.json", _redact_job_raw(info.raw))
    visible = _wait_for_visible_job(settings.project_id, info.id)
    _write_json(artifacts_dir / "stage1-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "Genesis demo Job spec.subnet_id is empty"
    return job


def _submit_lerobot_train(
    *,
    settings: PipelineSettings,
    input_path: str,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage3_lerobot_train",
        job_name=_job_name("lerobot-train"),
        output_path=output_path,
    )
    result = _run_npa(
        _lerobot_train_command(
            settings,
            job_name=job.job_name,
            input_path=input_path,
            output_path=output_path,
        ),
        artifacts_dir=artifacts_dir,
        label="stage3-submit",
        timeout=int(os.environ.get("NPA_E2E_PIPELINE_SUBMIT_TIMEOUT", "900")),
    )
    assert result.returncode == 0, _format_result(result)
    payload = json.loads(result.stdout)
    assert payload["status"] == "submitted"
    job.job_id = str(payload["job_id"])
    visible = _wait_for_visible_job(settings.project_id, job.job_id)
    _write_json(artifacts_dir / "stage3-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "LeRobot train Job spec.subnet_id is empty"
    return job


def _submit_genesis_eval(
    *,
    settings: PipelineSettings,
    checkpoint_path: str,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage4_genesis_eval",
        job_name=_job_name("genesis-eval"),
        output_path=output_path,
    )
    client = ServerlessClient()
    info = client.create_job(
        project_id=settings.project_id,
        name=job.job_name,
        image=settings.genesis_image,
        command=_genesis_eval_container_command(),
        gpu_type=GENESIS_GPU_TYPE,
        gpu_count=1,
        preset=GENESIS_GPU_PRESET,
        subnet_id=_subnet_id(settings.project_id),
        output_path=output_path,
        env={
            **_safe_job_env(settings, job.job_name),
            "NPA_STAGE_INPUT_PATH": checkpoint_path,
        },
        extra_env=_secret_job_env(settings),
        timeout="2h",
    )
    job.job_id = info.id
    _write_json(artifacts_dir / "stage4-submit.json", _redact_job_raw(info.raw))
    visible = _wait_for_visible_job(settings.project_id, info.id)
    _write_json(artifacts_dir / "stage4-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "Genesis eval Job spec.subnet_id is empty"
    return job


def _genesis_demo_container_command() -> str:
    local_dir = "/tmp/npa-pipeline-genesis-demos"
    script = f"""
import json
import os
from pathlib import Path

from npa.genesis.generate_demos import generate_demos
from npa.genesis.train_teacher import train_teacher

root = Path("/tmp/npa-pipeline-genesis")
teacher_dir = root / "teacher"
log_dir = root / "logs"
demos_dir = Path({local_dir!r})
teacher_dir.mkdir(parents=True, exist_ok=True)
demos_dir.mkdir(parents=True, exist_ok=True)

train_result = train_teacher(
    n_envs={SMOKE_N_ENVS},
    max_iterations=1,
    output_dir=teacher_dir,
    log_dir=log_dir,
    seed={SMOKE_SEED},
    action_space={ACTION_SPACE!r},
)
demos_result = generate_demos(
    checkpoint_path=Path(train_result["checkpoint_path"]),
    n_envs={SMOKE_N_ENVS},
    n_episodes={SMOKE_NUM_DEMOS},
    output_dir=demos_dir,
    domain_randomize=True,
    fps={FPS},
    seed={SMOKE_SEED},
    allow_failure_demos=True,
    action_space={ACTION_SPACE!r},
)
(demos_dir / "stage1_summary.json").write_text(
    json.dumps(
        {{
            "status": "success",
            "job": os.environ.get("NPA_JOB_NAME", ""),
            "train_result": train_result,
            "demos_result": demos_result,
            "domain_randomize": True,
            "action_space": {ACTION_SPACE!r},
        }},
        indent=2,
    )
)
print("NPA_PIPELINE_STAGE1_DEMOS_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    body = (
        "set -euo pipefail\n"
        "export PYTHONUNBUFFERED=1\n"
        f"python3 <<'PY'\n{script}\nPY\n"
        f"{build_serverless_output_upload_cmd(local_dir, '')}"
    )
    return _remote_bash(body)


def _genesis_eval_container_command() -> str:
    local_dir = "/tmp/npa-pipeline-genesis-eval"
    script = f"""
import json
import os
from pathlib import Path

from npa.clients.storage import StorageClient
from npa.genesis.eval_student import eval_student

checkpoint_uri = os.environ["NPA_STAGE_INPUT_PATH"]
checkpoint_dir = Path("/tmp/npa-pipeline-student-checkpoint")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
StorageClient.from_environment().download_directory(checkpoint_uri, str(checkpoint_dir))

output_dir = Path({local_dir!r})
output_dir.mkdir(parents=True, exist_ok=True)
result = eval_student(
    checkpoint_path=checkpoint_dir,
    n_envs={SMOKE_EVAL_EPISODES},
    n_episodes={SMOKE_EVAL_EPISODES},
    output_dir=output_dir,
    domain_randomize=True,
    seed={SMOKE_SEED + 1000},
    action_space={ACTION_SPACE!r},
)
(output_dir / "stage4_summary.json").write_text(json.dumps(result, indent=2))
print("NPA_PIPELINE_STAGE4_EVAL_DONE", os.environ.get("NPA_OUTPUT_PATH", ""), flush=True)
""".strip()
    body = (
        "set -euo pipefail\n"
        "export PYTHONUNBUFFERED=1\n"
        f"python3 <<'PY'\n{script}\nPY\n"
        f"{build_serverless_output_upload_cmd(local_dir, '')}"
    )
    return _remote_bash(body)


def _adapter_command(*, input_path: str, output_path: str) -> list[str]:
    return [
        "adapter",
        "convert",
        "--input-path",
        input_path,
        "--output-path",
        output_path,
        "--fps",
        str(FPS),
        "--robot",
        "franka_panda",
        "--task",
        "Pick and place cube to target",
    ]


def _lerobot_train_command(
    settings: PipelineSettings,
    *,
    job_name: str,
    input_path: str,
    output_path: str,
) -> list[str]:
    return [
        "workbench",
        "lerobot",
        "-p",
        settings.project_alias,
        "-n",
        WORKBENCH_NAME,
        "train",
        "--runtime",
        "serverless",
        "--project-id",
        settings.project_id,
        "--policy-type",
        "act",
        "--input-path",
        input_path,
        "--job-name",
        job_name,
        "--steps",
        str(SMOKE_TRAIN_STEPS),
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--gpu-type",
        LEROBOT_GPU_TYPE,
        "--gpu-count",
        "1",
        "--image",
        settings.lerobot_image,
        "--output-path",
        output_path,
        "--smoke",
        "--poll-interval",
        str(POLL_INTERVAL),
        "--wait-timeout",
        str(int(MAX_WAIT)),
        "--submit-only",
        "--output",
        "json",
    ]


def _run_adapter(
    *,
    settings: PipelineSettings,
    input_path: str,
    output_path: str,
    artifacts_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_npa(
        _adapter_command(input_path=input_path, output_path=output_path),
        artifacts_dir=artifacts_dir,
        label="stage2-adapter",
        timeout=int(os.environ.get("NPA_E2E_PIPELINE_ADAPTER_TIMEOUT", "1800")),
        env_overrides=_storage_env(settings),
    )


def _run_npa(
    args: list[str],
    *,
    artifacts_dir: Path,
    label: str,
    timeout: int,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    (artifacts_dir / f"{label}-command.json").write_text(
        json.dumps(args, indent=2) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [_npa_executable(), *args],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    (artifacts_dir / f"{label}-stdout.txt").write_text(result.stdout, encoding="utf-8")
    (artifacts_dir / f"{label}-stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def _npa_executable() -> str:
    script = Path(sys.executable).with_name("npa")
    return str(script) if script.exists() else "npa"


def _safe_job_env(settings: PipelineSettings, job_name: str) -> dict[str, str]:
    return {
        "NPA_JOB_NAME": job_name,
        "PYTHONUNBUFFERED": "1",
        "AWS_ENDPOINT_URL": settings.endpoint_url,
        "NEBIUS_S3_ENDPOINT": settings.endpoint_url,
    }


def _secret_job_env(settings: PipelineSettings) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": settings.access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.secret_access_key,
    }


def _storage_env(settings: PipelineSettings) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": settings.access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.secret_access_key,
        "AWS_ENDPOINT_URL": settings.endpoint_url,
        "NEBIUS_S3_ENDPOINT": settings.endpoint_url,
    }


def _remote_bash(script: str) -> str:
    return f"bash -lc {shlex.quote(script)}"


def _job_name(label: str) -> str:
    raw = f"{JOB_PREFIX}-{label}-{uuid.uuid4().hex[:8]}"
    return raw[:63]


def _subnet_id(project_id: str) -> str:
    return resolve_subnet(
        project_id=project_id,
        explicit_subnet_id=os.environ.get("NPA_E2E_SERVERLESS_SUBNET_ID", ""),
    )


def _wait_for_visible_job(project_id: str, job_id: str):
    client = ServerlessClient()
    deadline = time.monotonic() + 60
    last: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            return client.get_job(job_id, project_id)
        except Exception as exc:
            last = exc
            time.sleep(2)
    pytest.fail(f"Job {job_id} was not visible after submission: {last}")


def _poll_job(project_id: str, job_id: str, artifacts_dir: Path):
    client = ServerlessClient()
    deadline = time.monotonic() + MAX_WAIT
    startup_deadline = time.monotonic() + STARTING_WAIT
    last = None
    tick = 0
    while time.monotonic() <= deadline:
        tick += 1
        current = client.get_job(job_id, project_id)
        last = current
        _write_json(
            artifacts_dir / f"job-detail-{job_id}-tick-{tick:03d}.json",
            _redact_job_raw(current.raw),
        )
        _capture_logs(job_id, artifacts_dir / f"job-logs-{job_id}-tick-{tick:03d}.txt")
        if current.status in {"running", "succeeded", "failed", "cancelled"}:
            startup_deadline = 0
        if current.status in {"succeeded", "failed", "cancelled"}:
            return current
        if startup_deadline and time.monotonic() > startup_deadline:
            pytest.fail(
                f"Job {job_id} did not leave queue/startup within {STARTING_WAIT}s; "
                f"last={current.raw}"
            )
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} did not finish within {MAX_WAIT}s; last={last}")


def _capture_logs(job_id: str, path: Path) -> None:
    result = subprocess.run(
        ["nebius", "ai", "job", "logs", job_id, "--tail", "500", "--timestamps"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    path.write_text(result.stdout, encoding="utf-8")


def _cleanup_job(project_id: str, ref: str, artifacts_dir: Path) -> None:
    if not ref:
        return
    client = ServerlessClient()
    try:
        info = client.cancel_job(ref, project_id)
        job_id = info.id or ref
    except EndpointNotFoundError:
        return
    except Exception as exc:
        (artifacts_dir / f"cleanup-cancel-{ref}.err").write_text(
            str(exc),
            encoding="utf-8",
        )
        job_id = ref
    result = subprocess.run(
        ["nebius", "ai", "job", "delete", "--id", job_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    (artifacts_dir / f"cleanup-delete-{job_id}.log").write_text(
        result.stdout,
        encoding="utf-8",
    )
    orphan = subprocess.run(
        ["nebius", "ai", "job", "get", "--id", job_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (artifacts_dir / f"cleanup-orphan-check-{job_id}.log").write_text(
        orphan.stdout,
        encoding="utf-8",
    )


def _assert_genesis_demos_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage1-demos"
    _download_s3_prefix(settings, output_path, local_dir)
    episodes = sorted(path for path in local_dir.glob("episode_*") if path.is_dir())
    assert len(episodes) >= SMOKE_NUM_DEMOS
    for episode in episodes[:SMOKE_NUM_DEMOS]:
        arrays = {
            name: np.load(episode / name)
            for name in ("obs_workspace.npy", "obs_wrist.npy", "state.npy", "actions.npy")
        }
        frame_count = arrays["state.npy"].shape[0]
        assert frame_count > 0
        assert arrays["obs_workspace.npy"].shape[0] == frame_count
        assert arrays["obs_wrist.npy"].shape[0] == frame_count
        assert arrays["actions.npy"].shape[0] == frame_count
        assert arrays["obs_workspace.npy"].ndim == 4
        assert arrays["obs_workspace.npy"].shape[-1] == 3
        assert arrays["obs_workspace.npy"].dtype == np.uint8
        assert arrays["obs_wrist.npy"].dtype == np.uint8
        assert arrays["state.npy"].ndim == 2
        assert arrays["actions.npy"].ndim == 2
        assert arrays["state.npy"].shape[1] >= 2
        assert arrays["actions.npy"].shape[1] >= 2


def _assert_lerobot_dataset_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage2-lerobot-dataset"
    _download_s3_prefix(settings, output_path, local_dir)
    info_path = local_dir / "meta" / "info.json"
    data_path = local_dir / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = local_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    assert info_path.exists()
    assert (local_dir / "meta" / "tasks.parquet").exists()
    assert episodes_path.exists()
    assert data_path.exists()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] >= SMOKE_NUM_DEMOS
    assert info["total_frames"] > 0
    features = info["features"]
    for key in (
        "observation.images.workspace",
        "observation.images.wrist",
        "observation.state",
        "action",
    ):
        assert key in features
    table = pq.read_table(data_path)
    assert table.num_rows == info["total_frames"]
    for column in (
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "index",
        "task_index",
    ):
        assert column in table.column_names


def _assert_checkpoint_loadable(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage3-student-checkpoint"
    _download_s3_prefix(settings, output_path, local_dir)
    pretrained_dirs = [
        path.parent for path in local_dir.rglob("config.json")
        if (path.parent / "model.safetensors").exists()
    ]
    assert pretrained_dirs, f"No LeRobot pretrained_model directory under {output_path}"
    for pretrained_dir in pretrained_dirs[:1]:
        config = json.loads((pretrained_dir / "config.json").read_text(encoding="utf-8"))
        policy_type = str(config.get("type", config.get("_target_", ""))).lower()
        assert "act" in policy_type


def _assert_eval_metrics_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage4-eval-metrics"
    _download_s3_prefix(settings, output_path, local_dir)
    metrics_files = sorted(local_dir.glob("eval_*.json"))
    assert metrics_files, f"No Genesis eval metrics JSON under {output_path}"
    metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert "success_rate" in metrics
    assert 0.0 <= float(metrics["success_rate"]) <= 1.0
    assert int(metrics.get("episode_count", metrics.get("n_episodes", 0))) == SMOKE_EVAL_EPISODES
    assert "mean_reward" in metrics or "mean_steps_to_success" in metrics


def _download_s3_prefix(settings: PipelineSettings, output_path: str, local_dir: Path) -> None:
    parsed = urlparse(output_path)
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    assert keys, f"No artifacts found under {output_path}"
    for key in keys:
        rel = key.removeprefix(prefix).lstrip("/")
        if not rel:
            continue
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(parsed.netloc, key, str(target))


def _submitted_subnet_id(raw: dict[str, object]) -> str:
    spec = raw.get("spec") if isinstance(raw, dict) else None
    if isinstance(spec, dict):
        return str(spec.get("subnet_id") or spec.get("subnetId") or "").strip()
    return ""


def _redact_job_raw(raw: dict[str, object]) -> dict[str, object]:
    def redact(value: object) -> object:
        if isinstance(value, dict):
            name = str(value.get("name", ""))
            if name in SECRET_ENV_NAMES and "value" in value:
                return {**value, "value": "<redacted>"}
            return {
                key: "<redacted>" if _looks_secret_key(str(key)) else redact(inner)
                for key, inner in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    redacted = redact(raw)
    assert isinstance(redacted, dict)
    return redacted


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ("secret", "token", "password"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
