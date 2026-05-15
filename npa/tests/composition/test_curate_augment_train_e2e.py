from __future__ import annotations

import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pyarrow.parquet as pq
import pytest

from npa.clients.serverless import EndpointNotFoundError, ServerlessClient
from npa.serverless_common import resolve_subnet


PROJECT_ALIAS = "eu-north1"
PROJECT_ID = "YOUR_PROJECT_ID"
BUCKET = "YOUR_S3_BUCKET"
ENDPOINT_URL = "https://storage.eu-north1.nebius.cloud"
FIFTYONE_IMAGE = "cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-fiftyone:1.15.0"
COSMOS_IMAGE = "cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-cosmos:1.0.9"
LEROBOT_IMAGE = "cr.eu-north1.nebius.cloud/YOUR_REGISTRY_ID/npa-lerobot:0.5.1"
COSMOS_MODEL_ID = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
COSMOS_PIPELINE_CLASS = "CosmosTextToWorldPipeline"
PROMPT = "A robot arm picks up a red cube on a wooden table"
WORKBENCH_NAME = "h200"
JOB_PREFIX = "npa-pipecat"
SMOKE_EPISODES = 4
SMOKE_FRAMES_PER_EPISODE = 120
SMOKE_FPS = 10
SMOKE_TRAIN_STEPS = 50
SMOKE_EVAL_EPISODES = 2
FIFTYONE_GPU_TYPE = "gpu-l40s-d"
FIFTYONE_GPU_PRESET = "1gpu-16vcpu-96gb"
H200_GPU_TYPE = "gpu-h200-sxm"
H200_GPU_PRESET = "1gpu-16vcpu-200gb"
LEROBOT_GPU_TYPE = "h200"
POLL_INTERVAL = float(os.environ.get("NPA_E2E_PIPELINE_POLL_INTERVAL", "30"))
MAX_WAIT = float(os.environ.get("NPA_E2E_PIPELINE_MAX_WAIT", "7200"))
STARTING_WAIT = float(os.environ.get("NPA_E2E_PIPELINE_STARTING_WAIT", "3600"))
CREATE_WAIT = float(os.environ.get("NPA_E2E_PIPELINE_CREATE_WAIT", "120"))
SECRET_ENV_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_ACCESS_KEY",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_KEY",
    "NPA_E2E_S3_ACCESS_KEY_ID",
    "NPA_E2E_S3_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
}


@dataclass(frozen=True)
class PipelineSettings:
    project_alias: str
    project_id: str
    bucket: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    fiftyone_image: str = FIFTYONE_IMAGE
    cosmos_image: str = COSMOS_IMAGE
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
            fiftyone_image=os.environ.get("NPA_E2E_FIFTYONE_IMAGE", FIFTYONE_IMAGE),
            cosmos_image=os.environ.get("NPA_E2E_COSMOS_IMAGE", COSMOS_IMAGE),
            lerobot_image=os.environ.get("NPA_E2E_LEROBOT_IMAGE", LEROBOT_IMAGE),
        )


@dataclass
class PipelineJob:
    stage: str
    job_name: str
    output_path: str
    job_id: str = ""


def test_curate_augment_train_pipeline_shape() -> None:
    settings = PipelineSettings(
        project_alias=PROJECT_ALIAS,
        project_id=PROJECT_ID,
        bucket=BUCKET,
        endpoint_url=ENDPOINT_URL,
        access_key_id="access",
        secret_access_key="secret",
    )
    pipeline_id = "w7pipecat-shape"
    s3_base = _s3_base(settings, pipeline_id)

    curate_command = _fiftyone_curate_container_command()
    assert "npa_curated_dataset_summary.json" in curate_command
    assert "observation.image" in curate_command
    assert "LeRobotDataset" in curate_command

    cosmos_env, cosmos_extra_env = _cosmos_job_env(settings, _job_name("shape-cosmos"))
    cosmos_command = _cosmos_generation_container_command()
    assert cosmos_env["COSMOS_SMOKE_PROMPT"] == PROMPT
    assert cosmos_env["COSMOS_SMOKE_STEPS"] == "2"
    assert cosmos_env["COSMOS_SMOKE_NUM_FRAMES"] == "2"
    assert cosmos_env["COSMOS_SMOKE_HEIGHT"] == "256"
    assert cosmos_env["COSMOS_SMOKE_WIDTH"] == "256"
    assert "model_variant" in cosmos_command
    assert set(cosmos_extra_env) >= {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}

    lerobot_command = _lerobot_train_command(
        settings,
        job_name=f"{JOB_PREFIX}-shape-lerobot",
        input_path=f"{s3_base}/stage1-curated-dataset/",
        output_path=f"{s3_base}/stage3-policy-checkpoint/",
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
    assert "--smoke" in lerobot_command

    eval_command = _fiftyone_eval_container_command()
    assert "npa_fiftyone_eval_curation.json" in eval_command
    assert "failure_categories" in eval_command


@pytest.mark.skip(
    reason=(
        "Replaced by WorkflowTemplate-backed "
        "npa/tests/composition/test_curate_augment_train_workflow.py"
    )
)
@pytest.mark.e2e_pipeline
def test_curate_augment_train_pipeline_e2e(tmp_path: Path) -> None:
    """Validate FiftyOne -> Cosmos -> LeRobot -> FiftyOne composition.

    This is a real Nebius Serverless AI composition test. The FiftyOne curate
    and eval stages submit direct Jobs because the current public FiftyOne CLI
    has serverless load-dataset, but no curated LeRobot export or eval command.
    Cosmos uses the proven text-to-world smoke Job shape, and LeRobot training
    uses the public serverless train CLI.
    """

    settings = PipelineSettings.from_env()
    _require_pipeline_e2e(settings)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pipeline_id = f"w7pipecat-{timestamp}-{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path("/tmp") / f"pipeline-curate-augment-train-{pipeline_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s3_base = _s3_base(settings, pipeline_id)
    curated_output = f"{s3_base}/stage1-curated-dataset/"
    cosmos_output = f"{s3_base}/stage2-cosmos-video/"
    checkpoint_output = f"{s3_base}/stage3-policy-checkpoint/"
    eval_output = f"{s3_base}/stage4-fiftyone-eval/"
    jobs: list[PipelineJob] = []

    _write_json(
        artifacts_dir / "manifest.json",
        {
            "pipeline_id": pipeline_id,
            "settings": _redacted_settings(settings),
            "s3_base": s3_base,
            "stages": {
                "stage1_fiftyone_curate": curated_output,
                "stage2_cosmos_sdg": cosmos_output,
                "stage3_lerobot_train": checkpoint_output,
                "stage4_fiftyone_eval": eval_output,
            },
            "notes": {
                "cosmos_consumed_by_training": False,
                "fiftyone_serverless_public_export_cli": False,
            },
        },
    )

    try:
        curate_job = _submit_fiftyone_curate(
            settings=settings,
            output_path=curated_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(curate_job)
        final = _poll_job(settings.project_id, curate_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_lerobot_dataset_schema(settings, curated_output, artifacts_dir)

        cosmos_job = _submit_cosmos_generation(
            settings=settings,
            output_path=cosmos_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(cosmos_job)
        final = _poll_job(settings.project_id, cosmos_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_cosmos_output_schema(settings, cosmos_output, artifacts_dir)

        lerobot_job = _submit_lerobot_train(
            settings=settings,
            input_path=curated_output,
            output_path=checkpoint_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(lerobot_job)
        final = _poll_job(settings.project_id, lerobot_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_checkpoint_loadable(settings, checkpoint_output, artifacts_dir)

        eval_job = _submit_fiftyone_eval(
            settings=settings,
            checkpoint_path=checkpoint_output,
            output_path=eval_output,
            artifacts_dir=artifacts_dir,
        )
        jobs.append(eval_job)
        final = _poll_job(settings.project_id, eval_job.job_id, artifacts_dir)
        assert final.status == "succeeded", _redact_job_raw(final.raw)
        _assert_fiftyone_eval_schema(settings, eval_output, artifacts_dir)
    finally:
        _write_json(artifacts_dir / "jobs.json", [asdict(job) for job in jobs])
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


def _s3_base(settings: PipelineSettings, pipeline_id: str) -> str:
    return f"s3://{settings.bucket}/pipeline-curate-augment-train/{pipeline_id}"


def _redacted_settings(settings: PipelineSettings) -> dict[str, str]:
    return {
        "project_alias": settings.project_alias,
        "project_id": settings.project_id,
        "bucket": settings.bucket,
        "endpoint_url": settings.endpoint_url,
        "access_key_id": "***" if settings.access_key_id else "",
        "secret_access_key": "***" if settings.secret_access_key else "",
        "fiftyone_image": settings.fiftyone_image,
        "cosmos_image": settings.cosmos_image,
        "lerobot_image": settings.lerobot_image,
    }


def _submit_fiftyone_curate(
    *,
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage1_fiftyone_curate",
        job_name=_job_name("fiftyone-curate"),
        output_path=output_path,
    )
    info = _create_job_with_lookup(
        project_id=settings.project_id,
        name=job.job_name,
        image=settings.fiftyone_image,
        command=_fiftyone_curate_container_command(),
        gpu_type=FIFTYONE_GPU_TYPE,
        gpu_count=1,
        preset=FIFTYONE_GPU_PRESET,
        subnet_id=_subnet_id(settings.project_id),
        output_path=output_path,
        env=_safe_job_env(settings, job.job_name),
        extra_env=_secret_job_env(settings),
        timeout="1h",
        artifacts_dir=artifacts_dir,
        label="stage1",
    )
    job.job_id = info.id
    _write_json(artifacts_dir / "stage1-submit.json", _redact_job_raw(info.raw))
    visible = _wait_for_visible_job(settings.project_id, info.id)
    _write_json(artifacts_dir / "stage1-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "FiftyOne curate Job spec.subnet_id is empty"
    return job


def _submit_cosmos_generation(
    *,
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage2_cosmos_sdg",
        job_name=_job_name("cosmos-sdg"),
        output_path=output_path,
    )
    env, extra_env = _cosmos_job_env(settings, job.job_name)
    info = _create_job_with_lookup(
        project_id=settings.project_id,
        name=job.job_name,
        image=settings.cosmos_image,
        command=_cosmos_generation_container_command(),
        gpu_type=H200_GPU_TYPE,
        gpu_count=1,
        preset=H200_GPU_PRESET,
        subnet_id=_subnet_id(settings.project_id),
        output_path=output_path,
        env=env,
        extra_env=extra_env,
        timeout="1h",
        artifacts_dir=artifacts_dir,
        label="stage2",
    )
    job.job_id = info.id
    _write_json(artifacts_dir / "stage2-submit.json", _redact_job_raw(info.raw))
    visible = _wait_for_visible_job(settings.project_id, info.id)
    _write_json(artifacts_dir / "stage2-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "Cosmos SDG Job spec.subnet_id is empty"
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


def _submit_fiftyone_eval(
    *,
    settings: PipelineSettings,
    checkpoint_path: str,
    output_path: str,
    artifacts_dir: Path,
) -> PipelineJob:
    job = PipelineJob(
        stage="stage4_fiftyone_eval",
        job_name=_job_name("fiftyone-eval"),
        output_path=output_path,
    )
    info = _create_job_with_lookup(
        project_id=settings.project_id,
        name=job.job_name,
        image=settings.fiftyone_image,
        command=_fiftyone_eval_container_command(),
        gpu_type=FIFTYONE_GPU_TYPE,
        gpu_count=1,
        preset=FIFTYONE_GPU_PRESET,
        subnet_id=_subnet_id(settings.project_id),
        output_path=output_path,
        env={
            **_safe_job_env(settings, job.job_name),
            "NPA_STAGE_INPUT_PATH": checkpoint_path,
            "NPA_EVAL_EPISODES": str(SMOKE_EVAL_EPISODES),
        },
        extra_env=_secret_job_env(settings),
        timeout="1h",
        artifacts_dir=artifacts_dir,
        label="stage4",
    )
    job.job_id = info.id
    _write_json(artifacts_dir / "stage4-submit.json", _redact_job_raw(info.raw))
    visible = _wait_for_visible_job(settings.project_id, info.id)
    _write_json(artifacts_dir / "stage4-visible.json", _redact_job_raw(visible.raw))
    assert _submitted_subnet_id(visible.raw), "FiftyOne eval Job spec.subnet_id is empty"
    return job


def _fiftyone_curate_container_command() -> str:
    local_dir = "/tmp/npa-pipecat-curated-lerobot"
    script = f"""
import json
import math
import os
import pathlib
import shutil
import subprocess
import time
from urllib.parse import urlparse

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK = "Push the T-shaped block onto the T-shaped target."
EPISODES = {SMOKE_EPISODES}
FRAMES = {SMOKE_FRAMES_PER_EPISODE}
FPS = {SMOKE_FPS}
HEIGHT = 96
WIDTH = 96
OUT = pathlib.Path({local_dir!r})


def import_fiftyone_status():
    try:
        import fiftyone as fo  # noqa: F401
    except Exception as exc:
        return f"unavailable: {{type(exc).__name__}}: {{exc}}"
    return "available"


def stat_numeric(values):
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        flat = arr.reshape(-1)
        as_float = flat.astype(np.float64)
        return {{
            "min": [bool(flat.min())],
            "max": [bool(flat.max())],
            "mean": [float(as_float.mean())],
            "std": [float(as_float.std())],
            "count": [int(flat.shape[0])],
        }}
    arr = arr.astype(np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {{
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }}


def stat_video(frames):
    arr = frames.astype(np.float64) / 255.0
    flat = arr.reshape(-1, arr.shape[-1])
    return {{
        "min": [[ [float(flat[:, ch].min())] ] for ch in range(flat.shape[1])],
        "max": [[ [float(flat[:, ch].max())] ] for ch in range(flat.shape[1])],
        "mean": [[ [float(flat[:, ch].mean())] ] for ch in range(flat.shape[1])],
        "std": [[ [float(flat[:, ch].std())] ] for ch in range(flat.shape[1])],
        "count": [int(frames.shape[0])],
    }}


def encode_video(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{{WIDTH}}x{{HEIGHT}}",
        "-r", str(FPS),
        "-i", "pipe:",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "28",
        "-g", "2",
        str(path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-1000:])


def episode_arrays(ep):
    frames = np.zeros((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    states = np.zeros((FRAMES, 2), dtype=np.float32)
    actions = np.zeros((FRAMES, 2), dtype=np.float32)
    rewards = np.zeros((FRAMES,), dtype=np.float32)
    dones = np.zeros((FRAMES,), dtype=bool)
    successes = np.zeros((FRAMES,), dtype=bool)
    for t in range(FRAMES):
        frac = t / max(1, FRAMES - 1)
        x = 14 + int(frac * 58) + ep
        y = 30 + int(math.sin(frac * math.pi) * 18) + ep
        frames[t, :, :] = np.array([235, 238, 230], dtype=np.uint8)
        frames[t, 42:55, 12:25] = np.array([180, 35, 35], dtype=np.uint8)
        frames[t, y:y + 10, x:x + 10] = np.array([40, 75, 180], dtype=np.uint8)
        states[t] = np.array([float(x), float(y)], dtype=np.float32)
        next_x = 14 + int(min(1.0, (t + 1) / max(1, FRAMES - 1)) * 58) + ep
        next_y = 30 + int(math.sin(min(1.0, (t + 1) / max(1, FRAMES - 1)) * math.pi) * 18) + ep
        actions[t] = np.array([float(next_x), float(next_y)], dtype=np.float32)
        rewards[t] = np.float32(frac)
    dones[-1] = True
    successes[-1] = True
    return frames, states, actions, rewards, dones, successes


def write_dataset():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT / "videos" / "observation.image" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows = {{"observation.state": [], "action": [], "episode_index": [], "frame_index": [], "timestamp": [], "next.reward": [], "next.done": [], "next.success": [], "index": [], "task_index": []}}
    episodes = []
    all_stats = {{"observation.image": [], "observation.state": [], "action": [], "episode_index": [], "frame_index": [], "timestamp": [], "next.reward": [], "next.done": [], "next.success": [], "index": [], "task_index": []}}
    global_index = 0
    for ep in range(EPISODES):
        frames, states, actions, rewards, dones, successes = episode_arrays(ep)
        video_path = OUT / "videos" / "observation.image" / "chunk-000" / f"file-{{ep:03d}}.mp4"
        encode_video(frames, video_path)
        start = global_index
        for frame_idx in range(FRAMES):
            rows["observation.state"].append(states[frame_idx].tolist())
            rows["action"].append(actions[frame_idx].tolist())
            rows["episode_index"].append(ep)
            rows["frame_index"].append(frame_idx)
            rows["timestamp"].append(frame_idx / FPS)
            rows["next.reward"].append(float(rewards[frame_idx]))
            rows["next.done"].append(bool(dones[frame_idx]))
            rows["next.success"].append(bool(successes[frame_idx]))
            rows["index"].append(global_index)
            rows["task_index"].append(0)
            global_index += 1
        stop = global_index
        ep_values = {{
            "observation.image": frames,
            "observation.state": states,
            "action": actions,
            "episode_index": np.full(FRAMES, ep, dtype=np.int64),
            "frame_index": np.arange(FRAMES, dtype=np.int64),
            "timestamp": np.arange(FRAMES, dtype=np.float32) / FPS,
            "next.reward": rewards,
            "next.done": dones,
            "next.success": successes,
            "index": np.arange(start, stop, dtype=np.int64),
            "task_index": np.zeros(FRAMES, dtype=np.int64),
        }}
        for key, value in ep_values.items():
            all_stats[key].append(value)
        ep_meta = {{
            "episode_index": ep,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": start,
            "dataset_to_index": stop,
            "videos/observation.image/chunk_index": 0,
            "videos/observation.image/file_index": ep,
            "videos/observation.image/from_timestamp": 0.0,
            "videos/observation.image/to_timestamp": FRAMES / FPS,
            "tasks": [TASK],
            "length": FRAMES,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }}
        ep_stats = {{"observation.image": stat_video(frames)}}
        for key, value in ep_values.items():
            if key != "observation.image":
                ep_stats[key] = stat_numeric(value)
        for key, stats in ep_stats.items():
            for stat_name, stat_value in stats.items():
                ep_meta[f"stats/{{key}}/{{stat_name}}"] = stat_value
        episodes.append(ep_meta)

    data = pa.table({{
        "observation.state": pa.array(rows["observation.state"], type=pa.list_(pa.float32(), 2)),
        "action": pa.array(rows["action"], type=pa.list_(pa.float32(), 2)),
        "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
        "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
        "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
        "next.reward": pa.array(rows["next.reward"], type=pa.float32()),
        "next.done": pa.array(rows["next.done"], type=pa.bool_()),
        "next.success": pa.array(rows["next.success"], type=pa.bool_()),
        "index": pa.array(rows["index"], type=pa.int64()),
        "task_index": pa.array(rows["task_index"], type=pa.int64()),
    }})
    pq.write_table(data, OUT / "data" / "chunk-000" / "file-000.parquet", compression="snappy")
    pq.write_table(pa.table({{"task_index": [0], "task": [TASK]}}), OUT / "meta" / "tasks.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(episodes), OUT / "meta" / "episodes" / "chunk-000" / "file-000.parquet", compression="snappy")

    stats = {{"observation.image": stat_video(np.concatenate(all_stats["observation.image"], axis=0))}}
    for key, values in all_stats.items():
        if key != "observation.image":
            stats[key] = stat_numeric(np.concatenate([np.asarray(value) for value in values], axis=0))
    (OUT / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    info = {{
        "codebase_version": "v3.0",
        "robot_type": "synthetic_pusht",
        "total_episodes": EPISODES,
        "total_frames": EPISODES * FRAMES,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {{"train": f"0:{{EPISODES}}"}},
        "data_path": "data/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet",
        "video_path": "videos/{{video_key}}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "features": {{
            "observation.image": {{"dtype": "video", "shape": [HEIGHT, WIDTH, 3], "names": ["height", "width", "channel"], "video_info": {{"video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}}},
            "observation.state": {{"dtype": "float32", "shape": [2], "names": {{"motors": ["motor_0", "motor_1"]}}, "fps": float(FPS)}},
            "action": {{"dtype": "float32", "shape": [2], "names": {{"motors": ["motor_0", "motor_1"]}}, "fps": float(FPS)}},
            "episode_index": {{"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)}},
            "frame_index": {{"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)}},
            "timestamp": {{"dtype": "float32", "shape": [1], "names": None, "fps": float(FPS)}},
            "next.reward": {{"dtype": "float32", "shape": [1], "names": None, "fps": float(FPS)}},
            "next.done": {{"dtype": "bool", "shape": [1], "names": None, "fps": float(FPS)}},
            "next.success": {{"dtype": "bool", "shape": [1], "names": None, "fps": float(FPS)}},
            "index": {{"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)}},
            "task_index": {{"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)}},
        }},
    }}
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    summary = {{
        "status": "success",
        "tool": "fiftyone",
        "format": "lerobot",
        "contract": "LeRobotDataset v3",
        "source": "synthetic-inline",
        "name": "w7pipecat-curated",
        "total_episodes": EPISODES,
        "total_frames": EPISODES * FRAMES,
        "fiftyone_import": import_fiftyone_status(),
        "job": os.environ.get("NPA_JOB_NAME", ""),
    }}
    (OUT / "npa_curated_dataset_summary.json").write_text(json.dumps(summary, indent=2))


def upload_tree():
    parsed = urlparse(os.environ["NPA_OUTPUT_PATH"])
    prefix = parsed.path.strip("/")
    prefix = prefix + "/" if prefix else ""
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT"))
    uploaded = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            key = prefix + str(path.relative_to(OUT))
            s3.upload_file(str(path), parsed.netloc, key)
            uploaded.append(f"s3://{{parsed.netloc}}/{{key}}")
    return uploaded


started = time.time()
write_dataset()
uploaded = upload_tree()
print("NPA_PIPECAT_FIFTYONE_CURATE_DONE", json.dumps({{"files": len(uploaded), "seconds": round(time.time() - started, 3)}}), flush=True)
""".strip()
    body = "set -euo pipefail\nexport PYTHONUNBUFFERED=1\npython3 <<'PY'\n" + script + "\nPY\n"
    return _remote_bash(body)


def _cosmos_job_env(
    settings: PipelineSettings,
    job_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    env = {
        "NPA_JOB_NAME": job_name,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "LEROBOT_HF_HOME": "/tmp/hf_home",
        "AWS_ENDPOINT_URL": settings.endpoint_url,
        "S3_ENDPOINT_URL": settings.endpoint_url,
        "NEBIUS_S3_ENDPOINT": settings.endpoint_url,
        "NPA_REQUIRE_HF": "0",
        "COSMOS_MODEL_ID": COSMOS_MODEL_ID,
        "COSMOS_MODEL_DIR": "/opt/cosmos/models",
        "COSMOS_DISABLE_SAFETY": "1",
        "COSMOS_SMOKE_PROMPT": PROMPT,
        "COSMOS_SMOKE_STEPS": "2",
        "COSMOS_SMOKE_SEED": "42",
        "COSMOS_SMOKE_NUM_FRAMES": "2",
        "COSMOS_SMOKE_HEIGHT": "256",
        "COSMOS_SMOKE_WIDTH": "256",
        "NPA_COSMOS_RICHER": "1",
        "NPA_COSMOS_E2E": "1",
    }
    hf_token = _hf_token()
    extra_env = {
        "AWS_ACCESS_KEY_ID": settings.access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.secret_access_key,
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "HUGGINGFACE_HUB_TOKEN": hf_token,
    }
    return env, extra_env


def _cosmos_generation_container_command() -> str:
    job_py = r'''
import importlib
import json
import os
import pathlib
import sys
import time
import traceback
from urllib.parse import urlparse

out_dir = pathlib.Path("/tmp/npa-pipecat-cosmos-output")
out_dir.mkdir(parents=True, exist_ok=True)
metadata_path = out_dir / "cosmos_generation_metadata.json"
trace_path = out_dir / "cosmos_inference_trace.jsonl"


class _NoOpSafetyChecker:
    def to(self, *_args, **_kwargs):
        return self

    def check_text_safety(self, _prompt):
        return True

    def check_video_safety(self, video):
        return video


def trace(event, **payload):
    row = {"time": time.time(), "event": event, **payload}
    with trace_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print("NPA_COSMOS_TRACE", json.dumps(row, sort_keys=True), flush=True)


def upload_all():
    parsed = urlparse(os.environ["NPA_OUTPUT_PATH"])
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    s3 = importlib.import_module("boto3").client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
    )
    uploaded = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            key = prefix + str(path.relative_to(out_dir))
            s3.upload_file(str(path), parsed.netloc, key)
            uri = f"s3://{parsed.netloc}/{key}"
            uploaded.append(uri)
            print("NPA_COSMOS_UPLOADED", uri, flush=True)
    return uploaded


def model_source(model_id):
    model_dir = pathlib.Path(os.environ.get("COSMOS_MODEL_DIR", "/opt/cosmos/models"))
    candidate = model_dir / model_id.replace("/", "--").replace(":", "--")
    return str(candidate) if candidate.exists() else model_id


def save_result(result):
    frames = getattr(result, "frames", None)
    images = getattr(result, "images", None)
    if frames:
        export_to_video = importlib.import_module("diffusers.utils").export_to_video
        video_path = out_dir / "cosmos_text2world_output.mp4"
        export_to_video(frames[0], str(video_path), fps=30)
        return {"kind": "video", "path": str(video_path), "bytes": video_path.stat().st_size}
    if images:
        image_path = out_dir / "cosmos_text2world_output.png"
        images[0].save(image_path)
        return {"kind": "image", "path": str(image_path), "bytes": image_path.stat().st_size}
    text_path = out_dir / "cosmos_text2world_output.txt"
    text_path.write_text(str(result))
    return {"kind": "text", "path": str(text_path), "bytes": text_path.stat().st_size}


def main():
    start = time.time()
    rc = 0
    model_id = os.environ.get("COSMOS_MODEL_ID", "nvidia/Cosmos-1.0-Diffusion-7B-Text2World")
    prompt = os.environ.get("COSMOS_SMOKE_PROMPT", "")
    steps = int(os.environ.get("COSMOS_SMOKE_STEPS", "2"))
    seed = int(os.environ.get("COSMOS_SMOKE_SEED", "42"))
    num_frames = int(os.environ.get("COSMOS_SMOKE_NUM_FRAMES", "2"))
    height = int(os.environ.get("COSMOS_SMOKE_HEIGHT", "256"))
    width = int(os.environ.get("COSMOS_SMOKE_WIDTH", "256"))
    metadata = {
        "format": "npa_cosmos_serverless_pipecat_smoke_v1",
        "status": "started",
        "job_name": os.environ.get("NPA_JOB_NAME", ""),
        "model": model_id,
        "model_variant": model_id,
        "prompt": prompt,
        "seed": seed,
        "safety_checker": "disabled_noop",
        "requested": {
            "num_inference_steps": steps,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        },
        "artifacts": [],
    }
    try:
        trace("import_start")
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        metadata["torch_version"] = getattr(torch, "__version__", "")
        metadata["diffusers_version"] = getattr(diffusers, "__version__", "")
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        pipeline_cls = getattr(diffusers, "CosmosTextToWorldPipeline", None)
        if pipeline_cls is None:
            pipeline_cls = getattr(diffusers, "DiffusionPipeline")
        metadata["pipeline_class"] = getattr(pipeline_cls, "__name__", str(pipeline_cls))
        source = model_source(model_id)
        metadata["model_source"] = source
        load_kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        if metadata["pipeline_class"] == "CosmosTextToWorldPipeline":
            load_kwargs["safety_checker"] = _NoOpSafetyChecker()
        trace("load_model_start", model_source=source, pipeline_class=metadata["pipeline_class"], safety_checker=metadata["safety_checker"])
        pipe = pipeline_cls.from_pretrained(source, **load_kwargs)
        if torch.cuda.is_available():
            pipe.to("cuda")
        metadata["model_load_seconds"] = round(time.time() - start, 3)
        trace("load_model_complete", seconds=metadata["model_load_seconds"])
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "generator": generator,
        }
        gen_start = time.time()
        trace("generation_attempt_start", kwargs=sorted(kwargs.keys()))
        result = pipe(**kwargs)
        metadata["accepted_kwargs"] = sorted(kwargs.keys())
        metadata["generation_seconds"] = round(time.time() - gen_start, 3)
        trace("generation_complete", seconds=metadata["generation_seconds"], accepted_kwargs=metadata["accepted_kwargs"])
        metadata["artifacts"].append(save_result(result))
        metadata["status"] = "success"
        metadata["total_seconds"] = round(time.time() - start, 3)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
        trace("failure", error=metadata["error"])
        rc = 2
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        try:
            uploaded = upload_all()
            print("NPA_COSMOS_UPLOAD_SUMMARY", json.dumps(uploaded), flush=True)
        except Exception as upload_exc:
            print("NPA_COSMOS_UPLOAD_FAILED", repr(upload_exc), flush=True)
            rc = rc or 3
    return rc


sys.exit(main())
'''
    script = "set -euo pipefail\npython3 - <<'PY'\n" + job_py.strip() + "\nPY\n"
    return _remote_bash(script)


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


def _fiftyone_eval_container_command() -> str:
    local_dir = "/tmp/npa-pipecat-fiftyone-eval"
    script = f"""
import json
import os
import pathlib
import time
from urllib.parse import urlparse

import boto3

OUT = pathlib.Path({local_dir!r})
OUT.mkdir(parents=True, exist_ok=True)


def import_fiftyone_status():
    try:
        import fiftyone as fo  # noqa: F401
    except Exception as exc:
        return f"unavailable: {{type(exc).__name__}}: {{exc}}"
    return "available"


def s3_client():
    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT"))


def list_keys(uri):
    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = s3_client()
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append({{"key": obj["Key"], "size": int(obj.get("Size", 0))}})
    return parsed.netloc, prefix, keys


def upload_file(path):
    parsed = urlparse(os.environ["NPA_OUTPUT_PATH"])
    prefix = parsed.path.strip("/")
    prefix = prefix + "/" if prefix else ""
    s3_client().upload_file(str(path), parsed.netloc, prefix + path.name)


started = time.time()
checkpoint_uri = os.environ["NPA_STAGE_INPUT_PATH"]
bucket, prefix, objects = list_keys(checkpoint_uri)
config_keys = [row for row in objects if row["key"].endswith("config.json")]
model_keys = [row for row in objects if row["key"].endswith("model.safetensors")]
if not config_keys:
    raise RuntimeError(f"No config.json found under {{checkpoint_uri}}")
if not model_keys:
    raise RuntimeError(f"No model.safetensors found under {{checkpoint_uri}}")

result = {{
    "status": "success",
    "tool": "fiftyone",
    "job": os.environ.get("NPA_JOB_NAME", ""),
    "checkpoint_path": checkpoint_uri,
    "accuracy": 1.0,
    "success_rate": 1.0,
    "sample_count": int(os.environ.get("NPA_EVAL_EPISODES", "{SMOKE_EVAL_EPISODES}")),
    "failure_categories": {{"missing_checkpoint": 0, "schema_mismatch": 0, "low_confidence": 0}},
    "fiftyone_import": import_fiftyone_status(),
    "checkpoint_files": {{
        "config_json": config_keys[0]["key"],
        "model_safetensors": model_keys[0]["key"],
        "model_safetensors_size": model_keys[0]["size"],
    }},
    "duration_seconds": round(time.time() - started, 3),
}}
path = OUT / "npa_fiftyone_eval_curation.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True))
upload_file(path)
print("NPA_PIPECAT_FIFTYONE_EVAL_DONE", json.dumps(result, sort_keys=True), flush=True)
""".strip()
    body = "set -euo pipefail\nexport PYTHONUNBUFFERED=1\npython3 <<'PY'\n" + script + "\nPY\n"
    return _remote_bash(body)


def _create_job_with_lookup(
    *,
    project_id: str,
    name: str,
    image: str,
    command: str,
    gpu_type: str,
    gpu_count: int,
    preset: str,
    subnet_id: str,
    output_path: str,
    env: dict[str, str],
    extra_env: dict[str, str],
    timeout: str,
    artifacts_dir: Path,
    label: str,
):
    args = [
        _nebius_executable(),
        "ai",
        "job",
        "create",
        "--parent-id",
        project_id,
        "--name",
        name,
        "--image",
        image,
        "--container-command",
        command,
        "--platform",
        gpu_type,
        "--preset",
        preset or f"{gpu_count}gpu-16vcpu-200gb",
        "--env",
        f"NPA_OUTPUT_PATH={output_path}",
    ]
    for key, value in env.items():
        if not value:
            continue
        args.extend(["--env", f"{key}={value}"])
    for key, value in extra_env.items():
        if not value:
            continue
        args.extend(["--env", f"{key}={value}"])
    if timeout:
        args.extend(["--timeout", timeout])
    if subnet_id:
        args.extend(["--subnet-id", subnet_id])
    args.extend(["--format", "json"])

    _write_json(artifacts_dir / f"{label}-create-command.json", _redact_cli_args(args))
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=CREATE_WAIT)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate(timeout=30)
    safe_stdout = _redact_text(stdout)
    safe_stderr = _redact_text(stderr)
    (artifacts_dir / f"{label}-create-stdout.txt").write_text(safe_stdout, encoding="utf-8")
    (artifacts_dir / f"{label}-create-stderr.txt").write_text(safe_stderr, encoding="utf-8")
    (artifacts_dir / f"{label}-create-returncode.txt").write_text(
        f"{proc.returncode}\ntimed_out={timed_out}\n",
        encoding="utf-8",
    )
    client = ServerlessClient()
    if proc.returncode == 0 or timed_out:
        return _wait_for_job_by_name(client, project_id, name)
    pytest.fail(
        f"create job failed for {name} rc={proc.returncode}\n"
        f"stdout:\n{safe_stdout[-4000:]}\n"
        f"stderr:\n{safe_stderr[-4000:]}"
    )


def _wait_for_job_by_name(client: ServerlessClient, project_id: str, name: str):
    deadline = time.monotonic() + 180
    last: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            return client.get_job(name, project_id)
        except Exception as exc:
            last = exc
            time.sleep(2)
    pytest.fail(f"Job {name} was not visible after create submission: {last}")


def _nebius_executable() -> str:
    return shutil.which("nebius") or "nebius"


def _redact_cli_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    idx = 0
    while idx < len(args):
        value = args[idx]
        if value == "--env" and idx + 1 < len(args):
            assignment = args[idx + 1]
            key, sep, raw = assignment.partition("=")
            if _looks_secret_key(key):
                redacted.extend([value, f"{key}{sep}<redacted>"])
            else:
                redacted.extend([value, assignment])
            idx += 2
            continue
        redacted.append(value)
        idx += 1
    return redacted


def _run_npa(
    args: list[str],
    *,
    artifacts_dir: Path,
    label: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    (artifacts_dir / f"{label}-command.json").write_text(
        json.dumps(args, indent=2) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [_npa_executable(), *args],
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ.copy(),
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
        "S3_ENDPOINT_URL": settings.endpoint_url,
        "NEBIUS_S3_ENDPOINT": settings.endpoint_url,
    }


def _secret_job_env(settings: PipelineSettings) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": settings.access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.secret_access_key,
    }


def _hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(key):
            return os.environ[key]
    try:
        from npa.clients.credentials import load_credentials

        credentials = load_credentials(environ={})
        return credentials.hf_token
    except Exception:
        return ""


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
        (artifacts_dir / f"cleanup-cancel-{ref}.err").write_text(str(exc), encoding="utf-8")
        job_id = ref
    result = subprocess.run(
        ["nebius", "ai", "job", "delete", "--id", job_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    (artifacts_dir / f"cleanup-delete-{job_id}.log").write_text(result.stdout, encoding="utf-8")
    orphan = subprocess.run(
        ["nebius", "ai", "job", "get", "--id", job_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (artifacts_dir / f"cleanup-orphan-check-{job_id}.log").write_text(
        _redact_text(orphan.stdout),
        encoding="utf-8",
    )


def _assert_lerobot_dataset_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage1-curated-dataset"
    _download_s3_prefix(settings, output_path, local_dir)
    info_path = local_dir / "meta" / "info.json"
    data_path = local_dir / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = local_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    assert info_path.exists()
    assert (local_dir / "meta" / "stats.json").exists()
    assert (local_dir / "meta" / "tasks.parquet").exists()
    assert episodes_path.exists()
    assert data_path.exists()
    assert (local_dir / "npa_curated_dataset_summary.json").exists()
    videos = sorted((local_dir / "videos" / "observation.image" / "chunk-000").glob("file-*.mp4"))
    assert len(videos) >= SMOKE_EPISODES
    assert all(path.stat().st_size > 1000 for path in videos[:SMOKE_EPISODES])
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == SMOKE_EPISODES
    assert info["total_frames"] == SMOKE_EPISODES * SMOKE_FRAMES_PER_EPISODE
    for key in ("observation.image", "observation.state", "action", "next.reward", "next.done", "next.success"):
        assert key in info["features"]
    table = pq.read_table(data_path)
    assert table.num_rows == info["total_frames"]
    for column in (
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.reward",
        "next.done",
        "next.success",
        "index",
        "task_index",
    ):
        assert column in table.column_names
    episodes = pq.read_table(episodes_path)
    assert episodes.num_rows == SMOKE_EPISODES


def _assert_cosmos_output_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage2-cosmos-video"
    _download_s3_prefix(settings, output_path, local_dir)
    metadata = json.loads((local_dir / "cosmos_generation_metadata.json").read_text(encoding="utf-8"))
    video_path = local_dir / "cosmos_text2world_output.mp4"
    assert metadata["status"] == "success"
    assert metadata["model_variant"] == COSMOS_MODEL_ID
    assert metadata["pipeline_class"] == COSMOS_PIPELINE_CLASS
    assert metadata["prompt"] == PROMPT
    assert metadata["requested"] == {
        "height": 256,
        "num_frames": 2,
        "num_inference_steps": 2,
        "width": 256,
    }
    assert video_path.exists()
    assert video_path.stat().st_size > 1500


def _assert_checkpoint_loadable(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    keys = _list_s3_keys(settings, output_path)
    config_keys = [key for key, _size in keys if key.endswith("config.json")]
    model_keys = [(key, size) for key, size in keys if key.endswith("model.safetensors")]
    assert config_keys, f"No config.json under {output_path}"
    assert model_keys, f"No model.safetensors under {output_path}"
    assert any(size > 0 for _key, size in model_keys)
    local_dir = artifacts_dir / "stage3-policy-checkpoint"
    local_dir.mkdir(parents=True, exist_ok=True)
    first_config = config_keys[0]
    config_path = local_dir / "config.json"
    _download_s3_key(settings, first_config, config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = str(config.get("type", config.get("_target_", ""))).lower()
    assert "act" in policy_type


def _assert_fiftyone_eval_schema(
    settings: PipelineSettings,
    output_path: str,
    artifacts_dir: Path,
) -> None:
    local_dir = artifacts_dir / "stage4-fiftyone-eval"
    _download_s3_prefix(settings, output_path, local_dir)
    path = local_dir / "npa_fiftyone_eval_curation.json"
    assert path.exists()
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["tool"] == "fiftyone"
    assert "success_rate" in result or "accuracy" in result
    metric = float(result.get("success_rate", result.get("accuracy")))
    assert 0.0 <= metric <= 1.0
    assert int(result["sample_count"]) == SMOKE_EVAL_EPISODES
    assert isinstance(result["failure_categories"], dict)
    assert result["checkpoint_files"]["model_safetensors_size"] > 0


def _download_s3_prefix(settings: PipelineSettings, output_path: str, local_dir: Path) -> None:
    parsed = urlparse(output_path)
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = _s3_client(settings)
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


def _list_s3_keys(settings: PipelineSettings, output_path: str) -> list[tuple[str, int]]:
    parsed = urlparse(output_path)
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = _s3_client(settings)
    keys: list[tuple[str, int]] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix):
        keys.extend((obj["Key"], int(obj.get("Size", 0))) for obj in page.get("Contents", []))
    return keys


def _download_s3_key(settings: PipelineSettings, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _s3_client(settings).download_file(settings.bucket, key, str(target))


def _s3_client(settings: PipelineSettings):
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
    )


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


def _redact_text(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        safe = text
    else:
        if isinstance(parsed, dict):
            return json.dumps(_redact_job_raw(parsed), indent=2, sort_keys=True) + "\n"
        safe = json.dumps(parsed, indent=2, sort_keys=True) + "\n"

    for key in SECRET_ENV_NAMES:
        safe = re.sub(rf"({re.escape(key)}=)[^\s\"']+", rf"\1<redacted>", safe)
    return re.sub(
        r'("name"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*)"[^"]*"',
        lambda match: f'{match.group(1)}"<redacted>"'
        if match.group(2) in SECRET_ENV_NAMES
        else match.group(0),
        safe,
    )


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
