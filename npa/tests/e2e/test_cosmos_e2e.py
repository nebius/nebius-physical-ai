from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pytest

import npa.clients.serverless as serverless_mod
from npa.clients.serverless import EndpointNotFoundError, ServerlessClient

from ._serverless_images import resolve_image, resolve_serverless_gpu_type


PROJECT_ID = "project-test-00000000000"
BUCKET = "your-bucket-name"
ENDPOINT_URL = "https://storage.eu-north1.nebius.cloud"
IMAGE = "cr.eu-north1.nebius.cloud/your-registry-id/npa-cosmos:1.0.9"
MODEL_ID = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
PIPELINE_CLASS = "CosmosTextToWorldPipeline"
PROMPT = "A robot arm picks up a red cube on a wooden table"
JOB_PREFIX = "npa-e2e-cosmos-t2w"
POLL_INTERVAL = float(os.environ.get("NPA_E2E_COSMOS_POLL_INTERVAL", "30"))
MAX_WAIT = float(os.environ.get("NPA_E2E_COSMOS_MAX_WAIT", "1800"))


def test_cosmos_smoke_helper_request_shape() -> None:
    test_id = "shape"
    output_path = _output_path(test_id)
    env, extra_env = _job_env(test_id, output_path, access_key="ak", secret_key="sk")
    command = _cosmos_smoke_command()

    assert IMAGE.endswith("/npa-cosmos:1.0.9")
    assert env["COSMOS_MODEL_ID"] == MODEL_ID
    assert env["COSMOS_DISABLE_SAFETY"] == "1"
    assert env["COSMOS_SMOKE_PROMPT"] == PROMPT
    assert env["COSMOS_SMOKE_STEPS"] == "2"
    assert env["COSMOS_SMOKE_NUM_FRAMES"] == "2"
    assert env["COSMOS_SMOKE_HEIGHT"] == "256"
    assert env["COSMOS_SMOKE_WIDTH"] == "256"
    assert env["NPA_OUTPUT_PATH"] == output_path
    assert set(extra_env) == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    }
    assert "CosmosTextToWorldPipeline" in command
    assert "safety_checker" in command
    assert "generation_complete" in command
    assert "cosmos_text2world_output.mp4" in command


@pytest.mark.e2e_serverless
def test_cosmos_text2world_serverless_generation(tmp_path: Path) -> None:
    _require_cosmos_e2e()
    test_id = f"w7cosmos-e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    artifacts_dir = Path("/tmp") / f"cosmos-e2e-artifacts-{test_id}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    project_id = os.environ.get("NPA_E2E_SERVERLESS_PROJECT", PROJECT_ID)
    bucket = os.environ.get("NPA_E2E_S3_BUCKET", BUCKET)
    endpoint_url = os.environ.get("NPA_E2E_S3_ENDPOINT", ENDPOINT_URL)
    output_path = _output_path(test_id, bucket=bucket)
    access_key = os.environ["NPA_E2E_S3_ACCESS_KEY_ID"]
    secret_key = os.environ["NPA_E2E_S3_SECRET_ACCESS_KEY"]
    env, extra_env = _job_env(
        test_id,
        output_path,
        access_key=access_key,
        secret_key=secret_key,
        endpoint_url=endpoint_url,
    )
    name = f"{JOB_PREFIX}-{uuid.uuid4().hex[:8]}"
    client = ServerlessClient()
    job_id = ""
    (artifacts_dir / "output-path.txt").write_text(output_path + "\n", encoding="utf-8")
    (artifacts_dir / "job-name.txt").write_text(name + "\n", encoding="utf-8")

    try:
        serverless_mod._JOB_CREATE_TIMEOUT = int(os.environ.get("NPA_E2E_COSMOS_CREATE_TIMEOUT", "120"))
        info = client.create_job(
            project_id=project_id,
            name=name,
            image=resolve_image(os.environ.get("NPA_E2E_COSMOS_IMAGE", IMAGE)),
            command=_cosmos_smoke_command(),
            gpu_type=resolve_serverless_gpu_type(
                os.environ.get("NPA_E2E_COSMOS_GPU_TYPE", "gpu-h200-sxm")
            ),
            gpu_count=1,
            preset="1gpu-16vcpu-200gb",
            subnet_id=_subnet_id(project_id),
            output_path=output_path,
            env=env,
            extra_env=extra_env,
            timeout="1h",
        )
        job_id = info.id or name
        (artifacts_dir / "job-id.txt").write_text(job_id + "\n", encoding="utf-8")

        final = _poll_job(client, project_id, job_id, artifacts_dir)
        assert final.status == "succeeded", final.raw
        _capture_job(project_id, job_id, artifacts_dir, label="final")

        local_dir = artifacts_dir / "s3"
        _download_s3_prefix(output_path, local_dir, access_key, secret_key, endpoint_url)
        metadata = json.loads((local_dir / "cosmos_generation_metadata.json").read_text(encoding="utf-8"))
        trace_events = [
            json.loads(line)["event"]
            for line in (local_dir / "cosmos_inference_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        video_path = local_dir / "cosmos_text2world_output.mp4"

        assert metadata["status"] == "success"
        assert metadata["format"] == "npa_cosmos_serverless_richer_smoke_v2"
        assert metadata["model"] == MODEL_ID
        assert metadata["pipeline_class"] == PIPELINE_CLASS
        assert metadata["prompt"] == PROMPT
        assert metadata["safety_checker"] == "disabled_noop"
        assert metadata["cuda_device_name"] == "NVIDIA H200"
        assert metadata["accepted_kwargs"] == [
            "generator",
            "height",
            "num_frames",
            "num_inference_steps",
            "prompt",
            "width",
        ]
        assert metadata["requested"] == {
            "height": 256,
            "num_frames": 2,
            "num_inference_steps": 2,
            "width": 256,
        }
        assert trace_events[-1] == "generation_complete"
        assert "load_model_complete" in trace_events
        assert video_path.exists()
        assert video_path.stat().st_size > 1500
        assert _ffprobe_video(video_path, artifacts_dir) == {
            "codec_name": "h264",
            "codec_type": "video",
            "height": 256,
            "width": 256,
        }
    finally:
        if job_id or name:
            _cleanup_job(project_id, job_id or name, artifacts_dir)


def _require_cosmos_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if not os.environ.get("NPA_E2E_SERVERLESS_PROJECT"):
        pytest.skip("NPA_E2E_SERVERLESS_PROJECT not set")
    if not os.environ.get("NPA_E2E_S3_ACCESS_KEY_ID"):
        pytest.skip("NPA_E2E_S3_ACCESS_KEY_ID not set")
    if not os.environ.get("NPA_E2E_S3_SECRET_ACCESS_KEY"):
        pytest.skip("NPA_E2E_S3_SECRET_ACCESS_KEY not set")


def _output_path(test_id: str, *, bucket: str = BUCKET) -> str:
    return f"s3://{bucket}/w7cosmos-e2e/{test_id}/"


def _job_env(
    test_id: str,
    output_path: str,
    *,
    access_key: str,
    secret_key: str,
    endpoint_url: str = ENDPOINT_URL,
) -> tuple[dict[str, str], dict[str, str]]:
    env = {
        "NPA_OUTPUT_PATH": output_path,
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/tmp/hf_home",
        "LEROBOT_HF_HOME": "/tmp/hf_home",
        "AWS_ENDPOINT_URL": endpoint_url,
        "S3_ENDPOINT_URL": endpoint_url,
        "NEBIUS_S3_ENDPOINT": endpoint_url,
        "NPA_REQUIRE_HF": "0",
        "COSMOS_MODEL_ID": MODEL_ID,
        "COSMOS_MODEL_DIR": "/opt/cosmos/models",
        "COSMOS_DISABLE_SAFETY": "1",
        "COSMOS_SMOKE_PROMPT": PROMPT,
        "COSMOS_SMOKE_STEPS": "2",
        "COSMOS_SMOKE_SEED": "42",
        "COSMOS_SMOKE_NUM_FRAMES": "2",
        "COSMOS_SMOKE_HEIGHT": "256",
        "COSMOS_SMOKE_WIDTH": "256",
        "NPA_JOB_NAME": test_id,
        "NPA_COSMOS_RICHER": "1",
        "NPA_COSMOS_E2E": "1",
    }
    hf_token = _hf_token()
    extra_env = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "HF_TOKEN": hf_token,
        "HUGGING_FACE_HUB_TOKEN": hf_token,
        "HUGGINGFACE_HUB_TOKEN": hf_token,
    }
    return env, extra_env


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


def _cosmos_smoke_command() -> str:
    job_py = r'''
import importlib
import json
import os
import pathlib
import sys
import time
import traceback
from urllib.parse import urlparse

out_dir = pathlib.Path("/tmp/npa-cosmos-e2e-output")
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
        "format": "npa_cosmos_serverless_richer_smoke_v2",
        "status": "started",
        "job_name": os.environ.get("NPA_JOB_NAME", ""),
        "model": model_id,
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
    return "bash -lc " + shlex.quote(script)


def _subnet_id(project_id: str) -> str:
    override = os.environ.get("NPA_E2E_SERVERLESS_SUBNET_ID", "")
    if override:
        return override
    result = subprocess.run(
        ["nebius", "vpc", "subnet", "list", "--parent-id", project_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout or "{}")
    items = data.get("items") if isinstance(data, dict) else data
    ready = [
        item
        for item in items or []
        if str(((item.get("status") or {}).get("state") or "")).upper() in {"READY", ""}
    ]
    ranked = sorted(
        ready,
        key=lambda item: (
            "cosmos" not in str((item.get("metadata") or {}).get("name", "")).lower(),
            "default" not in str((item.get("metadata") or {}).get("name", "")).lower(),
        ),
    )
    assert ranked, f"No READY subnet found for {project_id}"
    return str((ranked[0].get("metadata") or {}).get("id") or "")


def _poll_job(client: ServerlessClient, project_id: str, job_id: str, artifacts_dir: Path):
    deadline = time.monotonic() + MAX_WAIT
    last = None
    tick = 0
    while time.monotonic() <= deadline:
        tick += 1
        current = client.get_job(job_id, project_id)
        last = current
        (artifacts_dir / f"job-detail-tick-{tick}.json").write_text(
            json.dumps(current.raw, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _capture_logs(job_id, artifacts_dir / f"job-logs-tick-{tick}.txt")
        if current.status in {"succeeded", "failed", "cancelled"}:
            return current
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"Job {job_id} did not finish within {MAX_WAIT}s; last={last}")


def _capture_job(project_id: str, job_id: str, artifacts_dir: Path, *, label: str) -> None:
    try:
        info = ServerlessClient().get_job(job_id, project_id)
        (artifacts_dir / f"job-detail-{label}.json").write_text(
            json.dumps(info.raw, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        (artifacts_dir / f"job-detail-{label}.err").write_text(str(exc), encoding="utf-8")
    _capture_logs(job_id, artifacts_dir / f"job-logs-{label}.txt")


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
    client = ServerlessClient()
    try:
        info = client.cancel_job(ref, project_id)
        job_id = info.id or ref
    except EndpointNotFoundError:
        return
    except Exception as exc:
        (artifacts_dir / "cleanup-cancel.err").write_text(str(exc), encoding="utf-8")
        job_id = ref
    result = subprocess.run(
        ["nebius", "ai", "job", "delete", "--id", job_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    (artifacts_dir / "cleanup-delete.log").write_text(result.stdout, encoding="utf-8")
    orphan = subprocess.run(
        ["nebius", "ai", "job", "get", "--id", job_id, "--format", "json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    (artifacts_dir / "cleanup-orphan-check.log").write_text(orphan.stdout, encoding="utf-8")


def _download_s3_prefix(
    output_path: str,
    local_dir: Path,
    access_key: str,
    secret_key: str,
    endpoint_url: str,
) -> None:
    parsed = urlparse(output_path)
    prefix = parsed.path.lstrip("/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    assert keys, f"No Cosmos artifacts found under {output_path}"
    for key in keys:
        rel = key.removeprefix(prefix).lstrip("/")
        target = local_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(parsed.netloc, key, str(target))


def _ffprobe_video(video_path: Path, artifacts_dir: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(video_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    (artifacts_dir / "video-ffprobe.json").write_text(result.stdout, encoding="utf-8")
    (artifacts_dir / "video-ffprobe.err").write_text(result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    streams = json.loads(result.stdout).get("streams") or []
    assert streams, result.stdout
    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "codec_type": stream.get("codec_type"),
        "height": stream.get("height"),
        "width": stream.get("width"),
    }
