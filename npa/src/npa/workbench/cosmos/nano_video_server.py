"""Ray Serve wrapper for the single-GPU Cosmos3 vLLM-Omni diffusion service.

Only ``app`` imports Ray. The request and artifact contracts can be validated on
an ordinary CPU host without importing vLLM, torch, or allocating a GPU.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib.metadata
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

MODEL_REVISION = "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"
MODEL_NAME = "nvidia/Cosmos3-Nano"
PIPELINE = "Cosmos3OmniDiffusersPipeline"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LOG = logging.getLogger(__name__)


async def _await_accepted_generation(generation: asyncio.Task) -> Any:
    """Drain accepted thread work despite repeated HTTP cancellation signals."""
    cancelled = False
    while not generation.done():
        try:
            await asyncio.shield(generation)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            if not cancelled:
                raise
            # Retrieve the terminal exception below before propagating the
            # caller's cancellation; never detach a still-running GPU thread.
    if cancelled:
        if not generation.cancelled():
            try:
                generation.result()
            except Exception:
                LOG.exception("Accepted generation failed while draining a disconnected request")
        raise asyncio.CancelledError
    return generation.result()


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value) or len(value) > 128:
        raise HTTPException(
            400, "request_id must contain only letters, digits, _, -, and ."
        )
    return value


def validate_weights(model_path: Path) -> dict[str, Any]:
    """Require the completed, immutable pre-stage contract before server launch."""
    ready = json.loads((model_path / "READY.json").read_text())
    if ready.get("revision") != MODEL_REVISION or ready.get("precision") != "BF16":
        raise ValueError(
            "Shared checkpoint revision or precision does not match the deployment"
        )
    if not ready.get("tensor_count") or not ready.get("files"):
        raise ValueError("Shared checkpoint has no verified tensor/file manifest")
    index = json.loads((model_path / "model_index.json").read_text())
    if index.get("_class_name") != PIPELINE:
        raise ValueError("Shared checkpoint is not the Cosmos3 diffusion pipeline")
    vae = json.loads((model_path / "vae" / "config.json").read_text())
    if vae.get("scale_factor_temporal") != 4 or vae.get("scale_factor_spatial") != 16:
        raise ValueError(
            "Shared checkpoint VAE stride differs from the rollout contract"
        )
    return ready


def diffusion_stage_config() -> dict[str, Any]:
    """Use the pinned engine's explicit stage path; fallback drops stage overrides."""
    return {"stage_args": [{
        "stage_id": 0,
        "stage_type": "diffusion",
        "runtime": {"process": True, "devices": "0"},
        "engine_args": {
            "model_stage": "diffusion",
            "model_class_name": PIPELINE,
            "dtype": "bfloat16",
            "parallel_config": {"tensor_parallel_size": 1},
            "model_config": {"sound_gen": False, "guardrails": False},
            "enable_diffusion_pipeline_profiler": True,
        },
        "default_sampling_params": {"num_inference_steps": 35},
        "final_output": True,
        "final_output_type": "image",
    }]}


def server_argv(model_path: Path, port: int, stage_config_path: Path) -> list[str]:
    return [
        "vllm",
        "serve",
        str(model_path),
        "--omni",
        "--model-class-name",
        PIPELINE,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--stage-configs-path",
        str(stage_config_path),
        "--enable-diffusion-pipeline-profiler",
        "--init-timeout",
        "1800",
        "--no-guardrails",
    ]


class NanoVideoRuntime:
    """One vLLM subprocess and sequential complete rollouts on its assigned GPU."""

    def __init__(self) -> None:
        self.model_path = Path(os.environ["NPA_COSMOS3_MODEL_PATH"])
        self.output_root = Path(os.environ["NPA_COSMOS3_VIDEO_OUTPUT_ROOT"])
        self.token = os.environ.get("NPA_COSMOS3_VIDEO_TOKEN", "")
        if not self.token:
            raise ValueError("NPA_COSMOS3_VIDEO_TOKEN is required")
        self.weights = validate_weights(self.model_path)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.replica_id = uuid.uuid4().hex
        self.process: subprocess.Popen | None = None
        self.endpoint = ""
        self._log_stream: Any = None
        self._generation_lock = asyncio.Lock()

    def authorize(self, request: Request) -> None:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {self.token}".encode()):
            raise HTTPException(401, "Bearer authentication required")

    async def start(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "inf",
                "VLLM_NO_USAGE_STATS": "1",
                "DO_NOT_TRACK": "1",
            }
        )
        # Upstream inference does not need credentials when loading prestaged
        # weights with guardrails off. Keep the API secret in the parent only.
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NPA_COSMOS3_VIDEO_TOKEN"):
            environment.pop(name, None)
        logs = self.output_root / ".server-logs"
        logs.mkdir(exist_ok=True)
        from .nano_video import write_json

        stage_config_path = logs / f"{self.replica_id}.stage.json"
        write_json(stage_config_path, diffusion_stage_config())
        self._log_stream = (logs / f"{self.replica_id}.log").open("ab", buffering=0)
        self.process = subprocess.Popen(
            server_argv(self.model_path, port, stage_config_path),
            env=environment,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            while self.process.poll() is None:
                try:
                    response = await client.get(f"{self.endpoint}/v1/models")
                    if response.is_success and response.json().get("data"):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(1)
        raise RuntimeError(
            "vLLM diffusion initialization failed; inspect private replica logs"
        )

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self._log_stream is not None:
            self._log_stream.close()

    async def check_health(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("vLLM diffusion process is unavailable")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.endpoint}/v1/models")
            response.raise_for_status()

    async def run(self, request: Request) -> dict[str, Any]:
        self.authorize(request)
        if request.headers.get("content-type", "").split(";")[0].strip() == "multipart/form-data":
            return await self._run_augmentation(request)
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected JSON request") from exc
        if not isinstance(body, dict) or set(body) != {"prompt", "seed", "request_id"}:
            raise HTTPException(400, "Expected prompt, seed, and request_id")
        request_id = _request_id(body["request_id"])
        prompt, seed = body["prompt"], body["seed"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(400, "prompt must be nonempty text")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise HTTPException(400, "seed must be a nonnegative signed 64-bit integer")
        directory = self.output_root / request_id
        # Lazy import preserves a GPU-free request-contract test surface.
        from .nano_video import run_rollout

        try:
            async with self._generation_lock:
                generation = asyncio.create_task(
                    asyncio.to_thread(
                        run_rollout,
                        endpoint=self.endpoint,
                        output_dir=directory,
                        prompt=prompt,
                        seed=seed,
                        replica_id=self.replica_id,
                    )
                )
                report = await _await_accepted_generation(generation)
            report["request_id"] = request_id
            return report
        except FileExistsError as exc:
            raise HTTPException(
                409, "request_id already exists; inspect its status"
            ) from exc
        except Exception as exc:
            LOG.exception("Cosmos3 rollout failed for request %s", request_id)
            raise HTTPException(
                500, "Generation failed; inspect the private workload evidence"
            ) from exc

    async def _run_augmentation(self, request: Request) -> dict[str, Any]:
        from .nano_video_augment import (
            AugmentationInputError,
            run_augmentation,
            validate_request,
        )

        temporary: Path | None = None
        request_id = "unparsed"
        try:
            async with request.form(max_files=1, max_fields=1) as form:
                if set(form) != {"request", "input_reference"} or any(len(form.getlist(key)) != 1 for key in form):
                    raise HTTPException(400, "Expected request JSON and one input_reference video")
                metadata, upload = form["request"], form["input_reference"]
                if not isinstance(metadata, str) or not isinstance(upload, UploadFile):
                    raise HTTPException(400, "Invalid augmentation multipart fields")
                try:
                    body = validate_request(json.loads(metadata))
                except (ValueError, AugmentationInputError) as exc:
                    raise HTTPException(400, "Invalid augmentation request or unsupported controls") from exc
                request_id = _request_id(body["request_id"])
                directory = self.output_root / request_id
                if directory.exists():
                    raise HTTPException(409, "request_id already exists; retrieve its result")
                uploads = self.output_root / ".uploads"
                uploads.mkdir(mode=0o700, exist_ok=True)
                if uploads.is_symlink():
                    raise RuntimeError("Upload directory cannot be a symlink")
                with tempfile.NamedTemporaryFile(dir=uploads, suffix=".mp4", delete=False) as stream:
                    temporary = Path(stream.name)
                    received = 0
                    while block := await upload.read(1024 * 1024):
                        received += len(block)
                        if received > body["source_bytes"]:
                            raise AugmentationInputError("Uploaded source exceeds its declared byte count")
                        stream.write(block)
                    if received != body["source_bytes"]:
                        raise AugmentationInputError("Uploaded source byte count differs from its declaration")
                    stream.flush()
                    os.fsync(stream.fileno())
                async with self._generation_lock:
                    generation = asyncio.create_task(asyncio.to_thread(
                        run_augmentation, endpoint=self.endpoint, output_dir=directory,
                        input_video=temporary, request=body, replica_id=self.replica_id,
                    ))
                    return await _await_accepted_generation(generation)
        except StarletteHTTPException:
            raise
        except AugmentationInputError as exc:
            raise HTTPException(422, "Source video failed hash, decode, timestamp or 480p/24fps validation") from exc
        except FileExistsError as exc:
            raise HTTPException(409, "request_id already exists; retrieve its result") from exc
        except Exception as exc:
            LOG.exception("Cosmos3 augmentation failed for request %s", request_id)
            raise HTTPException(500, "Augmentation failed; inspect the private workload evidence") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def result(self, request: Request, request_id: str) -> JSONResponse:
        """Recover durable results on any replica; never admit generation here."""
        self.authorize(request)
        request_id = _request_id(request_id)
        directory = self.output_root / request_id
        if not directory.exists():
            raise HTTPException(404, "Unknown request; absence does not authorize a retry")
        if directory.is_symlink() or not directory.is_dir():
            raise HTTPException(503, "Request state unavailable")
        path = directory / "report.json"
        if not path.exists():
            return JSONResponse({"request_id": request_id, "status": "initializing"}, status_code=202)
        if path.is_symlink() or not path.is_file():
            raise HTTPException(503, "Request state unavailable")
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict) or report.get("status") not in {"queued", "running", "succeeded", "failed"}:
                raise ValueError("Invalid durable state")
            from .nano_video_augment import SCHEMA, request_sha256, validate_report

            if report.get("schema_version") == SCHEMA:
                if report["request_id"] != request_id or report["request_sha256"] != request_sha256(report["request"]):
                    raise ValueError("Request identity mismatch")
                if report["status"] == "succeeded":
                    validate_report(report, report["request"])
            # Legacy continuation stored the ID in its directory, then added it
            # only to the HTTP response. Preserve that established format.
            report["request_id"] = request_id
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise HTTPException(503, "Request state unavailable") from exc
        return JSONResponse(report, status_code=200 if report["status"] in {"succeeded", "failed"} else 202)

    def status(self, request: Request, request_id: str) -> dict[str, Any]:
        self.authorize(request)
        path = self.output_root / _request_id(request_id) / "report.json"
        if not path.is_file() or path.is_symlink() or path.parent.is_symlink():
            raise HTTPException(404, "Unknown request")
        report = json.loads(path.read_text())
        return {
            "request_id": request_id,
            **{
                key: report[key]
                for key in ("status", "replica_id", "started_at", "finished_at", "request_sha256", "schema_version")
                if key in report
            },
        }

    def artifact(self, request: Request, request_id: str, name: str) -> FileResponse:
        self.authorize(request)
        request_id = _request_id(request_id)
        if not SAFE_NAME.fullmatch(name) or name.startswith("."):
            raise HTTPException(400, "Invalid artifact name")
        directory = self.output_root / request_id
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or directory.is_symlink()
            or path.resolve().parent != directory.resolve()
            or path.suffix not in {".mp4", ".mkv", ".png", ".json"}
        ):
            raise HTTPException(404, "Unknown artifact")
        report_path = directory / "report.json"
        if not report_path.is_file() or report_path.is_symlink():
            raise HTTPException(404, "Unknown artifact")
        report = json.loads(report_path.read_text())
        if name not in {item["path"] for item in report.get("artifacts", [])}:
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(path, filename=name)

    def list_runs(self, request: Request) -> dict[str, Any]:
        self.authorize(request)
        runs = []
        for directory in sorted(self.output_root.iterdir()):
            if (
                not directory.is_dir()
                or directory.is_symlink()
                or directory.name.startswith(".")
            ):
                continue
            path = directory / "report.json"
            if path.is_file() and not path.is_symlink():
                runs.append(self.status(request, directory.name))
        return {"runs": runs}

    def system_info(self, request: Request) -> dict[str, Any]:
        self.authorize(request)
        versions = {}
        for name in ("ray", "vllm", "vllm-omni", "torch"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unavailable"
        return {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "pipeline": PIPELINE,
            "precision": "BF16",
            "guardrails": False,
            "replica_id": self.replica_id,
            "gpus_per_replica": 1,
            "tensor_parallel_size": 1,
            "init_timeout": 1800,
            "versions": versions,
        }


def app(args: dict[str, Any] | None = None) -> Any:
    """Ray Serve application builder; one ranked set gives least-outstanding routing."""
    from ray import serve

    from .nano_video_router import LeastOutstandingRouter

    api = FastAPI(title="Cosmos3-Nano video generation")

    @serve.deployment(
        name="Cosmos3NanoVideo",
        num_replicas=16,
        max_ongoing_requests=1,
        ray_actor_options={"num_gpus": 1, "num_cpus": 2},
        request_router_config={"request_router_class": LeastOutstandingRouter},
    )
    @serve.ingress(api)
    class Cosmos3NanoVideo:
        async def __init__(self):
            self.runtime = NanoVideoRuntime()
            await self.runtime.start()

        def __del__(self):
            if hasattr(self, "runtime"):
                self.runtime.close()

        async def check_health(self):
            await self.runtime.check_health()

        @api.post("/run")
        async def run(self, request: Request):
            return await self.runtime.run(request)

        @api.get("/health")
        async def health(self, request: Request):
            self.runtime.authorize(request)
            try:
                await self.runtime.check_health()
            except (RuntimeError, httpx.HTTPError):
                return JSONResponse({"status": "unavailable"}, status_code=503)
            return {"status": "ready", "replica_id": self.runtime.replica_id}

        @api.get("/system-info")
        async def system_info(self, request: Request):
            return self.runtime.system_info(request)

        @api.get("/status")
        async def status(self, request: Request, request_id: str):
            return self.runtime.status(request, request_id)

        @api.get("/result")
        async def result(self, request: Request, request_id: str):
            return self.runtime.result(request, request_id)

        @api.get("/list")
        async def list_runs(self, request: Request):
            return self.runtime.list_runs(request)

        @api.get("/artifacts/{request_id}/{name}")
        async def artifact(self, request: Request, request_id: str, name: str):
            return self.runtime.artifact(request, request_id, name)

    return Cosmos3NanoVideo.bind()


if __name__ == "__main__":
    if sys.argv[1:] != ["--stage-weights"]:
        raise SystemExit("Use Ray Serve's application builder or --stage-weights")
    from .nano_video_stage import stage_weights

    stage_weights(Path(os.environ["NPA_COSMOS3_MODEL_PATH"]))
