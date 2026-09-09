"""Authenticated ingress for NVIDIA Cosmos Framework's native Ray Serve path."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

from npa.workbench.cosmos.ray_inputs import stage_sample_inputs
from npa.workbench.cosmos.ray_serve import (
    RayArtifact,
    RayBatchRequest,
    RayBatchResponse,
)
from npa.workbench.storage_scope import StorageAuthorizationError, StorageScope


def main() -> None:
    """Load Cosmos3-Nano once and serve real dynamically batched generation.

    Args:
        None; service configuration comes from environment variables.

    Returns:
        None; serves until the runtime stops.

    Raises:
        RuntimeError: Authentication, guarded startup, or service setup fails.
        OSError: Private credential or tokenizer files cannot be created.
    """

    token_file = _require_ray_authentication()
    try:
        _prepare_guardrail_tokenizer()
        _run_server()
    finally:
        token_file.unlink(missing_ok=True)


def _prepare_guardrail_tokenizer() -> None:
    """Reuse the pinned regular-file tokenizer cache before importing NLTK.

    NLTK's enforced path checks reject Hub snapshot symlinks. The shared Cosmos
    materializer preserves that boundary and verifies cached bytes on reuse.
    Setting NLTK_DATA before Ray starts also configures its model workers.
    """
    if not _env_bool("NPA_COSMOS3_RAY_GUARDRAILS", True):
        return
    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError(
            "HF_TOKEN is required when Cosmos guardrails are enabled; "
            "no tokenizer download was attempted"
        )
    from npa.workbench.cosmos.transfer import (
        _guardrail_nltk_data_path,
        prepare_guardrail_nltk_data,
    )

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    prepare_guardrail_nltk_data(hf_home=hf_home)
    safe_data = str(_guardrail_nltk_data_path(hf_home))
    os.environ["NLTK_DATA"] = os.pathsep.join(
        part for part in (safe_data, os.environ.get("NLTK_DATA", "")) if part
    )


def _run_server() -> None:

    import fastapi
    import ray
    import ray.serve
    from fastapi import Body, Header
    from fastapi.responses import FileResponse

    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.ray.serve import OmniModelDeployment

    output_root = Path(
        os.environ.get("NPA_COSMOS3_RAY_OUTPUT_DIR", "/outputs")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_name = os.environ.get("NPA_COSMOS3_RAY_MODEL", "Cosmos3-Nano")
    checkpoint = os.environ.get("NPA_COSMOS3_RAY_CHECKPOINT", "Cosmos3-Nano")
    framework_revision = os.environ.get("NPA_COSMOS3_FRAMEWORK_REVISION", "unknown")
    source_revision = os.environ.get("NPA_IMAGE_SOURCE_SHA", "")
    if len(source_revision) != 40 or any(
        c not in "0123456789abcdef" for c in source_revision
    ):
        raise RuntimeError(
            "NPA_IMAGE_SOURCE_SHA must be the exact 40-character source commit"
        )
    guardrails = _env_bool("NPA_COSMOS3_RAY_GUARDRAILS", True)
    world_size = _env_int("NPA_COSMOS3_RAY_WORLD_SIZE", 1, minimum=1)
    max_batch_size = _env_int("NPA_COSMOS3_RAY_MAX_BATCH_SIZE", 4, minimum=1)
    max_ongoing_requests = max_batch_size
    wait_timeout = _env_float("NPA_COSMOS3_RAY_BATCH_WAIT_TIMEOUT_S", 0.05, minimum=0.0)
    host = os.environ.get("NPA_COSMOS3_RAY_HOST", "0.0.0.0")
    port = _env_int("NPA_COSMOS3_RAY_PORT", 8000, minimum=1024)
    token = os.environ.get("NPA_COSMOS3_RAY_TOKEN", "")
    input_scope = StorageScope.from_config(
        s3_roots=os.environ.get("NPA_COSMOS3_RAY_ALLOWED_S3_ROOTS", "").split(",")
    )
    if not token:
        raise RuntimeError(
            "NPA_COSMOS3_RAY_TOKEN is required; unauthenticated GPU serving is refused"
        )
    if guardrails and not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError(
            "HF_TOKEN is required when Cosmos guardrails are enabled; "
            "run `npa workbench health access --capability cosmos3` before launch"
        )

    setup_overrides = OmniSetupOverrides.model_validate(
        {
            "checkpoint_path": checkpoint,
            "guardrails": guardrails,
            "parallelism_preset": os.environ.get(
                "NPA_COSMOS3_RAY_PARALLELISM_PRESET", "throughput"
            ),
            "output_dir": str(output_root),
        }
    )
    setup_args = setup_overrides.build_setup(world_size=world_size)
    model = (
        cast(ray.serve.Deployment, OmniModelDeployment)
        .options(
            name=model_name,
            max_ongoing_requests=max_ongoing_requests,
            user_config={
                "max_batch_size": max_batch_size,
                "batch_wait_timeout_s": wait_timeout,
            },
        )
        .bind(setup_args)
    )

    api = fastapi.FastAPI(title="NPA Cosmos3 Native Ray Serve")

    @ray.serve.deployment(num_replicas=1)
    @ray.serve.ingress(api)
    class NpaCosmosRouter:
        def __init__(self, handle: Any):
            self.handle = handle

        def _authorize(self, authorization: str) -> None:
            if not hmac.compare_digest(authorization, f"Bearer {token}"):
                raise fastapi.HTTPException(
                    status_code=401, detail="invalid bearer token"
                )

        @api.get("/health")
        async def health(
            self, authorization: str = Header(default="")
        ) -> dict[str, Any]:
            self._authorize(authorization)
            return {"status": "ok", "backend": "cosmos-framework-native-ray-serve"}

        @api.get("/ready")
        async def ready(
            self, authorization: str = Header(default="")
        ) -> dict[str, Any]:
            self._authorize(authorization)
            try:
                # Ingress can accept requests while its model replica is still
                # initializing. Query the controller through Ray's public API.
                status = await asyncio.to_thread(ray.serve.status)
                application = status.applications.get("npa_cosmos3_ray_serve")
                deployment = (
                    application.deployments.get(model_name)
                    if application is not None
                    else None
                )
                running = (
                    deployment.replica_states.get("RUNNING", 0)
                    if deployment is not None
                    else 0
                )
                model_ready = (
                    application is not None
                    and application.status == "RUNNING"
                    and deployment is not None
                    and deployment.status == "HEALTHY"
                    and type(running) is int
                    and running > 0
                )
            except Exception:
                model_ready = False
            if not model_ready:
                raise fastapi.HTTPException(
                    status_code=503, detail="model deployment is not ready"
                )
            return {
                "status": "ready",
                "backend": "cosmos-framework-native-ray-serve",
                "model": model_name,
                "guardrails": guardrails,
                "world_size": world_size,
                "max_batch_size": max_batch_size,
                "model_max_ongoing_requests": max_ongoing_requests,
                "batch_wait_timeout_s": wait_timeout,
                "framework_revision": framework_revision,
                "server_source_revision": source_revision,
                "weights_baked": False,
            }

        @api.get("/models")
        async def models(self, authorization: str = Header(default="")) -> list[str]:
            self._authorize(authorization)
            return [model_name]

        @api.get("/system-info")
        async def system_info(
            self, authorization: str = Header(default="")
        ) -> dict[str, Any]:
            self._authorize(authorization)
            return {
                "accelerators": _nvidia_accelerators(),
                "ray_cluster_resources": ray.cluster_resources(),
                "framework_revision": framework_revision,
                "server_source_revision": source_revision,
            }

        @api.post("/v1/batches")
        async def batches(
            self,
            body: dict[str, Any] = Body(...),
            authorization: str = Header(default=""),
        ) -> dict[str, Any]:
            self._authorize(authorization)
            # Ray rewrites and freezes the class-based endpoint. Keep its JSON
            # body explicit so FastAPI cannot reclassify it as a query field.
            # Keep Pydantic models outside FastAPI's route metadata.  Ray 2.46
            # cloudpickles that metadata when freezing an ingress app, and the
            # pinned Python 3.13/Pydantic combination recursively serializes a
            # model's mock validator.  Explicit validation preserves the exact
            # wire contract without making the model part of Ray's app pickle.
            request = RayBatchRequest.model_validate(body)
            if request.model != model_name:
                raise fastapi.HTTPException(
                    status_code=404, detail=f"model {request.model!r} is not loaded"
                )
            request_id = request.request_id or uuid.uuid4().hex
            request_root = (output_root / request_id).resolve()
            if output_root not in request_root.parents:
                raise fastapi.HTTPException(
                    status_code=400, detail="invalid request id"
                )
            request_root.mkdir(parents=True, exist_ok=False)

            samples = []
            for raw in request.samples:
                try:
                    staged = await asyncio.to_thread(
                        stage_sample_inputs, raw,
                        request_root / "inputs" / str(raw["name"]), scope=input_scope,
                    )
                except StorageAuthorizationError as exc:
                    raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
                sample = OmniSampleOverrides.model_validate(staged)
                sample.output_dir = request_root / str(raw["name"])
                sample.download(sample.output_dir / "inputs")
                samples.append(sample)

            # Concurrent handle calls are intentional: NVIDIA's deployment owns
            # @ray.serve.batch and coalesces these into its generate_batch path.
            outputs = await asyncio.gather(
                *(self.handle.generate.remote(sample) for sample in samples)
            )
            artifacts: list[RayArtifact] = []
            for sample, result in zip(samples, outputs, strict=True):
                for output in result.outputs:
                    for relative in output.files:
                        path = (output_root / relative).resolve()
                        if request_root not in path.parents or not path.is_file():
                            raise RuntimeError(
                                f"Cosmos output file is missing: {relative}"
                            )
                        artifacts.append(
                            RayArtifact(
                                sample=str(sample.name),
                                path=str(path.relative_to(output_root)),
                                bytes=path.stat().st_size,
                                sha256=_sha256(path),
                            )
                        )
            return RayBatchResponse(
                request_id=request_id,
                model=model_name,
                batch_size=len(samples),
                outputs=[output.model_dump(mode="json") for output in outputs],
                artifacts=artifacts,
                guardrails=guardrails,
                max_batch_size=max_batch_size,
                framework_revision=framework_revision,
                server_source_revision=source_revision,
            ).model_dump(mode="json")

        @api.get("/v1/artifacts/{artifact_path:path}")
        async def artifact(
            self, artifact_path: str, authorization: str = Header(default="")
        ):
            self._authorize(authorization)
            path = (output_root / artifact_path).resolve()
            if output_root not in path.parents or not path.is_file():
                raise fastapi.HTTPException(
                    status_code=404, detail="artifact not found"
                )
            return FileResponse(path)

    # Own this service's runtime instead of joining an ambient RAY_ADDRESS.
    # Inference does not need the management dashboard or Jobs HTTP API.
    ray.init(address="local", include_dashboard=False, _node_ip_address="127.0.0.1")
    try:
        ray.serve.start(http_options={"host": host, "port": port})
        app = cast(ray.serve.Deployment, NpaCosmosRouter).bind(model)
        ray.serve.run(app, name="npa_cosmos3_ray_serve", route_prefix="/", blocking=True)
    finally:
        ray.shutdown()


def _require_ray_authentication() -> Path:
    """Configure Ray's management boundary before importing its runtime."""
    mode = os.environ.get("RAY_AUTH_MODE", "token")
    if mode != "token":
        raise RuntimeError("Cosmos Ray serving requires RAY_AUTH_MODE=token")
    # This service owns its local cluster. Use a fresh owner-only credential,
    # without reusing the application token or an operator's ~/.ray token.
    # A path keeps the credential itself out of inherited environment dumps.
    with tempfile.NamedTemporaryFile(prefix="npa-ray-auth-", delete=False) as token:
        token.write(secrets.token_urlsafe(32).encode("ascii"))
        token.flush()
    os.environ.pop("RAY_AUTH_TOKEN", None)
    os.environ["RAY_AUTH_TOKEN_PATH"] = token.name
    os.environ["RAY_AUTH_MODE"] = "token"
    return Path(token.name)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_accelerators() -> list[dict[str, str]]:
    """Return non-identifying physical GPU model/capability evidence."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    accelerators = []
    for line in output.splitlines():
        name, separator, capability = line.partition(",")
        if separator:
            accelerators.append(
                {"name": name.strip(), "compute_capability": capability.strip()}
            )
    return accelerators


if __name__ == "__main__":
    main()
