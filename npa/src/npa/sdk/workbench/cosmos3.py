"""SDK helpers for Cosmos3 generation and reason workflow contracts."""

from __future__ import annotations

from typing import Any

from npa.workbench.cosmos.generate import (
    DEFAULT_MODE,
    DEFAULT_NAME,
    DEFAULT_PARALLELISM_PRESET,
    GENERATE_MODES,
    Cosmos3GenerateError,
    generate_and_publish,
)
from npa.workbench.cosmos.ray_serve import (
    DEFAULT_TOKEN_ENV,
    Cosmos3RayServeError,
    service_health,
    submit_batch,
)
from npa.workbench.cosmos.super_benchmark import (
    PRIMARY_SUITE,
    TOPOLOGY_ORDER,
    Cosmos3SuperBenchmarkError,
    run_benchmark,
)
from npa.workflows.cosmos_split import Cosmos3ReasonConfig, build_cosmos3_reason_manifest


def generate(
    *,
    prompt: str,
    output_path: str,
    mode: str = DEFAULT_MODE,
    input_path: str = "",
    checkpoint: str = "",
    name: str = DEFAULT_NAME,
    negative_prompt: str = "",
    seed: int = 0,
    num_steps: int = 0,
    guidance: float = 0.0,
    no_guardrails: bool = False,
    parallelism_preset: str = DEFAULT_PARALLELISM_PRESET,
    run_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a real Cosmos 3 generation and return its artifact manifest.

    Same implementation as ``npa workbench cosmos3 generate``: needs the
    ``npa-cosmos3`` runtime on a GPU plus the operator's Hugging Face token for the
    gated checkpoint (no weights ship in the image). ``output_path`` may be a local
    directory or an ``s3://`` prefix; ``dry_run=True`` resolves the plan only.
    """

    return generate_and_publish(
        mode=mode,
        prompt=prompt,
        name=name,
        checkpoint=checkpoint,
        input_path=input_path,
        output_path=output_path,
        negative_prompt=negative_prompt,
        seed=seed,
        num_steps=num_steps,
        guidance=guidance,
        no_guardrails=no_guardrails,
        parallelism_preset=parallelism_preset,
        run_id=run_id,
        dry_run=dry_run,
    )


def reason(
    *,
    input_uri: str,
    output_uri: str,
    model: str = "nvidia/Cosmos-Reason1-7B",
    image: str = "",
    prompt: str = "",
    run_id: str = "",
) -> dict[str, object]:
    """Return a Cosmos3 reason manifest."""

    return build_cosmos3_reason_manifest(
        Cosmos3ReasonConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            model=model,
            image=image,
            prompt=prompt,
            run_id=run_id,
        )
    )


def ray_batch(
    *,
    input_path: str,
    output_path: str,
    endpoint: str = "",
    token_env: str = DEFAULT_TOKEN_ENV,
    timeout: float = 1800.0,
    run_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a durable batch to native Cosmos Framework Ray Serve."""

    return submit_batch(
        input_path=input_path,
        output_path=output_path,
        endpoint=endpoint,
        token_env=token_env,
        timeout=timeout,
        run_id=run_id,
        dry_run=dry_run,
    )


def ray_health(
    *, endpoint: str = "", token_env: str = DEFAULT_TOKEN_ENV, timeout: float = 30.0
) -> dict[str, Any]:
    """Return model-backed readiness for native Cosmos Framework Ray Serve."""

    return service_health(endpoint=endpoint, token_env=token_env, timeout=timeout)


def super_benchmark(
    *,
    output_path: str,
    topologies: str = ",".join(TOPOLOGY_ORDER),
    attempts: int = 24,
    suite: str = PRIMARY_SUITE,
    gpu_family: str = "B200",
    base_port: int = 8100,
    run_id: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the immutable Cosmos3-Super node or H200 single-GPU suite."""

    return run_benchmark(
        output_path=output_path,
        topologies=topologies,
        attempts=attempts,
        suite=suite,
        gpu_family=gpu_family,
        base_port=base_port,
        run_id=run_id,
        dry_run=dry_run,
    )


__all__ = [
    "Cosmos3GenerateError",
    "Cosmos3RayServeError",
    "Cosmos3SuperBenchmarkError",
    "Cosmos3ReasonConfig",
    "GENERATE_MODES",
    "generate",
    "ray_batch",
    "ray_health",
    "reason",
    "super_benchmark",
]
