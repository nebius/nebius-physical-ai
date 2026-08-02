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


__all__ = [
    "Cosmos3GenerateError",
    "Cosmos3ReasonConfig",
    "GENERATE_MODES",
    "generate",
    "reason",
]
