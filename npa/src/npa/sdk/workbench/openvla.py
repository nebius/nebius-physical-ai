"""OpenVLA workbench SDK; mirrors the npa openvla CLI stages.

The three stages are stubs in this release: they expose the intended
signatures but raise NotImplementedError naming the tracking issue,
matching the honest stub behavior of npa.cli.workbench.openvla and
npa.workflows.byof.openvla_pipeline. The three-tier contract checks
signatures, not behavior, so the SDK functions stay thin until the
OpenVLA-OFT fine-tuning pipeline lands (nebius/nebius-physical-ai#500).

Model weights are never bundled: the base model (default
``openvla/openvla-7b``) is resolved at runtime through the HF Hub cache
(``npa.workbench.model_access``), using the operator's HF token.
"""

from __future__ import annotations

_ISSUE_REF = "nebius/nebius-physical-ai#500"

_NOT_IMPLEMENTED = (
    "OpenVLA workbench pipeline stages are stubs in this release "
    f"(tracking issue {_ISSUE_REF})."
)


def _not_implemented(stage: str) -> None:
    """Raise the stub notice for *stage*."""
    raise NotImplementedError(f"stage {stage!r}: {_NOT_IMPLEMENTED}")


def train(
    *,
    model_id: str = "openvla/openvla-7b",
    dataset_uri: str,
    dataset_name: str = "finetune",
    output_dir: str = "runs/openvla-oft",
    batch_size: int = 16,
    max_steps: int = 200_000,
    learning_rate: float = 5e-4,
    lora_rank: int = 32,
    lora_dropout: float = 0.0,
    image_aug: bool = True,
    seed: int = 7,
    dry_run: bool = False,
) -> None:
    """Fine-tune OpenVLA with the OpenVLA-OFT LoRA recipe (stub)."""
    _not_implemented("train")


def serve(
    *,
    checkpoint: str = "openvla/openvla-7b",
    host: str = "127.0.0.1",
    port: int = 8000,
    dry_run: bool = False,
) -> None:
    """Serve an OpenVLA checkpoint over HTTP (stub)."""
    _not_implemented("serve")


def eval(
    *,
    checkpoint: str,
    dataset_uri: str,
    num_episodes: int = 10,
    output_uri: str = "",
    seed: int = 7,
    dry_run: bool = False,
) -> None:
    """Evaluate an OpenVLA checkpoint (stub)."""
    _not_implemented("eval")
