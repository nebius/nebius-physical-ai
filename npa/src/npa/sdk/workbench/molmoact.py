"""MolmoAct workbench SDK; mirrors the npa molmoact CLI stages.

The three stages are stubs in this release: they expose the intended
signatures but raise NotImplementedError naming the tracking issue,
matching the honest stub behavior of npa.cli.workbench.molmoact and
npa.workflows.byof.molmoact_pipeline. The three-tier contract checks
signatures, not behavior, so the SDK functions stay thin until the
MolmoAct fine-tuning pipeline lands (nebius/nebius-physical-ai#502).

Model weights are never bundled: the base model (default
``allenai/MolmoAct-7B-O-0812``) is resolved at runtime through the HF Hub
cache (``npa.workbench.model_access``), using the operator's HF token.
"""

from __future__ import annotations

_ISSUE_REF = "nebius/nebius-physical-ai#502"

_NOT_IMPLEMENTED = (
    "MolmoAct workbench pipeline stages are stubs in this release "
    f"(tracking issue {_ISSUE_REF})."
)


def _not_implemented(stage: str) -> None:
    """Raise the stub notice for *stage*."""
    raise NotImplementedError(f"stage {stage!r}: {_NOT_IMPLEMENTED}")


def finetune(
    *,
    model_id: str = "allenai/MolmoAct-7B-O-0812",
    dataset_uri: str = "",
    output_s3_uri: str = "",
    max_steps: int = 5000,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    num_gpus: int = 1,
    run_name: str = "",
) -> None:
    """Fine-tune a MolmoAct policy on a demonstration dataset (stub)."""
    _not_implemented("finetune")


def serve(
    *,
    model_id: str = "allenai/MolmoAct-7B-O-0812",
    checkpoint: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
) -> None:
    """Serve a MolmoAct policy behind an HTTP endpoint (stub)."""
    _not_implemented("serve")


def eval(
    *,
    model_id: str = "allenai/MolmoAct-7B-O-0812",
    dataset_uri: str = "",
    checkpoint: str = "",
    num_episodes: int = 50,
    output_s3_uri: str = "",
) -> None:
    """Evaluate a MolmoAct policy on an evaluation dataset (stub)."""
    _not_implemented("eval")
