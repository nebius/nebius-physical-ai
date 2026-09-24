"""Newton physics workbench SDK; mirrors the npa newton CLI stages.

The three stages are stubs in this release: they expose the intended
signatures but raise NotImplementedError naming the tracking issue,
matching the honest stub behavior of npa.cli.newton and
npa.workflows.byof.newton_pipeline. The three-tier contract checks
signatures, not behavior, so the SDK functions stay thin until the Newton
simulation pipeline lands (nebius/nebius-physical-ai#499).
"""

from __future__ import annotations

_ISSUE_REF = "nebius/nebius-physical-ai#499"

_NOT_IMPLEMENTED = (
    "Newton workbench pipeline stages are stubs in this release "
    f"(tracking issue {_ISSUE_REF})."
)


def _not_implemented(stage: str) -> None:
    """Raise the stub notice for *stage*."""
    raise NotImplementedError(f"stage {stage!r}: {_NOT_IMPLEMENTED}")


def train_teacher(
    *,
    dataset_uri: str,
    output_uri: str,
    config_name: str = "newton_double_pendulum",
    train_steps: int = 100,
    seed: int | None = None,
) -> None:
    """Train a teacher policy in Newton simulation (stub)."""
    _not_implemented("train-teacher")


def generate_demos(
    *,
    checkpoint_uri: str,
    output_uri: str,
    num_demos: int = 10,
    seed: int | None = None,
) -> None:
    """Generate demonstration rollouts from a Newton teacher policy (stub)."""
    _not_implemented("generate-demos")


def eval(
    *,
    checkpoint_uri: str,
    dataset_uri: str,
    output_uri: str,
    num_episodes: int = 5,
    seed: int | None = None,
) -> None:
    """Evaluate a policy in Newton simulation (stub)."""
    _not_implemented("eval")
