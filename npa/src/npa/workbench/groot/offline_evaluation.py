"""Compare recorded matched robot trials without treating training loss as success."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from npa.workbench.groot.offline_feedback import file_digest


class MatchedTrial(BaseModel):
    """Recorded outcomes for two policies under a shared trial protocol.

    Args:
        trial_id: Unique matched trial identifier.
        baseline_success: Observed baseline outcome.
        candidate_success: Observed candidate outcome.
    Returns:
        None.
    Raises:
        ValueError: An outcome is not a literal boolean.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    trial_id: str = Field(min_length=1)
    baseline_success: bool
    candidate_success: bool


class RolloutComparison(BaseModel):
    """Checkpoint and protocol identities for operator-recorded evaluation trials.

    Args:
        schema_version: Rollout comparison format identity.
        baseline_checkpoint_sha256: Baseline model identity.
        candidate_checkpoint_sha256: Candidate model identity.
        protocol_sha256: Frozen evaluation conditions and trial schedule identity.
        trials: Nonempty matched trials from that protocol.
    Returns:
        None.
    Raises:
        ValueError: Trials repeat or checkpoint identities are identical.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["npa.groot-offline-rollouts/v1"]
    baseline_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trials: list[MatchedTrial] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_identity(self):
        identifiers = [trial.trial_id for trial in self.trials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("trial_id must be unique")
        if self.baseline_checkpoint_sha256 == self.candidate_checkpoint_sha256:
            raise ValueError(
                "baseline and candidate must identify different checkpoints"
            )
        return self


def compare_rollouts(
    rollouts_path: Path, training_receipt_path: Path, output_path: Path
) -> dict:
    """Summarize paired outcomes while retaining operator control of deployment.

    Args:
        rollouts_path: JSON with recorded matched trials and checkpoint identities.
        training_receipt_path: Native training receipt identifying both checkpoints.
        output_path: New report file.
    Returns:
        Success counts, rates, Wilson intervals, and paired wins/losses.
    Raises:
        ValueError: Rollout schema or identities are invalid.
        OSError: Input cannot be read or output already exists.
    """
    comparison = RolloutComparison.model_validate_json(rollouts_path.read_text())
    receipt = json.loads(training_receipt_path.read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "npa.groot-offline-training/v1"
    ):
        raise ValueError("expected an offline N1.5 training receipt")
    for role in ("baseline", "candidate"):
        key = f"{role}_checkpoint_sha256"
        receipt_key = "base_checkpoint_sha256" if role == "baseline" else key
        if getattr(comparison, key) != receipt.get(receipt_key):
            raise ValueError(
                "rollout checkpoint identity differs from training receipt"
            )
    count = len(comparison.trials)
    baseline = sum(trial.baseline_success for trial in comparison.trials)
    candidate = sum(trial.candidate_success for trial in comparison.trials)
    report = {
        "schema_version": "npa.groot-offline-comparison/v1",
        "evidence_kind": "operator-recorded-matched-rollouts",
        "training_receipt_sha256": file_digest(training_receipt_path),
        "baseline_checkpoint_sha256": comparison.baseline_checkpoint_sha256,
        "candidate_checkpoint_sha256": comparison.candidate_checkpoint_sha256,
        "protocol_sha256": comparison.protocol_sha256,
        "trials": count,
        "baseline": _success_summary(baseline, count),
        "candidate": _success_summary(candidate, count),
        "success_rate_delta": (candidate - baseline) / count,
        "paired_wins": sum(
            not t.baseline_success and t.candidate_success for t in comparison.trials
        ),
        "paired_losses": sum(
            t.baseline_success and not t.candidate_success for t in comparison.trials
        ),
        "deployment_decision": "operator-review-required",
    }
    with output_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report


def _success_summary(successes, count):
    rate = successes / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
        / denominator
    )
    return {
        "successes": successes,
        "rate": rate,
        "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
    }
