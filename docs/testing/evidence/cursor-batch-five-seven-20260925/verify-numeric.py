"""Recheck the public numeric export on CPU without storage credentials or Torch.

This verifies numerical consistency, not physical GPU allocation. Provider and
artifact readback evidence are summarized separately in the GPU proof report.
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "gpu-workload"))
from artifact_validation import (
    _loss_gradient,
    _oracle,
    _validate_journal,
    _validate_parameters,
)
from regression_contract import RECIPE, _close, _dataset


def verify(document):
    """Compare published measured values against the scalar CPU SGD oracle."""
    metrics = document["metrics"]
    if metrics["recipe"] != RECIPE:
        raise ValueError("recipe differs")
    parameters, momentum, journal = _oracle()
    _validate_journal(metrics["journal"], journal)
    _validate_parameters(document["checkpoint_numeric_state"], parameters, momentum)
    rows, targets = _dataset(RECIPE["seed"], RECIPE["samples"])
    final_loss, _ = _loss_gradient(parameters, rows, targets)
    rows, targets = _dataset(RECIPE["seed"] + 1, RECIPE["heldout_samples"])
    heldout_loss, _ = _loss_gradient(parameters, rows, targets)
    _close(metrics["final_train_loss"], final_loss, "final train loss")
    _close(metrics["heldout_loss"], heldout_loss, "held-out loss")
    _close(
        document["independent_cpu_verification"]["initial_train_loss"],
        journal[0]["loss"],
        "initial loss",
    )
    if not 0 < final_loss < journal[0]["loss"] * 0.02:
        raise ValueError("loss improvement gate failed")
    return {
        "steps_verified": len(journal),
        "final_train_loss": final_loss,
        "heldout_loss": heldout_loss,
    }


def controls(document):
    """Reject three altered copies while preserving the actual public data."""
    outcomes = {}
    for case in ("changed_weight", "zero_parameter_motion", "missing_step"):
        changed = copy.deepcopy(document)
        if case == "changed_weight":
            changed["checkpoint_numeric_state"]["parameters"][0] += 0.1
        elif case == "zero_parameter_motion":
            changed["metrics"]["journal"][0]["parameter_delta"] = 0.0
        else:
            changed["metrics"]["journal"].pop()
        try:
            verify(changed)
        except ValueError:
            outcomes[case] = "rejected"
        else:
            raise AssertionError(f"altered evidence accepted: {case}")
    return outcomes


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: verify-numeric.py [numeric-export.json]")
    filename = Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / "baseline-numeric.json"
    source = json.loads(filename.read_text())
    result = {
        "classification": "CPU consistency check of published measurements",
        "actual_export": verify(source),
        "copied_negative_controls": controls(source),
        "gpu_execution_claim": False,
    }
    print(json.dumps(result, indent=2))
