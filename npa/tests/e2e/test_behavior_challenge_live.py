"""Opt-in execution of the real BEHAVIOR evaluator in an authorized GPU runtime."""

import argparse
import json
import os
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge.execution import evaluate


@pytest.mark.e2e
@pytest.mark.gpu
def test_behavior_challenge_live():
    """Run the prescribed selection from a private worker configuration file."""
    config = os.environ.get("NPA_BEHAVIOR_LIVE_CONFIG")
    if not config:
        pytest.skip(
            "Requires licensed assets, fixed policy, and NPA_BEHAVIOR_LIVE_CONFIG"
        )
    settings = json.loads(Path(config).read_text())
    for field in (
        "upstream_root",
        "policy_root",
        "policy_python",
        "policy_checkpoint",
        "policy_archive",
    ):
        if settings.get(field):
            settings[field] = Path(settings[field])
    result = evaluate(argparse.Namespace(**settings))
    assert result["complete"]
    assert result["completed"] == result["planned"]
    assert result["completed"] > 0
