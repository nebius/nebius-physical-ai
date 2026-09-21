"""Opt-in execution of the real BEHAVIOR evaluator in an authorized GPU runtime."""

import argparse
import json
import os
from pathlib import Path

import pytest

from npa.workflows.behavior_challenge.execution import evaluate
from npa.workflows.behavior_challenge.campaign_runner import evaluate_partition


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


@pytest.mark.e2e
@pytest.mark.gpu
def test_behavior_campaign_live():
    """Run or recover one real frozen partition through its durable case store."""
    config = os.environ.get("NPA_BEHAVIOR_CAMPAIGN_LIVE_CONFIG")
    if not config:
        pytest.skip("Requires a licensed runtime and NPA_BEHAVIOR_CAMPAIGN_LIVE_CONFIG")
    settings = json.loads(Path(config).read_bytes())
    path_fields = (
        "workspace",
        "upstream_root",
        "policy_root",
        "policy_python",
        "policy_checkpoint",
        "policy_archive",
        "policy_selected_export_receipt",
        "policy_correlation_manifest",
        "policy_validation_receipt",
        "policy_stock_correlation_asset",
    )
    for field in path_fields:
        if settings.get(field):
            settings[field] = Path(settings[field])
    result = evaluate_partition(argparse.Namespace(**settings))
    assert result["completed"] > 0
    assert result["completed"] == len(result["case_ids"])
    assert len(set(result["case_ids"])) == result["completed"]
