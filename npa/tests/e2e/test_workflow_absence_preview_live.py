"""Opt-in original Sky absence preview; never release leases or mutate cloud resources."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from npa.orchestration.skypilot.absence_recovery import reconcile_absent
from npa.provisioning_journal import load_operation

pytestmark = pytest.mark.e2e


def test_original_sky_absence_preview_live():
    selected = os.environ.get("NPA_SKY_ABSENCE_PREVIEW_LIVE_CONFIG")
    if not selected:
        pytest.skip("requires exact private original-operation preview configuration")
    config = json.loads(Path(selected).read_bytes())
    repo = Path(__file__).resolve().parents[3]
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        == config["source_revision"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    manifest = Path(config["evidence_file"])
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() == config["evidence_sha256"]
    )
    operation = load_operation(config["operation_id"])
    before = operation.path.read_bytes()
    result = reconcile_absent(manifest)
    assert result["status"] == "absence-verified-not-applied"
    assert result["historical_workload_outcome"] == "unknown"
    assert result["cloud_mutations"] is False
    assert operation.path.read_bytes() == before
