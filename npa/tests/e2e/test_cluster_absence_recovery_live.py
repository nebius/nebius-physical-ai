"""Opt-in real terminal reconciliation; never provision or delete cloud resources."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from npa.cluster.reconcile_absent import reconcile_absent

pytestmark = pytest.mark.e2e


def test_owned_cluster_absence_reconciliation_live():
    selected = os.environ.get("NPA_ABSENCE_RECOVERY_LIVE_CONFIG")
    if not selected:
        pytest.skip("requires explicit private owned-operation recovery configuration")
    config = json.loads(Path(selected).read_text())
    assert config["allow_terminal_reconciliation"] is True
    repo = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    assert source == config["source_revision"]
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo
    )
    manifest = Path(config["evidence_file"])
    assert (
        hashlib.sha256(manifest.read_bytes()).hexdigest() == config["evidence_sha256"]
    )
    result = reconcile_absent(manifest)
    assert result["operation_id"] == config["operation_id"]
    assert result["status"] == "reconciled-destroyed"
    assert result["fresh_provider_verification"] is True
