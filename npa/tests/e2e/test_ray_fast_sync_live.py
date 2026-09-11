"""Run the two-revision source-sync proof on an operator-selected Ray endpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import uuid

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]
EXAMPLE = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development"


def _configuration() -> tuple[dict[str, str], Path]:
    """Load an owner-only endpoint and evidence root.

    Args:
        None.
    Returns:
        Private live configuration and its owner-only evidence root.
    Raises:
        AssertionError: The selected configuration or evidence root is unsafe.
        OSError: Private configuration cannot be read.
    """
    if not os.environ.get("NPA_RAY_FAST_SYNC_LIVE_CONFIG"):
        pytest.skip("requires an operator-selected Ray 2.58 Jobs endpoint")
    selected = os.environ["NPA_RAY_FAST_SYNC_LIVE_CONFIG"]
    path = Path(selected)
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_uid == os.getuid() and path.stat().st_mode & 0o077 == 0
    config = json.loads(path.read_text(encoding="utf-8"))
    assert set(config) == {"address", "evidence_dir"}
    evidence_root = Path(config["evidence_dir"])
    evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not evidence_root.is_symlink()
    assert evidence_root.stat().st_uid == os.getuid() and evidence_root.stat().st_mode & 0o077 == 0
    return config, evidence_root


def _load_harness():
    """Import the exact standalone qualification harness.

    Args:
        None.
    Returns:
        Imported fast-sync module.
    Raises:
        ImportError: The checked-in harness cannot be loaded.
    """
    specification = importlib.util.spec_from_file_location("ray_fast_sync_live", EXAMPLE / "fast_sync.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_local_edit_changes_real_ray_task_result() -> None:
    """Prove two uploaded source revisions change one real Ray task result.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Native working-directory delivery is stale or incomplete.
    """
    config, evidence_root = _configuration()
    token = uuid.uuid4().hex
    result = _load_harness().qualify(
        config["address"], evidence_root / token, run_token=token[:12]
    )
    assert result["status"] == "passed"
    assert [revision["value"] for revision in result["revisions"]] == ["before", "after"]
    assert len({revision["source_sha256"] for revision in result["revisions"]}) == 2
