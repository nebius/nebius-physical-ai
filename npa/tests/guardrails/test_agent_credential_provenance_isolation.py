"""Keep staged agent credentials out of non-live provider request tests."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_PROVIDER_TRIPWIRE = """
import pytest
from npa.cli import agent_resources

def test_non_live_discovery_does_not_inherit_agent_provenance(monkeypatch):
    def reject_provider(*args, **kwargs):
        pytest.fail("non-live discovery inherited staged credential provenance")
    monkeypatch.setattr(agent_resources, "run_bounded_agent_command", reject_provider)
    assert agent_resources.run_resource_discovery_command(
        ["nebius", "--profile", "synthetic-profile", "iam", "project", "list"]
    ) == (2, "", "agent credential source is unavailable")
"""


def test_non_live_fixtures_scrub_inherited_agent_provenance(tmp_path):
    tests = Path(__file__).resolve().parents[1]
    case = tmp_path / "test_provenance_tripwire.py"
    case.write_text(_PROVIDER_TRIPWIRE, encoding="utf-8")
    launcher = (
        "import importlib.util, sys, pytest; "
        f"spec=importlib.util.spec_from_file_location('isolated_npa_fixtures', {str(tests / 'conftest.py')!r}); "
        "plugin=importlib.util.module_from_spec(spec); spec.loader.exec_module(plugin); "
        f"sys.exit(pytest.main([{str(case)!r}, '-q', '--tb=short', "
        f"'--confcutdir', {str(tmp_path)!r}], plugins=[plugin]))"
    )
    env = dict(os.environ)
    for key in (
        "NPA_CI_SHARD_INDEX",
        "NPA_CI_TOTAL_SHARDS",
        "NPA_CI_TIMING_OUTPUT",
        "PYTEST_ADDOPTS",
    ):
        env.pop(key, None)
    env["NPA_NEBIUS_CREDENTIAL_SOURCE"] = "instance_metadata"
    env["PYTHONPATH"] = str(tests.parent / "src")
    result = subprocess.run(
        [sys.executable, "-c", launcher],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
