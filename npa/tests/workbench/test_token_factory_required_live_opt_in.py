"""Prove actual childpytest live gating without contacting a provider."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("required", "opt_in", "key_present", "exit_code", "spy_reached"),
    [
        (False, False, True, 0, False),
        (True, False, True, 4, False),
        (True, True, False, 4, False),
        (True, True, True, 1, True),
    ],
)
def test_required_live_cannot_succeed_as_a_skipped_run(
    tmp_path, required, opt_in, key_present, exit_code, spy_reached
):
    repo = Path(__file__).resolve().parents[3]
    live_node = next(
        "npa/tests/e2e/" + name
        for name in (
            "test_vlm_disclosures_live_e2e.py",
            "test_token_factory_caption_thinking_live.py",
            "test_vlm_benchmark_scope_live_e2e.py",
            "test_token_factory_image_availability_live.py",
        )
        if (repo / "npa/tests/e2e" / name).is_file()
    )
    plugin = tmp_path / "blocked_provider.py"
    reached = tmp_path / "provider-spy-reached"
    plugin.write_text(
        "from pathlib import Path\n"
        "from npa.clients.token_factory import TokenFactoryClient\n"
        "def pytest_configure(config):\n"
        "    def blocked(*args, **kwargs):\n"
        f"        Path({str(reached)!r}).write_text('blocked model-list spy')\n"
        "        raise AssertionError('MODEL_LIST_BLOCKED_NO_NETWORK')\n"
        "    TokenFactoryClient.list_models = blocked\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(repo / "npa/src"))),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if key_present:
        env["NEBIUS_TOKEN_FACTORY_KEY"] = "synthetic-test-key"
    if opt_in:
        env["NPA_INTEGRATION_E2E"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        live_node,
        "-p",
        "blocked_provider",
        "--basetemp",
        str(tmp_path / "child-pytest"),
        "-q",
        "-ra",
    ]
    if required:
        command.append("--require-token-factory-live")
    result = subprocess.run(
        command, cwd=repo, env=env, capture_output=True, text=True
    )
    (tmp_path / "child-output.log").write_text(result.stdout + result.stderr)
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert reached.exists() is spy_reached
    if required and not opt_in:
        assert "needs NPA_INTEGRATION_E2E=1" in result.stderr
    elif required and not key_present:
        assert "needs NEBIUS_TOKEN_FACTORY_KEY" in result.stderr
    elif not opt_in:
        assert "1 skipped" in result.stdout
    else:
        assert "MODEL_LIST_BLOCKED_NO_NETWORK" in result.stdout
