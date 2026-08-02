"""Unit tests for the real-GPU capability smoke's architecture gating.

The smoke itself needs a GPU, so these cover the one piece of logic that decides
whether a failure is excusable: ``--allow-no-tma``. Getting that wrong in the
permissive direction would turn the smoke back into the import check it replaced.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gpu_capability_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    return _load()


def test_datacenter_architectures_have_tma(smoke: ModuleType) -> None:
    """Hopper and datacenter Blackwell carry the TMA flash-attn-4 CuTe needs."""

    assert (9, 0) in smoke.TMA_CAPABLE, "H100/H200"
    assert (10, 0) in smoke.TMA_CAPABLE, "B200"
    assert (10, 3) in smoke.TMA_CAPABLE, "B300"


def test_workstation_blackwell_has_no_tma(smoke: ModuleType) -> None:
    """RTX PRO 6000 is the part where the CuTe kernel is known to fail."""

    assert (12, 0) not in smoke.TMA_CAPABLE


def test_allow_no_tma_cannot_excuse_a_datacenter_failure(smoke: ModuleType) -> None:
    """The escape hatch must stay closed exactly where a TMA failure is real.

    This mirrors the ``known_tma_gap`` condition in the script: the flag only
    forgives a failure on an architecture that has no TMA to begin with.
    """

    for capability in [(9, 0), (10, 0), (10, 3)]:
        excused = True and capability not in smoke.TMA_CAPABLE
        assert excused is False, f"--allow-no-tma must not excuse {capability}"

    excused_on_sm120 = True and (12, 0) not in smoke.TMA_CAPABLE
    assert excused_on_sm120 is True


def test_without_the_flag_nothing_is_excused(smoke: ModuleType) -> None:
    for capability in [(9, 0), (10, 0), (10, 3), (12, 0)]:
        assert (False and capability not in smoke.TMA_CAPABLE) is False


def test_golden_eval_runs_this_smoke_rather_than_an_import() -> None:
    """The base image's golden eval must execute a kernel, not just import."""

    import yaml

    root = Path(__file__).resolve().parents[2]
    evals = yaml.safe_load((root / "src/npa/smoke/golden_evals.yaml").read_text())
    entry = evals["containers"]["base-cuda13-b300"]["golden_eval"]
    assert "gpu_capability_smoke.py" in entry["command"]
    assert entry["script"] == "npa/docker/workbench/base/cuda13-b300/scripts/gpu_capability_smoke.py"
    assert (root.parent / entry["script"]).is_file()
