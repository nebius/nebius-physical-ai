"""Unit tests for the container-side torch architecture checker.

The script itself runs inside workbench images (where only torch is available),
so these tests import it by path and exercise the pure parsing/compatibility
helpers - no torch, no GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker/workbench/base/cuda13-b300/scripts/check_torch_gpu_arch.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_torch_gpu_arch", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load()


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("sm_80", (8, 0)),
        ("sm_90", (9, 0)),
        ("sm_100", (10, 0)),
        ("sm_103", (10, 3)),
        ("sm_120", (12, 0)),
        ("10.0", (10, 0)),
        ("10.3", (10, 3)),
        ("120", (12, 0)),
    ],
)
def test_parse_arch(checker: ModuleType, token: str, expected: tuple[int, int]) -> None:
    assert checker.parse_arch(token) == expected


def test_parse_arch_rejects_garbage(checker: ModuleType) -> None:
    with pytest.raises(ValueError):
        checker.parse_arch("blackwell")


def test_wheel_arch_set_ignores_ptx_entries(checker: ModuleType) -> None:
    arch_list = ["sm_80", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert checker.wheel_arch_set(arch_list) == {(8, 0), (9, 0), (10, 0), (12, 0)}


def test_sm_100_sass_covers_a_sm_103_device(checker: ModuleType) -> None:
    """Minor-version forward compat inside a major: B200 SASS runs on B300."""

    available = {(8, 0), (9, 0), (10, 0), (12, 0)}
    assert checker.arch_is_covered((10, 3), available, exact=False) is True


def test_sm_103_sass_does_not_cover_a_sm_100_device(checker: ModuleType) -> None:
    """Forward compat is one-way: a 10.3-only build does not run on B200."""

    available = {(10, 3)}
    assert checker.arch_is_covered((10, 0), available, exact=False) is False


def test_majors_never_cross(checker: ModuleType) -> None:
    """sm_120 (major 12) proves nothing about sm_100/sm_103 (major 10)."""

    workstation_only = {(12, 0)}
    assert checker.arch_is_covered((10, 0), workstation_only, exact=False) is False
    assert checker.arch_is_covered((10, 3), workstation_only, exact=False) is False

    datacenter_only = {(10, 0), (10, 3)}
    assert checker.arch_is_covered((12, 0), datacenter_only, exact=False) is False


def test_hopper_capped_wheel_misses_blackwell(checker: ModuleType) -> None:
    """cu124/cu126 stop at sm_90, which is the whole reason npa-cosmos is a PORT."""

    cu126 = checker.wheel_arch_set(["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"])
    assert checker.arch_is_covered((10, 0), cu126, exact=False) is False
    assert checker.arch_is_covered((9, 0), cu126, exact=False) is True


def test_exact_arch_disables_forward_compat(checker: ModuleType) -> None:
    available = {(10, 0)}
    assert checker.arch_is_covered((10, 3), available, exact=True) is False
    assert checker.arch_is_covered((10, 0), available, exact=True) is True


def test_format_arch_round_trips(checker: ModuleType) -> None:
    for token in ("sm_80", "sm_90", "sm_100", "sm_103", "sm_120"):
        assert checker.format_arch(checker.parse_arch(token)) == token


def test_known_capabilities_name_the_blackwell_parts(checker: ModuleType) -> None:
    assert "B200" in checker.KNOWN_CAPABILITIES[(10, 0)]
    assert "B300" in checker.KNOWN_CAPABILITIES[(10, 3)]
    assert "RTX PRO 6000" in checker.KNOWN_CAPABILITIES[(12, 0)]
