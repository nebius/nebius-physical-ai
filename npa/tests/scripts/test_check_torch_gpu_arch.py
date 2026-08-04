"""Unit tests for the container-side torch architecture checker.

The script itself runs inside workbench images (where only torch is available),
so these tests import it by path. They cover the pure parsing/compatibility
helpers, plus the report assembly and failure text driven through a stub torch -
no real torch, no GPU. The device paths themselves are only provable on real
hardware; see docs/workbench/image-gpu-compatibility-matrix.md for those runs.
"""

from __future__ import annotations

import importlib.util
import sys
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


@pytest.mark.parametrize("token", ["9", "8", "sm_9"])
def test_parse_arch_rejects_a_bare_single_digit(checker: ModuleType, token: str) -> None:
    """The bare form drops the sm_ prefix, so one digit would mean sm_9.

    It must fail with this function's own actionable message rather than an
    opaque int() error from slicing the string to nothing.
    """

    with pytest.raises(ValueError, match="use sm_100, 10.0, or 100"):
        checker.parse_arch(token)


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
    """cu124/cu126 stop at sm_90, which forced the historical npa-cosmos port."""

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


CU130_ARCHES = "sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120"
CU126_ARCHES = "sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90"


def _fake_torch(arch_flags: str, devices: list[tuple[str, tuple[int, int]]]):
    """Stand in for torch so build_report's device branch is unit-testable.

    The real device paths only run on a GPU; this covers the report assembly and
    the failure text, which are pure logic over what torch reports.
    """

    from types import SimpleNamespace

    return SimpleNamespace(
        __version__="2.9.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        _C=SimpleNamespace(_cuda_getArchFlags=lambda: arch_flags),
        cuda=SimpleNamespace(
            is_available=lambda: bool(devices),
            device_count=lambda: len(devices),
            get_device_name=lambda i: devices[i][0],
            get_device_capability=lambda i=0: devices[i][1],
            get_arch_list=lambda: arch_flags.split(),
        ),
    )


@pytest.fixture
def with_fake_torch(monkeypatch):
    def _install(arch_flags: str, devices: list[tuple[str, tuple[int, int]]]):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(arch_flags, devices))

    return _install


def _args(**overrides):
    from argparse import Namespace

    base = {
        "require_arch": [],
        "require_capability": [],
        "exact_arch": False,
        "require_sass_coverage": False,
        "json": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_build_report_describes_a_visible_device(checker, with_fake_torch) -> None:
    with_fake_torch(CU130_ARCHES, [("NVIDIA H100 80GB HBM3", (9, 0))])

    report, failures = checker.build_report(
        _args(require_arch=["sm_90"], require_capability=["9.0"])
    )

    assert not failures and report["ok"] is True
    assert report["cuda_available"] is True
    device = report["devices"][0]
    assert device["arch"] == "sm_90"
    assert device["capability"] == [9, 0]
    assert device["sass_covered"] is True
    assert "Hopper" in device["known_as"]


def test_build_report_flags_a_device_with_no_matching_sass(checker, with_fake_torch) -> None:
    """A cu126 wheel on B300: the device runs, but every kernel would PTX-JIT."""

    with_fake_torch(CU126_ARCHES, [("NVIDIA B300", (10, 3))])

    report, failures = checker.build_report(
        _args(require_capability=["10.3"], require_sass_coverage=True)
    )

    assert report["devices"][0]["sass_covered"] is False
    assert any("no matching SASS" in f and "PTX JIT" in f for f in failures)
    assert report["ok"] is False


def test_build_report_rejects_the_wrong_gpu_family(checker, with_fake_torch) -> None:
    """The guard that stops an sm_120 pass being read as Blackwell readiness."""

    with_fake_torch(CU130_ARCHES, [("NVIDIA RTX PRO 6000 Blackwell", (12, 0))])

    _, failures = checker.build_report(_args(require_capability=["10.0"]))

    assert any("expected one of [(10, 0)]" in f for f in failures)
    assert any("different CUDA majors" in f for f in failures)


def test_build_report_will_not_claim_a_capability_with_no_device(
    checker, with_fake_torch
) -> None:
    with_fake_torch(CU130_ARCHES, [])

    report, failures = checker.build_report(_args(require_capability=["10.0"]))

    assert report["devices"] == []
    assert any("no CUDA device is visible" in f for f in failures)


def test_build_report_reports_a_hopper_capped_wheel(checker, with_fake_torch) -> None:
    with_fake_torch(CU126_ARCHES, [])

    _, failures = checker.build_report(_args(require_arch=["sm_100"]))

    assert any("does not cover sm_100" in f for f in failures)
