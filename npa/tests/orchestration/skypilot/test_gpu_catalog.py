from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from npa.orchestration.skypilot import gpu_catalog as catalog_module
from npa.orchestration.skypilot.gpu_catalog import (
    InvalidNebiusGpuRequestError,
    NebiusGpuCatalog,
    discover_nebius_gpu_catalog,
    parse_accelerator_request,
    parse_nebius_gpu_catalog,
    resolve_nebius_gpu_preferences,
)


SKY_SHOW_GPUS_OUTPUT = """\
WARNING: `sky show-gpus` has been renamed to `sky gpus list` and will be removed in a future release.

The --cloud, --region, and --zone options are deprecated. Use --infra instead.
COMMON_GPU  AVAILABLE_QUANTITIES
B200        8
H100        1, 8
H200        1, 8
L40S        1, 2, 4

Hint: use -a/--all to see all accelerators (including non-common ones) and pricing.
"""


def _catalog() -> NebiusGpuCatalog:
    return parse_nebius_gpu_catalog(SKY_SHOW_GPUS_OUTPUT)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_parse_nebius_gpu_catalog_discovers_quantity_sets() -> None:
    catalog = _catalog()

    assert catalog.quantities_by_accelerator == {
        "B200": frozenset({8}),
        "H100": frozenset({1, 8}),
        "H200": frozenset({1, 8}),
        "L40S": frozenset({1, 2, 4}),
    }
    assert "A100" not in catalog.quantities_by_accelerator
    assert "RTX6000" not in catalog.quantities_by_accelerator


def test_parse_accelerator_request_defaults_bare_names_to_one() -> None:
    assert parse_accelerator_request("h100") == catalog_module.AcceleratorRequest("h100", 1)
    assert parse_accelerator_request("B200:8") == catalog_module.AcceleratorRequest("B200", 8)

    with pytest.raises(ValueError, match="positive integer"):
        parse_accelerator_request("H100:many")


def test_resolver_filters_invalid_nebius_strings_and_quantities() -> None:
    resolution = resolve_nebius_gpu_preferences(
        "A100:1",
        "RTX6000:1,B200:1,H200:8,L40S:4",
        catalog=_catalog(),
    )

    assert resolution.selected == "H200:8"
    assert resolution.accelerators == ("H200:8", "L40S:4")
    assert any("A100:1" in rejected for rejected in resolution.rejected)
    assert any("RTX6000:1" in rejected for rejected in resolution.rejected)
    assert any("B200:1" in rejected for rejected in resolution.rejected)


def test_resolver_accepts_explicit_b200_multi_gpu_request() -> None:
    resolution = resolve_nebius_gpu_preferences("B200:8", "H100:1", catalog=_catalog())

    assert resolution.selected == "B200:8"
    assert resolution.accelerators == ("B200:8", "H100:1")


def test_resolver_raises_clear_error_when_none_are_valid() -> None:
    with pytest.raises(InvalidNebiusGpuRequestError) as exc_info:
        resolve_nebius_gpu_preferences("A100:1", "RTX6000:1,B200:1", catalog=_catalog())

    message = str(exc_info.value)
    assert "Requested: A100:1, RTX6000:1, B200:1" in message
    assert "Currently valid Nebius VM accelerators:" in message
    assert "H100: 1, 8" in message
    assert "L40S: 1, 2, 4" in message


def test_discover_nebius_gpu_catalog_retries_transient_and_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sky = _executable(tmp_path / "sky")
    responses = [
        subprocess.CompletedProcess([str(sky)], 1, stdout="", stderr="temporary auth cache miss"),
        subprocess.CompletedProcess([str(sky)], 0, stdout="COMMON_GPU  AVAILABLE_QUANTITIES\n", stderr=""),
        subprocess.CompletedProcess([str(sky)], 0, stdout=SKY_SHOW_GPUS_OUTPUT, stderr=""),
    ]
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return responses.pop(0)

    monkeypatch.setattr(catalog_module.subprocess, "run", fake_run)

    catalog = discover_nebius_gpu_catalog(
        sky_bin=sky,
        backoff_seconds=0.25,
        sleep=sleeps.append,
    )

    assert catalog.quantities_by_accelerator["H100"] == frozenset({1, 8})
    assert calls == [[str(sky.resolve()), "show-gpus", "--cloud", "nebius"]] * 3
    assert sleeps == [0.25, 0.5]
