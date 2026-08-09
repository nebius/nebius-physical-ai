"""Fail-closed provenance for image source and operator-fetched Isaac closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workflows.sim2real.runtime_attestation import attest_runtime_source


SOURCE_SHA = "1" * 40
IMAGE = f"registry.example/npa-isaac@sha256:{'a' * 64}"


def _source_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SIM2REAL_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", SOURCE_SHA)
    monkeypatch.setenv("NPA_SIM2REAL_RUNTIME_IMAGE", IMAGE)


def _cache_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, bytes]:
    cache = tmp_path / "cache"
    stamp = "b" * 64
    target = cache / "v" / stamp
    (target / "venv" / "bin").mkdir(parents=True)
    (target / "venv" / "bin" / "python").write_text("python", encoding="utf-8")
    (target / ".complete").touch()
    bootstrap = tmp_path / "isaac-bootstrap"
    bootstrap.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bootstrap_sha = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    manifest_bytes = json.dumps(
        {
            "format": "npa_isaac_runtime_cache_v1",
            "cache_stamp": stamp,
            "bootstrap_sha256": bootstrap_sha,
        },
        sort_keys=True,
    ).encode()
    (target / "MANIFEST.json").write_bytes(manifest_bytes)
    (cache / "current").symlink_to(target)
    monkeypatch.setenv("NPA_ISAAC_CACHE_DIR", str(cache))
    monkeypatch.setenv("NPA_ISAAC_CACHE_READONLY", "1")
    monkeypatch.setenv("NPA_ISAAC_BOOTSTRAP_OFFLINE", "1")
    monkeypatch.setenv("NPA_ISAAC_BOOTSTRAP_PATH", str(bootstrap))
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_CACHE_PVC", "npa-isaac-cache")
    return target, bootstrap, manifest_bytes


def test_non_isaac_runtime_attests_exact_source_and_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source_env(monkeypatch)
    assert attest_runtime_source() == {
        "source_sha": SOURCE_SHA,
        "runtime_image": IMAGE,
    }


def test_readonly_isaac_runtime_attests_content_addressed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_env(monkeypatch)
    target, bootstrap, manifest_bytes = _cache_tree(tmp_path, monkeypatch)

    result = attest_runtime_source()

    assert result["isaac_cache_stamp"] == target.name
    assert result["isaac_cache_pvc"] == "npa-isaac-cache"
    assert result["isaac_cache_mode"] == "offline-readonly"
    assert (
        result["isaac_cache_manifest_sha256"]
        == hashlib.sha256(manifest_bytes).hexdigest()
    )
    assert (
        result["isaac_bootstrap_sha256"]
        == hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    )


def test_readonly_isaac_cache_fails_if_network_is_still_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_env(monkeypatch)
    _cache_tree(tmp_path, monkeypatch)
    monkeypatch.setenv("NPA_ISAAC_BOOTSTRAP_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="also be offline"):
        attest_runtime_source()


def test_isaac_cache_warmed_by_other_bootstrap_fails_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_env(monkeypatch)
    _, bootstrap, _ = _cache_tree(tmp_path, monkeypatch)
    bootstrap.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different bootstrap"):
        attest_runtime_source()


def test_isaac_cache_pointer_cannot_escape_the_version_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_env(monkeypatch)
    target, _, _ = _cache_tree(tmp_path, monkeypatch)
    current = target.parents[1] / "current"
    current.unlink()
    current.symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="escapes or is incomplete"):
        attest_runtime_source()
