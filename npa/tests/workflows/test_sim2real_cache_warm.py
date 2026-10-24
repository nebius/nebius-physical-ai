"""Functional unit contracts for the canonical Isaac cache warm state."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from npa.workflows.sim2real import cache_warm


def _bootstrap(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def test_warm_cache_runs_real_bootstrap_then_publishes_source_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = tmp_path / "isaac-bootstrap"
    _bootstrap(bootstrap)
    calls: list[list[str]] = []
    writes: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setenv("NPA_ISAAC_BOOTSTRAP_PATH", str(bootstrap))
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.example/isaac@sha256:" + "a" * 64)
    monkeypatch.setattr(cache_warm, "source_sha", lambda: "b" * 40)
    monkeypatch.setattr(
        cache_warm,
        "attest_warmed_isaac_cache",
        lambda: {"isaac_cache_mode": "warm-readwrite", "isaac_cache_stamp": "c" * 64},
    )

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ready=yes\n", "")

    def fake_write(uri, payload, *, directory):
        assert directory.is_dir()
        writes.append((uri, payload))
        return uri

    monkeypatch.setattr(cache_warm.subprocess, "run", fake_run)
    monkeypatch.setattr(cache_warm, "write_json", fake_write)

    result = cache_warm.warm_cache(root_uri="s3://customer-proof/run-1", run_id="run-1")

    assert calls == [[str(bootstrap), "warm"], [str(bootstrap), "status"]]
    assert result["uri"] == "s3://customer-proof/run-1/cache/isaac-warm.json"
    assert result["status"] == "ready"
    assert result["source_sha"] == "b" * 40
    assert writes == [
        (result["uri"], {key: result[key] for key in result if key != "uri"})
    ]


def test_warm_cache_refuses_mutable_image_before_downloading_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = tmp_path / "isaac-bootstrap"
    _bootstrap(bootstrap)
    monkeypatch.setenv("NPA_ISAAC_BOOTSTRAP_PATH", str(bootstrap))
    monkeypatch.setenv("NPA_TASK_IMAGE", "registry.example/isaac:mutable")
    monkeypatch.setattr(
        cache_warm.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "mutable image must fail before bootstrap"
        ),
    )

    with pytest.raises(RuntimeError, match="immutable NPA_TASK_IMAGE"):
        cache_warm.warm_cache(root_uri="s3://customer-proof/run-1", run_id="run-1")


def test_warm_cache_rejects_non_s3_artifact_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_ISAAC_BOOTSTRAP_PATH", raising=False)

    with pytest.raises(RuntimeError, match="s3://"):
        cache_warm.warm_cache(root_uri="file:///tmp/not-a-run", run_id="run-1")
