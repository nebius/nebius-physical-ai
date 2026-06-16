"""Tests for Sim2Real K8s submit script resolution."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_submit_script_prefers_explicit_env(monkeypatch, tmp_path: Path) -> None:
    from npa.workflows.sim2real import k8s_submit

    script = tmp_path / "submit-k8s-staged-job.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setenv("NPA_SIM2REAL_SUBMIT_SCRIPT", str(script))

    assert k8s_submit._resolve_submit_script() == script.resolve()


def test_resolve_submit_script_walkthrough_layout(monkeypatch, tmp_path: Path) -> None:
    from npa.workflows.sim2real import k8s_submit

    walkthrough = tmp_path / "demo"
    npa_root = walkthrough / "nebius-physical-ai"
    (npa_root / "npa").mkdir(parents=True)
    (npa_root / "npa" / "pyproject.toml").write_text("[project]\nname='npa'\n", encoding="utf-8")
    script = walkthrough / "operator" / "sim2real-rtxpro" / "submit-k8s-staged-job.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    monkeypatch.delenv("NPA_SIM2REAL_SUBMIT_SCRIPT", raising=False)
    monkeypatch.setattr(k8s_submit, "_repo_root", lambda: npa_root)

    assert k8s_submit._resolve_submit_script() == script.resolve()


def test_resolve_submit_script_missing_raises(monkeypatch, tmp_path: Path) -> None:
    from npa.workflows.sim2real import k8s_submit

    npa_root = tmp_path / "nebius-physical-ai"
    (npa_root / "npa").mkdir(parents=True)
    (npa_root / "npa" / "pyproject.toml").write_text("[project]\nname='npa'\n", encoding="utf-8")

    monkeypatch.delenv("NPA_SIM2REAL_SUBMIT_SCRIPT", raising=False)
    monkeypatch.setattr(k8s_submit, "_repo_root", lambda: npa_root)

    with pytest.raises(FileNotFoundError, match="missing sim2real operator submit script"):
        k8s_submit._resolve_submit_script()
