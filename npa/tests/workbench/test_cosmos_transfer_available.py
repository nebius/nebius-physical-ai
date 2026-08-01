"""Cosmos-Transfer availability probe must degrade gracefully, never crash."""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workbench.cosmos import transfer


def test_venv_has_torch_false_when_missing(tmp_path: Path) -> None:
    assert transfer._venv_has_torch(tmp_path / "nope" / "python") is False


def test_venv_has_torch_swallows_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hardened image can make the venv python unreadable (stat -> EACCES)."""

    def _raise(self, *args, **kwargs):  # noqa: ANN001
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "exists", _raise)
    # Must return False rather than propagating the PermissionError.
    assert transfer._venv_has_torch(tmp_path / "python") is False


def test_cosmos_transfer_available_false_on_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(transfer, "cosmos_transfer_repo", lambda: tmp_path)
    inference = tmp_path / "examples" / "inference.py"
    inference.parent.mkdir(parents=True, exist_ok=True)
    inference.write_text("# stub\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    def _raise_perm(_py: Path) -> bool:
        raise PermissionError(13, "Permission denied")

    # If _venv_has_torch itself raised, cosmos_transfer_available would crash;
    # with the fix it returns False and falls through to the uv-availability check.
    monkeypatch.setattr(transfer, "_venv_has_torch", lambda _py: False)
    monkeypatch.setattr(transfer.shutil, "which", lambda _name: None)
    assert transfer.cosmos_transfer_available() is False
