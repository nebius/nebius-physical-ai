"""Tests for destructive deploy confirmation helpers."""

from __future__ import annotations

import pytest
import typer

from npa.deploy.confirm import confirm_or_exit, confirm_vm_destroy


def test_confirm_or_exit_aborts_when_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: False)
    with pytest.raises(typer.Exit) as exc:
        confirm_or_exit("Destroy?")
    assert exc.value.exit_code == 1


def test_confirm_or_exit_continues_when_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(typer, "confirm", lambda *args, **kwargs: True)
    confirm_or_exit("Destroy?")


@pytest.mark.parametrize(
    ("byovm", "dry_run", "yes", "should_prompt"),
    [
        (False, False, False, True),
        (False, False, True, False),
        (False, True, False, False),
        (True, False, False, False),
    ],
)
def test_confirm_vm_destroy_prompt_gates(
    monkeypatch: pytest.MonkeyPatch,
    byovm: bool,
    dry_run: bool,
    yes: bool,
    should_prompt: bool,
) -> None:
    called: list[str] = []

    def fake_confirm(prompt: str, default: bool = False) -> bool:
        called.append(prompt)
        return False

    monkeypatch.setattr(typer, "confirm", fake_confirm)
    if should_prompt:
        with pytest.raises(typer.Exit):
            confirm_vm_destroy(
                "proj",
                "alias",
                byovm=byovm,
                dry_run=dry_run,
                yes=yes,
            )
        assert called
        assert "proj/alias" in called[0]
    else:
        confirm_vm_destroy(
            "proj",
            "alias",
            byovm=byovm,
            dry_run=dry_run,
            yes=yes,
        )
        assert called == []
