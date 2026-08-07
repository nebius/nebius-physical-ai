"""Shared non-TTY contract used by agent and storage destructive commands."""

from __future__ import annotations

import json

import pytest
import typer

from npa.cli import destructive


class _Input:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.mark.parametrize(
    "prompt",
    ["Destroy agent prod/a?", "Delete bucket b?", "Delete service account sa?"],
)
def test_non_tty_requires_yes_for_every_destructive_sibling(
    monkeypatch: pytest.MonkeyPatch, prompt: str, capsys
) -> None:
    monkeypatch.setattr(destructive.sys, "stdin", _Input(False))

    with pytest.raises(typer.Exit) as excinfo:
        destructive.require_destructive_confirmation(yes=False, prompt=prompt)

    assert excinfo.value.exit_code == 1
    assert "Re-run with --yes" in capsys.readouterr().out


def test_yes_bypasses_tty_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(destructive.sys, "stdin", _Input(False))
    monkeypatch.setattr(
        destructive.typer,
        "confirm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompted")),
    )

    destructive.require_destructive_confirmation(yes=True, prompt="Delete?")


def test_tty_confirmation_allows_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(destructive.sys, "stdin", _Input(True))
    monkeypatch.setattr(destructive.typer, "confirm", lambda *args, **kwargs: True)

    destructive.require_destructive_confirmation(yes=False, prompt="Delete?")


def test_tty_cancellation_has_stable_exit_and_no_mutation(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(destructive.sys, "stdin", _Input(True))
    monkeypatch.setattr(destructive.typer, "confirm", lambda *args, **kwargs: False)

    with pytest.raises(typer.Exit) as excinfo:
        destructive.require_destructive_confirmation(yes=False, prompt="Delete?")

    assert excinfo.value.exit_code == 1
    assert "No destructive action was attempted" in capsys.readouterr().out


def test_non_tty_json_refusal_is_one_machine_document(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(destructive.sys, "stdin", _Input(False))

    with pytest.raises(typer.Exit):
        destructive.require_destructive_confirmation(
            yes=False,
            prompt="Delete?",
            output_json=True,
            payload={"identity_source": "receipt"},
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "identity_source": "receipt",
        "message": "Delete? Re-run with --yes; no TTY is available to confirm.",
        "mutated": False,
        "result": "confirmation_required",
    }
