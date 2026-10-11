"""Failure-output contract for Cosmos3 DROID workflow stage commands."""

from __future__ import annotations

import json

import pytest
import typer

from npa.cli.workbench.cosmos3_droid_forward_dynamics import _invoke
from npa.workbench.cosmos.droid_forward_dynamics import DroidForwardDynamicsError


def test_droid_contract_failure_exposes_safe_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_kwargs: object) -> dict[str, object]:
        raise DroidForwardDynamicsError(
            "selection frames must contain exactly 17 observations"
        )

    with pytest.raises(typer.Exit) as raised:
        _invoke(fail)

    assert raised.value.exit_code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error_message": "selection frames must contain exactly 17 observations",
        "error_type": "DroidForwardDynamicsError",
        "status": "failed",
    }


def test_droid_untyped_failure_does_not_expose_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("private endpoint and credential must not be printed")

    with pytest.raises(typer.Exit):
        _invoke(fail)

    result = json.loads(capsys.readouterr().out)
    assert result == {"error_type": "RuntimeError", "status": "failed"}
    assert "private endpoint" not in json.dumps(result)
