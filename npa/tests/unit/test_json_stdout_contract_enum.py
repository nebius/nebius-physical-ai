"""``json_stdout_contract`` must recognise enum-typed ``output_format`` options.

Typer hands enum-typed options over as enum members. A ``(str, Enum)`` member
stringifies to ``OutputFormat.json`` rather than ``json``, so the contract
has to read the member's value before comparing.
"""

from __future__ import annotations

import json
from enum import Enum

import pytest

from npa.lifecycle_intent import json_stdout_contract


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@json_stdout_contract
def _command(output_format: OutputFormat = OutputFormat.text) -> int:
    print("diagnostic line that must not leak into JSON stdout")
    print(json.dumps({"result": "completed", "format": output_format.value}))
    return 0


def test_enum_member_stringifies_to_its_qualified_name() -> None:
    # The premise of the fix: without reading ``.value`` this never matched "json".
    assert str(OutputFormat.json) != "json"


def test_enum_json_member_activates_the_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert _command(output_format=OutputFormat.json) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"result": "completed", "format": "json"}
    assert "diagnostic line" not in captured.out
    assert "diagnostic line" in captured.err or "separated from JSON stdout" in captured.err


def test_enum_text_member_leaves_stdout_untouched(capsys: pytest.CaptureFixture[str]) -> None:
    assert _command(output_format=OutputFormat.text) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "diagnostic line that must not leak into JSON stdout"


def test_plain_string_json_still_activates_the_contract(capsys: pytest.CaptureFixture[str]) -> None:
    @json_stdout_contract
    def command(output_format: str = "text") -> None:
        print(json.dumps({"ok": True}))

    command(output_format="JSON")
    assert json.loads(capsys.readouterr().out) == {"ok": True}
