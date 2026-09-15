"""Strict single-document JSON parsing for CLI output with presentation noise."""

from __future__ import annotations

from npa.clients.json_output import parse_single_json_document


def test_parses_document_with_diagnostic_preamble() -> None:
    assert parse_single_json_document('note\n[{"a": 1}]') == [{"a": 1}]


def test_tolerates_trailing_ansi_spinner_frame() -> None:
    # SkyPilot's rich status spinner can flush one final frame to stdout after
    # the JSON payload; ANSI erase/cursor sequences contain ESC+"[" which must
    # not be mistaken for a second JSON document.
    output = (
        '[{"job_id": 5, "status": "RUNNING"}]\n'
        "\x1b[2K\x1b[32m⠏\x1b[0m \x1b[36mChecking managed jobs\x1b[0m\n"
        "\x1b[?25h\x1b[1A\x1b[2K"
    )
    assert parse_single_json_document(output) == [{"job_id": 5, "status": "RUNNING"}]


def test_rejects_second_document_after_payload() -> None:
    assert parse_single_json_document('{"a": 1}\n{"b": 2}') is None


def test_rejects_ambiguous_preamble_value() -> None:
    assert parse_single_json_document('junk [1, 2] trailing {') is None


def test_rejects_trailing_json_start_outside_ansi() -> None:
    assert parse_single_json_document('[{"a": 1}]\n[') is None
