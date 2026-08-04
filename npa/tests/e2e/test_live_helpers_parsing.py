"""Offline coverage for the live harness's CLI output parsing.

These tests run in the normal suite (no ``NPA_INTEGRATION_E2E``) because the parsing
bug they pin cost a real live run: a successful ``sonic-export`` submit (SkyPilot job
189) was reported as a test failure because ``json.loads(result.output)`` choked on the
advisory line the CLI writes before its JSON ("Hint: consider --secret-env ...").
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HELPERS_PATH = Path(__file__).resolve().parent / "npa_workflow_live_helpers.py"


def _helpers():
    spec = importlib.util.spec_from_file_location("npa_workflow_live_helpers", HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parses_a_bare_json_document() -> None:
    parse = _helpers()._json_document_from_stream

    assert parse('{"status": "SUBMITTED"}') == {"status": "SUBMITTED"}


def test_parses_json_after_an_advisory_stderr_line() -> None:
    parse = _helpers()._json_document_from_stream
    stream = 'Hint: consider --secret-env NGC_API_KEY\n{\n  "job_id": "189"\n}\n'

    assert parse(stream) == {"job_id": "189"}


def test_parses_json_after_runtime_progress_lines() -> None:
    parse = _helpers()._json_document_from_stream
    stream = (
        "[runtime] wave 001|sweep|...: submitted job_id=185\n"
        "[runtime] wave 001: 2 tasks running concurrently\n"
        '{"status": "succeeded", "waves": []}\n'
    )

    assert parse(stream)["status"] == "succeeded"


def test_parses_a_json_array_document() -> None:
    parse = _helpers()._json_document_from_stream

    assert parse("noise\n[1, 2, 3]\n") == [1, 2, 3]


def test_reports_the_stream_when_there_is_no_json() -> None:
    parse = _helpers()._json_document_from_stream

    with pytest.raises(AssertionError, match="no JSON document in CLI output"):
        parse("Error: something went wrong and nothing was printed\n")


def test_empty_stream_is_an_assertion_not_a_json_error() -> None:
    parse = _helpers()._json_document_from_stream

    with pytest.raises(AssertionError):
        parse("")
