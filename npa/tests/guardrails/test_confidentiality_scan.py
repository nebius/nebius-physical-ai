from __future__ import annotations

import re

import pytest

from npa.guardrails.confidentiality import compile_denylist, scan_text


def test_confidentiality_matcher_reports_redacted_locations_only() -> None:
    denylist = compile_denylist(r"synthetic-secret|project-codename")

    hits = scan_text(
        "public line\ncontains synthetic-secret here\ncontains project-codename here\n",
        denylist,
        source="fixture.txt",
    )

    assert [(hit.source, hit.line_number) for hit in hits] == [
        ("fixture.txt", 2),
        ("fixture.txt", 3),
    ]
    assert not hasattr(hits[0], "line")


def test_confidentiality_matcher_rejects_empty_pattern() -> None:
    with pytest.raises(ValueError, match="empty"):
        compile_denylist("")


def test_confidentiality_matcher_surfaces_invalid_regex() -> None:
    with pytest.raises(re.error):
        compile_denylist("[")
