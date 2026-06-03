from __future__ import annotations

import re
import subprocess

import pytest

from npa.guardrails.confidentiality import (
    compile_denylist,
    scan_paths,
    scan_text,
    tracked_text_files,
)


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


def test_confidentiality_matcher_can_ignore_case() -> None:
    denylist = compile_denylist(r"synthetic-secret", ignore_case=True)

    hits = scan_text("contains SYNTHETIC-SECRET here\n", denylist, source="fixture.txt")

    assert [(hit.source, hit.line_number) for hit in hits] == [("fixture.txt", 1)]


def test_tree_scan_skips_binary_files_but_keeps_text_hits(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "public.txt").write_text("contains synthetic-secret\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00synthetic-secret\n")
    subprocess.run(["git", "add", "public.txt", "image.png"], cwd=tmp_path, check=True)

    denylist = compile_denylist(r"synthetic-secret")
    paths = tracked_text_files(tmp_path)
    hits = scan_paths(paths, denylist, repo_root=tmp_path)

    assert {path.name for path in paths} == {"public.txt"}
    assert [(hit.source, hit.line_number) for hit in hits] == [("public.txt", 1)]


def test_confidentiality_matcher_rejects_empty_pattern() -> None:
    with pytest.raises(ValueError, match="empty"):
        compile_denylist("")


def test_confidentiality_matcher_names_empty_pattern_source() -> None:
    with pytest.raises(ValueError, match="INFRA_DENYLIST is empty"):
        compile_denylist("", source="INFRA_DENYLIST")


def test_confidentiality_matcher_surfaces_invalid_regex() -> None:
    with pytest.raises(re.error):
        compile_denylist("[")
