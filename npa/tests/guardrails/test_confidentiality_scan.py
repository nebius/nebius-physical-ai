from __future__ import annotations

import io
import json
from pathlib import Path
import re
import subprocess

import pytest

from npa.guardrails.confidentiality import (
    compile_builtin_nebius_infra,
    compile_denylist,
    load_denylist_pattern,
    main,
    scan_diff_text,
    scan_paths,
    scan_text,
    should_skip_unconfigured_fork_pull_request,
    tracked_text_files,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


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


@pytest.mark.parametrize(
    "text",
    [
        "tenant: " + "e00" + "a" * 16,
        "project: project-" + "u00" + "b" * 16,
        "cluster: mk8scluster-" + "u00" + "c" * 16,
        "node group: mk8snodegroup-" + "u00" + "d" * 16,
        "instance: computeinstance-" + "u00" + "e" * 16,
        "capacity block: capacityblockgroup-" + "u00" + "f" * 16,
        "bucket: storagebucket-" + "u00" + "g" * 16,
        "registry: registry-" + "u00" + "h" * 16,
        "network: network-" + "u00" + "j" * 16,
        "object: s3://campaign-bucket-" + "a1b2c3d4" + "/evidence.json",
        (
            "image: registry.example/validation-registry-"
            + "20260815"
            + "/npa-tool@sha256:"
            + "1" * 64
        ),
        "campaign: checkpoint-evaluation-" + "20260815",
    ],
)
def test_builtin_nebius_infra_guard_detects_synthetic_leak_classes(text: str) -> None:
    hits = scan_text(text, compile_builtin_nebius_infra(), source="synthetic.txt")

    assert [(hit.source, hit.line_number) for hit in hits] == [("synthetic.txt", 1)]


def test_builtin_nebius_infra_guard_allows_placeholders_and_hashes() -> None:
    sha40 = "e02" + "a" * 37
    sha64 = "e07" + "b" * 61
    text = "\n".join(
        [
            "Nebius project and tenant identifiers are configured externally.",
            "project: project-test",
            "cluster: mk8scluster-a",
            "image: <your-registry>/npa-tool:tag",
            "object: s3://example-bucket/runs/example/",
            "object: s3://${NPA_ARTIFACT_BUCKET}/runs/example/",
            f"framework revision: {sha40}",
            f"image digest: sha256:{sha64}",
        ]
    )

    assert scan_text(text, compile_builtin_nebius_infra(), source="safe.txt") == []


def test_builtin_nebius_infra_cli_scans_pr_text_without_echoing_match(
    monkeypatch, capsys
) -> None:
    private_value = "project-" + "u00" + "z" * 16
    monkeypatch.setattr("sys.stdin", io.StringIO(f"project: {private_value}\n"))

    assert main(["--built-in-nebius-infra", "--stdin-source", "pr-body"]) == 1
    captured = capsys.readouterr()
    assert "pr-body:1" in captured.err
    assert private_value not in captured.err


def test_builtin_nebius_infra_diff_scan_ignores_removed_values() -> None:
    private_value = "project-" + "u00" + "q" * 16
    diff = "\n".join(
        [
            "diff --git a/report.md b/report.md",
            "--- a/report.md",
            "+++ b/report.md",
            f"-{private_value}",
            "+exact project identifier retained externally",
        ]
    )

    assert (
        scan_diff_text(
            diff,
            compile_builtin_nebius_infra(),
            source="staged-diff",
        )
        == []
    )

    added = diff + f"\n+{private_value}\n"
    hits = scan_diff_text(
        added,
        compile_builtin_nebius_infra(),
        source="staged-diff",
    )
    assert [(hit.source, hit.line_number) for hit in hits] == [("staged-diff", 6)]


def test_builtin_nebius_infra_guard_is_wired_into_existing_gitleaks_ci() -> None:
    workflow = (REPO_ROOT / ".github/workflows/gitleaks.yml").read_text(
        encoding="utf-8"
    )
    gitleaks = (REPO_ROOT / ".gitleaks.toml").read_text(encoding="utf-8")

    assert "--config .gitleaks.toml" in workflow
    assert "[eu]00" in gitleaks
    assert "capacityblockgroup" in gitleaks
    assert "nebius-private-object-location" in gitleaks
    assert "nebius-private-registry-location" in gitleaks
    assert "nebius-dated-operational-name" in gitleaks


def test_tree_scan_skips_binary_files_but_keeps_text_hits(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    (tmp_path / "public.txt").write_text(
        "contains synthetic-secret\n", encoding="utf-8"
    )
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


def test_denylist_loader_prefers_env_pattern_over_file(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("file-only-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST",
        pattern_file=pattern_file,
        environ={"CUSTOMER_DENYLIST": "env-only-pattern"},
    )

    assert loaded.pattern == "env-only-pattern"
    assert loaded.source == "CUSTOMER_DENYLIST"


def test_denylist_loader_reads_configured_file_env(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("configured-file-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST",
        environ={"CUSTOMER_DENYLIST_FILE": str(pattern_file)},
    )

    assert loaded.pattern == "configured-file-pattern\n"
    assert loaded.source == f"CUSTOMER_DENYLIST_FILE:{pattern_file}"


def test_denylist_loader_reads_explicit_pattern_file(tmp_path) -> None:
    pattern_file = tmp_path / "customer.regex"
    pattern_file.write_text("explicit-file-pattern\n", encoding="utf-8")

    loaded = load_denylist_pattern(
        "CUSTOMER_DENYLIST", pattern_file=pattern_file, environ={}
    )

    assert loaded.pattern == "explicit-file-pattern\n"
    assert loaded.source == f"--pattern-file:{pattern_file}"


def test_denylist_loader_fails_closed_when_source_missing(tmp_path) -> None:
    missing_file = tmp_path / "missing.regex"

    with pytest.raises(ValueError, match="CUSTOMER_DENYLIST is empty"):
        load_denylist_pattern(
            "CUSTOMER_DENYLIST", pattern_file=missing_file, environ={}
        )


def test_confidentiality_scan_cli_fails_closed_when_source_missing(tmp_path) -> None:
    missing_file = tmp_path / "missing.regex"

    assert (
        main(["--repo-root", str(tmp_path), "--pattern-file", str(missing_file)]) == 2
    )


def test_should_skip_unconfigured_fork_pull_request(tmp_path) -> None:
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "repository": {"full_name": "nebius/nebius-physical-ai"},
                "pull_request": {
                    "head": {"repo": {"full_name": "contributor/nebius-physical-ai"}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert should_skip_unconfigured_fork_pull_request(
        "CUSTOMER_DENYLIST",
        environ={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        },
    )


def test_should_not_skip_same_repo_pull_request(tmp_path) -> None:
    event_file = tmp_path / "event.json"
    event_file.write_text(
        json.dumps(
            {
                "repository": {"full_name": "nebius/nebius-physical-ai"},
                "pull_request": {
                    "head": {"repo": {"full_name": "nebius/nebius-physical-ai"}},
                },
            }
        ),
        encoding="utf-8",
    )

    assert not should_skip_unconfigured_fork_pull_request(
        "CUSTOMER_DENYLIST",
        environ={
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_EVENT_PATH": str(event_file),
            "GITHUB_REPOSITORY": "nebius/nebius-physical-ai",
        },
    )
