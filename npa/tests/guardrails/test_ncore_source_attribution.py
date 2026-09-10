"""Tests for the narrow NCore source confidentiality disposition."""

from __future__ import annotations

import io
import re
from pathlib import Path
import subprocess

import pytest

from npa.guardrails import confidentiality
from npa.guardrails.ncore_attribution import REPOSITORY_PATH


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_NOTICE = REPO_ROOT / REPOSITORY_PATH
ATTRIBUTION_TEXT = CANONICAL_NOTICE.read_text().splitlines()[632]
ATTRIBUTION_PATTERN = "|".join(
    re.escape(CANONICAL_NOTICE.read_text().splitlines()[line - 1])
    for line in (633, 640)
)


def _source_repo(tmp_path: Path, *, extra_path: str | None = None) -> Path:
    repo = tmp_path / "source"
    notice = repo / REPOSITORY_PATH
    notice.parent.mkdir(parents=True)
    notice.write_bytes(CANONICAL_NOTICE.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", REPOSITORY_PATH], cwd=repo, check=True)
    if extra_path:
        foreign = repo / extra_path
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_bytes(CANONICAL_NOTICE.read_bytes())
        subprocess.run(["git", "add", extra_path], cwd=repo, check=True)
    return repo


def _proof_directory(tmp_path: Path) -> Path:
    proof = tmp_path / "proof"
    proof.mkdir(mode=0o700)
    return proof


def _run_tree(
    repo: Path,
    proof: Path,
    monkeypatch: pytest.MonkeyPatch,
    pattern: str,
    *,
    pattern_env: str = "CUSTOMER_DENYLIST",
) -> int:
    monkeypatch.setenv(pattern_env, pattern)
    return confidentiality.main(
        [
            "--repo-root",
            str(repo),
            "--tree",
            "--pattern-env",
            pattern_env,
            "--ncore-attribution-proof-directory",
            str(proof),
        ]
    )


def test_exact_customer_attribution_hits_are_dispositioned_after_dual_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    proof = _proof_directory(tmp_path)
    calls: list[tuple[bytes, Path]] = []

    def verified(notice: bytes, proof_directory: Path) -> dict[str, object]:
        calls.append((notice, proof_directory))
        return {"archives": [{}, {}]}

    monkeypatch.setattr(confidentiality, "verify_public_notice", verified)

    assert _run_tree(repo, proof, monkeypatch, ATTRIBUTION_PATTERN) == 0
    captured = capsys.readouterr()
    assert f"{REPOSITORY_PATH}:633" in captured.err
    assert f"{REPOSITORY_PATH}:640" in captured.err
    assert "raw=2 dispositioned=2 unresolved=0" in captured.out
    assert ATTRIBUTION_TEXT not in captured.out + captured.err
    assert calls == [(CANONICAL_NOTICE.read_bytes(), proof)]


def test_changed_notice_and_appended_private_marker_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    private_marker = "synthetic-private-marker"
    notice = repo / REPOSITORY_PATH
    notice.write_bytes(notice.read_bytes() + private_marker.encode() + b"\n")

    assert (
        _run_tree(
            repo,
            _proof_directory(tmp_path),
            monkeypatch,
            rf"{ATTRIBUTION_PATTERN}|{private_marker}",
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "raw=3 dispositioned=0 unresolved=3" in captured.err
    assert private_marker not in captured.out + captured.err


def test_foreign_copy_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    foreign_path = "docs/foreign-notice.txt"
    repo = _source_repo(tmp_path, extra_path=foreign_path)
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert (
        _run_tree(repo, _proof_directory(tmp_path), monkeypatch, ATTRIBUTION_PATTERN) == 1
    )
    captured = capsys.readouterr()
    assert f"{foreign_path}:633" in captured.err
    assert "raw=4 dispositioned=0 unresolved=4" in captured.err


def test_tracked_symlink_alias_prevents_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    alias = repo / "notice-alias"
    alias.symlink_to(REPOSITORY_PATH)
    subprocess.run(["git", "add", "notice-alias"], cwd=repo, check=True)
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert (
        _run_tree(repo, _proof_directory(tmp_path), monkeypatch, ATTRIBUTION_PATTERN) == 1
    )
    assert "dispositioned=0" in capsys.readouterr().err


def test_private_marker_elsewhere_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    private_marker = "synthetic-private-marker"
    foreign_path = repo / "docs/private.txt"
    foreign_path.parent.mkdir()
    foreign_path.write_text(private_marker)
    subprocess.run(["git", "add", "docs/private.txt"], cwd=repo, check=True)
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert (
        _run_tree(
            repo,
            _proof_directory(tmp_path),
            monkeypatch,
            rf"{ATTRIBUTION_PATTERN}|{private_marker}",
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "docs/private.txt:1" in captured.err
    assert "raw=3 dispositioned=2 unresolved=1" in captured.err
    assert private_marker not in captured.out + captured.err


def test_other_notice_line_and_non_customer_policy_remain_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    proof = _proof_directory(tmp_path)
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert _run_tree(repo, proof, monkeypatch, "Lucent Technologies") == 1
    assert "raw=1 dispositioned=0 unresolved=1" in capsys.readouterr().err

    assert (
        _run_tree(
            repo,
            proof,
            monkeypatch,
            ATTRIBUTION_PATTERN,
            pattern_env="INFRA_DENYLIST",
        )
        == 1
    )
    assert "raw=2 dispositioned=0 unresolved=2" in capsys.readouterr().err


def test_non_regular_git_mode_cannot_receive_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    subprocess.run(
        ["git", "update-index", "--chmod=+x", REPOSITORY_PATH],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert (
        _run_tree(repo, _proof_directory(tmp_path), monkeypatch, ATTRIBUTION_PATTERN) == 1
    )
    assert "raw=2 dispositioned=0 unresolved=2" in capsys.readouterr().err


def test_git_diff_hits_use_post_image_path_and_line() -> None:
    diff = "\n".join(
        [
            f"diff --git a/{REPOSITORY_PATH} b/{REPOSITORY_PATH}",
            f"--- a/{REPOSITORY_PATH}",
            f"+++ b/{REPOSITORY_PATH}",
            "@@ -632,0 +633,2 @@",
            f"+{ATTRIBUTION_TEXT}",
            "+public continuation",
        ]
    )

    hits = confidentiality._scan_git_diff_text(
        diff,
        confidentiality.compile_denylist(ATTRIBUTION_PATTERN),
        diff_range="base..HEAD",
    )

    assert len(hits) == 1
    assert hits[0].repository_path == REPOSITORY_PATH
    assert hits[0].line_number == 633


def test_stdin_diff_cannot_claim_repository_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    proof = _proof_directory(tmp_path)
    monkeypatch.setenv("CUSTOMER_DENYLIST", ATTRIBUTION_PATTERN)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"@@ -0,0 +633 @@\n+{ATTRIBUTION_TEXT}\n"),
    )
    monkeypatch.setattr(
        confidentiality, "verify_public_notice", lambda _notice, _proof: {}
    )

    assert (
        confidentiality.main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--stdin-source",
                "untrusted-patch",
                "--stdin-is-diff",
                "--ncore-attribution-proof-directory",
                str(proof),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "untrusted-patch:2" in captured.err
    assert "raw=1 dispositioned=0 unresolved=1" in captured.err


def test_added_content_cannot_spoof_a_canonical_diff_header() -> None:
    diff = "\n".join([
        "diff --git a/foreign.txt b/foreign.txt", "--- a/foreign.txt", "+++ b/foreign.txt",
        "@@ -0,0 +632,2 @@", f"+++ b/{REPOSITORY_PATH}", "+synthetic-private-marker",
    ])
    hits = confidentiality._scan_git_diff_text(
        diff, re.compile("synthetic-private-marker"), diff_range="base..HEAD")
    assert len(hits) == 1
    assert hits[0].repository_path == "foreign.txt"
    assert hits[0].line_number == 633


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", message], cwd=repo, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def test_diff_disposition_requires_the_actual_canonical_post_image(tmp_path: Path) -> None:
    repo = _source_repo(tmp_path)
    canonical = _commit(repo, "Canonical notice")
    notice = repo / REPOSITORY_PATH
    lines = notice.read_text().splitlines()
    lines[632] = "synthetic-private-marker"
    notice.write_text("\n".join(lines) + "\n")
    changed = _commit(repo, "Changed notice")
    notice.write_bytes(CANONICAL_NOTICE.read_bytes())
    restored = _commit(repo, "Restore canonical notice")
    hits = confidentiality.scan_git_diff(repo, f"{canonical}..{changed}",
                                          re.compile("synthetic-private-marker"))
    assert len(hits) == 1 and hits[0].repository_path is None
    assert not confidentiality._canonical_diff_matches(repo, f"{canonical}..{changed}")
    assert confidentiality._canonical_diff_matches(repo, f"{changed}..{restored}")


def test_public_transport_failure_retains_all_unresolved_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from npa.guardrails import ncore_attribution
    from npa._public_https import PublicDownloadError

    def refused(*_args: object, **_kwargs: object) -> None:
        raise PublicDownloadError("synthetic-sensitive-network-detail")

    repo = _source_repo(tmp_path)
    monkeypatch.setattr(ncore_attribution, "download_public_https", refused)
    assert _run_tree(repo, _proof_directory(tmp_path), monkeypatch, ATTRIBUTION_PATTERN) == 1
    output = capsys.readouterr()
    assert "raw=2 dispositioned=0 unresolved=2" in output.err
    assert "NCore public-attribution proof could not be verified" in output.err
    assert "synthetic-sensitive-network-detail" not in output.out + output.err
