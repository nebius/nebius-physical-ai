"""Tests for the narrow robomimic Debian confidentiality disposition."""

from __future__ import annotations

import hashlib
import lzma
from pathlib import Path
import re
import subprocess

import pytest

from npa._public_https import PublicDownloadError
from npa.guardrails import confidentiality, robomimic_attribution
from npa.guardrails.robomimic_attribution import REPOSITORY_PATH


REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_LOCK = REPO_ROOT / REPOSITORY_PATH
ATTRIBUTION_TEXT = CANONICAL_LOCK.read_text().splitlines()[248]
ATTRIBUTION_PATTERN = re.escape(ATTRIBUTION_TEXT)


def _source_repo(tmp_path: Path, *, extra_path: str | None = None) -> Path:
    repo = tmp_path / "source"
    lock = repo / REPOSITORY_PATH
    lock.parent.mkdir(parents=True)
    lock.write_bytes(CANONICAL_LOCK.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", REPOSITORY_PATH], cwd=repo, check=True)
    if extra_path:
        foreign = repo / extra_path
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text(ATTRIBUTION_TEXT + "\n", encoding="utf-8")
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
    pattern: str = ATTRIBUTION_PATTERN,
) -> int:
    monkeypatch.setenv("CUSTOMER_DENYLIST", pattern)
    return confidentiality.main(
        [
            "--repo-root",
            str(repo),
            "--tree",
            "--ncore-attribution-proof-directory",
            str(proof),
        ]
    )


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            message,
        ],
        cwd=repo,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def test_exact_lock_hit_is_dispositioned_after_official_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    proof = _proof_directory(tmp_path)
    calls: list[tuple[bytes, Path]] = []

    def verified(lock: bytes, proof_directory: Path) -> dict[str, object]:
        calls.append((lock, proof_directory))
        return {"copyright_sources": [{}, {}]}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", verified)

    assert _run_tree(repo, proof, monkeypatch) == 0
    captured = capsys.readouterr()
    assert f"{REPOSITORY_PATH}:249" in captured.err
    assert "raw=1 dispositioned=1 unresolved=0" in captured.out
    assert ATTRIBUTION_TEXT not in captured.out + captured.err
    assert calls == [(CANONICAL_LOCK.read_bytes(), proof)]


def test_foreign_copy_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    foreign_path = "docs/foreign-license-record.txt"
    repo = _source_repo(tmp_path, extra_path=foreign_path)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert _run_tree(repo, _proof_directory(tmp_path), monkeypatch) == 1
    captured = capsys.readouterr()
    assert f"{foreign_path}:1" in captured.err
    assert "raw=2 dispositioned=1 unresolved=1" in captured.err


def test_changed_lock_and_private_marker_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    marker = "synthetic-private-marker"
    lock = repo / REPOSITORY_PATH
    lock.write_bytes(lock.read_bytes() + marker.encode() + b"\n")
    called = False

    def unexpected(_lock: bytes, _proof: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", unexpected)

    assert (
        _run_tree(
            repo,
            _proof_directory(tmp_path),
            monkeypatch,
            rf"{ATTRIBUTION_PATTERN}|{marker}",
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "raw=2 dispositioned=0 unresolved=2" in captured.err
    assert marker not in captured.out + captured.err
    assert not called


def test_wrong_line_cannot_receive_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _source_repo(tmp_path)
    called = False

    def unexpected(_lock: bytes, _proof: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(confidentiality, "verify_public_license_lock", unexpected)
    hits = [
        confidentiality.ScanHit(
            REPOSITORY_PATH,
            250,
            repository_path=REPOSITORY_PATH,
        )
    ]

    assert confidentiality._robomimic_disposition_indexes(
        hits, repo, _proof_directory(tmp_path)
    ) == set()
    assert not called


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
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert _run_tree(repo, _proof_directory(tmp_path), monkeypatch) == 1
    assert "raw=1 dispositioned=0 unresolved=1" in capsys.readouterr().err


def test_tracked_symlink_alias_prevents_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    alias = repo / "lock-alias"
    alias.symlink_to(REPOSITORY_PATH)
    subprocess.run(["git", "add", "lock-alias"], cwd=repo, check=True)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert _run_tree(repo, _proof_directory(tmp_path), monkeypatch) == 1
    assert "raw=1 dispositioned=0 unresolved=1" in capsys.readouterr().err


def test_diff_disposition_requires_exact_canonical_post_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    lock = repo / REPOSITORY_PATH
    lock.unlink()
    subprocess.run(["git", "rm", "--cached", REPOSITORY_PATH], cwd=repo, check=True)
    base = _commit(repo, "Empty base")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(CANONICAL_LOCK.read_bytes())
    head = _commit(repo, "Canonical lock")
    monkeypatch.setenv("CUSTOMER_DENYLIST", ATTRIBUTION_PATTERN)
    monkeypatch.setattr(
        confidentiality, "verify_public_license_lock", lambda _lock, _proof: {}
    )

    assert (
        confidentiality.main(
            [
                "--repo-root",
                str(repo),
                "--diff-range",
                f"{base}..{head}",
                "--ncore-attribution-proof-directory",
                str(_proof_directory(tmp_path)),
            ]
        )
        == 0
    )
    assert "raw=1 dispositioned=1 unresolved=0" in capsys.readouterr().out

    lines = lock.read_text().splitlines()
    lines[248] += " "
    lock.write_text("\n".join(lines) + "\n")
    changed = _commit(repo, "Changed lock")
    hits = confidentiality.scan_git_diff(
        repo,
        f"{head}..{changed}",
        confidentiality.compile_denylist(ATTRIBUTION_PATTERN),
    )
    assert len(hits) == 1
    assert hits[0].repository_path is None


def _synthetic_source_proof(
    *, version: str = "0.11.7-2"
) -> tuple[bytes, bytes]:
    checksum_lines = "\n".join(
        f" {digest} {size} {name}"
        for digest, size, name in robomimic_attribution._SOURCE_FILES
    )
    source_text = (
        "Package: libbsd\n"
        f"Version: {version}\n"
        "Directory: pool/main/libb/libbsd\n"
        f"Checksums-Sha256:\n{checksum_lines}\n"
    ).encode()
    source_index = lzma.compress(source_text)
    digest = hashlib.sha256(source_index).hexdigest()
    inrelease = (
        "-----BEGIN PGP SIGNED MESSAGE-----\n"
        "Hash: SHA256\n\n"
        "Origin: Debian\n"
        "SHA256:\n"
        f" {digest} {len(source_index)} main/source/Sources.xz\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "synthetic-signature-framing\n"
        "-----END PGP SIGNATURE-----\n"
    ).encode()
    return inrelease, source_index


def test_signed_snapshot_source_binding_accepts_exact_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inrelease, source_index = _synthetic_source_proof()
    monkeypatch.setattr(
        robomimic_attribution,
        "_SOURCES",
        robomimic_attribution._PublicProof(
            "source-index",
            "https://snapshot.debian.org/source-index",
            hashlib.sha256(source_index).hexdigest(),
            len(source_index),
            "snapshot.debian.org",
        ),
    )

    robomimic_attribution._verify_snapshot_source(inrelease, source_index)


def test_signed_snapshot_or_source_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inrelease, source_index = _synthetic_source_proof()
    proof = robomimic_attribution._PublicProof(
        "source-index",
        "https://snapshot.debian.org/source-index",
        hashlib.sha256(source_index).hexdigest(),
        len(source_index),
        "snapshot.debian.org",
    )
    monkeypatch.setattr(robomimic_attribution, "_SOURCES", proof)

    with pytest.raises(ValueError, match="signed source-index pin mismatch"):
        robomimic_attribution._verify_snapshot_source(
            inrelease.replace(proof.sha256.encode(), b"0" * 64), source_index
        )

    changed_inrelease, changed_source = _synthetic_source_proof(version="0.11.7-3")
    monkeypatch.setattr(
        robomimic_attribution,
        "_SOURCES",
        robomimic_attribution._PublicProof(
            "source-index",
            "https://snapshot.debian.org/source-index",
            hashlib.sha256(changed_source).hexdigest(),
            len(changed_source),
            "snapshot.debian.org",
        ),
    )
    with pytest.raises(ValueError, match="source identity"):
        robomimic_attribution._verify_snapshot_source(
            changed_inrelease, changed_source
        )


def test_lock_size_digest_and_private_proof_mode_fail_closed(tmp_path: Path) -> None:
    canonical = CANONICAL_LOCK.read_bytes()
    with pytest.raises(ValueError, match="canonical public bytes"):
        robomimic_attribution.verify_public_license_lock(
            canonical + b"\n", _proof_directory(tmp_path)
        )

    proof = tmp_path / "public-proof"
    proof.mkdir(mode=0o755)
    proof.chmod(0o755)
    with pytest.raises(ValueError, match="caller-owned and private"):
        robomimic_attribution.verify_public_license_lock(canonical, proof)


def test_cached_proof_digest_mismatch_remains_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _source_repo(tmp_path)
    proof = _proof_directory(tmp_path)
    (proof / robomimic_attribution._INRELEASE.name).write_bytes(b"wrong")

    assert _run_tree(repo, proof, monkeypatch) == 1
    captured = capsys.readouterr()
    assert "raw=1 dispositioned=0 unresolved=1" in captured.err
    assert "robomimic public-attribution proof could not be verified" in captured.err


def test_independent_copyright_bytes_must_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = _proof_directory(tmp_path)
    monkeypatch.setattr(
        robomimic_attribution, "_verify_openpgp_signature", lambda *_args: None
    )
    monkeypatch.setattr(
        robomimic_attribution, "_verify_snapshot_source", lambda *_args: None
    )

    def payload(
        _directory: Path, source: robomimic_attribution._PublicProof
    ) -> bytes:
        if source is robomimic_attribution._FTP_MASTER_COPYRIGHT:
            return b"official-a"
        if source is robomimic_attribution._SOURCES_COPYRIGHT:
            return b"official-b"
        return b"snapshot"

    monkeypatch.setattr(robomimic_attribution, "_proof_payload", payload)
    with pytest.raises(ValueError, match="independent official copyright bytes"):
        robomimic_attribution.verify_public_license_lock(
            CANONICAL_LOCK.read_bytes(), proof
        )


def test_openpgp_verifier_absence_or_signature_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = _proof_directory(tmp_path)
    monkeypatch.setattr(robomimic_attribution.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="verification tools are unavailable"):
        robomimic_attribution._verify_openpgp_signature(proof)

    monkeypatch.setattr(
        robomimic_attribution.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        robomimic_attribution, "_proof_payload", lambda *_args: b"key"
    )
    results = iter(
        [
            subprocess.CompletedProcess([], 0, b"", b""),
            subprocess.CompletedProcess([], 0, b"keyring", b""),
            subprocess.CompletedProcess([], 1, b"", b"invalid"),
        ]
    )
    monkeypatch.setattr(
        robomimic_attribution.subprocess, "run", lambda *_args, **_kwargs: next(results)
    )
    with pytest.raises(ValueError, match="signature verification failed"):
        robomimic_attribution._verify_openpgp_signature(proof)


def test_public_transport_failure_is_redacted_and_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "synthetic-sensitive-network-detail"

    def refused(*_args: object, **_kwargs: object) -> None:
        raise PublicDownloadError(marker)

    repo = _source_repo(tmp_path)
    monkeypatch.setattr(robomimic_attribution, "download_public_https", refused)

    assert _run_tree(repo, _proof_directory(tmp_path), monkeypatch) == 1
    captured = capsys.readouterr()
    assert "raw=1 dispositioned=0 unresolved=1" in captured.err
    assert marker not in captured.out + captured.err
