"""Tests for byte-exact NCore public notice provenance verification."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from npa.guardrails import ncore_attribution as attribution


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTICE = REPO_ROOT / attribution.REPOSITORY_PATH


def _tar_bytes(member_name: str, payload: bytes, *, copies: int = 1) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for _ in range(copies):
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def test_canonical_notice_constants_match_tracked_complete_bytes() -> None:
    notice = NOTICE.read_bytes()

    assert len(notice) == attribution.NOTICE_SIZE == 54_130
    assert hashlib.sha256(notice).hexdigest() == attribution.NOTICE_SHA256
    assert attribution.ATTRIBUTION_LINES == frozenset({633, 640})
    assert attribution.IMAGE_PATH == (
        "usr/share/doc/npa-ncore/cpython/LICENSE.third-party"
    )


def test_official_archive_constants_match_ncore_source_lock() -> None:
    lock = json.loads(
        (REPO_ROOT / "npa/docker/workbench/ncore/base-source-lock.json").read_text()
    )["cpython_provenance"]

    assert lock["source_archive"] == {
        "url": attribution.SOURCE_ARCHIVE_URL,
        "sha256": attribution.SOURCE_ARCHIVE_SHA256,
        "size": attribution.SOURCE_ARCHIVE_SIZE,
    }
    assert lock["commit_archive"] == {
        "url": attribution.COMMIT_ARCHIVE_URL,
        "sha256": attribution.COMMIT_ARCHIVE_SHA256,
        "size": attribution.COMMIT_ARCHIVE_SIZE,
    }


def test_verify_public_notice_requires_exact_complete_bytes(tmp_path: Path) -> None:
    notice = NOTICE.read_bytes()

    for changed in (notice[:-1], notice + b"private-marker\n", b"x" + notice[1:]):
        with pytest.raises(ValueError, match="canonical public bytes"):
            attribution.verify_public_notice(changed, tmp_path)


def test_verify_public_notice_checks_both_complete_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notice = NOTICE.read_bytes()
    observed: list[str] = []

    def archive_payload(proof_directory: Path, archive: attribution._Archive) -> bytes:
        assert proof_directory == tmp_path
        observed.append(archive.url)
        return _tar_bytes(archive.member, notice)

    monkeypatch.setattr(attribution, "_archive_payload", archive_payload)

    proof = attribution.verify_public_notice(notice, tmp_path)

    assert observed == [
        attribution.SOURCE_ARCHIVE_URL,
        attribution.COMMIT_ARCHIVE_URL,
    ]
    assert proof["notice"] == {
        "sha256": attribution.NOTICE_SHA256,
        "size": attribution.NOTICE_SIZE,
    }
    assert all(
        set(archive)
        == {"url", "sha256", "size", "member", "member_sha256", "member_size"}
        for archive in proof["archives"]
    )


def test_verify_public_notice_rejects_changed_archive_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notice = NOTICE.read_bytes()

    def archive_payload(_proof_directory: Path, archive: attribution._Archive) -> bytes:
        member = (
            notice
            if archive.url == attribution.SOURCE_ARCHIVE_URL
            else b"x" + notice[1:]
        )
        return _tar_bytes(archive.member, member)

    monkeypatch.setattr(attribution, "_archive_payload", archive_payload)

    with pytest.raises(ValueError, match="differs from the notice"):
        attribution.verify_public_notice(notice, tmp_path)


def test_notice_member_rejects_missing_duplicate_and_non_regular_members() -> None:
    payload = b"public notice"
    archive = attribution._Archive(
        "https://example.test", "0" * 64, 1, "notice", "x", "example.test"
    )

    with pytest.raises(ValueError, match="unique regular"):
        attribution._notice_member(_tar_bytes("other", payload), archive)
    with pytest.raises(ValueError, match="unique regular"):
        attribution._notice_member(_tar_bytes("notice", payload, copies=2), archive)

    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        member = tarfile.TarInfo("notice")
        member.type = tarfile.SYMTYPE
        member.linkname = "other"
        bundle.addfile(member)
    with pytest.raises(ValueError, match="unique regular"):
        attribution._notice_member(output.getvalue(), archive)
    with pytest.raises(tarfile.ReadError):
        attribution._notice_member(b"not an archive", archive)


def test_cached_archive_must_be_exact_regular_bytes(tmp_path: Path) -> None:
    payload = b"archive bytes"
    archive = attribution._Archive(
        "https://example.test/archive",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        "notice",
        "archive.tar.gz",
        "example.test",
    )
    path = tmp_path / archive.cache_name
    for forged in (payload + b"appended", b"x" + payload[1:]):
        path.write_bytes(forged)
        with pytest.raises(ValueError, match="public pin"):
            attribution._verified_file(path, archive)

    path.unlink()
    target = tmp_path / "target"
    target.write_bytes(payload)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        attribution._verified_file(path, archive)

    path.unlink()
    os.chmod(target, 0o600)
    os.link(target, path)
    with pytest.raises(ValueError, match="caller-owned and private"):
        attribution._verified_file(path, archive)


def test_proof_directory_must_be_private_and_caller_owned(tmp_path: Path) -> None:
    original_mode = tmp_path.stat().st_mode & 0o777
    os.chmod(tmp_path, 0o755)
    try:
        with pytest.raises(ValueError, match="caller-owned and private"):
            attribution.verify_public_notice(NOTICE.read_bytes(), tmp_path)
    finally:
        os.chmod(tmp_path, original_mode)


def test_missing_archives_fail_closed_without_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic network refusal")

    monkeypatch.setattr(attribution, "download_public_https", unavailable)

    with pytest.raises(OSError, match="synthetic network refusal"):
        attribution.verify_public_notice(NOTICE.read_bytes(), tmp_path)

def test_archive_output_rejects_bytes_beyond_the_pinned_size() -> None:
    output = io.BytesIO()
    bounded = attribution._PinnedArchiveOutput(output, 3)
    assert bounded.write(b"abc") == 3
    with pytest.raises(ValueError, match="public size pin"):
        bounded.write(b"private-marker")
    assert output.getvalue() == b"abc"
