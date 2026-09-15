from __future__ import annotations

import hashlib
import io
import json
import socket
import tarfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import zstandard

import npa.deploy.corresponding_source as SOURCE
from npa.deploy.corresponding_source import (
    CorrespondingSourceError,
)

ROOT = Path(__file__).resolve().parents[3]
REAL_LOCK = (
    ROOT / "npa/docker/workbench/gymnasium-robotics/corresponding-source.lock.json"
)
SOURCE_REVISION = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64
PLATFORM_DIGEST = "sha256:" + "c" * 64
CONFIG_DIGEST = "sha256:" + "d" * 64
PUBLIC_ORIGIN = "https://downloads.example.test"
PUBLIC_ADDRESS = "93.184.216.34"
SOURCE_FILES = {
    "pool/main/f/fixture/fixture_1.dsc": b"Format: 3.0 (quilt)\n",
    "pool/main/f/fixture/fixture_1.orig.tar.xz": b"fixture source archive\n",
}


def _source_manifest(files: dict[str, bytes] = SOURCE_FILES) -> bytes:
    payload = {
        "schema": "npa.gymnasium-robotics.corresponding-source-manifest.v1",
        "files": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in files.items()
        ],
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _compressed_tar(members: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for member, content in members:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content) if content else None)
    return zstandard.ZstdCompressor(level=3).compress(raw.getvalue())


def _archive(
    files: dict[str, bytes] = SOURCE_FILES,
    *,
    manifest: bytes | None = None,
    extra_members: list[tuple[tarfile.TarInfo, bytes]] | None = None,
    include_manifest: bool = True,
) -> bytes:
    members: list[tuple[tarfile.TarInfo, bytes]] = []
    if include_manifest:
        members.append(
            (
                tarfile.TarInfo("source-manifest.json"),
                manifest or _source_manifest(files),
            )
        )
    members.extend((tarfile.TarInfo(name), content) for name, content in files.items())
    members.extend(extra_members or [])
    return _compressed_tar(members)


SOURCE_MANIFEST = _source_manifest()
ARCHIVE = _archive(manifest=SOURCE_MANIFEST)


class _Response(io.BytesIO):
    def __init__(self, content: bytes, url: str, status: int = 200) -> None:
        super().__init__(content)
        self._url = url
        self.status = status

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _accepted_lock(manifest: bytes = SOURCE_MANIFEST) -> dict[str, Any]:
    payload = json.loads(REAL_LOCK.read_text(encoding="utf-8"))
    payload["public_corresponding_source_delivery"] = "accepted-public-immutable"
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    delivery = payload["deliveries"][0]
    delivery["source_manifest_sha256"] = manifest_sha256
    for artifact in delivery["artifacts"]:
        if artifact["role"] == "source_archives":
            artifact["manifest_sha256"] = manifest_sha256
    return payload


def _record(
    lock_bytes: bytes,
    lock: dict[str, Any],
    archive: bytes = ARCHIVE,
) -> dict[str, Any]:
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    delivery = lock["deliveries"][0]
    return {
        "format": "npa_gymnasium_robotics_accepted_image_manifest_v1",
        "status": "accepted-for-publication",
        "tool": "gymnasium-robotics",
        "source_revision": SOURCE_REVISION,
        "image": {
            "digest": IMAGE_DIGEST,
            "platform": {"os": "linux", "architecture": "amd64"},
            "platform_manifest_digest": PLATFORM_DIGEST,
            "config_digest": CONFIG_DIGEST,
        },
        "corresponding_source": {
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "delivery": delivery,
            "artifact": {
                "reference": f"https://downloads.example.test/{archive_sha256}.tar.zst",
                "sha256": archive_sha256,
                "size_bytes": len(archive),
                "media_type": "application/zstd",
                "anonymous": True,
                "immutable": True,
                "contents_manifest_sha256": delivery["source_manifest_sha256"],
            },
        },
    }


def _fixture(
    tmp_path: Path,
    *,
    archive: bytes = ARCHIVE,
    manifest: bytes = SOURCE_MANIFEST,
) -> tuple[Path, Path, dict[str, Any]]:
    lock = _accepted_lock(manifest)
    lock_bytes = json.dumps(lock, sort_keys=True).encode()
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(lock_bytes)
    record = _record(lock_bytes, lock, archive)
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, lock_path, record


def _opener(expected_url: str, content: bytes = ARCHIVE, status: int = 200):
    def open_response(request: Any, *, timeout: float) -> _Response:
        assert request.full_url == expected_url
        assert request.get_header("Authorization") is None
        assert timeout == 60
        return _Response(content, expected_url, status)

    return open_response


def _resolver(
    hostname: str,
    port: int,
    *,
    type: socket.SocketKind,
) -> list[tuple[Any, ...]]:
    assert hostname == "downloads.example.test"
    assert port == 443
    assert type == socket.SOCK_STREAM
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_ADDRESS, 443))]


def _verify(
    record_path: Path,
    lock_path: Path,
    record: dict[str, Any],
    *,
    content: bytes = ARCHIVE,
    resolver: Any = _resolver,
    reviewed_origins: frozenset[str] = frozenset({PUBLIC_ORIGIN}),
    response_url: str | None = None,
    status: int = 200,
) -> None:
    reference = record["corresponding_source"]["artifact"]["reference"]
    opener = _opener(reference, content, status)
    if response_url is not None:

        def redirected_opener(request: Any, *, timeout: float) -> _Response:
            assert request.full_url == reference
            assert timeout == 60
            return _Response(content, response_url)

        opener = redirected_opener
    with (
        mock.patch.object(
            SOURCE,
            "REVIEWED_PUBLIC_DELIVERY_ORIGINS",
            reviewed_origins,
        ),
        mock.patch.object(SOURCE.socket, "getaddrinfo", resolver),
    ):
        SOURCE.verify_corresponding_source_delivery(
            record_path,
            lock_path,
            source_revision=SOURCE_REVISION,
            image_digest=IMAGE_DIGEST,
            platform_manifest_digest=PLATFORM_DIGEST,
            config_digest=CONFIG_DIGEST,
            opener=opener,
        )


def test_exact_anonymous_delivery_binding_passes(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    _verify(record_path, lock_path, record)


def test_missing_accepted_record_refuses(tmp_path: Path) -> None:
    _, lock_path, record = _fixture(tmp_path)
    with pytest.raises(CorrespondingSourceError, match="cannot open accepted"):
        _verify(tmp_path / "missing.json", lock_path, record)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("tool",), "other", "wrong subject"),
        (("source_revision",), "e" * 40, "source revision"),
        (("image", "digest"), "sha256:" + "e" * 64, "image digest"),
        (
            ("image", "platform_manifest_digest"),
            "sha256:" + "e" * 64,
            "platform_manifest",
        ),
        (("image", "config_digest"), "sha256:" + "e" * 64, "config_digest"),
        (
            ("image", "platform"),
            {"os": "linux", "architecture": "arm64"},
            "linux/amd64",
        ),
    ],
)
def test_wrong_image_subject_binding_refuses(
    tmp_path: Path, path: tuple[str, ...], value: Any, message: str
) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match=message):
        _verify(record_path, lock_path, record)


def test_wrong_lock_and_coverage_refuse(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    record["corresponding_source"]["lock_sha256"] = "f" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match="lock digest"):
        _verify(record_path, lock_path, record)

    record_path, lock_path, record = _fixture(tmp_path)
    record["corresponding_source"]["delivery"]["source_manifest_sha256"] = "f" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match="delivery does not match"):
        _verify(record_path, lock_path, record)


def test_mutable_or_non_anonymous_reference_refuses(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    artifact = record["corresponding_source"]["artifact"]
    artifact["reference"] = "https://downloads.example.test/latest.tar.zst"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match="digest-addressed"):
        _verify(record_path, lock_path, record)

    record_path, lock_path, record = _fixture(tmp_path)
    record["corresponding_source"]["artifact"]["anonymous"] = False
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match="not anonymous"):
        _verify(record_path, lock_path, record)


def test_malformed_record_and_unaccepted_lock_refuse(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    record_path.write_text(
        '{"tool":"gymnasium-robotics","tool":"other"}', encoding="utf-8"
    )
    with pytest.raises(CorrespondingSourceError, match="duplicate JSON field"):
        _verify(record_path, lock_path, record)

    current_lock = tmp_path / "withheld-lock.json"
    current_lock.write_bytes(REAL_LOCK.read_bytes())
    current_bytes = current_lock.read_bytes()
    record = _record(current_bytes, json.loads(current_bytes))
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CorrespondingSourceError, match="delivery is not accepted"):
        _verify(record_path, current_lock, record)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (ARCHIVE + b"extra", "exceeds accepted size"),
        (ARCHIVE[:-1], "size does not match"),
        (b"x" * len(ARCHIVE), "digest does not match"),
    ],
)
def test_anonymous_readback_bytes_must_match(
    tmp_path: Path, content: bytes, message: str
) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    with pytest.raises(CorrespondingSourceError, match=message):
        _verify(
            record_path,
            lock_path,
            record,
            content=content,
        )


def test_anonymous_readback_denial_refuses(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    with pytest.raises(
        CorrespondingSourceError, match="anonymous source retrieval failed"
    ):
        _verify(
            record_path,
            lock_path,
            record,
            status=401,
        )


def test_unreviewed_delivery_origin_refuses_before_retrieval(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    with pytest.raises(CorrespondingSourceError, match="not in the reviewed"):
        _verify(
            record_path,
            lock_path,
            record,
            reviewed_origins=frozenset(),
        )


@pytest.mark.parametrize(
    "address",
    (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "240.0.0.1",
        "224.0.0.1",
        "0.0.0.0",
        "::1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
        "::",
    ),
)
def test_resolved_non_global_destinations_refuse(
    tmp_path: Path,
    address: str,
) -> None:
    record_path, lock_path, record = _fixture(tmp_path)

    def resolver(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (address, 443))]

    with pytest.raises(CorrespondingSourceError, match="not globally routable"):
        _verify(record_path, lock_path, record, resolver=resolver)


def test_literal_private_destination_refuses_without_resolution(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    artifact = record["corresponding_source"]["artifact"]
    artifact["reference"] = artifact["reference"].replace(
        "downloads.example.test", "127.0.0.1"
    )
    record_path.write_text(json.dumps(record), encoding="utf-8")

    def unexpected_resolver(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        raise AssertionError("literal addresses must not be resolved")

    with pytest.raises(CorrespondingSourceError, match="not globally routable"):
        _verify(
            record_path,
            lock_path,
            record,
            resolver=unexpected_resolver,
            reviewed_origins=frozenset({"https://127.0.0.1"}),
        )


def test_hostname_resolution_failure_refuses_before_retrieval(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)

    def failed_resolver(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        raise socket.gaierror("unavailable")

    with pytest.raises(CorrespondingSourceError, match="resolution failed"):
        _verify(record_path, lock_path, record, resolver=failed_resolver)


def test_redirect_refuses_without_accepting_destination(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    with pytest.raises(CorrespondingSourceError, match="redirected"):
        _verify(
            record_path,
            lock_path,
            record,
            response_url="https://127.0.0.1/private",
        )


@pytest.mark.parametrize(
    ("archive", "message"),
    (
        (b"arbitrary source bytes", "Zstandard stream is malformed"),
        (
            zstandard.ZstdCompressor().compress(b"not a TAR archive"),
            "truncated",
        ),
        (_archive(include_manifest=False), "has no source manifest"),
    ),
)
def test_arbitrary_or_malformed_archive_cannot_satisfy_delivery(
    tmp_path: Path,
    archive: bytes,
    message: str,
) -> None:
    record_path, lock_path, record = _fixture(tmp_path, archive=archive)
    with pytest.raises(CorrespondingSourceError, match=message):
        _verify(record_path, lock_path, record, content=archive)


def test_archive_members_must_match_locked_manifest(tmp_path: Path) -> None:
    changed = dict(SOURCE_FILES)
    changed["pool/main/f/fixture/fixture_1.dsc"] = b"changed source\n"
    archive = _archive(changed, manifest=SOURCE_MANIFEST)
    record_path, lock_path, record = _fixture(
        tmp_path,
        archive=archive,
        manifest=SOURCE_MANIFEST,
    )
    with pytest.raises(CorrespondingSourceError, match="members do not match"):
        _verify(record_path, lock_path, record, content=archive)


def test_archive_manifest_bytes_must_match_locked_identity(tmp_path: Path) -> None:
    changed_manifest = _source_manifest(
        {"pool/main/f/other/other.dsc": b"Format: 3.0 (native)\n"}
    )
    archive = _archive(manifest=changed_manifest)
    record_path, lock_path, record = _fixture(
        tmp_path,
        archive=archive,
        manifest=SOURCE_MANIFEST,
    )
    with pytest.raises(CorrespondingSourceError, match="identity does not match"):
        _verify(record_path, lock_path, record, content=archive)


@pytest.mark.parametrize(
    ("member", "message"),
    (
        (tarfile.TarInfo("../escape"), "path is unsafe"),
        (tarfile.TarInfo("pool/main/f/fixture/fixture_1.dsc"), "duplicate path"),
    ),
)
def test_archive_traversal_and_duplicate_paths_refuse(
    tmp_path: Path,
    member: tarfile.TarInfo,
    message: str,
) -> None:
    archive = _archive(extra_members=[(member, b"hostile")])
    record_path, lock_path, record = _fixture(tmp_path, archive=archive)
    with pytest.raises(CorrespondingSourceError, match=message):
        _verify(record_path, lock_path, record, content=archive)


def test_archive_link_member_refuses(tmp_path: Path) -> None:
    link = tarfile.TarInfo("pool/main/f/fixture/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../etc/passwd"
    archive = _archive(extra_members=[(link, b"")])
    record_path, lock_path, record = _fixture(tmp_path, archive=archive)
    with pytest.raises(CorrespondingSourceError, match="link or special"):
        _verify(record_path, lock_path, record, content=archive)


def test_duplicate_manifest_json_field_refuses(tmp_path: Path) -> None:
    manifest = b'{"schema":"wrong","schema":"also-wrong","files":[]}\n'
    archive = _archive(manifest=manifest)
    record_path, lock_path, record = _fixture(
        tmp_path,
        archive=archive,
        manifest=manifest,
    )
    with pytest.raises(CorrespondingSourceError, match="duplicate JSON field"):
        _verify(record_path, lock_path, record, content=archive)


def test_both_publication_paths_consume_the_same_record() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text(
        encoding="utf-8"
    )
    release = (ROOT / "npa/src/npa/deploy/publish_public.py").read_text(
        encoding="utf-8"
    )
    assert "gymnasium_robotics_image_manifest.json" in workflow
    assert "verify_gymnasium_corresponding_source(item)" in release
    assert "CORRESPONDING SOURCE GATE" in release


def test_current_candidate_remains_fail_closed() -> None:
    assert not (
        ROOT / "npa/src/npa/deploy/gymnasium_robotics_image_manifest.json"
    ).exists()
    lock = json.loads(REAL_LOCK.read_text(encoding="utf-8"))
    assert lock["public_corresponding_source_delivery"] == (
        "withheld-until-separate-publication-acceptance"
    )
