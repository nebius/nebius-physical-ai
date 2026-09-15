from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from npa.deploy.corresponding_source import (
    CorrespondingSourceError,
    verify_corresponding_source_delivery,
)

ROOT = Path(__file__).resolve().parents[3]
REAL_LOCK = ROOT / "npa/docker/workbench/gymnasium-robotics/corresponding-source.lock.json"
SOURCE_REVISION = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64
PLATFORM_DIGEST = "sha256:" + "c" * 64
CONFIG_DIGEST = "sha256:" + "d" * 64
ARCHIVE = b"synthetic corresponding source\n"


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


def _accepted_lock() -> dict[str, Any]:
    payload = json.loads(REAL_LOCK.read_text(encoding="utf-8"))
    payload["public_corresponding_source_delivery"] = "accepted-public-immutable"
    return payload


def _record(lock_bytes: bytes, lock: dict[str, Any]) -> dict[str, Any]:
    archive_sha256 = hashlib.sha256(ARCHIVE).hexdigest()
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
                "size_bytes": len(ARCHIVE),
                "media_type": "application/zstd",
                "anonymous": True,
                "immutable": True,
                "contents_manifest_sha256": delivery["source_manifest_sha256"],
            },
        },
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    lock = _accepted_lock()
    lock_bytes = json.dumps(lock, sort_keys=True).encode()
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(lock_bytes)
    record = _record(lock_bytes, lock)
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


def _verify(record_path: Path, lock_path: Path, record: dict[str, Any]) -> None:
    reference = record["corresponding_source"]["artifact"]["reference"]
    verify_corresponding_source_delivery(
        record_path,
        lock_path,
        source_revision=SOURCE_REVISION,
        image_digest=IMAGE_DIGEST,
        platform_manifest_digest=PLATFORM_DIGEST,
        config_digest=CONFIG_DIGEST,
        opener=_opener(reference),
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
        (("image", "platform_manifest_digest"), "sha256:" + "e" * 64, "platform_manifest"),
        (("image", "config_digest"), "sha256:" + "e" * 64, "config_digest"),
        (("image", "platform"), {"os": "linux", "architecture": "arm64"}, "linux/amd64"),
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
    record_path.write_text('{"tool":"gymnasium-robotics","tool":"other"}', encoding="utf-8")
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
    [(ARCHIVE + b"extra", "exceeds accepted size"), (ARCHIVE[:-1], "size does not match"), (b"x" * len(ARCHIVE), "digest does not match")],
)
def test_anonymous_readback_bytes_must_match(
    tmp_path: Path, content: bytes, message: str
) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    reference = record["corresponding_source"]["artifact"]["reference"]
    with pytest.raises(CorrespondingSourceError, match=message):
        verify_corresponding_source_delivery(
            record_path,
            lock_path,
            source_revision=SOURCE_REVISION,
            image_digest=IMAGE_DIGEST,
            platform_manifest_digest=PLATFORM_DIGEST,
            config_digest=CONFIG_DIGEST,
            opener=_opener(reference, content),
        )


def test_anonymous_readback_denial_refuses(tmp_path: Path) -> None:
    record_path, lock_path, record = _fixture(tmp_path)
    reference = record["corresponding_source"]["artifact"]["reference"]
    with pytest.raises(CorrespondingSourceError, match="anonymous source retrieval failed"):
        verify_corresponding_source_delivery(
            record_path,
            lock_path,
            source_revision=SOURCE_REVISION,
            image_digest=IMAGE_DIGEST,
            platform_manifest_digest=PLATFORM_DIGEST,
            config_digest=CONFIG_DIGEST,
            opener=_opener(reference, status=401),
        )


def test_both_publication_paths_consume_the_same_record() -> None:
    workflow = (ROOT / ".github/workflows/publish-public-images.yml").read_text(encoding="utf-8")
    release = (ROOT / "npa/src/npa/deploy/publish_public.py").read_text(encoding="utf-8")
    assert "gymnasium_robotics_image_manifest.json" in workflow
    assert "verify_gymnasium_corresponding_source(item)" in release
    assert "CORRESPONDING SOURCE GATE" in release


def test_current_candidate_remains_fail_closed() -> None:
    assert not (ROOT / "npa/src/npa/deploy/gymnasium_robotics_image_manifest.json").exists()
    lock = json.loads(REAL_LOCK.read_text(encoding="utf-8"))
    assert lock["public_corresponding_source_delivery"] == (
        "withheld-until-separate-publication-acceptance"
    )
