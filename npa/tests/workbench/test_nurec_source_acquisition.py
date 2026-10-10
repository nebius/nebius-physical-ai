from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from npa.workbench.nurec import source_acquisition


def test_public_source_acquisition_is_anonymous_exact_and_private(
    monkeypatch, tmp_path: Path
) -> None:
    payload = b"synthetic pinned public archive"
    monkeypatch.setattr(
        source_acquisition, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest()
    )
    calls = []

    def downloader(url, output, **kwargs):
        calls.append((url, kwargs))
        output.write(payload)

    output = tmp_path / "source.zip"
    receipt_path = tmp_path / "source.json"
    receipt = source_acquisition.acquire_public_source(
        output, receipt_path, downloader=downloader
    )
    assert calls[0][0] == source_acquisition.SOURCE_URL
    assert calls[0][1]["allowed_hosts"] == frozenset({"huggingface.co"})
    policy = calls[0][1]["redirect_host_policy"]
    assert policy("cas-bridge.xethub.hf.co")
    assert policy("us.aws.cdn.hf.co")
    assert not policy("cdn.hf.co.attacker.invalid")
    assert receipt["credentials_forwarded"] is False
    assert receipt["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert json.loads(receipt_path.read_text()) == receipt


def test_public_source_acquisition_removes_wrong_or_partial_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "source.zip"
    receipt_path = tmp_path / "source.json"

    def downloader(_url, stream, **_kwargs):
        stream.write(b"wrong")

    with pytest.raises(source_acquisition.NcoreSourceAcquisitionError, match="SHA-256"):
        source_acquisition.acquire_public_source(
            output, receipt_path, downloader=downloader
        )
    assert not output.exists()
    assert not receipt_path.exists()


def test_public_source_acquisition_never_overwrites_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "source.zip"
    output.write_bytes(b"owned")
    with pytest.raises(
        source_acquisition.NcoreSourceAcquisitionError, match="must be new"
    ):
        source_acquisition.acquire_public_source(
            output,
            tmp_path / "source.json",
            downloader=lambda *_args, **_kwargs: None,
        )
    assert output.read_bytes() == b"owned"
