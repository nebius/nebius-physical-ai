"""Verify the maintained wheel cannot conceal changed model code or notices."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def integration():
    path = ROOT / "npa/docker/workbench/lerobot/prepare_secure_wheel.py"
    spec = importlib.util.spec_from_file_location("secure_lerobot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(module, model, logger):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path, value in (
            ("src/lerobot/model.py", model),
            ("src/" + module.LOGGER_PATH, logger),
            ("LICENSE", b"Apache notice"),
        ):
            item = tarfile.TarInfo(f"lerobot-{module.UPSTREAM_COMMIT}/{path}")
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    return output.getvalue()


def _upstream(module, monkeypatch, *, model=b"native model source", wheel_model=None):
    logger = b"logging.info(wandb.run.get_url())\n"
    original = (
        "Metadata-Version: 2.4\nName: lerobot\nVersion: 0.5.1\n"
        "License-File: LICENSE\nRequires-Dist: requests>=2.0\n"
        + "".join(f"Requires-Dist: {value}\n" for value in module.DEPENDENCIES)
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as wheel:
        wheel.writestr(
            "lerobot/model.py", model if wheel_model is None else wheel_model
        )
        wheel.writestr(module.LOGGER_PATH, logger)
        wheel.writestr(module.ORIGINAL_DIST_INFO + "METADATA", original)
        wheel.writestr(
            module.ORIGINAL_DIST_INFO + "WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
        wheel.writestr(module.ORIGINAL_DIST_INFO + "licenses/LICENSE", b"Apache notice")
        wheel.writestr(module.ORIGINAL_DIST_INFO + "RECORD", "original record")
    content, archive = output.getvalue(), _archive(module, model, logger)
    for field, value in (
        ("WHEEL_SHA256", content),
        ("ARCHIVE_SHA256", archive),
        ("METADATA_SHA256", original),
        ("LOGGER_SHA256", logger),
    ):
        monkeypatch.setattr(module, field, hashlib.sha256(value).hexdigest())
    return content, archive


def test_versioned_integration_preserves_native_source_notices_and_record(
    integration, monkeypatch, tmp_path
):
    wheel, archive = _upstream(integration, monkeypatch)
    first = integration._prepare(wheel, archive, tmp_path / "first")
    second = integration._prepare(wheel, archive, tmp_path / "second")
    assert first["integration_wheel_sha256"] == second["integration_wheel_sha256"]
    built = tmp_path / "first" / f"lerobot-{integration.VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(built) as contents:
        assert contents.read("lerobot/model.py") == b"native model source"
        assert (
            contents.read(integration.LOGGER_PATH)
            == integration.LOGGER_MODIFICATION_NOTICE + b"logging.info(wandb.run.url)\n"
        )
        assert (
            contents.read(integration.DIST_INFO + "licenses/LICENSE")
            == b"Apache notice"
        )
        metadata = contents.read(integration.DIST_INFO + "METADATA").decode()
        assert f"Version: {integration.VERSION}" in metadata
        assert "Requires-Dist: wandb<0.31.0,>=0.30.0" in metadata
        assert "Requires-Dist: requests>=2.0" in metadata
        assert "License-File: LICENSE" in metadata
        receipt = json.loads(
            contents.read(integration.DIST_INFO + "npa-source-integration.json")
        )
        assert receipt["upstream_commit"] == integration.UPSTREAM_COMMIT
        assert receipt["native_model_source_unchanged"] is True
        assert receipt["package_source_unchanged"] is False
        assert list(receipt["source_patches"]) == [integration.LOGGER_PATH]
        assert (
            "modification notice"
            in receipt["source_patches"][integration.LOGGER_PATH]["change"]
        )
        assert receipt["capability_qualification"].startswith("pending")
        for name, digest, size in csv.reader(
            contents.read(integration.DIST_INFO + "RECORD").decode().splitlines()
        ):
            if not digest:
                assert name == integration.DIST_INFO + "RECORD"
                continue
            value = contents.read(name)
            expected = (
                base64.urlsafe_b64encode(hashlib.sha256(value).digest())
                .decode()
                .rstrip("=")
            )
            assert digest == "sha256=" + expected and int(size) == len(value)


def test_changed_native_model_fails_before_output_creation(
    integration, monkeypatch, tmp_path
):
    wheel, archive = _upstream(integration, monkeypatch, wheel_model=b"changed model")
    with pytest.raises(RuntimeError, match="pinned upstream commit"):
        integration._prepare(wheel, archive, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_unreviewed_logger_fails_before_output_creation(
    integration, monkeypatch, tmp_path
):
    wheel, archive = _upstream(integration, monkeypatch)
    monkeypatch.setattr(integration, "LOGGER_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="unreviewed upstream logger source"):
        integration._prepare(wheel, archive, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("artifact", ["wheel", "archive"])
def test_unreviewed_artifact_fails_before_output_creation(
    integration, monkeypatch, tmp_path, artifact
):
    wheel, archive = _upstream(integration, monkeypatch)
    if artifact == "wheel":
        wheel += b"unreviewed"
    else:
        archive += b"unreviewed"
    with pytest.raises(RuntimeError, match="unreviewed upstream"):
        integration._prepare(wheel, archive, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_dependency_drift_is_rejected(integration, monkeypatch):
    _upstream(integration, monkeypatch)
    invalid = (
        "Version: 0.5.1\n"
        + "Requires-Dist: "
        + next(iter(integration.DEPENDENCIES))
        + "\n"
    ).encode()
    with pytest.raises(RuntimeError, match="dependency edges changed"):
        integration._metadata(invalid)


def test_download_refuses_unreviewed_url_before_network(integration, monkeypatch):
    def unexpected_network(*_args, **_kwargs):
        pytest.fail("unreviewed URL reached the transport")

    monkeypatch.setattr(integration.http.client, "HTTPSConnection", unexpected_network)
    with pytest.raises(RuntimeError, match="unreviewed upstream download URL"):
        integration._download("https://example.invalid/unreviewed", "unused")


def test_download_uses_a_bounded_https_connection(integration, monkeypatch):
    content = b"pinned source bytes"
    url = "https://files.pythonhosted.org/lerobot.whl"
    calls = {}

    class Response:
        status = 200

        def read(self):
            return content

    class Connection:
        def __init__(self, host, timeout):
            calls["connection"] = (host, timeout)

        def request(self, method, path):
            calls["request"] = (method, path)

        def getresponse(self):
            return Response()

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(integration, "WHEEL_URL", url)
    monkeypatch.setattr(
        integration, "WHEEL_SHA256", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(integration.http.client, "HTTPSConnection", Connection)

    assert integration._download(url, integration.WHEEL_SHA256) == content
    assert calls == {
        "connection": ("files.pythonhosted.org", 60),
        "request": ("GET", "/lerobot.whl"),
        "closed": True,
    }
