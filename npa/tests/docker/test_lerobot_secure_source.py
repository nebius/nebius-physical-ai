"""Verify the maintained wheel cannot conceal changed model code or notices."""

from __future__ import annotations

import ast
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
EXPECTED_DEPENDENCIES = {
    "torch<2.11.0,>=2.7": "torch<2.14.0,>=2.13.0",
    "torchvision<0.26.0,>=0.22.0": "torchvision<0.29.0,>=0.28.0",
    "torchcodec<0.11.0,>=0.3.0": "torchcodec<0.17.0,>=0.16.0",
    "diffusers<0.36.0,>=0.27.2": "diffusers<0.39.0,>=0.38.0",
    "Pillow<13.0.0,>=10.0.0": "Pillow<14.0.0,>=13.0.0",
    "setuptools<81.0.0,>=71.0.0": "setuptools<85.0.0,>=84.1.0",
    "datasets<5.0.0,>=4.0.0": "datasets<5.1.0,>=5.0.1",
    "transformers==5.3.0": "transformers<5.11.0,>=5.10.0",
    "wandb<0.25.0,>=0.24.0": "wandb<0.31.0,>=0.30.0",
}
GROOT_INIT = b"""#!/usr/bin/env python

# Copyright 2025 Nvidia and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .configuration_groot import GrootConfig
from .modeling_groot import GrootPolicy
from .processor_groot import make_groot_pre_post_processors

__all__ = [\"GrootConfig\", \"GrootPolicy\", \"make_groot_pre_post_processors\"]
"""


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
            ("src/" + module.GROOT_INIT_PATH, GROOT_INIT),
            ("LICENSE", b"Apache notice"),
        ):
            item = tarfile.TarInfo(f"{module.SOURCE_ARCHIVE_ROOT}/{path}")
            item.size = len(value)
            archive.addfile(item, io.BytesIO(value))
    return output.getvalue()


def _upstream(module, monkeypatch, *, model=b"native model source", wheel_model=None):
    logger = b"logging.info(wandb.run.get_url())\n"
    original = (
        "Metadata-Version: 2.4\nName: lerobot\nVersion: 0.5.1\n"
        "License-File: LICENSE\nRequires-Dist: requests>=2.0\n"
        + "".join(f"Requires-Dist: {value}\n" for value in EXPECTED_DEPENDENCIES)
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as wheel:
        wheel.writestr(
            "lerobot/model.py", model if wheel_model is None else wheel_model
        )
        wheel.writestr(module.LOGGER_PATH, logger)
        wheel.writestr(module.GROOT_INIT_PATH, GROOT_INIT)
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
        ("GROOT_INIT_SHA256", GROOT_INIT),
    ):
        monkeypatch.setattr(module, field, hashlib.sha256(value).hexdigest())
    return content, archive


def test_versioned_integration_preserves_native_source_notices_and_record(
    integration, monkeypatch, tmp_path
):
    assert integration.DEPENDENCIES == EXPECTED_DEPENDENCIES
    assert hashlib.sha256(GROOT_INIT).hexdigest() == integration.GROOT_INIT_SHA256
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
        groot_tree = ast.parse(contents.read(integration.GROOT_INIT_PATH))
        top_level_imports = [
            node
            for node in groot_tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "modeling_groot"
        ]
        assert top_level_imports == []
        lazy_getattr = next(
            node
            for node in groot_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        )
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "modeling_groot"
            for node in ast.walk(lazy_getattr)
        )
        assert (
            contents.read(integration.DIST_INFO + "licenses/LICENSE")
            == b"Apache notice"
        )
        metadata = contents.read(integration.DIST_INFO + "METADATA").decode()
        assert f"Version: {integration.VERSION}" in metadata
        assert "Requires-Dist: wandb<0.31.0,>=0.30.0" in metadata
        assert "Requires-Dist: Pillow<14.0.0,>=13.0.0" in metadata
        assert "Requires-Dist: setuptools<85.0.0,>=84.1.0" in metadata
        assert "Requires-Dist: transformers<5.11.0,>=5.10.0" in metadata
        assert "Requires-Dist: setuptools<81.0.0,>=71.0.0" not in metadata
        assert "Requires-Dist: transformers==5.3.0" not in metadata
        assert "Requires-Dist: requests>=2.0" in metadata
        assert "License-File: LICENSE" in metadata
        receipt = json.loads(
            contents.read(integration.DIST_INFO + "npa-source-integration.json")
        )
        assert receipt["declared_upstream_commit"] == integration.UPSTREAM_COMMIT
        assert "upstream_commit" not in receipt
        assert receipt["upstream_source_distribution"] == "lerobot-0.5.1.tar.gz"
        assert receipt["native_model_source_unchanged"] is True
        assert receipt["package_source_unchanged"] is False
        assert list(receipt["source_patches"]) == [
            integration.LOGGER_PATH,
            integration.GROOT_INIT_PATH,
        ]
        assert (
            "modification notice"
            in receipt["source_patches"][integration.LOGGER_PATH]["change"]
        )
        assert (
            "deferred optional GR00T runtime imports"
            in receipt["source_patches"][integration.GROOT_INIT_PATH]["change"]
        )
        assert receipt["deferred_optional_policy_qualification"] == {
            "groot": "unqualified under the security-pinned transformers runtime; a request to instantiate GrootPolicy requires a separate runtime qualification"
        }
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


def test_unreviewed_groot_initializer_fails_before_output_creation(
    integration, monkeypatch, tmp_path
):
    wheel, archive = _upstream(integration, monkeypatch)
    monkeypatch.setattr(integration, "GROOT_INIT_SHA256", "0" * 64)
    with pytest.raises(
        RuntimeError, match="unreviewed upstream GR00T package initializer"
    ):
        integration._prepare(wheel, archive, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_eager_groot_model_imports_are_rejected(integration):
    with pytest.raises(RuntimeError, match="eager optional GR00T model import remains"):
        integration._validate_groot_import_surface(
            b"from .modeling_groot import GrootPolicy\n"
        )

    with pytest.raises(RuntimeError, match="eager optional GR00T model import remains"):
        integration._validate_groot_import_surface(
            b"from .processor_groot import make_groot_pre_post_processors\n"
        )


@pytest.mark.parametrize("member", ["licenses/LICENSE", "METADATA"])
def test_missing_required_wheel_member_has_a_structured_failure(
    integration, monkeypatch, tmp_path, member
):
    wheel, archive = _upstream(integration, monkeypatch)
    rebuilt = io.BytesIO()
    missing = integration.ORIGINAL_DIST_INFO + member
    with (
        zipfile.ZipFile(io.BytesIO(wheel)) as source,
        zipfile.ZipFile(rebuilt, "w") as output,
    ):
        for entry in source.infolist():
            if entry.filename != missing:
                output.writestr(entry, source.read(entry.filename))
    wheel = rebuilt.getvalue()
    monkeypatch.setattr(integration, "WHEEL_SHA256", hashlib.sha256(wheel).hexdigest())

    with pytest.raises(RuntimeError, match="unreviewed upstream wheel layout"):
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


def test_source_archive_uses_the_immutable_pypi_distribution(integration):
    assert integration.ARCHIVE_URL.startswith("https://files.pythonhosted.org/")
    assert integration.ARCHIVE_URL.endswith("/lerobot-0.5.1.tar.gz")
    assert integration.SOURCE_ARCHIVE_ROOT == "lerobot-0.5.1"
    assert len(integration.ARCHIVE_SHA256) == 64


def test_native_source_claim_is_derived_from_the_patch_paths(integration):
    assert integration._native_model_source_unchanged({integration.LOGGER_PATH: {}})
    assert integration._native_model_source_unchanged({integration.GROOT_INIT_PATH: {}})
    assert not integration._native_model_source_unchanged(
        {"lerobot/policies/act/modeling_act.py": {}}
    )
    assert not integration._native_model_source_unchanged(
        {"lerobot/policies/groot/processor_groot.py": {}}
    )
