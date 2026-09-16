"""Synthetic package fixtures for the exact EVG tokenizer adaptation contract.

These fixtures are not vendor source or real model/tokenizer execution. Every
fixture byte hash is explicitly substituted for the production source hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from npa.workflows import paidf_evg_tokenizer as tokenizer
from npa.workflows.paidf_guardrails import PaidfGuardrailError


SYNTHETIC_SOURCE = b'''class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_path, **kwargs):
        if kwargs.get("tokenizer_type") != "qwen2":
            raise RuntimeError("synthetic explicit tokenizer dispatch required")
        return {"model": model_path, **kwargs}

class SyntheticPipeline:
    def __init__(self):
        model_path = "synthetic-pinned-model"
        local_files_only = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="text_tokenizer",
            local_files_only=local_files_only,
        )
'''
SYNTHETIC_PATCHED = SYNTHETIC_SOURCE.replace(
    b'            subfolder="text_tokenizer",\n',
    b'            tokenizer_type="qwen2",\n            subfolder="text_tokenizer",\n',
)
SYNTHETIC_CONFIG = json.dumps(
    {"tokenizer_class": "Qwen2Tokenizer", "synthetic_test_fixture": True}
).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def synthetic_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenizer, "PIPELINE_SOURCE_SHA256", _sha(SYNTHETIC_SOURCE))
    monkeypatch.setattr(tokenizer, "PIPELINE_PATCHED_SHA256", _sha(SYNTHETIC_PATCHED))
    monkeypatch.setattr(tokenizer, "MODEL_CONFIG_SHA256", _sha(SYNTHETIC_CONFIG))
    home = tmp_path / "home"
    hub = home / "hub"
    hub.mkdir(parents=True)
    config = (
        hub
        / ("models--" + tokenizer.COSMOS3_SUPER_IMAGE2VIDEO_MODEL.replace("/", "--"))
        / "snapshots"
        / tokenizer.COSMOS3_SUPER_IMAGE2VIDEO_REVISION
        / "text_tokenizer/tokenizer_config.json"
    )
    config.parent.mkdir(parents=True)
    config.write_bytes(SYNTHETIC_CONFIG)
    package = tmp_path / "vendor/vllm_omni"
    pipeline = package / tokenizer.PIPELINE_PATH
    pipeline.parent.mkdir(parents=True)
    pipeline.write_bytes(SYNTHETIC_SOURCE)
    for directory in (pipeline.parent, *pipeline.parent.parents):
        (directory / "__init__.py").write_text("")
        if directory == package:
            break
    (package / "retained_vendor_module.py").write_text("VALUE = 'synthetic'\n")
    return home, hub, config, package


def _prepare(runtime):
    home, hub, _config, package = runtime
    return tokenizer.prepare_evg_tokenizer_overlay(home, hub, source_package=package)


def test_fixture_is_not_accepted_as_production_source():
    assert _sha(SYNTHETIC_SOURCE) != tokenizer.PIPELINE_SOURCE_SHA256
    assert _sha(SYNTHETIC_PATCHED) != tokenizer.PIPELINE_PATCHED_SHA256
    assert _sha(SYNTHETIC_CONFIG) != tokenizer.MODEL_CONFIG_SHA256
    with pytest.raises(PaidfGuardrailError, match="reviewed image"):
        tokenizer.patch_pipeline_source(SYNTHETIC_SOURCE)


def test_patch_changes_only_the_explicit_tokenizer_keyword(synthetic_runtime):
    assert tokenizer.patch_pipeline_source(SYNTHETIC_SOURCE) == SYNTHETIC_PATCHED
    adaptation = tokenizer.tokenizer_source_adaptation()
    assert adaptation["license"] == "Apache-2.0"
    assert adaptation["original_sha256"] == _sha(SYNTHETIC_SOURCE)
    assert adaptation["patched_sha256"] == _sha(SYNTHETIC_PATCHED)
    assert adaptation["tokenizer_type"] == "qwen2"
    assert adaptation["model_config_sha256"] == _sha(SYNTHETIC_CONFIG)


@pytest.mark.parametrize("mutation", ["source", "patched", "missing-anchor", "duplicate-anchor"])
def test_source_patch_rejects_drift(synthetic_runtime, monkeypatch, mutation):
    source = SYNTHETIC_SOURCE
    if mutation == "source":
        source += b"# unreviewed change\n"
    elif mutation == "patched":
        monkeypatch.setattr(tokenizer, "PIPELINE_PATCHED_SHA256", "0" * 64)
    else:
        source = (
            source.replace(tokenizer.TOKENIZER_CALL.encode(), b"        pass\n")
            if mutation == "missing-anchor"
            else source + tokenizer.TOKENIZER_CALL.encode()
        )
        # Only source identity changes here; the unique anchor guard must still run.
        monkeypatch.setattr(tokenizer, "PIPELINE_SOURCE_SHA256", _sha(source))
    with pytest.raises(PaidfGuardrailError):
        tokenizer.patch_pipeline_source(source)


def test_overlay_import_reuse_and_vendor_preservation(synthetic_runtime):
    _home, _hub, _config, package = synthetic_runtime
    before = tokenizer._package_files(package, installed=True)
    overlay = _prepare(synthetic_runtime)
    environment = {
        **os.environ,
        "PYTHONPATH": str(overlay),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = """import importlib, json, sys
from pathlib import Path
module = importlib.import_module('vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3')
assert Path(module.__file__).resolve() == Path(sys.argv[1]).resolve()
value = module.SyntheticPipeline().tokenizer
assert value == {'model': 'synthetic-pinned-model', 'tokenizer_type': 'qwen2',
                 'subfolder': 'text_tokenizer', 'local_files_only': True}
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(overlay / "vllm_omni" / tokenizer.PIPELINE_PATH)],
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert _prepare(synthetic_runtime) == overlay
    assert before == tokenizer._package_files(package, installed=True)
    assert (overlay / "vllm_omni" / tokenizer.PIPELINE_PATH).read_bytes() == SYNTHETIC_PATCHED
    assert not list(overlay.rglob("*.pyc"))


@pytest.mark.parametrize("mutation", ["missing", "malformed", "array", "changed-class", "changed-bytes"])
def test_configuration_must_match_the_pinned_class_and_bytes(
    synthetic_runtime, monkeypatch, mutation
):
    home, _hub, config, _package = synthetic_runtime
    if mutation == "missing":
        config.unlink()
    else:
        payload = {
            "malformed": b"{invalid",
            "array": b"[]",
            "changed-class": b'{"tokenizer_class":"AnotherTokenizer"}',
            "changed-bytes": SYNTHETIC_CONFIG + b"\n",
        }[mutation]
        config.write_bytes(payload)
        if mutation in {"array", "changed-class"}:
            monkeypatch.setattr(tokenizer, "MODEL_CONFIG_SHA256", _sha(payload))
    with pytest.raises(PaidfGuardrailError):
        _prepare(synthetic_runtime)
    assert not (home / "npa-paidf-tokenizer-code").exists()


@pytest.mark.parametrize("missing", ["__init__.py", tokenizer.PIPELINE_PATH])
def test_incomplete_vendor_package_refuses(synthetic_runtime, missing):
    (synthetic_runtime[3] / missing).unlink()
    with pytest.raises(PaidfGuardrailError, match="incomplete"):
        _prepare(synthetic_runtime)


@pytest.mark.parametrize("location", ["package", "source-file", "home", "root", "destination"])
def test_directory_and_source_redirects_refuse(synthetic_runtime, tmp_path, location):
    home, hub, config, package = synthetic_runtime
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_bytes(b"untouched")
    if location == "package":
        link = tmp_path / "package-link"
        link.symlink_to(package, target_is_directory=True)
        package = link
    elif location == "source-file":
        source = package / tokenizer.PIPELINE_PATH
        original = outside / "pipeline.py"
        original.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(original)
    elif location == "home":
        link = tmp_path / "home-link"
        link.symlink_to(home, target_is_directory=True)
        home = link
    else:
        root = home / "npa-paidf-tokenizer-code"
        if location == "root":
            root.symlink_to(outside, target_is_directory=True)
        else:
            root.mkdir()
            (root / tokenizer.PIPELINE_PATCHED_SHA256).symlink_to(outside, target_is_directory=True)
    with pytest.raises(PaidfGuardrailError, match="redirect"):
        _prepare((home, hub, config, package))
    assert marker.read_bytes() == b"untouched"


@pytest.mark.parametrize("mutation", ["pipeline", "extra-file", "extra-root", "bytecode", "vendor-change"])
def test_overlay_reuse_rejects_mutation(synthetic_runtime, mutation):
    overlay = _prepare(synthetic_runtime)
    if mutation == "pipeline":
        target = overlay / "vllm_omni" / tokenizer.PIPELINE_PATH
    elif mutation == "extra-root":
        target = overlay / "unexpected"
    elif mutation == "bytecode":
        target = overlay / "vllm_omni/unexpected.pyc"
    elif mutation == "vendor-change":
        target = synthetic_runtime[3] / "retained_vendor_module.py"
    else:
        target = overlay / "vllm_omni/unexpected.py"
    if target.exists():
        target.chmod(0o600)
    target.write_bytes(b"unreviewed mutation")
    with pytest.raises(PaidfGuardrailError, match="differs"):
        _prepare(synthetic_runtime)
    assert target.read_bytes() == b"unreviewed mutation"


@pytest.mark.parametrize("location", ["vendor", "overlay"])
def test_special_package_files_refuse_without_reading(synthetic_runtime, monkeypatch, location):
    package = synthetic_runtime[3]
    if location == "overlay":
        package = _prepare(synthetic_runtime) / "vllm_omni"
    special = package / "unexpected.fifo"
    os.mkfifo(special)
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path != special, "special file reached a blocking content read"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(PaidfGuardrailError, match="non-regular"):
        _prepare(synthetic_runtime)


def test_normal_hf_snapshot_link_uses_the_same_model_blob(synthetic_runtime):
    _home, _hub, config, _package = synthetic_runtime
    model = config.parents[3]
    blob = model / "blobs" / _sha(SYNTHETIC_CONFIG)
    blob.parent.mkdir()
    blob.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(blob)
    overlay = _prepare(synthetic_runtime)
    assert (overlay / "vllm_omni" / tokenizer.PIPELINE_PATH).is_file()
    assert blob.read_bytes() == SYNTHETIC_CONFIG


@pytest.mark.parametrize("mutation", ["outside", "other-model", "directory", "fifo", "cycle", "ancestor"])
def test_config_redirects_and_special_targets_fail_before_read(
    synthetic_runtime, tmp_path, monkeypatch, mutation
):
    home, hub, config, package = synthetic_runtime
    if mutation == "ancestor":
        linked_hub = tmp_path / "hub-link"
        linked_hub.symlink_to(hub, target_is_directory=True)
        runtime = (home, linked_hub, config, package)
    else:
        config.unlink()
        runtime = synthetic_runtime
        if mutation in {"outside", "other-model"}:
            outside = (
                tmp_path / "outside.json"
                if mutation == "outside"
                else hub / "models--synthetic--other/blobs/fixture"
            )
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_bytes(SYNTHETIC_CONFIG)
            config.symlink_to(outside)
        elif mutation == "directory":
            config.mkdir()
        elif mutation == "fifo":
            os.mkfifo(config)
        else:
            config.symlink_to(config.name)
    read_bytes = Path.read_bytes

    def guarded_read(path):
        assert path.name != "tokenizer_config.json", "unsafe configuration reached content read"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(PaidfGuardrailError, match="configuration"):
        _prepare(runtime)
    assert not (home / "npa-paidf-tokenizer-code").exists()


def test_vendor_mutation_during_staging_is_rejected(synthetic_runtime, monkeypatch):
    vendor = synthetic_runtime[3] / "retained_vendor_module.py"
    original_rename = Path.rename

    def mutate_after_stage(path, destination):
        result = original_rename(path, destination)
        vendor.write_text("VALUE = 'changed during staging'\n")
        return result

    monkeypatch.setattr(Path, "rename", mutate_after_stage)
    with pytest.raises(PaidfGuardrailError, match="differs"):
        _prepare(synthetic_runtime)


def test_partial_staging_failure_removes_only_its_temporary_directory(
    synthetic_runtime, monkeypatch
):
    home, _hub, _config, package = synthetic_runtime
    before = tokenizer._package_files(package, installed=True)
    root = home / "npa-paidf-tokenizer-code"
    root.mkdir()
    marker = root / "unrelated-owner-marker"
    marker.write_bytes(b"retained")
    original_write = Path.write_bytes

    def interrupted_write(path, payload):
        if path.name == "pipeline_cosmos3.py":
            raise OSError("synthetic interrupted staging")
        return original_write(path, payload)

    monkeypatch.setattr(Path, "write_bytes", interrupted_write)
    with pytest.raises(OSError, match="interrupted staging"):
        _prepare(synthetic_runtime)
    assert list(root.iterdir()) == [marker]
    assert marker.read_bytes() == b"retained"
    assert tokenizer._package_files(package, installed=True) == before
