"""DIG-specific source adaptation and artifact enforcement contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys

import pytest

from npa.workflows import paidf_dig_guardrails as dig
from npa.workflows.paidf_guardrails import PaidfGuardrailError, _digest_document


PROTOCOL_FIXTURE = b"""import re
class Qwen3Guard:
    def extract_label_and_categories(self, content):
        if isinstance(content, Exception):
            raise content
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        return label.lower() != "unsafe", label

    def is_safe(self, content):
        try:
            return self.extract_label_and_categories(content)
        except Exception as e:
            return True, "Unexpected error occurred when running Qwen3Guard guardrail."
"""


@pytest.fixture
def reviewed_fixture(monkeypatch):
    patched = dig._qwen_guardrail_patch_bytes(PROTOCOL_FIXTURE)
    monkeypatch.setattr(
        dig, "DIG_QWEN_SOURCE_SHA256", hashlib.sha256(PROTOCOL_FIXTURE).hexdigest()
    )
    monkeypatch.setattr(
        dig, "DIG_QWEN_PATCHED_SHA256", hashlib.sha256(patched).hexdigest()
    )
    return PROTOCOL_FIXTURE


def _package(tmp_path, original):
    package = tmp_path / "vendor/cosmos_framework"
    source = package / "auxiliary/guardrail/qwen3guard/qwen3guard.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(original)
    for parent in (source.parent, *source.parent.parents):
        (parent / "__init__.py").write_text("")
        if parent == package:
            break
    return package


def _source_record():
    return {
        "source_adaptation": dig.dig_qwen_source_adaptation(),
        "installed_package_tree_sha256": "a" * 64,
        "overlay_tree_sha256": "b" * 64,
        "package_file_count": 5,
    }


def _summary():
    flags = {
        "guardrail_enabled": True,
        "text_guardrail_enforcing": True,
        "image_guardrail_enforcing": False,
    }
    return {
        **flags,
        "world_size": 1,
        "generated_images_total": 2,
        "guardrail_blocked_total": 0,
        "rank_timings": [{**flags, "generated_images": 2}],
    }


@pytest.mark.parametrize(
    "verdict,allowed", [("Safe", True), ("Unsafe", False), ("Controversial", True)]
)
def test_dig_preserves_published_verdicts(reviewed_fixture, verdict, allowed):
    namespace = {"re": re}
    exec(dig.patch_dig_qwen_source(reviewed_fixture), namespace)
    assert namespace["Qwen3Guard"]().is_safe(
        f"Safety: {verdict}\nCategories: None"
    ) == (allowed, verdict)


@pytest.mark.parametrize(
    "verdict",
    [
        "",
        "Safety: Safely",
        "Safety: Unknown",
        "Safety: Safe extra",
        "Safety: Safe\nSafety: Unsafe",
        "Safety: Safe\nSafety : Safe",
        ImportError("synthetic inference dependency missing"),
    ],
)
def test_dig_verdict_and_inference_failures_cannot_reach_generation(
    reviewed_fixture, verdict
):
    namespace = {"re": re}
    exec(dig.patch_dig_qwen_source(reviewed_fixture), namespace)
    with pytest.raises(RuntimeError, match="failed closed"):
        namespace["Qwen3Guard"]().is_safe(verdict)


def test_dig_does_not_accept_evg_or_unknown_source():
    from npa.workflows.paidf_guardrails import QWEN_GUARDRAIL_SOURCE_SHA256

    assert dig.DIG_QWEN_SOURCE_SHA256 != QWEN_GUARDRAIL_SOURCE_SHA256
    with pytest.raises(PaidfGuardrailError, match="reviewed framework"):
        dig.patch_dig_qwen_source(PROTOCOL_FIXTURE)
    adaptation = dig.dig_qwen_source_adaptation()
    assert adaptation["license"] == "OpenMDW-1.1"
    assert adaptation["revision"] == "a904d2d36b774a51dd06ff9ff906816b1a04f579"


def test_overlay_imports_selected_bytes_and_preserves_installed_package(
    tmp_path, reviewed_fixture
):
    package = _package(tmp_path, reviewed_fixture)
    original = dig._package_files(package, installed=True)
    destination = tmp_path / "overlay"
    child, record = dig.prepare_dig_guardrail_overlay(
        destination, dict(os.environ), source_package=package
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            dig.DIG_IMPORT_PROBE,
            str(destination / dig.DIG_QWEN_PATH),
            dig.DIG_QWEN_PATCHED_SHA256,
        ],
        env=child,
        check=True,
        capture_output=True,
    )
    dig.verify_dig_guardrail_overlay(destination, record, source_package=package)
    assert original == dig._package_files(package, installed=True)
    assert not list(destination.rglob("*.pyc"))
    assert child["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize(
    "mutation", ["source", "overlay", "added-bytecode", "installed-change"]
)
def test_overlay_rejects_changed_or_unreviewed_bytes(
    tmp_path, reviewed_fixture, mutation
):
    package = _package(tmp_path, reviewed_fixture)
    if mutation == "source":
        (package / "auxiliary/guardrail/qwen3guard/qwen3guard.py").write_bytes(
            b"changed"
        )
        with pytest.raises(PaidfGuardrailError, match="reviewed framework"):
            dig.prepare_dig_guardrail_overlay(
                tmp_path / "overlay", {}, source_package=package
            )
        return
    destination = tmp_path / "overlay"
    _, record = dig.prepare_dig_guardrail_overlay(
        destination, {}, source_package=package
    )
    if mutation == "overlay":
        path = destination / dig.DIG_QWEN_PATH
    elif mutation == "installed-change":
        path = package / "__init__.py"
    else:
        path = destination / "cosmos_framework/unexpected.pyc"
    if path.exists():
        path.chmod(0o600)
    path.write_bytes(b"changed")
    with pytest.raises(PaidfGuardrailError, match="changed after"):
        dig.verify_dig_guardrail_overlay(destination, record, source_package=package)


@pytest.mark.parametrize("location", ["source", "destination", "ancestor"])
def test_overlay_refuses_symlink_redirects(tmp_path, reviewed_fixture, location):
    package = _package(tmp_path, reviewed_fixture)
    destination = tmp_path / "overlay"
    if location == "source":
        link = tmp_path / "source-link"
        link.symlink_to(package, target_is_directory=True)
        package = link
    elif location == "destination":
        destination.symlink_to(tmp_path / "outside", target_is_directory=True)
    else:
        ancestor = tmp_path / "ancestor"
        ancestor.symlink_to(tmp_path, target_is_directory=True)
        destination = ancestor / "overlay"
    with pytest.raises(PaidfGuardrailError, match="redirected"):
        dig.prepare_dig_guardrail_overlay(destination, {}, source_package=package)
    assert not (tmp_path / "outside").exists()


def test_import_probe_rejects_a_different_selected_module(tmp_path, reviewed_fixture):
    package = _package(tmp_path, reviewed_fixture)
    child = {**os.environ, "PYTHONPATH": str(package.parent)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dig.DIG_IMPORT_PROBE,
            str(tmp_path / "missing.py"),
            dig.DIG_QWEN_PATCHED_SHA256,
        ],
        env=child,
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"did not import the reviewed Qwen adaptation" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "disabled",
        "missing-text",
        "false-image-claim",
        "rank-disabled",
        "count",
        "bool-count",
        "missing-rank",
        "malformed",
    ],
)
def test_output_summary_must_prove_actual_upstream_behavior(tmp_path, mutation):
    summary = _summary()
    if mutation == "disabled":
        summary["guardrail_enabled"] = False
    elif mutation == "missing-text":
        del summary["text_guardrail_enforcing"]
    elif mutation == "false-image-claim":
        summary["image_guardrail_enforcing"] = True
    elif mutation == "rank-disabled":
        summary["rank_timings"][0]["guardrail_enabled"] = False
    elif mutation == "count":
        summary["generated_images_total"] = 3
    elif mutation == "bool-count":
        summary["generated_images_total"] = True
    elif mutation == "missing-rank":
        summary["rank_timings"] = []
    else:
        summary = []
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(PaidfGuardrailError):
        dig.dig_guardrail_runtime(path, _source_record(), 2)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "license",
        "revision",
        "source-hash",
        "disabled",
        "false-image-claim",
        "count",
        "import",
        "hash",
    ],
)
def test_runtime_lineage_rejects_missing_changed_or_false_provenance(
    tmp_path, mutation
):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary()))
    value = dig.dig_guardrail_runtime(path, _source_record(), 2)
    dig.require_dig_guardrail_runtime(value, 2)
    assert (
        value["timing_summary_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    value = copy.deepcopy(value)
    if mutation == "none":
        value = None
    elif mutation in {"license", "revision", "source-hash"}:
        key = "original_sha256" if mutation == "source-hash" else mutation
        value["source_adaptation"][key] = "changed"
    elif mutation == "disabled":
        value["text_guardrail_enforcing"] = False
    elif mutation == "false-image-claim":
        value["image_guardrail_enforcing"] = True
    elif mutation == "count":
        value["generated_images"] = 3
    elif mutation == "import":
        value["vendor_import_verified"] = False
    else:
        value["overlay_tree_sha256"] = "invalid"
    if isinstance(value, dict):
        value["contract_sha256"] = _digest_document(
            {k: v for k, v in value.items() if k != "contract_sha256"}
        )
    with pytest.raises(PaidfGuardrailError):
        dig.require_dig_guardrail_runtime(value, 2)
