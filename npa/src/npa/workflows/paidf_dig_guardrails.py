"""Fail-closed adaptation of AnomalyGen's pinned OpenMDW Qwen guardrail."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from npa.workflows.paidf_guardrails import (
    PaidfGuardrailError,
    _digest_document,
    _qwen_guardrail_patch_bytes,
)
from npa.workflows.paidf_upstream import DIG_FRAMEWORK_REVISION

DIG_QWEN_PATH = "cosmos_framework/auxiliary/guardrail/qwen3guard/qwen3guard.py"
DIG_QWEN_SOURCE_SHA256 = (
    "3e102cef96d0f11b3af2d0b22eec54f0c7e41f9355ad8742a432681d56211be6"
)
DIG_QWEN_PATCHED_SHA256 = (
    "8d11d043d30f3794f5e4e44ac2bfe6b0fe3328a969be748c846f1e592b628bdb"
)
DIG_VENDOR_PYTHON = "/opt/venv/bin/python"
DIG_VENDOR_PACKAGE = Path("/opt/venv/lib/python3.13/site-packages/cosmos_framework")
DIG_IMPORT_PROBE = """\
import hashlib
import importlib
from pathlib import Path
import sys
module = importlib.import_module('cosmos_framework.auxiliary.guardrail.qwen3guard.qwen3guard')
actual = Path(module.__file__).resolve()
if actual != Path(sys.argv[1]).resolve() or hashlib.sha256(actual.read_bytes()).hexdigest() != sys.argv[2]:
    raise RuntimeError('DIG vendor interpreter did not import the reviewed Qwen adaptation')
"""


def dig_qwen_source_adaptation() -> dict[str, Any]:
    value = {
        "schema": "npa.paidf.dig-guardrail-source-adaptation.v1",
        "repository": "https://github.com/NVIDIA/cosmos-framework",
        "revision": DIG_FRAMEWORK_REVISION,
        "license": "OpenMDW-1.1",
        "path": DIG_QWEN_PATH,
        "original_sha256": DIG_QWEN_SOURCE_SHA256,
        "patched_sha256": DIG_QWEN_PATCHED_SHA256,
        "verdict_protocol": "one complete Safety line: Safe, Unsafe, or Controversial",
        "controversial_policy": "allow-as-upstream",
        "invalid_verdict_or_inference_exception": "raise-before-generation",
    }
    value["patch_sha256"] = _digest_document(value)
    return value


def patch_dig_qwen_source(original: bytes) -> bytes:
    """Reuse the identical two anchors only after verifying DIG's own source."""
    if hashlib.sha256(original).hexdigest() != DIG_QWEN_SOURCE_SHA256:
        raise PaidfGuardrailError("DIG Qwen source differs from the reviewed framework")
    patched = _qwen_guardrail_patch_bytes(original)
    if hashlib.sha256(patched).hexdigest() != DIG_QWEN_PATCHED_SHA256:
        raise PaidfGuardrailError("DIG Qwen adaptation differs from its reviewed bytes")
    return patched


def _package_files(package: Path, *, installed: bool = False) -> dict[str, bytes]:
    if (
        any(path.is_symlink() for path in (package, *package.parents))
        or not package.is_dir()
    ):
        raise PaidfGuardrailError("DIG framework package is missing or redirected")
    files = {}
    for path in package.rglob("*"):
        relative = path.relative_to(package)
        if installed and ("__pycache__" in relative.parts or path.suffix == ".pyc"):
            continue
        if path.is_symlink():
            raise PaidfGuardrailError("DIG framework package contains a source link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PaidfGuardrailError(
                "DIG framework package contains a non-regular file"
            )
        files[relative.as_posix()] = path.read_bytes()
    if "__init__.py" not in files:
        raise PaidfGuardrailError("DIG framework package has no initializer")
    return files


def _tree_hash(files: dict[str, bytes]) -> str:
    return _digest_document(
        {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        }
    )


def prepare_dig_guardrail_overlay(
    destination: Path,
    environment: dict[str, str],
    *,
    source_package: Path = DIG_VENDOR_PACKAGE,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Copy the installed framework into a new run-private directory."""
    files = _package_files(source_package, installed=True)
    original_tree = _tree_hash(files)
    relative = str(Path(DIG_QWEN_PATH).relative_to("cosmos_framework"))
    if relative not in files:
        raise PaidfGuardrailError("DIG framework package has no Qwen source")
    files[relative] = patch_dig_qwen_source(files[relative])
    if destination.exists() or any(
        path.is_symlink() for path in (destination, *destination.parents)
    ):
        raise PaidfGuardrailError(
            "DIG source overlay must be a new unredirected directory"
        )
    destination.mkdir(mode=0o700)
    for name, payload in files.items():
        output = destination / "cosmos_framework" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        output.chmod(0o444)
    source = {
        "source_adaptation": dig_qwen_source_adaptation(),
        "installed_package_tree_sha256": original_tree,
        "overlay_tree_sha256": _tree_hash(files),
        "package_file_count": len(files),
    }
    verify_dig_guardrail_overlay(destination, source, source_package=source_package)
    child = dict(environment)
    child["PYTHONPATH"] = str(destination) + (
        os.pathsep + child["PYTHONPATH"] if child.get("PYTHONPATH") else ""
    )
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child, source


def verify_dig_guardrail_overlay(
    destination: Path,
    source: dict[str, Any],
    *,
    source_package: Path = DIG_VENDOR_PACKAGE,
) -> None:
    files = _package_files(destination / "cosmos_framework")
    relative = str(Path(DIG_QWEN_PATH).relative_to("cosmos_framework"))
    if (
        len(files) != source["package_file_count"]
        or _tree_hash(files) != source["overlay_tree_sha256"]
        or hashlib.sha256(files.get(relative, b"")).hexdigest()
        != DIG_QWEN_PATCHED_SHA256
        or _tree_hash(_package_files(source_package, installed=True))
        != source["installed_package_tree_sha256"]
        or source["source_adaptation"] != dig_qwen_source_adaptation()
    ):
        raise PaidfGuardrailError(
            "DIG framework overlay changed after source verification"
        )


def dig_guardrail_runtime(
    summary_path: Path, source: dict[str, Any], image_count: int
) -> dict[str, Any]:
    """Bind completed upstream enforcement flags and counts to the executed code."""
    try:
        summary_bytes = summary_path.read_bytes()
        summary = json.loads(summary_bytes)
    except (OSError, ValueError) as exc:
        raise PaidfGuardrailError(
            "DIG output has no readable guardrail timing summary"
        ) from exc
    if not isinstance(summary, dict):
        raise PaidfGuardrailError("DIG output guardrail summary is malformed")
    ranks = summary.get("rank_timings")
    if (
        type(summary.get("world_size")) is not int
        or summary["world_size"] != 1
        or not isinstance(ranks, list)
        or len(ranks) != 1
        or not isinstance(ranks[0], dict)
        or type(summary.get("generated_images_total")) is not int
        or summary["generated_images_total"] != image_count
        or type(ranks[0].get("generated_images")) is not int
        or ranks[0]["generated_images"] != image_count
        or type(summary.get("guardrail_blocked_total")) is not int
        or summary["guardrail_blocked_total"] < 0
    ):
        raise PaidfGuardrailError(
            "DIG guardrail summary disagrees with generated output"
        )
    for record in (summary, ranks[0]):
        if (
            record.get("guardrail_enabled") is not True
            or record.get("text_guardrail_enforcing") is not True
            or record.get("image_guardrail_enforcing") is not False
        ):
            raise PaidfGuardrailError(
                "DIG output lacks the reviewed upstream guardrail behavior"
            )
    value = {
        "schema": "npa.paidf.dig-guardrail-runtime.v1",
        **source,
        "vendor_import_verified": True,
        "guardrail_enabled": True,
        "text_guardrail_enforcing": True,
        "image_guardrail_enforcing": False,
        "generated_images": image_count,
        "guardrail_blocked": summary["guardrail_blocked_total"],
        "timing_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }
    value["contract_sha256"] = _digest_document(value)
    require_dig_guardrail_runtime(value, image_count)
    return value


def require_dig_guardrail_runtime(value: Any, image_count: int) -> None:
    """Reject missing or changed code/enforcement provenance at a DIG handoff."""
    expected_fields = {
        "schema",
        "source_adaptation",
        "installed_package_tree_sha256",
        "overlay_tree_sha256",
        "package_file_count",
        "vendor_import_verified",
        "guardrail_enabled",
        "text_guardrail_enforcing",
        "image_guardrail_enforcing",
        "generated_images",
        "guardrail_blocked",
        "timing_summary_sha256",
        "contract_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise PaidfGuardrailError(
            "DIG result has no complete guardrail runtime provenance"
        )
    if (
        value["schema"] != "npa.paidf.dig-guardrail-runtime.v1"
        or value["source_adaptation"] != dig_qwen_source_adaptation()
        or value["vendor_import_verified"] is not True
        or value["guardrail_enabled"] is not True
        or value["text_guardrail_enforcing"] is not True
        or value["image_guardrail_enforcing"] is not False
        or type(value["generated_images"]) is not int
        or value["generated_images"] != image_count
        or type(image_count) is not int
        or image_count < 1
        or type(value["guardrail_blocked"]) is not int
        or value["guardrail_blocked"] < 0
        or type(value["package_file_count"]) is not int
        or value["package_file_count"] < 2
        or any(
            not isinstance(value[key], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value[key])
            for key in (
                "installed_package_tree_sha256",
                "overlay_tree_sha256",
                "timing_summary_sha256",
            )
        )
        or value["contract_sha256"]
        != _digest_document(
            {key: item for key, item in value.items() if key != "contract_sha256"}
        )
    ):
        raise PaidfGuardrailError(
            "DIG guardrail runtime provenance changed or disables protection"
        )
