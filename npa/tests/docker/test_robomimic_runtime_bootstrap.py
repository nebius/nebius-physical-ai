from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "npa" / "docker" / "workbench" / "robomimic"
SPEC = importlib.util.spec_from_file_location(
    "verify_robomimic_image", IMAGE_ROOT / "verify_image.py"
)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(tmp_path: Path) -> tuple[Path, Path]:
    runtime_root = tmp_path / "runtime"
    interpreter = runtime_root / "payload" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    lock_path = IMAGE_ROOT / "runtime-requirements.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = {
        "schema": "npa.robomimic.runtime-inventory.v1",
        "runtime_id": lock["runtime_id"],
        "lock_sha256": _sha(lock_path),
        "source_revision": lock["source_revision"],
        "abi": lock["abi"],
        "packages": lock["packages"],
        "artifacts": [
            {
                "name": name,
                "version": version,
                "filename": f"{name}-{version}.whl",
                "source": "https://operator.invalid/runtime-wheelhouse/",
                "sha256": hashlib.sha256(f"{name}=={version}".encode()).hexdigest(),
            }
            for name, version in lock["packages"].items()
        ],
        "files": [
            {
                "path": "payload/bin/python",
                "size": interpreter.stat().st_size,
                "sha256": _sha(interpreter),
            }
        ],
        "symlinks": [],
    }
    inventory_path = runtime_root / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "schema": "npa.robomimic.runtime-ready.v1",
        "runtime_id": lock["runtime_id"],
        "lock_sha256": _sha(lock_path),
        "source_revision": lock["source_revision"],
        "inventory_sha256": _sha(inventory_path),
    }
    (runtime_root / ".ready.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runtime_root, lock_path


def test_exact_external_runtime_inventory_verifies_without_mutation(
    tmp_path: Path,
) -> None:
    runtime_root, lock_path = _runtime(tmp_path)
    before = {
        path.relative_to(runtime_root).as_posix(): _sha(path)
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    result = verifier.verify_external_runtime(
        runtime_root=runtime_root, runtime_lock_path=lock_path
    )
    after = {
        path.relative_to(runtime_root).as_posix(): _sha(path)
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    assert result["package_count"] == 22
    assert result["artifact_count"] == 22
    assert result["payload_file_count"] == 1
    assert before == after


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "wrong-package", "extra"])
def test_runtime_inventory_fails_closed(tmp_path: Path, mutation: str) -> None:
    runtime_root, lock_path = _runtime(tmp_path)
    if mutation == "missing":
        (runtime_root / ".ready.json").unlink()
    elif mutation == "corrupt":
        (runtime_root / "payload" / "bin" / "python").write_text("changed")
    elif mutation == "wrong-package":
        inventory_path = runtime_root / "inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["packages"]["torch"] = "0"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        marker_path = runtime_root / ".ready.json"
        marker = json.loads(marker_path.read_text())
        marker["inventory_sha256"] = _sha(inventory_path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    else:
        (runtime_root / "payload" / "extra").write_text("undeclared")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_external_runtime(
            runtime_root=runtime_root, runtime_lock_path=lock_path
        )


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    runtime_root, lock_path = _runtime(tmp_path)
    inventory_path = runtime_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())
    link = runtime_root / "payload" / "escape"
    os.symlink("../../../outside", link)
    inventory["symlinks"] = [{"path": "payload/escape", "target": "../../../outside"}]
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    marker_path = runtime_root / ".ready.json"
    marker = json.loads(marker_path.read_text())
    marker["inventory_sha256"] = _sha(inventory_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="escapes"):
        verifier.verify_external_runtime(
            runtime_root=runtime_root, runtime_lock_path=lock_path
        )


def test_bootstrap_has_no_fetch_install_or_cache_population_path() -> None:
    text = (IMAGE_ROOT / "runtime_bootstrap.sh").read_text(encoding="utf-8")
    for forbidden in (
        "curl ",
        "wget ",
        "pip install",
        "uv sync",
        "git clone",
        "ensure",
    ):
        assert forbidden not in text
    assert "assert-refusal" in text
    assert "verify" in text
