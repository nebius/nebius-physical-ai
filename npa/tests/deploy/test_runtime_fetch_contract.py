from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from npa.deploy.runtime_fetch_contract import (
    RuntimeFetchContractError,
    validate_runtime_fetch_contract,
)


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/gymnasium-robotics"
MANIFEST = IMAGE / "runtime-fetch-manifest.json"
SOURCE_LOCK = IMAGE / "source-lock.json"
CORRESPONDING_LOCK = IMAGE / "corresponding-source.lock.json"


def _copies(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (
        tmp_path / "manifest.json",
        tmp_path / "source-lock.json",
        tmp_path / "corresponding-source.lock.json",
    )
    for target, source in zip(paths, (MANIFEST, SOURCE_LOCK, CORRESPONDING_LOCK)):
        target.write_bytes(source.read_bytes())
    return paths


def test_payload_free_runtime_fetch_contract_passes() -> None:
    result = validate_runtime_fetch_contract(MANIFEST, SOURCE_LOCK, CORRESPONDING_LOCK)
    assert result["status"] == "passed"
    assert result["payload_free"] is True
    assert result["runtime_artifact_count"] == 20


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "withheld", "not complete"),
        ("rights_boundary", "acceptance proxy", "rights boundary"),
    ],
)
def test_manifest_status_and_rights_are_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    manifest, source_lock, corresponding_lock = _copies(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload[field] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeFetchContractError, match=message):
        validate_runtime_fetch_contract(manifest, source_lock, corresponding_lock)


def test_missing_customer_gate_is_not_a_publication_waiver(tmp_path: Path) -> None:
    manifest, source_lock, corresponding_lock = _copies(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_fetch"].pop("customer_gate")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeFetchContractError, match="fields are incomplete"):
        validate_runtime_fetch_contract(manifest, source_lock, corresponding_lock)


def test_corresponding_source_lock_must_be_runtime_only(tmp_path: Path) -> None:
    manifest, source_lock, corresponding_lock = _copies(tmp_path)
    payload = json.loads(corresponding_lock.read_text(encoding="utf-8"))
    payload["public_corresponding_source_delivery"] = (
        "withheld-until-separate-publication-acceptance"
    )
    corresponding_lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeFetchContractError, match="runtime-fetch-only"):
        validate_runtime_fetch_contract(manifest, source_lock, corresponding_lock)


def test_runtime_artifact_must_remain_runtime_only(tmp_path: Path) -> None:
    manifest, source_lock, corresponding_lock = _copies(tmp_path)
    payload = json.loads(source_lock.read_text(encoding="utf-8"))
    payload["artifacts"][0]["delivery_status"] = "baked"
    source_raw = json.dumps(payload).encode()
    source_lock.write_bytes(source_raw)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["runtime_fetch"]["source_lock_sha256"] = hashlib.sha256(
        source_raw
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(RuntimeFetchContractError, match="not runtime-only"):
        validate_runtime_fetch_contract(manifest, source_lock, corresponding_lock)


def test_runtime_artifact_origin_must_be_official_and_immutable(tmp_path: Path) -> None:
    manifest, source_lock, corresponding_lock = _copies(tmp_path)
    payload = json.loads(source_lock.read_text(encoding="utf-8"))
    payload["artifacts"][0]["url"] = "https://evil.example.invalid/source.tar.gz"
    source_raw = json.dumps(payload).encode()
    source_lock.write_bytes(source_raw)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["runtime_fetch"]["source_lock_sha256"] = hashlib.sha256(
        source_raw
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(RuntimeFetchContractError, match="origin is not approved"):
        validate_runtime_fetch_contract(manifest, source_lock, corresponding_lock)
