from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "npa/src/npa/assets/paidf_starter_video.json"


def test_paidf_starter_contract_is_pinned_real_and_runtime_fetched() -> None:
    asset = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert asset["source_kind"] == "upstream_sample"
    assert asset["input_origin"] == "actual_capture"
    assert len(asset["source"]["immutable_revision"]) == 40
    assert asset["source"]["immutable_revision"] in asset["source"]["asset_url"]
    assert (
        asset["source"]["immutable_revision"] in asset["source"]["episode_metadata_url"]
    )
    assert asset["integrity"] == {
        "sha256": "caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198",
        "byte_size": 607681,
    }
    assert asset["media"]["container"] == "mp4"
    assert asset["media"]["codec"] == "h264"
    assert asset["license"]["spdx_id"] == "CC-BY-4.0"
    assert asset["license"]["redistribution_permitted"] is True
    assert asset["license"]["hosted_service_use_permitted"] is True
    assert asset["license"]["field_of_use_restrictions"] == "none"
    assert asset["license"]["acceptance_required"] is False
    assert asset["license"]["authentication_required"] is False
    assert asset["delivery"]["mode"] == "operator_runtime_fetch"
    assert asset["delivery"]["binary_redistributed_by_npa"] is False
    assert asset["cosmos_media_decision"]["selected"] is False


def test_paidf_notice_and_packaging_guards_cover_delivery_decision() -> None:
    notice = (REPO_ROOT / "skills/NOTICE-PAIDF-STARTER-MEDIA").read_text()
    packaging = (REPO_ROOT / "npa/docker/workbench/packaging-contract.yaml").read_text()
    pyproject = (REPO_ROOT / "npa/pyproject.toml").read_text()

    assert "Zhiyuan Li" in notice
    assert "CC BY 4.0" in notice
    assert "caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198" in notice
    assert "does not bundle" in notice
    assert "RoboPro starter" in packaging
    assert (
        "upstream Cosmos repository's ambiguously licensed example media" in packaging
    )
    assert '"src/npa/assets/paidf_starter_video.json"' in pyproject


def test_paidf_starter_binary_is_not_bundled() -> None:
    asset = json.loads(CONTRACT.read_text(encoding="utf-8"))
    leaf = Path(asset["source"]["asset_path"]).name

    assert not any(
        path.name == leaf and "docs/demos/assets" not in str(path)
        for path in REPO_ROOT.rglob("*.mp4")
    )
