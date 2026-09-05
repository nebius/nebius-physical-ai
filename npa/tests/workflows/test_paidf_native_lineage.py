"""Reject foreign producer identities and changed bytes at native handoffs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from npa.workflows import paidf_native as native
from npa.workflows.paidf_upstream import QWEN_IMAGE_EDIT_MODEL


def write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload))
    return str(path)


def generation(tmp_path: Path, workflow: str = "iaa") -> tuple[str, str, dict]:
    media = tmp_path / "image.bmp"
    Image.new("RGB", (96, 96), "blue").save(media)
    (tmp_path / "caption.txt").write_text("A person wearing blue.")
    (tmp_path / "metadata.json").write_text('{"attribute_verification":{"passed":true}}')
    (tmp_path / "config.yaml").write_text("evaluators:\n- attribute_verification:\n    enabled: true\n")
    item = {
        "input_key": "person", "augmentation_index": 0,
        "media_uri": str(media), "caption_uri": str(tmp_path / "caption.txt"),
        "metadata_uri": str(tmp_path / "metadata.json"),
        "config_uri": str(tmp_path / "config.yaml"),
    }
    manifest = write(tmp_path / "configs.json", {
        "schema": f"npa.paidf.native.{workflow}-configs.v1", "workflow": workflow,
        "run_id": "lineage-run", "configs": [item],
    })
    payload = {
        "schema": f"npa.paidf.native.{workflow}-augmentation.v1", "workflow": workflow,
        "run_id": "lineage-run", "runtime_image": "example.test/worker@sha256:" + "a" * 64,
        "source_adaptation": native._paidf_image_output_adaptation(),
        "count": 1, "attempted_count": 1, "failed_count": 0, "failed": [],
        "config_manifest_uri": manifest,
        "outputs": [{**item, "artifacts": native._artifact_fingerprints({
            field: item[field] for field in ("media_uri", "config_uri", "caption_uri", "metadata_uri")
        })}],
    }
    return manifest, write(tmp_path / "augmentation.json", payload), payload


def test_validation_preserves_executed_generation_identity(tmp_path: Path) -> None:
    manifest, producer_uri, producer = generation(tmp_path)
    result = native.validate_augmentation(manifest, str(tmp_path / "validation.json"), "lineage-run", producer_uri)
    assert result["accepted_count"] == 1
    assert result["producers"] == [native._producer_descriptor(producer_uri, producer)]
    assert result["producers"][0]["runtime_image"] == producer["runtime_image"]
    assert result["producers"][0]["source_adaptation"] == (
        native._paidf_image_output_adaptation()
    )


@pytest.mark.parametrize("mutation", [
    "run", "schema", "missing", "duplicate", "foreign-config", "foreign-media",
    "stale-bytes", "no-content-manifest", "missing-adaptation", "changed-adaptation",
])
def test_validation_rejects_unbound_generation_outputs(tmp_path: Path, mutation: str) -> None:
    manifest, producer_uri, producer = generation(tmp_path)
    if mutation == "run":
        producer["run_id"] = "prior-run"
    elif mutation == "schema":
        producer["schema"] = "npa.paidf.native.iaa-configs.v1"
    elif mutation == "missing":
        producer["outputs"] = []
    elif mutation == "duplicate":
        producer["outputs"] *= 2
    elif mutation == "foreign-config":
        producer["config_manifest_uri"] = "another-config.json"
    elif mutation == "foreign-media":
        producer["outputs"][0]["media_uri"] = "another-image.bmp"
    elif mutation == "stale-bytes":
        (tmp_path / "caption.txt").write_text("Changed after the producer finished.")
    elif mutation == "no-content-manifest":
        producer["outputs"][0].pop("artifacts")
    elif mutation == "missing-adaptation":
        producer.pop("source_adaptation")
    else:
        producer["source_adaptation"]["patched_sha256"] = "f" * 64
    write(Path(producer_uri), producer)
    with pytest.raises(native.PaidfNativeError):
        native.validate_augmentation(manifest, str(tmp_path / "validation.json"), "lineage-run", producer_uri)
    assert not (tmp_path / "validation.json").exists()


def test_terminal_lineage_rejects_self_consistent_unreviewed_adaptation(
    tmp_path: Path,
) -> None:
    _manifest, producer_uri, producer = generation(tmp_path, "evg")
    producer["source_adaptation"]["patch_sha256"] = "f" * 64
    write(Path(producer_uri), producer)
    descriptor = native._producer_descriptor(producer_uri, producer)

    with pytest.raises(native.PaidfNativeError, match="reviewed image MIME"):
        native._verified_producers(
            [descriptor], ["evg-augmentation"], "lineage-run", "evg"
        )


@pytest.mark.parametrize("mutation", ["runtime", "run", "missing", "stage", "broken-predecessor"])
def test_producer_chain_rejects_missing_or_replaced_stage(tmp_path: Path, mutation: str) -> None:
    manifest, producer_uri, producer = generation(tmp_path, "evg")
    validation_uri = str(tmp_path / "validation.json")
    validation = native.validate_augmentation(manifest, validation_uri, "lineage-run", producer_uri)
    descriptors = [native._producer_descriptor(producer_uri, producer), native._producer_descriptor(validation_uri, validation)]
    label = {
        "schema": "npa.paidf.native.evg-auto-label-detection.v1", "workflow": "evg",
        "run_id": "lineage-run", "stage": "detection", "producers": descriptors[:],
    }
    label_uri = write(tmp_path / "detection.json", label)
    descriptors.append(native._producer_descriptor(label_uri, label))
    if mutation == "runtime":
        producer["runtime_image"] = "example.test/foreign@sha256:" + "b" * 64
        write(Path(producer_uri), producer)
    elif mutation == "run":
        label["run_id"] = "prior-run"
        write(Path(label_uri), label)
        descriptors[-1] = native._producer_descriptor(label_uri, label)
    elif mutation == "missing":
        descriptors.pop(1)
    else:
        if mutation == "stage":
            label["stage"] = "captioning"
        else:
            label["producers"] = []
        write(Path(label_uri), label)
        descriptors[-1] = native._producer_descriptor(label_uri, label)
    with pytest.raises(native.PaidfNativeError):
        native._verified_producers(descriptors, native._lineage_kinds("evg")[:3], "lineage-run", "evg")


@pytest.mark.parametrize("mutation", ["extra", "missing", "duplicate", "foreign-media", "foreign-root", "changed-sidecar"])
def test_label_join_binds_scene_set_and_content(tmp_path: Path, mutation: str) -> None:
    scene = tmp_path / "labels/person/0"
    scene.mkdir(parents=True)
    sidecar = scene / "objects.json"
    sidecar.write_text('{"objects":[]}')
    accepted = [{"input_key": "person", "augmentation_index": 0, "media_uri": "image.bmp"}]
    output = {
        "key": "person_aug0", "media_uri": "image.bmp", "data_path": str(scene),
        "artifacts": native._artifact_fingerprints({"objects.json": str(sidecar)}),
    }
    document = {"schema": "npa.paidf.native.evg-auto-label-detection.v1", "stage": "detection", "count": 1, "outputs": [output]}
    if mutation == "extra":
        document["outputs"].append({**output, "key": "foreign_aug0"})
    elif mutation == "missing":
        document["outputs"] = []
    elif mutation == "duplicate":
        document["outputs"].append(output)
    elif mutation == "foreign-media":
        output["media_uri"] = "other.bmp"
    elif mutation == "foreign-root":
        output["data_path"] = str(tmp_path / "other")
    else:
        sidecar.write_text('{"objects":["changed"]}')
    document["count"] = len(document["outputs"])
    with pytest.raises(native.PaidfNativeError):
        native._verify_label_handoffs([document], accepted, str(tmp_path / "labels"))


def test_vendor_credentials_are_bound_to_validated_token_in_child_only(monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "approved-token")
    for key in ("VLM_API_KEY", "LLM_API_KEY", "NVIDIA_API_KEY"):
        monkeypatch.setenv(key, "unrelated-token")
    monkeypatch.setenv("GENERATION_API_KEY", "unrelated-generation-token")
    child = native._token_factory_child_env()
    assert child["GENERATION_API_KEY"] == "local"
    assert os.environ["GENERATION_API_KEY"] == "unrelated-generation-token"
    for key in ("VLM_API_KEY", "LLM_API_KEY", "NVIDIA_API_KEY"):
        assert child[key] == "approved-token"
        assert os.environ[key] == "unrelated-token"


def test_vendor_credentials_require_the_validated_source(monkeypatch) -> None:
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-token")
    with pytest.raises(native.PaidfNativeError, match="NEBIUS_TOKEN_FACTORY_KEY"):
        native._token_factory_child_env()


@pytest.mark.parametrize("mutation", ["foreign-origin", "wrong-port", "wrong-model", "wrong-key", "duplicate", "missing", "foreign-alias"])
def test_local_service_cannot_be_replaced_by_another_generation_endpoint(mutation: str) -> None:
    endpoint = {
        "role": "image_edit", "url": "http://127.0.0.1:8000/v1",
        "model": QWEN_IMAGE_EDIT_MODEL, "api_key_env": "GENERATION_API_KEY",
    }
    config = {"endpoints": [endpoint]}
    if mutation == "foreign-origin":
        endpoint["url"] = "https://foreign.example.test/v1"
    elif mutation == "wrong-port":
        endpoint["url"] = "http://127.0.0.1:9000/v1"
    elif mutation == "wrong-model":
        endpoint["model"] = "another-model"
    elif mutation == "wrong-key":
        endpoint["api_key_env"] = "NVIDIA_API_KEY"
    elif mutation == "duplicate":
        config["endpoints"].append(endpoint.copy())
    elif mutation == "missing":
        config["endpoints"] = []
    else:
        config["endpoints"].append({"role": "vlm", "url": "https://foreign.example.test/v1", "api_key_env": "GENERATION_API_KEY"})
    with pytest.raises(native.PaidfNativeError):
        native._validate_local_generation_endpoint(config, "iaa", 8000)


def test_foreign_generation_endpoint_fails_before_starting_model_server(tmp_path: Path, monkeypatch) -> None:
    from npa.workflows.paidf_upstream import QWEN_IMAGE_EDIT_REVISION

    config = write(tmp_path / "config.yaml", {"endpoints": [{
        "role": "image_edit", "url": "https://foreign.example.test/v1",
        "model": QWEN_IMAGE_EDIT_MODEL, "api_key_env": "GENERATION_API_KEY",
    }]})
    manifest = write(tmp_path / "configs.json", {
        "schema": "npa.paidf.native.iaa-configs.v1", "run_id": "lineage-run",
        "workflow": "iaa", "configs": [{"config_uri": config}],
    })
    monkeypatch.setattr(native.subprocess, "Popen", lambda *_a, **_k: pytest.fail("foreign endpoint started a model server"))
    with pytest.raises(native.PaidfNativeError, match="local model service"):
        native.run_local_augmentation(manifest, "unused", QWEN_IMAGE_EDIT_MODEL, QWEN_IMAGE_EDIT_REVISION, "image-edit", 8000, 1, "lineage-run")


@pytest.mark.parametrize("workflow", ["iaa", "evg"])
def test_missing_output_after_successful_cli_obeys_upstream_join(tmp_path: Path, monkeypatch, workflow: str) -> None:
    from npa.workflows.paidf_upstream import COSMOS3_SUPER_IMAGE2VIDEO_MODEL

    items = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        manifest, _, _ = generation(root, workflow)
        item = json.loads(Path(manifest).read_text())["configs"][0]
        item["augmentation_index"] = index
        write(Path(item["config_uri"]), {"endpoints": [{
            "role": "image_edit" if workflow == "iaa" else "image2video",
            "url": "http://127.0.0.1:8000/v1", "api_key_env": "GENERATION_API_KEY",
            "model": QWEN_IMAGE_EDIT_MODEL if workflow == "iaa" else COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
        }]})
        items.append(item)
    Path(items[0]["media_uri"]).unlink()
    manifest = write(tmp_path / "batch.json", {
        "schema": f"npa.paidf.native.{workflow}-configs.v1", "workflow": workflow,
        "run_id": "lineage-run", "configs": items,
    })
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    monkeypatch.setattr(native, "_runtime_fetch", lambda _r, _v, destination: destination)
    monkeypatch.setattr(
        native,
        "_patch_paidf_image_output_contract",
        lambda _source: {
            "schema": "npa.paidf.upstream-source-adaptation.v1",
            "patch_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(native.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(native, "_run_component", lambda *_a, **_k: None)
    if workflow == "evg":
        with pytest.raises(native.PaidfNativeError):
            native.run_augmentation(manifest, str(tmp_path / "result.json"), "lineage-run")
    else:
        result = native.run_augmentation(manifest, str(tmp_path / "result.json"), "lineage-run")
        assert result["count"] == 1
        assert result["failed_count"] == 1
        assert result["failed"][0]["reason"] == "component_output_missing_or_empty"
        assert result["outputs"][0]["augmentation_index"] == 1


@pytest.mark.parametrize("parallel_size", [0, 3, 4])
def test_unsupported_cfg_override_fails_before_model_start(tmp_path: Path, monkeypatch, parallel_size: int) -> None:
    from npa.workflows.paidf_upstream import (
        COSMOS3_SUPER_IMAGE2VIDEO_MODEL, COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
    )

    config = write(tmp_path / "config.yaml", {"endpoints": [{
        "role": "image2video", "url": "http://127.0.0.1:8000/v1",
        "model": COSMOS3_SUPER_IMAGE2VIDEO_MODEL, "api_key_env": "GENERATION_API_KEY",
    }]})
    manifest = write(tmp_path / "configs.json", {
        "schema": "npa.paidf.native.evg-configs.v1", "run_id": "lineage-run",
        "workflow": "evg", "configs": [{"config_uri": config}],
    })
    monkeypatch.setattr(native.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unsupported CFG override started a model server"))
    with pytest.raises(native.PaidfNativeError, match="parallel_size 1 or 2"):
        native.run_local_augmentation(manifest, "unused", COSMOS3_SUPER_IMAGE2VIDEO_MODEL, COSMOS3_SUPER_IMAGE2VIDEO_REVISION, "image2video", 8000, parallel_size, "lineage-run")
