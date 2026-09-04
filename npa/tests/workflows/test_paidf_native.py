from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from npa.workflows import paidf_native


def test_component_runner_preserves_three_upstream_retries(monkeypatch) -> None:
    attempts = 0

    def fail_then_succeed(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise subprocess.CalledProcessError(1, ["component"])

    monkeypatch.setattr(paidf_native.subprocess, "run", fail_then_succeed)
    monkeypatch.setattr(paidf_native.time, "sleep", lambda _seconds: None)
    paidf_native._run_component(["component"])
    assert attempts == 4


def test_multistorage_config_uses_run_scoped_s3_environment(monkeypatch) -> None:
    monkeypatch.delenv("MULTISTORAGECLIENT_CONFIGURATION", raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://objects.example.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "run-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "run-secret")

    paidf_native._configure_multistorage(
        "s3://input-bucket/inputs/image.png",
        "s3://output-bucket/results/",
    )

    config = json.loads(os.environ["MULTISTORAGECLIENT_CONFIGURATION"])
    assert sorted(config["profiles"]) == ["input-bucket", "output-bucket"]
    assert config["path_mapping"]["s3://input-bucket/"] == "msc://input-bucket/"
    assert config["profiles"]["output-bucket"]["storage_provider"]["options"] == {
        "base_path": "output-bucket",
        "endpoint_url": "https://objects.example.test",
        "infer_content_type": True,
        "region_name": "us-east-1",
    }


def test_evg_local_service_preserves_upstream_two_way_hsdp(monkeypatch) -> None:
    launched: list[str] = []

    class Service:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def popen(argv, **_kwargs):
        launched.extend(argv)
        return Service()

    monkeypatch.setattr(paidf_native.subprocess, "Popen", popen)
    monkeypatch.setattr(
        paidf_native.urllib.request, "urlopen", lambda *_a, **_k: Response()
    )
    monkeypatch.setattr(
        paidf_native,
        "run_augmentation",
        lambda *_args: {"schema": "npa.paidf.native.evg-augmentation.v1"},
    )

    paidf_native.run_local_augmentation(
        "configs.json",
        "result.json",
        "nvidia/model",
        "deadbeef",
        "image2video",
        8000,
        2,
        "run",
    )

    assert launched == [
        "vllm",
        "serve",
        "nvidia/model",
        "--revision",
        "deadbeef",
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--cfg-parallel-size",
        "2",
        "--use-hsdp",
        "--hsdp-shard-size",
        "2",
        "--init-timeout",
        "1800",
    ]


def test_prepare_images_writes_verified_pane_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (128, 96), "navy").save(source / "person.png")
    output = tmp_path / "prepared"
    manifest_path = tmp_path / "manifest.json"

    result = paidf_native.prepare_images(
        str(source), str(output), str(manifest_path), "unit-run"
    )

    assert result["count"] == 1
    item = result["images"][0]
    assert len(item["sha256"]) == 64
    pane = json.loads(Path(item["pane_metadata_uri"]).read_text(encoding="utf-8"))
    assert pane == {
        "height": 96,
        "image_order": ["input-0000.png"],
        "original_resolutions": [{"height": 96, "width": 128}],
        "widths": [128],
    }


def test_build_configs_mutates_pinned_upstream_protocol_without_replacing_it(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "input_key": "input-0000",
                        "prepared_uri": "s3://example/input.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch(_repository: str, _revision: str, destination: Path) -> Path:
        config = (
            destination
            / "airflow/dags/workflows/image_attribute_augmentation_dag/configs/cosmos_config.yaml"
        )
        config.parent.mkdir(parents=True)
        config.write_text(
            yaml.safe_dump(
                {
                    "data": [{"inputs": {"rgb": "<>"}, "output": {}}],
                    "endpoints": [
                        {"id": "vlm", "role": "vlm", "url": "<>", "model": "<>"},
                        {"id": "llm", "role": "llm", "url": "<>", "model": "<>"},
                        {
                            "id": "image_edit",
                            "role": "image_edit",
                            "url": "<>",
                            "model": "<>",
                            "adapter": "openai.chat.completions",
                        },
                    ],
                    "pipeline": {"retry": 0},
                    "captioning": {
                        "llm": {
                            "text": "upstream prompt {top_outer_color}",
                            "variables": {},
                            "verification_options": {"top_outer_color": ["blue"]},
                        }
                    },
                    "augmentation": {"parameters": {"extra_body": {"seed": None}}},
                    "evaluators": [{"attribute_verification": {"enabled": True}}],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return destination

    monkeypatch.setattr(paidf_native, "_runtime_fetch", fake_fetch)
    output = tmp_path / "out"
    result = paidf_native.build_augmentation_configs(
        "iaa",
        str(prepared),
        str(output),
        str(tmp_path / "configs.json"),
        1,
        7,
        "https://vlm.example/v1",
        "vlm-model",
        "https://llm.example/v1",
        "llm-model",
        "http://127.0.0.1:8000/v1",
        "image-edit-model",
        "unit-run",
    )

    config = yaml.safe_load(Path(result["configs"][0]["config_uri"]).read_text())
    assert config["captioning"]["llm"]["text"] == "upstream prompt {top_outer_color}"
    assert config["captioning"]["llm"]["verification_options"] == {
        "top_outer_color": ["blue"]
    }
    assert config["evaluators"] == [{"attribute_verification": {"enabled": True}}]
    assert config["augmentation"]["parameters"]["extra_body"]["seed"] == 7
    assert config["data"][0]["output"]["metadata"].endswith("output_metadata.json")


def test_evg_finalize_and_terminal_validation_require_published_sidecars(
    tmp_path: Path,
) -> None:
    media = tmp_path / "output.mp4"
    caption = tmp_path / "caption.txt"
    metadata = tmp_path / "metadata.json"
    for path in (media, caption, metadata):
        path.write_text("real-artifact", encoding="utf-8")
    data_path = tmp_path / "auto_labeling/input-0000/0"
    required = (
        "contextual/objects.json",
        "contextual/instances.json",
        "sidecars/captioning/video_captions.json",
        "sidecars/visual_qa_anomaly/items.json",
        "sidecars/visual_qa_per_track/items.json",
        "sidecars/visual_qa_per_track/windows.normalized.json",
    )
    for relative in required:
        target = data_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "accepted": [
                    {
                        "input_key": "input-0000",
                        "augmentation_index": 0,
                        "media_uri": str(media),
                        "caption_uri": str(caption),
                        "metadata_uri": str(metadata),
                        "variables": {"anomaly_type": "person_falling"},
                        "sha256": "a" * 64,
                        "size_bytes": media.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps({"schema": "npa.paidf.upstream.v1"}), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "outputs": [
                    {"key": "input-0000_aug0", "data_path": str(data_path)}
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.json"

    result = paidf_native.finalize_dataset(
        "evg",
        str(validation),
        str(upstream),
        str(labels),
        str(dataset),
        "unit-run",
    )
    report = paidf_native.validate_dataset(
        str(dataset), str(tmp_path / "terminal.json"), "unit-run"
    )

    assert result["validated_artifact_count"] == len(required)
    assert result["trackless_scene_count"] == 1
    assert report["status"] == "passed"
    assert len(report["dataset_manifest_sha256"]) == 64


def test_evg_finalize_fails_closed_when_a_required_sidecar_is_missing(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "accepted": [
                    {
                        "input_key": "input-0000",
                        "augmentation_index": 0,
                        "media_uri": str(tmp_path / "video.mp4"),
                        "caption_uri": str(tmp_path / "caption.txt"),
                        "metadata_uri": str(tmp_path / "metadata.json"),
                        "variables": {},
                        "sha256": "b" * 64,
                        "size_bytes": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text("{}", encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "key": "input-0000_aug0",
                        "data_path": str(tmp_path / "empty-labels"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(paidf_native.PaidfNativeError, match="missing published sidecar"):
        paidf_native.finalize_dataset(
            "evg",
            str(validation),
            str(upstream),
            str(labels),
            str(tmp_path / "dataset.json"),
            "unit-run",
        )
