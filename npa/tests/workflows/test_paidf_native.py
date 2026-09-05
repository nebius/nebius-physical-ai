from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import av
import numpy as np
import pytest
import yaml
from PIL import Image

from npa.workflows import paidf_native
from npa.workflows.paidf_upstream import (
    COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
    COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
    QWEN_IMAGE_EDIT_MODEL,
    QWEN_IMAGE_EDIT_REVISION,
)


TOKEN_FACTORY_ENDPOINT = "https://api.tokenfactory.nebius.com/v1"


def _generation_endpoint(workflow: str) -> dict:
    return {
        "role": "image_edit" if workflow == "iaa" else "image2video",
        "url": "http://127.0.0.1:8000/v1",
        "model": QWEN_IMAGE_EDIT_MODEL if workflow == "iaa" else COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
        "api_key_env": "GENERATION_API_KEY",
    }


@pytest.mark.parametrize("configured", [None, "", paidf_native.RFDETR_BASE_SHA256])
def test_detection_custom_cache_cannot_skip_published_hash(monkeypatch, configured):
    monkeypatch.setenv("RFDETR_MODEL_PATH", "/tmp/custom-checkpoint.pth")
    if configured is None:
        monkeypatch.delenv("RFDETR_MODEL_SHA256", raising=False)
    else:
        monkeypatch.setenv("RFDETR_MODEL_SHA256", configured)

    paidf_native._bind_detection_checkpoint()

    assert os.environ["RFDETR_MODEL_SHA256"] == paidf_native.RFDETR_BASE_SHA256


def test_detection_rejects_a_different_model_digest_before_materialization(monkeypatch):
    monkeypatch.setenv("RFDETR_MODEL_SHA256", "f" * 64)
    monkeypatch.setattr(
        paidf_native, "_read_run_artifact",
        lambda *_args: pytest.fail("must reject unsupported weights before reading input"),
    )
    with pytest.raises(paidf_native.PaidfNativeError, match="published RF-DETR"):
        paidf_native.run_auto_label(
            "evg", "detection", "unused", "unused", "unused",
            TOKEN_FACTORY_ENDPOINT, "vlm", TOKEN_FACTORY_ENDPOINT, "llm", "unit-run",
        )


def _native_identity(kind: str, workflow: str | None = None) -> dict:
    identity = {"schema": f"npa.paidf.native.{kind}.v1", "run_id": "unit-run"}
    if workflow is not None:
        identity["workflow"] = workflow
    return identity


def _upstream_identity(workflow: str) -> dict:
    return {
        "schema": "npa.paidf.upstream.v1",
        "run_id": "unit-run",
        "workflow_variant": {
            "iaa": "image-attribute-augmentation",
            "evg": "event-video-generation",
        }[workflow],
    }


def _write_tiny_video(path: Path, *, frame_count: int = 2) -> Path:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.full((24, 32, 3), index * 80, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_video_validation_decodes_multiple_real_frames(tmp_path: Path) -> None:
    paidf_native._validate_video(_write_tiny_video(tmp_path / "valid.mp4"))


@pytest.mark.parametrize("fixture", ["corrupt", "one-frame", "empty"])
def test_video_validation_rejects_invalid_or_incomplete_mp4(
    tmp_path: Path, fixture: str
) -> None:
    path = tmp_path / f"{fixture}.mp4"
    if fixture == "corrupt":
        path.write_bytes(b"not-an-mp4" * 256)
    elif fixture == "one-frame":
        _write_tiny_video(path, frame_count=1)
    else:
        path.write_bytes(b"")
    with pytest.raises(paidf_native.PaidfNativeError, match="decoded|decode"):
        paidf_native._validate_video(path)


def test_video_validation_rejects_audio_only_media(tmp_path: Path) -> None:
    path = tmp_path / "audio-only.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 800)
    with pytest.raises(paidf_native.PaidfNativeError, match="no video stream"):
        paidf_native._validate_video(path)


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


def test_evg_local_service_preserves_upstream_two_way_hsdp(
    tmp_path: Path, monkeypatch
) -> None:
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
        lambda *_args, **_kwargs: {"schema": "npa.paidf.native.evg-augmentation.v1"},
    )

    config = tmp_path / "service.yaml"
    config.write_text(yaml.safe_dump({"endpoints": [_generation_endpoint("evg")]}))
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {**_native_identity("evg-configs", "evg"), "configs": [{"config_uri": str(config)}]},
    )
    paidf_native.run_local_augmentation(
        str(manifest),
        "result.json",
        COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
        COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
        "image2video",
        8000,
        2,
        "unit-run",
    )

    assert launched == [
        "vllm",
        "serve",
        COSMOS3_SUPER_IMAGE2VIDEO_MODEL,
        "--revision",
        COSMOS3_SUPER_IMAGE2VIDEO_REVISION,
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


@pytest.mark.parametrize(
    ("vlm_url", "llm_url"),
    [
        ("https://credentials.invalid/v1", TOKEN_FACTORY_ENDPOINT),
        (TOKEN_FACTORY_ENDPOINT, "http://api.tokenfactory.nebius.com/v1"),
    ],
)
def test_build_configs_rejects_token_factory_credential_routing_to_other_origin(
    tmp_path: Path, vlm_url: str, llm_url: str
) -> None:
    with pytest.raises(paidf_native.PaidfNativeError, match="approved Token Factory"):
        paidf_native.build_augmentation_configs(
            "iaa",
            str(tmp_path / "prepared.json"),
            str(tmp_path / "output"),
            str(tmp_path / "configs.json"),
            1,
            7,
            vlm_url,
            "vlm-model",
            llm_url,
            "llm-model",
            "http://127.0.0.1:8000/v1",
            QWEN_IMAGE_EDIT_MODEL,
            "unit-run",
        )


@pytest.mark.parametrize(
    ("model", "revision"),
    [
        ("unreviewed/model", COSMOS3_SUPER_IMAGE2VIDEO_REVISION),
        (COSMOS3_SUPER_IMAGE2VIDEO_MODEL, "a" * 40),
    ],
)
def test_local_service_rejects_unreviewed_generation_artifact(
    monkeypatch, model: str, revision: str
) -> None:
    monkeypatch.setattr(
        paidf_native.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("unreviewed model server was launched"),
    )
    with pytest.raises(paidf_native.PaidfNativeError, match="reviewed generation"):
        paidf_native.run_local_augmentation(
            "configs.json",
            "result.json",
            model,
            revision,
            "image2video",
            8000,
            2,
            "run",
        )


def test_augmentation_revalidates_rendered_credential_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "endpoints": [
                    _generation_endpoint("iaa"),
                    {
                        "role": "vlm",
                        "url": "https://credentials.invalid/v1",
                        "api_key_env": "VLM_API_KEY",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                **_native_identity("iaa" + "-configs", "iaa"),
                "workflow": "iaa",
                "configs": [
                    {
                        "config_uri": str(config),
                        "media_uri": str(tmp_path / "output.jpg"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    monkeypatch.setattr(paidf_native.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(
        paidf_native,
        "_run_component",
        lambda *_args, **_kwargs: pytest.fail("cross-origin endpoint was invoked"),
    )

    with pytest.raises(paidf_native.PaidfNativeError, match="approved Token Factory"):
        paidf_native.run_augmentation(
            str(manifest), str(tmp_path / "result.json"), "unit-run"
        )




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
                **_native_identity("prepared-input"),
                "images": [
                    {
                        "input_key": "input-0000",
                        "prepared_uri": "s3://example/input.png",
                    }
                ],
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
        TOKEN_FACTORY_ENDPOINT,
        "vlm-model",
        TOKEN_FACTORY_ENDPOINT,
        "llm-model",
        "http://127.0.0.1:8000/v1",
        QWEN_IMAGE_EDIT_MODEL,
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
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}\n", encoding="utf-8")
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
                **_native_identity("evg-validation", "evg"),
                "accepted": [
                    {
                        "input_key": "input-0000",
                        "augmentation_index": 0,
                        "media_uri": str(media),
                        "caption_uri": str(caption),
                        "metadata_uri": str(metadata),
                        "config_uri": str(config),
                        "variables": {"anomaly_type": "person_falling"},
                        "sha256": paidf_native._sha256(media),
                        "size_bytes": media.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps(_upstream_identity("evg")), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                **_native_identity("evg-auto-label-person-attribute-search", "evg"),
                "outputs": [{"key": "input-0000_aug0", "data_path": str(data_path)}],
            }
        ),
        encoding="utf-8",
    )
    _link_label_fixtures(validation, labels)
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
    scene = Path(result["entries"][0]["scene_path"])
    assert (scene / "raw/video.mp4").read_bytes() == media.read_bytes()
    assert (scene / "sidecars/cosmos/config.yaml").read_bytes() == config.read_bytes()
    assert (
        scene / "sidecars/cosmos/metadata.json"
    ).read_bytes() == metadata.read_bytes()
    assert result["entries"][0]["labels"] == str(scene)
    assert result["metadata"]["total_scenes"] == 1
    assert result["assembled_file_count"] == len(required) + 4
    assert report["status"] == "passed"
    assert len(report["dataset_manifest_sha256"]) == 64



def test_evg_finalize_fails_closed_when_a_required_sidecar_is_missing(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                **_native_identity("evg-validation", "evg"),
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
                ],
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps(_upstream_identity("evg")), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                **_native_identity("evg-auto-label-person-attribute-search", "evg"),
                "outputs": [
                    {
                        "key": "input-0000_aug0",
                        "data_path": str(tmp_path / "empty-labels"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _link_label_fixtures(validation, labels)
    with pytest.raises(
        paidf_native.PaidfNativeError, match="output content manifest"
    ):
        paidf_native.finalize_dataset(
            "evg",
            str(validation),
            str(upstream),
            str(labels),
            str(tmp_path / "dataset.json"),
            "unit-run",
        )



def _write_fixture_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_augmentation_producer(manifest: Path) -> Path:
    """Record the actual fixture files emitted by a completed generation batch."""
    payload = json.loads(manifest.read_text())
    workflow = payload["workflow"]
    outputs = [
        {
            **item,
            "artifacts": paidf_native._artifact_fingerprints({
                field: item[field]
                for field in ("config_uri", "media_uri", "caption_uri", "metadata_uri")
            }),
        }
        for item in payload["configs"]
    ]
    return _write_fixture_json(
        manifest.with_name(manifest.stem + "-generation.json"),
        {
            **_native_identity(f"{workflow}-augmentation", workflow),
            "component": "NVIDIA paidf-augmentation 1.1.0",
            "upstream_revision": paidf_native.PAIDF_AUGMENTATION_REVISION,
            "config_manifest_uri": str(manifest),
            "count": len(outputs),
            "attempted_count": len(outputs),
            "failed_count": 0,
            "failed": [],
            "outputs": outputs,
        },
    )


def _link_validation_fixture(validation: Path) -> None:
    """Give a unit-test validation its complete, content-bound producer history."""
    document = json.loads(validation.read_text())
    workflow = document["workflow"]
    for index, item in enumerate(document["accepted"]):
        media = Path(item["media_uri"])
        if not media.exists():
            media.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (96, 96), "blue").save(media, format="BMP")
        for field, suffix, content in (
            ("config_uri", ".yaml", "evaluators: []\n"),
            ("caption_uri", ".txt", "A person wearing blue."),
            ("metadata_uri", ".json", "{}"),
        ):
            path = Path(item.setdefault(field, str(validation.parent / f"fixture-{index}-{field}{suffix}")))
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        item["sha256"] = paidf_native._sha256(media)
        item["size_bytes"] = media.stat().st_size
    configs = _write_fixture_json(
        validation.with_name(validation.stem + "-configs.json"),
        {
            **_native_identity(f"{workflow}-configs", workflow),
            "configs": document["accepted"],
        },
    )
    producer = _write_augmentation_producer(configs)
    producers = [paidf_native._producer_descriptor(str(producer), json.loads(producer.read_text()))]
    if document["schema"].endswith("iaa-postprocess.v1"):
        checked = _write_fixture_json(
            validation.with_name(validation.stem + "-generation-validation.json"),
            {
                **_native_identity("iaa-validation", "iaa"),
                "accepted": document["accepted"],
                "accepted_count": len(document["accepted"]),
                "skipped": [],
                "skipped_count": 0,
                "producers": producers,
            },
        )
        producers = [*producers, paidf_native._producer_descriptor(str(checked), json.loads(checked.read_text()))]
    document.update(producers=producers, accepted_count=len(document["accepted"]))
    _write_fixture_json(validation, document)


def _link_label_fixtures(validation: Path, labels: Path) -> None:
    """Bind each labeling stage to its predecessor and existing fixture sidecars."""
    _link_validation_fixture(validation)
    checked = json.loads(validation.read_text())
    original = json.loads(labels.read_text())
    workflow = checked["workflow"]
    by_key = {f"{item['input_key']}_aug{item['augmentation_index']}": item for item in checked["accepted"]}
    producers = [*checked["producers"], paidf_native._producer_descriptor(str(validation), checked)]
    stages = (
        {"person-attribute-search": paidf_native.IAA_LABEL_ARTIFACTS}
        if workflow == "iaa" else {
            "detection": ("contextual/objects.json", "contextual/instances.json"),
            "captioning": ("sidecars/captioning/video_captions.json",),
            "visual-qa-anomaly": ("sidecars/visual_qa_anomaly/items.json",),
            "visual-qa-person": ("sidecars/visual_qa_per_track/items.json", "sidecars/visual_qa_per_track/windows.normalized.json"),
            "person-attribute-search": (),
        }
    )
    for stage, required in stages.items():
        outputs = []
        for output in original["outputs"]:
            item = by_key[output["key"]]
            existing = {name: f"{output['data_path']}/{name}" for name in required if Path(output["data_path"], name).is_file()}
            outputs.append({
                **output, "media_uri": item["media_uri"],
                "required_artifacts": list(required),
                "artifacts": paidf_native._artifact_fingerprints(existing),
                "trackless": stage == "person-attribute-search" and not required,
            })
        path = labels if stage == "person-attribute-search" else labels.with_name(f"{stage}.json")
        payload = {
            **_native_identity(f"{workflow}-auto-label-{stage}", workflow),
            "stage": stage, "count": len(outputs), "outputs": outputs,
            "producers": producers, "validation_uri": str(validation),
            "component": "NVIDIA paidf-auto-labeling 1.1.0",
            "upstream_revision": paidf_native.PAIDF_AUTO_LABELING_REVISION,
        }
        _write_fixture_json(path, payload)
        producers = [*producers, paidf_native._producer_descriptor(str(path), payload)]


def _write_iaa_bundles(data_path: Path) -> None:
    config = (
        data_path
        / "sidecars/person_attribute_search/assets/event_and_person_attribute_search_config.yaml"
    )
    config.parent.mkdir(parents=True, exist_ok=True)
    if not config.exists():
        config.write_text("bundle_query_generation: true\n", encoding="utf-8")
    _write_fixture_json(
        data_path / paidf_native.IAA_LABEL_ARTIFACTS[0],
        {
            "chunk_id": "scene",
            "n_people": 1,
            "people": {
                "person_aug0": {"track_id": 0, "attributes": {"top color": "blue"}}
            },
        },
    )
    _write_fixture_json(
        data_path / paidf_native.IAA_LABEL_ARTIFACTS[1],
        {
            "chunk_id": "scene",
            "n_people": 1,
            "people": {
                "person_aug0": {
                    "image_filename": "person.jpg",
                    "queries": {
                        tier: ["person wearing blue"]
                        for tier in ("easy", "medium", "hard")
                    },
                }
            },
        },
    )


@pytest.mark.parametrize("produce_labels", [True, False])
def test_iaa_labeling_consumes_postprocessing_and_stages_query_prompt(
    tmp_path: Path, monkeypatch, produce_labels: bool
) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    attributes = _write_fixture_json(
        tmp_path / "postprocessing/augmented_data.json",
        {
            "entries": [
                {"person_key": "person_aug0", "attributes": {"top_outer_color": "blue"}}
            ]
        },
    )
    validation = _write_fixture_json(
        tmp_path / "validation.json",
        {
            **_native_identity("iaa-postprocess", "iaa"),
            "accepted": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(tmp_path / "split.jpg"),
                    "metadata_uri": str(tmp_path / "output_metadata.json"),
                    "postprocess_dataset_uri": str(attributes),
                }
            ],
        },
    )

    _link_validation_fixture(validation)

    def fetch(_repository, _revision, destination):
        assets = (
            destination
            / "airflow/dags/workflows/image_attribute_augmentation_dag/configs"
        )
        assets.mkdir(parents=True)
        (assets / "event_and_person_attribute_search_config.yaml").write_text(
            "attribute_json: s3://example/placeholder.json\n"
            "query_prompt_file: s3://example/placeholder-prompt.json\n"
            "bundle_query_generation: true\nbundle_query_count: 3\n",
            encoding="utf-8",
        )
        _write_fixture_json(
            assets / "image_attribute_augmentation_synonymous_query_prompt.json",
            {"system_prompt": "published prompt"},
        )
        return destination

    def component(argv, *, env):
        assert env["VLM_API_KEY"] == "test-token"
        assert env["LLM_API_KEY"] == "test-token"
        assert argv[0] == "/app/.venv/bin/main"
        assert argv[argv.index("--attribute-json") + 1] == str(attributes)
        config = yaml.safe_load(Path(argv[argv.index("--config-file") + 1]).read_text())
        assert config["attribute_json"] == str(attributes)
        assert config["llm_endpoint_url"] == TOKEN_FACTORY_ENDPOINT
        assert config["llm_model"] == "llm-model"
        assert config["bundle_query_generation"] is True
        assert config["bundle_query_count"] == 3
        assert json.loads(Path(config["query_prompt_file"]).read_text()) == {
            "system_prompt": "published prompt"
        }
        data_path = Path(json.loads(argv[argv.index("--input") + 1])[0]["data_path"])
        if produce_labels:
            _write_iaa_bundles(data_path)

    monkeypatch.setattr(paidf_native, "_runtime_fetch", fetch)
    monkeypatch.setattr(paidf_native, "_run_component", component)
    args = (
        "iaa",
        "person-attribute-search",
        str(validation),
        str(tmp_path / "labels"),
        str(tmp_path / "result.json"),
        TOKEN_FACTORY_ENDPOINT,
        "vlm-model",
        TOKEN_FACTORY_ENDPOINT,
        "llm-model",
        "unit-run",
    )
    if produce_labels:
        result = paidf_native.run_auto_label(*args)
        assert result["outputs"][0]["required_artifacts"] == list(
            paidf_native.IAA_LABEL_ARTIFACTS
        )
        assert result["outputs"][0]["trackless"] is False
    else:
        with pytest.raises(
            paidf_native.PaidfNativeError, match="required published sidecar"
        ):
            paidf_native.run_auto_label(*args)
        assert not (tmp_path / "result.json").exists()



@pytest.mark.parametrize(
    "defect", ["empty", "count", "identity", "query", "attributes"]
)
def test_iaa_label_bundles_fail_closed_on_invalid_published_content(
    tmp_path: Path, defect: str
) -> None:
    _write_iaa_bundles(tmp_path)
    target = tmp_path / paidf_native.IAA_LABEL_ARTIFACTS[1]
    document = json.loads(target.read_text())
    if defect == "empty":
        document.update(n_people=0, people={})
    elif defect == "count":
        document["n_people"] = 2
    elif defect == "identity":
        document["people"]["another-person"] = document["people"].pop("person_aug0")
    elif defect == "query":
        document["people"]["person_aug0"]["queries"]["hard"] = []
    else:
        target = tmp_path / paidf_native.IAA_LABEL_ARTIFACTS[0]
        document = json.loads(target.read_text())
        document["people"]["person_aug0"].pop("attributes")
    _write_fixture_json(target, document)
    with pytest.raises(paidf_native.PaidfNativeError):
        paidf_native._validate_iaa_labels(str(tmp_path))


def test_iaa_postprocessing_records_actual_split_image_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    original = tmp_path / "cosmos/person/0/output.jpg"
    original.parent.mkdir(parents=True)
    Image.new("RGB", (96, 96), "navy").save(original)
    validation = _write_fixture_json(
        tmp_path / "validation.json",
        {
            **_native_identity("iaa-validation", "iaa"),
            "accepted": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(original),
                    "sha256": paidf_native._sha256(original),
                    "size_bytes": original.stat().st_size,
                }
            ],
        },
    )
    _link_validation_fixture(validation)
    prepared = _write_fixture_json(
        tmp_path / "prepared.json",
        {
            **_native_identity("prepared-input"),
            "images": [
                {
                    "input_key": "person",
                    "pane_metadata_uri": str(tmp_path / "person.json"),
                }
            ],
        },
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    sync_commands = []
    monkeypatch.setattr(
        paidf_native.subprocess,
        "run",
        lambda argv, **_kwargs: sync_commands.append(argv),
    )

    def component(argv, *, env):
        assert env["VLM_API_KEY"] == "test-token"
        assert env["LLM_API_KEY"] == "test-token"
        assert argv[:3] == ["uv", "run", "--project"]
        assert argv[3] == sync_commands[0][3]
        assert argv[4:8] == [
            "--no-sync", "--python", paidf_native.sys.executable, "python"
        ]
        output = Path(argv[argv.index("--output-dir") + 1])
        media = output / "augmented_imgs/person_aug0/person.jpg"
        media.parent.mkdir(parents=True)
        Image.new("RGB", (64, 64), "blue").save(media)
        _write_fixture_json(
            output / "augmented_data.json",
            {
                "entries": [
                    {
                        "person_key": "person_aug0",
                        "attributes": {"top_outer_color": "blue"},
                        "selected_attributes": {"top outer color": "blue"},
                        "queries": {
                            "easy": ["blue jacket"],
                            "medium": ["person wearing blue jacket"],
                            "hard": ["Person wearing blue jacket."],
                        },
                        "attribute_verification": {"passed": True},
                        "images": [
                            f"{output.name}/augmented_imgs/person_aug0/person.jpg"
                        ],
                    }
                ]
            },
        )

    monkeypatch.setattr(paidf_native, "_run_component", component)
    result = paidf_native.postprocess_iaa(
        str(validation),
        str(prepared),
        str(tmp_path / "postprocessing"),
        str(tmp_path / "result.json"),
        TOKEN_FACTORY_ENDPOINT,
        "vlm-model",
        "unit-run",
    )
    item = result["accepted"][0]
    split = Path(item["media_uri"])
    assert item["sha256"] == paidf_native._sha256(split)
    assert item["size_bytes"] == split.stat().st_size
    assert item["sha256"] != paidf_native._sha256(original)
    assert item["generation_sha256"] == paidf_native._sha256(original)
    assert item["generation_media_uri"] == str(original)
    assert len(sync_commands) == 1
    assert sync_commands[0][:3] == ["uv", "sync", "--project"]
    assert sync_commands[0][4:] == [
        "--frozen", "--python", paidf_native.sys.executable
    ]



@pytest.mark.parametrize("workflow", ["iaa", "evg"])
def test_augmentation_batch_preserves_workflow_specific_partial_failure_policy(
    tmp_path: Path, monkeypatch, workflow: str
) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    configs = []
    for index in range(2):
        config = tmp_path / f"config-{index}.yaml"
        config.write_text(yaml.safe_dump({"index": index, "endpoints": [_generation_endpoint(workflow)]}))
        configs.append(
            {
                "config_uri": str(config),
                "media_uri": str(tmp_path / f"output-{index}.bmp"),
                "caption_uri": str(tmp_path / f"caption-{index}.txt"),
                "metadata_uri": str(tmp_path / f"metadata-{index}.json"),
                "input_key": "person",
                "augmentation_index": index,
            }
        )
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {
            **_native_identity(workflow + "-configs", workflow),
            "workflow": workflow,
            "configs": configs,
        },
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    sync_commands = []
    monkeypatch.setattr(
        paidf_native.subprocess,
        "run",
        lambda argv, **_kwargs: sync_commands.append(argv),
    )
    attempted = []

    def component(argv, *, env):
        assert env["VLM_API_KEY"] == "test-token"
        assert env["LLM_API_KEY"] == "test-token"
        assert env["GENERATION_API_KEY"] == "local"
        assert argv[:3] == ["uv", "run", "--project"]
        assert argv[3] == sync_commands[0][3]
        assert argv[4:8] == [
            "--no-sync", "--python", paidf_native.sys.executable, "python"
        ]
        index = yaml.safe_load(Path(argv[-1]).read_text())["index"]
        attempted.append(index)
        if index == 0:
            raise subprocess.CalledProcessError(1, argv)
        Image.new("RGB", (96, 96), "blue").save(configs[index]["media_uri"])
        Path(configs[index]["caption_uri"]).write_text("A person wearing blue.")
        _write_fixture_json(Path(configs[index]["metadata_uri"]), {"attribute_verification": {"passed": True}})

    monkeypatch.setattr(paidf_native, "_run_component", component)
    if workflow == "evg":
        with pytest.raises(subprocess.CalledProcessError):
            paidf_native.run_augmentation(
                str(manifest), str(tmp_path / "result.json"), "unit-run"
            )
        assert attempted == [0]
    else:
        result = paidf_native.run_augmentation(
            str(manifest), str(tmp_path / "result.json"), "unit-run"
        )
        assert attempted == [0, 1]
        assert result["count"] == 1
        assert result["attempted_count"] == 2
        assert result["failed_count"] == 1
        assert result["failed"][0]["exit_code"] == 1
    assert len(sync_commands) == 1
    assert sync_commands[0][:3] == ["uv", "sync", "--project"]
    assert sync_commands[0][4:] == [
        "--frozen", "--python", paidf_native.sys.executable
    ]




@pytest.mark.parametrize("workflow", ["iaa", "evg"])
def test_output_validation_rejects_explicit_evaluator_failure(
    tmp_path: Path, workflow: str
) -> None:
    configs = []
    for index, passed in enumerate((False, True)):
        media = tmp_path / f"image-{index}.bmp"
        Image.new("RGB", (96, 96), "blue").save(media)
        caption = tmp_path / f"caption-{index}.txt"
        caption.write_text("A person wearing blue.")
        metadata = _write_fixture_json(
            tmp_path / f"metadata-{index}.json",
            {
                "attribute_verification": {"passed": passed},
            },
        )
        config = tmp_path / f"config-{index}.yaml"
        config.write_text(
            "evaluators:\n  - attribute_verification:\n      enabled: true\n",
            encoding="utf-8",
        )
        configs.append(
            {
                "input_key": "person",
                "augmentation_index": index,
                "media_uri": str(media),
                "caption_uri": str(caption),
                "metadata_uri": str(metadata),
                "config_uri": str(config),
            }
        )
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {
            **_native_identity(workflow + "-configs", workflow),
            "workflow": workflow,
            "configs": configs,
        },
    )
    producer = _write_augmentation_producer(manifest)
    if workflow == "evg":
        with pytest.raises(
            paidf_native.PaidfNativeError, match="requires every expected"
        ):
            paidf_native.validate_augmentation(
                str(manifest), str(tmp_path / "result.json"), "unit-run", str(producer)
            )
    else:
        result = paidf_native.validate_augmentation(
            str(manifest), str(tmp_path / "result.json"), "unit-run", str(producer)
        )
        assert result["accepted_count"] == 1
        assert result["accepted"][0]["augmentation_index"] == 1
        assert result["skipped_count"] == 1
        assert (
            result["skipped"][0]["reason"]
            == "attribute_verification did not affirmatively pass"
        )



def test_all_failed_iaa_batch_records_failure_without_promoting_empty_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"endpoints": [_generation_endpoint("iaa")]}))
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {
            **_native_identity("iaa" + "-configs", "iaa"),
            "workflow": "iaa",
            "configs": [
                {
                    "config_uri": str(config),
                    "media_uri": "output.jpg",
                    "input_key": "person",
                    "augmentation_index": 0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    monkeypatch.setattr(paidf_native.subprocess, "run", lambda *_a, **_k: None)

    def component(argv, *, env):
        assert env["VLM_API_KEY"] == "test-token"
        assert env["LLM_API_KEY"] == "test-token"
        assert env["GENERATION_API_KEY"] == "local"
        raise subprocess.CalledProcessError(7, argv)

    monkeypatch.setattr(paidf_native, "_run_component", component)
    report = tmp_path / "result.json"
    with pytest.raises(paidf_native.PaidfNativeError, match="every mapped"):
        paidf_native.run_augmentation(str(manifest), str(report), "unit-run")
    failed = json.loads(report.read_text())
    assert failed["count"] == 0
    assert failed["outputs"] == []
    assert failed["failed_count"] == 1
    assert failed["failed"][0]["exit_code"] == 7




def test_iaa_terminal_validation_reopens_bundles_after_dataset_assembly(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "labels"
    _write_iaa_bundles(data_path)
    media = tmp_path / "split/person.jpg"
    media.parent.mkdir()
    Image.new("RGB", (64, 64), "blue").save(media)
    caption = tmp_path / "caption.txt"
    caption.write_text("A person wearing blue.")
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}\n", encoding="utf-8")
    metadata = _write_fixture_json(
        tmp_path / "metadata.json",
        {
            "attribute_verification": {"passed": True},
        },
    )
    postprocess = _write_fixture_json(tmp_path / "attributes.json", {"entries": [{}]})
    validation = _write_fixture_json(
        tmp_path / "validation.json",
        {
            **_native_identity("iaa-postprocess", "iaa"),
            "accepted": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(media),
                    "caption_uri": str(caption),
                    "metadata_uri": str(metadata),
                    "config_uri": str(config),
                    "variables": {"top_outer_color": "blue"},
                    "sha256": paidf_native._sha256(media),
                    "size_bytes": media.stat().st_size,
                    "postprocess_dataset_uri": str(postprocess),
                    "generation_media_uri": str(media),
                    "generation_sha256": paidf_native._sha256(media),
                    "selected_attributes": {"top outer color": "blue"},
                    "queries": {
                        "easy": ["blue jacket"],
                        "medium": ["person wearing blue jacket"],
                        "hard": ["Person wearing blue jacket."],
                    },
                    "attribute_verification": {"passed": True},
                }
            ],
        },
    )
    upstream = _write_fixture_json(
        tmp_path / "upstream.json", _upstream_identity("iaa")
    )
    labels = _write_fixture_json(
        tmp_path / "labels.json",
        {
            **_native_identity("iaa-auto-label-person-attribute-search", "iaa"),
            "outputs": [
                {
                    "key": "person_aug0",
                    "data_path": str(data_path),
                }
            ],
        },
    )
    _link_label_fixtures(validation, labels)
    dataset = tmp_path / "dataset.json"
    assembled = paidf_native.finalize_dataset(
        "iaa",
        str(validation),
        str(upstream),
        str(labels),
        str(dataset),
        "unit-run",
    )
    assert assembled["validated_artifact_count"] == 2
    entry = assembled["entries"][0]
    scene = Path(entry["scene_path"])
    assert entry["postprocess_dataset"] == str(scene / "sidecars/augmented_data.json")
    assert (
        scene / "sidecars/augmented_data.json"
    ).read_bytes() == postprocess.read_bytes()
    assert (scene / "raw/person.jpg").read_bytes() == media.read_bytes()
    assert (tmp_path / "augmented_data.json").is_file()
    assert entry["person_key"] == "person_aug0"
    assert entry["auto_labeling_source_path"] == str(data_path)
    report = tmp_path / "terminal.json"
    result = paidf_native.validate_dataset(str(dataset), str(report), "unit-run")
    assert result["status"] == "passed"
    report.unlink()
    # A config/prompt may remain present, but loss of model-produced queries
    # between assembly and the terminal gate must still fail the workflow.
    (scene / paidf_native.IAA_LABEL_ARTIFACTS[1]).unlink()
    with pytest.raises(paidf_native.PaidfNativeError, match="input does not exist"):
        paidf_native.validate_dataset(str(dataset), str(report), "unit-run")
    assert not report.exists()



def test_native_reports_record_executed_image_without_polluting_upstream_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    image = "registry.example.test/runtime@sha256:" + "a" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    native = paidf_native._write_json(
        {**_native_identity("iaa-augmentation", "iaa")},
        str(tmp_path / "native.json"),
    )
    assert native["runtime_image"] == image
    protocol = paidf_native._write_json(
        {"images": ["image.jpg"]}, str(tmp_path / "protocol.json")
    )
    assert protocol == {"images": ["image.jpg"]}


def _write_dig_runtime_cache(pretrained: Path) -> None:
    revisions = paidf_native._dig_model_revisions()
    for repository, probe in paidf_native.DIG_RUNTIME_CACHE_PROBES.items():
        target = (
            pretrained
            / "hf"
            / f"models--{repository.replace('/', '--')}"
            / "snapshots"
            / revisions[repository]
            / probe
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if probe == "blocklist":
            target.mkdir(exist_ok=True)
            (target / "words.txt").write_text("synthetic-fixture\n", encoding="utf-8")
        else:
            target.write_text("synthetic-fixture", encoding="utf-8")


def test_dig_runtime_uses_only_verified_preflight_pinned_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _write_dig_runtime_cache(tmp_path)
    manifest = paidf_native._dig_cache_manifest(tmp_path, "unit-run", initialize=True)
    monkeypatch.setenv("HF_HUB_CACHE", "/unrelated/cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("CKPT_DIR", "/unrelated/checkpoints")
    monkeypatch.setenv(
        "PATH",
        "/tmp/npa-shim:/opt/npa-venv/bin:/usr/local/bin:/opt/venv/bin:/usr/bin",
    )
    monkeypatch.setenv(
        "PYTHONPATH", "/tmp/npa-src-overlay/src:/workspace/paidf-anomalygen"
    )
    env = paidf_native._dig_offline_environment(tmp_path, "unit-run")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hf")
    assert env["CKPT_DIR"] == str(tmp_path)
    assert env["VIRTUAL_ENV"] == "/opt/venv"
    assert env["PATH"].split(":") == [
        "/opt/venv/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]
    assert env["PYTHONPATH"] == (
        "/tmp/npa-src-overlay/src:/workspace/paidf-anomalygen"
    )
    assert env["UV_PYTHON"] == "/opt/venv/bin/python"
    assert len(manifest["models"]) == 4
    assert all(len(model["revision"]) == 40 for model in manifest["models"])


@pytest.mark.parametrize(
    "defect", ["missing_snapshot", "changed_ref", "changed_manifest"]
)
def test_dig_runtime_refuses_missing_or_drifted_cache(
    tmp_path: Path, defect: str
) -> None:
    _write_dig_runtime_cache(tmp_path)
    manifest = paidf_native._dig_cache_manifest(tmp_path, "unit-run", initialize=True)
    cache = tmp_path / "hf/models--Qwen--Qwen3Guard-Gen-0.6B"
    if defect == "missing_snapshot":
        revision = paidf_native._dig_model_revisions()["Qwen/Qwen3Guard-Gen-0.6B"]
        (cache / "snapshots" / revision / "model.safetensors").unlink()
    elif defect == "changed_ref":
        (cache / "refs/main").write_text("b" * 40)
    else:
        manifest["models"][0]["revision"] = "c" * 40
        _write_fixture_json(tmp_path / "runtime-hf-snapshots.json", manifest)
    with pytest.raises(paidf_native.PaidfNativeError, match="cache"):
        paidf_native._dig_offline_environment(tmp_path, "unit-run")


def test_dig_training_and_inference_children_use_the_vendor_environment(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "physical-ai-data-factory"
    dataset = tmp_path / "dataset"
    pretrained = tmp_path / "pretrained"
    checkpoint = tmp_path / "checkpoint"
    for directory in (source, dataset, pretrained, checkpoint):
        directory.mkdir()
    (dataset / "defect_spec.jsonl").write_text("{}\n", encoding="utf-8")
    selected = checkpoint / "training/model/checkpoint-10.pt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"checkpoint")
    real_path = Path
    monkeypatch.setattr(
        paidf_native,
        "Path",
        lambda path: (
            workspace if str(path) == "/workspace/paidf-anomalygen" else real_path(path)
        ),
    )
    materialized = {
        "dataset": dataset,
        "pretrained": pretrained,
        "checkpoint": checkpoint,
    }
    monkeypatch.setattr(
        paidf_native,
        "_materialize",
        lambda uri, _target: materialized[uri],
    )
    monkeypatch.setattr(
        paidf_native,
        "_dig_pretrained_content_manifest",
        lambda *_args, **_kwargs: ({}, "a" * 64),
    )
    monkeypatch.setattr(
        paidf_native, "_dig_cache_manifest", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(paidf_native, "_runtime_fetch", lambda *_args: source)
    monkeypatch.setattr(paidf_native, "_publish", lambda *_args: None)
    monkeypatch.setattr(
        paidf_native,
        "_verify_dig_finetune_handoff",
        lambda *_args: (
            {
                "selected_checkpoint": "training/model/checkpoint-10.pt",
                "selected_checkpoint_sha256": "b" * 64,
            },
            selected,
        ),
    )
    monkeypatch.setenv(
        "PATH",
        "/tmp/npa-shim:/opt/npa-venv/bin:/usr/local/bin:/opt/venv/bin:/usr/bin",
    )
    monkeypatch.setenv("PYTHONPATH", "/workspace/paidf-anomalygen")
    child_envs = []

    def component(_argv, **kwargs):
        env = kwargs["env"]
        child_envs.append(env)
        if "TRAIN_OUTPUT" in env:
            pointer = Path(env["TRAIN_OUTPUT"]) / "training/best_checkpoint.txt"
            pointer.parent.mkdir(parents=True)
            (pointer.parent / "model").mkdir()
            (pointer.parent / "model/checkpoint-10.pt").write_bytes(b"checkpoint")
            pointer.write_text("checkpoint-10.pt", encoding="utf-8")
        if "OUTPUT_DIR" in env:
            generated = Path(env["OUTPUT_DIR"])
            (generated / "reconstructed_image").mkdir(parents=True)
            (generated / "reconstructed_image/image.png").write_bytes(b"image")
            (generated / "pseudo_labels").mkdir()
            (generated / "pseudo_labels/coco_annotations.json").write_text(
                "{}", encoding="utf-8"
            )

    monkeypatch.setattr(paidf_native, "_run_component", component)

    paidf_native.run_dig_train(
        "dataset",
        "pretrained",
        str(tmp_path / "finetune"),
        str(tmp_path / "train-result.json"),
        "pcb",
        "unit-run",
    )
    paidf_native.run_dig_inference(
        "dataset",
        "pretrained",
        "checkpoint",
        str(tmp_path / "finetune-result.json"),
        str(tmp_path / "generated"),
        str(tmp_path / "infer-result.json"),
        1,
        "unit-run",
    )

    assert len(child_envs) == 2
    for env in child_envs:
        assert env["VIRTUAL_ENV"] == "/opt/venv"
        assert env["UV_PYTHON"] == "/opt/venv/bin/python"
        assert env["PATH"].split(":") == [
            "/opt/venv/bin",
            "/usr/local/bin",
            "/usr/bin",
        ]
        assert env["PYTHONPATH"] == "/workspace/paidf-anomalygen"


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_dig_pretrained_content_manifest_covers_the_complete_published_tree(
    tmp_path: Path, mutation: str
) -> None:
    _write_dig_runtime_cache(tmp_path)
    paidf_native._dig_cache_manifest(tmp_path, "unit-run", initialize=True)
    converted = tmp_path / "Cosmos3-Nano/model/weights.bin"
    converted.parent.mkdir(parents=True)
    converted.write_bytes(b"reviewed-converted-checkpoint")
    manifest, manifest_sha256 = paidf_native._dig_pretrained_content_manifest(
        tmp_path, "unit-run", initialize=True
    )

    assert manifest["file_count"] == len(manifest["files"])
    assert len(manifest_sha256) == 64
    assert {
        record["path"] for record in manifest["files"]
    } >= {"runtime-hf-snapshots.json", "Cosmos3-Nano/model/weights.bin"}
    paidf_native._dig_pretrained_content_manifest(tmp_path, "unit-run")

    if mutation == "changed":
        converted.write_bytes(b"different-checkpoint-bytes")
    elif mutation == "added":
        (tmp_path / "unrecorded.bin").write_bytes(b"unrecorded")
    else:
        converted.unlink()
    with pytest.raises(paidf_native.PaidfNativeError, match="complete hash manifest"):
        paidf_native._dig_pretrained_content_manifest(tmp_path, "unit-run")


def _write_dig_finetune_handoff(
    checkpoint_root: Path, result_path: Path, checkpoint_uri: str
) -> tuple[dict, Path]:
    selected = checkpoint_root / "training/model/checkpoint-10.pt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"selected-model-weights")
    pointer = checkpoint_root / "training/best_checkpoint.txt"
    pointer.write_text(selected.name, encoding="utf-8")
    payload = {
        **_native_identity("dig-finetune", "dig"),
        "status": "completed",
        "selected_checkpoint": selected.relative_to(checkpoint_root).as_posix(),
        "selected_checkpoint_sha256": paidf_native._sha256(selected),
        "output_uri": checkpoint_uri,
    }
    _write_fixture_json(checkpoint_root / "npa-finetune.json", payload)
    _write_fixture_json(result_path, payload)
    return payload, selected


def test_dig_finetune_handoff_binds_external_result_pointer_and_selected_bytes(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "finetune"
    result_path = tmp_path / "finetune-result.json"
    checkpoint_uri = str(checkpoint_root)
    payload, selected = _write_dig_finetune_handoff(
        checkpoint_root, result_path, checkpoint_uri
    )

    verified, verified_selected = paidf_native._verify_dig_finetune_handoff(
        checkpoint_root, checkpoint_uri, str(result_path), "unit-run"
    )

    assert verified == payload
    assert verified_selected == selected


@pytest.mark.parametrize(
    "mutation", ["selected_bytes", "best_pointer", "external_result", "selected_path"]
)
def test_dig_finetune_handoff_fails_closed_on_identity_or_content_drift(
    tmp_path: Path, mutation: str
) -> None:
    checkpoint_root = tmp_path / "finetune"
    result_path = tmp_path / "finetune-result.json"
    checkpoint_uri = str(checkpoint_root)
    payload, selected = _write_dig_finetune_handoff(
        checkpoint_root, result_path, checkpoint_uri
    )
    if mutation == "selected_bytes":
        selected.write_bytes(b"mutated-model-weights")
        expected = "content hash"
    elif mutation == "best_pointer":
        other = selected.with_name("checkpoint-20.pt")
        other.write_bytes(b"other-model-weights")
        (checkpoint_root / "training/best_checkpoint.txt").write_text(
            other.name, encoding="utf-8"
        )
        expected = "best-checkpoint pointer"
    elif mutation == "external_result":
        payload["selected_checkpoint_sha256"] = "f" * 64
        _write_fixture_json(result_path, payload)
        expected = "embedded checkpoint record"
    else:
        payload["selected_checkpoint"] = "../outside.pt"
        _write_fixture_json(checkpoint_root / "npa-finetune.json", payload)
        _write_fixture_json(result_path, payload)
        expected = "identity"

    with pytest.raises(paidf_native.PaidfNativeError, match=expected):
        paidf_native._verify_dig_finetune_handoff(
            checkpoint_root, checkpoint_uri, str(result_path), "unit-run"
        )


def test_dig_inference_toolref_consumes_the_finetune_result_without_step_override() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    argv = TOOL_CATALOG["workflow.paidf.dig_infer"].argv_template
    assert "--finetune-result-uri" in argv
    assert "{{config.finetune_result_uri}}" in argv
    assert "--checkpoint-step" not in argv
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "workflows/workbench/npa-workflows/paidf-defect-image-generation.yaml"
        ).read_text(encoding="utf-8")
    )
    assert "finetune_result_uri" in workflow["config"]
    assert "checkpoint_step" not in workflow["config"]


def test_dig_preparation_pins_real_converter_and_original_downloader(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    script = workspace / "scripts/download_checkpoints.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    converted_manifest = workspace / "assets/checkpoint_manifest_converted.sha256"
    converted_manifest.parent.mkdir()
    converted_manifest.write_text("synthetic-fixture", encoding="utf-8")
    real_path = Path
    monkeypatch.setattr(
        paidf_native,
        "Path",
        lambda path: (
            workspace if str(path) == "/workspace/paidf-anomalygen" else real_path(path)
        ),
    )
    calls = []

    def component(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        if argv[0] == "uvx" and "--cache-dir" in argv:
            _write_dig_runtime_cache(Path(argv[argv.index("--cache-dir") + 1]).parent)
        elif argv[0] == "uvx":
            target = Path(argv[argv.index("--local-dir") + 1])
            target.mkdir(parents=True)
            _write_fixture_json(target / "config.json", {})
        elif argv[0] == "python":
            Path(argv[argv.index("-o") + 1]).mkdir(parents=True)
        elif argv[0] == "bash":
            target = Path(kwargs["env"]["CKPT_DIR"]) / "wan2pt2/Wan2.2_VAE.pth"
            target.parent.mkdir()
            target.write_text("synthetic-fixture", encoding="utf-8")

    monkeypatch.setattr(paidf_native, "_run_component", component)
    monkeypatch.setenv(
        "PATH",
        "/tmp/npa-shim:/opt/npa-venv/bin:/usr/local/bin:/opt/venv/bin:/usr/bin",
    )
    monkeypatch.setenv("PYTHONPATH", "/workspace/paidf-anomalygen")
    output = tmp_path / "published"
    result = paidf_native.prepare_dig_pretrained(
        str(output), str(tmp_path / "result.json"), "unit-run"
    )
    revisions = paidf_native._dig_model_revisions()
    downloads = [argv for argv, _env in calls if argv[0] == "uvx"]
    assert len(downloads) == 6
    assert calls
    for _argv, env in calls:
        assert env["VIRTUAL_ENV"] == "/opt/venv"
        assert env["UV_PYTHON"] == "/opt/venv/bin/python"
        assert env["PATH"].split(":") == [
            "/opt/venv/bin",
            "/usr/local/bin",
            "/usr/bin",
        ]
        assert env["PYTHONPATH"] == "/workspace/paidf-anomalygen"
    for argv in downloads:
        assert argv[1] == "hf==1.26.0"
        assert argv[argv.index("--revision") + 1] == revisions[argv[3]]
    converters = [(argv, env) for argv, env in calls if argv[0] == "python"]
    assert len(converters) == 2
    for argv, env in converters:
        assert argv[2] == "cosmos_framework.scripts.convert_model_to_dcp"
        assert argv[argv.index("--checkpoint-path") + 1] not in {
            "Cosmos3-Nano",
            "Cosmos3-Edge",
        }
        assert env["HF_HUB_OFFLINE"] == "1"
    upstream_env = next(env for argv, env in calls if argv[0] == "bash")
    assert upstream_env["EDGE_VLM_REV"] == revisions["nvidia/Cosmos3-Edge"]
    assert upstream_env["GUARDRAIL_REV"] == revisions["nvidia/Cosmos-Guardrail1"]
    assert (output / "runtime-hf-snapshots.json").is_file()
    assert (output / paidf_native.DIG_PRETRAINED_CONTENT_MANIFEST).is_file()
    assert len(result["content_manifest_sha256"]) == 64
    assert result["status"] == "completed"


def test_iaa_generation_service_binds_only_to_loopback(
    tmp_path: Path, monkeypatch
) -> None:
    launched = []

    def inspect_launch(argv, **_kwargs):
        launched.extend(argv)
        raise RuntimeError("inspected launch")

    monkeypatch.setattr(paidf_native.subprocess, "Popen", inspect_launch)
    config = tmp_path / "service.yaml"
    config.write_text(yaml.safe_dump({"endpoints": [_generation_endpoint("iaa")]}))
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {**_native_identity("iaa-configs", "iaa"), "configs": [{"config_uri": str(config)}]},
    )
    with pytest.raises(RuntimeError, match="inspected launch"):
        paidf_native.run_local_augmentation(
            str(manifest),
            "result.json",
            QWEN_IMAGE_EDIT_MODEL,
            QWEN_IMAGE_EDIT_REVISION,
            "image-edit",
            8000,
            1,
            "unit-run",
        )
    assert launched[launched.index("--host") + 1] == "127.0.0.1"


@pytest.mark.parametrize("mismatch", ["schema", "run_id", "workflow"])
def test_generation_rejects_foreign_manifest_before_model_server_startup(
    tmp_path: Path, monkeypatch, mismatch: str
) -> None:
    payload = _native_identity("iaa-configs", "iaa")
    if mismatch == "workflow":
        payload = _native_identity("evg-configs", "evg")
    else:
        payload[mismatch] = "foreign-artifact"
    manifest = _write_fixture_json(tmp_path / "configs.json", payload)
    monkeypatch.setattr(
        paidf_native.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "foreign manifest started a model server"
        ),
    )
    with pytest.raises(paidf_native.PaidfNativeError, match="identity|workflow"):
        paidf_native.run_local_augmentation(
            str(manifest),
            str(tmp_path / "result.json"),
            QWEN_IMAGE_EDIT_MODEL,
            QWEN_IMAGE_EDIT_REVISION,
            "image-edit",
            8000,
            1,
            "unit-run",
        )


@pytest.mark.parametrize(
    ("enabled", "verdict", "accepted"),
    [
        (True, None, False),
        (True, {}, False),
        (True, [], False),
        (True, {"passed": 1}, False),
        (True, {"passed": "true"}, False),
        (True, {"passed": False}, False),
        (True, {"passed": True}, True),
        (False, None, True),
    ],
)
def test_enabled_attribute_verification_requires_an_affirmative_boolean_verdict(
    tmp_path: Path, enabled: bool, verdict, accepted: bool
) -> None:
    media = tmp_path / "image.bmp"
    Image.new("RGB", (96, 96), "blue").save(media)
    caption = tmp_path / "caption.txt"
    caption.write_text("A person wearing blue.")
    metadata = _write_fixture_json(
        tmp_path / "metadata.json",
        {} if verdict is None else {"attribute_verification": verdict},
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {"evaluators": [{"attribute_verification": {"enabled": enabled}}]}
        )
    )
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {
            **_native_identity("iaa-configs", "iaa"),
            "configs": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(media),
                    "caption_uri": str(caption),
                    "metadata_uri": str(metadata),
                    "config_uri": str(config),
                }
            ],
        },
    )
    producer = _write_augmentation_producer(manifest)
    if accepted:
        result = paidf_native.validate_augmentation(
            str(manifest), str(tmp_path / "result.json"), "unit-run", str(producer)
        )
        assert result["accepted_count"] == 1
    else:
        with pytest.raises(paidf_native.PaidfNativeError, match="failed validation"):
            paidf_native.validate_augmentation(
                str(manifest), str(tmp_path / "result.json"), "unit-run", str(producer)
            )
        assert not (tmp_path / "result.json").exists()



@pytest.mark.parametrize(
    "consumer",
    ["configs", "augment", "validate", "postprocess", "label", "finalize", "terminal"],
)
@pytest.mark.parametrize("mismatch", ["schema", "run_id", "workflow"])
def test_native_consumers_reject_foreign_artifact_identity_before_execution(
    tmp_path: Path, monkeypatch, consumer: str, mismatch: str
) -> None:
    kind = {
        "configs": "prepared-input",
        "augment": "iaa-configs",
        "validate": "iaa-configs",
        "postprocess": "iaa-validation",
        "label": "iaa-postprocess",
        "finalize": "iaa-postprocess",
        "terminal": "iaa-dataset",
    }[consumer]
    if consumer == "configs" and mismatch == "workflow":
        pytest.skip("Image preparation is a shared protocol; schema and run bind it.")
    payload = _native_identity(kind, None if consumer == "configs" else "iaa")
    payload[mismatch] = "evg" if mismatch == "workflow" else "foreign-artifact"
    path = str(_write_fixture_json(tmp_path / "input.json", payload))
    output = str(tmp_path / "output.json")
    for name in ("_runtime_fetch", "_run_component", "_publish"):
        monkeypatch.setattr(
            paidf_native,
            name,
            lambda *_a, **_k: pytest.fail("foreign artifact was executed or published"),
        )
    calls = {
        "configs": lambda: paidf_native.build_augmentation_configs(
            "iaa",
            path,
            output,
            output,
            1,
            42,
            TOKEN_FACTORY_ENDPOINT,
            "vlm",
            TOKEN_FACTORY_ENDPOINT,
            "llm",
            "http://127.0.0.1:8000/v1",
            QWEN_IMAGE_EDIT_MODEL,
            "unit-run",
        ),
        "augment": lambda: paidf_native.run_augmentation(path, output, "unit-run"),
        "validate": lambda: paidf_native.validate_augmentation(
            path, output, "unit-run", path
        ),
        "postprocess": lambda: paidf_native.postprocess_iaa(
            path, path, output, output, TOKEN_FACTORY_ENDPOINT, "vlm", "unit-run"
        ),
        "label": lambda: paidf_native.run_auto_label(
            "iaa",
            "person-attribute-search",
            path,
            output,
            output,
            TOKEN_FACTORY_ENDPOINT,
            "vlm",
            TOKEN_FACTORY_ENDPOINT,
            "llm",
            "unit-run",
        ),
        "finalize": lambda: paidf_native.finalize_dataset(
            "iaa", path, path, path, output, "unit-run"
        ),
        "terminal": lambda: paidf_native.validate_dataset(path, output, "unit-run"),
    }
    with pytest.raises(paidf_native.PaidfNativeError, match="identity|workflow"):
        calls[consumer]()
    assert not Path(output).exists()



@pytest.mark.parametrize("artifact", ["upstream", "labels"])
@pytest.mark.parametrize("mismatch", ["schema", "run_id", "workflow"])
def test_dataset_join_rejects_foreign_upstream_or_label_identity(
    tmp_path: Path, artifact: str, mismatch: str
) -> None:
    validation = _write_fixture_json(
        tmp_path / "validation.json",
        {**_native_identity("iaa-postprocess", "iaa"), "accepted": [{}]},
    )
    upstream_payload = _upstream_identity("iaa")
    labels_payload = _native_identity("iaa-auto-label-person-attribute-search", "iaa")
    target = upstream_payload if artifact == "upstream" else labels_payload
    field = (
        "workflow_variant"
        if artifact == "upstream" and mismatch == "workflow"
        else mismatch
    )
    target[field] = (
        "event-video-generation" if field == "workflow_variant" else "foreign-artifact"
    )
    upstream = _write_fixture_json(tmp_path / "upstream.json", upstream_payload)
    labels = _write_fixture_json(tmp_path / "labels.json", labels_payload)
    with pytest.raises(paidf_native.PaidfNativeError, match="identity|workflow"):
        paidf_native.finalize_dataset(
            "iaa",
            str(validation),
            str(upstream),
            str(labels),
            str(tmp_path / "dataset.json"),
            "unit-run",
        )


def test_dig_runtime_cache_cannot_be_relabelled_as_another_run(tmp_path: Path) -> None:
    _write_dig_runtime_cache(tmp_path)
    paidf_native._dig_cache_manifest(tmp_path, "prior-run", initialize=True)
    with pytest.raises(paidf_native.PaidfNativeError, match="provenance"):
        paidf_native._dig_offline_environment(tmp_path, "unit-run")
