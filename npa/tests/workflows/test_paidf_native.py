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
                ]
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps({"schema": "npa.paidf.upstream.v1"}), encoding="utf-8"
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {"outputs": [{"key": "input-0000_aug0", "data_path": str(data_path)}]}
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

    with pytest.raises(
        paidf_native.PaidfNativeError, match="missing published sidecar"
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
            "accepted": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(tmp_path / "split.jpg"),
                    "metadata_uri": str(tmp_path / "output_metadata.json"),
                    "postprocess_dataset_uri": str(attributes),
                }
            ]
        },
    )

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

    def component(argv):
        assert argv[argv.index("--attribute-json") + 1] == str(attributes)
        config = yaml.safe_load(Path(argv[argv.index("--config-file") + 1]).read_text())
        assert config["attribute_json"] == str(attributes)
        assert config["llm_endpoint_url"] == "https://llm.example/v1"
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
        "https://vlm.example/v1",
        "vlm-model",
        "https://llm.example/v1",
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
            "accepted": [
                {
                    "input_key": "person",
                    "augmentation_index": 0,
                    "media_uri": str(original),
                    "sha256": paidf_native._sha256(original),
                    "size_bytes": original.stat().st_size,
                }
            ]
        },
    )
    prepared = _write_fixture_json(
        tmp_path / "prepared.json",
        {
            "images": [
                {
                    "input_key": "person",
                    "pane_metadata_uri": str(tmp_path / "person.json"),
                }
            ]
        },
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    monkeypatch.setattr(paidf_native.subprocess, "run", lambda *_a, **_k: None)

    def component(argv):
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
        "https://vlm.example/v1",
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


@pytest.mark.parametrize("workflow", ["iaa", "evg"])
def test_augmentation_batch_preserves_workflow_specific_partial_failure_policy(
    tmp_path: Path, monkeypatch, workflow: str
) -> None:
    configs = []
    for index in range(2):
        config = tmp_path / f"config-{index}.yaml"
        config.write_text(f"index: {index}\n")
        configs.append(
            {
                "config_uri": str(config),
                "media_uri": "output.jpg",
                "input_key": "person",
                "augmentation_index": index,
            }
        )
    manifest = _write_fixture_json(
        tmp_path / "configs.json", {"workflow": workflow, "configs": configs}
    )
    monkeypatch.setattr(
        paidf_native, "_runtime_fetch", lambda _r, _v, destination: destination
    )
    monkeypatch.setattr(paidf_native.subprocess, "run", lambda *_a, **_k: None)
    attempted = []

    def component(argv):
        index = yaml.safe_load(Path(argv[-1]).read_text())["index"]
        attempted.append(index)
        if index == 0:
            raise subprocess.CalledProcessError(1, argv)

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
        configs.append(
            {
                "input_key": "person",
                "augmentation_index": index,
                "media_uri": str(media),
                "caption_uri": str(caption),
                "metadata_uri": str(metadata),
            }
        )
    manifest = _write_fixture_json(
        tmp_path / "configs.json", {"workflow": workflow, "configs": configs}
    )
    if workflow == "evg":
        with pytest.raises(
            paidf_native.PaidfNativeError, match="requires every expected"
        ):
            paidf_native.validate_augmentation(
                str(manifest), str(tmp_path / "result.json"), "unit-run"
            )
    else:
        result = paidf_native.validate_augmentation(
            str(manifest), str(tmp_path / "result.json"), "unit-run"
        )
        assert result["accepted_count"] == 1
        assert result["accepted"][0]["augmentation_index"] == 1
        assert result["skipped_count"] == 1
        assert result["skipped"][0]["reason"] == "attribute_verification failed"


def test_all_failed_iaa_batch_records_failure_without_promoting_empty_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    manifest = _write_fixture_json(
        tmp_path / "configs.json",
        {
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

    def component(argv):
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
            ]
        },
    )
    upstream = _write_fixture_json(
        tmp_path / "upstream.json", {"schema": "npa.paidf.upstream.v1"}
    )
    labels = _write_fixture_json(
        tmp_path / "labels.json",
        {
            "outputs": [
                {
                    "key": "person_aug0",
                    "data_path": str(data_path),
                }
            ]
        },
    )
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
    with pytest.raises(paidf_native.PaidfNativeError, match="required bundle"):
        paidf_native.validate_dataset(str(dataset), str(report), "unit-run")
    assert not report.exists()


def test_native_reports_record_executed_image_without_polluting_upstream_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    image = "registry.example.test/runtime@sha256:" + "a" * 64
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    native = paidf_native._write_json(
        {"schema": "npa.paidf.native.iaa-augmentation.v1"},
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
    manifest = paidf_native._dig_cache_manifest(tmp_path, initialize=True)
    monkeypatch.setenv("HF_HUB_CACHE", "/unrelated/cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("CKPT_DIR", "/unrelated/checkpoints")
    env = paidf_native._dig_offline_environment(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hf")
    assert env["CKPT_DIR"] == str(tmp_path)
    assert len(manifest["models"]) == 4
    assert all(len(model["revision"]) == 40 for model in manifest["models"])


@pytest.mark.parametrize(
    "defect", ["missing_snapshot", "changed_ref", "changed_manifest"]
)
def test_dig_runtime_refuses_missing_or_drifted_cache(
    tmp_path: Path, defect: str
) -> None:
    _write_dig_runtime_cache(tmp_path)
    manifest = paidf_native._dig_cache_manifest(tmp_path, initialize=True)
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
        paidf_native._dig_offline_environment(tmp_path)


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
    output = tmp_path / "published"
    result = paidf_native.prepare_dig_pretrained(
        str(output), str(tmp_path / "result.json"), "unit-run"
    )
    revisions = paidf_native._dig_model_revisions()
    downloads = [argv for argv, _env in calls if argv[0] == "uvx"]
    assert len(downloads) == 6
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
    assert result["status"] == "completed"
