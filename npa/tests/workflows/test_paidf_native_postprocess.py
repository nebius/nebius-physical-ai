"""IAA post-processing preserves successful siblings from the mapped source DAG."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from npa.workflows import paidf_native as native


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _postprocess_inputs(tmp_path: Path) -> tuple[str, str, dict]:
    accepted = []
    augmentation_outputs = []
    prepared_images = []
    for index in range(2):
        source = tmp_path / f"cosmos/person/{index}/output.jpg"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (96, 96), "navy").save(source)
        config = tmp_path / f"cosmos/person/{index}/config.yaml"
        config.write_text("{}\n", encoding="utf-8")
        caption = tmp_path / f"cosmos/person/{index}/caption.txt"
        caption.write_text("A person wearing blue.\n", encoding="utf-8")
        metadata = tmp_path / f"cosmos/person/{index}/metadata.json"
        metadata.write_text("{}\n", encoding="utf-8")
        output = {
            "input_key": "person",
            "augmentation_index": index,
            "config_uri": str(config),
            "media_uri": str(source),
            "caption_uri": str(caption),
            "metadata_uri": str(metadata),
        }
        augmentation_outputs.append(
            {
                **output,
                "artifacts": native._artifact_fingerprints(
                    {
                        field: output[field]
                        for field in (
                            "config_uri",
                            "media_uri",
                            "caption_uri",
                            "metadata_uri",
                        )
                    }
                ),
            }
        )
        accepted.append(
            {
                **output,
                "sha256": native._sha256(source),
                "size_bytes": source.stat().st_size,
            }
        )
        prepared_images.append(
            {
                "input_key": "person",
                "pane_metadata_uri": str(tmp_path / "person.json"),
            }
        )

    augmentation = {
        "schema": "npa.paidf.native.iaa-augmentation.v1",
        "run_id": "postprocess-run",
        "workflow": "iaa",
        "count": 2,
        "outputs": augmentation_outputs,
    }
    augmentation_uri = _write_json(tmp_path / "augmentation.json", augmentation)
    validation = {
        "schema": "npa.paidf.native.iaa-validation.v1",
        "run_id": "postprocess-run",
        "workflow": "iaa",
        "accepted": accepted,
        "skipped": [],
        "producers": [native._producer_descriptor(augmentation_uri, augmentation)],
    }
    validation_uri = _write_json(tmp_path / "validation.json", validation)
    prepared_uri = _write_json(
        tmp_path / "prepared.json",
        {
            "schema": "npa.paidf.native.prepared-input.v1",
            "run_id": "postprocess-run",
            "images": prepared_images,
        },
    )
    return validation_uri, prepared_uri, validation


def _write_postprocess_output(argv: list[str]) -> None:
    output = Path(argv[argv.index("--output-dir") + 1])
    index = int(output.name.removeprefix("aug_"))
    media = output / f"augmented_imgs/person_aug{index}/person.jpg"
    media.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "blue").save(media)
    _write_json(
        output / "augmented_data.json",
        {
            "entries": [
                {
                    "person_key": f"person_aug{index}",
                    "attributes": {"top_outer_color": "blue"},
                    "selected_attributes": {"top outer color": "blue"},
                    "queries": {
                        "easy": ["blue jacket"],
                        "medium": ["person wearing blue jacket"],
                        "hard": ["Person wearing blue jacket."],
                    },
                    "attribute_verification": {"passed": True},
                    "images": [f"{output.name}/augmented_imgs/person_aug{index}/person.jpg"],
                }
            ]
        },
    )


def _install_runtime_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "test-token")
    monkeypatch.setattr(native, "_runtime_fetch", lambda _r, _v, destination: destination)
    monkeypatch.setattr(native.subprocess, "run", lambda *_a, **_k: None)


def test_postprocess_iaa_retains_successful_mapped_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_uri, prepared_uri, validation = _postprocess_inputs(tmp_path)
    _install_runtime_stubs(monkeypatch)

    def component(argv: list[str], *, env: dict[str, str]) -> None:
        assert env["VLM_API_KEY"] == "test-token"
        if argv[argv.index("--output-dir") + 1].endswith("aug_0"):
            raise subprocess.CalledProcessError(9, argv)
        _write_postprocess_output(argv)

    monkeypatch.setattr(native, "_run_component", component)
    result = native.postprocess_iaa(
        validation_uri,
        prepared_uri,
        str(tmp_path / "postprocessing"),
        str(tmp_path / "result.json"),
        "https://api.tokenfactory.nebius.com/v1",
        "vlm-model",
        "postprocess-run",
    )

    assert [item["augmentation_index"] for item in result["accepted"]] == [1]
    assert result["skipped"] == [
        {
            "input_key": "person",
            "augmentation_index": 0,
            "stage": "iaa-postprocess",
            "reason": "component_retries_exhausted",
            "error_type": "CalledProcessError",
            "exit_code": 9,
        }
    ]
    assert result["producers"] == [
        *validation["producers"],
        native._producer_descriptor(validation_uri, validation),
    ]


def test_postprocess_iaa_fails_when_every_mapped_item_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_uri, prepared_uri, _validation = _postprocess_inputs(tmp_path)
    _install_runtime_stubs(monkeypatch)
    monkeypatch.setattr(
        native,
        "_run_component",
        lambda argv, *, env: (_ for _ in ()).throw(
            subprocess.CalledProcessError(9, argv)
        ),
    )

    result_uri = tmp_path / "result.json"
    with pytest.raises(
        native.PaidfNativeError, match="every mapped IAA post-processing task failed"
    ):
        native.postprocess_iaa(
            validation_uri,
            prepared_uri,
            str(tmp_path / "postprocessing"),
            str(result_uri),
            "https://api.tokenfactory.nebius.com/v1",
            "vlm-model",
            "postprocess-run",
        )
    assert not result_uri.exists()
