from __future__ import annotations

import json
import subprocess
from pathlib import Path

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
    paidf_native._run_component(["component"])
    assert attempts == 4


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
                    {"input_key": "input-0000", "prepared_uri": "s3://example/input.png"}
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
