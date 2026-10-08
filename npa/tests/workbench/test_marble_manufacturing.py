"""Verify manufacturing data isolation, exact composite labels, and honest comparisons."""

import io
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from npa.workbench.marble import api, pallet_benchmark as benchmark, pallet_data
from npa.workbench.marble.pallet_report import _validate
from npa.workbench.marble.schemas import RunRequest


def _image_bytes(color, mode="RGB"):
    image = Image.new(mode, (80, 60), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _dataset():
    return {
        "train": [
            {
                "id": "train",
                "group": "session-1",
                "uri": "s3://example-bucket/train.png",
                "boxes": [[0, 0, 20, 30]],
            }
        ],
        "evaluation": [
            {
                "id": "test",
                "group": "session-2",
                "uri": "s3://example-bucket/test.png",
                "boxes": [[1, 2, 30, 40]],
            }
        ],
        "cutouts": [{"uri": "s3://example-bucket/cutout.png", "source_id": "train"}],
    }


def _cutout():
    image = Image.new("RGBA", (30, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((3, 2, 25, 17), fill=(255, 0, 0, 255))
    return image


@pytest.mark.parametrize("change", ["group", "source_id", "id", "uri"])
def test_dataset_rejects_overlap_and_non_s3_sources(change):
    value = _dataset()
    if change == "source_id":
        value["cutouts"][0][change] = "test"
    elif change == "uri":
        value["train"][0][change] = "/tmp/worker-invisible.png"
    else:
        value["evaluation"][0][change] = value["train"][0][change]
    with pytest.raises(ValueError):
        pallet_data.PalletDataset.model_validate(value)


def test_snapshot_rejects_same_pixels_in_different_partitions(monkeypatch, tmp_path):
    payload = _image_bytes("red")
    monkeypatch.setattr(pallet_data, "read_bytes_uri", lambda _: payload)
    with pytest.raises(api.MarbleError, match="identical decoded"):
        pallet_data._snapshot(
            tmp_path, pallet_data.PalletDataset.model_validate(_dataset()), "run"
        )


@pytest.mark.parametrize("box", [[0, 0, 81, 20], [20, 0, 10, 20], [-1, 0, 20, 20]])
def test_labels_must_be_inside_real_image(box):
    with pytest.raises(api.MarbleError, match="inside"):
        pallet_data._validate_boxes(Image.new("RGB", (80, 60)), [box])


def test_preflight_requires_key_before_reading_data(monkeypatch):
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    monkeypatch.setattr(
        api, "load_credentials", lambda: type("Credentials", (), {"tokens": {}})()
    )
    monkeypatch.setattr(
        pallet_data, "read_bytes_uri", lambda _: pytest.fail("data read without key")
    )
    with pytest.raises(api.MarbleError, match="WLT_API_KEY"):
        pallet_data.pallet_preflight(
            RunRequest(
                input_path="s3://example-bucket/input.json",
                output_path="s3://example-bucket/snapshot",
                run_id="run",
            )
        )


def test_snapshot_contains_real_labels_hashes_and_training_cutout(
    monkeypatch, tmp_path
):
    buffer = io.BytesIO()
    _cutout().save(buffer, format="PNG")
    payloads = {
        "train.png": _image_bytes("red"),
        "test.png": _image_bytes("blue"),
        "cutout.png": buffer.getvalue(),
    }
    monkeypatch.setattr(
        pallet_data, "read_bytes_uri", lambda uri: payloads[uri.rsplit("/", 1)[-1]]
    )
    result = pallet_data._snapshot(
        tmp_path, pallet_data.PalletDataset.model_validate(_dataset()), "run"
    )
    assert result["train"][0]["pixel_sha256"] != result["evaluation"][0]["pixel_sha256"]
    assert result["cutouts"][0]["source_id"] == "train"
    with Image.open(tmp_path / result["cutouts"][0]["file"]) as image:
        assert image.size == (23, 16)


def test_composite_box_matches_actual_foreground_pixels():
    background = Image.new("RGB", (200, 160), "black")
    payload, box = benchmark.composite_pallet(background, _cutout(), random.Random(42))
    array = np.asarray(Image.open(io.BytesIO(payload)))
    yy, xx = np.where(array[:, :, 0] > 0)
    assert box == [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    assert benchmark.composite_pallet(background, _cutout(), random.Random(42)) == (
        payload,
        box,
    )


def test_lance_views_match_training_budget_and_preserve_test_partition(tmp_path):
    import lancedb

    real = [benchmark._row(_image_bytes("red"), [[0, 0, 20, 30]])]
    synthetic = [benchmark._row(_image_bytes("green"), [[2, 3, 20, 30]])] * 3
    evaluation = [benchmark._row(_image_bytes("blue"), [[1, 2, 20, 30]])]
    assert benchmark._write_views(tmp_path / "lance", real, synthetic, evaluation) == 4
    db = lancedb.connect(str(tmp_path / "lance"))
    assert (
        db.open_table("baseline").count_rows()
        == db.open_table("augmented").count_rows()
        == 4
    )
    assert db.open_table("evaluation").to_arrow().to_pylist() == evaluation
    assert db.open_table("baseline").to_arrow().to_pylist() == real * 4


def test_negative_accuracy_change_is_preserved():
    result = benchmark.compare_metrics(
        {"mAP": 0.5, "mAP_50": 0.7, "mAP_75": 0.4},
        {"mAP": 0.4, "mAP_50": 0.6, "mAP_75": 0.3},
    )
    assert result["outcome"] == "no_improvement"
    assert result["delta"]["mAP"] == pytest.approx(-0.1)


@pytest.mark.parametrize("value", [-1, float("nan"), 1.1])
def test_invalid_evaluation_cannot_claim_improvement(value):
    metrics = {"mAP": value, "mAP_50": 0.5, "mAP_75": 0.2}
    with pytest.raises(api.MarbleError, match="Undefined"):
        benchmark.compare_metrics(metrics, metrics)


def test_report_rejects_fabricated_delta():
    metrics = {"mAP": 0.5, "mAP_50": 0.7, "mAP_75": 0.4}
    result = {
        "run_id": "run",
        "gpu": {"name": "test GPU"},
        "generated_this_run": True,
        "arms": {"baseline": metrics, "augmented": metrics},
        "delta": {"mAP": 1},
        "outcome": "improved",
    }
    with pytest.raises(api.MarbleError, match="does not match"):
        _validate(result, "run")
