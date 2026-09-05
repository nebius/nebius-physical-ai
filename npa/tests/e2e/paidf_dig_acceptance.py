"""Independent checks of the pinned AnomalyGen Day-1 deliverables.

This module is test-only: native workers retain their existing dependencies.
COCO compressed RLE is decoded independently from its published wire format:
https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c
The oracle fixtures were produced by the real BSD-2-Clause pycocotools package.
AnomalyGen selection/label semantics are pinned at dbaf7d7d9003f048230f9026da5969e9e5931785
(callbacks/training_report.py, eval/metric_specs.py, scripts/texture/pseudo_label.py).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import PurePosixPath

import numpy as np
import yaml
from PIL import Image

from npa.workflows.paidf_dig_guardrails import require_dig_guardrail_runtime
from npa.workflows.paidf_native import DIG_PRETRAINED_CONTENT_MANIFEST, SCHEMA_PREFIX
from npa.workflows.paidf_upstream import PHYSICAL_AI_DATA_FACTORY_REVISION

METRICS = {
    "nn": ("nn_score", "max"),
    "mnn": ("mnn_score", "max"),
    "fid": ("fid", "min"),
    "aq_nn": ("aq_nn", "max"),
    "completeness": ("completeness", "max"),
    "precision": ("precision", "max"),
    "boundary_iou": ("boundary_iou", "max"),
}
GENERATION_COLUMNS = (
    "output_filename",
    "image_filename",
    "mask_filename",
    "anomaly_type",
    "guidance",
    "num_steps",
    "seed",
    "num_generated_images",
    "crop_and_paste",
    "crop_ratio",
    "poisson_blend",
    "PSNR",
    "index",
)
BLOCKED_COLUMNS = (
    "index",
    "output_idx",
    "anomaly_type",
    "image_filename",
    "mask_filename",
    "guardrail",
    "message",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def relative_path(value: str) -> str:
    assert isinstance(value, str) and value, "missing artifact path"
    path = PurePosixPath(value)
    assert not path.is_absolute() and ".." not in path.parts
    assert path.as_posix() == value and "\\" not in value
    return value


def digest_string(value) -> str:
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    return value


def decode_coco_rle(value: dict, *, width: int, height: int) -> np.ndarray:
    """Decode signed differential five-bit runs with bounded uint32 counts."""
    assert type(width) is int and width > 0 and type(height) is int and height > 0
    assert isinstance(value, dict) and set(value) == {"size", "counts"}
    assert value["size"] == [height, width]
    assert all(type(dimension) is int for dimension in value["size"])
    encoded = value["counts"]
    assert isinstance(encoded, str) and encoded
    runs = []
    index = total = 0
    while index < len(encoded):
        number = groups = 0
        while True:
            assert index < len(encoded), "unterminated COCO run"
            code = ord(encoded[index]) - 48
            assert 0 <= code < 64, "invalid COCO character"
            index += 1
            number |= (code & 31) << (5 * groups)
            groups += 1
            # The format stores uint32 counts and signed differences thereof.
            assert groups <= 7, "COCO run overflows its count representation"
            if not code & 32:
                if code & 16:
                    number -= 1 << (5 * groups)
                break
        assert -(2**32 - 1) <= number <= 2**32 - 1
        if len(runs) > 2:
            number += runs[-2]
        assert 0 <= number <= 2**32 - 1, "negative or overflowing COCO run"
        total += number
        assert total <= width * height, "COCO run exceeds image dimensions"
        runs.append(number)
    assert total == width * height, "COCO runs do not cover the image"
    decoded = np.zeros(total, dtype=np.bool_)
    offset = 0
    for index, count in enumerate(runs):
        if index % 2:
            decoded[offset : offset + count] = True
        offset += count
    return decoded.reshape((height, width), order="F")


def decode_image(payload: bytes, mode: str) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        assert image.width > 0 and image.height > 0
        return np.asarray(image.convert(mode)).copy()


def csv_rows(payload: bytes, columns: tuple[str, ...]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    assert reader.fieldnames == list(columns), (
        "CSV header differs from pinned upstream schema"
    )
    rows = list(reader)
    assert all(None not in row and None not in row.values() for row in rows)
    return rows


def verify_checkpoint_selection(*, read_bytes, list_keys, hash_file, finetune, usecase):
    selected = relative_path(finetune["selected_checkpoint"])
    selected_path = PurePosixPath(selected)
    assert selected_path.parent.name == "model"
    assert selected_path.parent.parent.name == "checkpoints"
    match = re.fullmatch(r"iter_([0-9]{9})\.pt", selected_path.name)
    assert match, "unexpected checkpoint filename"
    selected_iteration = int(match[1])
    keys = set(list_keys("finetune/"))
    pointer = "finetune/" + str(selected_path.parent.parent / "best_checkpoint.txt")
    assert {key for key in keys if key.endswith("/best_checkpoint.txt")} == {pointer}
    assert read_bytes(pointer).decode().strip() == selected_path.name
    actual_hash, size = hash_file("finetune/" + selected)
    assert size > 0 and actual_hash == digest_string(
        finetune["selected_checkpoint_sha256"]
    )
    embedded = json.loads(read_bytes("finetune/npa-finetune.json"))
    assert embedded == finetune, "embedded finetune record differs"
    run_root = "finetune/" + str(selected_path.parents[2])
    recipe = yaml.safe_load(read_bytes(run_root + "/exp_texture_ft.yaml"))
    assert recipe["dataset_name"] == usecase
    metric = recipe.get("early_stop_metric", "nn")
    assert metric in METRICS
    row_name, direction = METRICS[metric]
    assert type(recipe["max_iter"]) is int and recipe["max_iter"] > 0
    prefix = run_root + "/valid/"
    scored = []
    for key in sorted(keys):
        if not key.startswith(prefix) or not key.endswith("/valid_kpi.csv"):
            continue
        relative = key[len(prefix) :].split("/")
        assert len(relative) == 2 and relative[0].isdigit()
        iteration = int(relative[0])
        reader = csv.reader(io.StringIO(read_bytes(key).decode()))
        header = next(reader)
        assert header[-1] == "Average"
        values = {}
        for row in reader:
            if not row:
                break
            assert row[0] not in values, "duplicate validation metric"
            values[row[0]] = row[-1]
        checkpoint = run_root + f"/checkpoints/model/iter_{iteration:09d}.pt"
        if row_name in values and checkpoint in keys:
            score = float(values[row_name])
            # Missing/NaN metrics are ineligible upstream; infinite metrics are
            # not meaningful evaluator evidence and must not certify a run.
            if math.isnan(score):
                continue
            assert math.isfinite(score), "nonfinite checkpoint-selection metric"
            scored.append((iteration, score))
    assert scored, "no scored saved checkpoint"
    scored.sort()
    if recipe["max_iter"] > 7500:
        settled = [item for item in scored if item[0] >= 7500]
        scored = settled or scored
    expected = (max if direction == "max" else min)(scored, key=lambda item: item[1])
    assert selected_iteration == expected[0], (
        "best checkpoint contradicts evaluator scores"
    )
    for name in ("training_curves.png", "training_loss.png"):
        decode_image(read_bytes(run_root + "/" + name), "RGB")
    # Only a triggered stop produces this file; metal_surface disables stopping.
    stop_path = run_root + "/early_stop.json"
    if stop_path in keys:
        stopped = json.loads(read_bytes(stop_path))
        assert recipe.get("early_stop_enabled") is True and stopped["triggered"] is True
        assert stopped["criteria"] == metric
        assert type(stopped["stop_iteration"]) is int
        assert (
            stopped["best_iteration"] <= stopped["stop_iteration"] <= recipe["max_iter"]
        )
    return {"metric": metric, "score": expected[1], "iteration": expected[0]}, recipe


def assert_dig_live_artifacts(
    *, read_bytes, list_keys, hash_file, run_id, prefix_uri, num_sdg, usecase
):
    """Reopen media and provenance independently of producer success reports."""

    def document(path):
        value = json.loads(read_bytes(relative_path(path)))
        assert isinstance(value, dict)
        return value

    def report(path, schema):
        value = document(path)
        assert value["schema"] == f"{SCHEMA_PREFIX}.{schema}.v1"
        assert value["run_id"] == run_id and value["workflow"] == "dig"
        assert value["status"] == "completed"
        return value

    pretrained = report("reports/pretrained-result.json", "dig-pretrained")
    finetune = report("reports/finetune-result.json", "dig-finetune")
    result = report("reports/dig-result.json", "dig-result")
    for value, directory in (
        (pretrained, "pretrained"),
        (finetune, "finetune"),
        (result, "anomaly"),
    ):
        assert (
            value["output_uri"].rstrip("/") == prefix_uri.rstrip("/") + "/" + directory
        )
    assert result["finetune_result_uri"] == prefix_uri + "reports/finetune-result.json"
    for value in (finetune, result):
        assert value["upstream_workflow_revision"] == PHYSICAL_AI_DATA_FACTORY_REVISION
    manifest_bytes = read_bytes("pretrained/" + DIG_PRETRAINED_CONTENT_MANIFEST)
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == f"{SCHEMA_PREFIX}.dig-pretrained-content.v1"
    assert manifest["run_id"] == run_id and manifest["workflow"] == "dig"
    manifest_hash = sha256(manifest_bytes)
    assert pretrained["content_manifest"] == DIG_PRETRAINED_CONTENT_MANIFEST
    assert pretrained["content_manifest_sha256"] == manifest_hash
    assert finetune["pretrained_content_manifest_sha256"] == manifest_hash
    assert result["pretrained_content_manifest_sha256"] == manifest_hash
    paths = [relative_path(item["path"]) for item in manifest["files"]]
    assert (
        len(set(paths))
        == len(paths)
        == manifest["file_count"]
        == pretrained["file_count"]
        > 0
    )
    assert set(list_keys("pretrained/")) == {
        "pretrained/" + path for path in [*paths, DIG_PRETRAINED_CONTENT_MANIFEST]
    }, "published pretrained file inventory differs from its manifest"
    assert (
        sum(item["size_bytes"] for item in manifest["files"])
        == manifest["total_bytes"]
        == pretrained["total_bytes"]
        > 0
    )
    for item in manifest["files"]:
        digest_string(item["sha256"])
        assert type(item["size_bytes"]) is int and item["size_bytes"] >= 0
    assert (
        sha256(read_bytes("pretrained/checkpoint_manifest_converted.sha256"))
        == pretrained["manifest_sha256"]
    )
    for key in ("selected_checkpoint", "selected_checkpoint_sha256"):
        assert result[key] == finetune[key]
    selection, recipe = verify_checkpoint_selection(
        read_bytes=read_bytes,
        list_keys=list_keys,
        hash_file=hash_file,
        finetune=finetune,
        usecase=usecase,
    )
    count = result["image_count"]
    assert type(count) is int and count > 0 and result["label_file_count"] == 1
    assert len(result["images"]) == count
    require_dig_guardrail_runtime(result["guardrail_runtime"], count)
    timing_bytes = read_bytes("anomaly/timing_summary.json")
    assert sha256(timing_bytes) == result["guardrail_runtime"]["timing_summary_sha256"]
    timing = json.loads(timing_bytes)
    assert timing["world_size"] == 1 and len(timing["rank_timings"]) == 1
    for record in (timing, timing["rank_timings"][0]):
        assert record["guardrail_enabled"] is True
        assert record["text_guardrail_enforcing"] is True
        assert record["image_guardrail_enforcing"] is False
    blocked = csv_rows(read_bytes("anomaly/guardrail_blocked.csv"), BLOCKED_COLUMNS)
    generated_rows = csv_rows(
        read_bytes("anomaly/texture_ft_generation_result.csv"), GENERATION_COLUMNS
    )
    assert (
        timing["generated_images_total"]
        == timing["rank_timings"][0]["generated_images"]
        == count
    )
    assert (
        timing["guardrail_blocked_total"]
        == timing["rank_timings"][0]["guardrail_blocked"]
        == len(blocked)
        == result["guardrail_runtime"]["guardrail_blocked"]
    )
    assert count + len(blocked) == num_sdg and len(generated_rows) == count
    classes = {"+".join(item) for item in recipe["anomaly_types"]}
    for row in blocked:
        # The pinned preset has no image-content classifier; only text can deny.
        assert row["anomaly_type"] in classes and row["guardrail"] == "text"
        assert int(row["output_idx"]) == -1
        assert row["image_filename"] and row["mask_filename"]
        assert row["message"].strip()
    indices = [int(row["index"]) for row in [*generated_rows, *blocked]]
    assert len(set(indices)) == num_sdg and set(indices) == set(range(num_sdg))
    rows = {row["output_filename"]: row for row in generated_rows}
    assert len(rows) == count
    images = {}
    for item in result["images"]:
        name = relative_path(item["name"])
        assert PurePosixPath(name).name == name and name not in images
        media = read_bytes("anomaly/reconstructed_image/" + name)
        assert (
            sha256(media) == digest_string(item["sha256"])
            and len(media) == item["size_bytes"]
        )
        images[name] = decode_image(media, "RGB")
    assert set(images) == set(rows)
    assert set(list_keys("anomaly/reconstructed_image/")) == {
        "anomaly/reconstructed_image/" + name for name in images
    }, "generated media inventory differs from the completed result"
    labels = document("anomaly/pseudo_labels/coco_annotations.json")
    assert len(labels["images"]) == count and labels["annotations"]
    image_ids = {item["id"]: item for item in labels["images"]}
    categories = {item["id"]: item["name"] for item in labels["categories"]}
    assert len(image_ids) == count and len(categories) == len(labels["categories"])
    assert all(type(key) is int and key > 0 for key in (*image_ids, *categories))
    assert len(set(categories.values())) == len(categories)
    assert (
        set(categories.values())
        == {row["anomaly_type"] for row in rows.values()}
        <= classes
    )
    assert {item["file_name"] for item in labels["images"]} == set(images)
    class_lines = (
        read_bytes("anomaly/pseudo_labels/classification/classes.txt")
        .decode()
        .splitlines()
    )
    assert class_lines == ["original", *sorted(categories.values())]
    mask_by_id, unions = {}, {}
    for image_id, item in image_ids.items():
        name = item["file_name"]
        generated = images[name]
        height, width = generated.shape[:2]
        assert type(item["height"]) is int and type(item["width"]) is int
        assert (item["height"], item["width"]) == (height, width)
        original = decode_image(read_bytes("anomaly/original_image/" + name), "RGB")
        mask = decode_image(read_bytes("anomaly/original_mask/" + name), "L")
        assert original.shape == generated.shape and mask.shape == (height, width)
        assert set(np.unique(mask)) <= {0, 255} and np.any(mask)
        assert np.array_equal(
            mask, decode_image(read_bytes("anomaly/pseudo_labels/masks/" + name), "L")
        )
        mask_by_id[image_id] = mask > 0
        unions[image_id] = np.zeros(mask.shape, dtype=np.bool_)
        for directory in ("images", "classification/" + rows[name]["anomaly_type"]):
            copied = decode_image(
                read_bytes("anomaly/pseudo_labels/" + directory + "/" + name), "RGB"
            )
            assert np.array_equal(copied, generated)
        copied_original = decode_image(
            read_bytes("anomaly/pseudo_labels/classification/original/" + name), "RGB"
        )
        assert np.array_equal(copied_original, original)
        overlay = decode_image(
            read_bytes("anomaly/pseudo_labels/visualization/" + name), "RGB"
        )
        assert overlay.shape == generated.shape and np.any(overlay != generated)
        row = rows[name]
        assert row["image_filename"] and row["mask_filename"]
        assert int(row["num_steps"]) > 0 and int(row["num_generated_images"]) == 1
        assert math.isfinite(float(row["guidance"])) and int(row["seed"]) >= 0
        assert int(row["index"]) >= 0
        assert row["crop_and_paste"] in {"True", "False"}
        assert row["poisson_blend"] in {"True", "False"}
        if row["crop_ratio"] != "none":
            assert (
                math.isfinite(float(row["crop_ratio"])) and float(row["crop_ratio"]) > 0
            )
        difference = (
            original.astype(np.float32)[mask > 0]
            - generated.astype(np.float32)[mask > 0]
        )
        mse = float(np.mean(difference**2))
        expected_psnr = 100.0 if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
        assert math.isclose(float(row["PSNR"]), expected_psnr, abs_tol=1e-4)
    annotation_ids = set()
    for item in labels["annotations"]:
        assert (
            type(item["id"]) is int
            and item["id"] > 0
            and item["id"] not in annotation_ids
        )
        annotation_ids.add(item["id"])
        image_id = item["image_id"]
        assert type(image_id) is int and type(item["category_id"]) is int
        assert image_id in image_ids and item["category_id"] in categories
        assert (
            categories[item["category_id"]]
            == rows[image_ids[image_id]["file_name"]]["anomaly_type"]
        )
        height, width = mask_by_id[image_id].shape
        decoded = decode_coco_rle(item["segmentation"], width=width, height=height)
        assert type(item["area"]) is int and item["area"] == int(decoded.sum()) > 0
        ys, xs = np.nonzero(decoded)
        assert isinstance(item["bbox"], list) and len(item["bbox"]) == 4
        assert all(type(value) is int for value in item["bbox"])
        assert item["bbox"] == [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        assert item["iscrowd"] == 0
        assert not np.any(unions[image_id] & decoded), "overlapping instance masks"
        unions[image_id] |= decoded
    assert all(np.array_equal(unions[key], mask) for key, mask in mask_by_id.items())
    return {
        "image_count": count,
        "annotation_count": len(annotation_ids),
        "blocked_count": len(blocked),
        "checkpoint_selection": selection,
    }
