"""Synthetic acceptance-contract regressions, never live inference evidence."""

from __future__ import annotations

import csv
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from npa.workflows.paidf_dig_guardrails import (
    dig_guardrail_runtime,
    dig_qwen_source_adaptation,
)
from npa.workflows.paidf_native import DIG_PRETRAINED_CONTENT_MANIFEST, SCHEMA_PREFIX
from npa.workflows.paidf_upstream import PHYSICAL_AI_DATA_FACTORY_REVISION


@pytest.fixture
def acceptance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    return importlib.import_module("tests.e2e.paidf_dig_acceptance")


def test_compressed_rle_matches_actual_pycocotools_oracle_vectors(acceptance):
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/paidf_coco_rle_vectors.json"
    )
    data = json.loads(fixture.read_text())
    assert data["oracle"]["version"] == "2.0.11"
    assert (
        data["oracle"]["wheel_sha256"]
        == "a82d1c9ed83f75da0b3f244f2a3cf559351a283307bd9b79a4ee2b93ab3231dd"
    )
    for vector in data["vectors"]:
        height, width = vector["size"]
        decoded = acceptance.decode_coco_rle(
            {"size": [height, width], "counts": vector["counts"]},
            width=width,
            height=height,
        )
        expected = np.array([value == "1" for value in vector["flat_fortran"]]).reshape(
            (height, width), order="F"
        )
        np.testing.assert_array_equal(decoded, expected)
        assert int(decoded.sum()) == vector["area"]
        if decoded.any():
            ys, xs = np.nonzero(decoded)
            assert [
                xs.min(),
                ys.min(),
                xs.max() - xs.min() + 1,
                ys.max() - ys.min() + 1,
            ] == vector["bbox"]


@pytest.mark.parametrize(
    "counts", ["", "~", "\x00", "P", "O", "1", "9", "oooooooo", "oooooo7", "000O"]
)
def test_compressed_rle_rejects_malformed_negative_overflow_and_wrong_totals(
    acceptance, counts
):
    with pytest.raises(AssertionError):
        acceptance.decode_coco_rle(
            {"counts": counts, "size": [2, 2]}, width=2, height=2
        )


@pytest.mark.parametrize("size", [[-1, 4], [True, 4], [4, 1], [2], "2,2"])
def test_compressed_rle_uses_actual_decoded_media_dimensions(acceptance, size):
    with pytest.raises(AssertionError):
        acceptance.decode_coco_rle({"counts": "04", "size": size}, width=2, height=2)


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def csv_bytes(rows, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def artifacts(acceptance, tmp_path):
    run_id = "synthetic-dig-acceptance"
    prefix = "s3://fixture-bucket/synthetic-dig-acceptance/"
    values = {}
    manifest_payload = b"converted checkpoint source manifest\n"
    pretrained_payload = b"synthetic pretrained bytes: no model"
    content = {
        "schema": f"{SCHEMA_PREFIX}.dig-pretrained-content.v1",
        "run_id": run_id,
        "workflow": "dig",
        "file_count": 2,
        "total_bytes": len(manifest_payload) + len(pretrained_payload),
        "files": [
            {
                "path": "checkpoint_manifest_converted.sha256",
                "sha256": acceptance.sha256(manifest_payload),
                "size_bytes": len(manifest_payload),
            },
            {
                "path": "checkpoints/synthetic.bin",
                "sha256": acceptance.sha256(pretrained_payload),
                "size_bytes": len(pretrained_payload),
            },
        ],
    }
    values["pretrained/" + DIG_PRETRAINED_CONTENT_MANIFEST] = content
    values["pretrained/checkpoint_manifest_converted.sha256"] = manifest_payload
    values["pretrained/checkpoints/synthetic.bin"] = pretrained_payload
    content_hash = acceptance.sha256(json.dumps(content).encode())

    def report(schema, output):
        return {
            "schema": f"{SCHEMA_PREFIX}.{schema}.v1",
            "run_id": run_id,
            "workflow": "dig",
            "status": "completed",
            "output_uri": prefix + output + "/",
        }

    values["reports/pretrained-result.json"] = {
        **report("dig-pretrained", "pretrained"),
        "content_manifest": DIG_PRETRAINED_CONTENT_MANIFEST,
        "content_manifest_sha256": content_hash,
        "manifest_sha256": acceptance.sha256(manifest_payload),
        "file_count": content["file_count"],
        "total_bytes": content["total_bytes"],
    }
    checkpoint = b"synthetic checkpoint bytes: no model"
    selected = "results/run/checkpoints/model/iter_000008000.pt"
    finetune = {
        **report("dig-finetune", "finetune"),
        "selected_checkpoint": selected,
        "selected_checkpoint_sha256": acceptance.sha256(checkpoint),
        "pretrained_content_manifest_sha256": content_hash,
        "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
    }
    values["reports/finetune-result.json"] = finetune
    values["finetune/npa-finetune.json"] = dict(finetune)
    values["finetune/results/run/checkpoints/best_checkpoint.txt"] = (
        b"iter_000008000.pt\n"
    )
    for iteration, score in ((1000, 99.0), (8000, 0.7), (9000, 0.7)):
        values[f"finetune/results/run/checkpoints/model/iter_{iteration:09d}.pt"] = (
            checkpoint
        )
        values[f"finetune/results/run/valid/{iteration}/valid_kpi.csv"] = (
            f"metric,metal_surface+MT_Blowhole,Average\nnn_score,{score},{score}\n\n".encode()
        )
    values["finetune/results/run/exp_texture_ft.yaml"] = yaml.safe_dump(
        {
            "dataset_name": "metal_surface",
            "anomaly_types": [["metal_surface", "MT_Blowhole"]],
            "max_iter": 15000,
            "early_stop_metric": "nn",
            "early_stop_enabled": False,
        }
    ).encode()
    image = np.full((2, 2, 3), 32, dtype=np.uint8)
    original = np.zeros_like(image)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    media, origin = png(image), png(original)
    for path in ("training_curves.png", "training_loss.png"):
        values["finetune/results/run/" + path] = media
    flags = {
        "guardrail_enabled": True,
        "text_guardrail_enforcing": True,
        "image_guardrail_enforcing": False,
    }
    timing = {
        **flags,
        "world_size": 1,
        "generated_images_total": 1,
        "guardrail_blocked_total": 0,
        "rank_timings": [{**flags, "generated_images": 1, "guardrail_blocked": 0}],
    }
    timing_path = tmp_path / "timing.json"
    timing_path.write_text(json.dumps(timing))
    guardrail = dig_guardrail_runtime(
        timing_path,
        {
            "source_adaptation": dig_qwen_source_adaptation(),
            "package_file_count": 2,
            "installed_package_tree_sha256": "1" * 64,
            "overlay_tree_sha256": "2" * 64,
        },
        1,
    )
    values["anomaly/timing_summary.json"] = timing
    result = {
        **report("dig-result", "anomaly"),
        "image_count": 1,
        "label_file_count": 1,
        "guardrail_runtime": guardrail,
        "selected_checkpoint": selected,
        "selected_checkpoint_sha256": acceptance.sha256(checkpoint),
        "pretrained_content_manifest_sha256": content_hash,
        "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
        "finetune_result_uri": prefix + "reports/finetune-result.json",
        "images": [
            {
                "name": "sample.png",
                "sha256": acceptance.sha256(media),
                "size_bytes": len(media),
            }
        ],
    }
    values["reports/dig-result.json"] = result
    for path in (
        "reconstructed_image",
        "pseudo_labels/images",
        "pseudo_labels/classification/metal_surface+MT_Blowhole",
    ):
        values[f"anomaly/{path}/sample.png"] = media
    for path in (
        "original_image",
        "pseudo_labels/classification/original",
        "pseudo_labels/visualization",
    ):
        values[f"anomaly/{path}/sample.png"] = origin
    for path in ("original_mask", "pseudo_labels/masks"):
        values[f"anomaly/{path}/sample.png"] = png(mask)
    values["anomaly/pseudo_labels/classification/classes.txt"] = (
        b"original\nmetal_surface+MT_Blowhole"
    )
    labels = {
        "images": [{"id": 1, "file_name": "sample.png", "width": 2, "height": 2}],
        "categories": [{"id": 1, "name": "metal_surface+MT_Blowhole"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "area": 4,
                "bbox": [0, 0, 2, 2],
                "iscrowd": 0,
                "segmentation": {"counts": "04", "size": [2, 2]},
            }
        ],
    }
    values["anomaly/pseudo_labels/coco_annotations.json"] = labels
    row = {
        "output_filename": "sample.png",
        "image_filename": "/synthetic/input.jpg",
        "mask_filename": "/synthetic/mask.png",
        "anomaly_type": "metal_surface+MT_Blowhole",
        "num_steps": 35,
        "num_generated_images": 1,
        "guidance": 3,
        "seed": 43,
        "index": 0,
        "crop_and_paste": True,
        "crop_ratio": "none",
        "poisson_blend": False,
        "PSNR": 20 * np.log10(255.0) - 10 * np.log10(32.0**2),
    }
    values["anomaly/texture_ft_generation_result.csv"] = csv_bytes(
        [row], acceptance.GENERATION_COLUMNS
    )
    values["anomaly/guardrail_blocked.csv"] = (
        b"index,output_idx,anomaly_type,image_filename,mask_filename,guardrail,message\n"
    )
    reads, hashes = [], []

    def read(path):
        reads.append(path)
        value = values[path]
        return value if isinstance(value, bytes) else json.dumps(value).encode()

    def hash_file(path):
        hashes.append(path)
        payload = read(path)
        return acceptance.sha256(payload), len(payload)

    arguments = {
        "read_bytes": read,
        "list_keys": lambda prefix: [key for key in values if key.startswith(prefix)],
        "hash_file": hash_file,
        "run_id": run_id,
        "prefix_uri": prefix,
        "num_sdg": 1,
        "usecase": "metal_surface",
    }
    return values, arguments, reads, hashes


def test_dig_acceptance_reopens_checkpoint_evaluator_masks_media_and_metadata(
    acceptance, artifacts
):
    _values, arguments, reads, hashes = artifacts
    result = acceptance.assert_dig_live_artifacts(**arguments)
    assert result["checkpoint_selection"] == {
        "metric": "nn",
        "score": 0.7,
        "iteration": 8000,
    }
    assert result["annotation_count"] == result["image_count"] == 1
    assert hashes == [
        "pretrained/checkpoint_manifest_converted.sha256",
        "pretrained/checkpoints/synthetic.bin",
        "finetune/results/run/checkpoints/model/iter_000008000.pt",
    ]
    assert any(path.endswith("valid_kpi.csv") for path in reads)
    assert "anomaly/original_mask/sample.png" in reads
    assert "anomaly/guardrail_blocked.csv" in reads


def test_dig_acceptance_rejects_same_size_retained_pretrained_mutation(
    acceptance, artifacts
):
    values, arguments, _reads, hashes = artifacts
    path = "pretrained/checkpoints/synthetic.bin"
    original = values[path]
    values[path] = bytes([original[0] ^ 1]) + original[1:]
    assert len(values[path]) == len(original)
    with pytest.raises(AssertionError, match="retained pretrained payload changed"):
        acceptance.assert_dig_live_artifacts(**arguments)
    assert path in hashes


def test_dig_acceptance_combines_disjoint_instance_masks(acceptance, artifacts):
    values, arguments, _reads, _hashes = artifacts
    labels = values["anomaly/pseudo_labels/coco_annotations.json"]
    first = labels["annotations"][0]
    first.update(
        area=2, bbox=[0, 0, 1, 2], segmentation={"counts": "022", "size": [2, 2]}
    )
    labels["annotations"].append(
        {
            **first,
            "id": 2,
            "bbox": [1, 0, 1, 2],
            "segmentation": {"counts": "22", "size": [2, 2]},
        }
    )
    assert acceptance.assert_dig_live_artifacts(**arguments)["annotation_count"] == 2


def test_dig_acceptance_accounts_for_real_text_denial_separately(
    acceptance, artifacts, tmp_path
):
    values, arguments, _reads, _hashes = artifacts
    timing = values["anomaly/timing_summary.json"]
    timing["guardrail_blocked_total"] = 1
    timing["rank_timings"][0]["guardrail_blocked"] = 1
    path = tmp_path / "blocked-timing.json"
    path.write_text(json.dumps(timing))
    old = values["reports/dig-result.json"]["guardrail_runtime"]
    source = {
        key: old[key]
        for key in (
            "source_adaptation",
            "package_file_count",
            "installed_package_tree_sha256",
            "overlay_tree_sha256",
        )
    }
    values["reports/dig-result.json"]["guardrail_runtime"] = dig_guardrail_runtime(
        path, source, 1
    )
    row = {
        "index": 1,
        "output_idx": -1,
        "anomaly_type": "metal_surface+MT_Blowhole",
        "image_filename": "/synthetic/blocked.jpg",
        "mask_filename": "/synthetic/blocked.png",
        "guardrail": "text",
        "message": "Synthetic denied verdict for contract testing",
    }
    values["anomaly/guardrail_blocked.csv"] = csv_bytes([row], list(row))
    arguments["num_sdg"] = 2
    assert acceptance.assert_dig_live_artifacts(**arguments)["blocked_count"] == 1
    row["index"] = 0
    values["anomaly/guardrail_blocked.csv"] = csv_bytes([row], list(row))
    with pytest.raises(AssertionError):
        acceptance.assert_dig_live_artifacts(**arguments)


@pytest.mark.parametrize(
    "mutation",
    [
        "rle",
        "bbox",
        "area",
        "mask",
        "overlap",
        "uncovered",
        "duplicate_image",
        "class",
        "checkpoint_bytes",
        "checkpoint_hash",
        "pointer",
        "embedded",
        "selection",
        "no_kpi",
        "infinite_kpi",
        "pretrained",
        "count",
        "classification",
        "original",
        "psnr",
        "blocked",
        "status",
        "run_id",
        "image_bytes",
        "unreported_image",
        "unreported_pretrained",
        "boolean_image_id",
        "float_bbox",
        "empty_blocked_header",
        "missing_generated_column",
        "nonfinite_guidance",
        "invalid_crop_metadata",
    ],
)
def test_dig_acceptance_refuses_inconsistent_real_deliverable_contracts(
    acceptance, artifacts, mutation
):
    values, arguments, _reads, _hashes = artifacts
    labels = values["anomaly/pseudo_labels/coco_annotations.json"]
    annotation = labels["annotations"][0]
    if mutation == "rle":
        annotation["segmentation"] = {"counts": "malformed", "size": [-1, 0]}
    elif mutation == "bbox":
        annotation["bbox"] = [1000, 1000, 2, 2]
    elif mutation == "area":
        annotation["area"] = 1
    elif mutation == "boolean_image_id":
        annotation["image_id"] = True
    elif mutation == "float_bbox":
        annotation["bbox"] = [0.0, 0.0, 2.0, 2.0]
    elif mutation == "empty_blocked_header":
        values["anomaly/guardrail_blocked.csv"] = b"not,the,published,header\n"
    elif mutation in {
        "missing_generated_column",
        "nonfinite_guidance",
        "invalid_crop_metadata",
    }:
        path = "anomaly/texture_ft_generation_result.csv"
        rows = list(csv.DictReader(io.StringIO(values[path].decode())))
        columns = list(acceptance.GENERATION_COLUMNS)
        if mutation == "missing_generated_column":
            columns.remove("crop_ratio")
            del rows[0]["crop_ratio"]
        elif mutation == "nonfinite_guidance":
            rows[0]["guidance"] = "nan"
        else:
            rows[0]["crop_and_paste"] = "possibly"
        values[path] = csv_bytes(rows, columns)
    elif mutation == "unreported_image":
        values["anomaly/reconstructed_image/extra.png"] = values[
            "anomaly/reconstructed_image/sample.png"
        ]
    elif mutation == "unreported_pretrained":
        values["pretrained/extra.bin"] = b"unexpected"
    elif mutation == "mask":
        values["anomaly/pseudo_labels/masks/sample.png"] = png(
            np.zeros((2, 2), dtype=np.uint8)
        )
    elif mutation == "overlap":
        labels["annotations"].append({**annotation, "id": 2})
    elif mutation == "uncovered":
        annotation.update(segmentation={"counts": "13", "size": [2, 2]}, area=3)
    elif mutation == "duplicate_image":
        labels["images"].append(dict(labels["images"][0]))
    elif mutation == "class":
        labels["categories"][0]["name"] = "other"
    elif mutation == "checkpoint_bytes":
        values["finetune/results/run/checkpoints/model/iter_000008000.pt"] = b"changed"
    elif mutation == "checkpoint_hash":
        values["reports/finetune-result.json"]["selected_checkpoint_sha256"] = "z" * 64
    elif mutation == "pointer":
        values["finetune/results/run/checkpoints/best_checkpoint.txt"] = (
            b"iter_000009000.pt"
        )
    elif mutation == "embedded":
        values["finetune/npa-finetune.json"]["run_id"] = "foreign"
    elif mutation in {"selection", "infinite_kpi"}:
        score = "0.9" if mutation == "selection" else "inf"
        values["finetune/results/run/valid/9000/valid_kpi.csv"] = (
            f"metric,Average\nnn_score,{score}\n".encode()
        )
    elif mutation == "no_kpi":
        for path in list(values):
            if path.endswith("valid_kpi.csv"):
                del values[path]
    elif mutation == "pretrained":
        values["reports/dig-result.json"]["pretrained_content_manifest_sha256"] = (
            "f" * 64
        )
    elif mutation == "count":
        values["reports/dig-result.json"]["images"] *= 2
    elif mutation == "classification":
        del values[
            "anomaly/pseudo_labels/classification/metal_surface+MT_Blowhole/sample.png"
        ]
    elif mutation == "original":
        values["anomaly/original_image/sample.png"] = png(
            np.zeros((3, 2, 3), dtype=np.uint8)
        )
    elif mutation == "psnr":
        values["anomaly/texture_ft_generation_result.csv"] = values[
            "anomaly/texture_ft_generation_result.csv"
        ].replace(b"18.02780404228098", b"0")
    elif mutation == "blocked":
        arguments["num_sdg"] = 2
    elif mutation == "status":
        values["reports/dig-result.json"]["status"] = "pending"
    elif mutation == "run_id":
        values["reports/dig-result.json"]["run_id"] = "foreign"
    else:
        values["anomaly/reconstructed_image/sample.png"] = b"invalid image"
    with pytest.raises((AssertionError, KeyError, ValueError)):
        acceptance.assert_dig_live_artifacts(**arguments)
