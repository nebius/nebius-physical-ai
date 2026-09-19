from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

from botocore.exceptions import ClientError
import pytest

from npa.workbench.cosmos import sam2_masks as sm


def test_sam2_config_is_pinned_and_validates_quality_bounds() -> None:
    config = sm.Sam2MaskConfig()
    config.validate()
    assert config.model_id == "facebook/sam2.1-hiera-tiny"
    assert len(config.model_revision) == 40
    with pytest.raises(sm.Sam2MaskError, match="area fractions"):
        sm.Sam2MaskConfig(min_area_fraction=0.7, max_area_fraction=0.6).validate()


def test_sam2_rejects_prompt_coordinates_in_the_workflow_surface() -> None:
    with pytest.raises(sm.Sam2MaskError, match="off or sam2-auto"):
        sm.Sam2MaskConfig(mode="sam2-boxes").validate()


def test_automatic_mask_selection_rejects_speckles_and_background() -> None:
    selected = sm._select_automatic_boxes(
        [
            {"area": 1, "bbox": [1, 1, 1, 1], "predicted_iou": 1, "stability_score": 1},
            {
                "area": 9_900,
                "bbox": [0, 0, 100, 100],
                "predicted_iou": 1,
                "stability_score": 1,
            },
            {
                "area": 1_600,
                "bbox": [20, 20, 40, 40],
                "predicted_iou": 0.95,
                "stability_score": 0.96,
            },
        ],
        width=100,
        height=100,
        min_area_fraction=0.002,
        max_area_fraction=0.65,
        limit=2,
    )
    assert selected == [(20.0, 20.0, 60.0, 60.0)]


def test_automatic_mask_selection_prefers_compact_distinct_foreground() -> None:
    selected = sm._select_automatic_boxes(
        [
            # The historical area score preferred this broad, high-confidence
            # background proposal because its area was closest to half a frame.
            {
                "area": 5_500,
                "bbox": [0, 0, 100, 55],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
            },
            {
                "area": 400,
                "bbox": [10, 10, 20, 20],
                "predicted_iou": 0.95,
                "stability_score": 0.96,
            },
            # A near-duplicate of the compact object must not consume another
            # propagation slot even though its confidence is also high.
            {
                "area": 420,
                "bbox": [10, 10, 21, 20],
                "predicted_iou": 0.94,
                "stability_score": 0.95,
            },
            {
                "area": 300,
                "bbox": [65, 60, 15, 20],
                "predicted_iou": 0.91,
                "stability_score": 0.93,
            },
        ],
        width=100,
        height=100,
        min_area_fraction=0.002,
        max_area_fraction=0.65,
        limit=2,
    )

    assert selected == [
        (10.0, 10.0, 30.0, 30.0),
        (65.0, 60.0, 80.0, 80.0),
    ]


def test_sam2_contract_versions_the_automatic_selection_policy() -> None:
    assert sm.Sam2MaskConfig().public_contract()["selection_policy"] == (
        "compact-motion-foreground-v3"
    )


def test_temporal_motion_support_includes_action_path_not_static_backdrop(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    frames = []
    for index, left in enumerate((8, 24, 40)):
        frame = Image.new("RGB", (64, 48), (180, 180, 180))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((2, 2, 12, 12), fill=(40, 70, 180))
        draw.rectangle((left, 29, left + 5, 36), fill=(20, 180, 40))
        path = tmp_path / f"frame-{index:02d}.png"
        frame.save(path)
        frames.append(path)

    mask, evidence = sm._temporal_motion_support(frames, width=64, height=48)

    assert mask[32, 10]
    assert mask[32, 43]
    assert not mask[7, 7]
    assert not mask[20, 30]
    assert evidence == {
        "engine": "temporal-rgb-range-v1",
        "sample_count": 3,
        "threshold": 24,
        "dilation_pixels": 11,
        "coverage": pytest.approx(float(mask.mean())),
    }


def test_temporal_motion_support_rejects_camera_wide_change(tmp_path: Path) -> None:
    from PIL import Image

    frames = []
    for index, value in enumerate((0, 255)):
        path = tmp_path / f"wide-{index}.png"
        Image.new("RGB", (32, 24), (value, value, value)).save(path)
        frames.append(path)

    with pytest.raises(sm.Sam2MaskError, match="camera-wide motion"):
        sm._temporal_motion_support(frames, width=32, height=24)


def test_publish_sam2_masks_requires_and_publishes_exact_frame_count(
    tmp_path: Path,
) -> None:
    from PIL import Image

    masks = tmp_path / "masks"
    masks.mkdir()
    for index in range(2):
        Image.new("L", (4, 3), 255).save(masks / f"mask-{index:06d}.png")
    manifest = {
        "schema": sm.MASK_SCHEMA,
        "engine": sm.SAM2_ENGINE,
        "component": "Meta Segment Anything Model 2",
        "component_version": "1.0",
        "component_revision": sm.SAM2_SOURCE_REVISION,
        "license": {"spdx": "Apache-2.0"},
        "config": {"mode": "sam2-auto"},
        "frame_count": 2,
        "width": 4,
        "height": 3,
        "object_count": 1,
        "mask_coverage": {"mean": 0.1},
        "runtime": {"device": "cuda"},
        "lineage": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    uploads: list[str] = []

    class Storage:
        def upload_file(self, _local: str, uri: str) -> str:
            uploads.append(uri)
            return uri

    published = sm.publish_sam2_masks(
        sm.Sam2MaskResult(manifest, manifest_path, masks),
        "s3://example/run/segmentation/",
        storage_client=Storage(),
    )

    assert uploads == [
        "s3://example/run/segmentation/masks/mask-000000.png",
        "s3://example/run/segmentation/masks/mask-000001.png",
        "s3://example/run/segmentation/manifest.json",
    ]
    assert published["manifest_uri"].endswith("/manifest.json")
    assert published["masks_uri"].endswith("/masks/")


def test_load_published_sam2_masks_reuses_only_an_exact_contract(
    tmp_path: Path,
) -> None:
    from PIL import Image

    config = sm.Sam2MaskConfig()
    base = "s3://example/run/segmentation/"
    png = tmp_path / "mask.png"
    mask = Image.new("L", (4, 3), 0)
    for x in range(2):
        for y in range(3):
            mask.putpixel((x, y), 255)
    mask.save(png)
    manifest = {
        "schema": sm.MASK_SCHEMA,
        "status": "executed",
        "engine": sm.SAM2_ENGINE,
        "component": "Meta Segment Anything Model 2",
        "component_version": sm.SAM2_DISTRIBUTION_VERSION,
        "component_source": "https://github.com/facebookresearch/sam2",
        "component_revision": sm.SAM2_SOURCE_REVISION,
        "license": {"spdx": sm.SAM2_LICENSE, "url": sm.SAM2_LICENSE_URL},
        "config": config.public_contract(),
        "frame_count": 2,
        "width": 4,
        "height": 3,
        "object_count": 1,
        "mask_coverage": {"mean": 0.5, "min": 0.5, "max": 0.5},
        "motion_support": {
            "engine": sm.SAM2_MOTION_ENGINE,
            "sample_count": 2,
            "threshold": sm.SAM2_MOTION_THRESHOLD,
            "dilation_pixels": sm.SAM2_MOTION_DILATION,
            "coverage": 0.2,
        },
        "runtime": {"device": "cuda", "seconds": 2.0, "frames_per_second": 1.0},
        "manifest_uri": f"{base}manifest.json",
        "masks_uri": f"{base}masks/",
        "lineage": {
            "source_kind": "frame-aligned-video",
            "source_frame_count": 2,
            "source_run_id": "run",
            "mask_pattern": f"{base}masks/mask-%06d.png",
        },
    }
    objects = {
        ("example", "run/segmentation/manifest.json"): json.dumps(manifest).encode(),
        ("example", "run/segmentation/masks/mask-000000.png"): png.read_bytes(),
        ("example", "run/segmentation/masks/mask-000001.png"): png.read_bytes(),
    }

    class S3:
        def get_object(self, *, Bucket: str, Key: str):
            try:
                return {"Body": BytesIO(objects[(Bucket, Key)])}
            except KeyError as exc:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}}, "GetObject"
                ) from exc

        def download_file(self, bucket: str, key: str, local: str) -> None:
            Path(local).write_bytes(objects[(bucket, key)])

    storage = type("Storage", (), {"s3": S3()})()
    reused = sm.load_published_sam2_masks(
        base,
        tmp_path / "reused",
        config=config,
        run_id="run",
        storage_client=storage,
    )

    assert reused is not None
    assert len(list(reused.masks_dir.glob("mask-*.png"))) == 2
    for invalid_coverage in (
        {"mean": 0.0, "min": 0.0, "max": 0.0},
        {"mean": 1.0, "min": 1.0, "max": 1.0},
    ):
        invalid_manifest = {**manifest, "mask_coverage": invalid_coverage}
        objects[("example", "run/segmentation/manifest.json")] = json.dumps(
            invalid_manifest
        ).encode()
        with pytest.raises(sm.Sam2MaskError, match="invalid evidence"):
            sm.load_published_sam2_masks(
                base,
                tmp_path / f"invalid-{invalid_coverage['mean']}",
                config=config,
                run_id="run",
                storage_client=storage,
            )
    objects[("example", "run/segmentation/manifest.json")] = json.dumps(
        manifest
    ).encode()
    with pytest.raises(sm.Sam2MaskError, match="immutable configuration"):
        sm.load_published_sam2_masks(
            base,
            tmp_path / "mismatch",
            config=sm.Sam2MaskConfig(max_objects=5),
            run_id="run",
            storage_client=storage,
        )
    with pytest.raises(sm.Sam2MaskError, match="invalid evidence"):
        sm.load_published_sam2_masks(
            base,
            tmp_path / "wrong-run",
            config=config,
            run_id="another-run",
            storage_client=storage,
        )


def test_load_published_sam2_masks_returns_none_only_when_manifest_is_absent(
    tmp_path: Path,
) -> None:
    class S3:
        def get_object(self, *, Bucket: str, Key: str):
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    storage = type("Storage", (), {"s3": S3()})()
    assert (
        sm.load_published_sam2_masks(
            "s3://example/new/segmentation/",
            tmp_path,
            config=sm.Sam2MaskConfig(),
            run_id="run",
            storage_client=storage,
        )
        is None
    )


def test_sam2_contract_rejects_non_binary_masks(
    tmp_path: Path,
) -> None:
    from PIL import Image

    mask = tmp_path / "mask.png"
    Image.new("L", (4, 3), 128).save(mask)
    with pytest.raises(sm.Sam2MaskError, match="binary PNG"):
        sm._verify_binary_mask(mask, width=4, height=3)
