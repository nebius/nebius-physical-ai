"""Keep native NRE validation modalities separate on the decoded Rerun timeline."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from npa.workflows import data_factory_viz as viz
from npa.workflows.data_factory_viz import build_run_rrd


def _native_validation_images(root: Path) -> dict[tuple[str, int], Path]:
    image = pytest.importorskip("PIL.Image")
    expected = {}
    for modality_index, modality in enumerate(
        ("pred_distance", "pred_opacity", "pred_rgb")
    ):
        for frame in range(38):
            relative = Path("val") / modality / "cam_00" / f"{frame:06d}.png"
            path = root / "reconstruction" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.new("RGB", (32, 24), (frame * 5, modality_index * 70, 120)).save(path)
            expected[(f"/reconstruction/val/{modality}/cam_00", frame)] = path
    return expected


def _review_pixels(path: Path) -> bytes:
    image = pytest.importorskip("PIL.Image")
    encoded = io.BytesIO()
    with image.open(path) as source:
        source.convert("RGB").save(encoded, format="JPEG", quality=75)
    with image.open(io.BytesIO(encoded.getvalue())) as source:
        return source.convert("RGB").tobytes()


def _decoded_images(recording: Path) -> list[tuple[str, int, bytes]]:
    reader = pytest.importorskip("rerun_bindings").RrdReaderInternal
    image = pytest.importorskip("PIL.Image")
    rows = []
    for chunk in reader(str(recording)).stream():
        for row in chunk.to_record_batch().to_pylist():
            if "EncodedImage:blob" not in row:
                continue
            assert "frame" in chunk.timeline_names
            assert len(row["EncodedImage:blob"]) == 1
            payload = bytes(row["EncodedImage:blob"][0])
            with image.open(io.BytesIO(payload)) as decoded:
                rows.append(
                    (
                        str(chunk.entity_path),
                        row["frame"],
                        decoded.convert("RGB").tobytes(),
                    )
                )
    return rows


def test_native_validation_modalities_keep_unique_entities_and_frame_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rerun")
    monkeypatch.setattr(viz, "RRD_MAX_FRAMES_PER_ENTITY", 24)
    monkeypatch.setattr(viz, "RRD_MAX_FRAME_DIM", 512)
    monkeypatch.setattr(viz, "RRD_JPEG_QUALITY", 75)
    root = tmp_path / "run"
    expected = _native_validation_images(root)
    recording = tmp_path / "review.rrd"

    result = build_run_rrd(str(root), str(recording), app_id="neural-reconstruction")
    rows = _decoded_images(recording)

    # Each native modality gets its own bounded review sequence. Reusing the
    # camera basename would collapse three images onto the same entity/frame.
    selected = {int(index * 38 / 24) for index in range(24)}
    identities = [(entity, frame) for entity, frame, _ in rows]
    assert result["frames_logged"] == len(rows) == 72
    assert len(set(identities)) == len(rows)
    assert set(identities) == {
        identity for identity in expected if identity[1] in selected
    }
    for entity, frame, pixels in rows:
        assert pixels == _review_pixels(expected[(entity, frame)])
