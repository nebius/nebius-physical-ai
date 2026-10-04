"""Check live RRD readback against independent modality and frame fixtures."""

import importlib.util
import io
import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def helpers():
    """Load live assertions. Args: None. Returns: module. Raises: ImportError."""
    path = Path(__file__).resolve().parents[2] / "e2e/npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("nurec_live_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_images(root):
    image = pytest.importorskip("PIL.Image")
    sources = {}
    for modality_index, modality in enumerate(
        ("pred_rgb", "pred_distance", "pred_opacity")
    ):
        for camera, frames in (
            ("front", (7, 13, 29, 41, 67)),
            ("rear", (107, 113, 131, 173, 191, 211)),
        ):
            for index, frame in enumerate(frames):
                path = (
                    root / "reconstruction/val" / modality / camera / f"{frame:06d}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                image.new(
                    "RGB", (16, 16), (17 + modality_index * 61, frame, 29 + index * 23)
                ).save(path)
                sources[f"/reconstruction/val/{modality}/{camera}", frame] = path
    for frame in (5, 11, 25, 50, 99):
        path = root / "novel_views/lens" / f"{frame:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.new("RGB", (16, 16), (frame, 61, 197)).save(path)
        sources["/novel_view/lens", frame] = path
    (root / "reconstruction/metrics.yaml").write_text(
        yaml.safe_dump({"test": {"psnr": 23.0, "ssim": 0.8, "lpips": 0.2}})
    )
    return sources


def _selected(frames, cap, pattern):
    if cap <= 0 or len(frames) <= cap:
        return frames
    if cap == 1:
        return frames[:1]
    if pattern == "late":
        return frames[:1] + frames[-(cap - 1) :]
    return frames[: cap - 1] + frames[-1:]


def _encoded(path, wrong=False):
    image = pytest.importorskip("PIL.Image")
    with image.open(path) as source:
        rgb = source.convert("RGB")
    if wrong:
        rgb = image.new("RGB", (16, 16), (0, 0, 0))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()


def _log_images(recording, sources, cap, pattern, defect):
    rr = pytest.importorskip("rerun")
    groups = defaultdict(list)
    for entity, frame in sources:
        groups[entity].append(frame)
    for entity, frames in groups.items():
        chosen = _selected(sorted(frames), cap, pattern)
        if defect == "missing_last":
            chosen = sorted(frames)[:cap]
        if defect == "missing_first":
            chosen = sorted(frames)[-cap:]
        if defect == "missing_modality" and "/pred_distance/" in entity:
            continue
        target = (
            "/reconstruction/" + entity.rsplit("/", 1)[-1]
            if defect == "merged" and entity.startswith("/reconstruction/")
            else entity
        )
        for ordinal, frame in enumerate(chosen):
            reset = defect == "reset_ids" and entity.startswith("/reconstruction/")
            recording.set_time("frame", sequence=ordinal if reset else frame)
            contents = _encoded(
                sources[entity, frame],
                defect == "wrong_reconstruction_image"
                and entity.startswith("/reconstruction/"),
            )
            recording.log(
                target, rr.EncodedImage(contents=contents, media_type="image/jpeg")
            )
            if defect == "duplicate" and entity.startswith("/reconstruction/"):
                recording.log(
                    target, rr.EncodedImage(contents=contents, media_type="image/jpeg")
                )


def _recording(root, sources, cap, *, pattern="early", defect=""):
    rr = pytest.importorskip("rerun")
    pytest.importorskip("rerun.chunk")
    path = root / "reports/sim2real.rrd"
    path.parent.mkdir()
    recording = rr.RecordingStream("neural-reconstruction", recording_id="rrd-contract")
    recording.save(str(path))
    settings = {
        "schema": "npa.nurec.rrd-review.v1",
        "max_frames_per_entity": cap,
        "max_frame_dim": 0,
        "jpeg_quality": 75,
    }
    metrics = yaml.safe_load((root / "reconstruction/metrics.yaml").read_text())
    for entity, document in (
        ("provenance/rrd_review", settings),
        ("gaussians/summary", metrics),
    ):
        recording.log(
            entity,
            rr.TextDocument(f"```json\n{json.dumps(document)}\n```"),
            static=True,
        )
    _log_images(recording, sources, cap, pattern, defect)
    recording.flush()
    recording.disconnect()
    return settings


def _assert_recording(helpers, root):
    from npa.viz.recordings import load_recording

    chunks = list(load_recording(root / "reports/sim2real.rrd").chunks())
    frames = {"lens": {5, 11, 25, 50, 99}}
    helpers._assert_nurec_rrd_frames(root, list(reversed(chunks)), frames)


@pytest.mark.parametrize("cap", [24, 3, 2, 1, 0, -1])
@pytest.mark.parametrize("pattern", ["early", "late"])
def test_live_accepts_independent_endpoint_selections(helpers, tmp_path, cap, pattern):
    """Accept valid interiors. Args: fixtures/cap/pattern. Returns: None. Raises: AssertionError."""
    sources = _source_images(tmp_path)
    _recording(tmp_path, sources, cap, pattern=pattern)
    _assert_recording(helpers, tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_last",
        "missing_first",
        "missing_modality",
        "merged",
        "reset_ids",
        "wrong_reconstruction_image",
        "duplicate",
    ],
)
def test_live_rejects_corrupt_reconstruction_semantics(helpers, tmp_path, defect):
    """Reject source drift. Args: fixtures/defect. Returns: None. Raises: AssertionError."""
    sources = _source_images(tmp_path)
    cap = 3 if defect in {"missing_first", "missing_last"} else 0
    _recording(tmp_path, sources, cap, defect=defect)
    with pytest.raises(AssertionError):
        _assert_recording(helpers, tmp_path)


def test_live_rejects_ambiguous_source_identity(helpers, tmp_path):
    """Reject ambiguity. Args: fixtures. Returns: None. Raises: AssertionError."""
    sources = _source_images(tmp_path)
    _recording(tmp_path, sources, 0)
    original = next(iter(sources.values()))
    original.with_name("alternate-" + original.name).write_bytes(original.read_bytes())
    with pytest.raises(AssertionError, match="duplicate source"):
        _assert_recording(helpers, tmp_path)


def test_live_oracle_does_not_use_writer_identity_or_selection(
    helpers, tmp_path, monkeypatch
):
    """Keep readback independent. Args: fixtures. Returns: None. Raises: AssertionError."""
    from npa.workflows import data_factory_viz as viz

    sources = _source_images(tmp_path)
    _recording(tmp_path, sources, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("writer helper used by readback")

    for name in ("_grouped_images", "_subsample", "_frame_index"):
        monkeypatch.setattr(viz, name, forbidden)
    _assert_recording(helpers, tmp_path)
