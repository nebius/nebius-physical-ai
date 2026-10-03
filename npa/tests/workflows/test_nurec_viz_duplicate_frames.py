"""Reject ambiguous source images before claiming a faithful NuRec recording."""

import hashlib

import pytest

from npa.workflows import data_factory_viz as viz


def _image(path, color):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


@pytest.mark.parametrize("cap", [1, 24])
@pytest.mark.parametrize(
    "relative",
    ["novel_views/front", "reconstruction/val/pred_rgb/front", "reconstruction"],
)
def test_nurec_recording_rejects_duplicate_source_frame_ids(
    tmp_path, monkeypatch, cap, relative
):
    """Reject ambiguity without changing input images.

    Args: tmp_path, monkeypatch, cap, relative: Fixture and review settings.
    Returns: None.
    Raises: AssertionError: A recording succeeds or source image bytes change.
    """
    pytest.importorskip("rerun")
    pytest.importorskip("PIL")
    monkeypatch.setattr(viz, "RRD_MAX_FRAMES_PER_ENTITY", cap)
    run = tmp_path / "ambiguous-review"
    first = run / relative / "000007.png"
    second = run / relative / "capture-000007.jpg"
    _image(first, (255, 0, 0))
    _image(second, (0, 0, 255))
    original = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)
    }

    with pytest.raises(viz.DataFactoryVizError, match="duplicate frame index 7"):
        viz.build_run_rrd(
            str(run), str(tmp_path / "review.rrd"), app_id="neural-reconstruction"
        )

    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in original
    } == original


def test_nurec_same_frame_id_remains_distinct_across_modality_and_camera(tmp_path):
    """Keep legitimate image tracks with the same source frame number.

    Args: tmp_path: Synthetic source directory.
    Returns: None.
    Raises: AssertionError: Modality or camera identities collapse.
    """
    pytest.importorskip("PIL")
    groups = ("val/pred_rgb/front", "val/pred_distance/front", "val/pred_rgb/rear")
    for index, group in enumerate(groups):
        _image(tmp_path / group / "000007.png", (index * 70, 20, 40))

    actual = viz._grouped_images(tmp_path)

    assert set(actual) == set(groups)
    assert all(
        paths == [tmp_path / group / "000007.png"] for group, paths in actual.items()
    )
