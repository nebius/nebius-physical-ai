"""The NuRec run recording must be viewable in the NPA agent's Rerun panel.

Covers the three properties the agent depends on:

1. ``build_run_rrd`` logs the neural-reconstruction entities (novel views, NRE
   validation renders, Gaussian quality) into a real ``.rrd``;
2. the resulting bytes satisfy ``recording_has_run_entities`` and are NOT
   classified as the stock franka demo;
3. adding the NuRec markers did not weaken stock-demo detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.cli.agent_recordings import (
    DEMO_MARKERS,
    RUN_ENTITY_MARKERS,
    is_stock_demo_recording,
    recording_has_run_entities,
)
from npa.workflows.data_factory_viz import (
    RUN_SUBDIRS,
    _grouped_images,
    _input_entity,
    _load_nurec_docs,
    build_run_rrd,
)

NUREC_MARKERS = (b"novel_view", b"reconstruction/", b"gaussians", b"nurec")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def _nurec_run(root: Path) -> Path:
    """A synthetic run tree shaped exactly like the workflow's S3 layout."""
    # Real capture frames exported by export-ncore-benchmark-gt (JPEG, nested).
    for index in range(3):
        _write_image(root / "input" / "camera_images" / "camera2" / f"00000{index}.jpg", (10, 20, 30))
    # Novel views rendered from the trained Gaussians (nre render -> <sensor>/<frame>).
    for index in range(4):
        _write_image(root / "novel_views" / "camera2" / f"00000{index}.png", (40, 50, 60))
    # NRE's own validation renders.
    _write_image(root / "reconstruction" / "val" / "frame_000" / "rgb.png", (70, 80, 90))
    (root / "ncore").mkdir(parents=True, exist_ok=True)
    (root / "ncore" / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "nvidia/PhysicalAI-NuRec-PPISP",
                "scene": "struktur28",
                "variant": "auto",
                "shard_count": 4,
                "camera_ids": ["camera1", "camera2"],
                "lidar_ids": ["virtual_lidar"],
                "rig_derivation": {
                    "reference_camera": "camera2",
                    "pose_count": 38,
                    "poses_component_group": "npa_rig",
                },
            }
        )
    )
    (root / "reconstruction" / "metrics.yaml").write_text(
        "test:\n  psnr: 31.2\n  ssim: 0.93\n  lpips: 0.09\n"
    )
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "final.json").write_text(
        json.dumps({"artifact_count": 12, "capability": "neural-reconstruction"})
    )
    return root


def test_run_subdirs_cover_the_neural_reconstruction_stages() -> None:
    for stage in ("ncore", "reconstruction", "novel_views", "input", "reports"):
        assert stage in RUN_SUBDIRS


def test_build_run_rrd_logs_nurec_entities(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    run = _nurec_run(tmp_path / "neural-reconstruction-struktur28-20260731t170500z")
    out = tmp_path / "reports" / "sim2real.rrd"

    result = build_run_rrd(str(run), str(out), app_id="neural-reconstruction")

    assert result["status"] == "completed"
    # 3 capture frames + 4 novel views + 1 validation render.
    assert result["frames_logged"] == 8
    assert out.is_file() and out.stat().st_size > 0


def test_recording_bytes_carry_the_nurec_run_entities(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    run = _nurec_run(tmp_path / "neural-reconstruction-toro-20260731t170500z")
    out = tmp_path / "reports" / "sim2real.rrd"

    build_run_rrd(str(run), str(out), app_id="neural-reconstruction")
    data = out.read_bytes()

    for marker in (b"novel_view", b"reconstruction", b"gaussians"):
        assert marker in data, marker
    # The agent only serves a recording as run data when this holds.
    assert recording_has_run_entities(data) is True
    # And it must never be mistaken for the stock franka demo.
    assert is_stock_demo_recording(data) is False


def test_nurec_markers_are_registered_for_the_agent_finish_guard() -> None:
    for marker in NUREC_MARKERS:
        assert marker in RUN_ENTITY_MARKERS, marker


def test_stock_demo_detection_still_works_after_adding_nurec_markers() -> None:
    # Geometry-only demo payload: no run entities, so it stays classified as demo.
    demo = b"".join(DEMO_MARKERS)

    assert recording_has_run_entities(demo) is False
    assert is_stock_demo_recording(demo) is True


def test_nurec_markers_do_not_appear_in_the_demo_markers() -> None:
    for marker in NUREC_MARKERS:
        assert all(marker not in demo for demo in DEMO_MARKERS), marker


def test_input_entity_groups_by_directory_for_nested_captures(tmp_path: Path) -> None:
    root = tmp_path / "input"
    nested = root / "camera_images" / "camera2" / "000001.jpg"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")

    assert _input_entity(nested, root) == "camera2"


def test_input_entity_keeps_the_data_factory_clip_prefix(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    flat = root / "video_0_frame_01.png"
    flat.write_text("x")

    assert _input_entity(flat, root) == "video_0"


def test_grouped_images_buckets_by_parent_directory(tmp_path: Path) -> None:
    _write_image(tmp_path / "camera1" / "000000.png", (1, 2, 3))
    _write_image(tmp_path / "camera2" / "000000.png", (4, 5, 6))
    _write_image(tmp_path / "loose.png", (7, 8, 9))

    groups = _grouped_images(tmp_path)

    assert set(groups) == {"camera1", "camera2", "frames"}


def test_nurec_stage_docs_describe_the_capture_metrics_and_novel_views(tmp_path: Path) -> None:
    run = _nurec_run(tmp_path / "run")
    stage_log: list[str] = []

    docs = _load_nurec_docs(run, stage_log)

    assert "pipeline/1_ncore" in docs
    assert "struktur28" in docs["pipeline/1_ncore"]
    # The derivation of the rig frame is part of the run's provenance.
    assert "camera2" in docs["pipeline/1_ncore"]
    assert "npa_rig" in docs["pipeline/1_ncore"]
    assert "pipeline/2_reconstruct" in docs
    assert "31.2" in docs["pipeline/2_reconstruct"]
    # gaussians/* is one of the markers the agent scans for.
    assert "gaussians/summary" in docs
    assert "pipeline/3_novel_views" in docs
    assert "novel_view/camera2" in docs["pipeline/3_novel_views"]
    assert stage_log


def test_nurec_stage_docs_are_empty_for_a_data_factory_run(tmp_path: Path) -> None:
    # The additions must be completely inert for the other producer.
    run = tmp_path / "df-run"
    (run / "input").mkdir(parents=True)

    assert _load_nurec_docs(run, []) == {}


# ---------------------------------------------------------------------------------
# agent-side classification of the recording
# ---------------------------------------------------------------------------------
# These helpers live INSIDE the agent bootstrap template string (agent.py is
# rendered into the agent VM backend, so they are not importable module
# attributes). The established pattern for guarding them is source inspection --
# see npa/tests/cli/test_agent.py::
# test_data_factory_recording_note_wired_in_apply_loaded_artifact.
def _agent_source() -> str:
    from npa.cli import agent as agent_module

    return Path(agent_module.__file__).read_text(encoding="utf-8")


def test_recording_identity_classifies_a_nurec_run() -> None:
    """Now a real importable function, not a source scan.

    Producer identity lives in agent_recordings.py -- next to the entity-marker
    scan it belongs with, and off the agent module's size ratchet.
    """
    from npa.cli.agent_recordings import (
        is_neural_reconstruction_recording,
        is_pipeline_recording,
    )

    key = (
        "checkpoints/neural-reconstruction/"
        "neural-reconstruction-struktur28-20260731t051728z/reports/sim2real.rrd"
    )
    assert is_neural_reconstruction_recording(key) is True
    assert is_pipeline_recording(key) is True


def test_recording_identity_does_not_claim_the_other_producers() -> None:
    from npa.cli.agent_recordings import is_neural_reconstruction_recording

    for key in (
        "checkpoints/sim2real-b/s2r-real-0725t222636z/reports/sim2real.rrd",
        "checkpoints/physical-ai-data-factory/paidf-demo/reports/sim2real.rrd",
    ):
        assert is_neural_reconstruction_recording(key) is False


def test_recording_identity_matches_a_path_segment_not_a_substring() -> None:
    from npa.cli.agent_recordings import is_neural_reconstruction_recording

    # No `<id>/` segment: a prefix that merely ENDS with the phrase must not match.
    assert (
        is_neural_reconstruction_recording(
            "checkpoints/neural-reconstruction-only/reports/sim2real.rrd"
        )
        is False
    )
    # And a non-recording key is never a match, whatever the prefix says.
    assert (
        is_neural_reconstruction_recording(
            "checkpoints/neural-reconstruction/run/reports/final.json"
        )
        is False
    )


def test_agent_wires_the_reconstruction_branch_before_the_generic_one() -> None:
    """The branch order still has to be checked in the bootstrap template source.

    _apply_loaded_artifact lives inside the agent bootstrap string, so it is not an
    importable attribute. If the generic Sim2Real branch ran first, a NuRec run
    would claim a held-out-simulation camera it does not have.
    """
    source = _agent_source()

    assert "elif is_neural_reconstruction_recording(key):" in source
    assert source.index("elif is_neural_reconstruction_recording(key):") < source.index(
        "elif _is_sim2real_pipeline_recording(key):"
    )
    assert "NEURAL_RECONSTRUCTION_PREVIEW_ENTITY" in source
    assert "NEURAL_RECONSTRUCTION_VIEWER_NOTE" in source


def test_agent_does_not_relabel_the_nurec_camera_as_heldout_sim() -> None:
    source = _agent_source()

    guard_start = source.index("camera = _sim2real_pipeline_camera_label(camera)")
    guard = source[max(0, guard_start - 400) : guard_start]
    assert "_is_data_factory_recording(key)" in guard
    assert "is_neural_reconstruction_recording(key)" in guard


def test_embedded_backend_defines_the_identity_helpers_before_using_them() -> None:
    """The bootstrap inlines agent_recordings.py; order matters.

    agent.py calls these names unqualified with no import, so they only resolve
    because the recordings module is spliced in above. If that splice ever moved
    below the usage, the agent VM would fail at import time -- far from the cause.
    """
    from npa.cli import agent as agent_module

    source = _agent_source()
    embed_marker = agent_module._AGENT_RECORDINGS_EMBED

    assert embed_marker in source
    assert source.index(embed_marker) < source.index(
        "elif is_neural_reconstruction_recording(key):"
    )
