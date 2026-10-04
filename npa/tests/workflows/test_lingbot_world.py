"""Exercise durable LingBot World Base (Cam) continuation-stage contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import pytest

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.solutions.lingbot_camera import FRAME_COUNT, create_camera_controls
from npa.workflows import lingbot_world


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/lingbot-world-controlled-continuation.yaml"


class MemoryStorage:
    """Store S3-shaped artifacts in a temporary directory for workflow tests."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def upload_directory(
        self, local_dir: str, uri: str, *, require_empty: bool = False
    ) -> str:
        target = self._path(uri)
        if require_empty and target.exists():
            raise RuntimeError("output prefix already exists")
        shutil.copytree(local_dir, target)
        return uri

    def download_file(self, uri: str, destination: str) -> str:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._path(uri), target)
        return str(target)

    def _path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        assert parsed.scheme == "s3" and parsed.netloc
        return self.root / parsed.netloc / parsed.path.lstrip("/")


def _write_video(path: Path, direction: float) -> None:
    """Encode a nonstationary 161-frame MP4 for native-inference replacement."""

    import av

    output = av.open(str(path), "w")
    stream = output.add_stream("libx264", rate=16)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    try:
        for index in range(FRAME_COUNT):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = (index * 3) % 255
            pixels[:, :, 1] = 32 if direction > 0 else 192
            pixels[:, :, 2] = (index * 7 + (0 if direction > 0 else 96)) % 255
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(pixels, format="rgb24")
            ):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()


def _fake_generation(
    image: Path,
    _prompt: str,
    _seed: int,
    output: Path,
    _degree: int,
    controls_dir: Path,
) -> dict:
    """Write native-shaped output while preserving the supplied control artifacts."""

    output.mkdir()
    shutil.copytree(controls_dir, output / "controls")
    poses = np.load(controls_dir / "poses.npy", allow_pickle=False)
    _write_video(output / "video.mp4", float(poses[-1, 0, 3]))
    return {
        "solution": "lingbot-world",
        "capability": "lingbot_world_camera_conditioned_video",
        "input_sha256": lingbot_world.file_sha256(image),
        "controls": {
            "poses.npy": lingbot_world.file_sha256(output / "controls" / "poses.npy")
        },
        "observed": {"sha256": lingbot_world.file_sha256(output / "video.mp4")},
    }


def _copy_input(source: Path):
    """Return a checksum-aware replacement for the remote BYOF input fetch."""

    def copy(_uri: str, expected_sha256: str, destination: Path) -> Path:
        assert expected_sha256 == lingbot_world.file_sha256(source)
        shutil.copy2(source, destination)
        return destination

    return copy


def _input_provenance(image: Path) -> dict[str, object]:
    """Build a complete selected-byte provenance record for one fixture image."""

    digest = lingbot_world.file_sha256(image)
    return {
        "schema": lingbot_world.MEDIA_PROVENANCE_SCHEMA,
        "selected_input": {
            "sha256": digest,
            "media_type": "image/png",
            "dimensions": [48, 32],
        },
        "source": {
            "title": "Workflow test image",
            "url": "https://example.invalid/lingbot-test-image",
            "author": "NPA test fixture",
            "license": "Apache-2.0",
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "sha256": digest,
        },
        "derivation": {"modification": "No transformation before the fixture upload."},
        "redistribution_decision": "Test-only fixture; not distributed with a runtime image.",
    }


def _write_input_provenance(
    storage: MemoryStorage, uri: str, image: Path
) -> None:
    """Publish one fixture provenance record where the preparation stage expects it."""

    target = storage._path(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_input_provenance(image)), encoding="utf-8")


def test_camera_controls_are_native_shape_and_matched(tmp_path: Path) -> None:
    """Paired authored trajectories retain LingBot's documented array contract."""

    prescribed = tmp_path / "prescribed"
    alternative = tmp_path / "alternative"
    create_camera_controls(
        prescribed, translation_x=0.7, translation_z=1.5, yaw_radians=0.1
    )
    create_camera_controls(
        alternative, translation_x=-0.7, translation_z=1.5, yaw_radians=-0.1
    )
    first = np.load(prescribed / "poses.npy", allow_pickle=False)
    second = np.load(alternative / "poses.npy", allow_pickle=False)
    intrinsics = np.load(prescribed / "intrinsics.npy", allow_pickle=False)
    assert first.shape == (FRAME_COUNT, 4, 4)
    assert intrinsics.shape == (FRAME_COUNT, 4)
    assert first[-1, 0, 3] == -second[-1, 0, 3]
    assert first[-1, 2, 3] == second[-1, 2, 3]


def test_workflow_has_five_real_stages_with_exact_artifact_handoffs() -> None:
    """Plan the checked-in graph and require every durable control handoff.

    This is intentionally a graph-level assertion in addition to the media test:
    a shell wrapper around one model call cannot satisfy the connected-stage
    contract by merely adding named workflow states.
    """

    plan = build_plan(load_spec(SPEC), run_id="lingbot-contract")
    prepare, prescribed, alternative, evaluate, visualize = plan.steps
    assert [step.state for step in plan.steps] == [
        "prepare",
        "generate-prescribed",
        "generate-alternative",
        "evaluate-control-response",
        "visualize",
    ]
    assert all("npa.workflows.lingbot_world" in step.shell for step in plan.steps)
    assert "prepare" in prepare.shell
    assert "--input-provenance-uri" in prepare.shell
    assert "generate" in prescribed.shell and "--controls prescribed" in prescribed.shell
    assert "generate" in alternative.shell and "--controls alternative" in alternative.shell
    assert "evaluate" in evaluate.shell
    assert "visualize" in visualize.shell
    spec = load_spec(SPEC)
    assert spec.config["gpu_type"] == "B200"
    assert spec.config["gpu_count"] == 8
    assert spec.resources["lingbot"]["accelerators"] == (
        "{{config.gpu_type}}:{{config.gpu_count}}"
    )
    assert prescribed.resources_profile["accelerators"] == "B200:8"
    assert alternative.resources_profile["accelerators"] == "B200:8"
    rtx_compatibility = build_plan(
        merge_config_overrides(
            load_spec(SPEC),
            {
                "gpu_type": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                "gpu_count": "8",
            },
        ),
        run_id="lingbot-rtx-compatibility",
    )
    assert [
        step.resources_profile["accelerators"]
        for step in rtx_compatibility.steps
        if step.state.startswith("generate-")
    ] == [
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:8",
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:8",
    ]
    assert "--degree 8" in prescribed.shell
    assert "--degree 8" in alternative.shell
    assert {item["schema"] for item in prepare.inputs} == {
        "application/json",
        "image/jpeg",
    }
    prepared_uri = prepare.outputs[0]["uri"]
    prescribed_uri = prescribed.outputs[0]["uri"]
    alternative_uri = alternative.outputs[0]["uri"]
    report_uri = evaluate.outputs[0]["uri"]
    assert prescribed.inputs[0]["uri"] == prepared_uri
    assert {item["uri"] for item in alternative.inputs} == {
        prepared_uri,
        prescribed_uri,
    }
    assert {item["uri"] for item in evaluate.inputs} == {
        prescribed_uri,
        alternative_uri,
    }
    assert {item["uri"] for item in visualize.inputs} == {
        prescribed_uri,
        alternative_uri,
        report_uri,
    }
    assert {item["schema"] for item in visualize.outputs} == {
        "npa.workbench.lingbot_world.controlled_continuation.v1",
        "video/mp4",
        "application/vnd.rerun.rrd",
    }
    readiness = json.loads(SPEC.with_suffix(".readiness.json").read_text())
    assert readiness["workflow_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC.name)
    assert case.tier == "gpu"
    assert not case.plan_only
    assert case.rotation_skip
    assert "eight" in case.skip_reason


def test_connected_stages_publish_and_reconsume_real_media(
    monkeypatch, tmp_path: Path
) -> None:
    """Every stage consumes predecessor artifacts and publishes its declared output."""

    source = tmp_path / "context.png"
    Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8)).save(source)
    storage = MemoryStorage(tmp_path / "objects")
    monkeypatch.setattr(lingbot_world, "_storage", lambda: storage)
    monkeypatch.setattr(lingbot_world, "download_input", _copy_input(source))
    monkeypatch.setattr(lingbot_world, "generate_camera_video", _fake_generation)
    prepared_uri = "s3://unit/runs/demo/prepared/manifest.json"
    provenance_uri = "s3://unit/inputs/context.provenance.json"
    prescribed_uri = "s3://unit/runs/demo/prescribed/manifest.json"
    alternative_uri = "s3://unit/runs/demo/alternative/manifest.json"
    report_uri = "s3://unit/runs/demo/reports/evaluation/control-response.json"
    visualization_uri = "s3://unit/runs/demo/reports/visualization/manifest.json"
    _write_input_provenance(storage, provenance_uri, source)
    prepared = lingbot_world.prepare_context(
        "s3://unit/inputs/context.png",
        lingbot_world.file_sha256(source),
        provenance_uri,
        prepared_uri,
        "demo",
    )
    prescribed = lingbot_world.generate_continuation(
        prepared_uri,
        prescribed_uri,
        "A controlled camera movement through a hangar.",
        7,
        8,
        "prescribed",
        "prescribed",
        "demo",
    )
    alternative = lingbot_world.generate_continuation(
        prepared_uri,
        alternative_uri,
        "A controlled camera movement through a hangar.",
        7,
        8,
        "alternative",
        "alternative",
        "demo",
        prescribed_uri,
    )
    report = lingbot_world.evaluate_control_response(
        prescribed_uri, alternative_uri, report_uri, "demo"
    )
    visualization = lingbot_world.visualize_comparison(
        prescribed_uri, alternative_uri, report_uri, visualization_uri, "demo"
    )
    assert prepared["input"]["context_sha256"]
    assert prepared["input"]["media_provenance"]["selected_input"]["sha256"] == (
        lingbot_world.file_sha256(source)
    )
    assert prepared["input"]["media_provenance"]["copied_uri"] == (
        "s3://unit/runs/demo/prepared/input-provenance.json"
    )
    assert prepared["upstream"]["source"]["license_url"].endswith("LICENSE.txt")
    assert prepared["upstream"]["citation_url"].endswith("README.md#-citation")
    assert (
        prescribed["native_generation"]["capability"]
        == "lingbot_world_camera_conditioned_video"
    )
    assert alternative["control"] == "alternative"
    assert alternative["consumed_prescribed"]["manifest_uri"] == prescribed_uri
    assert alternative["consumed_prescribed"]["video_sha256"] == prescribed[
        "native_generation"
    ]["observed"]["sha256"]
    assert report["visual_response"]["mean_absolute_rgb_delta"] > 0.001
    assert visualization["comparison_mp4"]["frame_count"] == FRAME_COUNT
    assert visualization["comparison_rrd"]["frames"] == FRAME_COUNT
    assert storage._path(
        "s3://unit/runs/demo/reports/visualization/comparison.mp4"
    ).is_file()
    assert storage._path(
        "s3://unit/runs/demo/reports/visualization/comparison.rrd"
    ).is_file()


def test_preparation_rejects_provenance_for_different_input(
    monkeypatch, tmp_path: Path
) -> None:
    """Preparation cannot make an unbound image look rights-cleared downstream."""

    source = tmp_path / "context.png"
    Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8)).save(source)
    storage = MemoryStorage(tmp_path / "objects")
    monkeypatch.setattr(lingbot_world, "_storage", lambda: storage)
    provenance_uri = "s3://unit/inputs/context.provenance.json"
    _write_input_provenance(storage, provenance_uri, source)
    path = storage._path(provenance_uri)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["selected_input"]["sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(
        lingbot_world.LingBotWorldStageError,
        match="does not bind input_sha256",
    ):
        lingbot_world.prepare_context(
            "s3://unit/inputs/context.png",
            lingbot_world.file_sha256(source),
            provenance_uri,
            "s3://unit/runs/demo/prepared/manifest.json",
            "demo",
        )


def test_controlled_contract_rejects_legacy_four_rank_topology() -> None:
    """The SwitchWorld contract cannot silently use the legacy smoke shape."""

    with pytest.raises(lingbot_world.LingBotWorldStageError, match="8-rank"):
        lingbot_world.generate_continuation(
            "s3://unit/prepared/manifest.json",
            "s3://unit/prescribed/manifest.json",
            "prompt",
            1,
            4,
            "prescribed",
            "prescribed",
            "demo",
        )


def test_alternative_generation_rejects_unmatched_context(
    monkeypatch, tmp_path: Path
) -> None:
    """Matched-control evaluation never compares outputs from different contexts."""

    storage = MemoryStorage(tmp_path / "objects")
    monkeypatch.setattr(lingbot_world, "_storage", lambda: storage)
    prepared = {
        "schema": lingbot_world.SCHEMA,
        "stage": "prepare",
        "input": {"context_uri": "s3://unit/context.png", "context_sha256": "a" * 64},
        "controls": {"prescribed": {}, "alternative": {}},
    }
    previous = {
        "schema": lingbot_world.SCHEMA,
        "stage": "generate",
        "role": "prescribed",
        "input": {"context_uri": "s3://unit/other.png", "context_sha256": "b" * 64},
        "video_uri": "s3://unit/previous.mp4",
    }
    prepared_path = storage._path("s3://unit/prepared/manifest.json")
    previous_path = storage._path("s3://unit/previous/manifest.json")
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    previous_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(json.dumps(prepared))
    previous_path.write_text(json.dumps(previous))
    with pytest.raises(lingbot_world.LingBotWorldStageError, match="context differs"):
        lingbot_world.generate_continuation(
            "s3://unit/prepared/manifest.json",
            "s3://unit/alternative/manifest.json",
            "prompt",
            1,
            8,
            "alternative",
            "alternative",
            "demo",
            "s3://unit/previous/manifest.json",
        )
