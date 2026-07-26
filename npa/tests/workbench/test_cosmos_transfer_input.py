"""Unit tests for optional input-conditioning in the Cosmos Transfer 2.5 runner.

These cover the code path that makes the augment a REAL augmentation of the
caller's input clip (edge control computed on-the-fly from ``video_path``) while
leaving the default bundled-example behavior — and the golden eval — unchanged.
No GPU / cosmos runtime is touched; the inference subprocess is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

from npa.workbench.cosmos import transfer as tx


def _fake_env(monkeypatch, repo: Path):
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda r: Path("/usr/bin/python3"))

    def fake_run(cmd, *args, **kwargs):
        cwd = Path(kwargs["cwd"])
        out = cmd[cmd.index("-o") + 1]
        outdir = cwd / out
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.mp4").write_bytes(b"y" * 200_001)
        return None

    monkeypatch.setattr(tx.subprocess, "run", fake_run)


def test_spec_for_input_video_builds_edge_control(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)
    rel, modality = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="rainy night, wet asphalt",
        control="edge",
        control_weight=0.8,
        guidance=4,
        name="run-1",
    )
    assert modality == "edge"
    spec = json.loads((repo / rel).read_text())
    assert spec["video_path"] == str(clip.resolve())
    assert spec["prompt"] == "rainy night, wet asphalt"
    assert spec["edge"] == {"control_weight": 0.8}
    assert spec["guidance"] == 4
    # depth/seg need a precomputed control file → fall back to edge for input-only.
    _rel2, modality2 = tx._spec_for_input_video(
        repo, input_video=str(clip), prompt="", control="depth",
        control_weight=1.0, guidance=3, name="run-2",
    )
    assert modality2 == "edge"


def test_run_cosmos_transfer_conditions_on_input(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    _fake_env(monkeypatch, repo)
    clip = tmp_path / "myinput.mp4"
    clip.write_bytes(b"x" * 1000)

    res = tx.run_cosmos_transfer(
        run_id="r1", input_video=str(clip), prompt="foggy morning", control="edge"
    )
    assert res["input_conditioned"] is True
    assert res["input_video"] == str(clip)
    assert res["control"] == "edge"
    assert Path(res["video_path"]).exists()
    # A conditioned spec was written that points at the input clip, not DEFAULT_SPEC.
    assert res["spec"] != tx.DEFAULT_SPEC
    # The synthesized controlnet spec is ephemeral (removed after inference to
    # avoid accumulating in the repo dir); its content is returned for inspection.
    assert not (repo / res["spec"]).exists()
    spec = res["spec_json"]
    assert spec["video_path"] == str(clip.resolve())
    assert "edge" in spec
    assert spec["prompt"] == "foggy morning"


def test_run_cosmos_transfer_default_uses_bundled_spec(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    _fake_env(monkeypatch, repo)
    monkeypatch.delenv("COSMOS_TRANSFER_SPEC", raising=False)
    monkeypatch.delenv("COSMOS_TRANSFER_PROMPT", raising=False)

    res = tx.run_cosmos_transfer(run_id="r2")
    assert res["input_conditioned"] is False
    assert res["spec"] == tx.DEFAULT_SPEC
    assert not list(repo.glob("_npa_input_spec_*.json"))


def test_publish_marks_real_gpu_mode_and_conditioning(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    recorded: dict[str, str] = {}

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            if uri.endswith("manifest.json"):
                recorded["manifest"] = Path(local).read_text()
            elif uri.endswith("metadata.json"):
                recorded["metadata"] = Path(local).read_text()
            return uri

    manifest = tx.publish_transfer_to_s3(
        {
            "video_path": str(video),
            "video_bytes": 200_000,
            "spec": "_npa_input_spec_r1.json",
            "input_conditioned": True,
            "input_video": "/tmp/robot_input.mp4",
            "control": "edge",
        },
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"weather": "rainy"},
        storage_client=FakeStorage(),
    )
    assert manifest["mode"] == "cosmos_transfer2.5_gpu"
    assert manifest["input_conditioned"] is True
    assert manifest["conditioned_input"] == "robot_input.mp4"
    assert manifest["control"] == "edge"
    meta = json.loads(recorded["metadata"])
    assert meta["mode"] == "cosmos_transfer2.5_gpu"
    assert meta["input_conditioned"] is True
    assert meta["conditioned_input"] == "robot_input.mp4"


def test_multi_variant_publish_writes_one_clip_per_combo(tmp_path: Path, monkeypatch) -> None:
    """publish_transfer_clip (per combo) + write_run_manifest (once) must emit one
    clip dir per sampled scenario and a run manifest that records the fan-out."""
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    uploaded: list[str] = []

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            uploaded.append(uri)
            return uri

    storage = FakeStorage()
    combos = [
        {"cloth_color": "blue", "prompt": "a blue cloth, bright daylight"},
        {"cloth_color": "red", "prompt": "a red cloth, dim evening light"},
        {"cloth_color": "white", "prompt": "a white cloth, cool overhead light"},
    ]
    clips = []
    for i, combo in enumerate(combos):
        video = tmp_path / f"out{i}.mp4"
        video.write_bytes(b"x" * 200_000)
        clips.append(
            tx.publish_transfer_clip(
                {"video_path": str(video), "video_bytes": 200_000, "spec": f"spec{i}",
                 "input_conditioned": True, "input_video": "/tmp/robot.mp4", "control": "edge"},
                "s3://bkt/run1/cosmos_augmented/",
                run_id="run1",
                clip_name=f"aug-run1-{i}",
                variables=combo,
                storage_client=storage,
            )
        )
    manifest = tx.write_run_manifest(clips, "s3://bkt/run1/cosmos_augmented/", run_id="run1", storage_client=storage)

    assert manifest["variant_count"] == 3
    assert manifest["multiply_mode"] == "multi-variant"
    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1", "aug-run1-2"]
    assert len(manifest["variants"]) == 3
    assert manifest["variants"][1]["prompt"] == "a red cloth, dim evening light"
    # One metadata.json + one augmented_video.mp4 per clip, plus one run manifest.
    assert sum(u.endswith("metadata.json") for u in uploaded) == 3
    assert sum(u.endswith("augmented_video.mp4") for u in uploaded) == 3
    assert sum(u.endswith("cosmos_augmented/manifest.json") for u in uploaded) == 1


def test_single_variant_manifest_reports_single_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])
    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            return uri

    manifest = tx.publish_transfer_to_s3(
        {"video_path": str(video), "video_bytes": 200_000, "spec": "s"},
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"cloth_color": "blue"},
        storage_client=FakeStorage(),
    )
    assert manifest["variant_count"] == 1
    assert manifest["multiply_mode"] == "single-variant"
    assert manifest["clips"] == ["aug-run1"]


def test_materialize_input_clip_local_path(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _materialize_input_clip

    clip = tmp_path / "local.mp4"
    clip.write_bytes(b"x" * 10)
    assert _materialize_input_clip(str(clip)) == str(clip)
    assert _materialize_input_clip("") == ""
    assert _materialize_input_clip(str(tmp_path / "missing.mp4")) == ""


def test_detect_gpu_count_from_cuda_visible_devices(monkeypatch) -> None:
    from npa.cli.workbench import cosmos2

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cosmos2._detect_gpu_count() == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Falls back to nvidia-smi (absent in CI) -> at least 1.
    assert cosmos2._detect_gpu_count() >= 1


def test_variant_parallelism_env_override_and_cap(monkeypatch) -> None:
    from npa.cli.workbench import cosmos2

    monkeypatch.setenv("NPA_COSMOS_VARIANT_PARALLELISM", "4")
    # Capped at the number of variants so we never spawn idle workers.
    assert cosmos2._variant_parallelism(2) == 2
    assert cosmos2._variant_parallelism(8) == 4
    monkeypatch.delenv("NPA_COSMOS_VARIANT_PARALLELISM", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cosmos2._variant_parallelism(6) == 4
    assert cosmos2._variant_parallelism(1) == 1


def test_run_cosmos_transfer_pins_gpu_and_unique_spec(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    # A default spec the prompt override copies from.
    spec_path = repo / "spec.json"
    spec_path.write_text(json.dumps({"prompt": "orig", "video_path": "x"}), encoding="utf-8")
    _fake_env(monkeypatch, repo)

    seen_env: dict[str, str] = {}

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001
        cwd = Path(kwargs["cwd"])
        out = cmd[cmd.index("-o") + 1]
        outdir = cwd / out
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "result.mp4").write_bytes(b"y" * 200_001)
        seen_env["CUDA_VISIBLE_DEVICES"] = kwargs["env"].get("CUDA_VISIBLE_DEVICES", "")
        return None

    monkeypatch.setattr(tx.subprocess, "run", fake_run)

    res = tx.run_cosmos_transfer(
        run_id="run1-v2",
        spec="spec.json",
        prompt="a red cloth",
        cuda_visible_devices="2",
        variant_tag="run1-v2",
    )
    assert seen_env["CUDA_VISIBLE_DEVICES"] == "2"
    # Variant-tagged patched spec keeps concurrent siblings from clobbering.
    assert "run1-v2" in res["spec"]
    assert Path(res["video_path"]).exists()


def test_write_run_manifest_records_variant_parallelism(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tx, "extract_frames", lambda vp, dest, max_frames=8: [])

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            return uri

    storage = FakeStorage()
    clips = []
    for i in range(4):
        video = tmp_path / f"out{i}.mp4"
        video.write_bytes(b"x" * 200_000)
        clips.append(
            tx.publish_transfer_clip(
                {"video_path": str(video), "video_bytes": 200_000, "spec": f"spec{i}"},
                "s3://bkt/run1/cosmos_augmented/",
                run_id="run1",
                clip_name=f"aug-run1-{i}",
                variables={"prompt": f"scene {i}"},
                storage_client=storage,
            )
        )
    manifest = tx.write_run_manifest(
        clips, "s3://bkt/run1/cosmos_augmented/", run_id="run1",
        storage_client=storage, variant_parallelism=4,
    )
    assert manifest["variant_count"] == 4
    assert manifest["variant_parallelism"] == 4
    assert manifest["multiply_mode"] == "multi-variant"
