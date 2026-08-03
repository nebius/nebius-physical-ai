"""Unit tests for optional input-conditioning in the Cosmos Transfer 2.5 runner.

These cover the code path that makes the augment a REAL augmentation of the
caller's input clip (edge control computed on-the-fly from ``video_path``). No
upstream fixture is used.
No GPU / cosmos runtime is touched; the inference subprocess is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import cosmos2
from npa.workbench.cosmos import transfer as tx

runner = CliRunner()


def _fake_env(monkeypatch, repo: Path):
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda r: Path("/usr/bin/python3"))
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")

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
    # A conditioned spec was written that points at the input clip.
    assert res["spec"]
    # The synthesized controlnet spec is ephemeral (removed after inference to
    # avoid accumulating in the repo dir); its content is returned for inspection.
    assert not (repo / res["spec"]).exists()
    spec = res["spec_json"]
    assert spec["video_path"] == str(clip.resolve())
    assert "edge" in spec
    assert spec["prompt"] == "foggy morning"


def test_run_cosmos_transfer_requires_input_or_explicit_spec(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    _fake_env(monkeypatch, repo)
    monkeypatch.delenv("COSMOS_TRANSFER_SPEC", raising=False)
    monkeypatch.delenv("COSMOS_TRANSFER_PROMPT", raising=False)

    with pytest.raises(ValueError, match="no upstream media is bundled"):
        tx.run_cosmos_transfer(run_id="r2")
    assert not list(repo.glob("_npa_input_spec_*.json"))


def test_run_cosmos_transfer_refuses_missing_token_before_env_or_download(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    called = False

    def fail_if_called(_repo: Path) -> Path:
        nonlocal called
        called = True
        raise AssertionError("environment/download path must not run")

    monkeypatch.setattr(tx, "ensure_env", fail_if_called)
    with pytest.raises(RuntimeError, match="no model download was attempted"):
        tx.run_cosmos_transfer(input_video=str(tmp_path / "missing.mp4"))
    assert called is False


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


def test_materialize_input_clip_empty_s3_prefix_is_no_video(monkeypatch) -> None:
    from npa.clients.storage import StorageClient

    materialized_dirs: list[Path] = []

    class EmptyPrefixStorage:
        def download_directory(self, _src: str, local_dir: str) -> str:
            materialized_dirs.append(Path(local_dir))
            assert materialized_dirs[-1].is_dir()
            return local_dir

    monkeypatch.setattr(StorageClient, "from_environment", lambda: EmptyPrefixStorage())

    assert cosmos2._materialize_input_clip("s3://test-bucket/empty/") == ""
    assert materialized_dirs and not materialized_dirs[0].exists()


@pytest.mark.parametrize("stage", ["authentication", "listing", "download"])
def test_materialize_input_clip_propagates_storage_failures(
    monkeypatch, stage: str
) -> None:
    from npa.clients.storage import StorageClient

    failure = PermissionError(f"{stage} failed: credential=must-not-surface")

    class FailingStorage:
        def download_directory(self, _src: str, _local_dir: str) -> str:
            raise failure

        def download_path(self, _src: str, _local_path: str) -> str:
            raise failure

    def from_environment():
        if stage == "authentication":
            raise failure
        return FailingStorage()

    monkeypatch.setattr(StorageClient, "from_environment", from_environment)
    source = "s3://test-bucket/input.mp4" if stage == "download" else "s3://test-bucket/input/"

    with pytest.raises(PermissionError) as caught:
        cosmos2._materialize_input_clip(source)
    assert caught.value is failure


def test_cli_materialization_error_is_sanitized_and_preserves_cause(monkeypatch) -> None:
    secret = "do-not-print-this-token"
    failure = PermissionError(f"access denied: token={secret}")
    source = f"s3://test-bucket/input/?token={secret}"

    def fail_materialize(_src: str) -> str:
        raise failure

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fail_materialize)

    with pytest.raises(cosmos2.typer.BadParameter) as caught:
        cosmos2._materialize_conditioning_input(source)
    assert isinstance(caught.value, cosmos2.typer.BadParameter)
    assert caught.value.__cause__ is failure
    message = str(caught.value)
    assert "could not inspect or download" in message
    assert "credentials" in message
    assert secret not in message


def test_transfer_cli_rejects_conditioning_without_input_video(monkeypatch) -> None:
    input_uri = "s3://test-bucket/run/input/"
    materialized: list[str] = []
    inference_called = False

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)

    def fake_materialize(src: str) -> str:
        materialized.append(src)
        return ""

    def fail_if_inferred(**_kwargs) -> dict:
        nonlocal inference_called
        inference_called = True
        raise AssertionError("inference must not run without an input video")

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fake_materialize)
    monkeypatch.setattr(tx, "run_cosmos_transfer", fail_if_inferred)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            input_uri,
            "--output-uri",
            "s3://test-bucket/run/augmented/",
            "--condition-on-input",
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert "no supported video" in result.output
    assert input_uri not in result.output
    assert materialized == [input_uri]
    assert inference_called is False


def test_transfer_cli_reports_storage_failure_without_secret(monkeypatch) -> None:
    secret = "do-not-print-this-token"
    input_uri = f"s3://test-bucket/run/input/?token={secret}"
    failure = PermissionError(f"access denied: credential={secret}")
    inference_called = False

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)

    def fail_materialize(_src: str) -> str:
        raise failure

    def fail_if_inferred(**_kwargs) -> dict:
        nonlocal inference_called
        inference_called = True
        raise AssertionError("inference must not run after an input-storage failure")

    monkeypatch.setattr(cosmos2, "_materialize_input_clip", fail_materialize)
    monkeypatch.setattr(tx, "run_cosmos_transfer", fail_if_inferred)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            input_uri,
            "--output-uri",
            "s3://test-bucket/run/augmented/",
            "--condition-on-input",
            "--execute",
        ],
    )

    assert result.exit_code != 0
    assert "could not inspect or download" in result.output
    assert "no supported video" not in result.output
    assert secret not in result.output
    assert inference_called is False


def test_detect_gpu_count_from_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    assert cosmos2._detect_gpu_count() == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Falls back to nvidia-smi (absent in CI) -> at least 1.
    assert cosmos2._detect_gpu_count() >= 1


def test_variant_parallelism_env_override_and_cap(monkeypatch) -> None:
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
