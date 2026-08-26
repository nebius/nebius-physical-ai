"""Segmentation-conditioned augmentation and region masks in Cosmos Transfer 2.5.

The pinned upstream revision computes edge/vis/seg controls on the fly from the
input clip -- ``seg`` via GroundingDINO-base + SAM2 -- while NPA requires depth
to be precomputed without restricted Video Depth Anything weights. It accepts a binary
spatiotemporal region mask per modality, either precomputed or generated from a
text prompt by SAM2. These tests cover the NPA surface for both: the controlnet
spec NPA writes, the artifacts it publishes, and the CLI/PAIDF wiring.

No GPU or Cosmos runtime is touched; the inference subprocess and storage client
are mocked.
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


class FakeStorage:
    """Records every upload so a test can assert the published layout."""

    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_file(self, local: str, uri: str) -> str:
        self.uploads[uri] = Path(local).read_bytes()
        return uri


def test_output_classifier_never_selects_control_evidence_as_generated(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "clip.mp4"
    control = tmp_path / "clip_control_seg.mp4"
    mask = tmp_path / "clip_mask_seg.mp4"
    unknown = tmp_path / "clip_control_evidence_extra.mp4"
    known_orphan = tmp_path / "orphan_control_depth.mp4"
    ordinary_orphan = tmp_path / "orphan_control_experiment.mp4"
    named_control = tmp_path / "control-surface.mp4"
    named_mask = tmp_path / "robot_mask_trial.mp4"
    for path, size in (
        (generated, 1),
        (control, 100),
        (mask, 200),
        (unknown, 300),
        (known_orphan, 350),
        (ordinary_orphan, 375),
        (named_control, 400),
        (named_mask, 500),
    ):
        path.write_bytes(b"x" * size)

    videos, controls, masks = tx._classify_output_videos(tmp_path)

    assert videos == sorted(
        str(path)
        for path in (generated, ordinary_orphan, named_control, named_mask)
    )
    assert controls == {"seg": str(control)}
    assert masks == {"seg": str(mask)}


def _fake_inference(monkeypatch, repo: Path, *, sidecars: dict[str, bytes] | None = None):
    """Run the real spec/argv path with a stubbed inference subprocess.

    ``sidecars`` are extra files the stub writes into the output directory, named
    the way upstream names its control and mask videos.
    """

    monkeypatch.setattr(tx, "cosmos_transfer_repo", lambda: repo)
    monkeypatch.setattr(tx, "ensure_env", lambda _repo: Path("/usr/bin/python3"))
    monkeypatch.setenv("HF_TOKEN", "unit-test-placeholder")
    specs: list[dict] = []

    def fake_run(cmd, *_args, **kwargs):
        spec_path = Path(kwargs["cwd"]) / cmd[cmd.index("-i") + 1]
        specs.append(json.loads(spec_path.read_text(encoding="utf-8")))
        outdir = Path(kwargs["cwd"]) / cmd[cmd.index("-o") + 1]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "npa_input.mp4").write_bytes(b"render" * 100)
        for name, body in (sidecars or {}).items():
            (outdir / name).write_bytes(body)

    monkeypatch.setattr(tx.subprocess, "run", fake_run)
    return specs


def test_seg_spec_asks_for_on_the_fly_segmentation_not_an_asset(tmp_path: Path) -> None:
    """`seg` needs no control file: upstream segments the input from a prompt."""

    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    rel, modality = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="a chrome robot arm in a bright warehouse",
        control="seg",
        control_weight=0.9,
        guidance=3,
        name="run-1",
        control_prompt="robot arm, conveyor, bin",
    )

    assert modality == "seg"
    spec = json.loads((repo / rel).read_text())
    assert spec["video_path"] == str(clip.resolve())
    assert spec["seg"] == {
        "control_weight": 0.9,
        "control_prompt": "robot arm, conveyor, bin",
    }
    assert "control_path" not in spec["seg"]


def test_a_region_mask_restricts_the_control_to_segmented_pixels(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)
    mask = tmp_path / "arm-mask.mp4"
    mask.write_bytes(b"mask")

    _rel, _modality = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="weathered steel",
        control="edge",
        control_weight=1.0,
        guidance=3,
        name="masked",
        mask_asset=str(mask),
    )
    spec = json.loads(next(repo.glob("_npa_input_spec_masked.json")).read_text())
    assert spec["edge"]["mask_path"] == str(mask.resolve())

    prompted, _ = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="weathered steel",
        control="seg",
        control_weight=1.0,
        guidance=3,
        name="prompted",
        mask_prompt="robot arm",
    )
    assert json.loads((repo / prompted).read_text())["seg"]["mask_prompt"] == "robot arm"


def test_a_precomputed_segmentation_map_can_replace_the_generated_one(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)
    seg = tmp_path / "seg.mp4"
    seg.write_bytes(b"seg")

    rel, _modality = tx._spec_for_input_video(
        repo,
        input_video=str(clip),
        prompt="a repainted cell",
        control="seg",
        control_weight=1.0,
        guidance=3,
        name="asset",
        control_asset=str(seg),
    )
    assert json.loads((repo / rel).read_text())["seg"]["control_path"] == str(seg.resolve())


def test_an_unsupported_modality_fails_instead_of_quietly_becoming_edge(
    tmp_path: Path,
) -> None:
    """The silent coercion this replaces meant a seg request rendered as edge."""

    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    with pytest.raises(tx.ControlModalityError, match="unsupported .* 'canny'"):
        tx._spec_for_input_video(
            repo,
            input_video=str(clip),
            prompt="",
            control="canny",
            control_weight=1.0,
            guidance=3,
            name="bad",
        )
    assert tx.resolve_control_modality("SEG ") == "seg"


def test_a_control_weight_outside_upstreams_range_fails_early(tmp_path: Path) -> None:
    """Upstream types control_weight ge=0.0 le=1.0 and validates it on the GPU."""

    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    with pytest.raises(tx.ControlModalityError, match="outside .* 0.0-1.0"):
        tx._spec_for_input_video(
            repo,
            input_video=str(clip),
            prompt="",
            control="seg",
            control_weight=1.5,
            guidance=3,
            name="heavy",
        )
    assert tx.resolve_control_weight("0.75") == 0.75
    assert tx.resolve_control_weight(0.0) == 0.0
    assert tx.resolve_control_modality("") == "edge"


def test_two_ways_of_naming_one_region_mask_are_rejected_together(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)
    mask = tmp_path / "mask.mp4"
    mask.write_bytes(b"mask")

    with pytest.raises(tx.ControlModalityError, match="not both"):
        tx._spec_for_input_video(
            repo,
            input_video=str(clip),
            prompt="",
            control="seg",
            control_weight=1.0,
            guidance=3,
            name="both",
            mask_asset=str(mask),
            mask_prompt="robot arm",
        )


def test_a_control_prompt_on_a_non_text_modality_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x" * 100)

    with pytest.raises(tx.ControlModalityError, match="not text-driven"):
        tx._spec_for_input_video(
            repo,
            input_video=str(clip),
            prompt="",
            control="edge",
            control_weight=1.0,
            guidance=3,
            name="edge-prompt",
            control_prompt="robot arm",
        )


def test_the_run_reports_the_control_and_mask_videos_upstream_wrote(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    clip = tmp_path / "input.mp4"
    clip.write_bytes(b"x" * 1000)
    specs = _fake_inference(
        monkeypatch,
        repo,
        sidecars={
            "npa_input_control_seg.mp4": b"seg-control",
            "npa_input_mask_seg.mp4": b"seg-mask",
        },
    )

    result = tx.run_cosmos_transfer(
        run_id="r1",
        input_video=str(clip),
        prompt="a chrome arm",
        control="seg",
        control_weight=0.7,
        control_prompt="robot arm",
        mask_prompt="robot arm",
        variant_tag="npa_input",
    )

    assert specs[0]["seg"]["control_prompt"] == "robot arm"
    assert specs[0]["seg"]["mask_prompt"] == "robot arm"
    assert result["control"] == "seg"
    assert result["control_weight"] == 0.7
    assert Path(result["control_videos"]["seg"]).read_bytes() == b"seg-control"
    assert Path(result["mask_videos"]["seg"]).read_bytes() == b"seg-mask"
    # The generated clip, not a sidecar.
    assert Path(result["video_path"]).name == "npa_input.mp4"


def test_a_large_region_mask_is_never_published_as_the_augmentation(
    tmp_path: Path, monkeypatch
) -> None:
    """A binary mask can outweigh a short render; size must not decide."""

    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    clip = tmp_path / "input.mp4"
    clip.write_bytes(b"x" * 1000)
    _fake_inference(
        monkeypatch,
        repo,
        sidecars={
            "npa_input_control_seg.mp4": b"c" * 5_000_000,
            "npa_input_mask_seg.mp4": b"m" * 5_000_000,
        },
    )

    result = tx.run_cosmos_transfer(
        run_id="r1", input_video=str(clip), control="seg", variant_tag="npa_input"
    )

    assert Path(result["video_path"]).name == "npa_input.mp4"
    assert result["video_bytes"] == len(b"render" * 100)


def test_control_artifacts_publish_beside_the_clips_not_inside_them(
    tmp_path: Path, monkeypatch
) -> None:
    """Nesting them under a variant dir would misdirect the evaluator.

    ``cosmos_evaluator`` treats every child directory of the augment prefix as a
    variant, and falls back to the alphabetically first PNG inside one when the
    variant has no mp4 -- which a nested ``control/`` would win.
    """

    video = tmp_path / "out.mp4"
    video.write_bytes(b"render")
    control = tmp_path / "out_control_seg.mp4"
    control.write_bytes(b"seg-control")
    mask = tmp_path / "out_mask_seg.mp4"
    mask.write_bytes(b"seg-mask")

    def fake_extract(source: str, dest: Path, *, max_frames: int = 8) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        frame = dest / "frame-00000.png"
        frame.write_bytes(b"png")
        return [frame]

    monkeypatch.setattr(tx, "extract_frames", fake_extract)
    storage = FakeStorage()

    clip = tx.publish_transfer_clip(
        {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "seg.json",
            "input_conditioned": True,
            "input_video": "/tmp/input.mp4",
            "control": "seg",
            "control_weight": 0.8,
            "control_prompt": "robot arm",
            "mask_prompt": "robot arm",
            "control_videos": {"seg": str(control)},
            "mask_videos": {"seg": str(mask)},
        },
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        clip_name="aug-run1-0",
        variables={"prompt": "a chrome arm"},
        control_output_uri="s3://bkt/run1/cosmos_control/",
        require_frames=True,
        storage_client=storage,
    )

    assert clip["control_uris"] == {
        "control_seg": "s3://bkt/run1/cosmos_control/aug-run1-0/control_seg.mp4",
        "mask_seg": "s3://bkt/run1/cosmos_control/aug-run1-0/mask_seg.mp4",
    }
    assert clip["control_frames"]["control_seg"] == [
        "s3://bkt/run1/cosmos_control/aug-run1-0/control_seg/frame-00000.png"
    ]
    assert storage.uploads[clip["control_uris"]["control_seg"]] == b"seg-control"
    assert storage.uploads[clip["control_uris"]["mask_seg"]] == b"seg-mask"
    # Nothing control-shaped landed under the augmented clip prefix.
    augmented = [u for u in storage.uploads if "/cosmos_augmented/" in u]
    assert sorted(augmented) == [
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4",
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/frame-00000.png",
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/metadata.json",
    ]
    meta = json.loads(storage.uploads[
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/metadata.json"
    ])
    assert meta["control"] == "seg"
    assert meta["control_prompt"] == "robot arm"
    assert meta["mask_prompt"] == "robot arm"
    assert meta["control_evidence"] == {"status": "published"}


def test_control_upload_failure_preserves_completed_generated_variant(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"generated")
    control = tmp_path / "out_control_seg.mp4"
    control.write_bytes(b"control")

    def fake_extract(_source: str, dest: Path, *, max_frames: int = 8) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        frame = dest / "frame-00000.png"
        frame.write_bytes(b"frame")
        return [frame]

    monkeypatch.setattr(tx, "extract_frames", fake_extract)

    class FailingControlStorage(FakeStorage):
        def __init__(self) -> None:
            super().__init__()
            self.order: list[str] = []

        def upload_file(self, local: str, uri: str) -> str:
            self.order.append(uri)
            if "/cosmos_control/" in uri:
                raise RuntimeError("signed-provider-detail-must-not-persist")
            return super().upload_file(local, uri)

    storage = FailingControlStorage()
    clip = tx.publish_transfer_clip(
        {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "seg.json",
            "control": "seg",
            "control_videos": {"seg": str(control)},
        },
        "s3://bkt/run1/cosmos_augmented/",
        clip_name="aug-run1-0",
        control_output_uri="s3://bkt/run1/cosmos_control/",
        require_frames=True,
        storage_client=storage,
    )

    core = "s3://bkt/run1/cosmos_augmented/aug-run1-0/"
    assert storage.order[:3] == [
        core + "augmented_video.mp4",
        core + "frame-00000.png",
        core + "metadata.json",
    ]
    assert storage.uploads[core + "augmented_video.mp4"] == b"generated"
    assert storage.uploads[core + "frame-00000.png"] == b"frame"
    metadata = json.loads(storage.uploads[core + "metadata.json"])
    assert metadata["control_uris"] == {}
    assert metadata["control_evidence"] == {
        "error_type": "RuntimeError",
        "status": "failed",
    }
    assert "signed-provider-detail" not in json.dumps(metadata)
    assert clip["control_uris"] == {}
    assert clip["control_evidence"]["status"] == "failed"

    # Core publication is complete enough for the run manifest to continue.
    manifest = tx.write_run_manifest(
        [clip],
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
    )
    assert manifest["variant_count"] == 1


def test_absent_control_signal_never_claims_evidence_was_published(
    tmp_path: Path, monkeypatch
) -> None:
    video = tmp_path / "out.mp4"
    video.write_bytes(b"generated")

    def fake_extract(_source: str, dest: Path, *, max_frames: int = 8) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        frame = dest / "frame-00000.png"
        frame.write_bytes(b"frame")
        return [frame]

    monkeypatch.setattr(tx, "extract_frames", fake_extract)
    storage = FakeStorage()
    clip = tx.publish_transfer_clip(
        {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "seg.json",
            "control": "seg",
            "control_videos": {"seg": str(tmp_path / "absent.mp4")},
        },
        "s3://bkt/run1/cosmos_augmented/",
        clip_name="aug-run1-0",
        control_output_uri="s3://bkt/run1/cosmos_control/",
        require_frames=True,
        storage_client=storage,
    )
    assert clip["control_uris"] == {}
    assert clip["control_evidence"] == {"status": "missing"}
    metadata = json.loads(
        storage.uploads[
            "s3://bkt/run1/cosmos_augmented/aug-run1-0/metadata.json"
        ]
    )
    assert metadata["control_evidence"] == {"status": "missing"}


def test_the_run_manifest_records_what_conditioned_the_batch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tx, "extract_frames", lambda *_a, **_k: [])
    storage = FakeStorage()
    clips = []
    for index in range(2):
        video = tmp_path / f"out{index}.mp4"
        video.write_bytes(b"render")
        clips.append(
            tx.publish_transfer_clip(
                {
                    "video_path": str(video),
                    "video_bytes": video.stat().st_size,
                    "spec": f"seg{index}.json",
                    "control": "seg",
                    "control_weight": 0.8,
                    "control_prompt": "robot arm",
                    "mask_prompt": "robot arm",
                },
                "s3://bkt/run1/cosmos_augmented/",
                run_id="run1",
                clip_name=f"aug-run1-{index}",
                variant_index=index,
                variables={"prompt": f"variant {index}"},
                storage_client=storage,
            )
        )

    manifest = tx.write_run_manifest(
        clips, "s3://bkt/run1/cosmos_augmented/", run_id="run1", storage_client=storage
    )

    assert manifest["control"] == "seg"
    assert manifest["control_weight"] == 0.8
    assert manifest["control_prompt"] == "robot arm"
    assert manifest["mask_prompt"] == "robot arm"
    assert [v["clip"] for v in manifest["variants"]] == ["aug-run1-0", "aug-run1-1"]


def test_an_edge_run_publishes_exactly_what_it_always_did(
    tmp_path: Path, monkeypatch
) -> None:
    """No control prefix and no sidecars: the artifact set must not change."""

    monkeypatch.setattr(tx, "extract_frames", lambda *_a, **_k: [])
    video = tmp_path / "out.mp4"
    video.write_bytes(b"render")
    storage = FakeStorage()

    clip = tx.publish_transfer_clip(
        {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "edge.json",
            "control": "edge",
        },
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        clip_name="aug-run1-0",
        variables={"prompt": "a red cloth"},
        storage_client=storage,
    )

    assert clip["control_uris"] == {}
    assert sorted(storage.uploads) == [
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4",
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/metadata.json",
    ]


def test_cli_refuses_an_unknown_modality_before_holding_the_gpu(monkeypatch) -> None:
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: pytest.fail("inference must not start for a bad modality"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/augment/",
            "--execute",
            "--condition-on-input",
            "--control",
            "segmentation",
        ],
    )

    assert result.exit_code != 0


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--control", "edge", "--control-prompt", "robot arm"], "not text-driven"),
        (["--control", "depth"], "requires an operator-owned precomputed"),
    ],
)
def test_cli_normalizes_deterministic_control_errors_before_runtime_or_storage(
    monkeypatch, args: list[str], expected: str
) -> None:
    monkeypatch.setattr(
        tx,
        "cosmos_transfer_available",
        lambda: pytest.fail("runtime probe must not run"),
    )
    monkeypatch.setattr(
        cosmos2,
        "_materialize_input_clip",
        lambda *_a, **_k: pytest.fail("input must not be materialized"),
    )
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/augment/",
            "--execute",
            "--condition-on-input",
            *args,
        ],
    )
    assert result.exit_code == 2
    assert expected in result.output
    assert "Traceback" not in result.output


def test_cli_refuses_an_out_of_range_control_weight_before_holding_the_gpu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: pytest.fail("inference must not start for a bad weight"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/augment/",
            "--execute",
            "--condition-on-input",
            "--control-weight",
            "2.0",
        ],
    )

    assert result.exit_code != 0
    # Rich wraps the message inside a box, so the phrase only reads contiguously
    # once the borders and line breaks are collapsed.
    output = " ".join(result.output.replace("│", " ").split())
    assert "is outside Cosmos Transfer's accepted range 0.0-1.0" in output


def test_cli_refuses_both_mask_forms_before_holding_the_gpu(monkeypatch) -> None:
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: pytest.fail("inference must not start with a conflicting mask"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/augment/",
            "--execute",
            "--condition-on-input",
            "--mask-asset",
            "s3://bkt/masks/arm.mp4",
            "--mask-prompt",
            "robot arm",
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_fails_when_a_named_control_asset_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_kwargs: pytest.fail("a missing asset must not fall back to on-the-fly"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/augment/",
            "--execute",
            "--condition-on-input",
            "--control",
            "seg",
            "--control-asset",
            str(tmp_path / "absent.mp4"),
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_failed_control_asset_download_removes_its_temporary_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    from npa.clients.storage import StorageClient

    scratch = tmp_path / "control-download"

    def make_scratch(*, prefix: str) -> str:
        assert prefix == "npa-cosmos-control-"
        scratch.mkdir()
        return str(scratch)

    class FailingStorage:
        def download_path(self, _uri: str, _local: str) -> str:
            raise PermissionError("denied")

    monkeypatch.setattr(tempfile, "mkdtemp", make_scratch)
    monkeypatch.setattr(StorageClient, "from_environment", lambda: FailingStorage())

    with pytest.raises(cosmos2.typer.BadParameter, match="could not download"):
        cosmos2._materialize_control_asset(
            "s3://bkt/controls/seg.mp4", label="--control-asset"
        )

    assert not scratch.exists()


def test_cli_threads_seg_conditioning_through_the_multiply_fan_out(monkeypatch) -> None:
    seen: list[dict] = []
    published: list[dict] = []
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(cosmos2, "_all_augmentations", lambda _uri: [{"prompt": "a chrome arm"}])
    monkeypatch.setattr(
        cosmos2, "_persist_generated_conditioning_clip", lambda *_a, **_k: ""
    )

    def fake_run(**kwargs):
        seen.append(kwargs)
        return {
            "video_path": "/tmp/generated.mp4",
            "video_bytes": 10,
            "spec": "seg.json",
            "input_conditioned": True,
            "input_video": "/tmp/in.mp4",
            "control": "seg",
            "control_weight": 0.6,
            "control_prompt": "robot arm",
            "mask_prompt": "robot arm",
        }

    def fake_publish(transfer, output_uri, **kwargs):
        published.append(kwargs)
        return {
            "clip": "aug-run-0",
            "variant_index": 0,
            "augmented_video_uri": f"{output_uri}aug-run-0/augmented_video.mp4",
            "frames_uri": f"{output_uri}aug-run-0/",
            "frames": [],
            "frame_count": 0,
            "control": "seg",
            "control_weight": 0.6,
            "control_prompt": "robot arm",
            "mask_prompt": "robot arm",
            "control_uris": {
                "control_seg": "s3://bkt/control/aug-run-0/control_seg.mp4"
            },
            "variables": kwargs.get("variables") or {},
        }

    monkeypatch.setattr(tx, "run_cosmos_transfer", fake_run)
    monkeypatch.setattr(tx, "publish_transfer_clip", fake_publish)
    monkeypatch.setattr(tx, "write_run_manifest", lambda clips, uri, **kw: tx.build_run_manifest(clips, **kw))

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "s3://bkt/cosmos_augmented/",
            "--configs-uri",
            "s3://bkt/configs/",
            "--run-id",
            "run",
            "--execute",
            "--condition-on-input",
            "--control",
            "seg",
            "--control-weight",
            "0.6",
            "--control-prompt",
            "robot arm",
            "--mask-prompt",
            "robot arm",
            "--control-output-uri",
            "s3://bkt/cosmos_control/",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen[0]["control"] == "seg"
    assert seen[0]["control_prompt"] == "robot arm"
    assert seen[0]["mask_prompt"] == "robot arm"
    assert published[0]["control_output_uri"] == "s3://bkt/cosmos_control/"
    payload = json.loads(result.output)
    # Prompt text and storage locations may be customer-derived. The invocation
    # and durable manifest assertions above prove they were threaded correctly;
    # the CLI response must not echo them.
    assert "control" not in payload
    assert "control_prompt" not in payload
    assert "mask_prompt" not in payload
    assert "control_output_uri" not in payload
    assert "control_uris" not in payload


def test_env_overrides_let_a_submit_switch_to_seg_without_new_argv(monkeypatch) -> None:
    """`NPA_COSMOS_*` is the established way to tune conditioning per submit."""

    seen: list[dict] = []
    monkeypatch.setenv("NPA_COSMOS_CONTROL", "seg")
    monkeypatch.setenv("NPA_COSMOS_CONTROL_PROMPT", "robot arm, bin")
    monkeypatch.setenv("NPA_COSMOS_MASK_PROMPT", "robot arm")
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")

    def fake_run(**kwargs):
        seen.append(kwargs)
        return {
            "video_path": "/tmp/generated.mp4",
            "video_bytes": 10,
            "spec": "seg.json",
            "input_conditioned": True,
            "input_video": "/tmp/in.mp4",
            "control": "seg",
        }

    monkeypatch.setattr(tx, "run_cosmos_transfer", fake_run)

    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/input/",
            "--output-uri",
            "/tmp/local-out",
            "--execute",
            "--condition-on-input",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen[0]["control"] == "seg"
    assert seen[0]["control_prompt"] == "robot arm, bin"
    assert seen[0]["mask_prompt"] == "robot arm"
