"""Unit tests for the non-stub Cosmos2 transfer reference augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from npa.workbench.cosmos.transfer import (
    AUGMENTED_FRAMES_SCHEMA,
    TRANSFER_MANIFEST_FILENAME,
    TRANSFER_MANIFEST_SCHEMA,
    augmented_frames_index_uri_for,
    reference_augment_frames,
    transfer_manifest_uri_for,
)


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path)


def test_reference_augment_produces_real_image_frames(tmp_path: Path) -> None:
    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (200, 40, 40))
    _write_png(src / "frame_001.png", (40, 200, 40))

    result = reference_augment_frames(
        str(src), str(out), run_id="unit", variants_per_frame=2
    )

    # 2 sources x 2 variants = 4 real augmented frames, plus an index manifest.
    assert result["frame_count"] == 4
    assert result["source_frame_count"] == 2
    frames = sorted(out.glob("frame-*.png"))
    assert len(frames) == 4
    for frame in frames:
        # Each output is a real, openable image (not a JSON descriptor).
        with Image.open(frame) as img:
            assert img.size == (64, 48)
    index = json.loads((out / "index.json").read_text())
    assert index["schema"] == AUGMENTED_FRAMES_SCHEMA
    assert index["frame_count"] == 4
    assert {f["perturbation"] for f in index["frames"]} <= {
        "lighting",
        "contrast",
        "color",
        "blur",
    }


def test_augmented_frames_index_uri_is_derived_from_the_output_prefix() -> None:
    assert (
        augmented_frames_index_uri_for("s3://bucket/run/augmented/")
        == "s3://bucket/run/augmented/index.json"
    )
    assert augmented_frames_index_uri_for("local://out") == "local://out/index.json"


def test_reference_fallback_reports_its_canonical_index_uri(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(source / "frame.png")

    result = reference_augment_frames(
        str(source), str(output), run_id="fallback", variants_per_frame=1
    )

    assert result["index_uri"] == augmented_frames_index_uri_for(str(output))
    assert Path(result["index_uri"]).is_file()


def test_reference_fallback_preserves_local_scheme_for_index_and_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(source / "frame.png")
    output_uri = f"local://{output}"

    result = reference_augment_frames(
        str(source), output_uri, run_id="fallback", variants_per_frame=1
    )

    assert result["augmented_frames_uri"] == output_uri
    assert result["index_uri"] == f"{output_uri}/index.json"
    assert [frame["uri"] for frame in result["frames"]] == [
        f"{output_uri}/frame-00000.png"
    ]
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert [frame["uri"] for frame in index["frames"]] == [
        f"{output_uri}/frame-00000.png"
    ]


def test_real_transfer_manifest_uri_is_canonical() -> None:
    assert TRANSFER_MANIFEST_FILENAME == "manifest.json"
    assert TRANSFER_MANIFEST_SCHEMA == "npa.cosmos2.transfer.v1"
    assert (
        transfer_manifest_uri_for("s3://bucket/run/augmented/")
        == "s3://bucket/run/augmented/manifest.json"
    )
    assert (
        transfer_manifest_uri_for("s3://bucket/run/augmented")
        == "s3://bucket/run/augmented/manifest.json"
    )


def test_reference_augment_actually_transforms_pixels(tmp_path: Path) -> None:
    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (128, 128, 128))

    reference_augment_frames(str(src), str(out), run_id="unit", variants_per_frame=1)

    original = Image.open(src / "frame_000.png").convert("RGB").tobytes()
    augmented = Image.open(next(out.glob("frame-*.png"))).convert("RGB").tobytes()
    assert original != augmented, "reference augmentation must change pixels, not copy"


def test_reference_augment_accepts_single_local_file(tmp_path: Path) -> None:
    src_file = tmp_path / "solo.png"
    out = tmp_path / "augment"
    _write_png(src_file, (77, 88, 99))

    result = reference_augment_frames(
        str(src_file), str(out), run_id="unit", variants_per_frame=2
    )

    # A single local image file is a valid source, not a "no source images" error.
    assert result["source_frame_count"] == 1
    assert result["frame_count"] == 2
    assert len(list(out.glob("frame-*.png"))) == 2


def test_reference_augment_without_sources_raises(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(RuntimeError, match="no source images"):
        reference_augment_frames(str(src), str(tmp_path / "out"), run_id="unit")


def test_cosmos2_transfer_cli_default_emits_real_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.workbench.cosmos import transfer as tx

    src = tmp_path / "scene"
    out = tmp_path / "augment"
    _write_png(src / "frame_000.png", (10, 20, 30))
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            str(src),
            "--output-uri",
            str(out),
            "--run-id",
            "cli-unit",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Not a descriptor stub: it ran a real reference augmentation with frames.
    assert payload["status"] == "executed_reference"
    assert payload["mode"] == "reference_augment"
    assert payload["output_kind"] == "frames"
    assert payload["frame_count"] >= 1
    assert list(out.glob("frame-*.png"))
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert index["schema"] == AUGMENTED_FRAMES_SCHEMA
    assert manifest["schema"] == TRANSFER_MANIFEST_SCHEMA
    assert manifest["status"] == "executed_reference"
    assert manifest["mode"] == "reference_augment"
    assert manifest["index_uri"] == str(out / "index.json")
    assert manifest["manifest_uri"] == str(out / "manifest.json")
    assert [frame["uri"] for frame in manifest["frames"]] == [
        str(frame) for frame in sorted(out.glob("frame-*.png"))
    ]


def test_real_transfer_cli_and_durable_manifest_use_the_same_gpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.workbench.cosmos import transfer as tx

    video = tmp_path / "output.mp4"
    video.write_bytes(b"video")
    durable_manifest = {
        "schema": TRANSFER_MANIFEST_SCHEMA,
        "mode": "cosmos_transfer2.5_gpu",
        "status": "executed",
        "augmented_video_uri": "s3://bucket/run/augment/augmented_video.mp4",
        "augmented_frames_uri": "s3://bucket/run/augment/",
        "frame_count": 1,
    }
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(
        tx,
        "run_cosmos_transfer",
        lambda **_: {
            "video_path": str(video),
            "video_bytes": video.stat().st_size,
            "spec": "spec.json",
            "input_conditioned": False,
        },
    )
    monkeypatch.setattr(
        tx,
        "publish_transfer_to_s3",
        lambda *_args, **_kwargs: durable_manifest,
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            str(tmp_path),
            "--output-uri",
            "s3://bucket/run/augment/",
            "--run-id",
            "real-mode",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == durable_manifest["mode"] == "cosmos_transfer2.5_gpu"


def test_reference_fallback_s3_publishes_index_and_canonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app
    from npa.workbench.cosmos import transfer as tx

    src = tmp_path / "scene"
    _write_png(src / "frame_000.png", (30, 40, 50))
    uploads: dict[str, bytes] = {}

    class FakeStorage:
        def upload_directory(self, local_dir: str, uri: str) -> str:
            base = uri.rstrip("/") + "/"
            root = Path(local_dir)
            for path in root.rglob("*"):
                if path.is_file():
                    uploads[base + path.relative_to(root).as_posix()] = path.read_bytes()
            return uri

        def upload_file(self, local: str, uri: str) -> str:
            uploads[uri] = Path(local).read_bytes()
            return uri

    storage = FakeStorage()
    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: False)
    monkeypatch.setattr(
        "npa.clients.storage.StorageClient.from_environment",
        staticmethod(lambda: storage),
    )

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            str(src),
            "--output-uri",
            "s3://bucket/run/augment/",
            "--run-id",
            "s3-fallback",
        ],
    )

    assert result.exit_code == 0, result.output
    index = json.loads(uploads["s3://bucket/run/augment/index.json"])
    manifest = json.loads(uploads["s3://bucket/run/augment/manifest.json"])
    assert index["schema"] == AUGMENTED_FRAMES_SCHEMA
    assert manifest["schema"] == TRANSFER_MANIFEST_SCHEMA
    assert manifest["index_uri"] == "s3://bucket/run/augment/index.json"
    assert [frame["uri"] for frame in manifest["frames"]] == [
        "s3://bucket/run/augment/frame-00000.png",
        "s3://bucket/run/augment/frame-00001.png",
    ]
