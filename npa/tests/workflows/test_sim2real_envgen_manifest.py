"""Manifest-backed frame references for the two-step Sim2Real workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.cosmos.transfer import (
    REFERENCE_AUGMENT_MODE,
    TRANSFER_MANIFEST_MODE,
    TRANSFER_MANIFEST_SCHEMA,
    TRANSFER_MANIFEST_STATUS,
)
from npa.workflows import sim2real_envgen as envgen


def _write_manifest(path: Path, frame_uris: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": TRANSFER_MANIFEST_SCHEMA,
                "mode": TRANSFER_MANIFEST_MODE,
                "status": TRANSFER_MANIFEST_STATUS,
                "frames": [
                    {"frame_id": f"frame-{index:05d}", "uri": uri}
                    for index, uri in enumerate(frame_uris)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(scene: envgen.SceneSpec, *, env_count: int = 1_000) -> envgen.EnvGenConfig:
    return envgen.EnvGenConfig(
        run_id="manifest-test",
        output_uri="s3://bucket/run/",
        env_count=env_count,
        scene_spec=scene,
    )


def test_one_thousand_envs_cycle_only_over_published_manifest_frames(
    tmp_path: Path,
) -> None:
    published = [f"s3://bucket/run/augment/frame-{index:05d}.png" for index in range(8)]
    manifest = _write_manifest(tmp_path / "manifest.json", published)

    scene = envgen.resolve_augmented_frames(
        envgen.build_scene_spec(), str(manifest)
    )
    records = envgen.generate_raw_envs(_config(scene))
    referenced = [record["scene"]["augmented_frame_uri"] for record in records]

    assert len(records) == 1_000
    assert scene.augmented_frames_manifest_uri == str(manifest)
    assert scene.augmented_frame_uris == tuple(published)
    assert referenced == [published[index % len(published)] for index in range(1_000)]
    assert set(referenced) == set(published)


def test_legacy_prefix_callers_keep_synthesized_frame_references() -> None:
    scene = envgen.resolve_augmented_frames(
        envgen.build_scene_spec(), "s3://bucket/legacy/frames/"
    )
    records = envgen.generate_raw_envs(_config(scene, env_count=3))

    assert [record["scene"]["augmented_frame_uri"] for record in records] == [
        "s3://bucket/legacy/frames/frame-00000.png",
        "s3://bucket/legacy/frames/frame-00001.png",
        "s3://bucket/legacy/frames/frame-00002.png",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema": TRANSFER_MANIFEST_SCHEMA,
            "mode": TRANSFER_MANIFEST_MODE,
            "status": TRANSFER_MANIFEST_STATUS,
            "frames": [],
        },
        {
            "schema": TRANSFER_MANIFEST_SCHEMA,
            "mode": TRANSFER_MANIFEST_MODE,
            "status": TRANSFER_MANIFEST_STATUS,
            "frames": [{}],
        },
        {
            "schema": "wrong.schema",
            "mode": TRANSFER_MANIFEST_MODE,
            "status": TRANSFER_MANIFEST_STATUS,
            "frames": [{"uri": "s3://bucket/frame.png"}],
        },
        ["not", "an", "object"],
    ],
)
def test_transfer_manifest_frame_list_fails_closed(
    tmp_path: Path, payload: object
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(envgen.Sim2RealEnvGenError):
        envgen.frame_uris_from_transfer_manifest(str(manifest))


def test_reference_output_cannot_masquerade_as_real_transfer(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": TRANSFER_MANIFEST_SCHEMA,
                "mode": REFERENCE_AUGMENT_MODE,
                "status": "executed_reference",
                "frames": [{"uri": "local://frames/frame-00000.png"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(envgen.Sim2RealEnvGenError, match="real-transfer mode"):
        envgen.frame_uris_from_transfer_manifest(str(manifest))


@pytest.mark.parametrize("scheme", ["file://", "local://"])
def test_local_transfer_manifest_schemes_are_supported(
    tmp_path: Path, scheme: str
) -> None:
    published = ["local://frames/frame-00000.png"]
    manifest = _write_manifest(tmp_path / "manifest.json", published)

    assert envgen.frame_uris_from_transfer_manifest(
        f"{scheme}{manifest}"
    ) == tuple(published)


def test_unreadable_local_transfer_manifest_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(envgen.Sim2RealEnvGenError, match="could not read transfer manifest"):
        envgen.frame_uris_from_transfer_manifest(f"local://{missing}")


def test_malformed_local_transfer_manifest_fails_clearly(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")

    with pytest.raises(envgen.Sim2RealEnvGenError, match="could not read transfer manifest"):
        envgen.frame_uris_from_transfer_manifest(f"local://{malformed}")


def test_s3_transfer_manifest_is_resolved_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    published = ["s3://bucket/run/augment/frame-00000.png"]

    class FakeStorage:
        def download_path(self, uri: str, local: str) -> str:
            assert uri == "s3://bucket/run/augment/manifest.json"
            _write_manifest(Path(local), published)
            return local

    monkeypatch.setattr(
        envgen.StorageClient,
        "from_environment",
        staticmethod(lambda: FakeStorage()),
    )

    assert envgen.frame_uris_from_transfer_manifest(
        "s3://bucket/run/augment/manifest.json"
    ) == tuple(published)


def test_raw_shard_cli_persists_resolved_frame_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    published = [
        "s3://bucket/run/augment/frame-00000.png",
        "s3://bucket/run/augment/frame-00001.png",
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", published)

    class FakeStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            return uri

    monkeypatch.setattr(
        envgen.StorageClient,
        "from_environment",
        staticmethod(lambda: FakeStorage()),
    )
    output_dir = tmp_path / "out"

    result = envgen.main(
        [
            "raw-shard",
            "--run-id",
            "cli-test",
            "--output-uri",
            "s3://bucket/run/",
            "--env-count",
            "10",
            "--augmented-frames-uri",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["raw_count"] == 10
    scene = json.loads((output_dir / "scene-spec.json").read_text(encoding="utf-8"))
    assert scene["augmented_frames_manifest_uri"] == str(manifest)
    assert scene["augmented_frame_uris"] == published
    records = envgen._read_jsonl(output_dir / "raw-shard-00-of-01.jsonl")
    assert [record["scene"]["augmented_frame_uri"] for record in records] == [
        published[index % 2] for index in range(10)
    ]


def test_public_workbench_raw_shard_persists_resolved_frame_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    published = [
        "s3://bucket/run/augment/exact-a.png",
        "s3://bucket/run/augment/exact-b.png",
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", published)

    class FakeStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            return uri

    monkeypatch.setattr(
        envgen.StorageClient,
        "from_environment",
        staticmethod(lambda: FakeStorage()),
    )
    output_dir = tmp_path / "public-out"

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "sim2real-envgen",
            "raw-shard",
            "--run-id",
            "public-cli-test",
            "--output-uri",
            "s3://bucket/run/",
            "--env-count",
            "4",
            "--augmented-frames-uri",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    scene = json.loads((output_dir / "scene-spec.json").read_text(encoding="utf-8"))
    assert scene["augmented_frames_manifest_uri"] == str(manifest)
    assert scene["augmented_frame_uris"] == published
    records = envgen._read_jsonl(output_dir / "raw-shard-00-of-01.jsonl")
    assert [record["scene"]["augmented_frame_uri"] for record in records] == [
        published[index % 2] for index in range(4)
    ]


def test_public_workbench_raw_shard_retains_legacy_prefix_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    class FakeStorage:
        def upload_file(self, _local: str, uri: str) -> str:
            return uri

    monkeypatch.setattr(
        envgen.StorageClient,
        "from_environment",
        staticmethod(lambda: FakeStorage()),
    )
    output_dir = tmp_path / "legacy-public-out"
    prefix = "s3://bucket/run/legacy-frames/"

    result = CliRunner().invoke(
        app,
        [
            "workbench",
            "sim2real-envgen",
            "raw-shard",
            "--run-id",
            "legacy-public-cli-test",
            "--output-uri",
            "s3://bucket/run/",
            "--env-count",
            "3",
            "--augmented-frames-uri",
            prefix,
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    records = envgen._read_jsonl(output_dir / "raw-shard-00-of-01.jsonl")
    assert [record["scene"]["augmented_frame_uri"] for record in records] == [
        f"{prefix}frame-{index:05d}.png" for index in range(3)
    ]
