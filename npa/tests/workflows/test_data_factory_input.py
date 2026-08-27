from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from urllib.error import URLError
from urllib.parse import urlparse

from botocore.exceptions import ClientError
import pytest

from npa.workflows import data_factory_input as dfi


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []
        self.list_requests: list[tuple[str, str]] = []

    @staticmethod
    def _missing(operation: str) -> ClientError:
        return ClientError({"Error": {"Code": "NoSuchKey"}}, operation)

    def get_object(self, *, Bucket: str, Key: str):
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise self._missing("GetObject") from exc
        return {"Body": BytesIO(body)}

    def list_objects_v2(self, *, Bucket: str, Prefix: str):
        self.list_requests.append((Bucket, Prefix))
        return {
            "Contents": [
                {"Key": key, "Size": len(body)}
                for (bucket, key), body in sorted(self.objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }

    def head_object(self, *, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise self._missing("HeadObject")
        return {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "Metadata": self.metadata.get((Bucket, Key), {}),
        }

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs=None) -> None:
        self.objects[(bucket, key)] = Path(path).read_bytes()
        self.metadata[(bucket, key)] = dict((ExtraArgs or {}).get("Metadata") or {})
        self.uploads.append((bucket, key))

    def download_file(self, bucket: str, key: str, path: str) -> None:
        try:
            body = self.objects[(bucket, key)]
        except KeyError as exc:
            raise self._missing("DownloadFile") from exc
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        self.downloads.append((bucket, key))


class FakeStorage:
    def __init__(self) -> None:
        self.s3 = FakeS3()

    def download_path(self, uri: str, path: str) -> str:
        parsed = urlparse(uri)
        target = Path(path)
        self.s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(target))
        return str(target)


@pytest.fixture
def h264_video(tmp_path: Path) -> Path:
    path = tmp_path / "capture.mp4"
    path.write_bytes(b"hermetic-test-h264-mp4")
    return path


@pytest.fixture
def fake_media_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise PAIDF staging without depending on host FFmpeg packages."""

    monkeypatch.setattr(dfi.shutil, "which", lambda name: f"/test-bin/{name}")

    def fake_probe(path: Path) -> dict:
        conditioning = path.name == "conditioning.mp4"
        return {
            "container": "mp4",
            "codec": "h264",
            "codec_profile": "High",
            "width": 1280 if conditioning else 320,
            "height": 720 if conditioning else 240,
            "duration_seconds": 5.8125 if conditioning else 0.6,
            "frame_rate": "16/1" if conditioning else "10/1",
            "frame_count": 93 if conditioning else 6,
            "pixel_format": "yuv420p",
        }

    def fake_derive(source: Path, output: Path) -> None:
        output.write_bytes(b"derived-conditioning:" + source.read_bytes())

    def fake_extract(_conditioning: Path, output_dir: Path) -> list[Path]:
        frames = []
        for index in range(1, dfi.CONDITIONING_FRAMES + 1):
            frame = output_dir / f"conditioning-frame-{index:04d}.png"
            frame.write_bytes(f"derived-frame-{index}".encode())
            frames.append(frame)
        return frames

    monkeypatch.setattr(dfi, "probe_video", fake_probe)
    monkeypatch.setattr(dfi, "_derive_conditioning", fake_derive)
    monkeypatch.setattr(dfi, "_extract_conditioning_frames", fake_extract)


def test_default_selection_is_pinned_real_starter() -> None:
    result = dfi.plan_paidf_input(run_id="paidf-one", bucket="bucket")

    assert result.selection == "starter"
    assert result.provenance["source_kind"] == "upstream_sample"
    assert result.provenance["input_origin"] == "actual_capture"
    assert result.provenance["input_origin_label"] == "Upstream real sample"
    assert result.provenance["sha256"] == (
        "caadec919abfebe7ac7f571f52d0c579dbe86ceacc0d0bdbf9a862ed1a908198"
    )
    assert result.config_overrides()["seed_fixture"] == "false"


@pytest.mark.parametrize(
    ("video", "uri", "fixture"),
    [
        (Path("input.mp4"), "s3://bucket/input.mp4", False),
        (Path("input.mp4"), "", True),
        (None, "s3://bucket/input.mp4", True),
    ],
)
def test_explicit_input_conflicts_fail(
    video: Path | None, uri: str, fixture: bool
) -> None:
    with pytest.raises(dfi.PaidfInputError, match="options conflict"):
        dfi.select_paidf_input(input_video=video, input_uri=uri, seed_fixture=fixture)


def test_explicit_selectors_override_default() -> None:
    assert dfi.select_paidf_input(input_video=Path("capture.mp4")) == "local_video"
    assert dfi.select_paidf_input(input_uri="s3://bucket/capture.mp4") == "object_uri"
    assert dfi.select_paidf_input(seed_fixture=True) == "synthetic_fixture"
    assert (
        dfi.select_paidf_input(lerobot_uri="s3://bucket/dataset/")
        == "lerobot_dataset"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_video": Path("capture.mov")}, "ending in .mp4"),
        ({"input_uri": "s3://bucket/capture.mkv"}, "MP4 object"),
        ({"input_uri": "https://example.test/capture.mp4"}, "one S3 object"),
        ({"lerobot_uri": "https://example.test/dataset/"}, "S3 LeRobotDataset"),
    ],
)
def test_invalid_selector_paths_fail_clearly(kwargs: dict, message: str) -> None:
    with pytest.raises(dfi.PaidfInputError, match=message):
        dfi.select_paidf_input(**kwargs)


def test_fixture_plan_is_explicit_and_synthetic() -> None:
    result = dfi.plan_paidf_input(
        run_id="paidf-fixture", bucket="bucket", seed_fixture=True
    )

    assert result.provenance["source_kind"] == "synthetic_fixture"
    assert result.provenance["input_origin_label"] == "Synthetic seeded fixture"
    assert result.config_overrides()["seed_fixture"] == "true"


def test_lerobot_selector_materializes_only_selected_camera_episode(
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    prefix = "datasets/example/"
    storage.s3.objects[("artifacts", prefix + "meta/info.json")] = json.dumps(
        {
            "codebase_version": "v2.1",
            "total_episodes": 2,
            "features": {
                "observation.images.front": {"dtype": "video"},
                "observation.images.wrist": {"dtype": "video"},
            },
        }
    ).encode()
    storage.s3.objects[
        (
            "artifacts",
            prefix
            + "videos/chunk-000/observation.images.front/episode_000001.mp4",
        )
    ] = b"front-episode-one"
    storage.s3.objects[
        (
            "artifacts",
            prefix
            + "videos/chunk-000/observation.images.wrist/episode_000001.mp4",
        )
    ] = b"wrist-episode-one"
    # An unrelated episode proves that PAIDF does not materialize the full dataset.
    storage.s3.objects[
        (
            "artifacts",
            prefix
            + "videos/chunk-000/observation.images.front/episode_000000.mp4",
        )
    ] = b"front-episode-zero"

    result = dfi.prepare_paidf_input(
        run_id="paidf-lerobot",
        bucket="artifacts",
        lerobot_uri=f"s3://artifacts/{prefix}",
        lerobot_camera="observation.images.wrist",
        lerobot_episode=1,
        storage_client=storage,
    )

    assert result.selection == "lerobot_dataset"
    assert result.provenance["source_format"] == "lerobot"
    assert result.provenance["lerobot_selection"] == {
        "episode_selector": "operator-supplied",
        "camera_selector": "explicit",
        "selection_contract": "compatibility-defaults",
        "media_kind": "video",
        "selected_object": "redacted",
    }
    serialized = json.dumps(result.provenance)
    assert prefix not in serialized
    assert "observation.images.wrist" not in serialized
    assert re.search(r"\b[0-9a-f]{64}\b", serialized) is None
    assert result.config_overrides()["input_sha256"] == ""
    assert storage.s3.downloads == [
        (
            "artifacts",
            prefix
            + "videos/chunk-000/observation.images.wrist/episode_000001.mp4",
        )
    ]
    assert storage.s3.list_requests == []
    assert result.config_overrides()["input_source_format"] == "lerobot"
    staged_prefix = "physical-ai-data-factory/paidf-lerobot/input/"
    assert all(
        "sha256" not in metadata
        for (bucket, key), metadata in storage.s3.metadata.items()
        if bucket == "artifacts" and key.startswith(staged_prefix)
    )


def test_lerobot_camera_and_episode_validation_is_fail_closed() -> None:
    with pytest.raises(dfi.PaidfInputError, match="non-negative"):
        dfi.plan_paidf_input(
            run_id="bad",
            bucket="artifacts",
            lerobot_uri="s3://artifacts/dataset/",
            lerobot_episode=-1,
        )
    with pytest.raises(dfi.PaidfInputError, match="require --lerobot-uri"):
        dfi.plan_paidf_input(
            run_id="bad", bucket="artifacts", lerobot_camera="front"
        )


@pytest.mark.parametrize(
    ("camera", "episode_was_explicit", "missing"),
    [
        ("", True, "--lerobot-camera"),
        ("observation.images.front", False, "--lerobot-episode"),
    ],
)
def test_lerobot_strict_selection_requires_explicit_camera_and_episode(
    camera: str, episode_was_explicit: bool, missing: str
) -> None:
    with pytest.raises(dfi.PaidfInputError, match=missing):
        dfi.validate_lerobot_selector(
            selection="lerobot_dataset",
            camera=camera,
            episode=0,
            require_explicit_selection=True,
            episode_was_explicit=episode_was_explicit,
        )

    dfi.validate_lerobot_selector(
        selection="lerobot_dataset",
        camera="observation.images.front",
        episode=0,
        require_explicit_selection=True,
        episode_was_explicit=True,
    )

    planned = dfi.plan_paidf_input(
        run_id="paidf-explicit-contract",
        bucket="artifacts",
        lerobot_uri="s3://artifacts/dataset/",
        lerobot_camera="observation.images.front",
        lerobot_episode=0,
        require_explicit_lerobot_selection=True,
        lerobot_episode_was_explicit=True,
    )
    assert planned.provenance["lerobot_selection"]["selection_contract"] == (
        "explicit-camera-and-episode"
    )


def test_lerobot_explicit_zero_episode_is_still_a_lerobot_only_selector() -> None:
    with pytest.raises(dfi.PaidfInputError, match="require --lerobot-uri"):
        dfi.validate_lerobot_selector(
            selection="starter",
            camera="",
            episode=0,
            episode_was_explicit=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_episodes", 0, "positive integer"),
        ("total_episodes", True, "positive integer"),
        ("chunks_size", 0, "positive integer"),
        ("chunks_size", True, "must be an integer"),
    ],
)
def test_lerobot_rejects_invalid_dataset_cardinality(
    field: str,
    value: object,
    message: str,
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "chunks_size": 1_000,
        "features": {"observation.images.front": {"dtype": "video"}},
    }
    info[field] = value
    storage.s3.objects[("artifacts", "dataset/meta/info.json")] = json.dumps(
        info
    ).encode()

    with pytest.raises(dfi.PaidfInputError, match=message):
        dfi.prepare_paidf_input(
            run_id="paidf-lerobot-cardinality",
            bucket="artifacts",
            lerobot_uri="s3://artifacts/dataset/",
            storage_client=storage,
        )


def test_lerobot_rejects_oversized_metadata(
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    storage.s3.objects[("artifacts", "dataset/meta/info.json")] = b"{" + (
        b" " * dfi.MAX_LEROBOT_INFO_BYTES
    )

    with pytest.raises(dfi.PaidfInputError, match="1 MB validation limit"):
        dfi.prepare_paidf_input(
            run_id="paidf-lerobot-oversized",
            bucket="artifacts",
            lerobot_uri="s3://artifacts/dataset/",
            storage_client=storage,
        )


def test_lerobot_rejects_unsafe_format_fields_and_nonfinite_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(dfi.PaidfInputError, match="unsupported format field"):
        dfi._format_lerobot_video_path(
            "videos/{video_key.__class__}/episode_{episode_index:06d}.mp4",
            feature="observation.images.front",
            episode=0,
            episode_chunk=0,
            chunk_index=0,
            file_index=0,
        )

    monkeypatch.setattr(
        dfi.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("ffmpeg must not run"),
    )
    with pytest.raises(dfi.PaidfInputError, match="invalid episode time range"):
        dfi._trim_lerobot_episode(
            tmp_path / "source.mp4",
            tmp_path / "episode.mp4",
            start_seconds=0,
            end_seconds=float("inf"),
        )


def test_lerobot_v3_selects_and_trims_one_episode_from_shared_video(
    tmp_path: Path,
    fake_media_pipeline: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    storage = FakeStorage()
    prefix = "datasets/v3/"
    feature = "observation.images.front"
    storage.s3.objects[("artifacts", prefix + "meta/info.json")] = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 8,
            "chunks_size": 1000,
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {feature: {"dtype": "video"}},
        }
    ).encode()
    table = pa.Table.from_pylist(
        [
            {
                "episode_index": 7,
                f"videos/{feature}/chunk_index": 0,
                f"videos/{feature}/file_index": 2,
                f"videos/{feature}/from_timestamp": 1.25,
                f"videos/{feature}/to_timestamp": 2.75,
            }
        ]
    )
    parquet = tmp_path / "episodes.parquet"
    pq.write_table(table, parquet)
    storage.s3.objects[
        ("artifacts", prefix + "meta/episodes/chunk-000/file-000.parquet")
    ] = parquet.read_bytes()
    shared_key = prefix + f"videos/{feature}/chunk-000/file-002.mp4"
    storage.s3.objects[("artifacts", shared_key)] = b"shared-video"
    trim_args: list[tuple[float, float]] = []

    def fake_trim(source: Path, destination: Path, *, start_seconds, end_seconds):
        trim_args.append((float(start_seconds), float(end_seconds)))
        destination.write_bytes(b"episode:" + source.read_bytes())

    monkeypatch.setattr(dfi, "_trim_lerobot_episode", fake_trim)

    result = dfi.prepare_paidf_input(
        run_id="paidf-lerobot-v3",
        bucket="artifacts",
        lerobot_uri=f"s3://artifacts/{prefix}",
        lerobot_episode=7,
        storage_client=storage,
    )

    assert result.selection == "lerobot_dataset"
    assert trim_args == [(1.25, 2.75)]
    assert [key for _bucket, key in storage.s3.downloads] == [
        prefix + "meta/episodes/chunk-000/file-000.parquet",
        shared_key,
    ]
    assert storage.s3.list_requests == [
        ("artifacts", prefix + "meta/episodes/chunk-000/")
    ]


def test_lerobot_rejects_unsupported_version_and_unsafe_video_template(
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    prefix = "datasets/unsupported/"
    storage.s3.objects[("artifacts", prefix + "meta/info.json")] = json.dumps(
        {
            "codebase_version": "v4.0",
            "features": {"observation.images.front": {"dtype": "video"}},
        }
    ).encode()
    with pytest.raises(dfi.PaidfInputError, match="v2.x and v3.x"):
        dfi.prepare_paidf_input(
            run_id="paidf-lerobot-unsupported",
            bucket="artifacts",
            lerobot_uri=f"s3://artifacts/{prefix}",
            storage_client=storage,
        )

    storage.s3.objects[("artifacts", prefix + "meta/info.json")] = json.dumps(
        {
            "codebase_version": "v2.1",
            "total_episodes": 1,
            "video_path": "../private/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.front": {"dtype": "video"}},
        }
    ).encode()
    with pytest.raises(dfi.PaidfInputError, match="relative videos"):
        dfi.prepare_paidf_input(
            run_id="paidf-lerobot-unsafe-template",
            bucket="artifacts",
            lerobot_uri=f"s3://artifacts/{prefix}",
            storage_client=storage,
        )


def test_lerobot_v3_refuses_shared_video_without_episode_timestamps(
    tmp_path: Path,
    fake_media_pipeline: None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    storage = FakeStorage()
    prefix = "datasets/v3-missing-time/"
    feature = "observation.images.front"
    storage.s3.objects[("artifacts", prefix + "meta/info.json")] = json.dumps(
        {
            "codebase_version": "v3.0",
            "total_episodes": 1,
            "chunks_size": 1000,
            "features": {feature: {"dtype": "video"}},
        }
    ).encode()
    table = pa.Table.from_pylist(
        [
            {
                "episode_index": 0,
                f"videos/{feature}/chunk_index": 0,
                f"videos/{feature}/file_index": 0,
            }
        ]
    )
    parquet = tmp_path / "episodes.parquet"
    pq.write_table(table, parquet)
    storage.s3.objects[
        ("artifacts", prefix + "meta/episodes/chunk-000/file-000.parquet")
    ] = parquet.read_bytes()
    storage.s3.objects[
        ("artifacts", prefix + f"videos/{feature}/chunk-000/file-000.mp4")
    ] = b"shared-video"

    with pytest.raises(dfi.PaidfInputError, match="missing.*time range"):
        dfi.prepare_paidf_input(
            run_id="paidf-lerobot-v3-missing-time",
            bucket="artifacts",
            lerobot_uri=f"s3://artifacts/{prefix}",
            storage_client=storage,
        )

    assert all("videos/" not in key for _bucket, key in storage.s3.downloads)


def test_fixture_cannot_replace_committed_user_input(
    h264_video: Path, fake_media_pipeline: None
) -> None:
    storage = FakeStorage()
    dfi.prepare_paidf_input(
        run_id="paidf-fixture-conflict",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    with pytest.raises(dfi.PaidfInputError, match="--seed-fixture cannot replace"):
        dfi.prepare_paidf_input(
            run_id="paidf-fixture-conflict",
            bucket="artifacts",
            seed_fixture=True,
            storage_client=storage,
        )


def test_implicit_retry_reuses_committed_fixture_without_starter_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    prefix = "physical-ai-data-factory/paidf-fixture-retry/input/"
    provenance = dfi._fixture_provenance(
        "paidf-fixture-retry", f"s3://artifacts/{prefix}"
    )
    storage.s3.objects[("artifacts", prefix + "provenance.json")] = json.dumps(
        provenance
    ).encode()
    monkeypatch.setattr(
        dfi,
        "_fetch_starter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("starter fetched")
        ),
    )

    result = dfi.prepare_paidf_input(
        run_id="paidf-fixture-retry", bucket="artifacts", storage_client=storage
    )

    assert result.selection == "synthetic_fixture"
    assert result.reused is True


def test_local_video_staging_records_lineage_and_is_idempotent(
    h264_video: Path, fake_media_pipeline: None
) -> None:
    storage = FakeStorage()

    first = dfi.prepare_paidf_input(
        run_id="paidf-local",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    upload_count = len(storage.s3.uploads)
    second = dfi.prepare_paidf_input(
        run_id="paidf-local",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    prefix = "physical-ai-data-factory/paidf-local/input/"
    provenance = json.loads(
        storage.s3.objects[("artifacts", prefix + "provenance.json")]
    )
    assert first.provenance["source_kind"] == "user_supplied"
    assert first.provenance["input_origin_label"] == "User-supplied input"
    assert first.provenance["staged_canonical_s3_uri"] == f"s3://artifacts/{prefix}"
    assert first.provenance["cosmos_conditioning"]["enabled"] is True
    assert first.provenance["cosmos_conditioning"]["cli_equivalent"] == (
        "--condition-on-input"
    )
    assert (
        first.provenance["derivation"]["derived_from_sha256"]
        == first.provenance["sha256"]
    )
    assert first.provenance["derivation"]["media"]["frame_count"] == 93
    assert len(first.provenance["derivation"]["frame_derivation"]["items"]) == 8
    assert provenance == first.provenance
    assert second.reused is True
    assert len(storage.s3.uploads) == upload_count


def test_implicit_retry_reuses_user_source_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
    h264_video: Path,
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    dfi.prepare_paidf_input(
        run_id="paidf-reuse",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    monkeypatch.setattr(
        dfi,
        "_fetch_starter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("starter fetched")
        ),
    )

    result = dfi.prepare_paidf_input(
        run_id="paidf-reuse", bucket="artifacts", storage_client=storage
    )

    assert result.reused is True
    assert result.provenance["source_kind"] == "user_supplied"


def test_existing_run_rejects_different_explicit_source(
    h264_video: Path, tmp_path: Path, fake_media_pipeline: None
) -> None:
    storage = FakeStorage()
    dfi.prepare_paidf_input(
        run_id="paidf-immutable",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    other = tmp_path / "other.mp4"
    other.write_bytes(h264_video.read_bytes() + b"different")

    with pytest.raises(dfi.PaidfInputError, match="run input is immutable"):
        dfi.prepare_paidf_input(
            run_id="paidf-immutable",
            bucket="artifacts",
            input_video=other,
            storage_client=storage,
        )


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _contract_for_bytes(body: bytes) -> dict:
    import hashlib

    return {
        "asset_id": "test-asset",
        "integrity": {
            "sha256": hashlib.sha256(body).hexdigest(),
            "byte_size": len(body),
        },
        "source": {"asset_url": "https://official.example/immutable.mp4"},
    }


def test_fetch_verifies_caches_and_supports_offline_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"verified physical video bytes"
    contract = _contract_for_bytes(body)
    monkeypatch.setattr(dfi, "urlopen", lambda *_args, **_kwargs: _Response(body))

    path, status = dfi._fetch_starter(
        contract, cache_dir=tmp_path, offline=False, reporter=lambda _message: None
    )
    assert status == "verified_fetch"
    assert path.read_bytes() == body

    monkeypatch.setattr(
        dfi,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    same, status = dfi._fetch_starter(
        contract, cache_dir=tmp_path, offline=True, reporter=lambda _message: None
    )
    assert same == path
    assert status == "verified_hit"


def test_offline_cache_miss_and_checksum_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b"expected"
    contract = _contract_for_bytes(body)
    with pytest.raises(dfi.PaidfInputError, match="offline PAIDF cache miss"):
        dfi._fetch_starter(
            contract, cache_dir=tmp_path, offline=True, reporter=lambda _message: None
        )

    monkeypatch.setattr(
        dfi, "urlopen", lambda *_args, **_kwargs: _Response(b"tampered")
    )
    with pytest.raises(dfi.PaidfInputError, match="SHA-256 mismatch"):
        dfi._fetch_starter(
            contract, cache_dir=tmp_path, offline=False, reporter=lambda _message: None
        )


def test_fetch_fails_clearly_when_contract_requires_acceptance_or_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract_for_bytes(b"asset")
    contract["license"] = {
        "url": "https://official.example/license",
        "acceptance_required": True,
        "authentication_required": False,
    }
    contract["delivery"] = {
        "acceptance_environment_variable": "NPA_PAIDF_TEST_ACCEPT",
        "authentication_environment_variable": "NPA_PAIDF_TEST_TOKEN",
    }
    monkeypatch.delenv("NPA_PAIDF_TEST_ACCEPT", raising=False)
    monkeypatch.delenv("NPA_PAIDF_TEST_TOKEN", raising=False)

    with pytest.raises(
        dfi.PaidfInputError, match="explicit upstream license acceptance"
    ):
        dfi._fetch_starter(
            contract, cache_dir=tmp_path, offline=False, reporter=lambda _message: None
        )

    monkeypatch.setenv("NPA_PAIDF_TEST_ACCEPT", "1")
    contract["license"]["authentication_required"] = True
    with pytest.raises(dfi.PaidfInputError, match="requires upstream authentication"):
        dfi._fetch_starter(
            contract, cache_dir=tmp_path, offline=False, reporter=lambda _message: None
        )


def test_invalid_media_fails_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    invalid = tmp_path / "invalid.mp4"
    invalid.write_text("not video", encoding="utf-8")
    storage = FakeStorage()
    monkeypatch.setattr(dfi.shutil, "which", lambda name: f"/test-bin/{name}")

    def reject_invalid(_path: Path) -> dict:
        raise dfi.PaidfInputError("video validation failed: invalid test media")

    monkeypatch.setattr(dfi, "probe_video", reject_invalid)

    with pytest.raises(dfi.PaidfInputError, match="video validation failed"):
        dfi.prepare_paidf_input(
            run_id="paidf-invalid",
            bucket="artifacts",
            input_video=invalid,
            storage_client=storage,
        )
    assert storage.s3.uploads == []


def test_missing_media_tools_fails_before_staging(
    monkeypatch: pytest.MonkeyPatch, h264_video: Path
) -> None:
    storage = FakeStorage()
    monkeypatch.setattr(dfi.shutil, "which", lambda _name: None)

    with pytest.raises(dfi.PaidfInputError, match="ffprobe and ffmpeg are required"):
        dfi.prepare_paidf_input(
            run_id="paidf-missing-ffmpeg",
            bucket="artifacts",
            input_video=h264_video,
            storage_client=storage,
        )
    assert storage.s3.uploads == []


def _ffprobe_payload(**stream_overrides) -> str:
    stream = {
        "codec_name": "h264",
        "profile": "High",
        "width": 1280,
        "height": 720,
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "16/1",
        "nb_frames": "93",
    }
    stream.update(stream_overrides)
    return json.dumps(
        {
            "streams": [stream],
            "format": {"format_name": "mov,mp4", "duration": "5.8125"},
        }
    )


def test_real_probe_parser_derives_missing_frame_count_from_valid_bounded_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=_ffprobe_payload(nb_frames="N/A"), stderr=""
        ),
    )

    media = dfi.probe_video(tmp_path / "capture.mp4")

    assert media["frame_count"] == 93
    assert media["frame_rate"] == "16/1"


@pytest.mark.parametrize(
    ("overrides", "format_patch", "field"),
    [
        ({"width": None}, {}, "width"),
        ({"avg_frame_rate": "0/0"}, {}, "avg_frame_rate"),
        ({"avg_frame_rate": "not-a-rate"}, {}, "avg_frame_rate"),
        ({"nb_frames": "N/A", "avg_frame_rate": "1000000000/1"}, {}, "nb_frames"),
        ({}, {"duration": "not-a-duration"}, "duration"),
    ],
)
def test_real_probe_parser_wraps_every_numeric_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict,
    format_patch: dict,
    field: str,
) -> None:
    payload = json.loads(_ffprobe_payload(**overrides))
    payload["format"].update(format_patch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(dfi.PaidfInputError, match=field):
        dfi.probe_video(tmp_path / "capture.mp4")


def test_retry_refuses_cross_version_conditioning_byte_drift(
    monkeypatch: pytest.MonkeyPatch,
    h264_video: Path,
    fake_media_pipeline: None,
) -> None:
    storage = FakeStorage()
    build = {"version": "ffmpeg-a"}

    def derive(source: Path, output: Path) -> dict:
        output.write_bytes(build["version"].encode() + b":" + source.read_bytes())
        return {
            "name": "ffmpeg",
            "version": build["version"],
            "arguments": ["<source-by-sha256>"],
        }

    monkeypatch.setattr(dfi, "_derive_conditioning", derive)
    dfi.prepare_paidf_input(
        run_id="paidf-version-drift",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    uploads = list(storage.s3.uploads)
    build["version"] = "ffmpeg-b"

    with pytest.raises(dfi.PaidfInputError, match="ffmpeg/libx264 changed"):
        dfi.prepare_paidf_input(
            run_id="paidf-version-drift",
            bucket="artifacts",
            input_video=h264_video,
            storage_client=storage,
        )

    assert storage.s3.uploads == uploads


def test_local_input_provenance_never_persists_operator_directory(
    h264_video: Path, fake_media_pipeline: None
) -> None:
    storage = FakeStorage()

    result = dfi.prepare_paidf_input(
        run_id="paidf-safe-local-ref",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    assert result.provenance["source_ref"] == h264_video.name
    assert str(h264_video.parent) not in json.dumps(result.provenance)


def test_interrupted_commit_reuses_exact_artifacts_and_writes_marker_last(
    h264_video: Path, fake_media_pipeline: None
) -> None:
    class InterruptingS3(FakeS3):
        fail_once = True

        def upload_file(self, path: str, bucket: str, key: str, ExtraArgs=None) -> None:
            if key.endswith("provenance.json") and self.fail_once:
                self.fail_once = False
                raise RuntimeError("interrupted marker upload")
            super().upload_file(path, bucket, key, ExtraArgs=ExtraArgs)

    storage = FakeStorage()
    storage.s3 = InterruptingS3()
    with pytest.raises(dfi.PaidfInputError, match="interrupted marker upload"):
        dfi.prepare_paidf_input(
            run_id="paidf-interrupted",
            bucket="artifacts",
            input_video=h264_video,
            storage_client=storage,
        )
    completed_artifact_uploads = len(storage.s3.uploads)

    result = dfi.prepare_paidf_input(
        run_id="paidf-interrupted",
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    assert result.reused is False
    assert len(storage.s3.uploads) == completed_artifact_uploads + 1
    assert storage.s3.uploads[-1][1].endswith("provenance.json")


@pytest.mark.parametrize(
    "missing_leaf", ["conditioning.mp4", "conditioning-frame-0004.png"]
)
def test_committed_retry_repairs_only_a_missing_verified_artifact(
    h264_video: Path, fake_media_pipeline: None, missing_leaf: str
) -> None:
    storage = FakeStorage()
    run_id = "paidf-repair-" + missing_leaf.replace(".", "-")
    dfi.prepare_paidf_input(
        run_id=run_id,
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    prefix = f"physical-ai-data-factory/{run_id}/input/"
    storage.s3.objects.pop(("artifacts", prefix + missing_leaf))
    storage.s3.metadata.pop(("artifacts", prefix + missing_leaf), None)
    before = len(storage.s3.uploads)

    result = dfi.prepare_paidf_input(
        run_id=run_id,
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    assert result.reused is True
    assert storage.s3.uploads[before:] == [("artifacts", prefix + missing_leaf)]


def test_artifacts_without_commit_marker_gain_only_the_marker_on_same_byte_retry(
    h264_video: Path, fake_media_pipeline: None
) -> None:
    storage = FakeStorage()
    run_id = "paidf-marker-repair"
    dfi.prepare_paidf_input(
        run_id=run_id,
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )
    prefix = f"physical-ai-data-factory/{run_id}/input/"
    storage.s3.objects.pop(("artifacts", prefix + "provenance.json"))
    storage.s3.metadata.pop(("artifacts", prefix + "provenance.json"), None)
    before = len(storage.s3.uploads)

    dfi.prepare_paidf_input(
        run_id=run_id,
        bucket="artifacts",
        input_video=h264_video,
        storage_client=storage,
    )

    assert storage.s3.uploads[before:] == [("artifacts", prefix + "provenance.json")]
