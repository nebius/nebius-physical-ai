"""Media staging rejects filesystem/SSRF inputs before any download."""

from pathlib import Path

import pytest

from npa.workbench.cosmos.ray_inputs import stage_sample_inputs
from npa.workbench.storage_scope import StorageAuthorizationError, StorageScope


@pytest.mark.parametrize("value", [
    "/etc/passwd", "file:///etc/passwd", "../secret", "http://localhost/secret",
    "https://example.invalid/redirect", "s3://foreign/media/image.png",
    "s3://test-bucket/media-other/image.png", "s3://test-bucket/media/../secret",
    "s3://test-bucket/media/%2e%2e/secret", "s3://test-bucket/media/",
])
@pytest.mark.parametrize("field", ["vision_path", "prompt_path", "negative_prompt_file"])
def test_inputs_fail_before_download(tmp_path: Path, value: str, field: str) -> None:
    with pytest.raises(StorageAuthorizationError):
        stage_sample_inputs(
            {"name": "sample", field: value}, tmp_path / "input",
            scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
            storage_client=object(),
        )
    assert not (tmp_path / "input").exists()


@pytest.mark.parametrize("raw", [
    {"defaults_file": "s3://test-bucket/media/config.json"},
    {"output_dir": "/tmp/escape"},
    {"future_path": "s3://test-bucket/media/a"},
    {"edge": {"control_path": "http://localhost/internal"}},
])
def test_nested_and_indirect_file_configuration_is_rejected(tmp_path, raw):
    with pytest.raises(StorageAuthorizationError):
        stage_sample_inputs(raw, tmp_path / "input", scope=StorageScope())
    assert not (tmp_path / "input").exists()


def test_authorizes_entire_sample_before_staging_any_media(tmp_path):
    with pytest.raises(StorageAuthorizationError):
        stage_sample_inputs(
            {"vision_path": "s3://test-bucket/media/a.png", "prompt_path": "/etc/passwd"},
            tmp_path / "input", storage_client=object(),
            scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
        )
    assert not (tmp_path / "input").exists()


def test_text_only_needs_no_storage_and_preserves_sampling_types(tmp_path):
    sample = {"name": "sample", "prompt": "robot workcell", "resolution": "720", "aspect_ratio": "1,1"}
    assert stage_sample_inputs(sample, tmp_path / "input", scope=StorageScope()) == sample
    assert not (tmp_path / "input").exists()


def test_valid_media_and_transfer_inputs_are_staged_as_local_files(tmp_path):
    class Storage:
        calls = []

        def download_file(self, uri, destination):
            self.calls.append(uri)
            Path(destination).write_bytes(b"verified object bytes")

    storage = Storage()
    original = {"name": "sample", "vision_path": "s3://test-bucket/media/a.png",
                "edge": {"control_path": "s3://test-bucket/media/b.mp4"}}
    output = stage_sample_inputs(
        original, tmp_path / "input", storage_client=storage,
        scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
    )
    assert storage.calls == ["s3://test-bucket/media/a.png", "s3://test-bucket/media/b.mp4"]
    for value in (output["vision_path"], output["edge"]["control_path"]):
        path = Path(value)
        assert path.parent == tmp_path / "input"
        assert path.read_bytes() == b"verified object bytes"
    assert original["vision_path"].startswith("s3://")


def test_existing_input_directory_cannot_redirect_staging(tmp_path):
    (tmp_path / "input").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(FileExistsError):
        stage_sample_inputs(
            {"vision_path": "s3://test-bucket/media/a.png"}, tmp_path / "input",
            storage_client=object(),
            scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
        )


@pytest.mark.parametrize("encoded", ["%3F", "%23", "%09", "%0D", "%0A"])
def test_reinterpreted_decoded_key_is_rejected_before_any_download(tmp_path, encoded):
    from unittest.mock import Mock

    from npa.clients.storage import StorageClient

    storage = object.__new__(StorageClient)
    storage._s3 = Mock()
    with pytest.raises(StorageAuthorizationError, match="downloaded exactly"):
        stage_sample_inputs(
            {"vision_path": "s3://test-bucket/media/good.png",
             "edge": {"control_path": f"s3://test-bucket/media/input{encoded}variant.png"}},
            tmp_path / "input", storage_client=storage,
            scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
        )
    storage._s3.get_object.assert_not_called()
    assert not (tmp_path / "input").exists()


@pytest.mark.parametrize(("encoded", "expected"), [
    ("input%2Epng", "input.png"),
    ("input.%70ng", "input.png"),
    ("input%20variant.png", "input variant.png"),
])
def test_authorized_key_reaches_real_storage_parser_unchanged(tmp_path, encoded, expected):
    import io
    from unittest.mock import Mock

    from botocore.response import StreamingBody

    from npa.clients.storage import StorageClient

    data = b"exact conditioning object"
    storage = object.__new__(StorageClient)
    storage._s3 = Mock()
    storage._s3.get_object.return_value = {"Body": StreamingBody(io.BytesIO(data), len(data))}
    sample = {"vision_path": f"s3://test-bucket/media/{encoded}"}
    result = stage_sample_inputs(
        sample, tmp_path / "input", storage_client=storage,
        scope=StorageScope.from_config(s3_roots=["s3://test-bucket/media"]),
    )
    storage._s3.get_object.assert_called_once_with(Bucket="test-bucket", Key=f"media/{expected}")
    assert Path(result["vision_path"]).read_bytes() == data
    assert sample["vision_path"].endswith(encoded)
