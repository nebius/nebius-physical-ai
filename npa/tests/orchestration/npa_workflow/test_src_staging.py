"""Unit tests for staging the npa package source to S3."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.src_staging import (
    DEFAULT_SRC_PREFIX,
    SrcStagingError,
    find_npa_package_root,
    iter_source_files,
    resolve_src_uri_from_env,
    stage_npa_source,
)

runner = CliRunner()


class FakeStorageClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploads.append((local_file, bucket_uri))
        return bucket_uri


def _fake_package(root: Path) -> Path:
    (root / "src" / "npa").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='npa'\n", encoding="utf-8")
    (root / "src" / "npa" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "npa" / "cli.py").write_text("x = 1\n", encoding="utf-8")
    # Noise that must never be uploaded.
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "big.py").write_text("", encoding="utf-8")
    (root / "src" / "npa" / "__pycache__").mkdir()
    (root / "src" / "npa" / "__pycache__" / "cli.pyc").write_text("", encoding="utf-8")
    (root / "npa.egg-info").mkdir()
    (root / "npa.egg-info" / "PKG-INFO").write_text("", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "junk.py").write_text("", encoding="utf-8")
    return root


def test_iter_source_files_excludes_build_artifacts(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "npa")

    files = {path.as_posix() for path in iter_source_files(root)}

    assert files == {"pyproject.toml", "src/npa/__init__.py", "src/npa/cli.py"}


def test_stage_npa_source_uploads_to_expected_uri(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "npa")
    client = FakeStorageClient()
    status: list[str] = []

    uri = stage_npa_source(
        bucket="my-bucket",
        source_root=root,
        client=client,
        on_status=status.append,
    )

    assert uri == f"s3://my-bucket/{DEFAULT_SRC_PREFIX}/"
    destinations = {dest for _local, dest in client.uploads}
    assert destinations == {
        "s3://my-bucket/npa-src/npa/pyproject.toml",
        "s3://my-bucket/npa-src/npa/src/npa/__init__.py",
        "s3://my-bucket/npa-src/npa/src/npa/cli.py",
    }
    assert any("staged 3 files" in line for line in status)


def test_stage_npa_source_normalizes_bucket_and_prefix(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "npa")
    client = FakeStorageClient()

    uri = stage_npa_source(
        bucket="s3://my-bucket/ignored-path",
        prefix="/custom/src/",
        source_root=root,
        client=client,
    )

    assert uri == "s3://my-bucket/custom/src/"
    assert all(dest.startswith("s3://my-bucket/custom/src/") for _l, dest in client.uploads)


def test_stage_npa_source_requires_a_bucket(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "npa")

    with pytest.raises(SrcStagingError, match="bucket is required"):
        stage_npa_source(bucket="  ", source_root=root, client=FakeStorageClient())


def test_stage_npa_source_rejects_a_non_package_root(tmp_path: Path) -> None:
    with pytest.raises(SrcStagingError, match="does not look like the npa package"):
        stage_npa_source(
            bucket="my-bucket", source_root=tmp_path, client=FakeStorageClient()
        )


def test_find_npa_package_root_locates_this_checkout() -> None:
    root = find_npa_package_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "npa" / "__init__.py").is_file()


def test_find_npa_package_root_raises_without_a_source_tree(tmp_path: Path) -> None:
    orphan = tmp_path / "site-packages" / "npa" / "mod.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("", encoding="utf-8")

    with pytest.raises(SrcStagingError, match="Could not locate the npa package source"):
        find_npa_package_root(orphan)


def test_resolve_src_uri_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    assert resolve_src_uri_from_env() == ""

    monkeypatch.setenv("NPA_E2E_NPA_SRC_S3_URI", "s3://b/e2e/npa")
    assert resolve_src_uri_from_env() == "s3://b/e2e/npa"

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://b/npa-src/npa")
    assert resolve_src_uri_from_env() == "s3://b/npa-src/npa"


def test_stage_src_command_prints_the_export_line(mocker) -> None:
    staged = mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        return_value="s3://my-bucket/npa-src/npa/",
    )

    result = runner.invoke(
        app, ["workbench", "workflow", "stage-src", "--bucket", "my-bucket"]
    )

    assert result.exit_code == 0, result.output
    assert "npa_src_s3_uri: s3://my-bucket/npa-src/npa/" in result.output
    assert "export NPA_SRC_S3_URI=s3://my-bucket/npa-src/npa/" in result.output
    assert staged.call_args.kwargs["bucket"] == "my-bucket"


def test_stage_src_command_requires_a_bucket() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "stage-src"])

    assert result.exit_code == 1
    assert "No bucket to stage into" in result.output


def test_submit_stage_src_sets_the_source_uri(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`submit --stage-src` uploads and satisfies the npa-source prerequisite."""
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    monkeypatch.delenv("NPA_E2E_NPA_SRC_S3_URI", raising=False)
    staged = mocker.patch(
        "npa.orchestration.npa_workflow.src_staging.stage_npa_source",
        return_value="s3://real-bucket/npa-src/npa/",
    )
    spec = (
        Path(__file__).resolve().parents[3]
        / "workflows"
        / "physical-ai-data-factory.yaml"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(spec),
            "--run-id",
            "stage-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--var",
            "bucket=real-bucket",
            "--stage-src",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert staged.call_args.kwargs["bucket"] == "real-bucket"
    assert "NPA_SRC_S3_URI is unset" not in result.output
    assert "s3://real-bucket/npa-src/npa/" in result.output


def test_submit_stage_src_refuses_a_placeholder_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_SRC_S3_URI", raising=False)
    spec = (
        Path(__file__).resolve().parents[3]
        / "workflows"
        / "physical-ai-data-factory.yaml"
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "submit",
            str(spec),
            "--run-id",
            "stage-demo",
            "--assume-decision",
            "promote_checkpoint",
            "--stage-src",
            "--plan-only",
        ],
    )

    assert result.exit_code == 1
    assert "--stage-src needs a real bucket" in result.output


def test_stage_npa_source_uploads_in_parallel_without_losing_files(
    tmp_path: Path,
) -> None:
    """Concurrency must not drop or duplicate objects."""
    import threading

    root = tmp_path / "npa"
    (root / "src" / "npa").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='npa'\n", encoding="utf-8")
    for index in range(50):
        (root / "src" / "npa" / f"mod_{index}.py").write_text("", encoding="utf-8")

    lock = threading.Lock()

    class ThreadSafeClient:
        def __init__(self) -> None:
            self.uploads: list[str] = []

        def upload_file(self, local_file: str, bucket_uri: str) -> str:
            with lock:
                self.uploads.append(bucket_uri)
            return bucket_uri

    client = ThreadSafeClient()
    stage_npa_source(bucket="b", source_root=root, client=client, max_workers=8)

    assert len(client.uploads) == 51
    assert len(set(client.uploads)) == 51


def test_stage_npa_source_serial_mode(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "npa")
    client = FakeStorageClient()

    stage_npa_source(bucket="b", source_root=root, client=client, max_workers=1)

    assert len(client.uploads) == 3


def test_storage_client_falls_back_to_saved_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`stage-src` must work on a machine configured only via `npa configure`.

    Regression: StorageClient.from_environment reads AWS_* env vars only, so
    staging died with "Unable to locate credentials" despite a populated
    ~/.npa/credentials.yaml.
    """
    from types import SimpleNamespace

    from npa.orchestration.npa_workflow import src_staging

    for var in ("AWS_ENDPOINT_URL", "NEBIUS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "npa.clients.credentials.load_credentials",
        lambda *a, **k: SimpleNamespace(
            s3_endpoint="https://storage.example.invalid",
            s3_access_key_id="AK_saved",
            s3_secret_access_key="SK_saved",
        ),
    )
    captured: dict = {}

    class _Client:
        @classmethod
        def from_environment(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr("npa.clients.storage.StorageClient", _Client)

    src_staging._storage_client()

    assert captured == {
        "endpoint_url": "https://storage.example.invalid",
        "aws_access_key_id": "AK_saved",
        "aws_secret_access_key": "SK_saved",
    }


def test_storage_client_prefers_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.orchestration.npa_workflow import src_staging

    def _must_not_load(*_a, **_k):
        raise AssertionError("saved credentials must not be read when all values are given")

    monkeypatch.setattr("npa.clients.credentials.load_credentials", _must_not_load)
    captured: dict = {}

    class _Client:
        @classmethod
        def from_environment(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr("npa.clients.storage.StorageClient", _Client)

    src_staging._storage_client(
        endpoint_url="https://explicit.invalid",
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
    )

    assert captured["endpoint_url"] == "https://explicit.invalid"
    assert captured["aws_access_key_id"] == "AK"
