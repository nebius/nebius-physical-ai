"""SONIC export object-storage staging (fully mocked; no infrastructure).

Regression context: the ``sonic-export`` npa.workflow twin failed live with
``checkpoint not found: s3://.../checkpoint.pt`` (run
``npa-wf-gpu-sonic-export-45b108b8``, SkyPilot job 184) because the tool only handled
local paths while the retired SkyPilot template did the S3 download/upload in bash.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from npa.workbench.sonic.staging import (
    DEFAULT_ONNX_NAME,
    ExportStaging,
    external_data_uri_candidates,
    is_object_uri,
    plan_export_staging,
    publish_outputs,
    resolve_object_onnx_uri,
    sidecar_uri_candidates,
    stage_eval_inputs,
    stage_inputs,
)


class FakeStorageClient:
    """Minimal StorageClient double recording both directions."""

    def __init__(self, contents: dict[str, bytes] | None = None) -> None:
        self.contents = contents or {}
        self.downloads: list[tuple[str, str]] = []
        self.uploads: list[tuple[str, str]] = []

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        self.downloads.append((bucket_uri, local_path))
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.contents.get(bucket_uri, b"staged"))
        return str(dest)

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        self.uploads.append((local_file, bucket_uri))
        return bucket_uri


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("s3://bucket/key.pt", True),
        ("  s3://bucket/key.pt", True),
        ("/tmp/key.pt", False),
        ("", False),
        (None, False),
        ({"policy": "x"}, False),
    ],
)
def test_is_object_uri(value: object, expected: bool) -> None:
    assert is_object_uri(value) is expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("s3://b/p/policy.onnx", "s3://b/p/policy.onnx"),
        ("s3://b/p/POLICY.ONNX", "s3://b/p/POLICY.ONNX"),
        ("s3://b/p/", f"s3://b/p/{DEFAULT_ONNX_NAME}"),
        ("s3://b/p", f"s3://b/p/{DEFAULT_ONNX_NAME}"),
    ],
)
def test_resolve_object_onnx_uri_matches_local_semantics(output: str, expected: str) -> None:
    assert resolve_object_onnx_uri(output) == expected


def test_stage_inputs_downloads_only_object_values(tmp_path: Path) -> None:
    client = FakeStorageClient()

    staged = stage_inputs(
        {
            "checkpoint": "s3://bucket/run/checkpoint.pt",
            "config": "/local/config.yaml",
            "obs_spec": None,
            "action_spec": {"inline": "spec"},
        },
        workdir=tmp_path,
        storage_client=client,
    )

    assert set(staged) == {"checkpoint"}
    assert staged["checkpoint"] == str(tmp_path / "checkpoint.pt")
    assert Path(staged["checkpoint"]).read_bytes() == b"staged"
    assert client.downloads == [("s3://bucket/run/checkpoint.pt", staged["checkpoint"])]


def test_stage_inputs_constructs_no_client_for_local_only_values(tmp_path: Path) -> None:
    """A purely local run must not touch object storage at all."""

    staged = stage_inputs(
        {"checkpoint": "/local/checkpoint.pt"}, workdir=tmp_path, storage_client=None
    )

    assert staged == {}


def test_plan_export_staging_for_an_object_output(tmp_path: Path) -> None:
    client = FakeStorageClient()

    plan = plan_export_staging(
        workdir=tmp_path,
        output="s3://bucket/run/export/",
        inputs={"checkpoint": "s3://bucket/run/checkpoint.pt"},
        storage_client=client,
    )

    assert plan.stages_output
    assert plan.onnx_uri == f"s3://bucket/run/export/{DEFAULT_ONNX_NAME}"
    assert plan.local_output.endswith(f"/export/{DEFAULT_ONNX_NAME}")
    assert Path(plan.local_output).parent.is_dir()
    assert plan.inputs["checkpoint"].endswith("checkpoint.pt")


def test_plan_export_staging_for_a_local_output_keeps_the_path(tmp_path: Path) -> None:
    plan = plan_export_staging(
        workdir=tmp_path,
        output="/out/policy.onnx",
        inputs={"checkpoint": "/local/checkpoint.pt"},
    )

    assert not plan.stages_output
    assert plan.local_output == "/out/policy.onnx"
    assert plan.onnx_uri == ""


def test_publish_outputs_uploads_the_onnx_and_its_sidecar(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    onnx = export_dir / DEFAULT_ONNX_NAME
    onnx.write_bytes(b"onnx")
    # The exporter writes `<stem>.metadata.json` (Path.with_suffix), not
    # `<name>.onnx.metadata.json`.
    sidecar = export_dir / "sonic_policy.metadata.json"
    sidecar.write_text("{}", encoding="utf-8")
    (export_dir / "subdir").mkdir()
    client = FakeStorageClient()
    plan = ExportStaging(
        workdir=tmp_path,
        local_output=str(onnx),
        onnx_uri=f"s3://bucket/run/export/{DEFAULT_ONNX_NAME}",
    )

    published = publish_outputs(plan, storage_client=client)

    assert published[str(onnx)] == f"s3://bucket/run/export/{DEFAULT_ONNX_NAME}"
    assert published[str(sidecar)] == "s3://bucket/run/export/sonic_policy.metadata.json"
    # Directories are skipped, and every upload lands under the ONNX's prefix.
    assert len(published) == 2
    assert all(uri.startswith("s3://bucket/run/export/") for _, uri in client.uploads)


def test_publish_outputs_is_a_no_op_for_local_outputs(tmp_path: Path) -> None:
    plan = ExportStaging(workdir=tmp_path, local_output=str(tmp_path / "p.onnx"))

    assert publish_outputs(plan, storage_client=FakeStorageClient()) == {}


@pytest.mark.parametrize(
    ("onnx_uri", "expected"),
    [
        (
            "s3://b/p/sonic_policy.onnx",
            ("s3://b/p/sonic_policy.metadata.json", "s3://b/p/sonic_policy.onnx.metadata.json"),
        ),
        ("s3://b/p/model", ("s3://b/p/model.metadata.json", "s3://b/p/model.metadata.json")),
    ],
)
def test_sidecar_uri_candidates_match_the_local_resolver(
    onnx_uri: str, expected: tuple[str, str]
) -> None:
    assert sidecar_uri_candidates(onnx_uri) == expected


def test_external_data_uri_candidates() -> None:
    assert external_data_uri_candidates("s3://b/p/sonic_policy.onnx") == (
        "s3://b/p/sonic_policy.onnx.data",
    )


def test_stage_eval_inputs_downloads_external_weights_next_to_the_model(
    tmp_path: Path,
) -> None:
    """A large ONNX is a PAIR of files; onnxruntime resolves .data relative to it."""

    client = FakeStorageClient(
        {
            "s3://bucket/run/sonic_policy.onnx": b"onnx",
            "s3://bucket/run/sonic_policy.onnx.data": b"weights",
            "s3://bucket/run/sonic_policy.metadata.json": b"{}",
        }
    )

    local_onnx, _ = stage_eval_inputs(
        onnx="s3://bucket/run/sonic_policy.onnx",
        metadata=None,
        workdir=tmp_path,
        storage_client=client,
    )

    sibling = Path(local_onnx).parent / "sonic_policy.onnx.data"
    assert sibling.is_file()
    assert sibling.read_bytes() == b"weights"


def test_stage_eval_inputs_downloads_the_onnx_and_its_sidecar(tmp_path: Path) -> None:
    client = FakeStorageClient(
        {
            "s3://bucket/run/sonic_policy.onnx": b"onnx",
            "s3://bucket/run/sonic_policy.metadata.json": b"{}",
        }
    )

    local_onnx, local_metadata = stage_eval_inputs(
        onnx="s3://bucket/run/sonic_policy.onnx",
        metadata=None,
        workdir=tmp_path,
        storage_client=client,
    )

    assert Path(local_onnx).name == "sonic_policy.onnx"
    # Landed under the name the local resolver tries first (Path.with_suffix).
    assert local_metadata is not None
    assert Path(local_metadata).name == "sonic_policy.metadata.json"
    assert client.downloads[0][0] == "s3://bucket/run/sonic_policy.onnx"


def test_stage_eval_inputs_falls_back_to_the_appended_sidecar_name(tmp_path: Path) -> None:
    class OnlyAppendedSidecar(FakeStorageClient):
        def download_path(self, bucket_uri: str, local_path: str) -> str:
            if bucket_uri.endswith("sonic_policy.metadata.json"):
                raise FileNotFoundError(bucket_uri)
            return super().download_path(bucket_uri, local_path)

    client = OnlyAppendedSidecar()

    _, local_metadata = stage_eval_inputs(
        onnx="s3://bucket/run/sonic_policy.onnx",
        metadata=None,
        workdir=tmp_path,
        storage_client=client,
    )

    assert local_metadata is not None
    assert "s3://bucket/run/sonic_policy.onnx.metadata.json" in [
        uri for uri, _ in client.downloads
    ]


def test_stage_eval_inputs_passes_local_paths_through(tmp_path: Path) -> None:
    assert stage_eval_inputs(
        onnx="/local/policy.onnx", metadata="/local/meta.json", workdir=tmp_path
    ) == ("/local/policy.onnx", "/local/meta.json")


def test_stage_eval_inputs_downloads_an_explicit_sidecar_uri(tmp_path: Path) -> None:
    client = FakeStorageClient()

    _, local_metadata = stage_eval_inputs(
        onnx="s3://bucket/run/sonic_policy.onnx",
        metadata="s3://bucket/run/custom.metadata.json",
        workdir=tmp_path,
        storage_client=client,
    )

    assert local_metadata is not None
    assert Path(local_metadata).name == "custom.metadata.json"


def test_export_onnx_round_trips_object_storage(tmp_path: Path) -> None:
    """End-to-end through the real exporter, with storage mocked."""

    torch = pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")
    from npa.workbench.sonic import export_onnx
    from npa.workflows.sonic_fixture import build_checkpoint

    checkpoint = tmp_path / "checkpoint.pt"
    build_checkpoint(checkpoint, obs_dim=6, act_dim=3, hidden=8)
    client = FakeStorageClient({"s3://bucket/run/checkpoint.pt": checkpoint.read_bytes()})

    result = export_onnx(
        checkpoint="s3://bucket/run/checkpoint.pt",
        output="s3://bucket/run/export/",
        verify=True,
        storage_client=client,
    )

    assert result.status == "exported"
    # The reported paths are the object URIs the next stage can consume, not temp dirs.
    assert result.onnx_path == f"s3://bucket/run/export/{DEFAULT_ONNX_NAME}"
    assert result.metadata_path == "s3://bucket/run/export/sonic_policy.metadata.json"
    assert result.checkpoint == "s3://bucket/run/checkpoint.pt"
    assert result.obs_dim == 6 and result.action_dim == 3
    assert result.parity and result.parity["passed"] is True
    uploaded = {uri for _, uri in client.uploads}
    assert result.onnx_path in uploaded and result.metadata_path in uploaded
    del torch  # only needed to gate the test


def test_export_onnx_local_path_needs_no_storage_client(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="torch ships in the npa[sonic] extra")
    from npa.workbench.sonic import export_onnx
    from npa.workflows.sonic_fixture import build_checkpoint

    checkpoint = tmp_path / "checkpoint.pt"
    build_checkpoint(checkpoint, obs_dim=5, act_dim=2, hidden=6)

    result = export_onnx(checkpoint=str(checkpoint), output=str(tmp_path / "out"))

    assert result.status == "exported"
    assert result.onnx_path == str(tmp_path / "out" / DEFAULT_ONNX_NAME)
    assert Path(result.onnx_path).is_file()
