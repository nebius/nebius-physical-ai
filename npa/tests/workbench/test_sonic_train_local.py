"""In-job (``--runtime local``) SONIC training and the train -> export -> eval chain.

torch is imported through ``importorskip`` so the standard unit suite (installed
without the ``sonic`` extra) skips instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.workbench.sonic.train import (
    CHECKPOINT_FILE_NAME,
    CHECKPOINT_FORMAT,
    MANIFEST_FILE_NAME,
    REFERENCE_TRAINER,
    TRAIN_MANIFEST_FORMAT,
    SonicTrainError,
    resolve_entrypoint,
    train_local,
)


def test_train_local_requires_output_path() -> None:
    with pytest.raises(SonicTrainError, match="requires --output-path"):
        train_local(output_path="")


def test_train_local_rejects_non_positive_iterations(tmp_path: Path) -> None:
    with pytest.raises(SonicTrainError, match="--max-iterations must be positive"):
        train_local(output_path=str(tmp_path / "out"), max_iterations=0)


def test_resolve_entrypoint_ignores_missing_file(tmp_path: Path) -> None:
    assert resolve_entrypoint(str(tmp_path / "nope.sh")) == ""


def test_resolve_entrypoint_finds_executable(tmp_path: Path) -> None:
    script = tmp_path / "entrypoint.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    assert resolve_entrypoint(str(script)) == str(script)


def test_train_local_reference_trainer_writes_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "training"
    result = train_local(
        output_path=str(out),
        max_iterations=2,
        num_envs=8,
        device="cpu",
        # A SONIC image would shell out to its own trainer; force the in-process
        # reference trainer so this stays hermetic.
        allow_entrypoint=False,
    )

    assert result["status"] == "trained"
    assert result["runtime"] == "local"
    assert result["trainer"] == REFERENCE_TRAINER
    assert result["iterations"] == 2
    assert result["observation_dim"] > 0
    assert result["action_dim"] > 0
    # Real gradient descent: the fitted loss must be below where it started.
    assert result["final_loss"] < result["initial_loss"]

    checkpoint = out / CHECKPOINT_FILE_NAME
    assert checkpoint.is_file()
    manifest = json.loads((out / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == TRAIN_MANIFEST_FORMAT
    assert manifest["trainer"] == REFERENCE_TRAINER
    assert len(manifest["metrics"]["iteration_loss"]) == 2


def test_train_local_checkpoint_exports_and_evaluates(tmp_path: Path) -> None:
    """The chain the sonic-export-eval twin runs, end to end on CPU."""

    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from npa.workbench.sonic import export_onnx
    from npa.workbench.sonic.eval import evaluate_onnx_policy

    train_local(
        output_path=str(tmp_path / "training"),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    exported = export_onnx(
        checkpoint=str(tmp_path / "training" / CHECKPOINT_FILE_NAME),
        output=str(tmp_path / "onnx"),
        verify=True,
    )
    assert exported.status == "exported"
    assert exported.parity is not None and exported.parity["passed"]
    # The exported graph carries the observation normalization the trainer
    # measured, so eval can feed it raw observations.
    assert exported.normalize == "baked"

    result = evaluate_onnx_policy(
        onnx=exported.onnx_path,
        episodes=2,
        env="locomotion-smoke",
        output=str(tmp_path / "eval.json"),
    )
    assert result["status"] == "completed"
    assert result["metrics"]["valid_action_rate"] == 1.0
    assert Path(result["result_uri"]).is_file()


class _FakeStorage:
    """Local-filesystem stand-in for ``StorageClient`` keyed by s3:// URI."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _local(self, uri: str) -> Path:
        return self.root / uri.removeprefix("s3://")

    def upload_file(self, local_file: str, bucket_uri: str) -> str:
        target = self._local(bucket_uri)
        if not target.suffix:
            target = target / Path(local_file).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local_file).read_bytes())
        return f"s3://{target.relative_to(self.root).as_posix()}"

    def upload_directory(self, local_dir: str, bucket_uri: str) -> str:
        for item in sorted(Path(local_dir).rglob("*")):
            if item.is_file():
                self.upload_file(
                    str(item),
                    bucket_uri.rstrip("/") + "/" + item.relative_to(local_dir).as_posix(),
                )
        return bucket_uri

    def download_path(self, bucket_uri: str, local_path: str) -> str:
        source = self._local(bucket_uri)
        if not source.is_file():
            raise FileNotFoundError(bucket_uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(source.read_bytes())
        return local_path


def test_sonic_chain_round_trips_through_object_storage(tmp_path: Path) -> None:
    """train -> export -> eval hand off artifacts over s3:// URIs.

    Each workflow stage runs on its own cluster, so object storage is the only
    channel between them; this is the contract the sonic-export-eval twin needs.
    """

    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from npa.workbench.sonic import export_onnx
    from npa.workbench.sonic.eval import evaluate_onnx_policy

    storage = _FakeStorage(tmp_path / "bucket")
    trained = train_local(
        output_path="s3://runs/sonic/training/",
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
        storage_client=storage,
    )
    assert trained["checkpoint_uri"] == "s3://runs/sonic/training/checkpoint.pt"
    assert storage._local(trained["checkpoint_uri"]).is_file()

    exported = export_onnx(
        checkpoint=trained["checkpoint_uri"],
        output="s3://runs/sonic/sonic_policy.onnx",
        storage_client=storage,
    )
    assert exported.onnx_path.startswith("s3://")
    assert exported.metadata_path.endswith(".metadata.json")
    assert exported.checkpoint == trained["checkpoint_uri"]

    result = evaluate_onnx_policy(
        onnx=exported.onnx_path,
        episodes=1,
        env="locomotion-smoke",
        output="s3://runs/sonic/eval.json",
        storage_client=storage,
    )
    assert result["status"] == "completed"
    assert storage._local(result["result_uri"]).is_file()


def test_export_reports_missing_staged_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from npa.workbench.sonic import SonicExportError, export_onnx

    storage = _FakeStorage(tmp_path / "bucket")
    with pytest.raises(SonicExportError, match="could not stage SONIC checkpoint"):
        export_onnx(
            checkpoint="s3://runs/sonic/training/checkpoint.pt",
            output=str(tmp_path / "onnx"),
            storage_client=storage,
        )


def test_checkpoint_holds_weights_not_a_pickled_module(tmp_path: Path) -> None:
    """The checkpoint must load under torch's restricted unpickler.

    A checkpoint that pickles the module binds the artifact to one class path
    and can only be read by executing what the file says.
    """

    torch = pytest.importorskip("torch")
    train_local(
        output_path=str(tmp_path / "training"),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    payload = torch.load(
        str(tmp_path / "training" / CHECKPOINT_FILE_NAME),
        map_location="cpu",
        weights_only=True,
    )
    assert payload["format"] == CHECKPOINT_FORMAT
    assert isinstance(payload["policy"], dict)
    assert payload["policy"]["class"].endswith("ReferenceLocomotionPolicy")
    assert isinstance(payload["policy_state_dict"], dict)
    assert not isinstance(payload["policy"], torch.nn.Module)


def test_export_bakes_the_normalization_the_trainer_measured(tmp_path: Path) -> None:
    """Rebuilding from a state dict must not drop the observation statistics.

    The exported graph takes RAW observations, so if the normalization the
    trainer measured were lost the ONNX would quietly expect pre-normalized
    input and every downstream score would be wrong while every check passed.
    """

    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from npa.workbench.sonic import export_onnx
    from npa.workbench.sonic.reference_policy import (
        ReferenceLocomotionPolicy,
        sample_observations,
    )

    train_local(
        output_path=str(tmp_path / "training"),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    exported = export_onnx(
        checkpoint=str(tmp_path / "training" / CHECKPOINT_FILE_NAME),
        output=str(tmp_path / "onnx"),
    )

    payload = torch.load(
        str(tmp_path / "training" / CHECKPOINT_FILE_NAME),
        map_location="cpu",
        weights_only=True,
    )
    policy = ReferenceLocomotionPolicy(**payload["policy"]["kwargs"])
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    stats = payload["normalization"]
    mean = torch.tensor(stats["mean"])
    std = torch.sqrt(torch.tensor(stats["var"]))

    obs = sample_observations(4, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        normalized = torch.clamp(
            (obs - mean) / std, -float(stats["clip"]), float(stats["clip"])
        )
        expected = policy(normalized).numpy()
        unnormalized = policy(obs).numpy()

    session = ort.InferenceSession(exported.onnx_path, providers=["CPUExecutionProvider"])
    actual = session.run(["action"], {"obs": obs.numpy().astype("float32")})[0]

    assert np.abs(expected - actual).max() < 1e-4
    # And the graph is genuinely doing the normalization, not passing through.
    assert np.abs(unnormalized - actual).max() > 1e-3


def test_train_local_cold_starts_from_an_unreadable_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    result = train_local(
        output_path=str(tmp_path / "training"),
        checkpoint=str(corrupt),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    assert result["warm_start"] == ""
    assert result["status"] == "trained"


def test_train_local_warm_start_ignores_hugging_face_ref(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    result = train_local(
        output_path=str(tmp_path / "training"),
        checkpoint="nvidia/GEAR-SONIC",
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    assert result["warm_start"] == ""


def test_train_local_warm_starts_from_previous_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    first = tmp_path / "run-1"
    train_local(
        output_path=str(first),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    result = train_local(
        output_path=str(tmp_path / "run-2"),
        checkpoint=str(first / CHECKPOINT_FILE_NAME),
        max_iterations=1,
        num_envs=8,
        device="cpu",
        allow_entrypoint=False,
    )
    assert result["warm_start"] == str(first / CHECKPOINT_FILE_NAME)
