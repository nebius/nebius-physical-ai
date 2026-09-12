from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "npa" / "scripts" / "scan_image_robomimic_payload.py"
SPEC = importlib.util.spec_from_file_location("scan_image_robomimic_payload", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def test_clean_neutral_layer_passes(tmp_path: Path) -> None:
    layer = _tar(
        tmp_path / "clean.tar",
        {
            "opt/robomimic/LICENSE": b"MIT\n",
            "opt/robomimic/robomimic/models/__init__.py": b"\n",
            "opt/robomimic/robomimic/models/obs_nets.py": b"class ObservationNet:\n    pass\n",
            "opt/robomimic-deps/transformers/models/auto/configuration_auto.py": b"\n",
            "opt/robomimic-deps/diffusers/models/transformers/transformer_2d.py": b"\n",
            "opt/robomimic-deps/numpy/lib/tests/data/python3.npy": b"fixture",
            "opt/robomimic-deps/numpy/core/tests/data/py3-objarr.npz": b"fixture",
        },
    )
    assert scanner.scan(layer, {"history": [{"created_by": "COPY source"}]}) == []


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("opt/deps/site-packages/torch/__init__.py", "torch_or_triton_distribution"),
        (
            "opt/deps/site-packages/nvidia/cudnn/lib/libcudnn.so.9",
            "nvidia_python_distribution",
        ),
        ("usr/local/cuda/bin/nvcc", "cuda_tool_or_header"),
        ("workspace/byof-inputs/lift.hdf5", "dataset_payload"),
        ("workspace/byof-runs/robomimic-smoke.json", "run_output_or_proof"),
        ("opt/npa-runtime/robomimic/payload/bin/python", "populated_runtime_cache"),
        (
            "opt/robomimic/robomimic/models/pretrained.pth",
            "checkpoint_or_weight",
        ),
        (
            "opt/robomimic/robomimic/models/checkpoints/pretrained.bin",
            "checkpoint_or_weight",
        ),
        ("opt/robomimic-deps/model.onnx", "checkpoint_or_weight"),
        ("opt/robomimic-deps/policy.msgpack", "checkpoint_or_weight"),
        ("opt/robomimic-deps/weights.npz", "checkpoint_or_weight"),
        ("opt/robomimic-deps/models/policy-array.npy", "checkpoint_or_weight"),
        ("opt/robomimic-deps/checkpoint.npz", "checkpoint_or_weight"),
        ("root/.docker/config.json", "credential_file"),
    ],
)
def test_forbidden_payload_is_detected(tmp_path: Path, path: str, kind: str) -> None:
    layer = _tar(tmp_path / "bad.tar", {path: b"payload"})
    assert kind in {finding.kind for finding in scanner.scan(layer, {})}


@pytest.mark.parametrize(
    ("wheel_path", "members", "kind"),
    [
        (
            "tmp/torch-2.7.1-py3-none-any.whl",
            {"torch/__init__.py": b""},
            "torch_or_triton_distribution",
        ),
        (
            "tmp/renamed-python-payload.whl",
            {"torchvision/__init__.py": b""},
            "torch_or_triton_distribution",
        ),
        (
            "tmp/renamed-vendor-payload.whl",
            {"nvidia/cudnn/lib/libcudnn.so.9": b""},
            "nvidia_python_distribution",
        ),
    ],
)
def test_forbidden_distribution_inside_wheel_is_detected(
    tmp_path: Path, wheel_path: str, members: dict[str, bytes], kind: str
) -> None:
    layer = _tar(tmp_path / "wheel-layer.tar", {wheel_path: _zip_bytes(members)})
    assert kind in {finding.kind for finding in scanner.scan(layer, {})}


def test_deleted_later_payload_still_fails_layer_scan(tmp_path: Path) -> None:
    first = _tar(tmp_path / "first.tar", {"workspace/byof-inputs/lift.hdf5": b"data"})
    second = _tar(tmp_path / "second.tar", {"workspace/byof-inputs/.wh.lift.hdf5": b""})
    findings = scanner.scan_tars([first, second], {})
    assert "dataset_payload" in {finding.kind for finding in findings}


@pytest.mark.parametrize(
    "history",
    [
        "FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",
        "RUN pip install torch==2.7.1+cu128",
        "/bin/sh -c pip install torch==2.7.1+cu128",
        "RUN robomimic-runtime warm",
        "/bin/sh -c robomimic-runtime warm # buildkit",
        "RUN huggingface-cli download robomimic/robomimic_datasets",
        "/bin/sh -c huggingface-cli download robomimic/robomimic_datasets",
        "ENV NPA_ROBOMIMIC_ACCEPT_EULA=YES",
        "/bin/sh -c #(nop)  ENV NPA_ROBOMIMIC_ACCEPT_EULA=YES",
    ],
)
def test_forbidden_build_history_is_detected(tmp_path: Path, history: str) -> None:
    layer = _tar(tmp_path / "empty.tar", {"neutral": b"ok"})
    assert scanner.scan(layer, {"history": [{"created_by": history}]})


def test_oci_config_env_rejects_invented_acceptance_proxy(tmp_path: Path) -> None:
    layer = _tar(tmp_path / "empty.tar", {"neutral": b"ok"})
    findings = scanner.scan(
        layer,
        {"config": {"Env": ["NPA_ROBOMIMIC_ACCEPT_EULA=YES"]}},
    )

    assert {finding.kind for finding in findings} == {"invented_acceptance_proxy"}
