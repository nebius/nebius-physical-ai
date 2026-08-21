"""Static and lightweight runtime gates for the redistributable Cosmos2 image."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from npa.deploy.images import supported_tool_version
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workbench.cosmos.fixture import generate_fixture

ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = ROOT / "npa" / "docker" / "workbench" / "cosmos2-transfer"
DOCKERFILE = IMAGE_DIR / "Dockerfile"
SOURCE_SHA = "67d56b7d550a3911024a32dc23ae0bae5258e633"
SAM2_SOURCE_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
BUILD_BASE_SHA = (
    "sha256:3986465b3dd3b4d602c07061f2cff417e0bfb24810129408d4eb12e111015a6c"
)
RUNTIME_BASE_SHA = (
    "sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
)
EXACT_TAG = "2.5.1-sam2-multigpu-20260817-r2"


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_source_base_uv_and_python_are_immutable() -> None:
    text = _dockerfile()
    assert f"COSMOS_TRANSFER_REVISION={SOURCE_SHA}" in text
    assert f"SAM2_REVISION={SAM2_SOURCE_SHA}" in text
    assert f"cudnn-devel-ubuntu24.04@{BUILD_BASE_SHA}" in text
    assert f"cudnn-runtime-ubuntu24.04@{RUNTIME_BASE_SHA}" in text
    assert "FROM ${CUDA_RUNTIME_IMAGE} AS runtime" in text
    assert re.search(r"UV_IMAGE=\S+@sha256:[0-9a-f]{64}", text)
    assert "COSMOS_PYTHON_VERSION=3.10.18" in text
    assert "UBUNTU_SNAPSHOT=20260801T053000Z" in text
    assert (
        text.count("URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/") == 2
    )
    assert "/etc/apt/sources.list.d/*.sources" in text
    assert "archive.ubuntu.com" not in text
    assert "security.ubuntu.com" not in text
    assert "developer.download.nvidia.com" not in text
    assert "uv sync --locked --no-dev --no-editable --extra=cu128" in text
    assert "uv pip uninstall --python .venv/bin/python sam2" in text
    assert "SAM2_BUILD_CUDA=0 uv pip install" in text
    assert "https://github.com/facebookresearch/sam2.git" in text
    assert 'version("SAM-2") == "1.0"' in text
    assert "apt-get upgrade -y" in text


def test_lfs_media_models_and_build_credentials_are_excluded() -> None:
    text = _dockerfile()
    assert text.count("GIT_LFS_SKIP_SMUDGE=1") >= 3
    assert "filter.lfs.smudge=" in text
    assert "rm -rf .git assets" in text
    assert "matplotlib/mpl-data/sample_data" in text
    assert "skimage/data" in text
    assert "wandb/bin" in text
    assert "site-packages/scipy" in text
    assert "numpy/_core/tests/_natype.py" in text
    assert "from transformers import SiglipModel" in text
    assert "assert-no-forbidden-cosmos2-payload" in text
    assert not re.search(r"(?im)^\s*(ARG|ENV)\s+(HF_TOKEN|NGC_API_KEY|.*SECRET)", text)
    assert not re.search(r"(?i)(huggingface|ngc).*(download|snapshot_download)", text)
    assert not re.search(r"(?im)^\s*(COPY|ADD)\s+.*assets", text)

    overrides = (IMAGE_DIR / "security-overrides.txt").read_text(encoding="utf-8")
    assert "nltk-3.10.0-py3-none-any.whl" in overrides
    assert (
        "sha256=54ff84d4916d3ef127e8953bee0023f6a6b320b75d634a19e06ef056d3d244bf"
        in overrides
    )
    assert "defusedxml-0.7.1-py2.py3-none-any.whl" in overrides
    assert (
        "sha256=a352e7e428770286cc899e2542b6cdaedb2b4953ff269a210103ec58f6198a61"
        in overrides
    )
    assert "pip-26.2-py3-none-any.whl" in overrides
    assert (
        "sha256=931c303696af6fa3417112103b1cad26890e5a07eccb5b99783700e33f2b8aad"
        in overrides
    )
    assert "msgpack-1.2.1-cp310-cp310-manylinux" in overrides
    assert (
        "sha256=83efa1c898e0fc5380fc0cabbf75164c52e3b5cbb45973710d75821928380c73"
        in overrides
    )
    assert "setuptools-83.0.0-py3-none-any.whl" in overrides
    assert (
        "sha256=29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3"
        in overrides
    )
    assert "files.pythonhosted.org" in overrides
    assert overrides.count("#sha256=") == 5
    assert re.search(
        r"uv pip install --python \.venv/bin/python --no-deps\s+\\\s+"
        r"--requirement /tmp/cosmos2-security-overrides\.txt",
        text,
    )
    # The direct-URL security overrides carry URL-fragment hashes; the separate
    # NPA CLI wheel overlay below deliberately uses pip-style --require-hashes.
    assert 'version("nltk") == "3.10.0"' in text
    assert 'version("defusedxml") == "0.7.1"' in text
    assert 'version("pip") == "26.2"' in text
    assert 'version("msgpack") == "1.2.1"' in text
    assert 'version("setuptools") == "83.0.0"' in text
    assert ".venv/bin/python -m pip --version" in text
    assert 'importlib.util.find_spec("pip") is None' in text
    assert 'importlib.util.find_spec("setuptools") is None' in text

    cli_requirements = (IMAGE_DIR / "npa-cli-requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "typer==0.24.1" in cli_requirements
    assert "kubernetes==33.1.0" in cli_requirements
    assert "fastapi==0.136.1" in cli_requirements
    requirement_lines = [
        line
        for line in cli_requirements.splitlines()
        if line and not line.startswith(("#", " "))
    ]
    assert len(requirement_lines) == 17
    assert cli_requirements.count("--hash=sha256:") == len(requirement_lines)
    assert "--no-deps --require-hashes" in text
    assert "pip install --no-deps /opt/npa" not in text
    assert "python -m npa.cli.entry" in text
    assert "python -m npa.cli.main" not in text
    assert "PYTHONPATH=/opt/npa/src" in text
    assert (
        "NPA_BAKED_PYTHON=/opt/cosmos/cosmos-transfer2.5/.venv/bin/python" in text
    )
    assert "ln -sfn /opt/npa/src/npa" in text
    assert 'env -u PYTHONPATH "${NPA_BAKED_PYTHON}" -c' in text
    assert "no build backend or package index is consulted" in text
    assert "import npa.cli.entry" in text
    assert "import npa.cli.main" not in text
    assert "workbench cosmos2 transfer --help" in text
    assert "grep -q -- '--control-asset'" in text


def test_forbidden_payload_guard_rejects_weight_and_media(tmp_path: Path) -> None:
    guard = IMAGE_DIR / "assert_no_forbidden_payload.sh"
    clean = tmp_path / "clean"
    clean.mkdir()
    subprocess.run(["bash", str(guard), str(clean)], check=True)

    weight = clean / "tiny.pt"
    weight.write_bytes(b"not harmless")
    proc = subprocess.run(
        ["bash", str(guard), str(clean)], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "model/checkpoint" in proc.stderr
    weight.unlink()

    allowed_site = clean / "lib" / "python3.10" / "site-packages"
    allowed_site.mkdir(parents=True)
    (allowed_site / "_virtualenv.pth").write_text(
        "import _virtualenv\n", encoding="utf-8"
    )
    (allowed_site / "distutils-precedence.pth").write_text(
        "import _distutils_hack\n", encoding="utf-8"
    )
    sobol = allowed_site / "scipy" / "stats" / "_sobol_direction_numbers.npz"
    sobol.parent.mkdir(parents=True)
    sobol.write_bytes(b"classified BSD runtime table")
    subprocess.run(["bash", str(guard), str(clean)], check=True)

    (allowed_site / "unclassified.pth").write_bytes(b"not a path config")
    proc = subprocess.run(
        ["bash", str(guard), str(clean)], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "unclassified.pth" in proc.stderr
    (allowed_site / "unclassified.pth").unlink()

    (clean / "sample.mp4").write_bytes(b"media")
    proc = subprocess.run(
        ["bash", str(guard), str(clean)], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "upstream media" in proc.stderr
    (clean / "sample.mp4").unlink()

    skimage_data = (
        clean / ".venv" / "lib" / "python3.10" / "site-packages" / "skimage" / "data"
    )
    skimage_data.mkdir(parents=True)
    proc = subprocess.run(
        ["bash", str(guard), str(clean)], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "scikit-image data/fetch" in proc.stderr
    skimage_data.rmdir()

    wandb_bin = (
        clean / ".venv" / "lib" / "python3.10" / "site-packages" / "wandb" / "bin"
    )
    wandb_bin.mkdir(parents=True)
    proc = subprocess.run(
        ["bash", str(guard), str(clean)], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "W&B native service" in proc.stderr


def test_final_runtime_is_non_root_relocated_and_cache_writable_by_design() -> None:
    text = _dockerfile()
    assert re.search(r"(?m)^USER ubuntu$", text)
    assert 'VOLUME ["/opt/cosmos/model-cache"]' in text
    assert "NLTK_DATA=/opt/cosmos/model-cache" in text
    assert "from nltk.pathsec import _get_allowed_roots" in text
    assert "LD_LIBRARY_PATH=/usr/local/cuda/lib64:" in text
    assert 'ln -sfn libcudart.so.12 "${cudart_dir}/libcudart.so"' in text
    assert 'ctypes.CDLL(\\"libcudart.so\\")' in text
    assert "UV_PYTHON_INSTALL_DIR=/opt/cosmos/uv-python" in text
    assert "! grep -q '/root' .venv/pyvenv.cfg" in text
    assert "COPY --from=build --chown=1000:1000 /opt/cosmos /opt/cosmos" in text
    assert 'exec /opt/cosmos/cosmos-transfer2.5/.venv/bin/python "$@"' in text
    assert "> /usr/local/bin/python3" in text
    assert "npa-cli-requirements.txt" in text
    assert "--no-deps --require-hashes" in text
    assert "-m npa.cli.entry" in text
    assert "-m npa.cli.main" not in text
    assert "/opt/cosmos/cosmos-transfer2.5/.venv/bin/npa" in text
    assert "workbench cosmos2 transfer --help" in text
    assert "rm -rf /opt/cosmos/model-cache/xdg/uv" in text
    assert "chown -R ubuntu:ubuntu /opt/cosmos/model-cache" in text
    assert "/opt/cosmos/model-cache/xdg/uv/.npa-write-probe" in text
    assert 'python3 -c "import cosmos_transfer2"' in text
    assert "python3 -m pip --version" in text
    assert "COPY --from=uv-bin /uv /usr/local/bin/uv" in text
    assert "COPY --from=uv-bin /uvx /usr/local/bin/uvx" in text
    assert 'test "$(command -v uvx)" = /usr/local/bin/uvx && uvx --version' in text
    assert "chown -R ubuntu:ubuntu /opt/cosmos " not in text
    assert "chown -R ubuntu:ubuntu /opt/cosmos/model-cache" in text
    assert "test -w /opt/cosmos/model-cache/huggingface" in text
    assert "rm -f /etc/ssh/ssh_host_*" in text


def test_entrypoint_passes_arbitrary_argv_and_refuses_tokenless_inference() -> None:
    entrypoint = IMAGE_DIR / "entrypoint.sh"
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    passthrough = subprocess.run(
        ["bash", str(entrypoint), "/bin/sh", "-c", "printf skypilot-ready"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert passthrough.stdout == "skypilot-ready"

    denied = subprocess.run(
        ["bash", str(entrypoint), "/bin/sh", "-c", "python examples/inference.py"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert denied.returncode == 78
    assert "no download was attempted" in denied.stderr
    assert "HF_TOKEN=" not in denied.stderr


def test_build_script_resolves_registry_without_committed_identifier() -> None:
    build_script = IMAGE_DIR / "build.sh"
    text = build_script.read_text(encoding="utf-8")
    assert os.access(build_script, os.X_OK)
    assert os.access(IMAGE_DIR / "assert_no_forbidden_payload.sh", os.X_OK)
    assert "resolve_container_registry" in text
    assert "docker buildx build" in text
    assert "--push --provenance=mode=max --sbom=true" in text
    assert "env -u HF_TOKEN" in text
    assert "cr.eu-" not in text
    assert re.search(r"\b[etu][0-9a-z]{18,}\b", text) is None


def test_procedural_fixture_is_real_multistep_video(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required for procedural-video validation")

    result = generate_fixture(
        tmp_path,
        width=320,
        height=240,
        fps=8,
        frames=16,
        num_steps=2,
    )
    spec = json.loads(Path(result["spec_path"]).read_text(encoding="utf-8"))
    assert spec["num_steps"] == 2
    assert spec["edge"]["control_weight"] == 1.0
    assert spec["video_path"] == str(Path(result["video_path"]).resolve())
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames",
            "-of",
            "json",
            result["video_path"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (320, 240)
    assert int(stream["nb_read_frames"]) == 16


def test_exact_pin_golden_eval_and_workflow_use_the_legal_path() -> None:
    assert supported_tool_version("cosmos2-transfer") == EXACT_TAG
    golden = yaml.safe_load(
        (ROOT / "npa" / "src" / "npa" / "smoke" / "golden_evals.yaml").read_text(
            encoding="utf-8"
        )
    )["containers"]["cosmos2-transfer"]
    assert (
        golden["golden_eval"]["command"]
        == "bash /opt/cosmos2-transfer/smoke_functional.sh"
    )
    smoke = (IMAGE_DIR / "smoke_functional.sh").read_text(encoding="utf-8")
    assert "from npa.workbench.cosmos.transfer import _classify_output_videos" in smoke
    assert "_classify_output_videos(out)" in smoke
    assert 'if "control" not in Path(p).name.lower()' not in smoke
    assert "procedural input" in golden["safety"]["notes"]
    assert "built outside this repo" not in golden["safety"]["notes"]
    argv = TOOL_CATALOG["workbench.cosmos2.transfer_execute"].argv_template
    assert "--condition-on-input" in argv
    assert "--execute" in argv
    workflow = yaml.safe_load(
        (
            ROOT
            / "npa"
            / "workflows"
            / "workbench"
            / "npa-workflows"
            / "cosmos2-transfer.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        workflow["states"]["transfer"]["toolRef"]
        == "workbench.cosmos2.transfer_execute"
    )
    assert workflow["config"]["trigger_uri"].endswith("/input/")
    assert workflow["resources"]["transfer-gpu"]["accelerators"] == "RTXPRO6000:1"
    pod_spec = workflow["resources"]["transfer-gpu"]["kubernetes"]["pod_config"]["spec"]
    assert pod_spec["volumes"] == [{"name": "cosmos2-model-cache", "emptyDir": {}}]
    assert pod_spec["containers"][0]["name"] == "ray-node"
    assert pod_spec["containers"][0]["volumeMounts"] == [
        {"name": "cosmos2-model-cache", "mountPath": "/opt/cosmos/model-cache"}
    ]


def test_redistribution_record_covers_every_artifact_category() -> None:
    record = (IMAGE_DIR / "REDISTRIBUTION.md").read_text(encoding="utf-8")
    for item in (
        "Cosmos Transfer source",
        "Git LFS objects",
        "model weights",
        "CUDA/cuDNN base runtime",
        "Python dependencies",
        "Live-test input",
        SOURCE_SHA,
        SAM2_SOURCE_SHA,
        BUILD_BASE_SHA,
        RUNTIME_BASE_SHA,
    ):
        assert item in record
    assert "not legal advice" in record
    assert "sam2==1.1.0" in record
    assert "unaffiliated PyPI repackaging" in record
    assert "SAM-2 1.0" in record
    assert "facebook/sam2.1-hiera-tiny" in record
    assert "Default/off runs do not fetch or invoke SAM2" in record
