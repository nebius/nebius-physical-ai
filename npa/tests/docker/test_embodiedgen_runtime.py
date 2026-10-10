"""Unit boundaries for the EmbodiedGen operator-private runtime fetcher."""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from xml.etree import ElementTree as XML_ET
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
import numpy as np
from PIL import Image

from npa.orchestration.npa_workflow import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    NpaWorkflowRenderError,
    SkypilotRenderOptions,
    render_skypilot_yaml,
    validate_image_override_selectors,
)
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import spec_requires_runtime
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa" / "docker" / "workbench" / "embodiedgen"
WORKFLOW = ROOT / "workflows" / "testing" / "byof-embodiedgen.yaml"
READINESS = WORKFLOW.with_suffix(".readiness.json")
SPEC = importlib.util.spec_from_file_location(
    "embodiedgen_runtime_bootstrap", IMAGE / "runtime-bootstrap.py"
)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOOTSTRAP
SPEC.loader.exec_module(BOOTSTRAP)
ASSET_BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "embodiedgen_asset_bundle", IMAGE / "asset_bundle.py"
)
assert ASSET_BUNDLE_SPEC and ASSET_BUNDLE_SPEC.loader
ASSET_BUNDLE = importlib.util.module_from_spec(ASSET_BUNDLE_SPEC)
sys.modules[ASSET_BUNDLE_SPEC.name] = ASSET_BUNDLE
ASSET_BUNDLE_SPEC.loader.exec_module(ASSET_BUNDLE)


def test_manifest_pins_the_selected_upstream_components() -> None:
    manifest = IMAGE / "runtime-manifest.json"
    payload = BOOTSTRAP._read_manifest(manifest)
    assert payload["solution"]["license"] == "Apache-2.0"
    assert payload["backend"]["license"] == "MIT"
    assert payload["backend"]["model_revision"] == BOOTSTRAP.MODEL_REVISION
    assert payload["runtime"]["upstream_requirements"] == {
        "path": "requirements.txt",
        "sha256": BOOTSTRAP.UPSTREAM_REQUIREMENTS_SHA256,
    }
    assert payload["runtime"]["upstream_install_basic"] == {
        "path": "install/install_basic.sh",
        "sha256": BOOTSTRAP.UPSTREAM_INSTALL_BASIC_SHA256,
    }
    assert payload["runtime"]["operator_runtime"] == BOOTSTRAP.RUNTIME_CONTRACT
    assert payload["runtime"]["validation_requirements"] == list(
        BOOTSTRAP.VALIDATION_REQUIREMENTS
    )
    assert payload["runtime"]["validation_system_packages"] == list(
        BOOTSTRAP.VALIDATION_SYSTEM_PACKAGES
    )
    assert payload["runtime"]["trellis_attention_runtime"] == (
        BOOTSTRAP.TRELLIS_ATTENTION_RUNTIME_CONTRACT
    )
    assert payload["runtime"]["urdf_property_response"] == (
        BOOTSTRAP.URDF_PROPERTY_RESPONSE_CONTRACT
    )
    assert payload["runtime"]["baked"] == {
        "source": False,
        "model": False,
        "input": False,
        "cache": False,
        "output": False,
        "credentials": False,
    }


def test_manifest_refuses_a_changed_model_revision(tmp_path: Path) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["backend"]["model_revision"] = "0" * 40
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model revision"):
        BOOTSTRAP._read_manifest(path)


def test_manifest_refuses_a_changed_validation_dependency_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["runtime"]["validation_requirements"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation dependency contract"):
        BOOTSTRAP._read_manifest(path)


def test_manifest_refuses_a_changed_blackwell_runtime_contract(tmp_path: Path) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["runtime"]["operator_runtime"]["cuda_variant"] = "cu126"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime contract"):
        BOOTSTRAP._read_manifest(path)


def test_manifest_refuses_a_changed_trellis_attention_contract(tmp_path: Path) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["runtime"]["trellis_attention_runtime"]["fa3_enabled"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="attention runtime contract"):
        BOOTSTRAP._read_manifest(path)


def test_manifest_refuses_a_changed_urdf_property_response_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads((IMAGE / "runtime-manifest.json").read_text())
    payload["runtime"]["urdf_property_response"]["query_or_parse_failure"] = "fallback"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="URDF property response contract"):
        BOOTSTRAP._read_manifest(path)


def test_runtime_bootstrap_pins_and_probes_validation_dependency_closure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = (
        "numpy==1.26.4",
        "scipy==1.14.1",
        "Pillow==11.3.0",
        "trimesh==4.11.1",
        "plyfile==1.0.3",
        "tifffile==2024.8.30",
        "contourpy==1.3.0",
        "imageio==2.37.4",
        "imageio-ffmpeg==0.6.0",
        "pybullet==3.2.7",
    )
    assert BOOTSTRAP.VALIDATION_REQUIREMENTS == expected
    assert "import imageio.v3 as iio" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import imageio_ffmpeg" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import numpy as np" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import contourpy" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import plyfile" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import pymeshfix" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import pyvista as pv" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import pybullet_data" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import scipy" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import spconv.pytorch" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import tifffile" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "from io import BytesIO" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert (
        'plyfile.PlyElement.describe(vertex, "vertex")'
        in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    )
    assert (
        'plyfile.PlyData.read(ply_buffer)["vertex"].count == 1'
        in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    )
    assert 'assert scipy.__version__ == "1.14.1"' in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert (
        'assert tifffile.__version__ == "2024.8.30"'
        in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    )
    assert "import torch" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import trimesh" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import torchvision" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import vtk" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "import xformers" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert 'torch.version.cuda == "12.8"' in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "from PIL import Image" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "pv.Sphere(theta_resolution=12, phi_resolution=12)" in (
        BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    )
    assert "pymeshfix.MeshFix(np.asarray(surface.points), faces)" in (
        BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    )
    assert "iio.imwrite(video, frame, fps=1)" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE
    assert "iio.imiter(video)" in BOOTSTRAP.VALIDATION_RUNTIME_PROBE

    calls: list[tuple[list[str], dict[str, str]]] = []

    def record(argv: list[str], **kwargs) -> None:
        calls.append((argv, kwargs["env"]))

    monkeypatch.setattr(BOOTSTRAP, "_run", record)
    venv = tmp_path / "venv"
    cache = tmp_path / "cache"
    BOOTSTRAP._verify_validation_runtime(venv, cache)
    assert calls == [
        (
            [str(venv / "bin" / "python"), "-c", BOOTSTRAP.VALIDATION_RUNTIME_PROBE],
            BOOTSTRAP._venv_environment(venv, cache),
        )
    ]


def test_runtime_bootstrap_refuses_changed_upstream_requirements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requirements = tmp_path / "requirements.txt"
    expected = b"pinned upstream requirements\n"
    requirements.write_bytes(expected)
    monkeypatch.setattr(
        BOOTSTRAP,
        "UPSTREAM_REQUIREMENTS_SHA256",
        hashlib.sha256(expected).hexdigest(),
    )
    BOOTSTRAP._verify_upstream_requirements(tmp_path)
    requirements.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requirements.txt"):
        BOOTSTRAP._verify_upstream_requirements(tmp_path)


def test_model_receipt_rejects_changed_cached_bytes(tmp_path: Path) -> None:
    model = tmp_path / "TRELLIS-image-large"
    model.mkdir()
    payload = model / "weights.safetensors"
    payload.write_bytes(b"pinned model bytes")
    receipt = model / ".npa-model-receipt.json"
    BOOTSTRAP._write_model_receipt(model)
    BOOTSTRAP._verify_model_receipt(model, receipt)
    payload.write_bytes(b"changed bytes")
    with pytest.raises(RuntimeError, match="do not match"):
        BOOTSTRAP._verify_model_receipt(model, receipt)


def test_venv_receipt_rejects_changed_cached_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    dependency = venv / "lib" / "python3.12" / "site-packages" / "trellis.py"
    executable = venv / "bin" / "img3d-cli"
    dependency.parent.mkdir(parents=True)
    executable.parent.mkdir(parents=True)
    dependency.write_text("pinned dependency")
    executable.write_text("pinned executable")
    marker = venv / ".npa-embodiedgen-installed.json"
    monkeypatch.setattr(BOOTSTRAP, "_installed_freeze", lambda *_: ["trellis==1"])
    BOOTSTRAP._write_install_receipt(venv, marker, tmp_path / "cache")
    BOOTSTRAP._verify_install_receipt(venv, marker, tmp_path / "cache")
    dependency.write_text("changed dependency")
    with pytest.raises(RuntimeError, match="venv bytes"):
        BOOTSTRAP._verify_install_receipt(venv, marker, tmp_path / "cache")


def test_smoke_requires_token_factory_before_creating_a_runtime_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    monkeypatch.delenv("NPA_TOKEN_FACTORY_API_KEY", raising=False)
    cache = tmp_path / "runtime-cache"
    args = SimpleNamespace(
        manifest=IMAGE / "runtime-manifest.json",
        cache_root=str(cache),
        smoke=IMAGE / "capability_smoke.py",
    )
    with pytest.raises(RuntimeError, match="Token Factory credential"):
        BOOTSTRAP.command_smoke(args)
    assert not cache.exists()


def test_trellis_only_patch_removes_unselected_sam_import(tmp_path: Path) -> None:
    inference = tmp_path / "embodied_gen" / "utils" / "inference.py"
    image_to_3d = tmp_path / "embodied_gen" / "scripts" / "imageto3d.py"
    inference.parent.mkdir(parents=True)
    image_to_3d.parent.mkdir(parents=True)
    inference.write_text(
        "from embodied_gen.models.sam3d import Sam3dInference\n"
        "def run(pipe: TrellisImageTo3DPipeline | Sam3dInference):\n"
        "    if isinstance(pipe, TrellisImageTo3DPipeline):\n        return 1\n"
        "    elif isinstance(pipe, Sam3dInference):\n        return 2\n"
        "    else:\n        raise ValueError()\n"
    )
    image_to_3d.write_text(
        "TrellisImageTo3DPipeline.from_pretrained(\n"
        '            "microsoft/TRELLIS-image-large"\n        )'
    )
    BOOTSTRAP._patch_trellis_only_import(tmp_path)
    assert "Sam3dInference" not in inference.read_text()
    assert "NPA_EMBODIEDGEN_TRELLIS_MODEL_DIR" in image_to_3d.read_text()


def test_trellis_xformers_patch_disables_fa3_before_trellis_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    patch = source / "embodied_gen" / "utils" / "monkey_patch" / "trellis.py"
    patch.parent.mkdir(parents=True)
    patch.write_text(
        "import os\nimport sys\n\n"
        "def monkey_path_trellis():\n"
        "    current_file_path = os.path.abspath(__file__)\n"
        "    current_dir = os.path.dirname(current_file_path)\n"
        '    sys.path.append(os.path.join(current_dir, "../../.."))\n\n'
        "    from thirdparty.TRELLIS.trellis.representations import Gaussian\n"
        "    return Gaussian\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "fa3-state"
    xformers = source / "xformers"
    (xformers / "ops").mkdir(parents=True)
    (xformers / "__init__.py").write_text('__version__ = "0.0.32.post2"\n')
    (xformers / "ops" / "__init__.py").write_text("")
    (xformers / "ops" / "fmha.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def _set_use_fa3(value):\n"
        '    Path(os.environ["NPA_TEST_FA3_SENTINEL"]).write_text(str(value))\n',
        encoding="utf-8",
    )
    representations = source / "thirdparty" / "TRELLIS" / "trellis" / "representations"
    representations.mkdir(parents=True)
    for package in (
        source / "thirdparty",
        source / "thirdparty" / "TRELLIS",
        source / "thirdparty" / "TRELLIS" / "trellis",
    ):
        (package / "__init__.py").write_text("")
    (representations / "__init__.py").write_text(
        "import os\nfrom pathlib import Path\n"
        'assert Path(os.environ["NPA_TEST_FA3_SENTINEL"]).read_text() == "False"\n'
        "class Gaussian:\n    pass\n",
        encoding="utf-8",
    )
    for name in tuple(sys.modules):
        if name == "xformers" or name.startswith(("xformers.", "thirdparty.")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setenv("NPA_TEST_FA3_SENTINEL", str(sentinel))

    BOOTSTRAP._patch_trellis_xformers_fa3(source)
    patched_spec = importlib.util.spec_from_file_location(
        "embodiedgen_fa3_patch_fixture", patch
    )
    assert patched_spec and patched_spec.loader
    patched = importlib.util.module_from_spec(patched_spec)
    patched_spec.loader.exec_module(patched)
    patched.monkey_path_trellis()
    assert sentinel.read_text() == "False"


def test_trellis_attention_probe_contract_is_recorded_before_model_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    venv = tmp_path / "venv"
    cache = tmp_path / "cache"
    source.mkdir()
    venv.mkdir()
    cache.mkdir()
    payload = {
        "status": "ok",
        "xformers_version": BOOTSTRAP.XFORMERS_VERSION,
        "fa3_enabled": False,
        "dense_backend": "xformers",
        "sparse_attention_backend": "xformers",
        "capability": [12, 0],
        "dispatches": {
            "dense_self": "fa2F@2.5.7-pt",
            "dense_cross": "fa2F@2.5.7-pt",
            "block_diagonal": "fa2F@2.5.7-pt",
        },
        "finite": {
            "dense_self": True,
            "dense_cross": True,
            "block_diagonal": True,
        },
        "max_abs_error_vs_sdpa": {
            "dense_self": 0.0,
            "dense_cross": 0.0,
            "block_diagonal": 0.0,
        },
    }
    captured: dict[str, object] = {}

    def completed(*args, **kwargs):
        captured["argv"] = args[0]
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr="", args=args[0]
        )

    monkeypatch.setattr(BOOTSTRAP.subprocess, "run", completed)
    result = BOOTSTRAP._verify_trellis_attention_runtime(source, venv, cache)
    assert captured["argv"] == [
        str(venv / "bin" / "python"),
        "-c",
        BOOTSTRAP.TRELLIS_ATTENTION_RUNTIME_PROBE,
    ]
    assert captured["cwd"] == source
    assert captured["env"] == BOOTSTRAP._venv_environment(venv, cache)
    assert json.loads(result.read_text()) == payload

    payload["fa3_enabled"] = True
    with pytest.raises(RuntimeError, match="attention probe"):
        BOOTSTRAP._verify_trellis_attention_runtime(source, venv, cache)


def test_smoke_runs_the_attention_probe_before_model_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    source, venv, model = tmp_path / "source", tmp_path / "venv", tmp_path / "model"
    attention_probe, runtime_receipt = (
        tmp_path / "attention.json",
        tmp_path / "runtime.json",
    )
    calls: list[str] = []
    cache.mkdir()
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "configured-for-test-only")
    monkeypatch.setattr(BOOTSTRAP, "_read_manifest", lambda *_: {})
    monkeypatch.setattr(BOOTSTRAP, "_safe_cache_root", lambda *_: cache)
    monkeypatch.setattr(BOOTSTRAP, "_prepare", lambda *_: (source, venv))
    monkeypatch.setattr(BOOTSTRAP, "_install", lambda *_: calls.append("install"))
    monkeypatch.setattr(
        BOOTSTRAP,
        "_verify_trellis_attention_runtime",
        lambda *_: calls.append("attention") or attention_probe,
    )
    monkeypatch.setattr(
        BOOTSTRAP, "_download_model", lambda *_: calls.append("model") or model
    )
    monkeypatch.setattr(
        BOOTSTRAP,
        "_runtime_receipt",
        lambda *_: calls.append("receipt") or runtime_receipt,
    )
    monkeypatch.setattr(BOOTSTRAP, "_token_factory_environment", lambda env: env)
    monkeypatch.setattr(
        BOOTSTRAP, "_run", lambda *_args, **_kwargs: calls.append("smoke")
    )
    args = SimpleNamespace(
        manifest=IMAGE / "runtime-manifest.json",
        cache_root=str(cache),
        smoke=IMAGE / "capability_smoke.py",
    )

    assert BOOTSTRAP.command_smoke(args) == 0
    assert calls == ["install", "attention", "model", "receipt", "smoke"]


def test_runtime_bootstrap_repairs_a_broken_incomplete_venv_before_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = tmp_path / "venv"
    python = venv / "bin" / "python"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    site_packages = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    pip_metadata = next(site_packages.glob("pip-*.dist-info"))
    bundled_version = subprocess.check_output(
        [str(python), "-c", "import ensurepip; print(ensurepip.version())"],
        text=True,
    ).strip()
    pip_metadata.rename(site_packages / "pip-0.0.1.dist-info")
    metadata = site_packages / "pip-0.0.1.dist-info" / "METADATA"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(
            f"Version: {bundled_version}", "Version: 0.0.1", 1
        ),
        encoding="utf-8",
    )
    (site_packages / "pip" / "__init__.py").write_text(
        'raise AttributeError("pkgutil.ImpImporter")\n', encoding="utf-8"
    )
    broken = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, text=True
    )
    assert broken.returncode != 0

    calls: list[list[str]] = []
    original_run = BOOTSTRAP._run

    def recover_then_record(argv: list[str], **kwargs) -> None:
        if argv[2:4] == ["ensurepip", "--upgrade"]:
            original_run(argv, **kwargs)
        else:
            calls.append(argv)

    monkeypatch.setattr(BOOTSTRAP, "_run", recover_then_record)
    BOOTSTRAP._repair_pip(venv, tmp_path / "cache")
    repaired = subprocess.run(
        [str(python), "-m", "pip", "--version"], capture_output=True, text=True
    )
    assert repaired.returncode == 0, repaired.stderr
    assert calls == [
        [
            str(python),
            "-m",
            "pip",
            "install",
            BOOTSTRAP.PIP_REQUIREMENT,
            "setuptools==80.10.2",
            "wheel",
        ]
    ]


def test_runtime_bootstrap_patches_the_upstream_installer_for_blackwell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer = tmp_path / "install" / "install_basic.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(
        "#!/bin/bash\n"
        "detect_cuda_variant() { printf '%s\\n' cu126; }\n"
        "CUDA_VARIANT=$(detect_cuda_variant)\n"
        'TORCH_INDEX_URL="${EMBODIEDGEN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_VARIANT}"\n'
        "PIP_INSTALL_PACKAGES=(\n"
        '    "pip==22.3.1"\n'
        '    "clip@git+https://github.com/openai/CLIP.git"\n'
        '    "kolors@git+https://github.com/HochCC/Kolors.git"\n'
        '    "--no-build-isolation diff-gaussian-rasterization@git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=mesh"\n'
        ")\n"
        'printf "%s\\n" "$CUDA_VARIANT" "$TORCH_INDEX_URL" "${PIP_INSTALL_PACKAGES[0]}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        BOOTSTRAP,
        "UPSTREAM_INSTALL_BASIC_SHA256",
        hashlib.sha256(installer.read_bytes()).hexdigest(),
    )

    BOOTSTRAP._pin_install_script(tmp_path)
    completed = subprocess.run(
        ["bash", str(installer)],
        env={
            **os.environ,
            "EMBODIEDGEN_CUDA_VARIANT": BOOTSTRAP.CUDA_VARIANT,
            "EMBODIEDGEN_TORCH_INDEX_URL": BOOTSTRAP.TORCH_INDEX_URL,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "cu128",
        BOOTSTRAP.TORCH_INDEX_URL,
        BOOTSTRAP.PIP_REQUIREMENT,
    ]


@pytest.mark.parametrize(
    ("image_format", "expected_mime"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png")],
)
def test_token_factory_mime_patch_uses_decoded_format_not_staged_suffix(
    tmp_path: Path, image_format: str, expected_mime: str
) -> None:
    source = tmp_path / "embodied_gen" / "utils"
    source.mkdir(parents=True)
    client = source / "gpt_clients.py"
    client.write_text(
        "import base64\nimport os\nfrom io import BytesIO\nfrom PIL import Image\n"
        "class Client:\n"
        "    def __init__(self):\n"
        '        self.image_formats = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}\n'
        "    def build(self, image_base64):\n"
        '        content_user = [{"type": "text", "text": "probe"}]\n'
        "        if image_base64 is not None:\n"
        "            if not isinstance(image_base64, list):\n"
        "                image_base64 = [image_base64]\n"
        "            for img in image_base64:\n"
        "                if isinstance(img, Image.Image):\n"
        "                    buffer = BytesIO()\n"
        '                    img.save(buffer, format=img.format or "PNG")\n'
        "                    buffer.seek(0)\n"
        "                    image_binary = buffer.read()\n"
        '                    img = base64.b64encode(image_binary).decode("utf-8")\n'
        "                elif (\n"
        "                    len(os.path.splitext(img)) > 1\n"
        "                    and os.path.splitext(img)[-1].lower() in self.image_formats\n"
        "                ):\n"
        "                    if not os.path.exists(img):\n"
        '                        raise FileNotFoundError(f"Image file not found: {img}")\n'
        '                    with open(img, "rb") as f:\n'
        '                        img = base64.b64encode(f.read()).decode("utf-8")\n'
        "                content_user.append(\n"
        "                    {\n"
        '                        "type": "image_url",\n'
        '                        "image_url": {"url": f"data:image/png;base64,{img}"},\n'
        "                    }\n"
        "                )\n"
        "        return content_user\n",
        encoding="utf-8",
    )
    # require_input stages every accepted format at input.jpg. Exercise both
    # real formats through that worker boundary rather than their original names.
    input_path = tmp_path / "input.jpg"
    image = Image.new("RGB", (2, 2), color=(12, 34, 56))
    image.save(input_path, format=image_format)
    image_bytes = input_path.read_bytes()

    BOOTSTRAP._patch_token_factory_image_mime(tmp_path)
    fixture_spec = importlib.util.spec_from_file_location(
        "embodiedgen_token_factory_mime_fixture", client
    )
    assert fixture_spec and fixture_spec.loader
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    sys.modules[fixture_spec.name] = fixture_module
    fixture_spec.loader.exec_module(fixture_module)
    content = fixture_module.Client().build(str(input_path))
    url = content[1]["image_url"]["url"]
    assert url.startswith(f"data:{expected_mime};base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == image_bytes


def test_urdf_property_response_patch_uses_only_completed_final_answer(
    tmp_path: Path,
) -> None:
    package = tmp_path / "embodied_gen"
    source = package / "validators"
    client_source = package / "utils"
    source.mkdir(parents=True)
    client_source.mkdir()
    converter = source / "urdf_convertor.py"
    client = client_source / "gpt_clients.py"
    client.write_text(
        "class _Message:\n"
        "    def __init__(self, content):\n"
        "        self.content = content\n\n"
        "class _Choice:\n"
        "    def __init__(self, content, finish_reason):\n"
        "        self.message = _Message(content)\n"
        "        self.finish_reason = finish_reason\n\n"
        "class _Completion:\n"
        "    def __init__(self, content, finish_reason):\n"
        "        self.choices = [_Choice(content, finish_reason)]\n\n"
        "class GPTclient:\n"
        "    def __init__(self, content, finish_reason):\n"
        "        self.content = content\n"
        "        self.finish_reason = finish_reason\n"
        "        self.last_payload = None\n\n"
        "    def completion_with_backoff(self, **payload):\n"
        "        self.last_payload = payload\n"
        "        return _Completion(self.content, self.finish_reason)\n\n"
        "    def query(self, text_prompt, image_base64=None, system_role=None, params=None):\n"
        "        if system_role is None:\n"
        "            system_role = 'physics'\n"
        "        payload = {'max_tokens': 500}\n"
        "        if params:\n"
        "            payload.update(params)\n"
        "        response = None\n"
        "        try:\n"
        "            response = self.completion_with_backoff(**payload)\n"
        "            response = response.choices[0].message.content\n"
        "        except Exception:\n"
        "            response = None\n"
        "        return response\n",
        encoding="utf-8",
    )
    converter.write_text(
        "from datetime import datetime\n"
        'VERSION = "test"\n\n'
        "class URDFGenerator:\n"
        "    def __init__(self, gpt_client):\n"
        "        self.gpt_client = gpt_client\n\n"
        "    def parse_response(self, response):\n"
        '        raw_lines = response.split("\\n")\n'
        "        lines = []\n"
        "        for line in raw_lines:\n"
        "            line = line.strip()\n"
        '            if line and not line.startswith("```") and ":" in line:\n'
        "                lines.append(line)\n\n"
        '        category = lines[0].split(": ")[1]\n'
        '        description = lines[1].split(": ")[1]\n'
        "        min_height, max_height = map(\n"
        '            lambda x: float(x.strip().replace(",", "").split()[0]),\n'
        '            lines[3].split(": ")[1].split("-"),\n'
        "        )\n"
        "        min_mass, max_mass = map(\n"
        '            lambda x: float(x.strip().replace(",", "").split()[0]),\n'
        '            lines[4].split(": ")[1].split("-"),\n'
        "        )\n"
        '        mu1 = float(lines[5].split(": ")[1].replace(",", ""))\n'
        '        mu2 = float(lines[6].split(": ")[1].replace(",", ""))\n\n'
        "        return {\n"
        '            "category": category.lower(),\n'
        '            "description": description.lower(),\n'
        '            "min_height": round(min_height, 4),\n'
        '            "max_height": round(max_height, 4),\n'
        '            "min_mass": round(min_mass, 4),\n'
        '            "max_mass": round(max_mass, 4),\n'
        '            "mu1": round(mu1, 2),\n'
        '            "mu2": round(mu2, 2),\n'
        '            "version": VERSION,\n'
        '            "generate_time": datetime.now().strftime("%Y%m%d%H%M%S"),\n'
        "        }\n\n"
        "    def resolve(self):\n"
        '        text_prompt = "prompt"\n'
        '        image_path = ["front.png"]\n'
        "        response = self.gpt_client.query(text_prompt, image_path)\n"
        "        if response is None:\n"
        "            asset_attrs = {\n"
        '                "category": "fallback",\n'
        '                "description": "fallback",\n'
        '                "min_height": 1,\n'
        '                "max_height": 1,\n'
        '                "min_mass": 1,\n'
        '                "max_mass": 1,\n'
        '                "mu1": 0.8,\n'
        '                "mu2": 0.6,\n'
        "            }\n"
        "        else:\n"
        "            asset_attrs = self.parse_response(response)\n"
        "        return asset_attrs\n",
        encoding="utf-8",
    )
    BOOTSTRAP._patch_urdf_property_response_contract(tmp_path)
    client_spec = importlib.util.spec_from_file_location(
        "embodiedgen_urdf_property_client_fixture", client
    )
    assert client_spec and client_spec.loader
    client_fixture = importlib.util.module_from_spec(client_spec)
    sys.modules[client_spec.name] = client_fixture
    client_spec.loader.exec_module(client_fixture)
    fixture_spec = importlib.util.spec_from_file_location(
        "embodiedgen_urdf_property_fixture", converter
    )
    assert fixture_spec and fixture_spec.loader
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)

    completed_response = """<think>
Pose: the object appears upright
Height: the object is standing
</think>
Category: cup
Description: a small ceramic cup
Pose: upright on its base
Height: 0.08-0.15 m
Weight: 0.05-0.10 kg
Static friction coefficient: 0.6
Dynamic friction coefficient: 0.5
"""
    completed_client = client_fixture.GPTclient(completed_response, "stop")
    attrs = fixture.URDFGenerator(completed_client).resolve()
    assert completed_client.last_completion_finish_reason == "stop"
    assert completed_client.last_payload == {"max_tokens": 2048}
    assert attrs | {"generate_time": "ignored"} == {
        "category": "cup",
        "description": "a small ceramic cup",
        "min_height": 0.08,
        "max_height": 0.15,
        "min_mass": 0.05,
        "max_mass": 0.1,
        "mu1": 0.6,
        "mu2": 0.5,
        "version": "test",
        "generate_time": "ignored",
    }
    parseable_prefix_client = client_fixture.GPTclient(
        """Category: cup
Description: a small ceramic cup
Pose: upright on its base
Height: 0.08-0.15 m
Weight: 0.05-0.10 kg
Static friction coefficient: 0.6
Dynamic friction coefficient: 0.5""",
        "length",
    )
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        fixture.URDFGenerator(parseable_prefix_client).resolve()
    assert parseable_prefix_client.last_payload == {"max_tokens": 2048}
    with pytest.raises(ValueError, match="incomplete reasoning block"):
        fixture.URDFGenerator(
            client_fixture.GPTclient("<think>unfinished", "stop")
        ).resolve()
    with pytest.raises(ValueError, match="required final fields"):
        fixture.URDFGenerator(
            client_fixture.GPTclient("Category: cup\nDescription: cup", "stop")
        ).resolve()
    with pytest.raises(RuntimeError, match="no final answer"):
        fixture.URDFGenerator(client_fixture.GPTclient(None, "stop")).resolve()


def test_asset_bundle_preserves_nested_generated_assets_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    collision = generated / "meshes" / "collision.obj"
    urdf = generated / "asset.urdf"
    collision.parent.mkdir(parents=True)
    urdf.write_text('<robot name="generated"/>', encoding="utf-8")
    collision.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf.chmod(0o600)

    first = tmp_path / "first.tar.gz"
    receipt = ASSET_BUNDLE.archive_asset_tree(generated, first)
    assert [member["path"] for member in receipt["members"]] == [
        "generated/asset.urdf",
        "generated/meshes/collision.obj",
    ]
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            "generated/asset.urdf",
            "generated/meshes/collision.obj",
        ]
        assert archive.getmember("generated/asset.urdf").mode == 0o644

    urdf.chmod(0o755)
    second = tmp_path / "second.tar.gz"
    ASSET_BUNDLE.archive_asset_tree(generated, second)
    assert first.read_bytes() == second.read_bytes()

    (generated / "unsafe-link").symlink_to(collision)
    with pytest.raises(RuntimeError, match="symbolic link"):
        ASSET_BUNDLE.archive_asset_tree(generated, tmp_path / "unsafe.tar.gz")
    assert not (tmp_path / "unsafe.tar.gz").exists()


def test_pybullet_validation_tracks_the_body_and_keeps_failure_media(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeBullet(ModuleType):
        DIRECT = 1
        ER_TINY_RENDERER = 2

        def __init__(self, *, settled: bool) -> None:
            super().__init__("pybullet")
            self.settled = settled
            self.steps = 0

        def connect(self, _: int) -> int:
            return 7

        def disconnect(self, _: int) -> None:
            return None

        def setAdditionalSearchPath(self, _: str) -> None:
            return None

        def loadURDF(self, _: str, **kwargs: object) -> int:
            return 2 if kwargs else 1

        def setGravity(self, *_: object) -> None:
            return None

        def stepSimulation(self) -> None:
            self.steps += 1

        def getAABB(self, _: int) -> tuple[list[float], list[float]]:
            z = max(0.05, 1.0 - self.steps / 1_200)
            return ([-0.06, -0.06, z - 0.05], [0.06, 0.06, z + 0.05])

        def computeProjectionMatrixFOV(self, *_: object) -> list[float]:
            return []

        def computeViewMatrix(self, *_: object) -> list[float]:
            return []

        def getCameraImage(
            self, *_: object, **__: object
        ) -> tuple[None, None, np.ndarray]:
            return None, None, np.zeros((480, 640, 4), dtype=np.uint8)

        def getBasePositionAndOrientation(
            self, _: int
        ) -> tuple[list[float], list[float]]:
            if self.steps == 0:
                return [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]
            return [0.02, 0.03, 0.05], [0.0, 0.0, 0.0, 1.0]

        def getBaseVelocity(self, _: int) -> tuple[list[float], list[float]]:
            speed = 0.0 if self.settled else 0.2
            return [speed, 0.0, 0.0], [0.0, speed, 0.0]

        def getContactPoints(self, **_: object) -> list[object]:
            return [object()]

    class FakeImageIo(ModuleType):
        def __init__(self) -> None:
            super().__init__("imageio.v3")
            self.videos: dict[Path, int] = {}

        def imwrite(self, path: Path, data: np.ndarray, **_: object) -> None:
            target = Path(path)
            target.write_bytes(b"decoded-test-media")
            if target.suffix == ".mp4":
                self.videos[target] = len(data)

        def imiter(self, path: Path):
            return iter(range(self.videos[Path(path)]))

    def load_smoke(*, settled: bool, suffix: str):
        fake_bullet = FakeBullet(settled=settled)
        fake_iio = FakeImageIo()
        imageio = ModuleType("imageio")
        imageio.v3 = fake_iio
        asset_bundle = ModuleType("asset_bundle")
        asset_bundle.archive_asset_tree = lambda *_: {}
        defusedxml = ModuleType("defusedxml")
        defusedxml.ElementTree = XML_ET
        pybullet_data = ModuleType("pybullet_data")
        pybullet_data.getDataPath = lambda: "/tmp"
        monkeypatch.setitem(sys.modules, "asset_bundle", asset_bundle)
        monkeypatch.setitem(sys.modules, "defusedxml", defusedxml)
        monkeypatch.setitem(sys.modules, "imageio", imageio)
        monkeypatch.setitem(sys.modules, "imageio.v3", fake_iio)
        monkeypatch.setitem(sys.modules, "pybullet", fake_bullet)
        monkeypatch.setitem(sys.modules, "pybullet_data", pybullet_data)
        monkeypatch.setitem(sys.modules, "trimesh", ModuleType("trimesh"))
        spec = importlib.util.spec_from_file_location(
            f"embodiedgen_capability_smoke_{suffix}", IMAGE / "capability_smoke.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, fake_bullet

    urdf = tmp_path / "generated.urdf"
    output = tmp_path / "success"
    output.mkdir()
    smoke, bullet = load_smoke(settled=True, suffix="success")
    result = smoke.pybullet_validation(urdf, output)
    assert (
        bullet.steps
        == smoke.PHYSICS_OBSERVATION_STEPS + smoke.PHYSICS_SETTLE_WINDOW_STEPS
    )
    assert result["observation_seconds"] == 40
    assert result["captured_frames"] == 160
    assert result["decoded_video_frames"] == 160
    assert result["camera"]["mode"] == "body_aabb_tracking"
    assert result["camera"]["first_capture"] != result["camera"]["last_capture"]
    assert (output / result["view_png"]).is_file()
    assert (output / result["view_mp4"]).is_file()

    failure_output = tmp_path / "failure"
    failure_output.mkdir()
    smoke, bullet = load_smoke(settled=False, suffix="failure")
    with pytest.raises(smoke.RigidBodySettleError):
        smoke.pybullet_validation(urdf, failure_output)
    failure = json.loads(
        (failure_output / "pybullet_validation_failure.json").read_text()
    )
    assert (
        bullet.steps
        == smoke.PHYSICS_OBSERVATION_STEPS + smoke.PHYSICS_SETTLE_WINDOW_STEPS
    )
    assert failure["status"] == "failed"
    assert failure["settle"]["angular_speed_rad_per_s"] == 0.2
    assert failure["captured_frames"] == failure["decoded_video_frames"] == 160
    assert (failure_output / failure["view_png"]).is_file()
    assert (failure_output / failure["view_mp4"]).is_file()


def test_capability_smoke_executes_the_real_generation_and_simulation_chain() -> None:
    tree = ast.parse((IMAGE / "capability_smoke.py").read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    def calls(name: str) -> set[str]:
        result = set()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    result.add(f"{node.func.value.id}.{node.func.attr}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                result.add(node.func.id)
        return result

    upstream_runs = [
        node
        for node in ast.walk(functions["run_upstream"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) == ("subprocess", "run")
    ]
    assert any(
        any(
            keyword.arg == "check" and keyword.value.value is True
            for keyword in run.keywords
        )
        for run in upstream_runs
    )
    assert "cvt_embodiedgen_asset_to_anysim" in calls("convert_to_mjcf")
    assert {"bullet.loadURDF", "bullet.setGravity"} <= calls("pybullet_validation")
    assert "bullet.stepSimulation" in calls("require_rigid_body_settle")
    assert {"iio.imwrite", "iio.imiter"} <= calls("_write_decoded_view")
    assert {
        "run_upstream",
        "collision_meshes",
        "convert_to_mjcf",
        "pybullet_validation",
    } <= calls("_generate_and_validate")
    assert {"_generate_and_validate", "atomic_json"} <= calls("main")


def test_prebuilt_image_and_standalone_profile_keep_the_byof_contract() -> None:
    dockerfile = (IMAGE / "Dockerfile").read_text(encoding="utf-8")
    profile = (
        ROOT
        / "npa/src/npa/workflows/byof/profiles"
        / "byof-solution-smoke-embodiedgen-rtxpro-gpu.yaml"
    ).read_text(encoding="utf-8")
    build = (IMAGE / "build.sh").read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "/opt/byof/npa_source_metadata.json" in dockerfile
    assert "git status --porcelain --untracked-files=all" in build
    assert "NPA_REGISTRY must name the operator-controlled private registry" in build
    assert "is_public_registry" in build
    assert "npa_byof_summary.json" in profile
    assert "client.head_object" in profile
    assert "allowPrivilegeEscalation: true" in profile
    assert "SETUID" in profile
    assert "allowPrivilegeEscalation: true" in workflow
    assert "SETUID" in workflow
    assert "npa-embodiedgen@sha256:" in workflow
    assert "setpriv --no-new-privs" in (IMAGE / "entrypoint.sh").read_text(
        encoding="utf-8"
    )


def test_build_script_rejects_docker_hub_shorthand_before_building() -> None:
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = subprocess.run(
        [
            "bash",
            str(IMAGE / "build.sh"),
            f"operator/npa-embodiedgen:dev-{source_sha}",
        ],
        cwd=ROOT,
        env={**os.environ, "NPA_REGISTRY": "operator", "NPA_SOURCE_SHA": source_sha},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "fully-qualified private registry host" in result.stderr


def test_token_factory_key_is_forwarded_only_to_embodiedgen() -> None:
    runner_path = ROOT / "npa/scripts/run_byof_container_verify.py"
    spec = importlib.util.spec_from_file_location(
        "embodiedgen_byof_runner", runner_path
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    assert runner.resolve_secret_envs(
        None,
        solution_name="embodiedgen",
        environment={"NEBIUS_TOKEN_FACTORY_KEY": "value"},
    ) == ["NEBIUS_TOKEN_FACTORY_KEY"]


def test_embodiedgen_byof_renderer_rejects_mutable_or_public_images() -> None:
    runner_path = ROOT / "npa/scripts/run_byof_container_verify.py"
    spec = importlib.util.spec_from_file_location("embodiedgen_byof_image", runner_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    digest = "a" * 64
    assert runner.validate_embodiedgen_image_reference(
        f"registry.example.invalid/npa-embodiedgen@sha256:{digest}"
    ).endswith(digest)
    with pytest.raises(ValueError, match="immutable"):
        runner.validate_embodiedgen_image_reference(
            "registry.example.invalid/npa-embodiedgen:latest"
        )
    with pytest.raises(ValueError, match="anonymous/public"):
        runner.validate_embodiedgen_image_reference(
            f"docker.io/npa-embodiedgen@sha256:{digest}"
        )
    with pytest.raises(ValueError, match="fully-qualified"):
        runner.validate_embodiedgen_image_reference(
            f"operator/npa-embodiedgen@sha256:{digest}"
        )


def test_native_workflow_rejects_nonprivate_or_mutable_embodiedgen_override() -> None:
    base_spec = load_spec(WORKFLOW)
    digest = "b" * 64
    validate_image_override_selectors(base_spec, SkypilotRenderOptions())
    mutable = replace(
        base_spec,
        config={**base_spec.config, "base_image": "private/npa-embodiedgen:tag"},
    )
    with pytest.raises(NpaWorkflowRenderError, match="immutable image"):
        validate_image_override_selectors(mutable, SkypilotRenderOptions())
    public = replace(
        base_spec,
        config={
            **base_spec.config,
            "base_image": f"docker.io/npa-embodiedgen@sha256:{digest}",
        },
    )
    with pytest.raises(NpaWorkflowRenderError, match="public registry"):
        validate_image_override_selectors(public, SkypilotRenderOptions())
    shorthand = replace(
        base_spec,
        config={
            **base_spec.config,
            "base_image": f"operator/npa-embodiedgen@sha256:{digest}",
        },
    )
    with pytest.raises(NpaWorkflowRenderError, match="fully-qualified"):
        validate_image_override_selectors(shorthand, SkypilotRenderOptions())
    private = f"registry.example.invalid/npa-embodiedgen@sha256:{digest}"
    accepted = replace(
        base_spec,
        config={**base_spec.config, "base_image": private},
    )
    validate_image_override_selectors(
        accepted,
        SkypilotRenderOptions(image_overrides={"workbench.byof.repo": private}),
    )
    with pytest.raises(NpaWorkflowRenderError, match="must name the same"):
        validate_image_override_selectors(
            accepted,
            SkypilotRenderOptions(
                image_overrides={
                    "workbench.byof.repo": (
                        "registry.example.invalid/npa-embodiedgen@sha256:" + "c" * 64
                    )
                }
            ),
        )


def test_input_uri_reaches_the_byof_shell_as_literal_data(tmp_path: Path) -> None:
    """Exercise plan → SkyPilot render → BYOF bash with hostile URI text."""

    marker = tmp_path / "uri-was-executed"
    capture = tmp_path / "received-uri"
    hostile = f'https://example.invalid/object.jpg?label="$(touch {marker})"`echo bad`'
    spec = load_spec(WORKFLOW)
    spec = replace(spec, config={**spec.config, "input_uri": hostile})
    plan = build_plan(spec, run_id="literal-uri")
    smoke_command = plan.steps[0].argv[plan.steps[0].argv.index("--smoke-command") + 1]
    encoded = base64.b64encode(hostile.encode("utf-8")).decode("ascii")

    assert hostile not in smoke_command
    assert encoded in smoke_command
    rendered = render_skypilot_yaml(spec, plan, run_id="literal-uri")
    assert hostile not in rendered
    assert encoded in rendered

    worker_command = smoke_command.replace(
        "exec /usr/local/bin/npa-embodiedgen-entrypoint run-smoke",
        'printf "%s" "$NPA_EMBODIEDGEN_INPUT_URI" > "$NPA_EMBODIEDGEN_CAPTURE_PATH"',
    )
    completed = subprocess.run(
        ["/bin/bash", "-lc", '/bin/bash -lc "$BYOF_SMOKE_COMMAND"'],
        env={
            **os.environ,
            "BYOF_SMOKE_COMMAND": worker_command,
            "NPA_EMBODIEDGEN_CAPTURE_PATH": str(capture),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert capture.read_text(encoding="utf-8") == hostile


def test_workflow_declares_worker_input_and_generated_asset_outputs() -> None:
    spec = load_spec(WORKFLOW)
    assert spec.metadata["executionMode"] == "runtime"
    assert spec_requires_runtime(spec)
    assert spec.resources["gpu"]["cpus"] == "16+"
    assert spec.resources["gpu"]["memory"] == "96+"
    state = spec.states["byof-run"]
    assert [(item.uri, item.schema) for item in state.inputs] == [
        ("{{config.input_uri}}", "image/*")
    ]
    assert {(item.uri, item.schema) for item in state.outputs} == {
        ("{{config.summary_uri}}", "npa.workbench.byof.summary.v1"),
        ("{{config.artifact_uri}}", "npa.embodiedgen.image-to-rigid-object.v1"),
        ("{{config.generated_asset_uri}}", "application/gzip"),
        ("{{config.mjcf_asset_uri}}", "application/gzip"),
        ("{{config.view_png_uri}}", "image/png"),
        ("{{config.view_mp4_uri}}", "video/mp4"),
    }


def test_embodiedgen_render_keeps_the_profiled_outer_worker_envelope() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="embodiedgen-resource-render")
    rendered = render_skypilot_yaml(spec, plan, run_id="embodiedgen-resource-render")
    tasks = list(yaml.safe_load_all(rendered))
    assert any(
        document.get("resources", {}).get("cpus") == "16+"
        and document["resources"].get("memory") == "96+"
        and document["resources"].get("accelerators")
        == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        for document in tasks
        if isinstance(document, dict)
    )


def test_readiness_record_binds_workflow_and_live_blockers() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    assert (
        f"sha256:{readiness['workflow_sha256']} workflows/testing/byof-embodiedgen.yaml"
        in readiness["planning"]["task_fidelity"]["evidence"]
    )
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["source_image"]["status"] == "blocked"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"


def test_embodiedgen_workflow_is_explicitly_plan_only_until_gpu_accepted() -> None:
    case = next(item for item in SUBMIT_LIVE_MATRIX if item.spec == WORKFLOW.name)
    assert case.tier == "gpu"
    assert case.plan_only
    assert case.requires_token_factory
    assert "private immutable" in case.plan_only_justification
