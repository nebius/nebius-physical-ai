#!/usr/bin/env python3
"""Fetch and prepare the pinned EmbodiedGen TRELLIS runtime outside image layers."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_REVISION = "f0124197888c2b733e4eaa65acd81ad9cfda3b79"
TRELLIS_REVISION = "55a8e8164b195bbf927e0978f00e76c835e6011f"
MODEL_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
SOURCE_URL = "https://github.com/HorizonRobotics/EmbodiedGen.git"
TRELLIS_URL = "https://github.com/microsoft/TRELLIS.git"
RUNTIME_NAME = "embodiedgen-v2-trellis"
UPSTREAM_REQUIREMENTS_SHA256 = (
    "acd142fb1d6d87931a5b4c783110526acade67d641d01ab2660f1fece7af9157"
)
UPSTREAM_INSTALL_BASIC_SHA256 = (
    "2969700d580e8a0f18e377cddd3f3976317748c744ce0408ba6721d0a88f84d1"
)
PIP_REQUIREMENT = "pip==24.0"
CUDA_VARIANT = "cu128"
TORCH_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_VARIANT}"
XFORMERS_VERSION = "0.0.32.post2"
RUNTIME_CONTRACT = {
    "python": "3.12",
    "pip": PIP_REQUIREMENT,
    "cuda_variant": CUDA_VARIANT,
    "torch_index_url": TORCH_INDEX_URL,
    "torch_cuda_arch_list": "12.0",
    "tcnn_cuda_architectures": "120",
}
TRELLIS_ATTENTION_RUNTIME_CONTRACT = {
    "xformers_version": XFORMERS_VERSION,
    "fa3_enabled": False,
    "dense_backend": "xformers",
    "sparse_attention_backend": "xformers",
    "expected_dispatch_prefix": "fa2F",
    "cuda_capability": [12, 0],
}

# EmbodiedGen's pinned requirements.txt names these validation dependencies but
# leaves several of them unversioned. Pin the imports used by
# capability_smoke.py after the upstream installer so the generated-view and
# physics checks do not depend on a future resolver result. The constrained
# numerical packages keep NumPy 1.26 compatible with packages the upstream
# installer otherwise resolves to NumPy-2-only releases.
VALIDATION_REQUIREMENTS = (
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
BOOTSTRAP_REQUIREMENTS = (
    "boto3==1.35.99",
    "defusedxml==0.7.1",
)
VALIDATION_SYSTEM_PACKAGES = ("libxrender1",)
VALIDATION_RUNTIME_PROBE = """from io import BytesIO
from pathlib import Path
import tempfile

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
import contourpy
import plyfile
import pymeshfix
import pyvista as pv
import pybullet_data
import scipy
import spconv.pytorch
import tifffile
import torch
import trimesh
import torchvision
import vtk
import xformers
from PIL import Image

assert Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
assert pybullet_data.getDataPath()
assert contourpy.__version__
assert pv.__version__ == "0.36.1"
assert scipy.__version__ == "1.14.1"
assert tifffile.__version__ == "2024.8.30"
assert trimesh.__version__
assert vtk.vtkVersion.GetVTKVersion() == "9.3.1"
assert Image
assert torch.__version__.startswith("2.8.0")
assert torch.version.cuda == "12.8"
assert torchvision.__version__.startswith("0.23.0")
assert xformers.__version__ == "0.0.32.post2"
vertex = np.array([(0.0, 0.0, 0.0)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
ply_buffer = BytesIO()
plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")]).write(ply_buffer)
ply_buffer.seek(0)
assert plyfile.PlyData.read(ply_buffer)["vertex"].count == 1
surface = pv.Sphere(theta_resolution=12, phi_resolution=12)
assert surface.n_points > 0 and surface.n_cells > 0
faces = surface.faces.reshape(-1, 4)[:, 1:]
meshfix = pymeshfix.MeshFix(np.asarray(surface.points), faces)
meshfix.repair(verbose=False)
assert len(meshfix.v) > 0 and len(meshfix.f) > 0
with tempfile.TemporaryDirectory() as directory:
    video = Path(directory) / "validation.mp4"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    iio.imwrite(video, frame, fps=1)
    assert video.is_file() and video.stat().st_size > 0
    decoded = list(iio.imiter(video))
    assert len(decoded) == 1 and decoded[0].shape == frame.shape
"""

# This runs after the pinned runtime installs and before model bytes are fetched.
# It imports the patched upstream EmbodiedGen boundary, then executes the exact
# xFormers dense and BlockDiagonalMask paths TRELLIS selects on the RTX worker.
TRELLIS_ATTENTION_RUNTIME_PROBE = """import json
from contextlib import redirect_stdout
from io import StringIO

import torch
import torch.nn.functional as functional
import xformers
import xformers.ops as xops
from xformers.ops import fmha
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from embodied_gen.utils.monkey_patch.trellis import monkey_path_trellis


def sdpa_reference(query, key, value):
    return functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    ).transpose(1, 2)


if not torch.cuda.is_available():
    raise RuntimeError("TRELLIS attention probe requires a CUDA GPU")
if xformers.__version__ != "0.0.32.post2":
    raise RuntimeError("unexpected xFormers version for TRELLIS attention probe")

with redirect_stdout(StringIO()):
    monkey_path_trellis()
    from thirdparty.TRELLIS.trellis.modules import attention as dense_attention
    from thirdparty.TRELLIS.trellis.modules import sparse as sparse_attention

if fmha._get_use_fa3() is not False:
    raise RuntimeError("EmbodiedGen did not disable xFormers FA3 before TRELLIS imports")
if dense_attention.BACKEND != "xformers" or sparse_attention.ATTN != "xformers":
    raise RuntimeError("EmbodiedGen TRELLIS attention backend changed unexpectedly")
if list(torch.cuda.get_device_capability(0)) != [12, 0]:
    raise RuntimeError("TRELLIS attention probe requires RTX PRO 6000 capability 12.0")

torch.manual_seed(20261010)
device = torch.device("cuda")
dtype = torch.float16
q_lengths, kv_lengths = [11, 17], [13, 19]
cases = {
    "dense_self": (
        torch.randn((1, 32, 4, 64), device=device, dtype=dtype),
        torch.randn((1, 32, 4, 64), device=device, dtype=dtype),
        torch.randn((1, 32, 4, 64), device=device, dtype=dtype),
        None,
    ),
    "dense_cross": (
        torch.randn((1, 19, 4, 64), device=device, dtype=dtype),
        torch.randn((1, 27, 4, 64), device=device, dtype=dtype),
        torch.randn((1, 27, 4, 64), device=device, dtype=dtype),
        None,
    ),
    "block_diagonal": (
        torch.randn((1, sum(q_lengths), 4, 64), device=device, dtype=dtype),
        torch.randn((1, sum(kv_lengths), 4, 64), device=device, dtype=dtype),
        torch.randn((1, sum(kv_lengths), 4, 64), device=device, dtype=dtype),
        BlockDiagonalMask.from_seqlens(q_lengths, kv_lengths),
    ),
}
dispatches = {}
finite = {}
errors = {}
for name, (query, key, value, bias) in cases.items():
    dispatches[name] = fmha.dispatch._dispatch_fw(
        fmha.Inputs(query, key, value, attn_bias=bias), needs_gradient=False
    ).NAME
    actual = xops.memory_efficient_attention(query, key, value, attn_bias=bias)
    torch.cuda.synchronize()
    finite[name] = bool(torch.isfinite(actual).all().item())
    if bias is None:
        reference = sdpa_reference(query, key, value)
    else:
        pieces = []
        q_start = kv_start = 0
        for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
            pieces.append(
                sdpa_reference(
                    query[:, q_start : q_start + q_length],
                    key[:, kv_start : kv_start + kv_length],
                    value[:, kv_start : kv_start + kv_length],
                )
            )
            q_start += q_length
            kv_start += kv_length
        reference = torch.cat(pieces, dim=1)
    torch.cuda.synchronize()
    errors[name] = float((actual.float() - reference.float()).abs().max().item())
    if not finite[name] or errors[name] > 0.05:
        raise RuntimeError("TRELLIS xFormers attention did not match synchronized SDPA")
if any(not name.startswith("fa2F") for name in dispatches.values()):
    raise RuntimeError("TRELLIS xFormers did not dispatch the FA2 fallback")

print(
    json.dumps(
        {
            "status": "ok",
            "xformers_version": xformers.__version__,
            "fa3_enabled": fmha._get_use_fa3(),
            "dense_backend": dense_attention.BACKEND,
            "sparse_attention_backend": sparse_attention.ATTN,
            "capability": list(torch.cuda.get_device_capability(0)),
            "dispatches": dispatches,
            "finite": finite,
            "max_abs_error_vs_sdpa": errors,
        },
        sort_keys=True,
    )
)
"""


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["solution"]["revision"] != SOURCE_REVISION:
        raise ValueError("unexpected EmbodiedGen source revision")
    if payload["backend"]["revision"] != TRELLIS_REVISION:
        raise ValueError("unexpected TRELLIS source revision")
    if payload["backend"]["model_revision"] != MODEL_REVISION:
        raise ValueError("unexpected TRELLIS model revision")
    runtime = payload["runtime"]
    if runtime.get("upstream_requirements") != {
        "path": "requirements.txt",
        "sha256": UPSTREAM_REQUIREMENTS_SHA256,
    }:
        raise ValueError("unexpected EmbodiedGen requirements contract")
    if runtime.get("upstream_install_basic") != {
        "path": "install/install_basic.sh",
        "sha256": UPSTREAM_INSTALL_BASIC_SHA256,
    }:
        raise ValueError("unexpected EmbodiedGen installer contract")
    if runtime.get("operator_runtime") != RUNTIME_CONTRACT:
        raise ValueError("unexpected EmbodiedGen runtime contract")
    if runtime.get("validation_requirements") != list(VALIDATION_REQUIREMENTS):
        raise ValueError("unexpected EmbodiedGen validation dependency contract")
    if runtime.get("validation_system_packages") != list(VALIDATION_SYSTEM_PACKAGES):
        raise ValueError("unexpected EmbodiedGen validation system package contract")
    if runtime.get("trellis_attention_runtime") != TRELLIS_ATTENTION_RUNTIME_CONTRACT:
        raise ValueError("unexpected EmbodiedGen attention runtime contract")
    return payload


def _run(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_cache_root(raw: str) -> Path:
    root = Path(raw).expanduser().resolve()
    if root == Path("/") or root == Path("/workspace"):
        raise ValueError("runtime cache must be a solution-scoped directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _clone_exact(source: Path, url: str, revision: str) -> None:
    if not source.exists():
        _run(["git", "clone", "--no-checkout", "--filter=blob:none", url, str(source)])
    if not (source / ".git").exists():
        raise RuntimeError("runtime source path is not the EmbodiedGen cache clone")
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=source, text=True
    ).strip()
    if origin != url:
        raise RuntimeError("runtime source cache has an unexpected origin")
    _run(["git", "reset", "--hard"], cwd=source)
    _run(["git", "clean", "-ffdx"], cwd=source)
    _run(["git", "fetch", "--depth", "1", "origin", revision], cwd=source)
    _run(["git", "checkout", "--detach", revision], cwd=source)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if observed != revision:
        raise RuntimeError(f"source revision mismatch: {observed}")


def _verify_upstream_requirements(source: Path) -> None:
    requirements = source / "requirements.txt"
    if (
        not requirements.is_file()
        or _sha256(requirements) != UPSTREAM_REQUIREMENTS_SHA256
    ):
        raise RuntimeError(
            "EmbodiedGen requirements.txt does not match the pinned source"
        )


def _install_trellis(source: Path) -> Path:
    trellis = source / "thirdparty" / "TRELLIS"
    if not (trellis / ".git").exists():
        _run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "--recursive",
                "thirdparty/TRELLIS",
            ],
            cwd=source,
        )
    _clone_exact(trellis, TRELLIS_URL, TRELLIS_REVISION)
    gitlink = subprocess.check_output(
        ["git", "rev-parse", "HEAD:thirdparty/TRELLIS"], cwd=source, text=True
    ).strip()
    if gitlink != TRELLIS_REVISION:
        raise RuntimeError("EmbodiedGen TRELLIS gitlink mismatch")
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=trellis)
    return trellis


def _venv_environment(venv: Path, cache: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{venv / 'bin'}:{env['PATH']}"
    env["HF_HOME"] = str(cache / "huggingface")
    env["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub")
    env["PIP_CACHE_DIR"] = str(cache / "pip")
    env["PYTHONPATH"] = str(cache / RUNTIME_NAME / "source")
    env["NPA_EMBODIEDGEN_TRELLIS_MODEL_REVISION"] = MODEL_REVISION
    env["EMBODIEDGEN_CUDA_VARIANT"] = CUDA_VARIANT
    env["EMBODIEDGEN_TORCH_INDEX_URL"] = TORCH_INDEX_URL
    env["TORCH_CUDA_ARCH_LIST"] = RUNTIME_CONTRACT["torch_cuda_arch_list"]
    env["TCNN_CUDA_ARCHITECTURES"] = RUNTIME_CONTRACT["tcnn_cuda_architectures"]
    return env


def _verify_upstream_install_script(script: Path) -> None:
    if not script.is_file() or _sha256(script) != UPSTREAM_INSTALL_BASIC_SHA256:
        raise RuntimeError(
            "EmbodiedGen install_basic.sh does not match the pinned source"
        )


def _pin_install_script(source: Path) -> None:
    script = source / "install" / "install_basic.sh"
    _verify_upstream_install_script(script)
    text = script.read_text(encoding="utf-8")
    replacements = {
        "CUDA_VARIANT=$(detect_cuda_variant)": (
            'CUDA_VARIANT="${EMBODIEDGEN_CUDA_VARIANT:-$(detect_cuda_variant)}"'
        ),
        '"pip==22.3.1"': f'"{PIP_REQUIREMENT}"',
        'https://github.com/openai/CLIP.git"': 'https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"',
        'https://github.com/HochCC/Kolors.git"': 'https://github.com/HochCC/Kolors.git@c59c0aa67587e472de657bc9f4f9c18272c94165"',
        "https://github.com/autonomousvision/mip-splatting.git#": "https://github.com/autonomousvision/mip-splatting.git@dda02ab5ecf45d6edb8c540d9bb65c7e451345a9#",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"upstream install script changed: {old}")
        text = text.replace(old, new)
    script.write_text(text, encoding="utf-8")


def _patch_trellis_only_import(source: Path) -> None:
    path = source / "embodied_gen" / "utils" / "inference.py"
    text = path.read_text(encoding="utf-8")
    old = "from embodied_gen.models.sam3d import Sam3dInference\n"
    if old not in text:
        raise RuntimeError(
            "upstream TRELLIS-only compatibility patch no longer applies"
        )
    text = text.replace(old, "")
    text = text.replace(
        "TrellisImageTo3DPipeline | Sam3dInference", "TrellisImageTo3DPipeline"
    )
    start = "    elif isinstance(pipe, Sam3dInference):"
    if start not in text:
        raise RuntimeError("upstream SAM3D branch changed")
    before, after = text.split(start, 1)
    _, tail = after.split("    else:\n", 1)
    text = before + "    else:\n" + tail
    path.write_text(text, encoding="utf-8")
    image_to_3d = source / "embodied_gen" / "scripts" / "imageto3d.py"
    text = image_to_3d.read_text(encoding="utf-8")
    old = 'TrellisImageTo3DPipeline.from_pretrained(\n            "microsoft/TRELLIS-image-large"\n        )'
    new = 'TrellisImageTo3DPipeline.from_pretrained(\n            os.environ["NPA_EMBODIEDGEN_TRELLIS_MODEL_DIR"]\n        )'
    if old not in text:
        raise RuntimeError("upstream TRELLIS model call changed")
    image_to_3d.write_text(text.replace(old, new), encoding="utf-8")


def _patch_token_factory_image_mime(source: Path) -> None:
    """Make the pinned upstream OpenAI image data URL match its encoded bytes."""

    path = source / "embodied_gen" / "utils" / "gpt_clients.py"
    text = path.read_text(encoding="utf-8")
    replacements = {
        "            for img in image_base64:\n                if isinstance(img, Image.Image):\n": (
            "            for img in image_base64:\n"
            '                image_mime_type = "image/png"\n'
            "                if isinstance(img, Image.Image):\n"
        ),
        """                if isinstance(img, Image.Image):
                    buffer = BytesIO()
                    img.save(buffer, format=img.format or "PNG")
                    buffer.seek(0)
                    image_binary = buffer.read()
                    img = base64.b64encode(image_binary).decode("utf-8")
""": """                if isinstance(img, Image.Image):
                    image_format = (img.format or "PNG").upper()
                    image_mime_type = {
                        "BMP": "image/bmp",
                        "GIF": "image/gif",
                        "JPEG": "image/jpeg",
                        "JPG": "image/jpeg",
                        "PNG": "image/png",
                        "WEBP": "image/webp",
                    }.get(image_format, "image/png")
                    buffer = BytesIO()
                    img.save(buffer, format=image_format if image_mime_type != "image/png" else "PNG")
                    buffer.seek(0)
                    image_binary = buffer.read()
                    img = base64.b64encode(image_binary).decode("utf-8")
""",
        """                elif (
                    len(os.path.splitext(img)) > 1
                    and os.path.splitext(img)[-1].lower() in self.image_formats
                ):
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    with open(img, "rb") as f:
                        img = base64.b64encode(f.read()).decode("utf-8")
""": """                elif (
                    len(os.path.splitext(img)) > 1
                    and os.path.splitext(img)[-1].lower() in self.image_formats
                ):
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    with Image.open(img) as source_image:
                        image_format = (source_image.format or "").upper()
                    image_mime_type = {
                        "BMP": "image/bmp",
                        "GIF": "image/gif",
                        "JPEG": "image/jpeg",
                        "PNG": "image/png",
                        "WEBP": "image/webp",
                    }.get(image_format)
                    if image_mime_type is None:
                        raise ValueError(
                            f"Unsupported image format for Token Factory: {image_format}"
                        )
                    with open(img, "rb") as f:
                        img = base64.b64encode(f.read()).decode("utf-8")
""",
        '"image_url": {"url": f"data:image/png;base64,{img}"},': (
            '"image_url": {"url": f"data:{image_mime_type};base64,{img}"},'
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError("upstream Token Factory image adapter changed")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _patch_trellis_xformers_fa3(source: Path) -> None:
    """Disable xFormers FA3 before the pinned upstream imports TRELLIS.

    The selected xFormers wheel dispatches FA3 on capability 12.0 by default,
    but the Hopper FA3 launch rejects the real TRELLIS shapes on the RTX PRO
    6000.  Keep EmbodiedGen's existing xFormers dense and sparse backend choice
    and use its supported version-bound FA2 fallback instead.  This exact-text
    patch fails closed if the upstream monkey-patch import boundary changes.
    """

    path = source / "embodied_gen" / "utils" / "monkey_patch" / "trellis.py"
    text = path.read_text(encoding="utf-8")
    old = """    sys.path.append(os.path.join(current_dir, "../../.."))

    from thirdparty.TRELLIS.trellis.representations import Gaussian
"""
    new = """    sys.path.append(os.path.join(current_dir, "../../.."))

    import xformers
    from xformers.ops.fmha import _set_use_fa3

    if xformers.__version__ != "0.0.32.post2":
        raise RuntimeError("unexpected xFormers version for the TRELLIS FA3 guard")
    _set_use_fa3(False)

    from thirdparty.TRELLIS.trellis.representations import Gaussian
"""
    if old not in text:
        raise RuntimeError("upstream TRELLIS monkey-patch import boundary changed")
    path.write_text(text.replace(old, new), encoding="utf-8")


def _prepare(cache: Path) -> tuple[Path, Path]:
    runtime = cache / RUNTIME_NAME
    source, venv = runtime / "source", runtime / "venv"
    runtime.mkdir(parents=True, exist_ok=True)
    _clone_exact(source, SOURCE_URL, SOURCE_REVISION)
    _verify_upstream_requirements(source)
    _install_trellis(source)
    _patch_trellis_only_import(source)
    _patch_token_factory_image_mime(source)
    _patch_trellis_xformers_fa3(source)
    _pin_install_script(source)
    if not (venv / "bin" / "python").is_file():
        _run([sys.executable, "-m", "venv", str(venv)])
    return source, venv


def _repair_pip(venv: Path, cache: Path) -> None:
    """Recover pip before reusing an incomplete runtime venv."""

    python = str(venv / "bin" / "python")
    env = _venv_environment(venv, cache)
    _run([python, "-m", "ensurepip", "--upgrade"], env=env)
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            PIP_REQUIREMENT,
            "setuptools==80.10.2",
            "wheel",
        ],
        env=env,
    )


def _install(source: Path, venv: Path, cache: Path) -> None:
    marker = venv / ".npa-embodiedgen-installed.json"
    if marker.is_file():
        _verify_install_receipt(venv, marker, cache)
        return
    env = _venv_environment(venv, cache)
    pip = str(venv / "bin" / "python")
    _repair_pip(venv, cache)
    _run(["bash", "install/install_basic.sh"], cwd=source, env=env)
    _run(
        [
            pip,
            "-m",
            "pip",
            "install",
            *VALIDATION_REQUIREMENTS,
            *BOOTSTRAP_REQUIREMENTS,
        ],
        env=env,
    )
    _verify_validation_runtime(venv, cache)
    _write_install_receipt(venv, marker, cache)


def _verify_validation_runtime(venv: Path, cache: Path) -> None:
    """Run the same import and MP4 encode/decode boundary as the smoke."""

    _run(
        [str(venv / "bin" / "python"), "-c", VALIDATION_RUNTIME_PROBE],
        env=_venv_environment(venv, cache),
    )


def _verify_trellis_attention_runtime(source: Path, venv: Path, cache: Path) -> Path:
    """Prove the patched FA2 dispatch before fetching model bytes."""

    completed = subprocess.run(
        [str(venv / "bin" / "python"), "-c", TRELLIS_ATTENTION_RUNTIME_PROBE],
        cwd=source,
        env=_venv_environment(venv, cache),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "TRELLIS attention probe did not emit one JSON result"
        ) from error
    expected = TRELLIS_ATTENTION_RUNTIME_CONTRACT
    dispatches = payload.get("dispatches")
    finite = payload.get("finite")
    errors = payload.get("max_abs_error_vs_sdpa")
    if not (
        payload.get("status") == "ok"
        and payload.get("xformers_version") == expected["xformers_version"]
        and payload.get("fa3_enabled") is expected["fa3_enabled"]
        and payload.get("dense_backend") == expected["dense_backend"]
        and payload.get("sparse_attention_backend")
        == expected["sparse_attention_backend"]
        and payload.get("capability") == expected["cuda_capability"]
        and isinstance(dispatches, dict)
        and set(dispatches) == {"dense_self", "dense_cross", "block_diagonal"}
        and all(
            isinstance(value, str)
            and value.startswith(expected["expected_dispatch_prefix"])
            for value in dispatches.values()
        )
        and isinstance(finite, dict)
        and set(finite) == set(dispatches)
        and all(value is True for value in finite.values())
        and isinstance(errors, dict)
        and set(errors) == set(dispatches)
        and all(
            isinstance(value, (int, float)) and value <= 0.05
            for value in errors.values()
        )
    ):
        raise RuntimeError(
            "TRELLIS attention probe did not satisfy its runtime contract"
        )
    result = cache / RUNTIME_NAME / "trellis-attention-probe.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    temporary = result.with_name(f".{result.name}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(result)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _venv_file_records(venv: Path) -> list[dict[str, str]]:
    records = []
    for root in (venv / "bin", venv / "lib"):
        for path in sorted(root.rglob("*")) if root.is_dir() else ():
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                records.append(
                    {"path": str(path.relative_to(venv)), "sha256": _sha256(path)}
                )
    return records


def _installed_freeze(venv: Path, cache: Path) -> list[str]:
    return subprocess.check_output(
        [str(venv / "bin" / "python"), "-m", "pip", "freeze", "--all"],
        env=_venv_environment(venv, cache),
        text=True,
    ).splitlines()


def _write_install_receipt(venv: Path, marker: Path, cache: Path) -> None:
    marker.write_text(
        json.dumps(
            {
                "schema": "npa.embodiedgen.venv-receipt.v1",
                "pip_freeze": _installed_freeze(venv, cache),
                "files": _venv_file_records(venv),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _verify_install_receipt(venv: Path, marker: Path, cache: Path) -> None:
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    expected = receipt.get("files")
    if receipt.get("schema") != "npa.embodiedgen.venv-receipt.v1":
        raise RuntimeError("EmbodiedGen venv receipt has an unexpected schema")
    if not isinstance(expected, list) or expected != _venv_file_records(venv):
        raise RuntimeError("EmbodiedGen venv bytes do not match their receipt")
    if receipt.get("pip_freeze") != _installed_freeze(venv, cache):
        raise RuntimeError("EmbodiedGen venv packages do not match their receipt")


def _download_model(venv: Path, cache: Path) -> Path:
    model = cache / RUNTIME_NAME / "models" / "TRELLIS-image-large"
    marker = model / ".npa-model-receipt.json"
    if marker.is_file():
        _verify_model_receipt(model, marker)
        return model
    if model.exists():
        raise RuntimeError(
            "TRELLIS model cache is incomplete; use a new scoped cache path"
        )
    temporary = model.with_name(f".{model.name}.download-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    script = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('microsoft/TRELLIS-image-large', "
        f"revision='{MODEL_REVISION}', local_dir={str(temporary)!r})"
    )
    env = _venv_environment(venv, cache)
    try:
        _run([str(venv / "bin" / "python"), "-c", script], env=env)
        _write_model_receipt(temporary)
        _verify_model_receipt(temporary, temporary / marker.name)
        temporary.rename(model)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return model


def _write_model_receipt(model: Path) -> None:
    files = []
    for path in sorted(model.rglob("*")):
        if path.is_file() and path.name != ".npa-model-receipt.json":
            files.append(
                {"path": str(path.relative_to(model)), "sha256": _sha256(path)}
            )
    receipt = {"revision": MODEL_REVISION, "files": files}
    (model / ".npa-model-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_model_receipt(model: Path, marker: Path) -> None:
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    files = receipt.get("files")
    if receipt.get("revision") != MODEL_REVISION or not isinstance(files, list):
        raise RuntimeError("TRELLIS model cache receipt is invalid")
    expected: dict[str, str] = {}
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeError("TRELLIS model cache receipt contains an invalid file")
        relative = str(record.get("path") or "")
        digest = str(record.get("sha256") or "")
        candidate = PurePosixPath(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise RuntimeError("TRELLIS model cache receipt contains an unsafe file")
        expected[relative] = digest
    if len(expected) != len(files):
        raise RuntimeError("TRELLIS model cache receipt duplicates a file")
    actual = {
        str(path.relative_to(model)): _sha256(path)
        for path in model.rglob("*")
        if path.is_file() and path.name != marker.name
    }
    if not expected or expected != actual:
        raise RuntimeError("TRELLIS model cache bytes do not match their receipt")


def _runtime_receipt(
    source: Path, venv: Path, cache: Path, attention_probe: Path
) -> Path:
    trellis = source / "thirdparty" / "TRELLIS"
    receipt = {
        "schema": "npa.embodiedgen.runtime-receipt.v1",
        "source_revision": SOURCE_REVISION,
        "source_requirements_sha256": _sha256(source / "requirements.txt"),
        "source_install_basic_sha256": UPSTREAM_INSTALL_BASIC_SHA256,
        "patched_install_basic_sha256": _sha256(
            source / "install" / "install_basic.sh"
        ),
        "trellis_revision": TRELLIS_REVISION,
        "trellis_model_revision": MODEL_REVISION,
        "operator_runtime": RUNTIME_CONTRACT,
        "trellis_attention_runtime": TRELLIS_ATTENTION_RUNTIME_CONTRACT,
        "validation_requirements": list(VALIDATION_REQUIREMENTS),
        "source_path": str(source),
        "venv_path": str(venv),
        "install_marker_sha256": _sha256(venv / ".npa-embodiedgen-installed.json"),
        "model_receipt_sha256": _sha256(
            cache
            / RUNTIME_NAME
            / "models"
            / "TRELLIS-image-large"
            / ".npa-model-receipt.json"
        ),
        "trellis_attention_probe_sha256": _sha256(attention_probe),
        "trellis_submodules": subprocess.check_output(
            ["git", "submodule", "status", "--recursive"], cwd=trellis, text=True
        ).splitlines(),
    }
    path = cache / RUNTIME_NAME / "runtime-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def _token_factory_environment(env: dict[str, str]) -> dict[str, str]:
    key = env.get("NEBIUS_TOKEN_FACTORY_KEY") or env.get("NPA_TOKEN_FACTORY_API_KEY")
    if not key:
        raise RuntimeError(
            "Token Factory credential is required for EmbodiedGen URDF estimates"
        )
    env["GPT_PROVIDER"] = "openai"
    env["ENDPOINT"] = "https://api.tokenfactory.nebius.com/v1/"
    env["API_KEY"] = key
    env["MODEL_NAME"] = "openbmb/MiniCPM-V-4_5"
    return env


def command_health(args: argparse.Namespace) -> int:
    payload = _read_manifest(args.manifest)
    print(json.dumps({"status": "ready", "capability": payload["capability"]}))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    _read_manifest(args.manifest)
    if not (
        os.environ.get("NEBIUS_TOKEN_FACTORY_KEY")
        or os.environ.get("NPA_TOKEN_FACTORY_API_KEY")
    ):
        raise RuntimeError(
            "Token Factory credential is required for EmbodiedGen URDF estimates"
        )
    cache = _safe_cache_root(args.cache_root)
    lock = cache / ".embodiedgen.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        source, venv = _prepare(cache)
        _install(source, venv, cache)
        attention_probe = _verify_trellis_attention_runtime(source, venv, cache)
        model = _download_model(venv, cache)
        receipt = _runtime_receipt(source, venv, cache, attention_probe)
    env = _token_factory_environment(_venv_environment(venv, cache))
    env["NPA_EMBODIEDGEN_RUNTIME_RECEIPT"] = str(receipt)
    env["NPA_EMBODIEDGEN_SOURCE_ROOT"] = str(source)
    env["NPA_EMBODIEDGEN_TRELLIS_MODEL_DIR"] = str(model)
    _run([str(venv / "bin" / "python"), args.smoke], env=env)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    health = sub.add_parser("health")
    health.add_argument("--manifest", type=Path, required=True)
    smoke = sub.add_parser("run-smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--smoke", type=Path, required=True)
    smoke.add_argument("--cache-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return command_health(args) if args.command == "health" else command_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
