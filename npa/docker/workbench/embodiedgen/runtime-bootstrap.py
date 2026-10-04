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


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["solution"]["revision"] != SOURCE_REVISION:
        raise ValueError("unexpected EmbodiedGen source revision")
    if payload["backend"]["revision"] != TRELLIS_REVISION:
        raise ValueError("unexpected TRELLIS source revision")
    if payload["backend"]["model_revision"] != MODEL_REVISION:
        raise ValueError("unexpected TRELLIS model revision")
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
    return env


def _pin_install_script(source: Path) -> None:
    script = source / "install" / "install_basic.sh"
    text = script.read_text(encoding="utf-8")
    replacements = {
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


def _prepare(cache: Path) -> tuple[Path, Path]:
    runtime = cache / RUNTIME_NAME
    source, venv = runtime / "source", runtime / "venv"
    runtime.mkdir(parents=True, exist_ok=True)
    _clone_exact(source, SOURCE_URL, SOURCE_REVISION)
    _install_trellis(source)
    _patch_trellis_only_import(source)
    _pin_install_script(source)
    if not (venv / "bin" / "python").is_file():
        _run([sys.executable, "-m", "venv", str(venv)])
    return source, venv


def _install(source: Path, venv: Path, cache: Path) -> None:
    marker = venv / ".npa-embodiedgen-installed.json"
    if marker.is_file():
        return
    env = _venv_environment(venv, cache)
    pip = str(venv / "bin" / "python")
    _run(
        [pip, "-m", "pip", "install", "pip==22.3.1", "setuptools==80.10.2", "wheel"],
        env=env,
    )
    _run(["bash", "install/install_basic.sh"], cwd=source, env=env)
    _run([pip, "-m", "pip", "install", "pybullet==3.2.7", "boto3==1.35.99"], env=env)
    freeze = subprocess.check_output(
        [pip, "-m", "pip", "freeze", "--all"], env=env, text=True
    )
    marker.write_text(json.dumps({"pip_freeze": freeze.splitlines()}, indent=2) + "\n")


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


def _runtime_receipt(source: Path, venv: Path, cache: Path) -> Path:
    trellis = source / "thirdparty" / "TRELLIS"
    receipt = {
        "schema": "npa.embodiedgen.runtime-receipt.v1",
        "source_revision": SOURCE_REVISION,
        "trellis_revision": TRELLIS_REVISION,
        "trellis_model_revision": MODEL_REVISION,
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
    cache = _safe_cache_root(args.cache_root)
    lock = cache / ".embodiedgen.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        source, venv = _prepare(cache)
        _install(source, venv, cache)
        model = _download_model(venv, cache)
        receipt = _runtime_receipt(source, venv, cache)
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
