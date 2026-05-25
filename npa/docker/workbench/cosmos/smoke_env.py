"""Cosmos container environment smoke checks."""

from __future__ import annotations

import importlib
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable


EXPECTED_COSMOS_VERSION = os.environ.get("COSMOS_VERSION", "1.0.9")
DEFAULT_MODEL = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
DEFAULT_MODEL_DIR = Path("/opt/cosmos-data/models")
SERVER_PATH = Path("/opt/cosmos/server.py")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _model_slug(model: str) -> str:
    return model.replace("/", "--").replace(":", "--")


def _import_cosmos_module() -> Any:
    for module_name in ("cosmos_predict2", "cosmos_predict1", "cosmos1", "cosmos"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
    raise ModuleNotFoundError("cosmos_predict2, cosmos_predict1, cosmos1, or cosmos")


def check_import_cosmos() -> CheckResult:
    try:
        module = _import_cosmos_module()
        version = metadata.version("cosmos-predict2")
        if version != EXPECTED_COSMOS_VERSION:
            return CheckResult(
                "import cosmos",
                False,
                f"expected version: {EXPECTED_COSMOS_VERSION}; found: {version}",
            )
        return CheckResult("import cosmos", True, f"module: {module.__name__}; version: {version}")
    except Exception as exc:
        return CheckResult("import cosmos", False, _format_exception(exc))


def check_cuda_gpu() -> CheckResult:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return CheckResult("check CUDA/GPU access", False, "torch.cuda.is_available() is false")
        before = torch.cuda.memory_allocated(0)
        tensor = torch.ones((512, 512), device="cuda")
        allocated = torch.cuda.memory_allocated(0) - before
        value = float(tensor.sum().item())
        del tensor
        torch.cuda.empty_cache()
        return CheckResult(
            "check CUDA/GPU access",
            True,
            f"devices: {torch.cuda.device_count()}; gpu0: {torch.cuda.get_device_name(0)}; "
            f"allocated_delta_bytes: {allocated}; sum: {value}",
        )
    except Exception as exc:
        return CheckResult("check CUDA/GPU access", False, _format_exception(exc))


def check_core_dependencies() -> CheckResult:
    packages = {
        "diffusers": "0.38.0",
        "peft": "0.19.1",
        "transformers": "4.51.3",
        "accelerate": "1.13.0",
        "flash-attn": "2.6.3",
        "natten": "0.21.0",
    }
    try:
        versions = []
        for package, expected in packages.items():
            version = metadata.version(package)
            if version != expected:
                return CheckResult(
                    "import core Cosmos dependencies",
                    False,
                    f"{package}: expected {expected}; found {version}",
                )
            versions.append(f"{package}=={version}")
        diffusers = importlib.import_module("diffusers")
        pipeline = getattr(diffusers, "CosmosTextToWorldPipeline", None)
        pipeline_name = getattr(pipeline, "__name__", "DiffusionPipeline fallback")
        return CheckResult(
            "import core Cosmos dependencies",
            True,
            "; ".join(versions) + f"; pipeline: {pipeline_name}",
        )
    except Exception as exc:
        return CheckResult("import core Cosmos dependencies", False, _format_exception(exc))


def check_server_script() -> CheckResult:
    try:
        py_compile.compile(str(SERVER_PATH), doraise=True)
        spec = importlib.util.spec_from_file_location("npa_cosmos_smoke_server", SERVER_PATH)
        if spec is None or spec.loader is None:
            return CheckResult("load Cosmos server script", False, f"unable to load spec for {SERVER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        app = getattr(module, "app", None)
        if app is None:
            return CheckResult("load Cosmos server script", False, "server module has no app")
        return CheckResult("load Cosmos server script", True, f"server: {SERVER_PATH}; app: {type(app).__name__}")
    except Exception as exc:
        return CheckResult("load Cosmos server script", False, _format_exception(exc))


def check_model_download_tooling() -> CheckResult:
    path = shutil.which("huggingface-cli")
    if path is None:
        return CheckResult("check model download tooling", False, "huggingface-cli not found on PATH")
    try:
        result = subprocess.run(
            [path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return CheckResult("check model download tooling", False, _format_exception(exc))
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        return CheckResult(
            "check model download tooling",
            False,
            f"exit code {result.returncode}; output: {output[:500]}",
        )
    return CheckResult("check model download tooling", True, f"executable: {path}")


def check_model_weights_mount() -> CheckResult:
    model = os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL)
    model_dir = Path(os.environ.get("COSMOS_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    try:
        if not model_dir.is_dir():
            return CheckResult("model weights directory is mounted", False, f"missing {model_dir}")
        model_path = model_dir / _model_slug(model)
        if not model_path.is_dir():
            return CheckResult("model weights directory is mounted", False, f"missing model {model_path}")
        return CheckResult("model weights directory is mounted", True, str(model_path))
    except Exception as exc:
        return CheckResult("model weights directory is mounted", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_cosmos,
        check_cuda_gpu,
        check_core_dependencies,
        check_server_script,
        check_model_download_tooling,
        check_model_weights_mount,
    ]
    results = []
    for check in checks:
        result = check()
        results.append(result)
        _print_result(result)

    passed = sum(result.ok for result in results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
