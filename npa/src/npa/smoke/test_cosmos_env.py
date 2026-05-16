"""Standalone NVIDIA Cosmos environment smoke checks.

Run with:
    python -m npa.smoke.test_cosmos_env
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from npa.smoke._versions import supported_tool_version

DEFAULT_MODEL = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
DEFAULT_MODEL_DIR = Path("/opt/cosmos/models")


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
        expected = supported_tool_version("cosmos", __file__)
        module = _import_cosmos_module()
        version = metadata.version("cosmos-predict2")
        if version != expected:
            return CheckResult(
                "import cosmos",
                False,
                f"expected version: {expected}; found: {version}",
            )
        return CheckResult(
            "import cosmos",
            True,
            f"module: {module.__name__}; version: {version}",
        )
    except Exception as exc:
        return CheckResult("import cosmos", False, _format_exception(exc))


def check_model_weights_dir() -> CheckResult:
    model = os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL)
    model_dir = Path(os.environ.get("COSMOS_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    try:
        if not model_dir.is_dir():
            return CheckResult("model weights directory exists", False, f"missing {model_dir}")
        model_path = model_dir / _model_slug(model)
        detail = f"directory: {model_dir}"
        if model_path.exists():
            detail += f"; model: {model_path}"
        return CheckResult("model weights directory exists", True, detail)
    except Exception as exc:
        return CheckResult("model weights directory exists", False, _format_exception(exc))


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_cosmos,
        check_model_weights_dir,
    ]
    results: list[CheckResult] = []

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
