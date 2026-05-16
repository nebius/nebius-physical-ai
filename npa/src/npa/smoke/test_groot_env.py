"""Standalone NVIDIA GR00T environment smoke checks.

Run with:
    python -m npa.smoke.test_groot_env
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable

from npa.smoke._versions import supported_tool_version

DEFAULT_GROOT_REPO = Path("/opt/groot/Isaac-GR00T")
DEFAULT_MODEL_DIR = Path("/opt/groot/models")
DEFAULT_MODEL = "nvidia/GR00T-N1.7-3B"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _format_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _model_slug(model: str) -> str:
    return model.removeprefix("ngc://").replace("/", "--").replace(":", "--")


def check_import_gr00t() -> CheckResult:
    try:
        release = supported_tool_version("groot", __file__)
        module = importlib.import_module("gr00t")
        version = metadata.version("gr00t")
        return CheckResult(
            "import gr00t",
            True,
            f"module: {module.__name__}; package_version: {version}; npa_release: {release}",
        )
    except Exception as exc:
        return CheckResult("import gr00t", False, _format_exception(exc))


def check_embodiment_tags() -> CheckResult:
    try:
        tags = importlib.import_module("gr00t.data.embodiment_tags")
        embodiment_tag = tags.EmbodimentTag.resolve("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        custom_tag = tags.EmbodimentTag.resolve("NEW_EMBODIMENT")
        return CheckResult(
            "resolve embodiment tags",
            True,
            f"base: {embodiment_tag.value}; custom: {custom_tag.value}",
        )
    except Exception as exc:
        return CheckResult("resolve embodiment tags", False, _format_exception(exc))


def check_repo_layout() -> CheckResult:
    repo = Path(os.environ.get("GROOT_REPO", str(DEFAULT_GROOT_REPO)))
    script = repo / "scripts" / "deployment" / "standalone_inference_script.py"
    finetune = repo / "gr00t" / "experiment" / "launch_finetune.py"
    missing = [str(path) for path in (script, finetune) if not path.exists()]
    if missing:
        return CheckResult("gr00t repo layout", False, "missing: " + ", ".join(missing))
    return CheckResult("gr00t repo layout", True, f"repo: {repo}")


def check_model_weights_dir() -> CheckResult:
    model = os.environ.get("GROOT_MODEL_PATH", DEFAULT_MODEL)
    model_dir = Path(os.environ.get("GROOT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    try:
        if not model_dir.is_dir():
            return CheckResult("model weights directory exists", False, f"missing {model_dir}")
        candidate = model_dir / _model_slug(model)
        detail = f"directory: {model_dir}"
        if candidate.exists():
            detail += f"; model: {candidate}"
        else:
            detail += f"; model cache not found for {model}"
        return CheckResult("model weights directory exists", True, detail)
    except Exception as exc:
        return CheckResult("model weights directory exists", False, _format_exception(exc))


def check_ngc_credentials() -> CheckResult:
    require_ngc = os.environ.get("GROOT_REQUIRE_NGC", "").strip().lower() in {"1", "true", "yes"}
    cfg = Path.home() / ".ngc" / "config"
    configured = bool(os.environ.get("NGC_API_KEY")) or (
        cfg.exists() and "apikey" in cfg.read_text(errors="ignore")
    )
    if require_ngc and not configured:
        return CheckResult("ngc credentials configured", False, "NGC_API_KEY or ~/.ngc/config missing")
    return CheckResult("ngc credentials configured", True, f"configured: {configured}")


def _print_result(result: CheckResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status}: {result.name}")
    if result.detail:
        print(f"  {result.detail}")


def main() -> int:
    checks: list[Callable[[], CheckResult]] = [
        check_import_gr00t,
        check_embodiment_tags,
        check_repo_layout,
        check_model_weights_dir,
        check_ngc_credentials,
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
