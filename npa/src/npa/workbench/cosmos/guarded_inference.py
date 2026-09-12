"""Fail-closed guardrail instrumentation for native Cosmos 3 inference.

The pinned framework enables prompt safety models, but its generated-media
runner is empty and treats that state as safe.  This entry point runs inside the
framework interpreter, restores the shipped ``VideoContentSafetyFilter`` when
upstream discovers no media model, and records what was actually discovered and
evaluated.  The host-side runner accepts a guarded result only when this receipt
proves both prompt and generated-media safety execution.

No weights are bundled by this module.  All model construction and downloads
still happen at runtime through the operator's framework environment.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

GUARDRAIL_STATE_SCHEMA = "npa.cosmos3.guardrail-state.v1"
GUARDRAIL_RECEIPT_ENV = "NPA_COSMOS3_GUARDRAIL_RECEIPT"
GUARDRAILS_REQUESTED_ENV = "NPA_COSMOS3_GUARDRAILS_REQUESTED"
INFERENCE_MODULE_ENV = "NPA_COSMOS3_INFERENCE_MODULE"
DEFAULT_INFERENCE_MODULE = "cosmos_framework.scripts.inference"
ALLOWED_INFERENCE_MODULES = frozenset(
    {DEFAULT_INFERENCE_MODULE, "npa.workbench.cosmos.structural_transfer_runner"}
)
_ROLES = ("prompt_input", "generated_media")


def _requested() -> bool:
    return os.environ.get(GUARDRAILS_REQUESTED_ENV, "") == "1"


def _new_state(requested: bool | None = None) -> dict[str, Any]:
    enabled = _requested() if requested is None else bool(requested)
    return {
        "schema": GUARDRAIL_STATE_SCHEMA,
        "requested": enabled,
        "discovered": {role: [] for role in _ROLES},
        "evaluated": {role: [] for role in _ROLES},
        "postprocessors": {"discovered": [], "evaluated": []},
        "effective": False,
        "status": "pending" if enabled else "explicit_opt_out",
        "failure": "",
    }


_STATE = _new_state()


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _model_names(models: Any) -> list[str]:
    return [model.__class__.__name__ for model in list(models or [])]


def _receipt_path() -> Path | None:
    value = os.environ.get(GUARDRAIL_RECEIPT_ENV, "").strip()
    return Path(value) if value else None


def _write_state() -> None:
    # Native tensor-parallel workers share the output directory. Only rank zero
    # runs the publishing path, so only rank zero may author its receipt.
    if os.environ.get("RANK", "0") != "0":
        return
    path = _receipt_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(_STATE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _log_state() -> None:
    if os.environ.get("RANK", "0") == "0":
        print(
            "NPA_COSMOS3_GUARDRAIL_STATE "
            + json.dumps(_STATE, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


def _role_for(value: Any) -> str:
    return "prompt_input" if isinstance(value, str) else "generated_media"


def _fail(role: str, category: str) -> None:
    _STATE["status"] = "ineffective"
    _STATE["failure"] = f"{role}:{category}"
    _STATE["effective"] = False
    _write_state()


def _guarded_safety_check(runner: Any, value: Any) -> tuple[bool, str]:
    """Evaluate every discovered model and reject empty or invalid runners."""

    role = _role_for(value)
    models = list(runner.safety_models or [])
    names = _model_names(models)
    _append_unique(_STATE["discovered"][role], names)
    if not models:
        _fail(role, "no_safety_models")
        raise RuntimeError(
            f"{role} guardrails requested but no safety model was discovered"
        )

    for model, name in zip(models, names, strict=True):
        try:
            outcome = model.is_safe(value)
        except Exception:
            _fail(role, f"{name}:evaluation_error")
            raise
        if not isinstance(outcome, tuple) or len(outcome) != 2:
            _fail(role, f"{name}:invalid_result")
            raise RuntimeError(f"{name} returned an invalid guardrail result")
        safe, message = outcome
        if not isinstance(safe, bool):
            _fail(role, f"{name}:invalid_result")
            raise RuntimeError(f"{name} returned a non-boolean safety decision")
        _append_unique(_STATE["evaluated"][role], [name])
        _write_state()
        if not safe:
            _STATE["status"] = "blocked"
            _STATE["failure"] = f"{role}:{name}:blocked"
            _write_state()
            block = runner.generic_block_msg or f"{name.upper()}: {message}"
            return False, block
    return True, runner.generic_safe_msg


def _guarded_postprocess(runner: Any, frames: Any) -> Any:
    models = list(runner.postprocessors or [])
    names = _model_names(models)
    _append_unique(_STATE["postprocessors"]["discovered"], names)
    for model, name in zip(models, names, strict=True):
        try:
            frames = model.postprocess(frames)
        except Exception:
            _fail("generated_media", f"{name}:postprocess_error")
            raise
        _append_unique(_STATE["postprocessors"]["evaluated"], [name])
        _write_state()
    return frames


def _media_safety_model_class() -> Any:
    from cosmos_framework.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
        VideoContentSafetyFilter,
    )

    return VideoContentSafetyFilter


def _restore_media_safety(original_factory: Any) -> Any:
    def create_video_guardrail_runner(*, offload_model_to_cpu: bool = False) -> Any:
        runner = original_factory(offload_model_to_cpu=offload_model_to_cpu)
        if runner.safety_models:
            return runner
        runner.safety_models = [
            _media_safety_model_class()(offload_model_to_cpu=offload_model_to_cpu)
        ]
        return runner

    return create_video_guardrail_runner


def _install_fail_closed_guardrails() -> None:
    from cosmos_framework.auxiliary.guardrail.common import presets
    from cosmos_framework.auxiliary.guardrail.common.core import GuardrailRunner

    presets.create_video_guardrail_runner = _restore_media_safety(
        presets.create_video_guardrail_runner
    )
    GuardrailRunner.run_safety_check = _guarded_safety_check
    GuardrailRunner.postprocess = _guarded_postprocess


def _finalize_state() -> None:
    if not _STATE["requested"]:
        _write_state()
        return
    complete = all(
        _STATE["discovered"][role]
        and _STATE["evaluated"][role] == _STATE["discovered"][role]
        for role in _ROLES
    )
    if complete and _STATE["status"] == "pending":
        _STATE["effective"] = True
        _STATE["status"] = "passed"
    elif _STATE["status"] == "pending":
        _fail("guardrail_suite", "incomplete_evaluation")
    _write_state()
    _log_state()


def main() -> None:
    """Run one allowlisted native entry point with fail-closed guardrails."""

    module_name = os.environ.get(INFERENCE_MODULE_ENV, DEFAULT_INFERENCE_MODULE)
    if module_name not in ALLOWED_INFERENCE_MODULES:
        raise RuntimeError(f"unsupported Cosmos 3 inference module: {module_name}")
    # Importing the native script performs its required process initialization.
    module = importlib.import_module(module_name)
    try:
        if _STATE["requested"]:
            _install_fail_closed_guardrails()
        module.main()
    except Exception:
        if _STATE["requested"] and _STATE["status"] == "pending":
            _fail("guardrail_suite", "inference_error")
        raise
    finally:
        _finalize_state()


if __name__ == "__main__":  # pragma: no cover - native module entry point
    main()
