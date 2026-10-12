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
import hashlib
import inspect
import json
import os
import threading
from pathlib import Path
from typing import Any

GUARDRAIL_STATE_SCHEMA = "npa.cosmos3.guardrail-state.v1"
GUARDRAIL_RECEIPT_ENV = "NPA_COSMOS3_GUARDRAIL_RECEIPT"
GUARDRAILS_REQUESTED_ENV = "NPA_COSMOS3_GUARDRAILS_REQUESTED"
INFERENCE_MODULE_ENV = "NPA_COSMOS3_INFERENCE_MODULE"
DEFAULT_INFERENCE_MODULE = "cosmos_framework.scripts.inference"
ACTION_POLICY_SERVER_MODULE = "cosmos_framework.scripts.action_policy_server_robolab"
# The source-derived ``RobolabPolicyService.infer`` at the pinned framework
# revision. The policy server bypasses OmniInference's normal guardrail call
# path, so this adapter deliberately refuses a changed service seam rather than
# silently wrapping a different method.
ROBOLAB_POLICY_SERVICE_INFER_SHA256 = (
    "8ec4f471e9fedfba42d6fbefe97bfba4d105901a0de25ffb6e827832d8892da2"
)
ALLOWED_INFERENCE_MODULES = frozenset(
    {
        DEFAULT_INFERENCE_MODULE,
        ACTION_POLICY_SERVER_MODULE,
        "npa.workbench.cosmos.structural_transfer_runner",
    }
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
        "evaluation_details": {role: {} for role in _ROLES},
        "postprocessors": {"discovered": [], "evaluated": []},
        "action_policy_requests": {
            "started": 0,
            "completed": 0,
            "failed": 0,
            "last": {
                "status": "not_started",
                "prompt_input": False,
                "generated_media": False,
            },
        },
        "effective": False,
        "status": "pending" if enabled else "explicit_opt_out",
        "failure": "",
    }


_STATE = _new_state()


def _prepare_guardrail_tokenizer() -> None:
    """Materialize pinned Blocklist data before the upstream NLTK import.

    Hugging Face snapshots expose files as links into their blob store. NLTK
    3.10's path hardening deliberately refuses those links at open time, even
    below an allowed root. Reuse the shared Cosmos materializer, which pins the
    Guardrail1 revision, copies only the tokenizer subtree to regular files,
    and verifies every byte before cache reuse.
    """

    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError(
            "HF_TOKEN is required when Cosmos 3 guardrails are enabled; "
            "no tokenizer download was attempted"
        )
    from npa.workbench.cosmos.transfer import (
        _guardrail_nltk_data_path,
        prepare_guardrail_nltk_data,
    )

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    prepare_guardrail_nltk_data(hf_home=hf_home)
    safe_data = str(_guardrail_nltk_data_path(hf_home))
    os.environ["NLTK_DATA"] = os.pathsep.join(
        part for part in (safe_data, os.environ.get("NLTK_DATA", "")) if part
    )


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


def _upstream_reported_evaluation_error(message: str) -> bool:
    """Recognize pinned upstream's fail-open model-error result.

    Qwen3Guard catches every internal exception and returns ``(True, <error>)``.
    Treating that tuple as a successful safety decision would merely move the
    original fail-open behavior into this wrapper.
    """

    normalized = message.strip().lower()
    return normalized.startswith("unexpected error occurred when running")


def _instrument_media_model(model: Any) -> Any:
    """Count successful frame classifications hidden by the upstream filter.

    ``VideoContentSafetyFilter.is_safe_frames`` catches frame-level exceptions
    and can return safe when every classifier call failed. Instrument its pinned
    private inference method so the outer runner can reject partial or empty
    classifier execution without changing any actual safety decision.
    """

    method_name = f"_{model.__class__.__name__}__infer"
    infer = getattr(model, method_name, None)
    if not callable(infer):
        raise RuntimeError(
            "VideoContentSafetyFilter cannot be instrumented for effective evaluation"
        )
    model._npa_guardrail_attempted_evaluations = 0
    model._npa_guardrail_successful_evaluations = 0

    def tracked_infer(*args: Any, **kwargs: Any) -> Any:
        model._npa_guardrail_attempted_evaluations += 1
        result = infer(*args, **kwargs)
        model._npa_guardrail_successful_evaluations += 1
        return result

    setattr(model, method_name, tracked_infer)
    return model


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
        attempted_before = getattr(model, "_npa_guardrail_attempted_evaluations", None)
        successful_before = getattr(
            model, "_npa_guardrail_successful_evaluations", None
        )
        try:
            outcome = model.is_safe(value)
        except Exception:
            _fail(role, f"{name}:evaluation_error")
            raise
        if not isinstance(outcome, tuple) or len(outcome) != 2:
            _fail(role, f"{name}:invalid_result")
            raise RuntimeError(f"{name} returned an invalid guardrail result")
        safe, message = outcome
        if not isinstance(safe, bool) or not isinstance(message, str):
            _fail(role, f"{name}:invalid_result")
            raise RuntimeError(f"{name} returned an invalid safety decision")
        detail: dict[str, Any] = {"decision": "safe" if safe else "blocked"}
        if attempted_before is not None and successful_before is not None:
            attempted = int(model._npa_guardrail_attempted_evaluations) - int(
                attempted_before
            )
            successful = int(model._npa_guardrail_successful_evaluations) - int(
                successful_before
            )
            detail.update(
                {"attempted_inputs": attempted, "successful_inputs": successful}
            )
            if attempted <= 0 or successful != attempted:
                _STATE["evaluation_details"][role][name] = detail
                _fail(role, f"{name}:ineffective_evaluation")
                raise RuntimeError(
                    f"{name} did not successfully evaluate every safety input"
                )
        if safe and _upstream_reported_evaluation_error(message):
            detail["decision"] = "error"
            _STATE["evaluation_details"][role][name] = detail
            _fail(role, f"{name}:evaluation_error")
            raise RuntimeError(f"{name} reported an internal evaluation error")
        _STATE["evaluation_details"][role][name] = detail
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
        model = _media_safety_model_class()(offload_model_to_cpu=offload_model_to_cpu)
        runner.safety_models = [_instrument_media_model(model)]
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


def _action_policy_requests() -> dict[str, Any]:
    requests = _STATE["action_policy_requests"]
    if not isinstance(requests, dict):  # pragma: no cover - internal invariant
        raise RuntimeError("invalid action-policy guardrail request state")
    return requests


def _begin_action_policy_request() -> None:
    """Durably mark a request pending before its prompt reaches the model."""

    if _STATE["status"] in {"blocked", "ineffective", "explicit_opt_out"}:
        raise RuntimeError(
            "action-policy guardrails are not available for another request"
        )
    requests = _action_policy_requests()
    request_id = int(requests["started"]) + 1
    requests["started"] = request_id
    requests["last"] = {
        "id": request_id,
        "status": "pending",
        "prompt_input": False,
        "generated_media": False,
    }
    _STATE["effective"] = False
    _STATE["status"] = "pending"
    _STATE["failure"] = ""
    _write_state()


def _complete_action_policy_step(role: str) -> None:
    request = _action_policy_requests()["last"]
    if request.get("status") != "pending":
        raise RuntimeError("action-policy guardrail request was not pending")
    request[role] = True
    _write_state()


def _complete_action_policy_request() -> None:
    """Publish success before the WebSocket server can emit the response."""

    requests = _action_policy_requests()
    request = requests["last"]
    if request.get("status") != "pending" or not all(
        request.get(role) is True for role in _ROLES
    ):
        raise RuntimeError("action-policy request did not complete both guardrails")
    request["status"] = "passed"
    requests["completed"] = int(requests["completed"]) + 1
    _STATE["effective"] = True
    _STATE["status"] = "passed"
    _STATE["failure"] = ""
    _write_state()
    _log_state()


def _fail_action_policy_request() -> None:
    requests = _action_policy_requests()
    request = requests["last"]
    if request.get("status") == "pending":
        request["status"] = "failed"
        requests["failed"] = int(requests["failed"]) + 1
    if _STATE["status"] == "pending":
        _fail("guardrail_suite", "action_policy_request_error")
    else:
        _write_state()


def _source_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decoded_video_to_cthw(decoded_video: Any) -> Any:
    """Adapt the policy server's documented uint8 THWC output for its guardrail."""

    import torch

    video = torch.as_tensor(decoded_video)
    if video.ndim != 4 or video.shape[-1] != 3:
        _fail("generated_media", "missing_decoded_video")
        raise RuntimeError("RoboLab policy did not produce decoded THWC video")
    if video.dtype != torch.uint8:
        _fail("generated_media", "invalid_decoded_video")
        raise RuntimeError("RoboLab policy decoded video was not uint8")
    return video.permute(3, 0, 1, 2).to(dtype=torch.float32).div(255.0)


def _guardrail_video_to_thwc(video_cthw: Any) -> Any:
    """Return the guarded/blurred video in the server's supported THWC shape."""

    import torch

    return (
        video_cthw.mul(255.0)
        .clamp(0.0, 255.0)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .detach()
        .cpu()
        .numpy()
    )


def _install_action_policy_service_guardrails(module: Any) -> None:
    """Patch the pinned RoboLab service's real prompt/generate/decode seam.

    The policy service calls ``model.generate_samples_from_batch`` directly,
    unlike the ordinary ``OmniInference`` path. Its generated video is only
    available after its own decode branch. Guard the exact model input at that
    direct call, force the documented decode branch through ``--decode-video``,
    then guard the decoded media before ``infer`` returns it to OpenPI.
    """

    service_class = getattr(module, "RobolabPolicyService", None)
    if service_class is None:
        raise RuntimeError(
            "RoboLab policy service is unavailable for guardrail binding"
        )
    original_infer = getattr(service_class, "infer", None)
    if not callable(original_infer):
        raise RuntimeError("RoboLab policy service has no infer method to guard")
    if (
        _source_sha256(inspect.getsource(original_infer))
        != ROBOLAB_POLICY_SERVICE_INFER_SHA256
    ):
        raise RuntimeError(
            "RoboLab policy-service infer seam differs from the pinned source"
        )
    if getattr(service_class, "_npa_guardrail_adapter_installed", False):
        return

    original_init = service_class.__init__

    def guarded_init(service: Any, *args: Any, **kwargs: Any) -> None:
        original_init(service, *args, **kwargs)
        if getattr(service.cfg, "decode_video", False) is not True:
            raise RuntimeError(
                "RoboLab policy guardrails require --decode-video True for media evaluation"
            )
        if getattr(service.pipe, "guardrails", None) is None:
            raise RuntimeError(
                "RoboLab policy server started without guardrail runners"
            )
        original_generate = getattr(service.model, "generate_samples_from_batch", None)
        if not callable(original_generate):
            raise RuntimeError("RoboLab policy model has no generation method to guard")
        service._npa_guardrail_request_lock = threading.RLock()

        def guarded_generate(
            data_batch: dict[str, Any], *call_args: Any, **call_kwargs: Any
        ) -> Any:
            input_caption_key = getattr(service.model, "input_caption_key", None)
            prompts = (
                data_batch.get(input_caption_key)
                if isinstance(input_caption_key, str)
                else None
            )
            if (
                not isinstance(prompts, list)
                or len(prompts) != 1
                or not isinstance(prompts[0], str)
            ):
                raise RuntimeError(
                    "RoboLab policy input did not contain one native prompt"
                )
            service.pipe._run_text_guardrail("robolab-policy-service", prompts[0])
            _complete_action_policy_step("prompt_input")
            return original_generate(data_batch, *call_args, **call_kwargs)

        service.model.generate_samples_from_batch = guarded_generate

    def guarded_infer(service: Any, obs: dict[str, Any]) -> dict[str, Any]:
        request_lock = getattr(service, "_npa_guardrail_request_lock", None)
        if request_lock is None:
            raise RuntimeError(
                "RoboLab policy service missed guardrail request initialization"
            )
        with request_lock:
            _begin_action_policy_request()
            try:
                outputs = original_infer(service, obs)
                video_cthw = _decoded_video_to_cthw(outputs.get("video"))
                guarded_video = service.pipe._run_video_guardrail(
                    "robolab-policy-service", video_cthw
                )
                outputs["video"] = _guardrail_video_to_thwc(guarded_video)
                _complete_action_policy_step("generated_media")
                _complete_action_policy_request()
                return outputs
            except Exception:
                _fail_action_policy_request()
                raise

    service_class.__init__ = guarded_init
    service_class.infer = guarded_infer
    service_class._npa_guardrail_adapter_installed = True


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
    try:
        if _STATE["requested"]:
            _prepare_guardrail_tokenizer()
        # Importing the native script performs its required process initialization.
        module = importlib.import_module(module_name)
        if _STATE["requested"]:
            _install_fail_closed_guardrails()
            if module_name == ACTION_POLICY_SERVER_MODULE:
                _install_action_policy_service_guardrails(module)
        module.main()
    except Exception:
        if _STATE["requested"] and _STATE["status"] == "pending":
            _fail("guardrail_suite", "inference_error")
        raise
    finally:
        _finalize_state()


if __name__ == "__main__":  # pragma: no cover - native module entry point
    main()
