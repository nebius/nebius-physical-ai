"""Cosmos Reason2 and hosted Cosmos3 rollout evaluation."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_REASON1_MODEL = "nvidia/Cosmos-Reason1-7B"
DEFAULT_REASON2_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_COSMOS3_MODEL = "nvidia/Cosmos3-Super-Reasoner"
DEFAULT_REASON1_CACHE = "/tmp/hf_home/cosmos-reason1"
DEFAULT_REASON2_CACHE = "/tmp/hf_home/cosmos-reason2"
DEFAULT_REASON3_CACHE = "/tmp/hf_home/cosmos-reason2-2b"
DEFAULT_HOSTED_EVENT_FRAMES = 8
DEFAULT_REASON_EVENT_FRAMES = 32
DEFAULT_REASON_MAX_NEW_TOKENS = 8192
REFERENCE_VLM_ALIASES = frozenset(
    {"", "npa-cosmos3-reason", "cosmos3-reason", "cosmos-reason", "reason2", "cosmos3"}
)
VLM_EVAL_SCHEMA = "npa.sim2real.vlm_eval.v3"
LEGACY_TWO_EVALUATOR_SCHEMA = "npa.sim2real.vlm_eval.v2"

ERROR_SEVERITY = {
    "collision": 0.95,
    "missed_target": 0.85,
    "unstable": 0.7,
    "late_grasp": 0.55,
    "minor_alignment": 0.3,
    "ok": 0.0,
}


class CosmosReasonError(RuntimeError):
    """Raised when Cosmos Reason inference or parsing fails."""


def cosmos_reason_family(model_id: str) -> str:
    """Return the real model family for a Hugging Face model id."""

    mid = str(model_id or "").strip().lower()
    if "cosmos3-edge" in mid or "cosmos3-super" in mid or "super-reasoner" in mid:
        return "cosmos3"
    if "reason2" in mid or "cosmos-reason2" in mid:
        return "reason2"
    if "reason1" in mid or "cosmos-reason1" in mid:
        return "reason1"
    return "reason2"


def default_reason_cache_dir(model_id: str) -> str:
    resolved = resolve_cosmos_reason_model_id(model_id)
    family = cosmos_reason_family(resolved)
    if "reason2-2b" in resolved.lower():
        return os.environ.get("NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE)
    if family == "cosmos3":
        raise CosmosReasonError("hosted Cosmos3 models do not use a local weight cache")
    if family == "reason2":
        return os.environ.get("NPA_COSMOS_REASON2_CACHE", DEFAULT_REASON2_CACHE)
    return os.environ.get("NPA_COSMOS_REASON_CACHE", DEFAULT_REASON1_CACHE)


_VLM_K8S_COMPONENTS = frozenset({"vlm_eval", "vlm_eval_reason2", "vlm_eval_cosmos3"})


def cosmos_reason_runtime_env() -> dict[str, str]:
    """Writable Hugging Face cache env for Cosmos Reason sibling Jobs.

    The Reason checkpoints are gated, so no image may bake them and every Job has
    to download them. When the operator configured durable weight storage that is
    where they land; otherwise the writable-but-ephemeral ``/tmp`` defaults apply,
    exactly as before.
    """

    from npa.workbench.model_cache import (
        RUNTIME_PREMOUNTED,
        model_cache_env,
        resolve_model_cache_root,
    )

    durable = model_cache_env(resolve_model_cache_root(runtime=RUNTIME_PREMOUNTED))

    def resolved(name: str, fallback: str) -> str:
        return os.environ.get(name) or durable.get(name) or fallback

    return {
        "HF_HOME": resolved("HF_HOME", "/tmp/hf_home"),
        "NPA_COSMOS_REASON2_CACHE": resolved(
            "NPA_COSMOS_REASON2_CACHE", DEFAULT_REASON2_CACHE
        ),
        "NPA_COSMOS_REASON3_CACHE": resolved(
            "NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE
        ),
        "NPA_COSMOS_REASON_CACHE": resolved(
            "NPA_COSMOS_REASON_CACHE", DEFAULT_REASON2_CACHE
        ),
    }


def prepare_cosmos_reason_cache(*, model_id: str) -> str:
    """Ensure the model cache directory exists and HF_HOME is set."""

    cache_dir = default_reason_cache_dir(model_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(cache_dir).parent))
    return cache_dir


def cosmos_reason_k8s_shell_preamble() -> str:
    """Shell snippet run before VLM sibling Jobs download gated HF weights."""

    return (
        'export HF_HOME="${HF_HOME:-/tmp/hf_home}"\n'
        'mkdir -p "${HF_HOME}" '
        '"${NPA_COSMOS_REASON2_CACHE:-/tmp/hf_home/cosmos-reason2}" '
        '"${NPA_COSMOS_REASON3_CACHE:-/tmp/hf_home/cosmos-reason2-2b}" '
        '"${NPA_COSMOS_REASON_CACHE:-/tmp/hf_home/cosmos-reason2}"\n'
    )


def apply_cosmos_reason_kubernetes_env(safe: dict[str, str]) -> dict[str, str]:
    """Merge writable HF cache defaults into a sibling Job env map."""

    for key, value in cosmos_reason_runtime_env().items():
        safe.setdefault(key, value)
    return safe


def vlm_k8s_component(component: str) -> bool:
    return component in _VLM_K8S_COMPONENTS


def task_description_from_manifest(manifest: dict[str, Any]) -> str:
    description = ""
    for key in ("task_description", "task", "instruction", "prompt"):
        value = str(manifest.get(key) or "").strip()
        if value:
            description = value
            break
    if not description:
        description = (
            "Evaluate whether the robot rollout completes the manipulation task. "
            "Use the camera frames and the listed actions to judge physical success, "
            "stability, target alignment, and contact mistakes."
        )
    augmentation = dict(manifest.get("scenario_source_augmentation") or {})
    if not any(
        (
            manifest.get("scenario_difficulty"),
            manifest.get("scenario_config_digest"),
            augmentation.get("lineage_id"),
        )
    ):
        return description
    context = (
        f" Scenario difficulty={manifest.get('scenario_difficulty', '')}; "
        f"applied_config_digest={manifest.get('scenario_config_digest', '')}; "
        f"Cosmos-Transfer lineage={augmentation.get('lineage_id', '')}. "
        "The visible frames are authoritative; lineage identifies the domain-"
        "randomization source and does not imply that Transfer pixels trained the state policy."
    )
    return description + context


def resolve_cosmos_reason_model_id(
    model: str, *, default: str = DEFAULT_REASON2_MODEL
) -> str:
    candidate = str(model or "").strip()
    if candidate in REFERENCE_VLM_ALIASES:
        env_default = (
            os.environ.get("NPA_COSMOS_REASON2_MODEL_ID", "")
            or os.environ.get("NPA_COSMOS_REASON_MODEL_ID", "")
            or default
        )
        candidate = env_default
    return candidate


def merge_reason_evaluations(
    reason2_eval: dict[str, Any],
    cosmos3_eval: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Fuse archived Reason2/Cosmos3 judgments for legacy artifact readers."""

    score2 = float(reason2_eval.get("score", 0.0))
    score3 = float(cosmos3_eval.get("score", 0.0))
    score = round((score2 + score3) / 2.0, 6)
    success = bool(reason2_eval.get("success")) and bool(cosmos3_eval.get("success"))
    steps2 = {
        int(item.get("step", index)): item
        for index, item in enumerate(reason2_eval.get("per_step") or [])
    }
    steps3 = {
        int(item.get("step", index)): item
        for index, item in enumerate(cosmos3_eval.get("per_step") or [])
    }
    merged_steps: list[dict[str, Any]] = []
    for step in sorted(set(steps2) | set(steps3)):
        left = steps2.get(step, {})
        right = steps3.get(step, {})
        left_truth = dict(left.get("simulator_ground_truth") or {})
        right_truth = dict(right.get("simulator_ground_truth") or {})
        if left_truth and right_truth and left_truth != right_truth:
            raise CosmosReasonError(
                f"Reason lanes disagree on simulator ground truth for step {step}"
            )
        simulator_ground_truth = left_truth or right_truth
        scenario_digests = {
            str(value)
            for value in (
                left.get("scenario_config_digest"),
                right.get("scenario_config_digest"),
                simulator_ground_truth.get("scenario_config_digest"),
            )
            if str(value or "").strip()
        }
        if len(scenario_digests) > 1:
            raise CosmosReasonError(
                f"Reason lanes disagree on scenario config digest for step {step}"
            )
        left_tags = _normalize_error_tags(left.get("error_tags") or [])
        right_tags = _normalize_error_tags(right.get("error_tags") or [])
        tags = list(dict.fromkeys(left_tags + right_tags))
        disagreement = bool(left and right and set(left_tags) != set(right_tags))
        source_values = {
            str(item.get("critique_source") or "model_per_step")
            for item in (left, right)
            if item
        }
        confidence = min(
            float(left.get("confidence", 0.65)) if left else 0.25,
            float(right.get("confidence", 0.65)) if right else 0.25,
        )
        if disagreement:
            confidence *= 0.5
        if "model_missing" in source_values or "model_malformed" in source_values:
            confidence = 0.0
        elif "summary_broadcast" in source_values:
            # Backward-compatible rejection for artifacts produced before the
            # per-step fail-closed contract. New inference never emits this.
            confidence = min(confidence, 0.10)
        critique_parts = [
            part.strip()
            for part in (
                str(left.get("critique_text") or "").strip(),
                str(right.get("critique_text") or "").strip(),
            )
            if part.strip()
        ]
        merged_steps.append(
            {
                "step": step,
                "critique_text": " | ".join(critique_parts),
                "error_tags": _normalize_error_tags(tags),
                "action": left.get("action") or right.get("action") or [],
                "simulator_ground_truth": simulator_ground_truth,
                "scenario_config_digest": next(iter(scenario_digests), ""),
                "camera_observation": str(
                    left.get("camera_observation")
                    or right.get("camera_observation")
                    or f"camera-{step:03d}.ppm"
                ),
                "reason2_critique": left.get("critique_text", ""),
                "cosmos3_critique": right.get("critique_text", ""),
                "reason2_tags": left_tags,
                "cosmos3_tags": right_tags,
                "model_disagreement": disagreement,
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
                "critique_source": (
                    "model_missing"
                    if "model_missing" in source_values
                    else (
                        "model_malformed"
                        if "model_malformed" in source_values
                        else (
                            "summary_broadcast"
                            if "summary_broadcast" in source_values
                            else "dual_model_per_step"
                        )
                    )
                ),
            }
        )
    summary_parts = [
        str(reason2_eval.get("summary") or "").strip(),
        str(cosmos3_eval.get("summary") or "").strip(),
    ]
    return {
        "schema": LEGACY_TWO_EVALUATOR_SCHEMA,
        "rollout_id": str(
            reason2_eval.get("rollout_id") or cosmos3_eval.get("rollout_id") or ""
        ),
        "success": success,
        "score": score,
        "per_step": merged_steps,
        "summary": " ".join(part for part in summary_parts if part),
        "model": f"{reason2_eval.get('model')} + {cosmos3_eval.get('model')}",
        "component_source": "cosmos_reason2_cosmos3_vlm",
        "reason2": {
            "model": reason2_eval.get("model"),
            "score": reason2_eval.get("score"),
            "success": reason2_eval.get("success"),
        },
        "cosmos3": {
            "model": cosmos3_eval.get("model"),
            "score": cosmos3_eval.get("score"),
            "success": cosmos3_eval.get("success"),
        },
        "two_evaluator": True,  # archived payload compatibility only
        "threshold": threshold,
    }


def merge_dual_reason_evaluations(
    reason2_eval: dict[str, Any],
    legacy_reason3_eval: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compatibility alias for archived callers; new payloads are Cosmos3-named."""

    return merge_reason_evaluations(
        reason2_eval, legacy_reason3_eval, threshold=threshold
    )


def run_cosmos_reason_vlm(
    *,
    model_id: str,
    image_paths: list[Path],
    actions: list[dict[str, Any]],
    task_description: str,
    rollout_id: str,
    threshold: float,
) -> dict[str, Any]:
    """Run self-hosted Cosmos Reason inference and parse structured VLM output."""

    resolved_model = resolve_cosmos_reason_model_id(model_id)
    family = cosmos_reason_family(resolved_model)
    if family == "cosmos3":
        raise CosmosReasonError(
            "Cosmos3-Super-Reasoner is hosted by Token Factory; use the "
            "token_factory Stage 8 backend instead of the self-hosted loader"
        )
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise CosmosReasonError(
            "Cosmos Reason inference requires torch, Pillow, and transformers "
            f"in the image: {exc}"
        ) from exc

    if not image_paths:
        raise CosmosReasonError("Cosmos Reason inference requires at least one frame")
    if not torch.cuda.is_available():
        raise CosmosReasonError("Cosmos Reason inference requires a CUDA GPU")

    cache_dir = prepare_cosmos_reason_cache(model_id=resolved_model)
    # One primary frame is captured for every decision/event sample. The
    # canonical task contract requires 32 such samples, so evaluating only the
    # first eight makes late grasp/place failures invisible to the VLM.
    max_frames = int(
        os.environ.get("NPA_COSMOS_REASON_MAX_FRAMES", str(DEFAULT_REASON_EVENT_FRAMES))
    )
    selected_paths = image_paths[: max(1, max_frames)]
    for path in selected_paths:
        with Image.open(path) as img:
            img.verify()

    prompt = _cosmos_reason_prompt(
        family=family,
        task_description=task_description,
        actions=actions,
        frame_names=[path.name for path in selected_paths],
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image", "image": str(path.resolve())} for path in selected_paths)
    messages = [{"role": "user", "content": content}]

    print(
        json.dumps(
            {
                "component": "vlm_eval",
                "event": "cosmos_reason_inference_start",
                "family": family,
                "model": resolved_model,
                "frames": [path.name for path in selected_paths],
            },
            sort_keys=True,
        )
    )
    processor = AutoProcessor.from_pretrained(
        resolved_model,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_cls = _reason_model_class(family, AutoModelForImageTextToText)
    model = model_cls.from_pretrained(
        resolved_model,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    first_device = next(model.parameters()).device
    inputs = _prepare_reason_inputs(
        processor=processor,
        messages=messages,
        device=first_device,
    )
    # A compact 32-entry JSON response does not reliably fit in the old 768-token
    # budget. Truncation caused otherwise valid models to fall back to a single
    # rollout summary. Keep this parameterized but make the real default large
    # enough for the required event-local contract.
    max_new_tokens = int(
        os.environ.get(
            "NPA_COSMOS_REASON_MAX_NEW_TOKENS", str(DEFAULT_REASON_MAX_NEW_TOKENS)
        )
    )
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated, strict=False)
    ]
    model_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = _parse_cosmos_reason_output(
        model_text,
        actions=actions,
        rollout_id=rollout_id,
        threshold=threshold,
        family=family,
    )
    payload["component_source"] = "cosmos_reason_vlm"
    payload["model"] = resolved_model
    payload["reason_family"] = family
    payload["frame_count"] = len(selected_paths)
    print(
        json.dumps(
            {
                "component": "vlm_eval",
                "event": "cosmos_reason_inference_complete",
                "family": family,
                "model": resolved_model,
                "score": payload.get("score"),
                "success": payload.get("success"),
            },
            sort_keys=True,
        )
    )
    return payload


def select_hosted_event_frames(
    image_paths: list[Path], *, max_frames: int = DEFAULT_HOSTED_EVENT_FRAMES
) -> list[Path]:
    """Select bounded, deterministic, rollout-wide keyframes."""

    if max_frames <= 0:
        raise CosmosReasonError("hosted max_frames must be positive")
    paths = list(image_paths)
    if len(paths) <= max_frames:
        return paths
    if max_frames == 1:
        return [paths[-1]]
    last = len(paths) - 1
    indices = [(index * last) // (max_frames - 1) for index in range(max_frames)]
    return [paths[index] for index in indices]


def run_token_factory_rollout_vlm(
    *,
    model_id: str,
    image_paths: list[Path],
    actions: list[dict[str, Any]],
    task_description: str,
    rollout_id: str,
    threshold: float,
    client: Any | None = None,
    max_frames: int = DEFAULT_HOSTED_EVENT_FRAMES,
) -> dict[str, Any]:
    """Score one real rollout with hosted Cosmos3 and retain request telemetry."""

    from npa.clients.token_factory import TokenFactoryClient, TokenFactoryError, split_reasoning

    resolved_model = str(model_id or DEFAULT_COSMOS3_MODEL).strip()
    if cosmos_reason_family(resolved_model) != "cosmos3":
        raise CosmosReasonError("token_factory backend requires a Cosmos3 model")
    selected_paths = select_hosted_event_frames(image_paths, max_frames=max_frames)
    if not selected_paths:
        raise CosmosReasonError("hosted Cosmos3 evaluation requires at least one frame")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _cosmos_reason_prompt(
                family="cosmos3",
                task_description=task_description,
                actions=actions,
                frame_names=[path.name for path in selected_paths],
            ),
        }
    ]
    for path in selected_paths:
        if not path.is_file():
            raise CosmosReasonError(f"rollout frame is missing: {path.name}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        )
    active = client or TokenFactoryClient()
    try:
        response = active.chat_completion(
            model=resolved_model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=DEFAULT_REASON_MAX_NEW_TOKENS,
        )
        message = response["choices"][0]["message"]
        model_text, _reasoning = split_reasoning(message)
    except (TokenFactoryError, KeyError, IndexError, TypeError) as exc:
        raise CosmosReasonError(f"hosted Cosmos3 rollout evaluation failed: {exc}") from exc
    if not model_text:
        raise CosmosReasonError("hosted Cosmos3 returned no visible structured evaluation")
    payload = _parse_cosmos_reason_output(
        model_text,
        actions=actions,
        rollout_id=rollout_id,
        threshold=threshold,
        family="cosmos3",
    )
    raw_usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    transport = getattr(active, "last_request_metrics", {}) or {}
    cost_value = raw_usage.get("cost")
    if not isinstance(cost_value, (int, float)):
        cost_value = raw_usage.get("cost_usd")
    cost = float(cost_value) if isinstance(cost_value, (int, float)) else None
    payload.update(
        {
            "component_source": "token_factory_cosmos3_rollout_vlm",
            "provider": "nebius",
            "backend": "token_factory",
            "model": resolved_model,
            "reason_family": "cosmos3",
            "frame_count": len(selected_paths),
            "action_count": len(actions),
            "selected_frames": [path.name for path in selected_paths],
            "request": {
                "request_id": str(response.get("id") or "") or None,
                "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
                "output_tokens": int(raw_usage.get("completion_tokens") or 0),
                "total_tokens": int(raw_usage.get("total_tokens") or 0),
                "latency_seconds": transport.get("latency_seconds"),
                "retries": int(transport.get("retries") or 0),
                "cost_usd": cost,
                "cost_source": "response_usage" if cost is not None else "unavailable",
            },
        }
    )
    return payload


def _reason_model_class(family: str, fallback: Any) -> Any:
    if family == "reason2":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            return fallback
    return fallback


def _prepare_reason_inputs(
    *, processor: Any, messages: list[dict[str, Any]], device: Any
) -> Any:
    """Apply the released self-hosted Reason processor path."""

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise CosmosReasonError(
            "Cosmos Reason2 inference requires qwen-vl-utils in the image"
        ) from exc
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)


def _cosmos_reason_prompt(
    *,
    family: str,
    task_description: str,
    actions: list[dict[str, Any]],
    frame_names: list[str],
) -> str:
    # Simulator ground truth is deliberately excluded: Cosmos labels are
    # calibrated *against* those measurements after inference and must not see
    # the answer in their prompt. Only policy actions and temporal identifiers
    # are model inputs.
    action_excerpt = json.dumps(
        [
            {
                "step": int(action.get("step", index)),
                "sim_step": int(action.get("sim_step", action.get("step", index))),
                "action": list(action.get("action") or []),
            }
            for index, action in enumerate(actions[:64])
        ],
        sort_keys=True,
    )
    expected_steps = [
        int(action.get("step", index)) for index, action in enumerate(actions[:64])
    ]
    label = {
        "reason1": "Cosmos-Reason1",
        "reason2": "Cosmos-Reason2",
        "cosmos3": "Cosmos3-Super-Reasoner",
    }.get(family, "Cosmos Reason")
    return (
        f"You are NVIDIA {label} evaluating a physical robot rollout.\n"
        f"Task description: {task_description}\n"
        f"Frame order: {frame_names}\n"
        f"Actions by step: {action_excerpt}\n"
        f"Required per_step indices: {expected_steps}\n"
        "Return one JSON object only; never use a top-level array. The object "
        "must contain: success (boolean), "
        "score (number from 0 to 1), summary (natural-language critique), and "
        "per_step (array of objects with step, critique_text, error_tags, "
        "camera_observation, confidence). per_step MUST contain exactly one "
        "compact, event-specific object for every required index; keep each "
        "critique_text at 12 words or fewer, and set camera_observation to the "
        "corresponding frame filename rather than another description; never copy or "
        "broadcast the rollout summary into step entries. If a step cannot be "
        "judged visually, use a step-specific 'insufficient visual evidence' "
        "critique with confidence 0. Use only these error tags when applicable: "
        "collision, missed_target, unstable, late_grasp, minor_alignment, ok. "
        "Judge actual visual rollout behavior, not metadata or requested actions."
    )


def _parse_cosmos_reason_output(
    model_text: str,
    *,
    actions: list[dict[str, Any]],
    rollout_id: str,
    threshold: float,
    family: str,
) -> dict[str, Any]:
    payload = _json_object_from_text(model_text)
    if payload is None:
        payload = _recover_truncated_cosmos_payload(model_text)
    if payload is None:
        payload = _parse_unstructured_vlm_output(model_text, threshold=threshold)
    if "score" not in payload:
        raise CosmosReasonError(f"{family} output did not include a numeric score")
    score = max(0.0, min(1.0, float(payload["score"])))
    success = bool(payload.get("success", score >= threshold)) and score >= threshold
    raw_steps = payload.get("per_step") or payload.get("steps") or []
    expected_actions = {
        int(action.get("step", index)): action for index, action in enumerate(actions)
    }
    model_steps: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            continue
        try:
            step = int(raw.get("step", index))
        except (TypeError, ValueError):
            continue
        if step in expected_actions and step not in model_steps:
            model_steps[step] = raw

    # Keep the real rollout-level model result in ``summary``, but never turn it
    # into fake temporal credit. Truncated or omitted local labels become
    # explicit zero-confidence rejections whose text identifies their own step.
    normalized_raw_steps: list[dict[str, Any]] = []
    for step in expected_actions:
        raw = model_steps.get(step)
        if raw is not None:
            normalized_raw_steps.append(raw)
            continue
        normalized_raw_steps.append(
            {
                "step": step,
                "critique_text": (
                    f"{family} returned no model-local critique for step {step}; "
                    "simulator state only."
                ),
                "error_tags": ["ok"],
                "critique_source": "model_missing",
                "confidence": 0.0,
                "camera_observation": f"camera-{step:03d}.ppm",
            }
        )
    per_step: list[dict[str, Any]] = []
    for index, raw in enumerate(normalized_raw_steps):
        step = int(raw.get("step", index))
        critique = str(
            raw.get("critique_text") or raw.get("critique") or raw.get("text") or ""
        ).strip()
        malformed = not critique
        if malformed:
            critique = (
                f"{family} returned a malformed critique for step {step}; "
                "simulator state only."
            )
            normalized_tags = ["ok"]
        else:
            tags = raw.get("error_tags") or raw.get("tags") or _tags_from_text(critique)
            if isinstance(tags, str):
                tags = [tags]
            normalized_tags = _normalize_error_tags(tags)
        critique_source = str(raw.get("critique_source") or "model_per_step")
        if malformed:
            critique_source = "model_malformed"
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": normalized_tags,
                "action": expected_actions[step].get("action", []),
                # Ground truth is deliberately excluded from the model prompt, then
                # reattached from the authoritative rollout row for calibration.
                # Without this post-inference join, temporal credit had no grounded
                # state and a stationary trace collapsed to zero PPO advantages.
                "simulator_ground_truth": dict(
                    expected_actions[step].get("simulator_ground_truth") or {}
                ),
                "scenario_config_digest": str(
                    expected_actions[step].get("scenario_config_digest")
                    or (
                        expected_actions[step].get("simulator_ground_truth") or {}
                    ).get("scenario_config_digest")
                    or ""
                ),
                "camera_observation": str(
                    raw.get("camera_observation") or f"camera-{step:03d}.ppm"
                ),
                "critique_source": critique_source,
                "confidence": max(
                    0.0,
                    min(
                        1.0,
                        float(
                            0.0
                            if malformed
                            else raw.get(
                                "confidence",
                                0.10
                                if raw.get("critique_source") == "summary_broadcast"
                                else 0.65,
                            )
                        ),
                    ),
                ),
            }
        )
    return {
        "schema": VLM_EVAL_SCHEMA if family == "cosmos3" else LEGACY_TWO_EVALUATOR_SCHEMA,
        "rollout_id": str(payload.get("rollout_id") or rollout_id),
        "success": success,
        "score": round(score, 6),
        "per_step": per_step,
        "summary": str(payload.get("summary") or payload.get("critique") or "").strip(),
    }


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
        # Cosmos commonly wraps its single requested evaluation object in a
        # one-element JSON array even when the prompt asks for an object. Keep
        # this narrow: a list of multiple candidate evaluations is ambiguous
        # and must not be silently selected.
        if (
            isinstance(payload, list)
            and len(payload) == 1
            and isinstance(payload[0], dict)
        ):
            return payload[0]
        return None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _recover_truncated_cosmos_payload(text: str) -> dict[str, Any] | None:
    """Recover complete fields and event rows from a token-truncated JSON answer.

    Cosmos occasionally exhausts its generation budget after writing valid event
    objects but before closing the surrounding array/object. We preserve only
    fully decodable rows; the caller marks every missing event as rejected at
    zero confidence. This is intentionally schema-specific rather than a general
    permissive JSON parser.
    """

    success_match = re.search(
        r'"success"\s*:\s*(true|false)', text, flags=re.IGNORECASE
    )
    score_match = re.search(
        r'"score"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', text
    )
    if score_match is None:
        return None
    payload: dict[str, Any] = {"score": float(score_match.group(1))}
    if success_match is not None:
        payload["success"] = success_match.group(1).lower() == "true"

    decoder = json.JSONDecoder()
    summary_match = re.search(r'"summary"\s*:\s*', text)
    if summary_match is not None:
        try:
            summary, _ = decoder.raw_decode(text, summary_match.end())
        except json.JSONDecodeError:
            summary = ""
        if isinstance(summary, str):
            payload["summary"] = summary

    steps_match = re.search(r'"(?:per_step|steps)"\s*:\s*\[', text)
    if steps_match is not None:
        cursor = steps_match.end()
        steps: list[dict[str, Any]] = []
        while cursor < len(text):
            while cursor < len(text) and (
                text[cursor].isspace() or text[cursor] == ","
            ):
                cursor += 1
            if cursor >= len(text) or text[cursor] == "]":
                break
            try:
                item, cursor = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                steps.append(item)
        if steps:
            payload["per_step"] = steps
    return payload


def _parse_unstructured_vlm_output(text: str, *, threshold: float) -> dict[str, Any]:
    lowered = text.lower()
    score_match = re.search(r"(?:score|confidence|rating)\D+([01](?:\.\d+)?)", lowered)
    if not score_match:
        raise CosmosReasonError("Cosmos Reason output was not parseable JSON")
    score = float(score_match.group(1))
    explicit_success = re.search(
        r'["\']?success["\']?\s*[:=]\s*(true|false)',
        lowered,
        flags=re.IGNORECASE,
    )
    if explicit_success is not None:
        success = explicit_success.group(1).lower() == "true"
    elif "fail" in lowered or "unsuccess" in lowered:
        success = False
    elif re.search(r"\b(success|pass(?:ed)?)\b", lowered):
        success = True
    else:
        success = score >= threshold
    return {
        "success": success,
        "score": score,
        "summary": text.strip(),
        "error_tags": _tags_from_text(text),
    }


def _tags_from_text(text: str) -> list[str]:
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    tags = [tag for tag in ERROR_SEVERITY if tag != "ok" and tag in lowered]
    if not tags and re.search(r"\b(ok|success|stable|complete)\b", text.lower()):
        tags = ["ok"]
    return tags or ["minor_alignment"]


def _normalize_error_tags(tags: list[Any]) -> list[str]:
    known = set(ERROR_SEVERITY)
    normalized = []
    for tag in tags:
        value = str(tag).strip().lower().replace("-", "_").replace(" ", "_")
        normalized.append(value if value in known else "minor_alignment")
    return normalized or ["minor_alignment"]
