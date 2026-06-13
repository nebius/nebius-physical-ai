"""Self-hosted Cosmos Reason2 and Reason3 inference for workbench and sim2real."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_REASON1_MODEL = "nvidia/Cosmos-Reason1-7B"
DEFAULT_REASON2_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_REASON3_MODEL = "nvidia/Cosmos-Reason1-7B"
DEFAULT_REASON1_CACHE = "/tmp/hf_home/cosmos-reason1"
DEFAULT_REASON2_CACHE = "/tmp/hf_home/cosmos-reason2"
DEFAULT_REASON3_CACHE = "/tmp/hf_home/cosmos-reason3"
REFERENCE_VLM_ALIASES = frozenset(
    {"", "npa-cosmos3-reason", "cosmos3-reason", "cosmos-reason", "reason2", "reason3"}
)
VLM_EVAL_SCHEMA = "npa.sim2real.vlm_eval.v1"

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
    """Return ``reason1``, ``reason2``, or ``reason3`` for a Hugging Face model id."""

    mid = str(model_id or "").strip().lower()
    if "super-reasoner" in mid or "cosmos3-super" in mid:
        return "reason3"
    if "reason2" in mid or "cosmos-reason2" in mid:
        return "reason2"
    if "reason1" in mid or "cosmos-reason1" in mid:
        return "reason1"
    return "reason2"


def default_reason_cache_dir(model_id: str) -> str:
    family = cosmos_reason_family(model_id)
    if family == "reason3":
        return os.environ.get("NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE)
    if family == "reason2":
        return os.environ.get("NPA_COSMOS_REASON2_CACHE", DEFAULT_REASON2_CACHE)
    return os.environ.get("NPA_COSMOS_REASON_CACHE", DEFAULT_REASON1_CACHE)


def task_description_from_manifest(manifest: dict[str, Any]) -> str:
    for key in ("task_description", "task", "instruction", "prompt"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    return (
        "Evaluate whether the robot rollout completes the manipulation task. "
        "Use the camera frames and the listed actions to judge physical success, "
        "stability, target alignment, and contact mistakes."
    )


def resolve_cosmos_reason_model_id(model: str, *, default: str = DEFAULT_REASON2_MODEL) -> str:
    candidate = str(model or "").strip()
    if candidate in REFERENCE_VLM_ALIASES:
        env_default = (
            os.environ.get("NPA_COSMOS_REASON3_MODEL_ID", "")
            or os.environ.get("NPA_COSMOS_REASON2_MODEL_ID", "")
            or os.environ.get("NPA_COSMOS_REASON_MODEL_ID", "")
            or default
        )
        candidate = env_default
    return candidate


def merge_dual_reason_evaluations(
    reason2_eval: dict[str, Any],
    reason3_eval: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Fuse Reason2 and Reason3 judgments into one sim2real VLM eval payload."""

    score2 = float(reason2_eval.get("score", 0.0))
    score3 = float(reason3_eval.get("score", 0.0))
    score = round((score2 + score3) / 2.0, 6)
    success = bool(reason2_eval.get("success")) and bool(reason3_eval.get("success"))
    if not success and score >= threshold:
        success = score >= threshold
    steps2 = {int(item.get("step", index)): item for index, item in enumerate(reason2_eval.get("per_step") or [])}
    steps3 = {int(item.get("step", index)): item for index, item in enumerate(reason3_eval.get("per_step") or [])}
    merged_steps: list[dict[str, Any]] = []
    for step in sorted(set(steps2) | set(steps3)):
        left = steps2.get(step, {})
        right = steps3.get(step, {})
        tags = list(
            dict.fromkeys(
                list(left.get("error_tags") or []) + list(right.get("error_tags") or [])
            )
        )
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
                "camera_observation": str(
                    left.get("camera_observation")
                    or right.get("camera_observation")
                    or f"camera-{step:03d}.ppm"
                ),
                "reason2_critique": left.get("critique_text", ""),
                "reason3_critique": right.get("critique_text", ""),
            }
        )
    summary_parts = [
        str(reason2_eval.get("summary") or "").strip(),
        str(reason3_eval.get("summary") or "").strip(),
    ]
    return {
        "schema": VLM_EVAL_SCHEMA,
        "rollout_id": str(reason2_eval.get("rollout_id") or reason3_eval.get("rollout_id") or ""),
        "success": success,
        "score": score,
        "per_step": merged_steps,
        "summary": " ".join(part for part in summary_parts if part),
        "model": f"{reason2_eval.get('model')} + {reason3_eval.get('model')}",
        "component_source": "cosmos_dual_reason_vlm",
        "reason2": {
            "model": reason2_eval.get("model"),
            "score": reason2_eval.get("score"),
            "success": reason2_eval.get("success"),
        },
        "reason3": {
            "model": reason3_eval.get("model"),
            "score": reason3_eval.get("score"),
            "success": reason3_eval.get("success"),
        },
        "dual_reason": True,
        "threshold": threshold,
    }


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
    try:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:
        raise CosmosReasonError(
            "Cosmos Reason inference requires torch, Pillow, transformers, "
            f"and qwen-vl-utils in the image: {exc}"
        ) from exc

    if not image_paths:
        raise CosmosReasonError("Cosmos Reason inference requires at least one frame")
    if not torch.cuda.is_available():
        raise CosmosReasonError("Cosmos Reason inference requires a CUDA GPU")

    cache_dir = default_reason_cache_dir(resolved_model)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(cache_dir).parent))
    max_frames = int(os.environ.get("NPA_COSMOS_REASON_MAX_FRAMES", "8"))
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
    content.extend({"type": "image", "image": str(path)} for path in selected_paths)
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
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    first_device = next(model.parameters()).device
    inputs = inputs.to(first_device)
    max_new_tokens = int(os.environ.get("NPA_COSMOS_REASON_MAX_NEW_TOKENS", "768"))
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


def _reason_model_class(family: str, fallback: Any) -> Any:
    if family in {"reason2", "reason3"}:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError:
            return fallback
    return fallback


def _cosmos_reason_prompt(
    *,
    family: str,
    task_description: str,
    actions: list[dict[str, Any]],
    frame_names: list[str],
) -> str:
    action_excerpt = json.dumps(actions[:16], sort_keys=True)
    label = {
        "reason1": "Cosmos-Reason1",
        "reason2": "Cosmos-Reason2",
        "reason3": "Cosmos3-Super-Reasoner",
    }.get(family, "Cosmos Reason")
    return (
        f"You are NVIDIA {label} evaluating a physical robot rollout.\n"
        f"Task description: {task_description}\n"
        f"Frame order: {frame_names}\n"
        f"Actions by step: {action_excerpt}\n"
        "Return JSON only. The JSON must contain: success (boolean), "
        "score (number from 0 to 1), summary (natural-language critique), and "
        "per_step (array of objects with step, critique_text, error_tags, "
        "camera_observation). Use only these error tags when applicable: "
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
        payload = _parse_unstructured_vlm_output(model_text, threshold=threshold)
    if "score" not in payload:
        raise CosmosReasonError(f"{family} output did not include a numeric score")
    score = max(0.0, min(1.0, float(payload["score"])))
    success = bool(payload.get("success", score >= threshold))
    raw_steps = payload.get("per_step") or payload.get("steps") or []
    if not raw_steps:
        critique = str(
            payload.get("summary")
            or payload.get("critique")
            or payload.get("critique_text")
            or model_text
        ).strip()
        tags = payload.get("error_tags") or _tags_from_text(critique)
        raw_steps = [
            {
                "step": int(action.get("step", index)),
                "critique_text": critique,
                "error_tags": tags,
                "critique_source": "summary_broadcast",
                "camera_observation": f"camera-{int(action.get('step', index)):03d}.ppm",
            }
            for index, action in enumerate(actions)
        ]
    per_step: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raw = {"critique_text": str(raw)}
        step = int(raw.get("step", index))
        tags = raw.get("error_tags") or raw.get("tags") or _tags_from_text(str(raw))
        if isinstance(tags, str):
            tags = [tags]
        normalized_tags = _normalize_error_tags(tags)
        critique = str(
            raw.get("critique_text")
            or raw.get("critique")
            or raw.get("text")
            or payload.get("summary")
            or ""
        ).strip()
        if not critique:
            raise CosmosReasonError(f"{family} per_step output lacks critique text")
        per_step.append(
            {
                "step": step,
                "critique_text": critique,
                "error_tags": normalized_tags,
                "action": actions[index].get("action", []) if index < len(actions) else [],
                "camera_observation": str(
                    raw.get("camera_observation") or f"camera-{step:03d}.ppm"
                ),
            }
        )
    return {
        "schema": VLM_EVAL_SCHEMA,
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
        return payload if isinstance(payload, dict) else None
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


def _parse_unstructured_vlm_output(text: str, *, threshold: float) -> dict[str, Any]:
    lowered = text.lower()
    score_match = re.search(r"(?:score|confidence|rating)\D+([01](?:\.\d+)?)", lowered)
    if not score_match:
        raise CosmosReasonError("Cosmos Reason output was not parseable JSON")
    score = float(score_match.group(1))
    if "success" in lowered or "pass" in lowered:
        success = True
    elif "fail" in lowered or "unsuccess" in lowered:
        success = False
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
