"""Self-hosted Cosmos Reason2 and Reason3 inference for workbench and sim2real."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_REASON1_MODEL = "nvidia/Cosmos-Reason1-7B"
DEFAULT_REASON2_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_REASON3_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_REASON1_CACHE = "/tmp/hf_home/cosmos-reason1"
DEFAULT_REASON2_CACHE = "/tmp/hf_home/cosmos-reason2"
DEFAULT_REASON3_CACHE = "/tmp/hf_home/cosmos-reason2-2b"
DEFAULT_REASON_EVENT_FRAMES = 32
DEFAULT_REASON_MAX_NEW_TOKENS = 8192
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
    resolved = resolve_cosmos_reason_model_id(model_id)
    mid = resolved.lower()
    if "reason2-2b" in mid:
        return os.environ.get("NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE)
    family = cosmos_reason_family(resolved)
    if family == "reason3":
        return os.environ.get("NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE)
    if family == "reason2":
        return os.environ.get("NPA_COSMOS_REASON2_CACHE", DEFAULT_REASON2_CACHE)
    return os.environ.get("NPA_COSMOS_REASON_CACHE", DEFAULT_REASON1_CACHE)


_VLM_K8S_COMPONENTS = frozenset({"vlm_eval", "vlm_eval_reason2", "vlm_eval_reason3"})


def cosmos_reason_runtime_env() -> dict[str, str]:
    """Writable Hugging Face cache env for Cosmos Reason sibling Jobs."""

    hf_home = os.environ.get("HF_HOME", "/tmp/hf_home")
    return {
        "HF_HOME": hf_home,
        "NPA_COSMOS_REASON2_CACHE": os.environ.get(
            "NPA_COSMOS_REASON2_CACHE", DEFAULT_REASON2_CACHE
        ),
        "NPA_COSMOS_REASON3_CACHE": os.environ.get(
            "NPA_COSMOS_REASON3_CACHE", DEFAULT_REASON3_CACHE
        ),
        "NPA_COSMOS_REASON_CACHE": os.environ.get(
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
    steps2 = {
        int(item.get("step", index)): item
        for index, item in enumerate(reason2_eval.get("per_step") or [])
    }
    steps3 = {
        int(item.get("step", index)): item
        for index, item in enumerate(reason3_eval.get("per_step") or [])
    }
    merged_steps: list[dict[str, Any]] = []
    for step in sorted(set(steps2) | set(steps3)):
        left = steps2.get(step, {})
        right = steps3.get(step, {})
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
                "camera_observation": str(
                    left.get("camera_observation")
                    or right.get("camera_observation")
                    or f"camera-{step:03d}.ppm"
                ),
                "reason2_critique": left.get("critique_text", ""),
                "reason3_critique": right.get("critique_text", ""),
                "reason2_tags": left_tags,
                "reason3_tags": right_tags,
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
        str(reason3_eval.get("summary") or "").strip(),
    ]
    return {
        "schema": VLM_EVAL_SCHEMA,
        "rollout_id": str(
            reason2_eval.get("rollout_id") or reason3_eval.get("rollout_id") or ""
        ),
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
        "reason3": "Cosmos3-Super-Reasoner",
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
    success = bool(payload.get("success", score >= threshold))
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
