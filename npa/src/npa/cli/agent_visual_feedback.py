"""Visual feedback helpers for NPA agent "Describe this" turns.

Pure, side-effect-free helpers embedded into the agent VM backend (same
mechanism as ``agent_chat`` / ``agent_routing``). No network I/O and no
project/secret hardcoding.

Supports per-visual prompts for Rerun, video, image, and data panes so the
vision (or reasoning) model gives operator-actionable feedback instead of a
generic caption.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Marker prefix so grounded intents never swallow Describe-this turns, and so
# skill routing can detect visual-feedback requests from text alone.
DESCRIBE_MARKER = "[npa-visual-feedback]"

# Soft cap so one screenshot cannot dominate Token Factory cost / payload size.
MAX_IMAGE_DATA_URL_CHARS = 2_500_000

VISUAL_KINDS = frozenset({"rerun", "video", "image", "data", "unknown"})

_KIND_GUIDANCE: dict[str, str] = {
    "rerun": (
        "This frame is from the embedded Rerun viewer (Sim2Real / robotics "
        "timeline). Identify what is shown (held-out sim camera, policy "
        "rollout, 3D proxy, UI chrome, or noise/static). If the image looks "
        "like random RGB noise or uninitialized pixels, say so plainly and "
        "suggest concrete checks: timeline scrub, entity selection, whether "
        "the .rrd is a real camera stream vs placeholder, and cross-check "
        "held-out eval report / video artifacts. Do not invent robot hardware "
        "footage that is not visible."
    ),
    "video": (
        "This is a frame from a loaded video artifact (often held-out success/"
        "failure MP4). Describe visible motion quality, task progress, and "
        "obvious success/failure cues. Call out blur, freezes, UI overlays, "
        "or empty frames. Suggest whether the operator should keep this clip, "
        "compare to Rerun, or re-run eval."
    ),
    "image": (
        "This is a still image artifact from the run. Describe the scene "
        "concretely (objects, robot, camera viewpoint, defects). Note "
        "corruption, saturation, or empty content. Suggest how it helps "
        "operator debugging or dataset review."
    ),
    "data": (
        "No pixel frame is attached — the operator is on the Data pane "
        "(JSON/text/report). Infer from metadata what the artifact is and "
        "what an operator should verify next (keys, success_rate, stage "
        "status, missing fields). Be concrete; do not pretend you can see "
        "pixels."
    ),
    "unknown": (
        "Describe whatever visual context is available and give practical "
        "operator next steps for the NPA agent workbench."
    ),
}


def normalize_visual_kind(kind: str | None) -> str:
    """Return a supported visual kind token."""
    token = str(kind or "").strip().lower()
    if token in VISUAL_KINDS:
        return token
    return "unknown"


def describe_user_prompt(kind: str, meta: Mapping[str, Any] | None = None) -> str:
    """Build the user-visible prompt text for a Describe-this turn."""
    visual_kind = normalize_visual_kind(kind)
    meta = meta if isinstance(meta, Mapping) else {}
    run_id = str(meta.get("run_id") or "").strip()
    camera = str(meta.get("camera") or "").strip()
    stage = str(meta.get("stage") or "").strip()
    artifact_key = str(meta.get("artifact_key") or meta.get("key") or "").strip()
    capture = str(meta.get("capture") or "").strip() or (
        "frame" if meta.get("has_image") else "metadata-only"
    )
    guidance = _KIND_GUIDANCE.get(visual_kind, _KIND_GUIDANCE["unknown"])
    lines = [
        f"{DESCRIBE_MARKER} Describe this {visual_kind} viewer and give "
        "operator feedback.",
        "",
        "Context:",
        f"- visual_kind: `{visual_kind}`",
        f"- capture: `{capture}`",
    ]
    if run_id:
        lines.append(f"- run_id: `{run_id}`")
    if stage:
        lines.append(f"- stage: `{stage}`")
    if camera:
        lines.append(f"- camera: `{camera}`")
    if artifact_key:
        lines.append(f"- artifact: `{artifact_key}`")
    lines.extend(
        [
            "",
            "Instructions:",
            guidance,
            "",
            "Reply structure:",
            "1. **What I see** — concrete visual description (or metadata-only "
            "limits).",
            "2. **Likely meaning** — how this relates to Sim2Real / the active "
            "run.",
            "3. **Operator feedback** — what looks healthy vs suspicious.",
            "4. **Next actions** — 2–4 concrete clicks/commands in the agent UI "
            "or CLI.",
        ]
    )
    return "\n".join(lines)


def format_visual_context_block(meta: Mapping[str, Any] | None) -> str:
    """Format non-secret visual metadata for the system prompt."""
    if not isinstance(meta, Mapping) or not meta:
        return ""
    allowed = (
        "kind",
        "visual_kind",
        "run_id",
        "stage",
        "camera",
        "artifact_render",
        "artifact_key",
        "artifact_uri",
        "capture",
        "note",
    )
    lines = ["Active visual context for this Describe-this turn:"]
    for key in allowed:
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text or len(text) > 400:
            continue
        # Never echo credentials-looking values into the prompt.
        lowered = text.lower()
        if any(token in lowered for token in ("password", "secret", "token=", "ak_", "sk-")):
            continue
        lines.append(f"- {key}: `{text}`")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def text_from_content(content: Any) -> str:
    """Extract plain text from string or multimodal content parts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            if part_type in {"text", "input_text"}:
                text = str(part.get("text") or "").strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content or "").strip()


def text_from_messages(messages: Sequence[Any] | None) -> str:
    """Return the latest user text from a message list."""
    if not messages:
        return ""
    for item in reversed(list(messages)):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").strip() != "user":
            continue
        text = text_from_content(item.get("content"))
        if text:
            return text
    return ""


def has_image_parts(content: Any) -> bool:
    """Return True when content includes image parts or a data-URI image."""
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and str(part.get("type") or "").startswith("image"):
                return True
        return False
    return isinstance(content, str) and "data:image/" in content


def is_visual_feedback_turn(
    *,
    user_text: str = "",
    messages: Sequence[Any] | None = None,
    visual_context: Mapping[str, Any] | None = None,
) -> bool:
    """Detect Describe-this / visual-feedback turns that must not be grounded away."""
    if isinstance(visual_context, Mapping) and visual_context:
        return True
    if messages:
        for item in messages:
            if isinstance(item, dict) and has_image_parts(item.get("content")):
                return True
    text = str(user_text or "").strip()
    if not text and messages:
        text = text_from_messages(messages)
    lowered = text.lower()
    if DESCRIBE_MARKER.lower() in lowered:
        return True
    if "describe this" in lowered and any(
        token in lowered for token in ("visual", "viewer", "rerun", "video", "image", "frame")
    ):
        return True
    return False


def _sanitize_image_url(url: str) -> str | None:
    value = str(url or "").strip()
    if not value.startswith("data:image/"):
        return None
    if len(value) > MAX_IMAGE_DATA_URL_CHARS:
        return None
    return value


def normalize_message_content_for_llm(content: Any) -> str | list[dict[str, Any]]:
    """Preserve OpenAI-style multimodal parts for Token Factory vision calls."""
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip()
            if part_type in {"text", "input_text"}:
                text = str(part.get("text") or "").strip()
                if text:
                    parts.append({"type": "text", "text": text})
            elif part_type in {"image_url", "image"}:
                image = part.get("image_url")
                url = ""
                if isinstance(image, dict):
                    url = str(image.get("url") or "").strip()
                elif isinstance(image, str):
                    url = image.strip()
                elif part_type == "image":
                    url = str(part.get("url") or part.get("image") or "").strip()
                safe = _sanitize_image_url(url)
                if safe:
                    parts.append({"type": "image_url", "image_url": {"url": safe}})
        if not parts:
            return ""
        if len(parts) == 1 and parts[0].get("type") == "text":
            return str(parts[0].get("text") or "")
        return parts
    return str(content or "").strip()


def normalize_messages_for_llm(raw: object) -> list[dict[str, Any]]:
    """Normalize chat messages while keeping multimodal user content intact."""
    history: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return history
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        if role not in {"user", "assistant", "system"}:
            continue
        content = normalize_message_content_for_llm(item.get("content"))
        if not content:
            continue
        history.append({"role": role, "content": content})
    return history[-80:]


def content_for_storage(content: Any, *, visual_kind: str = "") -> str:
    """Persist a text stub; never write large data-URL images into chat history."""
    text = text_from_content(content)
    if has_image_parts(content):
        kind = normalize_visual_kind(visual_kind) if visual_kind else "visual"
        suffix = " _(viewer frame attached; image omitted from stored history)_"
        if text:
            return f"{text}{suffix}"
        return f"{DESCRIBE_MARKER} Describe this {kind} viewer.{suffix}"
    return text


def normalize_messages_for_storage(
    raw: object,
    *,
    visual_kind: str = "",
) -> list[dict[str, str]]:
    """History shape for session persistence (text only, no image payloads)."""
    history: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return history
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = content_for_storage(item.get("content"), visual_kind=visual_kind)
        if content:
            history.append({"role": role, "content": content})
    return history[-80:]


def build_multimodal_user_content(
    text: str,
    image_data_url: str | None,
) -> str | list[dict[str, Any]]:
    """Build user content for /api/chat (text, or text+image parts)."""
    prompt = str(text or "").strip()
    safe = _sanitize_image_url(str(image_data_url or ""))
    if not safe:
        return prompt
    parts: list[dict[str, Any]] = []
    if prompt:
        parts.append({"type": "text", "text": prompt})
    parts.append({"type": "image_url", "image_url": {"url": safe}})
    return parts
