"""Visual feedback helpers for NPA agent "Describe this" turns.

Pure, side-effect-free helpers embedded into the agent VM backend (same
mechanism as ``agent_chat`` / ``agent_routing``). No network I/O and no
project/secret hardcoding.

Supports per-visual prompts for Rerun, video, image, and data panes so the
vision (or reasoning) model gives operator-actionable feedback instead of a
generic caption. Domain hints are inferred from free-text metadata tokens
(artifact key, notes, workflow name) — never from a hardcoded URI allowlist.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# Marker prefix so grounded intents never swallow Describe-this turns, and so
# skill routing can detect visual-feedback requests from text alone.
DESCRIBE_MARKER = "[npa-visual-feedback]"

# Soft cap so one screenshot cannot dominate Token Factory cost / payload size.
MAX_IMAGE_DATA_URL_CHARS = 2_500_000

VISUAL_KINDS = frozenset({"rerun", "video", "image", "data", "unknown"})

# Token → operator-facing hint. Matched against joined metadata text only.
_DOMAIN_HINT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("gr00t", "groot", "nvidia-gr00t", "eagle"),
        "Metadata suggests a foundation-policy / GR00T-style rollout view in sim — "
        "look for robot embodiment, end-effector, tabletop, or policy camera streams, "
        "not an empty page.",
    ),
    (
        ("isaac", "isaaclab", "isaac-lab", "omniverse", "replicator"),
        "Metadata suggests Isaac Lab / Omniverse sim content — expect synthetic RGB, "
        "depth, segmentation, or a 3D viewport rather than a blank canvas.",
    ),
    (
        ("heldout", "held-out", "eval"),
        "Metadata suggests held-out evaluation imagery — camera streams may look "
        "noisy, low-res, or tiled across envs; that can still be valid eval data.",
    ),
    (
        ("policy", "rollout", "manip", "franka", "cube", "lift"),
        "Metadata suggests policy-rollout / manipulation imagery — describe robot, "
        "object, and task progress when visible.",
    ),
    (
        ("genesis", "mujoco", "mjlab", "locomotion", "sonic"),
        "Metadata suggests locomotion / physics-sim imagery — describe gait, terrain, "
        "and contact cues when visible.",
    ),
    (
        ("cosmos", "world model", "synthetic"),
        "Metadata suggests synthetic / world-model imagery — judge temporal coherence "
        "and artifacts rather than assuming real-robot footage.",
    ),
)

_KIND_GUIDANCE: dict[str, str] = {
    "rerun": (
        "This frame is from the embedded Rerun viewer (robotics / sim timeline). "
        "Identify what is shown: sim RGB camera, depth/seg overlay, 3D scene graph, "
        "policy rollout strip, UI chrome, or true emptiness. "
        "IMPORTANT: dense RGB speckles, tiled env thumbnails, dark viewports with a "
        "robot mesh, or Isaac/GR00T-style synthetic frames are NOT 'blank' — describe "
        "them. Only call a frame blank when it is uniform black/white/gray with no "
        "structure. If noisy, say whether it looks like compressed camera bytes, "
        "uninitialized GPU memory, or a multi-env mosaic. Suggest timeline scrub, "
        "entity selection, and alternate Video/Image artifacts when helpful. "
        "Do not invent hardware footage that is not visible."
    ),
    "video": (
        "This is a frame from a loaded video artifact. Describe visible motion "
        "quality, task progress, and success/failure cues. Call out blur, freezes, "
        "UI overlays, or empty frames. Suggest whether to keep the clip, compare to "
        "Rerun, or re-run eval."
    ),
    "image": (
        "This is a still image artifact. Describe the scene concretely (objects, "
        "robot, camera viewpoint, defects). Note corruption or empty content. "
        "Suggest how it helps debugging or dataset review."
    ),
    "data": (
        "No pixel frame is attached — the operator is on the Data pane "
        "(JSON/text/report). Use any provided text excerpt plus metadata. Infer what "
        "the artifact is and what to verify next (keys, success_rate, stage status, "
        "missing fields). Do not pretend you can see pixels."
    ),
    "unknown": (
        "Describe whatever visual context is available and give practical operator "
        "next steps for the NPA agent workbench."
    ),
}


def normalize_visual_kind(kind: str | None) -> str:
    """Return a supported visual kind token."""
    token = str(kind or "").strip().lower()
    if token in VISUAL_KINDS:
        return token
    return "unknown"


def _meta_blob(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""
    parts: list[str] = []
    for key in (
        "artifact_key",
        "key",
        "artifact_uri",
        "s3_uri",
        "note",
        "visualization_note",
        "workflow_name",
        "run_id",
        "stage",
        "camera",
        "artifact_render",
        "text_excerpt",
    ):
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).lower()


def infer_visual_domain_hints(meta: Mapping[str, Any] | None) -> list[str]:
    """Infer domain hints from free-text metadata (no URI allowlists)."""
    blob = _meta_blob(meta)
    if not blob:
        return []
    # Soft-normalize separators so gr00t_n1 / isaac-lab match token rules.
    normalized = re.sub(r"[_\-/.:]+", " ", blob)
    hints: list[str] = []
    for tokens, hint in _DOMAIN_HINT_RULES:
        if any(token in normalized for token in tokens):
            hints.append(hint)
    return hints[:4]


def describe_user_prompt(kind: str, meta: Mapping[str, Any] | None = None) -> str:
    """Build the user-visible prompt text for a Describe-this turn."""
    visual_kind = normalize_visual_kind(kind)
    meta = meta if isinstance(meta, Mapping) else {}
    run_id = str(meta.get("run_id") or "").strip()
    camera = str(meta.get("camera") or "").strip()
    stage = str(meta.get("stage") or "").strip()
    artifact_key = str(meta.get("artifact_key") or meta.get("key") or "").strip()
    note = str(meta.get("note") or meta.get("visualization_note") or "").strip()
    text_excerpt = str(meta.get("text_excerpt") or "").strip()
    frame_quality = str(meta.get("frame_quality") or "").strip()
    capture = str(meta.get("capture") or "").strip() or (
        "frame" if meta.get("has_image") else "metadata-only"
    )
    guidance = _KIND_GUIDANCE.get(visual_kind, _KIND_GUIDANCE["unknown"])
    domain_hints = infer_visual_domain_hints(meta)
    lines = [
        f"{DESCRIBE_MARKER} Describe this {visual_kind} viewer and give "
        "operator feedback.",
        "",
        "Context:",
        f"- visual_kind: `{visual_kind}`",
        f"- capture: `{capture}`",
    ]
    if frame_quality:
        lines.append(f"- frame_quality: `{frame_quality}`")
    if run_id:
        lines.append(f"- run_id: `{run_id}`")
    if stage:
        lines.append(f"- stage: `{stage}`")
    if camera:
        lines.append(f"- camera: `{camera}`")
    if artifact_key:
        lines.append(f"- artifact: `{artifact_key}`")
    if note:
        lines.append(f"- note: {note[:320]}")
    if text_excerpt:
        lines.append("- data_excerpt:")
        lines.append("```")
        lines.append(text_excerpt[:2500])
        lines.append("```")
    if domain_hints:
        lines.append("")
        lines.append("Domain hints from metadata (not pixel labels):")
        for hint in domain_hints:
            lines.append(f"- {hint}")
    lines.extend(
        [
            "",
            "Instructions:",
            guidance,
        ]
    )
    if capture in {"metadata-only", "text"} or not meta.get("has_image"):
        lines.append(
            "CRITICAL: No viewer frame image is attached. Do NOT invent pixels, "
            "noise, robots, or scenes. State metadata-only limits up front, use "
            "domain hints + artifact/note fields, and suggest how to capture a "
            "real frame (Describe this after the Rerun canvas settles)."
        )
    else:
        lines.append(
            "Use a vision-capable reading of the attached frame. Never claim the "
            "visual is blank solely because the timeline is short or the "
            "application id is unfamiliar — inspect pixels first. Structured sim "
            "RGB, tiled envs, and 3D meshes are valid content."
        )
    lines.extend(
        [
            "",
            "Reply structure:",
            "1. **What I see** — concrete visual description (or metadata-only "
            "limits).",
            "2. **Likely meaning** — how this relates to the active run / stack.",
            "3. **Operator feedback** — what looks healthy vs suspicious.",
            "4. **Next actions** — 2–4 concrete clicks/commands in the agent UI "
            "or CLI.",
        ]
    )
    return "\n".join(lines)


def build_metadata_only_visual_reply(meta: Mapping[str, Any] | None) -> str:
    """Deterministic Describe-this reply when no frame is attached (0 tokens)."""
    meta = meta if isinstance(meta, Mapping) else {}
    kind = normalize_visual_kind(str(meta.get("kind") or meta.get("visual_kind") or "unknown"))
    run_id = str(meta.get("run_id") or "").strip() or "—"
    stage = str(meta.get("stage") or "").strip() or "—"
    camera = str(meta.get("camera") or "").strip() or "—"
    artifact = str(meta.get("artifact_key") or meta.get("key") or "").strip() or "—"
    note = str(meta.get("note") or meta.get("visualization_note") or "").strip()
    excerpt = str(meta.get("text_excerpt") or "").strip()
    hints = infer_visual_domain_hints(meta)
    lines = [
        "**What I see**: No viewer frame was attached to this turn — metadata only. "
        "I am **not** inventing pixels, noise, or a blank canvas claim.",
        "",
        "**Likely meaning**:",
        f"- visual_kind: `{kind}`",
        f"- run_id: `{run_id}`",
        f"- stage: `{stage}`",
        f"- camera: `{camera}`",
        f"- artifact: `{artifact}`",
    ]
    if note:
        lines.append(f"- note: {note[:320]}")
    if hints:
        lines.append("")
        lines.append("Domain hints from metadata:")
        for hint in hints:
            lines.append(f"- {hint}")
    if excerpt:
        lines.extend(["", "Data excerpt (truncated):", "```", excerpt[:1200], "```"])
    lines.extend(
        [
            "",
            "**Operator feedback**: Without a captured frame I cannot judge render "
            "quality. If the UI showed “blank”, wait for the Rerun canvas to settle "
            "(past bundle splash), scrub the timeline, then click **Describe this** again.",
            "",
            "**Next actions**:",
            "1. Stay on the Rerun/Video/Image tab until the viewer shows content.",
            "2. Click **Describe this** again (chat drawer opens; vision tier runs with a frame).",
            "3. If still empty, try **Reload Rerun data** or load the Video/Image artifact.",
            "4. Cross-check run stages / held-out report for this `run_id`.",
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
        "frame_quality",
        "note",
        "visualization_note",
        "workflow_name",
        "text_excerpt",
    )
    lines = ["Active visual context for this Describe-this turn:"]
    for key in allowed:
        value = meta.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        limit = 2500 if key == "text_excerpt" else 400
        if len(text) > limit:
            text = text[:limit] + "…"
        lowered = text.lower()
        if any(token in lowered for token in ("password", "secret", "token=", "ak_", "sk-")):
            continue
        lines.append(f"- {key}: `{text}`")
    for hint in infer_visual_domain_hints(meta):
        lines.append(f"- domain_hint: {hint}")
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
