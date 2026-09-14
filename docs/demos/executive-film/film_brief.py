"""Build an evidence-aware creative brief for an editor or coding agent to author."""

from film_cache import _fingerprint

_TEXT_FIELDS = ("prompt", "audience", "goal", "tone", "call_to_action")
_LAYOUTS = {"hero": 1, "cinematic": 1, "immersive": 1, "feature": 1, "reason": 1,
            "evidence": 1, "cameras": 1, "screen": 1, "close": 3,
            "triptych": 3, "split": 2, "comparison": 2, "review": 2, "pipeline": 4}


def _validate_brief(brief):
    if not isinstance(brief, dict):
        raise TypeError("Creative brief must be an object")
    for field in _TEXT_FIELDS:
        if not isinstance(brief.get(field), str) or not brief[field].strip():
            raise ValueError(f"Creative brief requires nonempty {field}")
    if type(brief.get("duration_seconds")) is not int or brief["duration_seconds"] <= 0:
        raise ValueError("Creative brief duration_seconds must be positive whole seconds")


def _brief_from_request(storyboard, overrides):
    duration = sum(scene["duration"] for scene in storyboard["scenes"])
    defaults = {"audience": "The audience described in the prompt",
                "goal": "Achieve the purpose described in the prompt",
                "tone": "Match the prompt and audience",
                "call_to_action": "github.com/nebius/nebius-physical-ai",
                "duration_seconds": duration}
    previous = storyboard.get("brief", {}) if overrides.get("prompt") is None else {}
    brief = {**defaults, **previous,
             **{key: value for key, value in overrides.items() if value is not None}}
    _validate_brief(brief)
    return brief


def _authoring_packet(storyboard, assets, overrides):
    brief = _brief_from_request(storyboard, overrides)
    return {"brief": brief, "brief_sha256": _fingerprint(brief),
            "instructions": [
                "Author a new storyboard from this brief; this packet does not generate one.",
                "Choose the narrative, scene count, order, evidence, wording and pacing for this audience and goal.",
                "For executives, connect capabilities to decisions; for engineers, show inputs, steps and inspectable outputs.",
                "Treat these as guidance, not fixed genres. The free-form prompt takes precedence.",
                "Use only available asset roles. Missing evidence is a production request, never an invented result.",
                "Asset names are not proof of their generating tool. Unattributed media requires provenance review.",
                "Preserve input/output/reference distinctions and limitations; separate runs are not one continuous pipeline.",
                "Do not imply task success, measured ROI or deployment readiness without supporting evidence.",
                "Write concise narration that fits each shot; change the script and pacing to fit the requested duration.",
                "Include the complete brief in storyboard.brief and make scene durations sum to duration_seconds.",
                "Review the script and rendered film for brief compliance; hashing a brief does not verify its meaning.",
            ],
            "storyboard_contract": {
                "required": ["title", "brief", "scenes"],
                "scene_fields": ["id", "duration", "layout", "eyebrow", "title", "subtitle",
                                 "assets", "labels", "narration"],
                "title": "Array of headline lines in each scene; use one line for screen layout.",
                "duration": "Positive whole seconds per scene; optional top-level duration must equal their sum.",
                "layouts_and_asset_counts": _LAYOUTS,
                "presentation": "Optional storyboard footer; scene footer overrides it. Close scenes use scene.cta.",
                "details": "Optional quote, quote_heading, source_notes, review_steps and pipeline_labels are scene text.",
            },
            "available_assets": {role: {"kind": asset["kind"], "sha256": asset["sha256"],
                                         "provenance": asset.get("provenance", {"status": "unattributed"})}
                                 for role, asset in assets.items()}}
