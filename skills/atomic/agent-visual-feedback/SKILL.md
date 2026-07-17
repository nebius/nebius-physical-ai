---
name: agent-visual-feedback
description: Use when the NPA agent should describe or critique the current viewer (Rerun, video, image, or data) via the Describe this control or multimodal chat.
---

# Agent Visual Feedback (Describe this)

Use this skill when the operator wants the agent to **look at the current
viewer** and give actionable feedback — not a generic caption.

## When To Use

- UI **Describe this** button next to Reload Rerun
- Chat turns containing `[npa-visual-feedback]` or “describe this viewer/visual”
- Interpreting noisy/static held-out Rerun frames, rollout video, or image artifacts
- Metadata-only feedback when a frame cannot be captured (Data pane / blocked canvas)

## Model

1. Prefer a **captured frame** (vision tier → `Qwen/Qwen2.5-VL-72B-Instruct`).
2. If capture fails, send **metadata-only** context and use the reasoning tier —
   never pretend pixels were seen.
3. Do **not** answer Describe-this from the grounded intent router (zero-token
   path). Vision/reasoning must run.

## Visual kinds

| Kind | Source | What to emphasize |
|------|--------|-------------------|
| `rerun` | Same-origin Rerun canvas | Held-out camera vs 3D proxy, noise/static, timeline/entity checks |
| `video` | `<video>` current frame | Task progress, success/failure cues, freezes/blur |
| `image` | Preview `<img>` | Scene contents, defects, dataset usefulness |
| `data` | JSON/text pane (no pixels) | Report fields, success_rate, missing keys, next verify steps |

## Agent API flow

```http
POST /api/chat
{
  "session_id": "default",
  "visual_context": {
    "kind": "rerun",
    "run_id": "agent-run-…",
    "stage": "artifact-loaded",
    "camera": "heldout-sim",
    "capture": "frame"
  },
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "[npa-visual-feedback] Describe this rerun viewer…"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

Constraints:

- Image parts must be `data:image/...` URLs (downscaled; oversized payloads dropped).
- Session history stores a **text stub** only — never persist megabyte data-URLs.
- No project IDs, bucket names, or secrets in prompts.

## Reply shape (required)

1. **What I see** — concrete description or explicit metadata-only limits  
2. **Likely meaning** — relation to Sim2Real / active run  
3. **Operator feedback** — healthy vs suspicious  
4. **Next actions** — 2–4 concrete UI/CLI steps  

## Implementation pointers

- Helpers: `npa/src/npa/cli/agent_visual_feedback.py` (embedded into agent backend)
- UI capture + button: embedded `ui.html` in `npa/src/npa/cli/agent.py`
- Routing: `has_image_content` / `TIER_VISION` in `agent_routing.py`
- Contract markers: `AGENT_VISUAL_FEEDBACK_CONTRACT`
