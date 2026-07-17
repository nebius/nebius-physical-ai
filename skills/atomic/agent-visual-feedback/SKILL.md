---
name: agent-visual-feedback
description: Use when the NPA agent should describe or critique the current viewer (Rerun, video, image, or data) via the Describe this control or multimodal chat.
---

# Agent Visual Feedback (Describe this)

Use this skill when the operator wants the agent to **look at the current
viewer** and give actionable feedback — not a generic caption.

## When To Use

- UI **Describe this** button (stays on the Rerun/viewer tab; chat opens as a drawer)
- Chat turns containing `[npa-visual-feedback]` or “describe this viewer/visual”
- Interpreting held-out Rerun frames, Isaac/GR00T-style sim views, rollout video, images, or Data-pane JSON
- Metadata-only feedback when a frame cannot be captured

## Model

1. Prefer a **quality-captured frame** (vision tier → `Qwen/Qwen2.5-VL-72B-Instruct`).
2. Wait for a non-blank canvas (skip uniform black/white; dense RGB noise is valid).
3. If capture fails, send **metadata/text** and use the reasoning tier — never pretend pixels were seen.
4. Do **not** answer Describe-this from the grounded intent router.

## Visual kinds (generalized — no URI allowlists)

| Kind | Source | What to emphasize |
|------|--------|-------------------|
| `rerun` | Largest same-origin Rerun canvas after quality wait | Sim RGB, depth/seg, 3D mesh, tiled envs, policy strips — **not** “blank” by default |
| `video` | `<video>` current frame | Task progress, success/failure cues |
| `image` | Preview `<img>` | Scene contents, defects |
| `data` | `<pre>` / text excerpt | Report fields, success_rate, missing keys |

Domain hints are inferred from free-text metadata tokens (`artifact_key`, notes,
workflow name) such as isaac / gr00t / heldout / genesis / cosmos — never from a
hardcoded path allowlist.

## UX

- On the Viewer tab (Rerun/Video/Image/Data), chat is a **bottom-right collapsible widget** (`viewer-focus` + FAB).
- Clicking the **Chat** main tab from Viewer opens the widget (stays on the media pane); **Full chat** expands to the Chat tab.
- **Describe this** captures in-place, opens the widget, and queues the vision turn.
- Chat sends are **queued** (`enqueueChatJob`) so Describe + typed messages do not drop.
- Frame captures are JPEG-capped (`maxChars`) and nginx allows `client_max_body_size 32m` so multimodal POSTs do not 413 / `Failed to fetch`.
- Do **not** prefetch `.rrd` bodies (many runs). Soft-swap uses `add_receiver` only; warm wasm/js separately.

## Agent API flow

```http
POST /api/chat
{
  "session_id": "default",
  "visual_context": {
    "kind": "rerun",
    "run_id": "demo-workbench-ui",
    "artifact_key": "…/reports/sim2real.rrd",
    "frame_quality": "rendered",
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

## Reply shape (required)

1. **What I see**  
2. **Likely meaning**  
3. **Operator feedback**  
4. **Next actions**

## Implementation pointers

- Helpers: `npa/src/npa/cli/agent_visual_feedback.py`
- UI: `describeVisual`, `waitForQualityRerunFrame`, `queueChatText`, viewer drawer
- Contracts: `AGENT_VISUAL_FEEDBACK_CONTRACT`, `AGENT_CHAT_QUEUE_CONTRACT`,
  `AGENT_VIEWER_CHAT_DRAWER_CONTRACT`
