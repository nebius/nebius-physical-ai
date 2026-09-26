"""Package a measured MJLab rollout as a self-contained browser video page."""

from __future__ import annotations

import base64
import hashlib
from html import escape
import json

from .schemas import MJLAB_VERSION


def write_video_report(outputs, request, report, checkpoint_sha256) -> None:
    """Embed the unchanged rollout bytes and measured provenance in HTML.

    Args:
        outputs: Local directory containing the verified rollout.mp4.
        request: Evaluation request that produced the video.
        report: Measured evaluation results, including verified frame count.
        checkpoint_sha256: Hash of the checkpoint loaded by this evaluation.
    Returns:
        None; writes rollout.html beside the MP4 for standard publication.
    Raises:
        OSError: The video cannot be read or the page cannot be written.
    """
    video = (outputs / "rollout.mp4").read_bytes()
    provenance = {
        "task": request.task,
        "mjlab_version": MJLAB_VERSION,
        "seed": request.seed,
        "checkpoint_sha256": checkpoint_sha256,
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "video_bytes": len(video),
        "video_frames": report["video_frames"],
        "episodes_completed": report["episodes_completed"],
        "mean_return": report["mean_return"],
        "mean_episode_length": report["mean_episode_length"],
        "survival_fraction": report["score"],
        "episodes": report["episodes"],
    }
    encoded_video = base64.b64encode(video).decode("ascii")
    page = _page(request.task, encoded_video, provenance)
    (outputs / "rollout.html").write_text(page, encoding="utf-8")


def _page(task, encoded_video, provenance):
    title = escape(task)
    details = escape(json.dumps(provenance, indent=2, allow_nan=False))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; media-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>MJLab rollout — {title}</title>
<style>
body {{ margin: 2rem auto; padding: 0 1rem; max-width: 70rem;
        font: 16px/1.5 system-ui, sans-serif; color: #e8edf5; background: #10151e; }}
video {{ display: block; width: 100%; max-height: 75vh; background: #000; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem;
       background: #1a2230; border-radius: .5rem; }}
</style>
</head>
<body>
<h1>MJLab policy rollout</h1>
<p>{title}</p>
<video controls playsinline preload="metadata" aria-label="Rendered MJLab policy episode"
       src="data:video/mp4;base64,{encoded_video}"></video>
<p>The video shows the first environment's first complete evaluation episode.
The metrics below cover every measured episode. Survival means reaching the
task horizon without termination; it does not by itself establish task success.</p>
<h2>Measured results and provenance</h2>
<pre>{details}</pre>
</body>
</html>
"""
