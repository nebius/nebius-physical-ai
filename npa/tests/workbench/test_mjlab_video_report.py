"""Verify portable rollout pages preserve video bytes and omit storage secrets."""

import base64
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from types import SimpleNamespace

from npa.workbench.mjlab.video_report import write_video_report


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_page_embeds_exact_video_and_measured_provenance(tmp_path):
    video = b"distinct generated video bytes"
    (tmp_path / "rollout.mp4").write_bytes(video)
    request = SimpleNamespace(
        task="<script>alert('task')</script>",
        seed=43,
        checkpoint="s3://private/checkpoint.pt",
        output_path="s3://private/eval",
    )
    report = {
        "video_frames": 1000,
        "episodes_completed": 1,
        "mean_return": 52.25,
        "mean_episode_length": 1000,
        "score": 0.0,
        "episodes": [{"return": 52.25, "survived": False}],
    }
    write_video_report(tmp_path, request, report, "a" * 64)
    html = (tmp_path / "rollout.html").read_text()
    elements = Page(html).elements
    video_element = next(attrs for tag, attrs in elements if tag == "video")
    assert base64.b64decode(video_element["src"].split(",", 1)[1]) == video
    assert "controls" in video_element
    assert not any(tag == "script" for tag, _ in elements)
    assert "s3://" not in html
    provenance = json.loads(unescape(html.split("<pre>")[1].split("</pre>")[0]))
    assert provenance["checkpoint_sha256"] == "a" * 64
    assert provenance["video_sha256"] == hashlib.sha256(video).hexdigest()
    assert provenance["survival_fraction"] == 0.0
    assert provenance["episodes"] == report["episodes"]
    assert provenance["task"] == request.task
    policy = next(
        attrs["content"]
        for tag, attrs in elements
        if tag == "meta" and attrs.get("http-equiv") == "Content-Security-Policy"
    )
    assert "default-src 'none'" in policy
    assert "media-src data:" in policy
