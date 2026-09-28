"""Keep workflow previews offline and prevent input text from becoming executable markup."""

from __future__ import annotations

import base64
from io import BytesIO
import json

from PIL import Image
import pytest

from npa.workflows.preview_html import image_preview, write_preview


def test_preview_escapes_metadata_and_keeps_full_image_outside_the_report(tmp_path):
    source = Image.new("RGB", (1280, 720), (50, 100, 150))
    data = image_preview(source, width=320)
    with Image.open(BytesIO(base64.b64decode(data.split(",")[1]))) as preview:
        assert preview.size == (320, 180)
    attack = '</script><img src="https://example.invalid" onerror="alert(1)">'
    groups = [
        {
            "title": attack,
            "note": "measured outputs",
            "frames": [{"label": attack, "images": [{"label": attack, "data": data}]}],
        }
    ]
    output = tmp_path / "index.html"
    write_preview(
        output, title=attack, summary=attack, metrics={attack: attack}, groups=groups
    )
    html = output.read_text()
    assert attack not in html
    encoded = html.split('<script id="preview-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(encoded) == groups
    assert "connect-src 'none'" in html
    assert "base-uri 'none'" in html
    assert source.size == (1280, 720)


def test_preview_refuses_empty_media_and_nonfinite_points(tmp_path):
    output = tmp_path / "index.html"
    with pytest.raises(ValueError, match="actual media"):
        write_preview(output, title="Demo", summary="", metrics={}, groups=[])
    groups = [{"frames": [{"points": [[float("nan"), 0, 0]]}]}]
    with pytest.raises(ValueError, match="JSON compliant"):
        write_preview(output, title="Demo", summary="", metrics={}, groups=groups)
    assert not output.exists()
