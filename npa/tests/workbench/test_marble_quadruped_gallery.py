"""Continuous proof packaging retains recorded payloads and offline containment."""

import base64
import hashlib
import json

import pytest

from npa.workbench.marble.api import MarbleError
from npa.workbench.marble.quadruped_report import _write_html, write_quadruped_gallery

MARKER = '<script type="application/json" id="capture-data">'


def _capture():
    return {
        "frames": 2,
        "trajectory": [{"frame": 0}, {"frame": 1}],
        "videos": {
            name: "data:video/mp4;base64," + base64.b64encode(name.encode()).decode()
            for name in ("observer", "frames", "depth-frames")
        },
    }


def test_gallery_preserves_every_embedded_video_and_telemetry_row(tmp_path):
    reports = {}
    expected = []
    for index, label in enumerate(
        ("Warehouse", "Home </script><script>alert(1)</script>")
    ):
        directory = tmp_path / str(index)
        directory.mkdir()
        capture = _capture()
        capture["source"] = index
        reports[label] = _write_html(directory, capture)
        expected.append({"label": label, "data": capture})
    page = write_quadruped_gallery(tmp_path, reports).read_text()
    payload = json.loads(page.split(MARKER, 1)[1].split("</script>", 1)[0])
    assert payload == {"captures": expected}
    assert "Home </script>" not in page
    script = page.split('<script id="player">', 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in page
    assert "connect-src 'none'" in page
    assert "__CAPTURE_DATA__" not in page and "__SCRIPT_SHA256__" not in page


@pytest.mark.parametrize(
    "change",
    [
        {"videos": {"observer": "https://example.invalid/video.mp4"}},
        {"videos": {}},
        {"trajectory": []},
        {"trajectory": None},
        {"frames": True},
    ],
)
def test_gallery_rejects_incomplete_or_external_recordings(tmp_path, change):
    capture = _capture() | change
    source = _write_html(tmp_path, capture)
    with pytest.raises(MarbleError):
        write_quadruped_gallery(tmp_path, {"Warehouse": source})


def test_gallery_requires_native_reports_and_named_situations(tmp_path):
    source = tmp_path / "source.html"
    source.write_text("<html>Not a recorded capture</html>")
    for reports in ({}, {" ": source}, {"Warehouse": source}):
        with pytest.raises(MarbleError):
            write_quadruped_gallery(tmp_path, reports)
