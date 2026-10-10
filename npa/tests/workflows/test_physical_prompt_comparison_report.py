"""Verify standalone media integrity, paired coverage, and safe report rendering."""

import base64
import hashlib
from html.parser import HTMLParser

import pytest

from npa.workflows.physical_prompt_comparison_contract import ARMS, SOLUTION
from npa.workflows.physical_prompt_comparison_report import write_gallery


class _Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def _receipt(arm, digest):
    return {
        "case_id": "test-case",
        "seed": 0,
        "arm": arm,
        "prompt": '<script>alert("prompt")</script>',
        "negative_prompt": "",
        "observed": {"sha256": digest},
        "runtime": {"devices": [{"name": "Synthetic test GPU"}]},
        "requested": {
            "model_id": "test-model",
            "revision": "test-revision",
            "width": 1280,
            "height": 720,
            "frames": 81,
            "fps": 16,
            "steps": 50,
        },
    }


def _judgment(arm, digest):
    return {
        "case_id": "test-case",
        "seed": 0,
        "arm": arm,
        "score": 0.0,
        "video_sha256": digest,
        "assertions": [
            {"index": 0, "verdict": "unknown", "evidence": "<script>bad()</script>"}
        ],
    }


@pytest.fixture
def comparison(tmp_path):
    case = {
        "id": "test-case",
        "prompt": '<img src="https://invalid.test/image">',
        "assertions": ["No disappearing objects."],
    }
    recipe = {
        "schema": "npa.physical-prompt-comparison.recipe.v1",
        "solution": SOLUTION,
        "arms": list(ARMS),
        "seeds": [0],
        "evaluation_frames": 16,
        "cases": [case],
        "run_id": "private-test-run-not-for-display",
    }
    rows, judgments = [], []
    for arm in ARMS:
        video = tmp_path / f"{arm}.mp4"
        video.write_bytes(f"synthetic-unit-fixture-{arm}".encode())
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        rows.append((_receipt(arm, digest), video))
        judgments.append(_judgment(arm, digest))
    report = {
        "judgments": judgments,
        "judge_model": "test-judge",
        "summary": {"arm_mean_scores": dict.fromkeys(ARMS, 0.0)},
    }
    output = tmp_path / "report"
    output.mkdir()
    return output, recipe, rows, report


def test_report_embeds_original_media_and_escapes_untrusted_text(comparison):
    output, recipe, rows, report = comparison
    write_gallery(*comparison)
    text = (output / "index.html").read_text()
    elements = _Document(text).elements
    videos = [attrs for tag, attrs in elements if tag == "video"]
    assert len(videos) == 3
    assert [row["arm"] for row, _ in rows] == list(ARMS)
    for attrs, (row, video) in zip(videos, rows, strict=True):
        prefix, encoded = attrs["src"].split(",", 1)
        assert prefix == "data:video/mp4;base64"
        assert base64.b64decode(encoded, validate=True) == video.read_bytes()
        assert attrs["data-sha256"] == row["observed"]["sha256"]
        assert (
            output / "videos" / f"test-case-seed-0-{row['arm']}.mp4"
        ).read_bytes() == video.read_bytes()
    assert not any(tag in {"script", "img", "iframe", "link"} for tag, _ in elements)
    assert all(
        attrs["href"].startswith("#") for _, attrs in elements if "href" in attrs
    )
    assert "&lt;script&gt;" in text and "&lt;img" in text
    assert recipe["run_id"] not in text
    assert "unknown" in text and "Unknown" in text


@pytest.mark.parametrize("changed", ["payload", "generation", "judgment"])
def test_report_rejects_media_receipt_mismatch(comparison, changed):
    _, _, rows, report = comparison
    if changed == "payload":
        rows[0][1].write_bytes(b"tampered")
    elif changed == "generation":
        rows[0][0]["observed"]["sha256"] = "0" * 64
    else:
        report["judgments"][0]["video_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from.*receipt"):
        write_gallery(*comparison)


@pytest.mark.parametrize("target", ["videos", "judgments"])
@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_report_rejects_incomplete_or_duplicate_pairs(comparison, target, change):
    _, _, rows, report = comparison
    collection = rows if target == "videos" else report["judgments"]
    if change == "missing":
        collection.pop()
    else:
        collection.append(collection[0])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        write_gallery(*comparison)
