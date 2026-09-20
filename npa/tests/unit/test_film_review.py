"""Exercise final-frame sampling and fail-closed narration/picture assessments."""

import copy
import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def review(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "npa/src/npa/studio_renderer"
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("film_review")


@pytest.fixture
def media(tmp_path):
    video = tmp_path / "film.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x90:r=30:d=1",
        "-f", "lavfi", "-i", "color=blue:s=160x90:r=30:d=1", "-f", "lavfi", "-i", "sine=duration=2",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video),
    ], check=True)
    captions = video.with_suffix(".srt")
    captions.write_text("1\n00:00:00,100 --> 00:00:01,900\nCompare <red> & blue.\n")
    shots = tmp_path / "edit.json"
    shots.write_text(json.dumps({"shots": [
        {"id": "red", "timeline_start": 0, "timeline_end": 1, "label": "Red", "private_evidence": "excluded"},
        {"id": "blue", "timeline_start": 1, "timeline_end": 2, "label": "Blue"},
    ]}))
    return video, captions, shots


def _complete(review, packet, verdict="aligned"):
    assessment = review._pending_assessment(packet)
    assessment["reviewer"] = {"name": "Local reviewer", "kind": "human", "method": "sampled frames"}
    for cue in assessment["cues"]:
        cue.update(verdict=verdict, visible_content="A red field followed by blue.",
                   reasoning="The color comparison is visible in the sampled frames.")
    return assessment


def test_offline_packet_samples_encoded_film_and_has_audible_cue_playback(review, media, tmp_path):
    packet, directory = review._prepare(*media, tmp_path / "review")
    cue = packet["cues"][0]
    assert cue["text"] == "Compare <red> & blue."
    assert {shot["id"] for shot in cue["shots"]} == {"red", "blue"}
    assert len(cue["frames"]) == 5
    assert all("private_evidence" not in shot for shot in cue["shots"])
    first = Image.open(directory / cue["frames"][0]["file"]).getpixel((50, 50))
    last = Image.open(directory / cue["frames"][-1]["file"]).getpixel((50, 50))
    assert first[0] > 200 and first[2] < 20
    assert last[2] > 200 and last[0] < 20
    assert review._hash(directory / "film.mp4") == review._hash(media[0])
    for frame in cue["frames"]:
        assert review._hash(directory / frame["file"]) == frame["sha256"]
    report = review._save_review(packet, review._pending_assessment(packet), directory)
    assert report["status"] == "pending" and report["pending_cues"] == ["cue-001"]
    page = (directory / "review.html").read_text()
    assert "Compare &lt;red&gt; &amp; blue." in page
    assert 'data-start="0.1" data-end="1.9"' in page
    assert 'src="film.mp4" controls' in page and "muted" not in page
    assert "continuous audiovisual viewing" in page


def test_changed_captions_invalidate_previous_judgment(review, media, tmp_path):
    packet, _ = review._prepare(*media, tmp_path / "review")
    assessment = _complete(review, packet)
    media[1].write_text(media[1].read_text().replace("Compare", "Inspect"))
    changed, _ = review._prepare(*media, tmp_path / "review")
    assert packet["packet_sha256"] != changed["packet_sha256"]
    with pytest.raises(ValueError, match="stale"):
        review._assess(changed, assessment)


@pytest.fixture
def packet():
    return {"packet_sha256": "packet", "video_sha256": "film", "scope": "sampled frames",
            "cues": [{"id": "cue-001"}, {"id": "cue-002"}]}


def test_summary_preserves_illustrative_assessment_order(review):
    cue_ids = [f"cue-{index:03d}" for index in range(1, 7)]
    packet = {"packet_sha256": "packet", "video_sha256": "film", "scope": "frames",
              "cues": [{"id": identifier} for identifier in cue_ids]}
    assessment = _complete(review, packet)
    by_id = {item["id"]: item for item in assessment["cues"]}
    verdicts = [("cue-006", "aligned"), ("cue-005", "illustrative"),
                ("cue-003", "illustrative"), ("cue-001", "mismatch"),
                ("cue-004", "uncertain"), ("cue-002", "not_reviewed")]
    assessment["cues"] = [{**by_id[identifier], "verdict": verdict}
                          for identifier, verdict in verdicts]
    report = review._assess(packet, assessment)
    assert report["illustrative_cues"] == ["cue-005", "cue-003"]
    assert report["problem_cues"] == ["cue-001", "cue-004"]
    assert report["pending_cues"] == ["cue-002"] and report["status"] == "needs_revision"


@pytest.mark.parametrize("verdict,status", [
    ("aligned", "reviewed"), ("illustrative", "reviewed"),
    ("mismatch", "needs_revision"), ("uncertain", "needs_revision"), ("not_reviewed", "pending"),
])
def test_timing_never_substitutes_for_explicit_judgment(review, packet, verdict, status):
    report = review._assess(packet, _complete(review, packet, verdict))
    assert report["status"] == status
    expected = ["cue-001", "cue-002"] if verdict == "illustrative" else []
    assert report["illustrative_cues"] == expected
    assert report["timing_is_not_semantic_evidence"] is True


@pytest.mark.parametrize("defect", ["missing", "duplicate", "extra", "stale", "reviewer", "reasoning", "verdict"])
def test_incomplete_or_unbound_assessments_cannot_pass(review, packet, defect):
    assessment = _complete(review, packet)
    if defect == "missing":
        assessment["cues"].pop()
    elif defect == "duplicate":
        assessment["cues"][1] = copy.deepcopy(assessment["cues"][0])
    elif defect == "extra":
        assessment["cues"].append({**assessment["cues"][0], "id": "unknown"})
    elif defect == "stale":
        assessment["packet_sha256"] = "older-film"
    elif defect == "reviewer":
        assessment["reviewer"]["name"] = ""
    else:
        assessment["cues"][0][defect] = ""
    with pytest.raises(ValueError):
        review._assess(packet, assessment)


@pytest.mark.parametrize("finish_reason,content", [
    ("length", '{"verdict":"aligned"}'), ("stop", "not JSON"), ("stop", '{"verdict":"pass"}'),
])
def test_partial_or_invalid_model_output_cannot_pass(review, finish_reason, content):
    judge = importlib.import_module("film_review_judge")
    with pytest.raises(ValueError):
        judge._model_judgment({"choices": [{"finish_reason": finish_reason,
                                           "message": {"content": content}}]}, "cue-001")


@pytest.mark.parametrize("start,end", [(0.1, 2), (0, 1), (0, float("nan"))])
def test_shot_context_cannot_hide_gaps_or_truncated_timeline(review, tmp_path, start, end):
    path = tmp_path / "edit.json"
    path.write_text(json.dumps({"shots": [{"id": "one", "timeline_start": start, "timeline_end": end}]}))
    with pytest.raises(ValueError):
        review._shots(path, 2)


def _options(tmp_path, media, **overrides):
    project = tmp_path / "project.json"
    project.write_text(json.dumps({"storyboard": "story.json", "assets": "assets.json",
                                   "voice_dir": "narration", "output_dir": "renders"}))
    values = dict(project=project, video=media[0], captions=media[1], shot_list=media[2],
                  output_dir=tmp_path / "review", assessment=None, judge=None,
                  model="vision-model", strict=True, open=False)
    return SimpleNamespace(**{**values, **overrides})


def test_strict_default_is_pending_and_does_not_call_a_model(review, media, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "_arguments", lambda: _options(tmp_path, media))
    monkeypatch.setattr(review, "_judge", lambda *args: pytest.fail("Offline review must not call inference"))
    with pytest.raises(SystemExit) as stopped:
        review._main()
    assert stopped.value.code == 1


def test_strict_accepts_illustrative_and_serializes_exact_summary(
    review, media, tmp_path, monkeypatch, capsys,
):
    media[1].write_text(
        "1\n00:00:00,100 --> 00:00:00,600\nFirst.\n\n"
        "2\n00:00:00,700 --> 00:00:01,200\nSecond.\n\n"
        "3\n00:00:01,300 --> 00:00:01,900\nThird.\n"
    )
    monkeypatch.setattr(review, "_arguments", lambda: _options(tmp_path, media))

    def reordered_assessment(args, packet, directory):
        assessment = _complete(review, packet, "illustrative")
        by_id = {item["id"]: item for item in assessment["cues"]}
        by_id["cue-001"]["verdict"] = "aligned"
        assessment["cues"] = [by_id[identifier] for identifier in
                              ("cue-003", "cue-001", "cue-002")]
        return assessment

    monkeypatch.setattr(
        review, "_selected_assessment", reordered_assessment,
    )
    review._main()
    summary = json.loads(capsys.readouterr().out)
    saved = json.loads((Path(summary["review_directory"]) / "review.json").read_text())
    assert summary["illustrative_cues"] == ["cue-003", "cue-002"]
    assert saved["illustrative_cues"] == ["cue-003", "cue-002"]
    assert summary["status"] == saved["status"] == "reviewed"


def test_film_changed_during_model_review_keeps_report_pending(review, media, tmp_path, monkeypatch):
    monkeypatch.setattr(review, "_arguments", lambda: _options(tmp_path, media, judge="token-factory"))

    def change_captions(packet, directory, model):
        media[1].write_text(media[1].read_text().replace("Compare", "Inspect"))
        return _complete(review, packet)

    monkeypatch.setattr(review, "_judge", change_captions)
    with pytest.raises(ValueError, match="changed during model review"):
        review._main()
    report = json.loads(next((tmp_path / "review").glob("*/review.json")).read_text())
    assert report["status"] == "pending"


def test_selected_film_dispatches_review_options(review, tmp_path):
    studio = importlib.import_module("studio")
    project = tmp_path / "project.json"
    command = studio._command(project, "review", ["--strict", "--open"])
    assert command[1].endswith("film_review.py")
    assert command[2:] == ["--project", str(project), "--strict", "--open"]


def test_reopening_same_film_retains_its_completed_assessment(review, media, tmp_path, monkeypatch):
    packet, directory = review._prepare(*media, tmp_path / "review")
    report = review._save_review(packet, _complete(review, packet), directory)
    monkeypatch.setattr(review, "_arguments", lambda: _options(tmp_path, media))
    monkeypatch.setattr(review, "_judge", lambda *args: pytest.fail("Reopening must reuse the existing review"))
    review._main()
    assert json.loads((directory / "review.json").read_text()) == report


def test_hosted_review_preserves_served_model_usage_and_all_judgments(review, packet, tmp_path, monkeypatch):
    judge = importlib.import_module("film_review_judge")
    response = {"model": "served-vision-model", "usage": {"total_tokens": 20}, "choices": [
        {"finish_reason": "stop", "message": {"content": json.dumps({
            "verdict": "illustrative", "visible_content": "A diagram.", "reasoning": "It explains the layout.",
        })}},
    ]}
    calls = []

    def complete(**options):
        calls.append(options)
        return response

    client = SimpleNamespace(list_models=lambda: ["vision-model"], chat_completion=complete)
    monkeypatch.setattr(judge, "_messages", lambda cue, directory: [{"role": "user", "content": cue["id"]}])
    assessment = judge._request_judgments(packet, tmp_path, "vision-model", client)
    assert len(calls) == len(packet["cues"])
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
    assert review._assess(packet, assessment)["status"] == "reviewed"
    for cue in packet["cues"]:
        receipt = json.loads((tmp_path / f"{cue['id']}-model.json").read_text())
        assert receipt["model"] == "served-vision-model" and receipt["usage"]["total_tokens"] == 20
        assert receipt["rubric_sha256"] == review._hash(tmp_path / "review-instructions.txt")


def test_unavailable_model_does_not_send_frames(review, packet, tmp_path):
    judge = importlib.import_module("film_review_judge")
    client = SimpleNamespace(list_models=lambda: [], chat_completion=lambda **options: pytest.fail("Unavailable model"))
    with pytest.raises(ValueError, match="not available"):
        judge._request_judgments(packet, tmp_path, "vision-model", client)
