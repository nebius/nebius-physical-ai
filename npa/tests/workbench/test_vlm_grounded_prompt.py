"""Verify ordinal image anchors and grounded instructions without changing evidence."""

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO

from PIL import Image
import pytest

from npa.workbench import vlm_eval


def _frames() -> list[vlm_eval.SelectedFrame]:
    frames = []
    for index, color in ((9, "blue"), (451, "green")):
        stream = BytesIO()
        Image.new("RGB", (9, 7), color).save(stream, format="PNG")
        frames.append(
            vlm_eval.SelectedFrame(
                f"source-{index:03}.png",
                "image/png",
                stream.getvalue(),
                source_kind="video",
                source_index=index,
                source_count=454,
                source_timestamp_s=index / 30,
            )
        )
    return frames


@pytest.mark.parametrize("selection", ["final", "keyframes", "sequence"])
def test_grounding_instructions_preserve_custom_rubric(selection: str) -> None:
    rubric = "Judge only whether a visible lever is horizontal."
    prompt = vlm_eval._build_prompt(
        task="Inspect the lever.",
        rubric=rubric,
        frame_selection=selection,
        frame_count=2,
    )
    assert f"Rubric: {rubric}" in prompt
    assert vlm_eval.DEFAULT_RUBRIC not in prompt
    assert "exact labels in supplied order: Frame 1, Frame 2." in prompt
    assert "not timestamps or source indices" in prompt
    assert "do not infer objects from the task text" in prompt
    assert "Do not invent timestamps, elapsed time, unseen actions" in prompt
    assert "Distinguish 'not shown' or 'cannot verify' from 'did not happen'" in prompt
    assert "rationale must still be factual and evidence-grounded" in prompt
    assert "only downstream contract" not in prompt


@pytest.mark.parametrize("frame_count", [1, 2])
def test_ordinals_bind_to_each_image_without_relabeling_source(
    frame_count: int,
) -> None:
    frames = _frames()[:frame_count]
    originals = [asdict(frame) for frame in frames]
    content = vlm_eval._openai_content("Judge visible evidence.", frames)
    assert content[0] == {"type": "text", "text": "Judge visible evidence."}
    assert len(content) == 1 + 2 * frame_count
    for ordinal, frame in enumerate(frames, start=1):
        assert content[2 * ordinal - 1] == {"type": "text", "text": f"Frame {ordinal}"}
        image = content[2 * ordinal]
        assert image["type"] == "image_url"
        prefix, encoded = image["image_url"]["url"].split(",", 1)
        assert prefix == f"data:{frame.media_type};base64"
        assert base64.b64decode(encoded, validate=True) == frame.data
        assert frame.label not in str(content)
    assert [asdict(frame) for frame in frames] == originals


@pytest.mark.parametrize(
    ("backend", "model", "temperature", "response_format"),
    [
        ("api", "MiniMaxAI/MiniMax-M3", True, False),
        ("api", "moonshotai/Kimi-K3", False, True),
        ("api", "vendor/explicit-vision", True, True),
        ("self-hosted", "vendor/local-vision", True, True),
    ],
)
def test_grounded_content_preserves_profiles_and_source_evidence(
    backend: str, model: str, temperature: bool, response_format: bool
) -> None:
    frames = _frames()
    prompt = vlm_eval._build_prompt(
        task="Inspect visible motion.",
        rubric=vlm_eval.DEFAULT_RUBRIC,
        frame_selection="sequence",
        frame_count=len(frames),
    )
    request, profile = vlm_eval._openai_request(
        backend=backend, model=model, prompt=prompt, frames=frames
    )
    assert request["model"] == model
    assert ("temperature" in request) is temperature
    assert ("response_format" in request) is response_format
    assert request["messages"][0]["content"] == vlm_eval._openai_content(prompt, frames)
    if backend == "self-hosted":
        assert profile is None
    else:
        assert profile is not None and profile.require_exact_model
    evidence = vlm_eval._build_request_evidence(
        backend=backend,
        model=model,
        prompt=prompt,
        rubric=vlm_eval.DEFAULT_RUBRIC,
        request=request,
        frames=frames,
        frame_selection="sequence",
        max_frames=2,
    )
    assert [item.label for item in evidence.frames] == [item.label for item in frames]
    assert [item.sha256 for item in evidence.frames] == [
        hashlib.sha256(frame.data).hexdigest() for frame in frames
    ]
    assert evidence.prompt_sha256 == hashlib.sha256(prompt.encode()).hexdigest()
    anchors = [part["text"] for part in request["messages"][0]["content"][1::2]]
    assert f"order: {', '.join(anchors)}." in prompt
    assert evidence.request_manifest_sha256 == vlm_eval._sha256_json(
        evidence.request_manifest
    )
    sampling = evidence.request_manifest["sampling"]
    assert sampling["selected_indices"] == [9, 451]
    assert sampling["selected_timestamps_s"] == [9 / 30, 451 / 30]
    assert [item.source_index for item in evidence.frames] == [9, 451]
    assert [item.source_timestamp_s for item in evidence.frames] == [9 / 30, 451 / 30]


def test_reordered_frames_keep_source_labels_and_receive_new_ordinals() -> None:
    frames = list(reversed(_frames()))
    content = vlm_eval._openai_content("Observe these frames.", frames)
    assert frames[0].label == "source-451.png"
    assert content[1]["text"] == "Frame 1"
    assert (
        base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1])
        == frames[0].data
    )
    assert content[3]["text"] == "Frame 2"


def test_paired_judges_share_the_same_ordinal_image_contract() -> None:
    frames = _frames()
    prompt = vlm_eval._build_prompt(
        task="Inspect visible motion.",
        rubric=vlm_eval.DEFAULT_RUBRIC,
        frame_selection="sequence",
        frame_count=len(frames),
    )
    common = vlm_eval._common_hosted_request(prompt=prompt, frames=tuple(frames))
    assert common["messages"][0]["content"] == vlm_eval._openai_content(prompt, frames)
    assert common["temperature"] == 0
    requests = [
        vlm_eval._request_for_model(common, model)
        for model in ("vendor/first-vision", "vendor/second-vision")
    ]
    assert vlm_eval._assert_model_only_request_difference(requests)
    assert requests[0]["messages"] == requests[1]["messages"]


@pytest.mark.parametrize("rubric", [vlm_eval.DEFAULT_RUBRIC, "Inspect the lever."])
def test_paired_output_contract_is_separate_and_bound_for_both_judges(
    rubric: str,
) -> None:
    frames = _frames()
    scalar_prompt = vlm_eval._build_prompt(
        task="Inspect visible motion.",
        rubric=rubric,
        frame_selection="sequence",
        frame_count=len(frames),
    )
    prompt = vlm_eval._comparison_prompt(
        "Inspect visible motion.", rubric, "sequence", len(frames)
    )
    prefix, output_contract = prompt.rsplit("\n", 1)
    assert prefix == scalar_prompt
    assert "Output contract for this paired audit" not in scalar_prompt
    assert "exactly one bare JSON object matching the schema above" in output_contract
    assert "Do not include Markdown fences" in output_contract
    assert (
        "preamble, commentary outside the object, or trailing text" in output_contract
    )
    assert output_contract.endswith("Put all explanation inside the rationale string.")
    common = vlm_eval._common_hosted_request(prompt=prompt, frames=frames)
    requests = [
        vlm_eval._request_for_model(common, model)
        for model in ("vendor/first-vision", "vendor/second-vision")
    ]
    assert vlm_eval._assert_model_only_request_difference(requests)
    for request in requests:
        content = request["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": prompt}
        assert content[1:] == vlm_eval._openai_content(scalar_prompt, frames)[1:]
        evidence = vlm_eval._build_request_evidence(
            backend="api",
            model=request["model"],
            prompt=prompt,
            rubric=rubric,
            request=request,
            frames=frames,
            frame_selection="sequence",
            max_frames=2,
        )
        assert evidence.prompt_sha256 == vlm_eval._sha256_text(prompt)
        assert evidence.prompt_sha256 != vlm_eval._sha256_text(scalar_prompt)
        assert evidence.request_manifest["prompt_sha256"] == evidence.prompt_sha256
        assert evidence.request_manifest_sha256 == vlm_eval._sha256_json(
            evidence.request_manifest
        )
        assert evidence.rubric_sha256 == vlm_eval._sha256_text(rubric)
