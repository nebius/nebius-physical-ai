"""Generate guarded Cosmos3 variants with complete source-edge conditioning."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from npa.workbench.cosmos.structural_transfer import TransferSettings
from npa.workflows.video_sweep import artifacts, vision

ENGINE = "cosmos3-nano"


def validate_variant(variant: dict) -> None:
    """Validate explicit native sampling controls without loading the model.

    Args:
        variant: An appearance hint or direct prompt plus native controls.
    Returns:
        None.
    Raises:
        ValueError: Fields or control values are invalid.
    """
    fields = {"seed", "control_guidance", "edge_threshold", "guidance", "num_steps"}
    if not isinstance(variant, dict) or set(variant) - {"cfg_normalization"} not in (
        fields | {"hint"},
        fields | {"prompt"},
    ):
        raise ValueError(
            "Cosmos3 variants require one hint or prompt and native controls"
        )
    text = variant.get("prompt", variant.get("hint"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The appearance hint or direct prompt must be nonempty")
    if type(variant["seed"]) is not int or variant["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if type(variant["num_steps"]) is not int or variant["num_steps"] < 1:
        raise ValueError("num_steps must be a positive integer")
    guidance = variant["guidance"]
    if (
        type(guidance) not in (int, float)
        or not math.isfinite(guidance)
        or guidance <= 0
    ):
        raise ValueError("guidance must be finite and positive")
    _settings(variant).validate()


def _settings(variant):
    return TransferSettings(
        control_guidance=variant["control_guidance"],
        edge_threshold=variant["edge_threshold"],
        first_chunk_conditional_frames=0,
        cfg_normalization=variant.get("cfg_normalization", "disabled"),
    )


def generate(item, args, plan) -> dict:
    """Run native Cosmos3 full-video transfer and retain its actual control map.

    Args:
        item: Source, prompt and validated native variant.
        args: Stage arguments containing the run artifact root.
        plan: Immutable plan containing the sampling count.
    Returns:
        Generated video, normalized source and actual edge-control artifacts.
    Raises:
        ValueError: Generation, guardrails or temporal alignment cannot be proved.
        OSError: Source retrieval or artifact publication fails.
    """
    from npa.workbench.cosmos.generate import run_cosmos3_generate
    from npa.workflows.paidf_cosmos3_media import prepare_reference, verify_pair

    validate_variant(item["variant"])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, prepared = root / "source.mp4", root / "prepared.mp4"
        artifacts.download(item["source"]["uri"], source, item["source"]["sha256"])
        normalization = prepare_reference(source, prepared)
        result = run_cosmos3_generate(
            mode="video2video",
            checkpoint="Cosmos3-Nano",
            name="variant",
            prompt=item["prompt"],
            vision_path=str(prepared),
            output_dir=root / "generated",
            seed=item["variant"]["seed"],
            guidance=item["variant"]["guidance"],
            num_steps=item["variant"]["num_steps"],
            no_guardrails=False,
            transfer=_settings(item["variant"]),
        )
        alignment = verify_pair(prepared, Path(result["output_path"]), 24)
        return _publish(item, args, plan, result, prepared, normalization, alignment)


def _media(path, root, samples):
    checksum = artifacts.file_digest(path)
    uri = root + "/" + checksum + path.suffix
    _, metadata = vision.sample_video(path, samples)
    artifacts.upload(path, uri)
    return {"uri": uri, "sha256": checksum, "video": metadata}


def _guarded_transfer(result):
    state = result.get("guardrail_state", {})
    transfer = result.get("structural_transfer", {})
    if state.get("status") != "passed" or state.get("effective") is not True:
        raise ValueError("Cosmos3 must prove effective model guardrails")
    if not all(
        transfer.get(key) is True
        for key in (
            "text_guardrail_passed",
            "video_guardrail_passed",
            "guardrail_postprocessing_applied",
        )
    ):
        raise ValueError("Cosmos3 must prove guarded full-source transfer")
    return transfer


def _publish(item, args, plan, result, prepared, normalization, alignment):
    transfer = _guarded_transfer(result)
    root = args.root_uri + "/candidates/" + item["id"]
    video = _media(Path(result["output_path"]), root, plan["samples"])
    control = _media(
        Path(transfer["control_path"]), root + "/controls", plan["samples"]
    )
    reference = _media(prepared, root + "/reference", plan["samples"])
    evidence = {
        "engine": ENGINE,
        "guardrail_state": result["guardrail_state"],
        "normalization": normalization,
        "temporal_alignment": alignment,
        "input_spec": result["input_spec"],
        "structural_transfer": transfer,
        "source_sha256": item["source"]["sha256"],
        "artifacts": {"video": video, "edge": control, "reference": reference},
    }
    artifacts.write_json(root + "/generation.json", evidence)
    return {
        **video,
        "id": item["id"],
        "engine": ENGINE,
        "seed": item["variant"]["seed"],
        "controls": {"edge": control},
        "reference": reference,
        "generation_sha256": artifacts.digest(evidence),
    }
