"""Build an offline review dashboard exclusively from completed native training and evaluation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .public_vla_data import PINS, write_json
from .public_vla_export import file_sha256


def build_report(root: Path, output: Path) -> dict:
    """Export measured learning curves, native rollout videos and reproducible provenance.

    Args:
        root: Completed pipeline directory.
        output: Public-safe HTML report directory.
    Returns:
        Evidence embedded in the report; no runtime identifiers or private paths.
    Raises:
        ValueError: Training or evaluation evidence is incomplete.
        OSError: Required artifacts cannot be read or copied.
    """
    completed = _read(root / "completed.json")
    if _read(root / "inputs.json") != PINS:
        raise ValueError("report export requires the exact public input revisions")
    if (
        completed.get("training_executed") is not True
        or completed.get("evaluation_executed") is not True
    ):
        raise ValueError("a completed real training and evaluation run is required")
    output.mkdir(parents=True, exist_ok=True)
    evidence = _evidence(root, output)
    write_json(output / "evidence.json", evidence)
    _write_html(output, evidence)
    _write_attribution(output)
    write_json(
        output / "checksums.json",
        {
            str(p.relative_to(output)): file_sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "checksums.json"
        },
    )
    return evidence


def _evidence(root, output):
    corpus = _read(root / "corpus.json")
    records = corpus["episodes"]
    return {
        "schema": "npa.public-vla.report.v1",
        "inputs": PINS,
        "task": _read(root / "task.json"),
        "recipe": _read(root / "recipe.json"),
        "corpus": {
            "episodes": len(records),
            "frames": sum(r["frames"] for r in records),
            "partitions": dict(Counter(r["partition"] for r in records)),
        },
        "training": {
            phase: _training(root, phase) for phase in ("generalist", "specialist")
        },
        "evaluation": {
            phase: _evaluation(root, output, phase)
            for phase in ("baseline", "generalist", "specialist", "test")
        },
        "selection": _read(root / "selection.json"),
        "rendering": _read(root / "rendering.json"),
        "playback": {"control_hz": 20, "native_video_fps": 80, "rate": 0.25},
        "runtime": _runtime(root),
        "execution": _read(root / "execution.json"),
        "export_verification": _read(root / "export-verification.json"),
        "recovery": _resume_proof(root),
        "training_executed": True,
        "evaluation_executed": True,
    }


def _read(path):
    return json.loads(path.read_text())


def _resume_proof(root):
    path = root / "resume-verification.json"
    if not path.is_file():
        return {"interruption_tested": False}
    proof = _read(path)
    fields = (
        "saved_step",
        "checkpoint_sha256",
        "recovery_roles",
        "step_before_interruption",
        "first_resumed_step",
        "resumed_past_interruption",
        "compared_updates",
        "maximum_replayed_loss_difference",
        "latest_resumed_step",
    )
    return {"interruption_tested": True} | {key: proof[key] for key in fields}


def _runtime(root):
    actual = _read(root / "evidence/generalist/runtime-rank-0.json")
    fields = (
        "gpu",
        "torch",
        "parameters",
        "trainable_parameters",
        "world_size",
        "rank",
    )
    return {key: actual[key] for key in fields}


def _training(root, phase):
    metrics = [
        json.loads(line)
        for line in (root / "evidence" / phase / "metrics-rank-0.jsonl")
        .read_text()
        .splitlines()
    ]
    plan = _read(root / "plans" / f"{phase}.json")
    # A resumed run can repeat updates after its last durable checkpoint.
    unique = {row["step"]: row for row in metrics}
    fields = (
        "step",
        "rank",
        "unix_time",
        "loss",
        "grad_norm",
        "lr",
        "update_s",
        "dataloading_s",
        "gpu_mem_gb",
    )
    metrics = [
        {key: unique[step][key] for key in fields if key in unique[step]}
        for step in sorted(unique)
    ]
    if not metrics or metrics[-1]["step"] != plan["steps"]:
        raise ValueError("optimizer evidence does not reach the planned final step")
    checkpoint = (
        root
        / phase
        / "checkpoints"
        / f"{plan['steps']:06d}"
        / "pretrained_model/model.safetensors"
    )
    return {
        "steps": plan["steps"],
        "frames": plan["frames"],
        "episodes": len(plan["episodes"]),
        "metrics": metrics,
        "weights_sha256": file_sha256(checkpoint),
        "stats_sha256": plan["stats_sha256"],
    }


def _evaluation(root, output, phase):
    source = root / "evaluation" / phase
    info = _read(source / "eval_info.json")
    videos = []
    for index, path in enumerate(sorted((source / "videos").rglob("*.mp4"))):
        name = f"{phase}-{index:02d}.mp4"
        shutil.copy2(path, output / name)
        videos.append(name)
    if len(videos) != min(10, info["overall"]["n_episodes"]):
        raise ValueError(
            "a measured simulator evaluation needs its native rollout video set"
        )
    measured = {
        key: info["overall"][key] for key in ("pc_success", "n_episodes", "eval_s")
    }
    successes = info["per_task"][0]["metrics"]["successes"]
    if len(info["per_task"]) != 1 or len(successes) != measured["n_episodes"]:
        raise ValueError("expected complete per-episode results for one task")
    return measured | {"videos": videos, "successes": successes}


def _write_html(output, evidence):
    template = Path(__file__).with_name("public_vla_report.html").read_text()
    encoded = json.dumps(evidence, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__RUN_EVIDENCE__", encoded))


def _write_attribution(output):
    text = (
        "Public VLA training reference\n\n"
        "Policy: HuggingFaceVLA/smolvla_libero (Apache-2.0).\n"
        "Backbone: HuggingFaceTB/SmolVLM2-500M-Instruct (Apache-2.0).\n"
        "Demonstrations: lerobot/libero (Apache-2.0).\n"
        "Training: Hugging Face LeRobot 0.6.0. Evaluation: native LIBERO/MuJoCo.\n"
        "LIBERO runtime assets: lerobot/libero-assets; retain upstream component terms.\n"
        "No raw simulation assets, customer data, credentials, or infrastructure identifiers are included.\n\n"
        "The initial policy was already trained on LIBERO. These results measure continued training and "
        "task adaptation, not from-scratch pretraining or generalization to an unseen benchmark.\n"
        "Validation initial states and final test initial states are disjoint within this run.\n"
        "All videos are actual native simulator rollouts. Curves contain recorded optimizer loss.\n"
        "See evidence.json for exact upstream revisions and measured results.\n"
    )
    (output / "attribution.txt").write_text(text)
    shutil.copy2(
        Path(__file__).with_name("apache-2.0.txt"), output / "license-apache-2.0.txt"
    )


def main() -> None:
    """Build an offline report from a complete local run.

    Args:
        None; process arguments provide input and output directories.
    Returns:
        None.
    Raises:
        ValueError: Required run evidence is incomplete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
