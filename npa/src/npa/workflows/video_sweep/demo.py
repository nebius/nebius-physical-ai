"""Export verified sweep results as a private, offline HTML and MP4 demo."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from npa.workflows.video_sweep import (
    artifacts,
    checkpoints,
    execution,
    matrix,
    matrix_view,
    planning,
    publication,
)
from npa.workflows.video_sweep.film import render_movie, transcode

_MODEL_LABELS = {
    "MiniMaxAI/MiniMax-M3": "MiniMax-M3",
    "nvidia/Cosmos3-Super-Reasoner": "Cosmos3 Super Reasoner",
}
_ENGINE_LABELS = {
    "cosmos-transfer2.5": "Cosmos Transfer 2.5",
    "cosmos3-nano": "Cosmos3-Nano",
}


def export_demo(args, output: Path) -> dict:
    """Verify publication or a fully tracked rejection and export actual footage.

    Args:
        args: Stage arguments identifying an existing reviewed, tracked run.
        output: New local directory; existing paths are never overwritten.
    Returns:
        Sanitized summary without source paths, prompts, or service identifiers.
    Raises:
        ValueError: Publication, lineage, or media integrity checks fail.
        OSError: Reading or creating artifacts fails.
    """
    if output.exists():
        raise FileExistsError("Choose a new demo directory")
    plan, report = execution.reviewed(args)
    published = _verify_publication(args, plan, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory)
        summary = _materialize(stage, plan, report)
        summary["published"] = published
        if not published:
            summary["evidence"] = (
                "Verified media, review and lineage receipts; all variants held out, no dataset published"
            )
        render_movie(stage, summary)
        _write_html(stage, summary)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for path in stage.iterdir():
            path.chmod(0o600)
        stage.rename(output)
    return summary


def _verify_publication(args, plan: dict, report: dict) -> bool:
    generated = {row["id"]: row for row in execution._join(args, plan)}
    for row in report["items"]:
        candidate = generated[row["id"]]
        if any(row[key] != candidate[key] for key in ("uri", "sha256", "engine")):
            raise ValueError("Review media differs from the worker receipt")
        if candidate["engine"] == "cosmos3-nano":
            item = next(item for item in plan["items"] if item["id"] == row["id"])
            checkpoints._verify(candidate, item, args.root_uri, plan)
            if any(
                row.get(key) != candidate.get(key)
                for key in ("controls", "reference", "generation_sha256")
            ):
                raise ValueError("Review controls differ from the worker receipt")
    receipt = artifacts.read_json(args.root_uri + "/lineage.json")
    publication._verify_lineage(receipt, report)
    expected = {row["id"]: row["sha256"] for row in report["items"] if row["accepted"]}
    if not expected:
        for name in ("manifest.json", "next-sources.json"):
            if artifacts.exists(args.root_uri + "/dataset/" + name):
                raise ValueError(
                    "An all-rejected review must not have a published dataset"
                )
        return False
    _verify_dataset(args, report, receipt, expected)
    return True


def _verify_dataset(args, report, receipt, expected):
    dataset = artifacts.read_json(args.root_uri + "/dataset/manifest.json")
    actual = {row["id"]: row["sha256"] for row in dataset["clips"]}
    if (
        dataset.get("schema") != "npa.video_sweep.dataset.v1"
        or dataset.get("run_id") != args.run_id
        or dataset.get("review_sha256") != artifacts.digest(report)
        or dataset.get("lineage_sha256") != artifacts.digest(receipt)
        or actual != expected
        or len(dataset["clips"]) != len(expected)
    ):
        raise ValueError("Dataset does not match the accepted review and lineage")
    following = artifacts.read_json(args.root_uri + "/dataset/next-sources.json")
    if (
        following.get("schema") != "npa.video_sweep.sources.v1"
        or following.get("parent_run_id") != args.run_id
        or following.get("clips") != [row["uri"] for row in dataset["clips"]]
    ):
        raise ValueError("Next-run inventory differs from the published dataset")
    with tempfile.TemporaryDirectory() as directory:
        for row in dataset["clips"]:
            artifacts.download(row["uri"], Path(directory) / "clip.mp4", row["sha256"])


def _materialize(stage: Path, plan: dict, report: dict) -> dict:
    sources, candidates, controls = {}, [], []
    reviews = {row["id"]: row for row in report["items"]}
    for item in plan["items"]:
        row = reviews[item["id"]]
        source = row.get("reference", item["source"])
        if source["sha256"] not in sources:
            name = f"source-{len(sources) + 1}"
            sources[source["sha256"]] = _media(stage, name, source)
        candidate = _candidate(stage, item, row, len(candidates) + 1, controls)
        candidate["source"] = sources[source["sha256"]]["name"]
        candidates.append(candidate)
    return {
        "schema": "npa.video_sweep.demo.v1",
        "sources": list(sources.values()),
        "candidates": candidates,
        "controls": controls,
        "generator": _ENGINE_LABELS.get(
            plan.get("generator", "cosmos-transfer2.5"), "Operator-selected generator"
        ),
        "threshold": float(report["threshold"]),
        "accepted": sum(row["accepted"] for row in candidates),
        "samples": int(plan["samples"]),
        "workers": int(plan["workers"]),
        "judge": _MODEL_LABELS.get(plan["reasoner_model"], "Operator-selected model"),
        "evidence": "Verified media, review, lineage and publication receipts",
        "scope": "Component results; workflow completion is not inferred from artifacts.",
        **_matrix_summary(plan),
    }


def _matrix_summary(plan):
    if "sweep" not in plan:
        return {}
    variants = matrix.expand(plan["sweep"])
    sources = len(plan["items"]) // len(variants)
    return {"matrix": matrix.describe(plan["sweep"], sources, plan["workers"])}


def _candidate(stage, item, row, index, controls):
    variant = item["variant"]
    if row["engine"] == "cosmos3-nano":
        from npa.workflows.video_sweep.cosmos3 import validate_variant

        validate_variant(variant)
    else:
        planning._validate_variant(variant)
    result = {
        **_media(stage, f"variant-{index}", row),
        "score": float(row["score"]),
        "passed": row["passed"],
        "accepted": row["accepted"],
        "seed": int(variant["seed"]),
        "engine": _ENGINE_LABELS.get(row["engine"], "Operator-selected generator"),
        "parameters": {
            key: value
            for key, value in variant.items()
            if key not in {"hint", "prompt"}
        },
    }
    if row.get("controls"):
        control = _media(stage, f"control-{index}", row["controls"]["edge"])
        controls.append(control)
        result["control"] = control["name"]
    return result


def _media(stage: Path, name: str, row: dict) -> dict:
    raw = stage / "input.mp4"
    artifacts.download(row["uri"], raw, row["sha256"])
    metadata = transcode(raw, stage / f"{name}.mp4", stage / f"{name}.jpg")
    raw.unlink()
    return {"name": name, "input_sha256": row["sha256"], **metadata}


def _write_html(stage: Path, summary: dict) -> None:
    media = {}
    for row in [
        *summary["sources"],
        *summary["candidates"],
        *summary.get("controls", []),
    ]:
        for extension, mime in (("mp4", "video/mp4"), ("jpg", "image/jpeg")):
            name = row["name"] + "." + extension
            encoded = base64.b64encode((stage / name).read_bytes()).decode("ascii")
            media[name] = f"data:{mime};base64,{encoded}"
    payload = json.dumps({"summary": summary, "media": media}).replace("<", "\\u003c")
    template = Path(__file__).with_name("demo.html").read_text()
    panel = matrix_view.render(summary.get("matrix"), summary["candidates"])
    template = template.replace("__MATRIX_PANEL__", panel)
    (stage / "index.html").write_text(template.replace("__DEMO_DATA__", payload))
