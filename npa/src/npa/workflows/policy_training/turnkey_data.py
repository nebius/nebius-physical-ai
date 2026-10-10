"""Fetch public demonstrations, run FiftyOne curation and seal independent partitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from .contracts import digest
from .public_vla_data import PINS, _assign_partitions, prepare_corpus, write_json
from .turnkey_store import inherit, materialize, read, record

DEFAULT_RECIPE = {
    "seed": 42,
    "batch_size": 64,
    "generalist_epochs": 1,
    "specialist_epochs": 40,
    "suite": "libero_spatial",
    "task_id": 0,
    "evaluation_episodes": 10,
    "deployment_state_offset": 30,
    "minimum_success": 0.7,
    "workers": 8,
    "min_brightness": 0.02,
    "max_brightness": 0.98,
    "min_sharpness": 0.00001,
}


def prepare(args, workspace: Path, output: Path) -> None:
    """Fetch revision-pinned public data and decode each episode's preview.

    Args:
        args: Immutable recipe and run identity.
        workspace: Private worker directory.
        output: Sealed preparation output.
    Returns:
        None.
    Raises:
        ValueError: Recipe, trajectory or preview decoding is invalid.
        OSError: Public inputs cannot be fetched.
    """
    recipe = _recipe(args)
    write_json(output / "recipe.json", recipe)
    dataset = _dataset(workspace / "dataset")
    corpus = prepare_corpus(dataset, output / "corpus.json", recipe["seed"])
    _previews(dataset, corpus, output)
    record(
        output,
        "prepare",
        {
            "engine": "public-huggingface-lerobot",
            "pins": PINS,
            "episodes": len(corpus["episodes"]),
            "frames": sum(r["frames"] for r in corpus["episodes"]),
        },
    )


def _recipe(args):
    from .public_vla import _validate_recipe

    incoming = json.loads(args.recipe)
    if not isinstance(incoming, dict) or set(incoming) - set(DEFAULT_RECIPE):
        raise ValueError("unknown public policy recipe keys")
    recipe = DEFAULT_RECIPE | incoming
    _validate_recipe(
        {
            k: v
            for k, v in recipe.items()
            if k
            not in {
                "min_brightness",
                "max_brightness",
                "min_sharpness",
                "deployment_state_offset",
            }
        }
    )
    if recipe["evaluation_episodes"] > 16:
        raise ValueError(
            "three disjoint LIBERO initial-state sets require at most 16 episodes"
        )
    _deployment_partition(recipe)
    from .data import _filter
    import fiftyone as fo

    _filter(
        fo,
        {k: recipe[k] for k in ("min_brightness", "max_brightness", "min_sharpness")}
        | {"min_detections": 0},
    )
    return recipe | {
        "run_id": args.run_id,
        "workflow_sha256": _workflow_identity(args.workflow_sha256),
        "pins": PINS,
        "training_kind": "continued-pretraining",
        "scheduler": "kubernetes",
    }


def _deployment_partition(recipe):
    # LIBERO supplies 50 initial states. Keep the deployment set disjoint from
    # both promotion sets, and reject an offset that would silently wrap.
    offset, trials = recipe["deployment_state_offset"], recipe["evaluation_episodes"]
    if type(offset) is not int or offset < 2 * trials or offset + trials > 50:
        raise ValueError("deployment states must be disjoint and fit the LIBERO set")


def _workflow_identity(explicit):
    import re

    runtime = os.environ.get("NPA_WORKFLOW_SHA256", "")
    identity = explicit or runtime
    if not re.fullmatch(r"[a-f0-9]{64}", identity) or (
        explicit and runtime and explicit != runtime
    ):
        raise ValueError("recipe requires the exact runtime workflow SHA-256")
    return identity


def _dataset(path):
    from huggingface_hub import snapshot_download

    snapshot_download(
        PINS["dataset"]["repo"],
        repo_type="dataset",
        token=False,
        revision=PINS["dataset"]["revision"],
        local_dir=path,
    )
    return path


def _previews(dataset, corpus, output):
    import pyarrow.parquet as pq

    info = read(dataset, "meta/info.json")
    rows = pq.read_table(dataset / "meta/episodes").to_pylist()
    lookup = {int(row["episode_index"]): row for row in rows}
    for episode in corpus["episodes"]:
        row = lookup[episode["episode_index"]]
        camera = "observation.images.image"
        video = info["video_path"].format(
            video_key=camera,
            chunk_index=row[f"videos/{camera}/chunk_index"],
            file_index=row[f"videos/{camera}/file_index"],
        )
        preview = output / "previews" / f"episode-{episode['episode_index']:04d}.png"
        preview.parent.mkdir(exist_ok=True)
        source = (dataset / video).resolve()
        if not source.is_relative_to(dataset.resolve()):
            raise ValueError("public video path escapes its verified dataset")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(row[f"videos/{camera}/from_timestamp"]),
                "-i",
                str(source),
                "-frames:v",
                "1",
                str(preview),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        episode["preview"] = preview.relative_to(output).as_posix()
    write_json(output / "corpus.json", corpus)


def curate(args, workspace: Path, output: Path) -> None:
    """Use real FiftyOne queries to flag and select decoded public previews.

    Args:
        args: Preparation input and curation output URIs.
        workspace: Private worker directory.
        output: Curation artifact directory.
    Returns:
        None.
    Raises:
        ValueError: No usable episodes remain.
        ImportError: FiftyOne is missing.
    """
    import fiftyone as fo
    from .data import _filter, _quality

    parent = materialize(args.input_uri, workspace / "prepared")
    inherit(parent, output)
    recipe, corpus = read(parent, "recipe.json"), read(parent, "corpus.json")
    policy = {
        k: recipe[k] for k in ("min_brightness", "max_brightness", "min_sharpness")
    } | {"min_detections": 0}
    dataset = fo.Dataset()
    try:
        _quality_samples(fo, dataset, corpus, parent, workspace, _quality)
        selected = set(dataset.match(_filter(fo, policy)).values("episode_index"))
        measurements = dataset.values(["episode_index", "brightness", "sharpness"])
        _curated_manifest(corpus, selected, output, measurements, policy)
    finally:
        from .diagnostics import _cleanup

        _cleanup(dataset.delete, args.output_uri.rstrip("/") + "-diagnostics/")
    _curation_evidence(output, corpus)


def _curation_evidence(output, corpus):
    record(
        output,
        "curate",
        {
            "engine": "fiftyone",
            "input_count": len(corpus["episodes"]),
            "selected_count": sum(not r["quality_flags"] for r in corpus["episodes"]),
            "scope": "preview-frame",
        },
    )
    # Keep one measured preview per task after FiftyOne has seen every episode.
    tasks, gallery = set(), set()
    for row in corpus["episodes"]:
        if row["quality_flags"] or set(row["tasks"]).issubset(tasks):
            continue
        tasks.update(row["tasks"])
        gallery.add(row["preview"])
    for preview in (output / "previews").glob("*.png"):
        if preview.relative_to(output).as_posix() not in gallery:
            preview.unlink()


def _quality_samples(fo, dataset, corpus, parent, workspace, quality):
    samples = []
    for row in corpus["episodes"]:
        preview = parent / row["preview"]
        brightness, sharpness = quality(
            {"preview_uri": str(preview)}, workspace / "preview.png"
        )
        samples.append(
            fo.Sample(
                filepath=str(preview),
                episode_index=row["episode_index"],
                brightness=brightness,
                sharpness=sharpness,
                detection_count=0,
            )
        )
    dataset.add_samples(samples)


def _curated_manifest(corpus, selected, output, measurements, policy):
    review = []
    for index, brightness, sharpness in zip(*measurements, strict=True):
        review.append(
            {
                "episode_index": index,
                "brightness": brightness,
                "sharpness": sharpness,
                "preview_quality_pass": index in selected,
            }
        )
    for row in corpus["episodes"]:
        if row["episode_index"] not in selected:
            row["quality_flags"].append("preview-quality")
        row["partition"] = "excluded"
    if not any(not r["quality_flags"] for r in corpus["episodes"]):
        raise ValueError("FiftyOne selected no usable episodes")
    write_json(output / "corpus.json", corpus)
    write_json(
        output / "quality.json",
        {
            "engine": "fiftyone",
            "scope": "preview-frame",
            "object_detector_executed": False,
            "inhouse_review": None,
            "policy": policy,
            "review": review,
        },
    )


def split(args, workspace: Path, output: Path) -> None:
    """Seal per-task source groups and two reserved demonstration partitions.

    Args:
        args: Curation input and split output URIs.
        workspace: Private worker directory.
        output: Split artifact directory.
    Returns:
        None.
    Raises:
        ValueError: Selected tasks cannot support independent partitions.
    """
    parent = materialize(args.input_uri, workspace / "curated")
    inherit(parent, output)
    corpus, recipe = read(output, "corpus.json"), read(output, "recipe.json")
    _assign_partitions(corpus["episodes"], recipe["seed"])
    write_json(output / "corpus.json", corpus)
    partitions = {
        name: [r["episode_index"] for r in corpus["episodes"] if r["partition"] == name]
        for name in ("train", "validation", "test", "excluded")
    }
    write_json(
        output / "split.json",
        {"partitions": partitions, "corpus_sha256": digest(corpus)},
    )
    record(
        output,
        "split",
        {
            "engine": "trajectory-group-per-task",
            "partitions": {k: len(v) for k, v in partitions.items()},
        },
    )
