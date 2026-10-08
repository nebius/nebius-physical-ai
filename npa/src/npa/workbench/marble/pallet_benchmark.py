"""Compare real-only and Marble-background pallet training on held-out real images."""

import hashlib
import io
import json
import math
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from npa.cli.path_contract import validate_read_path
from npa.workbench.dataset.storage import uri_join, write_bytes_uri

from .api import MarbleError
from .gpu import _cuda
from .runtime import _materialize, _paths, _publish, validate_result


def _row(image_bytes, boxes):
    return {
        "image_bytes": image_bytes,
        "ann_bboxes": boxes,
        "ann_categories": ["pallet"] * len(boxes),
    }


def _real_rows(root, records):
    return [
        _row((root / record["file"]).read_bytes(), record["boxes"])
        for record in records
    ]


def composite_pallet(background, cutout, randomizer):
    """Composite a training pallet onto a rendered background with exact pixel labels.

    Args: RGB background, RGBA cutout, and seeded Python random generator.
    Returns: PNG bytes and the inserted pallet's XYXY box.
    Raises: ValueError for a fully transparent cutout.
    """
    scale = randomizer.uniform(0.2, 0.45)
    ratio = min(
        background.width * scale / cutout.width,
        background.height * scale / cutout.height,
    )
    cutout = cutout.resize(
        (max(1, round(cutout.width * ratio)), max(1, round(cutout.height * ratio)))
    )
    bounds = cutout.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Pallet cutout is empty")
    x = randomizer.randint(0, background.width - cutout.width)
    y = randomizer.randint(0, background.height - cutout.height)
    canvas = background.convert("RGBA")
    canvas.alpha_composite(cutout, (x, y))
    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue(), [
        x + bounds[0],
        y + bounds[1],
        x + bounds[2],
        y + bounds[3],
    ]


def _synthetic_rows(capture_root, data_root, output_root, source, frames, seed):
    randomizer = random.Random(seed)
    rows = []
    (output_root / "synthetic").mkdir()
    for index in range(frames):
        cutout = randomizer.choice(source["cutouts"])
        with Image.open(capture_root / f"frames/{index:04d}.jpg") as background:
            with Image.open(data_root / cutout["file"]) as foreground:
                payload, box = composite_pallet(background, foreground, randomizer)
        (output_root / f"synthetic/{index:04d}.png").write_bytes(payload)
        rows.append(_row(payload, [box]))
    return rows


def _write_views(path, real, synthetic, evaluation):
    import lancedb
    import pyarrow as pa

    schema = pa.schema(
        [
            ("image_bytes", pa.binary()),
            ("ann_bboxes", pa.list_(pa.list_(pa.float32(), 4))),
            ("ann_categories", pa.list_(pa.string())),
        ]
    )
    augmented = real + synthetic
    # Equal row counts and epochs give both arms equal optimizer-update counts.
    baseline = [real[index % len(real)] for index in range(len(augmented))]
    database = lancedb.connect(str(path))
    for name, rows in (
        ("baseline", baseline),
        ("augmented", augmented),
        ("evaluation", evaluation),
    ):
        database.create_table(name, data=pa.Table.from_pylist(rows, schema=schema))
    return len(augmented)


def _seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _train_arm(request, database, name):
    from npa.workbench.detection_training.schemas import TrainRequest
    from npa.workbench.detection_training.training import train_detector

    trained = train_detector(
        TrainRequest(
            view=name,
            lance_uri=str(database),
            output_uri=uri_join(request.output_path, name),
            label_map={"pallet": 1},
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
        ),
        run_id=f"{request.run_id}-{name}",
    )
    checkpoint = next(
        a
        for a in trained.artifacts
        if a.role == "checkpoint" and a.epoch == request.epochs
    )
    return trained, checkpoint


def _train_and_evaluate(request, database, name, torch):
    from npa.workbench.detection_training.evaluation import evaluate_detector
    from npa.workbench.detection_training.schemas import EvalRequest

    _seed(torch, request.seed)
    started = time.perf_counter()
    output = uri_join(request.output_path, name)
    trained, checkpoint = _train_arm(request, database, name)
    evaluated = evaluate_detector(
        EvalRequest(
            checkpoint_uri=checkpoint.uri,
            eval_view="evaluation",
            lance_uri=str(database),
            output_uri=output,
            label_map={"pallet": 1},
        )
    )
    torch.cuda.synchronize()
    write_bytes_uri(
        uri_join(output, "training.json"), trained.model_dump_json().encode()
    )
    write_bytes_uri(
        uri_join(output, "evaluation.json"), evaluated.model_dump_json().encode()
    )
    return {
        "mAP": evaluated.mAP,
        "mAP_50": evaluated.mAP_50,
        "mAP_75": evaluated.mAP_75,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_epoch": checkpoint.epoch,
    }


def compare_metrics(baseline, augmented):
    """Report signed detection changes without turning negative results into success.

    Args: Actual baseline and augmented evaluation metric dictionaries.
    Returns: Signed AP differences and the observed mAP outcome.
    Raises: MarbleError for undefined, non-finite, or invalid AP metrics.
    """
    keys = ("mAP", "mAP_50", "mAP_75")
    if any(
        not math.isfinite(arm[k]) or not 0 <= arm[k] <= 1
        for arm in (baseline, augmented)
        for k in keys
    ):
        raise MarbleError(
            "Undefined or invalid real-test AP; refusing a claimed improvement"
        )
    delta = {key: augmented[key] - baseline[key] for key in keys}
    return {
        "delta": delta,
        "outcome": "improved" if delta["mAP"] > 0 else "no_improvement",
    }


def _comparison(request, capture, source, samples, arms, torch):
    return {
        "schema_version": "npa.marble.pallet-benchmark.v1",
        "run_id": request.run_id,
        "world_id": capture["world"]["world_id"],
        "generated_this_run": True,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "seed": request.seed,
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "samples_per_epoch_per_arm": samples,
        "optimizer_updates_per_arm": math.ceil(samples / request.batch_size)
        * request.epochs,
        "real_train_images": len(source["train"]),
        "synthetic_images": capture["frames"],
        "real_test_images": len(source["evaluation"]),
        "arms": arms,
        **compare_metrics(**arms),
        "evaluation_pixel_hashes": [r["pixel_sha256"] for r in source["evaluation"]],
        "limitations": "2D alpha compositing, not physical simulation. Labels cover inserted pallets; generated backgrounds may contain unlabeled objects. Single-seed result, not a production acceptance test.",
    }


def _experiment(request, root, capture, source, torch):
    output = root / "output"
    output.mkdir()
    real = _real_rows(root / "data", source["train"])
    evaluation = _real_rows(root / "data", source["evaluation"])
    synthetic = _synthetic_rows(
        root / "capture", root / "data", output, source, capture["frames"], request.seed
    )
    samples = _write_views(root / "lance", real, synthetic, evaluation)
    arms = {
        name: _train_and_evaluate(request, root / "lance", name, torch)
        for name in ("baseline", "augmented")
    }
    result = _comparison(request, capture, source, samples, arms, torch)
    result["files"] = {
        str(p.relative_to(output)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()
        }
        for p in output.rglob("*.png")
    }
    (output / "comparison.json").write_text(
        json.dumps(result, allow_nan=False, indent=2)
    )
    _publish(output, request.output_path)
    return {"run_id": request.run_id, **compare_metrics(**arms)}


def pallet_benchmark(request):
    """Train and evaluate both pallet-detector arms on a required CUDA worker.

    Args: PalletBenchmarkRequest pointing to captures and a validated data snapshot.
    Returns: The measured signed AP changes and experiment identity.
    Raises: MarbleError for absent CUDA, changed data, or an imported/sample world.
    """
    _paths(request)
    validate_read_path(request.dataset_path, tool="marble", allow_hf=False)
    torch = _cuda()
    with tempfile.TemporaryDirectory(prefix="npa-pallet-benchmark-") as directory:
        root = Path(directory)
        (root / "capture").mkdir()
        (root / "data").mkdir()
        capture = _materialize(request.input_path, "result.json", root / "capture")
        validate_result(capture, request.run_id)
        if (
            capture["kind"] != "capture"
            or capture["world"].get("source_kind") != "world-api"
            or not capture["world"].get("generated_this_run")
        ):
            raise MarbleError(
                "Manufacturing benchmark requires an API-generated Marble world"
            )
        source = _materialize(request.dataset_path, "dataset.json", root / "data")
        if source["run_id"] != request.run_id:
            raise MarbleError("Pallet snapshot belongs to a different run")
        return _experiment(request, root, capture, source, torch)
