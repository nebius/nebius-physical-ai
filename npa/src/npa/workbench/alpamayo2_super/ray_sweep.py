"""Execute Alpamayo scenario sweeps with Ray GPU actors and CPU reductions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from npa.clients.storage import StorageClient
from npa.workbench.alpamayo2_super.runtime import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_REVISION,
    DEFAULT_DATASET_REVISION,
    Alpamayo2SuperRequest,
    _publish,
    _resolve_model_snapshot,
    run_inference,
)
from npa.workbench.alpamayo2_super.ray_report import (
    case_grid, case_key, matched_comparisons, select_hard_samples,
    summarize_sample, validate_measurements,
)


@dataclass(frozen=True)
class AlpamayoSweepRequest:
    """Define a reproducible Ray experiment without provisioning infrastructure.

    Args:
        output_path: Report destination, with independent execution subdirectories.
        run_id: Operator-supplied experiment identity.
        sample_indices: Manifest indices for a new baseline.
        seeds: Random seeds for a new baseline; refinement inherits baseline seeds.
        diffusion_steps: Explicit integration-step settings to evaluate.
        workers: GPU actor count within an already allocated Ray cluster.
        manifest: Worker-readable pinned validation manifest.
        ray_address: Explicit application Ray address, or local for an owned runtime.
        input_path: Optional completed baseline report to refine.
        minimum_ade: Refine scenarios whose baseline mean ADE exceeds this value.
    Returns:
        None.
    Raises:
        None.
    """

    output_path: str
    run_id: str
    sample_indices: list[int] = field(default_factory=lambda: [0, 1])
    seeds: list[int] = field(default_factory=lambda: [42, 43])
    diffusion_steps: list[int] = field(default_factory=lambda: [10, 20])
    workers: int = 1
    manifest: str = DEFAULT_MANIFEST
    ray_address: str = "local"
    input_path: str = ""
    minimum_ade: float = 2.0


class _InferenceWorker:
    def __init__(self, output_path: str, manifest: str, run_id: str):
        self.output_path = output_path
        self.manifest = manifest
        self.run_id = run_id
        self.snapshot = ""

    def _resolve(self, request):
        if not self.snapshot:
            self.snapshot = _resolve_model_snapshot(request)
        return self.snapshot

    def infer(self, case: dict) -> dict:
        import ray

        label = "sample-{}-seed-{}-steps-{}".format(*case_key(case))
        print(json.dumps({"event": "alpamayo.case_started", **case}), flush=True)
        started = time.monotonic()
        result = run_inference(
            Alpamayo2SuperRequest(
                output_path=f"{self.output_path.rstrip('/')}/{label}/",
                manifest=self.manifest, run_id=self.run_id, **case,
            ),
            model_resolver=self._resolve,
        )
        measurement = {
            **case, **result["metrics"], "sample": result["sample"],
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": result["artifacts"],
            "ray_node_id": ray.get_runtime_context().get_node_id(),
            "ray_actor_id": str(ray.get_runtime_context().get_actor_id()),
        }
        print(json.dumps({"event": "alpamayo.case_completed", **case,
                          "elapsed_seconds": measurement["elapsed_seconds"],
                          "min_ade_m": measurement["min_ade_m"]}), flush=True)
        return measurement


def dispatch_cases(cases: list[dict], actors: list) -> list[dict]:
    """Keep one inference in flight per actor and collect every case exactly once.

    Args:
        cases: Explicit reproducible experiment cases.
        actors: Ray actors exposing an infer method.
    Returns:
        Validated measurements in deterministic case order.
    Raises:
        ValueError: Actors are absent or returned measurements are invalid.
        ray.exceptions.RayError: An actor or inference task fails.
    """
    import ray

    if cases and not actors:
        raise ValueError("at least one inference actor is required")
    remaining = iter(cases)
    pending = {}
    rows = []
    for actor in actors:
        case = next(remaining, None)
        if case is not None:
            pending[actor.infer.remote(case)] = actor
    while pending:
        completed, _ = ray.wait(list(pending), num_returns=1)
        reference = completed[0]
        actor = pending.pop(reference)
        rows.append(ray.get(reference))
        case = next(remaining, None)
        if case is not None:
            pending[actor.infer.remote(case)] = actor
    return validate_measurements(rows, cases)


def _reduce(rows: list[dict]) -> list[dict]:
    import ray

    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_index"]].append(row)
    reduce_sample = ray.remote(num_cpus=1)(summarize_sample)
    references = [reduce_sample.remote(group) for _, group in sorted(groups.items())]
    return [row for group in ray.get(references) for row in group]


def _read_report(path: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="npa-alpamayo-baseline-") as scratch:
        local = Path(scratch) / "report.json"
        if path.startswith("s3://"):
            StorageClient.from_environment().download_file(path, str(local))
        else:
            local = Path(path).expanduser()
        return json.loads(local.read_text())


def _cases_and_baseline(args) -> tuple[list[dict], dict | None]:
    if not args.input_path:
        return case_grid(args.sample_indices, args.seeds, args.diffusion_steps), None
    baseline = _read_report(args.input_path)
    samples = select_hard_samples(baseline, args.minimum_ade)
    expected = {"model_revision": DEFAULT_MODEL_REVISION, "dataset_revision": DEFAULT_DATASET_REVISION}
    if baseline.get("revisions") != expected:
        raise ValueError("baseline uses different model or dataset revisions")
    seeds = sorted({row["seed"] for row in baseline["cases"]})
    cases = case_grid(samples, seeds, args.diffusion_steps) if samples else []
    return cases, baseline


def _execute_sweep(args, cases: list[dict], execution: str) -> tuple[list[dict], list[dict]]:
    import ray

    if not cases:
        return [], []
    if args.workers < 1:
        raise ValueError("workers must be positive")
    resources = ray.cluster_resources()
    if resources.get("GPU", 0) < args.workers or resources.get("CPU", 0) < 2 * args.workers:
        raise ValueError("Ray cluster lacks the requested GPU/CPU actor capacity")
    worker = ray.remote(num_gpus=1, num_cpus=2, max_restarts=0)(_InferenceWorker)
    actors = [worker.remote(execution, args.manifest, args.run_id) for _ in range(args.workers)]
    try:
        rows = dispatch_cases(cases, actors)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
    return rows, _reduce(rows)


def _publish_report(report: dict, output_path: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="npa-alpamayo-report-") as scratch:
        local = Path(scratch)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        (local / "report.json").write_text(encoded)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        (local / "SHA256SUMS").write_text(f"{digest}  report.json\n")
        return _publish(local, output_path)


@contextmanager
def _ray_connection(address: str):
    import ray

    if ray.is_initialized():
        raise ValueError("run a sweep in a dedicated driver process with its own Ray connection")
    with tempfile.TemporaryDirectory(prefix="npa-alpamayo-ray-") as scratch:
        options = {}
        if address == "local":
            options = {"_temp_dir": scratch, "object_store_memory": 256 * 1024 * 1024,
                       "include_dashboard": False}
        try:
            ray.init(address=address, log_to_driver=True, **options)
            yield
        finally:
            ray.shutdown()


def _report(args, cases, rows, summaries, baseline):
    comparisons = matched_comparisons(baseline["measurements"], rows) if baseline else []
    return {
        "schema": "npa.alpamayo.ray-sweep.v1", "status": "complete",
        "run_id": args.run_id, "cases": cases, "measurements": rows,
        "summaries": summaries, "comparisons": comparisons,
        "revisions": {"model_revision": DEFAULT_MODEL_REVISION, "dataset_revision": DEFAULT_DATASET_REVISION},
        "baseline_uri": args.input_path, "selection_minimum_ade_m": args.minimum_ade if baseline else None,
        "model_lifetime": "one upstream subprocess per case; actor reuses downloaded snapshot",
    }


def run_sweep(args: AlpamayoSweepRequest) -> dict:
    """Run a Ray sweep or refine all baseline scenarios above the error threshold.

    Args:
        args: Experiment inputs, outputs, and explicit Ray connection settings.
    Returns:
        Complete report with exact measurements and published report locations.
    Raises:
        ValueError: Experiment, baseline, capacity, or metrics are invalid.
        RuntimeError: Inference or Ray execution fails.
        OSError: Artifact I/O fails.
    """
    if not args.ray_address or args.ray_address == "auto":
        raise ValueError("select local or an explicit application Ray address")
    case_grid([0], [0], args.diffusion_steps)
    cases, baseline = _cases_and_baseline(args)
    execution = args.output_path.rstrip("/") + "/executions/" + uuid.uuid4().hex
    with _ray_connection(args.ray_address):
        rows, summaries = _execute_sweep(args, cases, execution)
        report = _report(args, cases, rows, summaries, baseline)
        artifacts = _publish_report(report, args.output_path)
        return {**report, "artifacts": artifacts}


def _integers(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the workflow module's command-line contract.

    Args:
        None.
    Returns:
        Parser for Ray sweep and refinement inputs.
    Raises:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--input-path", default="")
    parser.add_argument("--sample-indices", type=_integers, default=[0, 1])
    parser.add_argument("--seeds", type=_integers, default=[42, 43])
    parser.add_argument("--diffusion-steps", type=_integers, default=[10, 20])
    parser.add_argument("--minimum-ade", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--ray-address", default="local")
    parser.add_argument("--run-id", required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_sweep(AlpamayoSweepRequest(**vars(build_parser().parse_args()))), indent=2, allow_nan=False))
