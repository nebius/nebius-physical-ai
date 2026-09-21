"""Turn automatically routed scene plans into real robot trajectories and LeRobot data."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field

from npa.clients.token_factory import TokenFactoryClient
from npa.workbench.token_factory import TokenFactoryToolError
from .sdg import SdgRequest, MODEL_CARDS, _jev_key, _jsonl, _read_text, _seeds, _summary
from .robot_artifacts import (
    export_robot_dataset,
    publish_robot_run,
    robot_artifact_hashes,
    write_robot_gallery,
)
from .robot_scene import ROBOT_PROMPT_REVISION, plan_robot_scene
from .robot_sim import FPS, runtime_versions, simulate_robot_episode


class RobotSdgRequest(SdgRequest):
    """Configure scene planning, real simulation, and LeRobot episode publication.

    Args:
        input_path: JSONL id/prompt scene seeds; local SDK path or S3 URI.
        output_path: New local directory or empty S3 run prefix.
        router: Hosted Token Factory classifier, or optional Jev.
        seed: Initial simulation seed, incremented for each input record.
        dry_run: Validate input without inference, simulation, or output writes.
    Returns:
        Validated robot SDG request.
    Raises:
        ValueError: Required paths or simulation seed are invalid.
    """

    seed: int = Field(default=0, ge=0)
    context_path: str = Field(default="", max_length=0)


@contextmanager
def _output_directory(output_path):
    if output_path.startswith("s3://"):
        with TemporaryDirectory(prefix="npa-robot-sdg-") as temporary:
            yield Path(temporary)
    else:
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=False)
        yield path


def _run_episode(root, client, seed, index, request, key):
    record = plan_robot_scene(client, seed, router=request.router, jev_key=key)
    if record["status"] == "error":
        return record
    relative = f"episodes/episode_{index:04d}"
    result = simulate_robot_episode(
        record["scene"], seed=request.seed + index, output=root / relative
    )
    return {
        **record,
        "simulation": result,
        "episode_path": relative,
        "simulation_seed": request.seed + index,
        "status": "accepted" if result["accepted"] else "rejected",
        "reason": "physics_checks_passed"
        if result["accepted"]
        else "physics_checks_failed",
    }


def _finish_run(root, records, seed_body, request, versions):
    accepted = export_robot_dataset(root, records)
    (root / "seeds.jsonl").write_text(seed_body)
    (root / "provenance.jsonl").write_text(_jsonl(records))
    (root / "rejected.jsonl").write_text(
        _jsonl([record for record in records if record["status"] != "accepted"])
    )
    write_robot_gallery(root, records)
    report = _summary(request, records, seed_body, "")
    report.update(
        schema="npa.token_factory.robot_sdg.v1",
        prompt_revision=ROBOT_PROMPT_REVISION,
        dataset_format="LeRobotDataset v3.0",
        dataset_path="dataset" if accepted else None,
        total_frames=sum(
            record["simulation"]["frames"]
            for record in records
            if record["status"] == "accepted"
        ),
        simulator_versions=versions,
        fps=FPS,
        success_judge="mujoco_state_and_contacts",
        policy_training_performed=False,
        physical_robot_execution=False,
        artifacts=robot_artifact_hashes(root),
    )
    publish_robot_run(root, request.output_path, report)
    return report


def run_robot_sdg(request: RobotSdgRequest, *, client=None):
    """Generate and filter camera/action robot demonstrations into a LeRobot dataset.

    Args:
        request: Scene seed input, new output destination, router, and RNG seed.
        client: Optional injected Token Factory client.
    Returns:
        Manifest binding real MP4s, physics traces, LeRobot data and inference usage.
    Raises:
        TokenFactoryToolError: Inputs, provider access, or simulation runtime are invalid.
        OSError: Output exists or artifact publication fails.
        RuntimeError: Actual simulation or video rendering fails.
    """
    seed_body = _read_text(request.input_path)
    seeds = _seeds(seed_body)
    if request.dry_run:
        return {
            "status": "planned",
            "seed_count": len(seeds),
            "router": request.router,
            "dataset_format": "LeRobotDataset v3.0",
            "inference_performed": False,
        }
    versions = runtime_versions()
    key = _jev_key(request.router)
    with _output_directory(request.output_path) as root:
        active = client or TokenFactoryClient()
        if not set(MODEL_CARDS) <= set(active.list_models()):
            raise TokenFactoryToolError(
                "Both robot SDG planning models must be available"
            )
        records = []
        for index, seed in enumerate(seeds):
            record = _run_episode(root, active, seed, index, request, key)
            records.append(record)
            (root / "provenance.jsonl").write_text(_jsonl(records))
        return _finish_run(root, records, seed_body, request, versions)
