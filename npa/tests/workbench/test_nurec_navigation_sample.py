"""Check sample integrity, public calibration, measured support and honest HTML results."""

import io
import json
from pathlib import Path
import tarfile
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from npa.workbench.nurec import navigation_sample as sample
from npa.workbench.nurec import navigation_sample_cases as cases
from npa.workbench.nurec import navigation_sample_report as report
from npa.workbench.nurec.navigation_assets import sha256
from npa.workbench.nurec.navigation_publication import publish_immutable
from npa.workflows.navigation.artifacts import publish, write_json


def _archive(tmp_path, members):
    archive = tmp_path / "sample.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = 1 if kind == tarfile.REGTYPE else 0
            member.linkname = "../escape" if kind == tarfile.SYMTYPE else ""
            handle.addfile(member, io.BytesIO(b"x") if member.size else None)
    return archive


def test_archive_mismatch_cannot_extract(tmp_path):
    archive = _archive(tmp_path, [(sample.SAMPLE_NAME + "/rgb.txt", tarfile.REGTYPE)])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="pinned SHA-256"):
        sample._extract(archive, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        (sample.SAMPLE_NAME + "/../../escape", tarfile.REGTYPE),
        ("/absolute/escape", tarfile.REGTYPE),
        (sample.SAMPLE_NAME + "/symbolic", tarfile.SYMTYPE),
        (sample.SAMPLE_NAME + "/hardlink", tarfile.LNKTYPE),
    ],
)
def test_archive_rejects_unsafe_members(tmp_path, monkeypatch, name, kind):
    archive = _archive(tmp_path, [(name, kind)])
    monkeypatch.setattr(sample, "ARCHIVE_SHA256", sha256(archive))
    with pytest.raises(ValueError, match="escaping|regular"):
        sample._extract(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_archive_rejects_duplicate_paths(tmp_path, monkeypatch):
    member = (sample.SAMPLE_NAME + "/rgb.txt", tarfile.REGTYPE)
    archive = _archive(tmp_path, [member, member])
    monkeypatch.setattr(sample, "ARCHIVE_SHA256", sha256(archive))
    with pytest.raises(ValueError, match="unique"):
        sample._extract(archive, tmp_path / "output")


def test_pose_interpolation_and_timestamp_gaps():
    poses = np.array([[0, 0, 0, 1, 0, 0, 0, 1], [0.04, 2, 0, 1, 0, 0, 1, 0]])
    rotations = Slerp(poses[:, 0], Rotation.from_quat(poses[:, 4:8]))
    matrix = np.array(sample._pose(0.02, poses, rotations))
    np.testing.assert_allclose(matrix[:3, 3], [1, 0, 1])
    np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert sample._association(0.02, poses, np.array([0.021]))[2]
    assert not sample._association(0.02, poses, np.array([0.05]))[2]
    poses[1, 0] = 0.10
    assert not sample._association(0.02, poses, np.array([0.021]))[2]


def test_original_indices_define_holdout_without_renumbering(tmp_path):
    for name in ("rgb.png", "depth.png"):
        (tmp_path / name).write_bytes(name.encode())
    times = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    rgb = [(str(value), "rgb.png") for value in times]
    depth = [(str(value), "depth.png") for value in times]
    poses = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0.04, 0, 0, 1, 0, 0, 0, 1],
            [0.08, 0, 0, 1, 0, 0, 0, 1],
        ]
    )
    frames, excluded = sample._frames(tmp_path, rgb, depth, poses)
    assert not excluded
    assert [row["id"] for row in frames if row["split"] == "validation"] == [
        "frame_000000",
        "frame_000005",
    ]
    assert all(row["rgb_sha256"] == sha256(tmp_path / "rgb.png") for row in frames)


def test_unknown_floor_does_not_become_supported(monkeypatch):
    monkeypatch.setattr(
        cases, "_rays", lambda scene, origins, directions: np.full(len(origins), np.inf)
    )
    supported, _ = cases._support(None, np.array([[0.0, 0.0]]))
    assert not supported.any()
    with pytest.raises(ValueError, match="connected support"):
        cases._free_grid(None)


def test_actual_rotated_reset_footprint_is_checked(monkeypatch):
    observations = []

    def rays(scene, origins, directions):
        observations.extend(origins)
        return np.full(len(origins), 0.6)

    monkeypatch.setattr(cases, "_rays", rays)
    pair = ({"xy_m": [0, 0], "floor_z_m": 0}, {"xy_m": [0, 2]})
    assert cases._supported_pairs(None, [pair]) == [pair]
    np.testing.assert_allclose(observations[1], [0.23, -0.35, 0.6], atol=1e-12)


def _published_reports(tmp_path, passed):
    scan, trained, scored = (tmp_path / name for name in ("scan", "trained", "scored"))
    for root in (scan, trained, scored):
        root.mkdir()
    write_json(
        scan / "reconstruction.json",
        {"integration_frames": 1984, "validation_frames": 501},
    )
    write_json(scan / "capture.json", {"public": "test-fixture"})
    recipe, binding = _training_fixture(scan, trained)
    _evaluation_fixture(trained, scored, recipe, binding, passed)
    destinations = [
        tmp_path / name
        for name in ("scan-published", "train-published", "eval-published")
    ]
    publish_immutable(scan, str(destinations[0]))
    publish(trained, str(destinations[1]))
    publish(scored, str(destinations[2]))
    return [str(path) for path in destinations]


def _training_fixture(scan, trained):
    from npa.workflows.navigation.reference_bundle import _recipe
    from npa.workflows.navigation.reference_scene import cases as reference_cases

    (trained / "scene.usdz").write_bytes(b"synthetic-contract-only-scene")
    recipe = _recipe(
        trained, "registry.example.invalid/isaac@sha256:" + "b" * 64, 500, 3, 2
    )
    recipe.update(reference_cases(2), minimum_success_rate=0.65)
    write_json(trained / "recipe.json", recipe)
    (trained / "policy.pt").write_bytes(b"synthetic-contract-only-checkpoint")
    evidence = trained / "scan"
    evidence.mkdir()
    for name in ("capture.json", "reconstruction.json"):
        shutil.copyfile(scan / name, evidence / name)
    for name in ("provenance.json", "physics_validation.json", "cases.json"):
        write_json(evidence / name, {"synthetic_fixture": name})
    write_json(
        evidence / "cases.json",
        {name: recipe[name] for name in ("train_cases", "eval_cases", "probe")},
    )
    hashes = {path.name: sha256(path) for path in evidence.iterdir()}
    hashes["scene.usdz"] = recipe["scene_sha256"]
    write_json(
        trained / "scan-lineage.json",
        {
            "schema": "npa.navigation.scan_handoff.v1",
            "source_sha256": hashes,
            "recipe_sha256": sha256(trained / "recipe.json"),
        },
    )
    binding = {name: recipe[name] for name in ("task", "image", "adapter_sha256")}
    binding["source_bundle_sha256"] = recipe["source_bundle_sha256"]
    binding.update(
        recipe_sha256=sha256(trained / "recipe.json"),
        checkpoint_sha256=sha256(trained / "policy.pt"),
        runtime={
            "scene_sha256": recipe["scene_sha256"],
            "reference_controller": {"sha256": recipe["reference_controller_sha256"]},
        },
    )
    write_json(
        trained / "training.json",
        {
            **binding,
            "schema": "npa.navigation.training.v1",
            "iterations": 500,
            "heldout_used_for_training": False,
        },
    )
    return recipe, binding


def _evaluation_fixture(trained, scored, recipe, binding, passed):
    from PIL import Image
    from npa.workflows.navigation.contract import Recipe
    from npa.workflows.navigation.runtime import _case_digest

    for name in ("policy.pt", "recipe.json"):
        shutil.copyfile(trained / name, scored / name)
    distance = 0.1 if passed else 3.0
    episodes = [
        {
            "case_id": case["id"],
            "seed": case["seed"],
            "steps": 1,
            "goal_distance_m": distance,
            "success": passed,
        }
        for case in recipe["eval_cases"]
    ]
    write_json(
        scored / "evaluation.json",
        {
            **binding,
            "schema": "npa.navigation.evaluation.v1",
            "success_rate": float(passed),
            "passed": passed,
            "episodes": episodes,
            "evaluation_inputs_sha256": _case_digest(Recipe.model_validate(recipe)),
            "policy_loaded": True,
        },
    )
    rows = [
        {
            "step": step,
            "simulation_seconds": step * 0.1,
            "goal_distance_m": distance if step == 1 else 4.0,
        }
        for step in range(4)
    ]
    write_json(
        scored / "rendered-rollout/frames.json",
        {"robot_index": 0, "renderer": "isaac-replicator-rgb", "frames": rows},
    )
    for row in rows:
        Image.new("RGB", (32, 24), (row["step"] * 50, 10, 70)).save(
            scored / "rendered-rollout" / f"{row['step']:06d}.png"
        )


@pytest.mark.parametrize("passed", [True, False])
def test_report_uses_actual_evaluation_and_preserves_failure(tmp_path, passed):
    scan, training, evaluation = _published_reports(tmp_path, passed)
    output = tmp_path / "report"
    value = report.build_report(scan, evaluation, training, str(output))
    assert value["passed"] is passed
    assert value["success_rate"] == float(passed)
    page = (output / "index.html").read_text()
    assert ("<dd>Passed</dd>" if passed else "<dd>Did not pass</dd>") in page
    assert "unseen buildings" in page
    assert "65%" in page and "80%" not in page
    assert "data:image/jpeg;base64," in page
    assert 'src="rollout.mp4"' not in page and 'href="' not in page
    assert "Step 2" not in page
    assert (output / "evaluation.json").is_file()


def test_report_refuses_a_different_checkpoint(tmp_path):
    scan, training, evaluation = _published_reports(tmp_path, True)
    value = json.loads((Path(evaluation) / "evaluation.json").read_text())
    value["checkpoint_sha256"] = "b" * 64
    write_json(Path(evaluation) / "evaluation.json", value)
    with pytest.raises(ValueError, match="checksum"):
        report.build_report(scan, evaluation, training, str(tmp_path / "report"))


def _reseal(root):
    write_json(
        root / "checksums.json",
        {
            path.relative_to(root).as_posix(): sha256(path)
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        },
    )


@pytest.mark.parametrize(
    "mutation", ["checkpoint-bytes", "cohort", "rate", "gate", "training-scan"]
)
def test_report_checks_relationships_even_when_each_bundle_is_sealed(
    tmp_path, mutation
):
    scan, training, evaluation = _published_reports(tmp_path, True)
    training, evaluation = Path(training), Path(evaluation)
    result = json.loads((evaluation / "evaluation.json").read_text())
    if mutation == "checkpoint-bytes":
        (training / "policy.pt").write_bytes(b"another-checkpoint")
    elif mutation == "cohort":
        result["episodes"][0]["seed"] += 1
    elif mutation == "rate":
        result["success_rate"] = 0.9
    elif mutation == "gate":
        result["passed"] = False
    else:
        write_json(training / "scan/capture.json", {"another": "capture"})
    write_json(evaluation / "evaluation.json", result)
    _reseal(training)
    _reseal(evaluation)
    with pytest.raises(ValueError):
        report.build_report(
            scan, str(evaluation), str(training), str(tmp_path / "report")
        )


def test_incomplete_evaluation_produces_honest_offline_failure_report(tmp_path):
    scan, training, evaluation = _published_reports(tmp_path, True)
    evaluation = Path(evaluation)
    (evaluation / "evaluation.json").rename(evaluation / "evaluation.incomplete.json")
    write_json(evaluation / "failure.json", {"status": "failed"})
    _reseal(evaluation)
    output = tmp_path / "report"
    summary = report.build_report(scan, str(evaluation), training, str(output))
    assert summary["passed"] is False and summary["evaluation_complete"] is False
    page = (output / "index.html").read_text()
    assert "Incomplete evaluation" in page
    assert "data:image/jpeg;base64," not in page
    assert (output / "evaluation.incomplete.json").exists()


def test_native_process_failure_keeps_the_original_failure_and_attempts_html(
    monkeypatch,
):
    import subprocess
    from npa.workflows.navigation import stages

    failure = subprocess.CalledProcessError(1, ["native-fixture"])
    calls = []

    def fail(*_arguments):
        raise failure

    monkeypatch.setattr(stages, "run_stage", fail)
    monkeypatch.setattr(report, "build_report", lambda *args: calls.append(args))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        report.evaluate_report("training", "evaluation", "scan", "reports")
    assert caught.value is failure
    assert calls == [("scan", "evaluation", "training", "reports")]


@pytest.mark.parametrize("report_committed", [False, True])
def test_retry_reuses_bound_native_evaluation_after_report_failure(
    tmp_path, monkeypatch, report_committed
):
    from npa.workflows.navigation import stages

    scan, training, completed = _published_reports(tmp_path, True)
    evaluation, output = tmp_path / "evaluation", tmp_path / "report"
    native_calls = []

    def native(stage, source, destination):
        native_calls.append((stage, source, destination))
        shutil.copytree(completed, destination)
        return json.loads((Path(completed) / "evaluation.json").read_text())

    original = report.publish_report

    def interrupted(directory, destination):
        if report_committed:
            original(directory, destination)
        raise OSError("report transport interrupted")

    monkeypatch.setattr(stages, "run_stage", native)
    monkeypatch.setattr(report, "publish_report", interrupted)
    with pytest.raises(OSError, match="transport interrupted"):
        report.evaluate_report(training, str(evaluation), scan, str(output))
    native_bytes = {p.name: p.read_bytes() for p in evaluation.iterdir() if p.is_file()}
    monkeypatch.setattr(report, "publish_report", original)
    result = report.evaluate_report(training, str(evaluation), scan, str(output))
    assert result["passed"] is True and len(native_calls) == 1
    assert (output / "index.html").is_file()
    assert native_bytes == {
        p.name: p.read_bytes() for p in evaluation.iterdir() if p.is_file()
    }


def test_recovered_losing_evaluation_retains_report_and_quality_failure(
    tmp_path, monkeypatch
):
    from npa.workflows.navigation import stages

    scan, training, evaluation = _published_reports(tmp_path, False)
    monkeypatch.setattr(stages, "run_stage", lambda *args: pytest.fail("reran native"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="below minimum_success_rate"):
            report.evaluate_report(training, evaluation, scan, str(tmp_path / "report"))
    assert "Did not pass" in (tmp_path / "report/index.html").read_text()


@pytest.mark.parametrize(
    "damage", ["missing-seal", "partial", "checkpoint", "recipe", "conflicting-failure"]
)
def test_occupied_native_output_requires_complete_matching_evidence(
    tmp_path, monkeypatch, damage
):
    from npa.workflows.navigation import stages

    scan, training, evaluation = _published_reports(tmp_path, True)
    root = Path(evaluation)
    if damage == "missing-seal":
        (root / "checksums.json").unlink()
    elif damage == "partial":
        (root / "evaluation.json").rename(root / "evaluation.incomplete.json")
        write_json(root / "failure.json", {"status": "failed"})
        _reseal(root)
    elif damage == "checkpoint":
        (root / "policy.pt").write_bytes(b"another-checkpoint")
        _reseal(root)
    elif damage == "recipe":
        recipe = json.loads((root / "recipe.json").read_text())
        recipe["iterations"] += 1
        write_json(root / "recipe.json", recipe)
        _reseal(root)
    else:
        write_json(root / "failure.json", {"status": "failed"})
        _reseal(root)
    monkeypatch.setattr(stages, "run_stage", lambda *args: pytest.fail("reran native"))
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    with pytest.raises((ValueError, OSError)):
        report.evaluate_report(training, evaluation, scan, str(tmp_path / "report"))
    assert not (tmp_path / "report").exists()
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
