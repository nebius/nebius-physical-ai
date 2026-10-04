"""Regression coverage for the Sylvest paired checkpoint comparison workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from npa.workflows import sylvest_oft_mixdata as workflow


def _rollouts(
    checkpoint: str, successes: list[int], protocol_sha256: str = "protocol-hash"
) -> dict[str, object]:
    categories = ["Camera Viewpoints", "Robot Initial States", "Camera Viewpoints"]
    episodes = [
        {
            "case_id": f"task-{index}::initial-state=0",
            "task_name": f"task-{index}",
            "category": categories[index],
            "difficulty_level": 1,
            "initial_state_index": 0,
            "initial_state_source_sha256": f"state-{index}",
            "success": success,
        }
        for index, success in enumerate(successes)
    ]
    return {
        "schema": workflow.WORKFLOW_SCHEMA,
        "checkpoint": {"id": checkpoint, "revision": "pinned"},
        "protocol_sha256": protocol_sha256,
        "benchmark": "original_libero",
        "source": {
            "repository": workflow.LIBERO_SOURCE_REPOSITORY,
            "revision": workflow.LIBERO_SOURCE_REVISION,
        },
        "runtime_sources": {
            "openvla_oft": {
                "repository": workflow.OFT_SOURCE_REPOSITORY,
                "revision": "pinned-oft",
                "license_sha256": workflow.OFT_LICENSE_SHA256,
            },
            "dlimp": {
                "repository": workflow.DLIMP_SOURCE_REPOSITORY,
                "revision": "pinned-dlimp",
                "license_sha256": workflow.DLIMP_LICENSE_SHA256,
            },
        },
        "comparison_scope": "training_coverage_unknown",
        "evaluation_config": {
            "seed": 7,
            "env_image_resolution": 256,
            "center_crop": True,
            "num_images_in_input": 2,
            "num_open_loop_steps": 8,
            "action_transform": "openvla_normalize_then_invert_gripper",
        },
        "episodes": episodes,
    }


def test_pairing_requires_identical_case_identity() -> None:
    """Reject a candidate result that is not a one-to-one paired comparison.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    baseline = _rollouts("baseline", [0, 1, 0])
    candidate = _rollouts("candidate", [1, 0, 1])
    candidate["episodes"].pop()

    try:
        workflow._paired_comparison(baseline, candidate)
    except workflow.SylvestComparisonError as exc:
        assert "different paired cases" in str(exc)
    else:  # pragma: no cover - documents the failure boundary
        raise AssertionError("unpaired rollouts were accepted")


def test_compare_and_report_emit_measured_artifacts(tmp_path: Path) -> None:
    """Produce a comparison and RRD only from supplied paired rollout evidence.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        None.

    Raises:
        None.
    """

    protocol = {
        "schema": workflow.WORKFLOW_SCHEMA,
        "benchmark": "original_libero",
        "comparison_scope": "training_coverage_unknown",
        "source": _rollouts("baseline", [0])["source"],
        "runtime_sources": _rollouts("baseline", [0])["runtime_sources"],
        "cases": [
            {
                "case_id": "task::initial-state=0",
                "initial_state_source_sha256": "source-state-hash",
            }
        ],
        "checkpoints": {
            "baseline": {
                "repo_id": "baseline/model",
                "revision": "pinned",
                "inventory_sha256": "baseline-inventory",
            },
            "candidate": {
                "repo_id": "candidate/model",
                "revision": "pinned",
                "inventory_sha256": "candidate-inventory",
            },
        },
        "evaluation_config": _rollouts("baseline", [0])["evaluation_config"],
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        workflow._canonical_bytes(protocol)
    ).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    notices_path = tmp_path / "notices.json"
    notices_path.write_text(
        json.dumps(
            {
                "schema": workflow.WORKFLOW_SCHEMA,
                "kind": "third_party_notices",
                "protocol_sha256": protocol["protocol_sha256"],
                "components": [{"name": f"component-{index}"} for index in range(5)],
            }
        )
    )

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(_rollouts("baseline", [0, 1, 0], protocol["protocol_sha256"]))
    )
    candidate_path.write_text(
        json.dumps(_rollouts("candidate", [1, 0, 1], protocol["protocol_sha256"]))
    )
    comparison_dir = tmp_path / "comparison"

    assert (
        workflow.main(
            [
                "compare",
                "--baseline-uri",
                str(baseline_path),
                "--candidate-uri",
                str(candidate_path),
                "--output-path",
                str(comparison_dir),
            ]
        )
        == 0
    )

    comparison_path = comparison_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    assert comparison["overall"]["mean_delta_success"] == 1 / 3
    assert comparison["overall"]["candidate_only_successes"] == 2
    assert comparison["overall"]["baseline_only_successes"] == 1
    assert comparison["comparison_scope"] == "training_coverage_unknown"
    assert any(
        "not a held-out generalization claim" in item
        for item in comparison["limitations"]
    )

    report_dir = tmp_path / "report"
    assert (
        workflow.main(
            [
                "report",
                "--comparison-uri",
                str(comparison_path),
                "--protocol-uri",
                str(protocol_path),
                "--notices-uri",
                str(notices_path),
                "--run-id",
                "test-sylvest-comparison",
                "--output-path",
                str(report_dir),
            ]
        )
        == 0
    )
    assert (report_dir / "report.json").is_file()
    assert (report_dir / "comparison.rrd").stat().st_size > 0
    checksums = json.loads((report_dir / "checksums.json").read_text())
    assert set(checksums["files"]) == {"comparison.rrd", "notices.json", "report.json"}
    assert (report_dir / "notices.json").is_file()
    rerun = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(rerun), "rrd", "verify", str(report_dir / "comparison.rrd")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    inspected = subprocess.run(
        [str(rerun), "rrd", "print", str(report_dir / "comparison.rrd")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert "metrics/delta_success" in inspected.stdout


def test_unlicensed_libero_plus_source_is_refused(tmp_path: Path) -> None:
    """Do not turn an upstream source-license gap into an NPA consent flag."""

    with pytest.raises(workflow.SylvestComparisonError, match="no license file") as error:
        workflow._require_libero_plus_license(tmp_path)
    assert "asset-only compatibility" in str(error.value)
    assert "does not license this source" in str(error.value)


def test_original_mit_libero_task_and_initial_state_bytes_are_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read original LIBERO task identity and state bytes without importing Torch."""

    task_map = tmp_path / workflow._LIBERO_TASK_MAP_FILE
    task_map.parent.mkdir(parents=True)
    task_map.write_text(
        "libero_task_map = {'libero_spatial': ['task_a', 'task_b']}\n",
        encoding="utf-8",
    )
    init_root = tmp_path / "libero/libero/init_files/libero_spatial"
    init_root.mkdir(parents=True)
    (init_root / "task_a.pruned_init").write_bytes(b"state-a")
    (init_root / "task_b.pruned_init").write_bytes(b"state-b")
    license_file = tmp_path / "LICENSE"
    license_file.write_text("MIT\n", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "LIBERO_LICENSE_SHA256",
        hashlib.sha256(license_file.read_bytes()).hexdigest(),
    )

    workflow._require_original_libero_license(tmp_path)
    cases, source_map = workflow._original_libero_suite_cases(
        tmp_path, "libero_spatial"
    )
    hashes = workflow._original_libero_initial_state_hashes(
        tmp_path, "libero_spatial", cases
    )

    assert source_map == task_map
    assert [case["name"] for case in cases] == ["task_a", "task_b"]
    assert hashes["task_a"] == hashlib.sha256(b"state-a").hexdigest()
    assert hashes["task_b"] == hashlib.sha256(b"state-b").hexdigest()
    assert workflow.LIBERO_CODE_LICENSE == "MIT"
    assert workflow.LIBERO_DATASET_LICENSE == "CC-BY-4.0"


def test_runtime_source_fetch_is_revision_pinned_and_atomically_marked(
    tmp_path: Path,
) -> None:
    """Reuse only a source cache that has both a Git and ready-marker identity."""

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init"], check=True, capture_output=True)
    (source / "README.md").write_text("licensed source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.email=npa@example.invalid",
            "-c",
            "user.name=NPA test",
            "commit",
            "-m",
            "source",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    destination = tmp_path / "cache" / "source"

    materialized = workflow._materialize_runtime_source(
        str(destination), revision, str(source)
    )

    assert materialized == destination.resolve()
    assert workflow._verified_source_root(str(destination), revision, str(source))
    assert workflow._read_json(destination / workflow._SOURCE_READY_FILE) == {
        "repository": str(source),
        "revision": revision,
        "tree": str(destination.resolve()),
    }
    assert (
        workflow._materialize_runtime_source(str(destination), revision, str(source))
        == destination.resolve()
    )


def test_licensed_dlimp_parent_becomes_noticed_deterministic_derivative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply only the reviewed deterministic override to an Apache parent.

    Args:
        tmp_path: Pytest-managed temporary directory.
        monkeypatch: Pytest configuration helper.

    Returns:
        None.

    Raises:
        None.
    """

    source = tmp_path / "dlimp-source"
    dataset = source / "dlimp" / "dataset.py"
    dataset.parent.mkdir(parents=True)
    license_file = source / "LICENSE"
    license_file.write_text("Apache License\n")
    dataset.write_text(
        "def options():\n    options.deterministic = False\n    return options\n"
    )
    (source / workflow._SOURCE_READY_FILE).write_text("runtime cache metadata\n")
    monkeypatch.setattr(
        workflow,
        "DLIMP_LICENSE_SHA256",
        hashlib.sha256(license_file.read_bytes()).hexdigest(),
    )

    derivative = workflow._prepare_deterministic_dlimp_runtime(
        source, tmp_path / "runtime" / "dlimp-deterministic"
    )

    rendered = (derivative / "dlimp" / "dataset.py").read_text()
    assert "options.deterministic = True" in rendered
    assert "options.deterministic = False" not in rendered
    assert (derivative / "LICENSE").read_text() == "Apache License\n"
    notice = (derivative / "NPA_MODIFICATIONS.md").read_text()
    assert "Apache-2.0" in notice
    assert "dlimp/dataset.py only" in notice
    assert not (derivative / workflow._SOURCE_READY_FILE).exists()
    assert workflow._dlimp_derivative_is_ready(derivative)
    assert (
        workflow._prepare_deterministic_dlimp_runtime(source, derivative) == derivative
    )


def test_initial_state_indices_are_upstream_trial_indices() -> None:
    """Reject index sets that cannot map to upstream run_task trials.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    assert workflow._parse_initial_state_indices("0,1,2") == [0, 1, 2]
    with pytest.raises(workflow.SylvestComparisonError, match="contiguous zero-based"):
        workflow._parse_initial_state_indices("1,2")


def test_protocol_scope_rejects_an_unsupported_or_unproven_held_out_label() -> None:
    """Keep held-out wording bound to immutable inventory evidence.

    Returns:
        None.
    """

    protocol: dict[str, object] = {
        "schema": workflow.WORKFLOW_SCHEMA,
        "benchmark": "original_libero",
        "comparison_scope": "held_out",
        "training_inventory_sha256": None,
        "source": {
            "repository": workflow.LIBERO_SOURCE_REPOSITORY,
            "revision": workflow.LIBERO_SOURCE_REVISION,
        },
        "evaluation_config": {
            "seed": 7,
            "env_image_resolution": 256,
            "center_crop": True,
            "num_images_in_input": 2,
            "num_open_loop_steps": 8,
            "action_transform": "openvla_normalize_then_invert_gripper",
        },
        "cases": [
            {
                "case_id": "task::initial-state=0",
                "initial_state_source_sha256": "source-state-hash",
            }
        ],
        "checkpoints": {
            "baseline": {
                "repo_id": "baseline/model",
                "revision": "baseline-revision",
                "inventory_sha256": "baseline-inventory",
            },
            "candidate": {
                "repo_id": "candidate/model",
                "revision": "candidate-revision",
                "inventory_sha256": "candidate-inventory",
            },
        },
        "runtime_sources": {
            "openvla_oft": {
                "repository": workflow.OFT_SOURCE_REPOSITORY,
                "revision": "pinned-oft",
                "license_sha256": workflow.OFT_LICENSE_SHA256,
            },
            "dlimp": {
                "repository": workflow.DLIMP_SOURCE_REPOSITORY,
                "revision": "pinned-dlimp",
                "license_sha256": workflow.DLIMP_LICENSE_SHA256,
            },
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        workflow._canonical_bytes(protocol)
    ).hexdigest()

    with pytest.raises(workflow.SylvestComparisonError, match="inventory provenance"):
        workflow._validate_protocol(protocol)


def test_checkpoint_cache_records_and_protects_exact_payload(tmp_path: Path) -> None:
    """Keep the immutable snapshot separate from evaluator-local compatibility edits.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        None.

    Raises:
        None.
    """

    snapshot = tmp_path / "snapshot"
    adapter = snapshot / "lora_adapter"
    adapter.mkdir(parents=True)
    for name in (
        "config.json",
        "dataset_statistics.json",
        "action_head--1_checkpoint.pt",
        "proprio_projector--1_checkpoint.pt",
    ):
        (snapshot / name).write_text(name)
    (adapter / "adapter_config.json").write_text("adapter")
    metadata = workflow._checkpoint_metadata(snapshot, "owner/model", "pinned")
    workflow._write_json(snapshot / ".npa-checkpoint-ready.json", metadata)
    assert workflow._checkpoint_is_ready(
        snapshot, snapshot / ".npa-checkpoint-ready.json", "owner/model", "pinned"
    )

    workspace = workflow._copy_checkpoint_workspace(snapshot, tmp_path / "workspace")
    (workspace / "config.json").write_text("evaluator rewrite")
    assert (snapshot / "config.json").read_text() == "config.json"
    assert not workflow._checkpoint_is_ready(
        workspace, snapshot / ".npa-checkpoint-ready.json", "owner/model", "pinned"
    )


def test_gpu_checkpoint_must_match_stage_one_inventory() -> None:
    """Do not let a same-named mutable checkpoint replace prepared model bytes."""

    prepared = {
        "checkpoints": {
            "baseline": {
                "repo_id": "owner/model",
                "revision": "pinned",
                "inventory_sha256": "prepared-inventory",
            },
            "candidate": {
                "repo_id": "owner/candidate",
                "revision": "candidate-pinned",
                "inventory_sha256": "candidate-inventory",
            },
        }
    }
    workflow._require_prepared_checkpoint(
        prepared,
        {
            "repo_id": "owner/model",
            "revision": "pinned",
            "inventory_sha256": "prepared-inventory",
        },
    )
    with pytest.raises(workflow.SylvestComparisonError, match="differs"):
        workflow._require_prepared_checkpoint(
            prepared,
            {
                "repo_id": "owner/model",
                "revision": "pinned",
                "inventory_sha256": "substituted-inventory",
            },
        )


def test_gpu_runtime_sources_must_match_stage_one_pins() -> None:
    """Do not replace the evaluator or dlimp dependency after preparation."""

    protocol = {
        "runtime_sources": {
            "openvla_oft": {
                "repository": workflow.OFT_SOURCE_REPOSITORY,
                "revision": "oft-pinned",
                "license_sha256": workflow.OFT_LICENSE_SHA256,
            },
            "dlimp": {
                "repository": workflow.DLIMP_SOURCE_REPOSITORY,
                "revision": "dlimp-pinned",
                "license_sha256": workflow.DLIMP_LICENSE_SHA256,
            },
        }
    }
    args = SimpleNamespace(
        openvla_oft_revision="oft-pinned", dlimp_revision="dlimp-pinned"
    )
    workflow._require_prepared_runtime_sources(protocol, args)
    args.dlimp_revision = "substituted"
    with pytest.raises(workflow.SylvestComparisonError, match="differs"):
        workflow._require_prepared_runtime_sources(protocol, args)


def test_rollout_stage_refuses_missing_upstream_mp4s(tmp_path: Path) -> None:
    """Do not publish a scored rollout artifact without upstream visual evidence.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        None.

    Raises:
        None.
    """

    with pytest.raises(
        workflow.SylvestComparisonError, match="rollout-video directory"
    ):
        workflow._collect_rollout_clips(tmp_path, tmp_path / "output", expected_count=1)


def test_upstream_task_capture_pairs_each_initial_state() -> None:
    """Keep one native run_task result for every selected initial state.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    class FakeEvaluator:
        class benchmark:
            @staticmethod
            def get_benchmark_dict() -> dict[str, object]:
                return {"libero_spatial": object}

        def __init__(self) -> None:
            self.seeds: list[int] = []
            self.episode_results = iter([False, True])

        def get_image_resize_size(self, cfg: object) -> int:
            return 224

        def set_seed_everywhere(self, seed: int) -> None:
            self.seeds.append(seed)

        def run_episode(
            self, *args: object, **kwargs: object
        ) -> tuple[bool, list[object]]:
            return next(self.episode_results), []

        def run_task(self, *args: object) -> tuple[int, int]:
            cfg = args[0]
            total_episodes = int(args[9])
            total_successes = int(args[10])
            successes = [self.run_episode()[0] for _ in range(cfg.num_trials_per_task)]
            return total_episodes + len(successes), total_successes + sum(successes)

    evaluator = FakeEvaluator()
    cfg = SimpleNamespace(
        seed=7, task_suite_name="libero_spatial", num_trials_per_task=0
    )
    cases = [
        {"task_name": "task", "initial_state_index": 0},
        {"task_name": "task", "initial_state_index": 1},
    ]
    results = workflow._run_cases(
        evaluator,
        cfg,
        cases,
        {"task": 0},
        object(),  # type: ignore[arg-type]
        object(),
        object(),
        object(),
        object(),
        None,
    )

    assert evaluator.seeds == [7]
    assert [row["success"] for row in results] == [0, 1]
    assert [row["initial_state_index"] for row in results] == [0, 1]


def test_original_libero_workflow_has_five_connected_substantive_stages() -> None:
    """Keep all five actual data, rollout, metric, and visualization stages wired.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    root = Path(__file__).resolve().parents[3]
    spec = yaml.safe_load(
        (
            root
            / "workflows/testing/sylvest-oft-mixdata-original-libero-comparison.yaml"
        ).read_text()
    )
    states = spec["states"]
    assert list(states) == [
        "prepare",
        "baseline_rollouts",
        "candidate_rollouts",
        "compare",
        "report",
    ]
    assert states["prepare"]["next"] == "baseline_rollouts"
    assert states["baseline_rollouts"]["next"] == "candidate_rollouts"
    assert states["candidate_rollouts"]["next"] == "compare"
    assert states["compare"]["next"] == "report"
    assert states["report"]["terminal"] is True
    assert spec["config"]["task_suite"] == "libero_spatial"
    assert spec["config"]["initial_state_indices"] == "0,1,2"
    assert spec["config"]["comparison_scope"] == "training_coverage_unknown"
    assert spec["config"]["benchmark"] == "original_libero"
    assert spec["config"]["libero_revision"] == workflow.LIBERO_SOURCE_REVISION
    assert spec["resources"]["gpu"]["accelerators"] == "{{config.gpu_type}}:1"
    prepare_shell = states["prepare"]["run"]["shell"]
    for option in (
        "--baseline-checkpoint-id",
        "--baseline-checkpoint-revision",
        "--candidate-checkpoint-id",
        "--candidate-checkpoint-revision",
        "--model-cache-root",
        "--openvla-oft-root",
        "--openvla-oft-revision",
        "--dlimp-root",
        "--dlimp-revision",
    ):
        assert option in prepare_shell
    assert all(
        "npa.workflows.sylvest_oft_mixdata" in states[name]["run"]["shell"]
        for name in states
    )
    assert len(states["compare"]["inputs"]) == 2
    assert any(
        output["uri"].endswith("comparison.rrd")
        for output in states["report"]["outputs"]
    )
    for state in ("baseline_rollouts", "candidate_rollouts"):
        shell = states[state]["run"]["shell"]
        assert "--dlimp-root" in shell
        assert "--dlimp-revision" in shell
        assert "--dlimp-runtime-root" in shell
        assert "--benchmark" in shell
        assert "--libero-root" in shell
