"""Contract tests for the distinct pinned LeRobot VLA-JEPA workflow."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from npa.orchestration.npa_workflow import load_spec
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.deploy.images import SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS
from npa.workflows import lerobot_vla_jepa as vla


WORKFLOW = Path("workflows/testing/lerobot-vla-jepa.yaml")
READINESS = WORKFLOW.with_suffix(".readiness.json")
DOCKERFILE = Path("npa/docker/workbench/lerobot-vla-jepa/Dockerfile")


def test_workflow_has_five_connected_native_stages() -> None:
    """Keep all four required real stages plus factual provenance connected."""
    spec = load_spec(WORKFLOW)
    expected = ["prepare", "train", "rollout", "evaluate", "report"]
    assert list(spec.states) == expected
    assert [spec.states[name].tool_ref for name in expected] == [
        f"workbench.lerobot.vla_jepa.{name}" for name in expected
    ]
    assert [spec.states[name].next for name in expected[:-1]] == expected[1:]
    assert spec.states["report"].terminal
    assert spec.resources["gpu"]["image"] == "{{config.vla_jepa_image}}"
    assert "registry.invalid" in spec.config["vla_jepa_image"]
    assert spec.config["train_steps"] == "30000"
    assert spec.config["heldout_task_ids"] == "[0, 1]"


def test_stage_templates_pass_exact_predecessor_artifacts() -> None:
    """Ensure downstream toolRefs receive checksum-sealed upstream roots."""
    templates = {
        name: TOOL_CATALOG[f"workbench.lerobot.vla_jepa.{name}"].argv_template
        for name in ("prepare", "train", "rollout", "evaluate", "report")
    }
    assert "{{config.prepared_uri}}" in templates["train"]
    assert "{{config.training_uri}}" in templates["rollout"]
    assert "{{config.rollout_uri}}" in templates["evaluate"]
    assert "{{config.prepared_uri}}" in templates["evaluate"]
    assert "{{config.evaluation_uri}}" in templates["report"]
    assert "{{config.training_uri}}" in templates["report"]


def test_live_matrix_registers_the_private_candidate_without_a_generic_fallback() -> (
    None
):
    """The rotation may plan the graph, but cannot silently select another image."""
    case = next(item for item in SUBMIT_LIVE_MATRIX if item.spec == WORKFLOW.name)
    assert case.tier == "gpu"
    assert case.plan_only
    assert not case.image_tool
    assert set(case.secret_envs) == {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
    }
    assert "unroutable image sentinel" in case.plan_only_justification
    assert "private" in case.plan_only_justification
    assert "LIBERO dataset endpoint" in case.plan_only_justification
    assert "no acceptance mechanism" in case.plan_only_justification


def test_candidate_image_removes_the_inherited_pip_cache() -> None:
    """The exact-byte scan must not retain a parent cache in layer history."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    cleanup = "rm -rf /home/ubuntu/.cache/pip"
    assert cleanup in dockerfile
    assert "test ! -e /home/ubuntu/.cache/pip" in dockerfile
    assert dockerfile.index(cleanup) > dockerfile.index('"rerun-sdk==0.38.1"')
    assert "FROM scratch AS vla_jepa_runtime" in dockerfile
    final_copy = "COPY --from=vla_jepa_build / /"
    assert final_copy in dockerfile
    assert dockerfile.index(cleanup) < dockerfile.index(final_copy)
    assert "PYTHONPATH=/opt/vla-jepa-npa/src:$PYTHONPATH" not in dockerfile
    assert "PYTHONPATH=/opt/vla-jepa-npa/src" in dockerfile
    assert "PATH=/opt/lerobot/venv/bin:$PATH" not in dockerfile
    assert "h5py/tests/data_files -type f -name '*.h5' -delete" in dockerfile
    assert "robosuite/models/assets/demonstrations -type f -name '*.hdf5'" in dockerfile
    assert "botocore/data -type f -name 'examples-1.json' -delete" in dockerfile


def test_candidate_image_declares_and_implements_skypilot_bootstrap_contract() -> None:
    """Keep the separately built candidate eligible for the real K8s preflight."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        'org.nebius.npa.skypilot-bootstrap-contract="skypilot-0.12.2-v1"' in dockerfile
    )
    for token in (
        "openssh-server",
        "rsync",
        "sudo",
        "ubuntu ALL=(ALL) NOPASSWD:ALL",
        "PasswordAuthentication no",
        "PermitRootLogin no",
        "rm -f /etc/ssh/ssh_host_*",
        'ENTRYPOINT ["/usr/local/bin/npa-lerobot-vla-jepa-entrypoint"]',
    ):
        assert token in dockerfile
    entrypoint = DOCKERFILE.parent / "entrypoint.sh"
    entrypoint_text = entrypoint.read_text(encoding="utf-8")
    assert "ssh-keygen -A" in entrypoint_text
    assert 'exec "$@"' in entrypoint_text
    assert "lerobot-vla-jepa" in SKYPILOT_BOOTSTRAP_ATTESTED_TOOLS


def test_task_disjoint_split_and_numeric_training_statistics() -> None:
    """Held-out tasks cannot leak into the train episode set or its statistics."""
    by_task = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    train, reserved = vla._selected_episodes(by_task, [0], 0.5, seed=7)
    assert not (set(train) & set(reserved))
    assert {0, 1}.issubset(reserved)
    rows = [
        {
            "episode_index": episode,
            "action": [float(episode)] * 7,
            "observation.state": [float(episode)] * 8,
        }
        for episode in range(6)
    ]
    stats = vla._training_stats(rows, set(train), {"observation.image": {"keep": True}})
    assert stats["observation.image"] == {"keep": True}
    assert stats["action"]["count"] == [len(train)]
    assert stats["action"]["mean"][0] == np.mean(train)


def test_native_commands_use_local_immutable_model_snapshots(tmp_path: Path) -> None:
    """Prevent an unpinned Qwen or V-JEPA Hub-main fetch from the stage adapter."""
    prepared = tmp_path / "prepared"
    vla.write_json(
        prepared / "prepare.json",
        {
            "dataset": {"repo": vla.LIBERO_REPO, "revision": vla.LIBERO_REVISION},
            "train_episodes": [3, 4],
        },
    )
    models = {name: tmp_path / name for name in ("pretrain", "qwen", "vjepa")}
    command = vla._train_command(
        prepared,
        tmp_path / "training",
        SimpleNamespace(train_steps=2, train_batch_size=1, num_workers=0),
        models,
    )
    assert f"--policy.path={models['pretrain']}" in command
    assert f"--policy.qwen_model_name={models['qwen']}" in command
    assert f"--policy.jepa_encoder_name={models['vjepa']}" in command
    assert f"--dataset.revision={vla.LIBERO_REVISION}" in command
    assert "--wandb.enable=false" in command
    assert "--wandb.mode=disabled" in command


def test_snapshot_creates_its_atomic_ready_marker_parent(
    monkeypatch, tmp_path: Path
) -> None:
    """The first real runtime fetch must not depend on a pre-created cache marker."""
    snapshot = tmp_path / "pinned-snapshot"
    snapshot.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_: str(snapshot)),
    )

    assert (
        vla._snapshot("author/payload", "deadbeef", "model", tmp_path / "cache")
        == snapshot
    )
    marker = (
        tmp_path / "cache" / "npa-vla-jepa-ready" / "author--payload--deadbeef.json"
    )
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "repo": "author/payload",
        "repo_type": "model",
        "revision": "deadbeef",
    }


def test_evaluation_rejects_missing_or_out_of_range_native_success(
    tmp_path: Path,
) -> None:
    """Do not publish a held-out metric unless it is native and bounded."""
    prepared = tmp_path / "prepared"
    training = tmp_path / "training"
    rollouts = tmp_path / "rollouts"
    vla.write_json(prepared / "prepare.json", {"seed": 42})
    vla.write_json(
        training / "training.json",
        {"prepare_sha256": vla.file_sha256(prepared / "prepare.json")},
    )
    vla.write_json(
        rollouts / "rollout.json",
        {"training_sha256": vla.file_sha256(training / "training.json")},
    )
    (training / "checkpoint").mkdir()
    original = vla._verify_training
    try:
        vla._verify_training = lambda *_: {
            "checkpoint_hashes": {"model.safetensors": "x"}
        }
        try:
            vla.evaluate(rollouts, training, prepared, tmp_path / "out")
        except ValueError as error:
            assert "pc_success" in str(error)
        else:
            raise AssertionError("missing native success unexpectedly accepted")
    finally:
        vla._verify_training = original


def test_report_writes_and_decodes_a_rerun_recording(tmp_path: Path) -> None:
    """Exercise the RRD writer and its independent CLI inspection on test data."""
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    vla.write_json(training / "training.json", {"training": "test"})
    vla.write_json(
        evaluation / "heldout-evaluation.json",
        {
            "training_sha256": vla.file_sha256(training / "training.json"),
            "checkpoint_hashes": {"model.safetensors": "test-only"},
            "pc_success": 0.25,
        },
    )
    output = tmp_path / "report"
    vla.report(evaluation, training, output, "vla-jepa-test-report")
    assert (output / "vla-jepa.rrd").is_file()
    assert "metrics/heldout_pc_success" in (
        output / "vla-jepa.inspection.txt"
    ).read_text(encoding="utf-8")
    assert (
        json.loads((output / "report.json").read_text(encoding="utf-8"))[
            "ready_for_robot_deployment"
        ]
        is False
    )


def test_readiness_is_hash_bound_and_does_not_claim_live_acceptance() -> None:
    """Keep planned structural evidence separate from the missing live gate."""
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert (
        readiness["workflow_sha256"]
        == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    )
    expected = (
        WORKFLOW,
        DOCKERFILE,
        Path("npa/docker/workbench/lerobot-vla-jepa/entrypoint.sh"),
        Path("npa/src/npa/deploy/images.py"),
        Path("npa/src/npa/workflows/lerobot_vla_jepa.py"),
        Path("npa/tests/workflows/test_lerobot_vla_jepa.py"),
    )
    hashes = {
        item.split(" ", 1)[1]: item.split(" ", 1)[0].removeprefix("sha256:")
        for item in readiness["planning"]["task_fidelity"]["evidence"]
        if item.startswith("sha256:")
    }
    assert set(hashes) == {str(path) for path in expected}
    for path in expected:
        assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["source_image"]["status"] == "unverified"
    assert readiness["prerequisites"]["target_runtime"]["status"] == "unverified"
