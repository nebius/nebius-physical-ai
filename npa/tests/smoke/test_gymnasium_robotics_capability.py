from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import runpy
import sys
import types
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SMOKE = ROOT / "npa/docker/workbench/gymnasium-robotics/capability_smoke.py"


def _constant_from(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def _constant(name: str) -> object:
    return _constant_from(SMOKE, name)


def _smoke_namespace() -> dict[str, object]:
    dependency_stubs = {
        name: types.ModuleType(name)
        for name in ("gymnasium", "gymnasium_robotics", "mujoco", "numpy")
    }
    with mock.patch.dict(sys.modules, dependency_stubs):
        return runpy.run_path(
            str(SMOKE), run_name="npa_gymnasium_capability_smoke_test"
        )


def _transition_guard():
    return _smoke_namespace()["_require_state_transition"]


def _digest_parser():
    return _smoke_namespace()["_digest_from_reference"]


def test_fixed_capability_identity_and_trajectory() -> None:
    assert (
        _constant("ENV_ID") == "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
    )
    assert _constant("ROLLOUT_STEPS") == 120
    assert _constant("MIN_TRANSITION_DELTA") == 1e-6
    assert _constant("RESET_SEED") == 20260910
    assert _constant("ACTION_SEED") == 11092026


def test_image_digest_parser_requires_one_anchored_digest() -> None:
    parse = _digest_parser()
    digest = "sha256:" + "a" * 64
    assert parse(f"registry.example.invalid/image@{digest}", "image") == digest
    assert (
        parse(f"docker-pullable://registry.example.invalid/image@{digest}", "image")
        == digest
    )
    for value in (
        digest,
        f"registry.example.invalid/image@{digest}-suffix",
        f"registry.example.invalid/image@sha256:{'b' * 64}@{digest}",
    ):
        with pytest.raises(RuntimeError, match="not digest-pinned"):
            parse(value, "image")


def test_every_runtime_verifier_binds_the_exact_asset_lock_bytes() -> None:
    asset_lock = ROOT / "npa/docker/workbench/gymnasium-robotics/asset-lock.json"
    expected = hashlib.sha256(asset_lock.read_bytes()).hexdigest()
    for path in (
        SMOKE,
        ROOT / "npa/docker/workbench/gymnasium-robotics/verify_image.py",
        ROOT / "npa/scripts/scan_image_gymnasium_robotics_payload.py",
    ):
        assert _constant_from(path, "EXPECTED_ASSET_LOCK") == expected


def test_smoke_requires_physics_touch_orientation_and_real_egl() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    for token in (
        'sum(value > 0 for value in rollout["contacts"])',
        '"steps_with_nonzero_readings": sum(',
        'max(rollout["orientation_changes"])',
        'max(rollout["position_changes"])',
        '"initial_object_pose": rollout["initial_goal"].tolist()',
        '"final_object_pose": rollout["final_goal"].tolist()',
        "len(set(frames)) < 2",
        '"backend": "egl"',
        '"synthetic_only_fixture": False',
        '"physics_substeps": substeps',
        '"pod_observed_image_digest": observed',
        'os.environ.get("NPA_GYMNASIUM_RUNTIME_ROOT", "")',
        'receipt.get("source_commit") != EXPECTED_SOURCE',
        '"cache_receipt": runtime_receipt',
    ):
        assert token in source


def test_smoke_accepts_position_orientation_and_state_evolution() -> None:
    _transition_guard()([0.0, 0.01], [0.0, 0.02], [0.0, 0.03])


@pytest.mark.parametrize(
    ("positions", "orientations", "states"),
    [
        ([0.0, 1e-6], [0.0, 0.02], [0.0, 0.03]),
        ([0.0, 0.01], [0.0, 1e-6], [0.0, 0.03]),
        ([0.0, 0.01], [0.0, 0.02], [0.0, 1e-6]),
    ],
)
def test_smoke_rejects_a_missing_transition_dimension(
    positions: list[float], orientations: list[float], states: list[float]
) -> None:
    with pytest.raises(RuntimeError, match="object position, orientation"):
        _transition_guard()(positions, orientations, states)


def test_smoke_emits_only_the_named_json() -> None:
    source = SMOKE.read_text(encoding="utf-8")
    assert 'output_dir = Path(os.environ["NPA_SMOKE_OUTPUT_DIR"])' in source
    assert 'output = output_dir / "gymnasium-robotics-smoke.json"' in source
    assert '"media_type": "application/json"' in source
    assert '"exit_status": 0' in source


def test_golden_eval_artifact_uses_the_smoke_output_contract() -> None:
    manifest = (ROOT / "npa/src/npa/smoke/golden_evals.yaml").read_text(
        encoding="utf-8"
    )
    assert "artifact: ${NPA_SMOKE_OUTPUT_DIR}/gymnasium-robotics-smoke.json" in manifest
