from __future__ import annotations

import ast
import hashlib
from pathlib import Path

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


def test_fixed_capability_identity_and_trajectory() -> None:
    assert (
        _constant("ENV_ID") == "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
    )
    assert _constant("ROLLOUT_STEPS") == 120
    assert _constant("RESET_SEED") == 20260910
    assert _constant("ACTION_SEED") == 11092026


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
        '"initial_object_pose": rollout["initial_goal"].tolist()',
        '"final_object_pose": rollout["final_goal"].tolist()',
        "len(set(frames)) < 2",
        '"backend": "egl"',
        '"synthetic_only_fixture": False',
        '"physics_substeps": substeps',
        '"pod_observed_image_digest": observed',
    ):
        assert token in source


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
