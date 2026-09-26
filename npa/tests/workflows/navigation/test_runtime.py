"""Exercise native driver boundaries with CPU contract fixtures, never a simulator."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from npa.workflows.navigation import runtime
from npa.workflows.navigation.artifacts import file_sha256
from npa.workflows.navigation.contract import finite_array


@pytest.mark.parametrize("stage", ["train", "evaluate", "evaluate-checkpoint"])
@pytest.mark.parametrize("builtin", [False, True])
def test_launcher_receives_resolved_backends_and_sealed_seed(
    stage, builtin, recipe, tmp_path, monkeypatch
):
    if builtin:
        recipe = recipe.model_copy(
            update={"adapter_module": "npa.workflows.navigation.reference"}
        )
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "recipe.json").write_text(recipe.model_dump_json())
    cases = recipe.train_cases if stage == "train" else recipe.eval_cases
    unresolved = SimpleNamespace(physics_alternatives=("PhysX", "Newton"))
    resolved = SimpleNamespace(physics="PhysX", seed=None)
    events = []

    callbacks = _startup_callbacks(unresolved, resolved, cases, stage, builtin, events)
    args = SimpleNamespace(
        stage=stage,
        input_path=source,
        output_path=output,
        kit_args="--/app/custom=true  --/app/other=7",
    )
    _stub_launcher(monkeypatch, callbacks, events, args, recipe, unresolved)
    assert runtime.main([]) == 0
    assert events == [cases[0].seed, "resolved", "Kit", resolved]


def _startup_callbacks(unresolved, resolved, cases, stage, builtin, events):
    def resolve(config):
        assert config is unresolved
        events.append("resolved")
        return resolved

    @contextmanager
    def launch(config, args):
        assert config is resolved and config.seed == cases[0].seed
        assert events == [cases[0].seed, "resolved"]
        if builtin and stage != "train":
            assert "--/rtx/hydra/supportMultiTickRate=true" in args.kit_args.split()
            assert "--/rtx/rendering/perSensorTickTlas=true" in args.kit_args.split()
        else:
            assert args.kit_args == "--/app/custom=true  --/app/other=7"
        events.append("Kit")
        yield

    return resolve, launch


def _stub_launcher(monkeypatch, callbacks, events, args, recipe, unresolved):
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        SimpleNamespace(launch_simulation=callbacks[1]),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils.hydra",
        SimpleNamespace(resolve_presets=callbacks[0]),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.seed",
        SimpleNamespace(configure_seed=events.append),
    )
    monkeypatch.setattr(runtime, "_arguments", lambda _: args)
    monkeypatch.setattr(
        runtime,
        "validate_scene_package",
        lambda _: pytest.fail("USD must not load before Kit starts"),
    )
    monkeypatch.setattr(runtime, "read_recipe", lambda _: recipe)
    # Request validation has separate rejection tests; this fixture isolates startup.
    monkeypatch.setattr(runtime, "_validate_control_request", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "task_adapter",
        lambda _: SimpleNamespace(configure=lambda **kwargs: unresolved),
    )
    monkeypatch.setattr(
        runtime, "_execute", lambda args, recipe, adapter, config: events.append(config)
    )


def test_native_execution_validates_usd_before_creating_scene(
    recipe, tmp_path, monkeypatch
):
    validated = []

    def make(*args, **kwargs):
        assert validated == [tmp_path / recipe.scene_file]
        raise RuntimeError("native scene creation reached")

    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=make))
    monkeypatch.setattr(runtime, "validate_scene_package", validated.append)
    with pytest.raises(RuntimeError, match="native scene creation reached"):
        runtime._execute(SimpleNamespace(input_path=tmp_path), recipe, None, None)


def test_measurements_do_not_alias_mutating_simulator_buffers():
    source = np.zeros((2, 3))
    measured = finite_array(source, (2, 3), "position")
    source[:] = 2
    assert measured.sum() == 0


def test_corrupt_checkpoint_rejected_before_native_loader(
    recipe, tmp_path, monkeypatch
):
    (tmp_path / "policy.pt").write_bytes(b"tampered")
    (tmp_path / "training.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    monkeypatch.setattr(
        "npa.workflows.navigation.initialization.load_native_checkpoint",
        lambda *args: pytest.fail("corrupt checkpoint must not be decoded"),
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        runtime._evaluate(None, None, None, None, recipe, tmp_path, tmp_path)


def test_policy_load_error_never_calls_rollout(recipe, tmp_path, monkeypatch):
    (tmp_path / "policy.pt").write_bytes(b"bad-native-checkpoint")
    (tmp_path / "training.json").write_text(
        json.dumps({"checkpoint_sha256": file_sha256(tmp_path / "policy.pt")})
    )

    def fail(*args):
        raise RuntimeError("native checkpoint decoder rejected payload")

    monkeypatch.setattr(runtime, "_verify_training_binding", lambda *args: None)
    monkeypatch.setattr(
        "npa.workflows.navigation.initialization.load_native_checkpoint", fail
    )
    monkeypatch.setattr(
        runtime, "_rollout", lambda *args: pytest.fail("no fallback policy")
    )
    with pytest.raises(RuntimeError, match="decoder rejected"):
        runtime._evaluate(None, None, None, None, recipe, tmp_path, tmp_path)


@pytest.mark.parametrize("failure", ["auto-reset", "nan-action", "changed-goal"])
def test_evaluation_fails_closed_on_invalid_native_transitions(
    recipe, monkeypatch, failure
):
    torch = pytest.importorskip("torch")
    initial = {
        "position_m": np.array([case.position_m for case in recipe.eval_cases]),
        "goal_m": np.array([case.goal_m for case in recipe.eval_cases]),
        "obstacle_contact": np.zeros(2),
        "peer_contact": np.zeros(2),
        "physical_failure": np.zeros(2),
    }
    measured = {key: array.copy() for key, array in initial.items()}
    if failure == "changed-goal":
        measured["goal_m"] += 1
    monkeypatch.setattr(runtime, "verify_reset", lambda *args: initial)
    monkeypatch.setattr(runtime, "snapshot", lambda *args: measured)
    wrapped = SimpleNamespace(
        get_observations=lambda: torch.zeros((2, 3)),
        step=lambda _: (
            torch.zeros((2, 3)),
            None,
            torch.tensor([failure == "auto-reset", False]),
            None,
        ),
    )

    def policy(obs):
        return torch.full((2, 2), float("nan") if failure == "nan-action" else 0.1)

    with pytest.raises(ValueError):
        runtime._rollout(None, wrapped, policy, None, recipe)


def test_rollout_actions_leave_native_metrics_resettable(recipe, monkeypatch):
    torch = pytest.importorskip("torch")
    initial = {
        "position_m": np.array([case.position_m for case in recipe.eval_cases]),
        "goal_m": np.array([case.goal_m for case in recipe.eval_cases]),
        "obstacle_contact": np.zeros(2),
        "peer_contact": np.zeros(2),
        "physical_failure": np.zeros(2),
    }
    metrics = {"error_pos": torch.zeros(2)}
    policy = torch.nn.Linear(3, 3).eval()

    def reset(*args):
        metrics["error_pos"][torch.arange(2)] = 0.0
        return initial

    def step(actions):
        assert not actions.requires_grad
        metrics["error_pos"] = torch.linalg.vector_norm(actions, dim=-1)
        return actions, None, torch.zeros(2, dtype=torch.bool), {}

    monkeypatch.setattr(runtime, "verify_reset", reset)
    monkeypatch.setattr(runtime, "snapshot", lambda *args: initial)
    wrapped = SimpleNamespace(get_observations=lambda: torch.zeros((2, 3)), step=step)
    recipe = recipe.model_copy(update={"episode_steps": 2})
    with torch.enable_grad():
        for _ in range(2):
            runtime._rollout(None, wrapped, policy, None, recipe)
        reset()
        assert torch.is_grad_enabled()


def test_usd_reference_count_binds_exact_scene(usd_environment, recipe, tmp_path):
    from pxr import UsdGeom
    from npa.workflows.navigation.native import _verify_scene_reference

    env, _, _ = usd_environment
    stage = env.unwrapped.sim.stage
    scene = tmp_path / "warehouse.usda"
    scene.write_text('#usda 1.0\n(defaultPrim = "Scene")\ndef Xform "Scene" {}\n')
    stage.GetPrimAtPath(recipe.scene_prim).GetReferences().AddReference(str(scene))
    _verify_scene_reference(stage, recipe, scene)
    duplicate = UsdGeom.Xform.Define(stage, "/World/Duplicate").GetPrim()
    duplicate.GetReferences().AddReference(str(scene))
    with pytest.raises(ValueError, match="exactly one"):
        _verify_scene_reference(stage, recipe, scene)


def test_training_recipe_cannot_be_rebound(recipe, tmp_path):
    (tmp_path / "recipe.json").write_text(recipe.model_dump_json())
    report = {
        "schema": "npa.navigation.training.v1",
        "task": recipe.task,
        "image": recipe.image,
        "adapter_sha256": recipe.adapter_sha256,
        "recipe_sha256": file_sha256(tmp_path / "recipe.json"),
        "runtime": {"scene_sha256": recipe.scene_sha256},
    }
    runtime._verify_training_binding(report, recipe, tmp_path)
    report["task"] = "Different-Navigation-v0"
    with pytest.raises(ValueError, match="does not bind"):
        runtime._verify_training_binding(report, recipe, tmp_path)


def test_native_scene_package_dependency_validation(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdUtils
    from npa.workflows.navigation.native import validate_scene_package

    source = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    prim = UsdGeom.Xform.Define(stage, "/Scene").GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.Cube.Define(stage, "/Scene/Wall")
    stage.GetRootLayer().Save()
    package = tmp_path / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(package))
    validate_scene_package(package)
    prim.GetReferences().AddReference("missing-layer.usda")
    stage.GetRootLayer().Save()
    broken = tmp_path / "broken.usdz"
    import zipfile

    with zipfile.ZipFile(broken, "w", compression=zipfile.ZIP_STORED) as writer:
        writer.write(source, "scene.usda")
    with pytest.raises(ValueError, match="self-contained"):
        validate_scene_package(broken)


def test_policy_state_digest_detects_normalization_buffer_changes():
    torch = pytest.importorskip("torch")
    from npa.workflows.navigation.native import policy_state_digest

    policy = torch.nn.Linear(2, 2)
    policy.register_buffer("running_mean", torch.zeros(2))
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))
    original = policy_state_digest(runner)
    policy.running_mean += 1
    assert policy_state_digest(runner) != original
