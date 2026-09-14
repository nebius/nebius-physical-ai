"""Verify Arena packaging, exact-source patches, and replay reset semantics."""

import importlib.util
import subprocess
import sys
import textwrap

import pytest
from pathlib import Path

from npa.deploy.images import CONTAINER_IMAGE_NAMES, SUPPORTED_TOOL_VERSIONS


ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "Dockerfile"
RUNTIME_REQUIREMENTS = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "runtime-requirements.txt"
)
SMOKE_SCRIPT = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "smoke_functional.sh"
)
THIRD_PARTY_NOTICES = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "THIRD_PARTY_NOTICES.md"
)
REDISTRIBUTION = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "REDISTRIBUTION.md"
)
EVIDENCE_PATCH = (
    ROOT / "npa" / "docker" / "workbench" / "isaac-arena" / "patch_npa_evidence.py"
)


def test_isaac_arena_image_is_exact_source_and_payload_clean_by_construction() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "ed0fd12be862078be316c73eb7cf423ba9b1c5cd" in text
    assert "4e62ddbd7edc40fb47e62a0d5ba523eebc129482b4ce083612f17c79f1fc40a8" in text
    assert (
        "npa-isaac-lab@sha256:e321e8631c7e318b5012dad210d9cd1001b7dc833cbff0369e420c5c12657ab6"
        in text
    )
    assert "--output /tmp/isaac-arena.tar.gz" in text
    assert "sha256sum -c -" in text
    assert "grep -q 'Apache License' /opt/isaac-arena/LICENSE.md" in text
    assert "grep -q 'Version 2.0, January 2004' /opt/isaac-arena/LICENSE.md" in text
    assert "NPA_LIGHT_WORKBENCH_TOOL=isaac-arena" in text
    assert "fffbafcc0cd39e54c394b7caccffc45a24a8584a64234f7c556ad830b4619afe" in text
    assert "npa-cli-requirements.txt" in text
    assert "runtime-requirements.txt" in text
    assert "NPA_ISAAC_BOOTSTRAP=/opt/npa/bin/isaac-bootstrap" in text
    assert "common/isaac_bootstrap.sh /opt/npa/bin/isaac-bootstrap" in text
    assert "_npa_image_hooks.pth" in text
    assert "--require-hashes" in text
    assert "import onnxruntime, pinocchio, pink" in text
    assert "import lightwheel_sdk.loader" in text
    assert "'pin':'4.0.0'" in text
    assert "'pin-pink':'3.3.0'" in text
    assert "'onnxruntime':'1.27.0'" in text
    assert "'daqp':'0.8.5'" in text
    assert "'qpsolvers':'4.12.0'" in text
    assert "'lightwheel-sdk':'1.0.3'" in text
    assert "'termcolor':'3.3.0'" in text
    assert "Licensed under the Apache License, Version 2.0" in text
    assert "test ! -e /home/ubuntu/.cache/lightwheel_sdk" in text
    assert "npa workbench isaac-arena evaluate" in text
    assert "patch_npa_evidence.py" in text
    assert "NPA_ISAAC_ARENA_VIEWPORT_ONLY" in text
    assert "--dry-run" in text
    assert 'test -z "$(find /opt/isaac-arena' in text
    assert text.rstrip().endswith(
        'ENTRYPOINT ["/usr/local/bin/npa-workflow-entrypoint"]'
    )
    assert "USER ubuntu" in text


def test_isaac_arena_viewport_patch_is_narrow_and_context_bound() -> None:
    text = EVIDENCE_PATCH.read_text(encoding="utf-8")
    assert "pinned Arena viewport-camera patch context changed" in text
    assert "embodiment-camera" in text
    assert "POLICY_RUNNER_CAMERA_CONTEXT" in text
    assert 'os.environ.get("NPA_ISAAC_ARENA_VIEWPORT_ONLY") == "1"' in text
    assert "args_cli.record_camera_video or args_cli.record_viewport_video" in text
    assert "self.enable_cameras = enable_cameras and not viewport_only" in text
    assert "args_cli.enable_cameras = False" not in text
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "/opt/isaac-arena/isaaclab_arena/evaluation/policy_runner.py" in dockerfile
    assert (
        "/opt/isaac-arena/isaaclab_arena/embodiments/embodiment_base.py" in dockerfile
    )


def _evidence_patch():
    spec = importlib.util.spec_from_file_location("arena_evidence_patch", EVIDENCE_PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_patch_preserves_render_and_applies_every_context(tmp_path: Path) -> None:
    module = _evidence_patch()
    policy = tmp_path / "policy_runner.py"
    policy.write_text(module.POLICY_RUNNER_RESET + module.POLICY_RUNNER_ENVIRONMENT
                      + module.POLICY_RUNNER_CAMERA_CONTEXT + module.POLICY_RUNNER_ROLLOUT_END)
    embodiment = tmp_path / "embodiment.py"
    embodiment.write_text(module.EMBODIMENT_IMPORT + module.EMBODIMENT_ASSIGNMENT)
    video = tmp_path / "video.py"
    video.write_text(module.VIDEO_TRIGGER)
    metric = tmp_path / "metric.py"
    metric.write_text(module.REVOLUTE_POST_STEP)
    module.patch_sources(policy, embodiment, video, metric)
    assert "env.unwrapped.reset_to(initial_state, None, is_relative=True)" in policy.read_text()
    assert "simulator_ground_truth_rank" in policy.read_text()
    assert ").unwrapped" in policy.read_text()
    assert module.POLICY_RUNNER_CAMERA_CONTEXT in policy.read_text()
    assert module.POLICY_RUNNER_ROLLOUT_END_PATCHED in policy.read_text()
    assert "self.enable_cameras = enable_cameras and not viewport_only" in embodiment.read_text()
    assert "step_trigger=lambda step: step == 0" in video.read_text()
    assert "episode_trigger" not in video.read_text()
    assert "def record_post_reset" in metric.read_text()
    with pytest.raises(RuntimeError, match="context changed"):
        module.patch_sources(policy, embodiment, video, metric)

def _replay_reset_assertions() -> str:
    return textwrap.dedent("""\
        from types import SimpleNamespace
        events = []
        state = {"articulation": {"microwave": {"joint_position": [0.2]}}}
        def reset_to(value, env_ids, *, is_relative):
            assert value is state and env_ids is None and is_relative is True
            events.append("recorded_state")
            return {"observation": "from_recorded_state"}, {}
        env = SimpleNamespace(unwrapped=SimpleNamespace(reset_to=reset_to))
        policy = SimpleNamespace(get_initial_state=lambda: state,
                                 reset=lambda: events.append("policy_reset"))
        assert apply_reset(env, policy) == {"observation": "from_recorded_state"}
        assert events == ["recorded_state", "policy_reset"]
        policy.get_initial_state = lambda: None
        try:
            apply_reset(env, policy)
        except RuntimeError as error:
            assert "did not provide an initial state" in str(error)
        else:
            raise AssertionError("missing replay state was accepted")
    """)


def test_replay_reset_applies_exact_state_once_before_policy_reset(tmp_path: Path) -> None:
    module = _evidence_patch()
    program = tmp_path / "patched_replay_reset.py"
    program.write_text("def apply_reset(env, policy):\n" + module.POLICY_RUNNER_RESET_PATCHED
                       + "    finally:\n        pass\n    return obs\n" + _replay_reset_assertions())
    completed = subprocess.run([sys.executable, str(program)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("fail", [False, True])
def test_rollout_finalizes_live_evidence_on_success_and_failure(tmp_path: Path, fail: bool) -> None:
    module = _evidence_patch()
    ending = module.POLICY_RUNNER_ROLLOUT_END_PATCHED.split("\n\ndef list_variations", 1)[0]
    program = tmp_path / "patched_rollout_finalization.py"
    body = "def rollout(env, fail):\n    try:\n        if fail:\n            raise RuntimeError('rollout failed')\n    except RuntimeError:\n        raise\n    else:\n"
    assertions = textwrap.dedent("""\
        from types import SimpleNamespace
        events = []
        recorder = SimpleNamespace(finalize=lambda: events.append('finalized-live'))
        base = SimpleNamespace(cfg=SimpleNamespace(metrics=True), compute_metrics=lambda: 0.0,
                               _npa_video_capture_recorder=recorder)
        env = SimpleNamespace(unwrapped=base)
        try:
            result = rollout(env, FAIL)
        except RuntimeError as error:
            assert FAIL and str(error) == 'rollout failed'
        else:
            assert not FAIL and result == 0.0
        del base._npa_video_capture_recorder
        events.append('teardown')
        assert events == ['finalized-live', 'teardown']
    """).replace("FAIL", repr(fail))
    program.write_text(body + ending + "\n" + assertions)
    completed = subprocess.run([sys.executable, str(program)], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_isaac_arena_runtime_dependency_closure_is_hash_locked() -> None:
    text = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    expected = {
        "daqp": "0.8.5",
        "onnxruntime": "1.27.0",
        "pin": "4.0.0",
        "pin-pink": "3.3.0",
        "qpsolvers": "4.12.0",
        "lightwheel-sdk": "1.0.3",
        "termcolor": "3.3.0",
    }
    for distribution, version in expected.items():
        assert f"{distribution}=={version} \\" in text
    requirement_lines = [
        line for line in text.splitlines() if line and not line.startswith((" ", "#"))
    ]
    assert requirement_lines
    assert all(line.endswith(" \\") for line in requirement_lines)
    assert text.count("--hash=sha256:") >= len(requirement_lines)
    notices = THIRD_PARTY_NOTICES.read_text(encoding="utf-8")
    assert "Lightwheel SDK 1.0.3" in notices
    assert "841ec064ab21a403de024e1e860541e9949e0ea2330d51961b1fdf49d0ec21cd" in notices
    assert "no Lightwheel registry object" in notices
    assert "NVIDIA viewport graphics userspace (runtime only)" in notices
    assert "Not included in image layers" in notices
    redistribution = REDISTRIBUTION.read_text(encoding="utf-8")
    assert "exactly matches the loaded kernel driver" in redistribution
    assert "never installs it on the node" in redistribution


def test_isaac_arena_image_catalog_identity() -> None:
    assert CONTAINER_IMAGE_NAMES["isaac-arena"] == "npa-isaac-arena"
    assert SUPPORTED_TOOL_VERSIONS["isaac-arena"] == "0.3.0-isaaclab3-20260912-r2"


def test_isaac_arena_golden_smoke_uses_hash_pinned_nonzero_replay() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "154ebea7839ec53e6ac441e18f1404b3fe140c3f004ad7e309519ba37274fa50" in text
    assert "--environment gr1_open_microwave" in text
    assert "--policy-type replay" in text
    assert "--input-path" in text
    assert "--embodiment gr1_pink" in text
    assert "--object tomato_soup_can" in text
    assert "--execution-device cpu" in text
    assert "--num-envs 1" in text
    assert "--record-video" in text
    assert "zero_action" not in text
