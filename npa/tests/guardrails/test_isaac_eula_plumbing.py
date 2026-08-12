"""Every automated path that runs an Isaac image must carry the operator's EULA acceptance.

The four Isaac workbench images ship no NVIDIA Isaac Sim or Isaac Lab. They fetch it on
first run and **refuse** (exit 78) unless the operator has set both
``OMNI_KIT_ACCEPT_EULA`` and ``ISAACSIM_ACCEPT_EULA``. That refusal is the legal mechanism
and is tested directly elsewhere.

The corollary is what this file guards, and it is easy to miss: any automated path that
launches one of those images has to *carry* that acceptance, or it simply cannot run them.
The serverless golden eval found this the expensive way — a real submitted job failed with

    isaac-bootstrap: refusing to download NVIDIA Isaac Sim / Isaac Lab.
    Not accepted (unset or not YES): OMNI_KIT_ACCEPT_EULA ISAACSIM_ACCEPT_EULA

which is correct behaviour and a useless test run. At that point none of the twelve
SkyPilot task templates that use an Isaac image declared the variables either.

Those templates are gone: the npa.workflow specs are the authoring surface now, and a spec
reaches an Isaac image through its ``toolRef`` rather than by naming the image. So the rule
follows the surface — the set is derived from the same toolRef-to-image map the renderer uses,
which means a new Isaac toolRef is covered the moment it is added.

The fix is emphatically NOT to hardcode ``YES`` anywhere in the repo: that would be us
accepting on the operator's behalf, which is precisely what the re-architecture exists to
avoid. Instead the variables are declared empty (so a task fails closed, with the
actionable message) and the operator supplies them at launch. These tests pin both halves:
the plumbing exists, and it does not pre-accept.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"
NPA_WORKFLOWS = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
EULA_VARS = ("OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA")
#: Images whose entrypoints reach Isaac through the bootstrap shim.
ISAAC_IMAGE_MARKERS = ("npa-isaac-lab", "npa-sonic")
#: Image keys in the renderer's toolRef map that resolve to an Isaac-based image.
ISAAC_IMAGE_KEYS = ("isaac-lab", "sonic")


def _isaac_tool_refs() -> tuple[str, ...]:
    """toolRef prefixes the RENDERER routes at an Isaac image.

    Derived rather than listed, so a new Isaac toolRef is covered the moment it is added.
    """

    from npa.orchestration.npa_workflow.skypilot_render import (
        ISAAC_IMAGE_TOOLS,
        TOOL_REF_IMAGE_TOOL,
    )

    return tuple(
        prefix for prefix, tool in TOOL_REF_IMAGE_TOOL.items() if tool in ISAAC_IMAGE_TOOLS
    )


def _isaac_templates() -> list[Path]:
    """Raw templates that name an Isaac image, plus specs whose toolRefs route at one."""

    paths = [
        path
        for path in SKYPILOT_DIR.glob("*.yaml")
        if any(marker in path.read_text(encoding="utf-8") for marker in ISAAC_IMAGE_MARKERS)
    ]
    prefixes = _isaac_tool_refs()
    for path in NPA_WORKFLOWS.glob("*.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        refs = [
            str((state or {}).get("toolRef") or "")
            for state in (document.get("states") or {}).values()
        ]
        if any(ref == p or ref.startswith(p + ".") for ref in refs for p in prefixes):
            paths.append(path)
    return sorted(set(paths))


def test_there_are_isaac_templates_to_check() -> None:
    """Guard the guard: a rename that empties the set must not silently pass."""
    assert len(_isaac_templates()) >= 10


@pytest.mark.parametrize("tool_ref", _isaac_tool_refs())
def test_the_renderer_declares_eula_acceptance_for_every_isaac_tool_ref(tool_ref: str) -> None:
    """The spec-surface half of #229's rule, moved to where the routing decision is made.

    A spec reaches an Isaac image through its toolRef, not by naming an image, so asking each
    spec to declare the variables would ask its author to know the renderer's routing table.
    The renderer knows it, so the renderer declares them — empty, so the task still fails
    closed with the actionable message.
    """

    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    envs = isaac_eula_envs(tool_ref)
    assert set(envs) == set(EULA_VARS), tool_ref


def test_the_renderer_declares_nothing_for_a_non_isaac_tool_ref() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs("workbench.lancedb.serve") == {}


def test_the_renderer_never_pre_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared empty unless the OPERATOR set it; never YES from the repo."""

    for var in EULA_VARS:
        monkeypatch.delenv(var, raising=False)
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs("workbench.sonic.train") == dict.fromkeys(EULA_VARS, "")


def test_the_operator_can_supply_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in EULA_VARS:
        monkeypatch.setenv(var, "YES")
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    assert isaac_eula_envs("workbench.sonic.train") == dict.fromkeys(EULA_VARS, "YES")


@pytest.mark.parametrize("path", _isaac_templates(), ids=lambda p: p.name)
def test_isaac_templates_declare_eula_in_every_envs_block(path: Path) -> None:
    """A multi-task file needs it in EVERY task, not just the first.

    isaac-lab-rl-sweep.yaml has four; sonic-locomotion-finetuning.yaml three. Missing one
    means that stage alone dies, which is a maddening way to discover the problem.
    """
    text = path.read_text(encoding="utf-8")
    envs_blocks = len(re.findall(r"(?m)^envs:\n", text))
    for var in EULA_VARS:
        assert text.count(f"{var}:") >= envs_blocks, (
            f"{path.name} has {envs_blocks} `envs:` block(s) but declares {var} "
            f"{text.count(f'{var}:')} time(s); every task that runs an Isaac image needs it"
        )


@pytest.mark.parametrize("path", _isaac_templates(), ids=lambda p: p.name)
def test_templates_do_not_pre_accept_the_licence(path: Path) -> None:
    """The declaration must be EMPTY. Pre-accepting would gut the whole mechanism."""
    for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
        if not isinstance(document, dict):
            continue
        for var, value in (document.get("envs") or {}).items():
            if var in EULA_VARS:
                assert value in ("", None), (
                    f"{path.name} pre-accepts NVIDIA's licence ({var}={value!r}). "
                    f"Acceptance is the operator's to give at launch; baking it here is "
                    f"the exact thing the runtime-fetch architecture exists to avoid."
                )


def test_the_forwarder_never_invents_acceptance() -> None:
    """Acceptance must come from the caller's environment, never a literal in our code."""
    from npa.serverless_common import env as env_module
    from npa.smoke import serverless_runner

    # Re-exported from the shared builder, so the golden-eval runner and every CLI path
    # agree on the variable names.
    assert set(serverless_runner.ISAAC_EULA_VARS) == set(EULA_VARS)
    assert set(env_module.ISAAC_EULA_VARS) == set(EULA_VARS)

    source = Path(env_module.__file__).read_text(encoding="utf-8")
    instructions = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for var in EULA_VARS:
        assert f'"{var}": "YES"' not in instructions, "must not pre-accept on the operator's behalf"
    assert "os.environ[name]" in instructions, "acceptance must come from the caller's environment"


def test_serverless_runner_omits_unset_acceptance(monkeypatch) -> None:
    """Absent acceptance must stay absent — not become an empty string that looks set."""
    from npa.smoke import serverless_runner

    for var in EULA_VARS:
        monkeypatch.delenv(var, raising=False)
    assert serverless_runner.isaac_eula_env() == {}

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    assert serverless_runner.isaac_eula_env() == {"OMNI_KIT_ACCEPT_EULA": "YES"}


# --------------------------------------------------------------------------------------
# Bootstrap ordering in SkyPilot setup blocks
# --------------------------------------------------------------------------------------


def _templates_asserting_the_isaac_tree() -> list[Path]:
    marker = "test -f /workspace/isaaclab/scripts"
    return sorted(
        path for path in SKYPILOT_DIR.glob("*.yaml") if marker in path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("path", _templates_asserting_the_isaac_tree(), ids=lambda p: p.name)
def test_isaac_tree_assertion_comes_after_the_bootstrap_is_triggered(path: Path) -> None:
    """Asserting the Isaac Lab tree exists before fetching it is a guaranteed FAILED_SETUP.

    The isaaclab wheel ships the library but no ``scripts/``, so /workspace/isaaclab is
    populated on first use of the interpreter — not by the image. A real
    ``sky launch`` of isaac-lab-rl-train-rtxpro-smoke failed exactly this way: the setup
    block ran ``test -f .../rsl_rl/train.py`` before anything had invoked
    ``/isaac-sim/python.sh``.
    """
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"(?m)^\s*test -f /workspace/isaaclab/scripts\S*", text):
        preceding = text[: match.start()]
        # The bootstrap fires on any invocation of the Isaac interpreter.
        last_setup = preceding.rfind("setup:")
        assert last_setup != -1, f"{path.name}: tree assertion outside a setup block"
        block = preceding[last_setup:]
        assert '"${PYTHON_BIN}"' in block or "isaac-bootstrap" in block, (
            f"{path.name}: asserts the Isaac Lab tree before anything triggers the "
            f"bootstrap that fetches it — this fails setup every time"
        )


def test_the_shared_serverless_builder_forwards_acceptance() -> None:
    """It belongs in the SHARED builder, not one caller.

    Every CLI serverless path (isaac_lab, groot, genesis, cosmos, fiftyone) and the
    golden-eval runner go through build_serverless_job_env, so putting the forwarding there
    is what makes `npa workbench isaac-lab train --runtime serverless` work too. An earlier
    version fixed only the golden-eval runner and left the CLI path broken.
    """
    import os

    from npa.serverless_common.env import build_serverless_job_env

    previous = {var: os.environ.get(var) for var in EULA_VARS}
    try:
        for var in EULA_VARS:
            os.environ[var] = "YES"
        env = build_serverless_job_env(output_path="s3://b/p/")
        for var in EULA_VARS:
            assert env[var] == "YES", var

        for var in EULA_VARS:
            del os.environ[var]
        env = build_serverless_job_env(output_path="s3://b/p/")
        for var in EULA_VARS:
            assert var not in env, f"{var} must stay absent, not become an empty string"
    finally:
        for var, value in previous.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def test_an_explicit_caller_value_beats_the_forwarded_one(monkeypatch) -> None:
    """extra_env is applied after the forward, so a caller can still override."""
    from npa.serverless_common.env import build_serverless_job_env

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    env = build_serverless_job_env(
        output_path="s3://b/p/", extra_env={"OMNI_KIT_ACCEPT_EULA": "no"}
    )
    assert env["OMNI_KIT_ACCEPT_EULA"] == "no"


#: CLI modules that can launch an Isaac image and therefore must carry acceptance.
#: Deliberately an explicit list rather than "every module that calls create_job": that
#: broader assertion also catches npa/src/npa/cli/workbench/lerobot.py, which hand-rolls
#: its serverless env. That is a real inconsistency and it means lerobot's submitter also
#: misses the standard S3/HF wiring — but LeRobot cannot run an Isaac image, so fixing it
#: is unrelated to this change and is reported rather than bundled in here.
ISAAC_CAPABLE_CLI_SUBMITTERS = (
    "cli/isaac_lab/__init__.py",
    "cli/groot/__init__.py",
)


@pytest.mark.parametrize("relative", ISAAC_CAPABLE_CLI_SUBMITTERS)
def test_isaac_capable_cli_submitters_use_the_shared_builder(relative: str) -> None:
    """A submitter that hand-rolls its env silently loses the acceptance forwarding.

    These are the CLI paths that can launch an Isaac image, and they are what the
    e2e_serverless tests drive. Going through build_serverless_job_env is what makes
    `npa workbench isaac-lab train --runtime serverless` work post-re-architecture.
    """
    path = REPO_ROOT / "npa" / "src" / "npa" / relative
    assert path.is_file(), path
    text = path.read_text(encoding="utf-8")
    assert "create_job(" in text, f"{relative} no longer submits a serverless job"
    assert "build_serverless_job_env" in text, (
        f"{relative} submits a serverless job without build_serverless_job_env, so it "
        f"will not carry the operator's Isaac EULA acceptance and the job will exit 78"
    )


# --------------------------------------------------------------------------------------
# K8s sim2real Isaac sibling jobs
# --------------------------------------------------------------------------------------

SIM2REAL_ISAAC_BUILDERS = (
    ("byo_isaac_eval", "build_isaac_eval_job_manifest"),
    ("byo_isaac_trainer", "build_isaac_job_manifest"),
    ("byo_isaac_policy_rollout", "build_isaac_rollout_job_manifest"),
)

_MANIFEST_KWARGS = {
    "byo_isaac_eval": dict(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:t", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4, checkpoint_uri="s3://b/m.pt", per_env_s3_uri="s3://b/p.json",
        s3_endpoint="https://s3.example", namespace="default", service_account="sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
    "byo_isaac_trainer": dict(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:t", task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64, iterations=10, s3_output_uri="s3://b/o/", s3_endpoint="https://s3.example",
        namespace="default", service_account="sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
    "byo_isaac_policy_rollout": dict(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:t", task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=2, steps_per_rollout=4, checkpoint_uri="s3://b/m.pt",
        out_s3_prefix="s3://b/o", s3_endpoint="https://s3.example", namespace="default",
        service_account="sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    ),
}


def _job_env(module_name: str, builder_name: str) -> dict[str, str]:
    import importlib

    module = importlib.import_module(f"npa.workflows.sim2real.{module_name}")
    manifest = getattr(module, builder_name)(**_MANIFEST_KWARGS[module_name])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return {entry["name"]: entry["value"] for entry in container.get("env", [])}


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_sim2real_isaac_jobs_forward_acceptance(monkeypatch, module_name, builder) -> None:
    """These jobs invoke /isaac-sim/python.sh, so without acceptance they exit 78."""
    for var in EULA_VARS:
        monkeypatch.setenv(var, "YES")
    env = _job_env(module_name, builder)
    for var in EULA_VARS:
        assert env.get(var) == "YES", f"{module_name}: {var} not forwarded into the job"


@pytest.mark.parametrize(("module_name", "builder"), SIM2REAL_ISAAC_BUILDERS)
def test_sim2real_isaac_jobs_do_not_invent_acceptance(monkeypatch, module_name, builder) -> None:
    """Unset must stay unset so the job fails with the refusal, not silent consent."""
    for var in EULA_VARS:
        monkeypatch.delenv(var, raising=False)
    env = _job_env(module_name, builder)
    for var in EULA_VARS:
        assert var not in env, f"{module_name}: {var} must not be invented"


def _generic_heldout_job_env(
    monkeypatch: pytest.MonkeyPatch, *, accepted: bool
) -> dict[str, str]:
    from npa.workflows.sim2real import engine
    from npa.workflows.sim2real_loop import Sim2RealLoopConfig

    for var in EULA_VARS:
        if accepted:
            monkeypatch.setenv(var, "YES")
        else:
            monkeypatch.delenv(var, raising=False)
    manifest = engine._component_job_manifest(
        "reg/npa-isaac-lab:t",
        component="heldout_eval",
        env={},
        config=Sim2RealLoopConfig(run_id="r", sim_backend="isaac"),
        namespace="default",
        job_name="j",
        timeout_s=60,
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return {entry["name"]: entry["value"] for entry in container["env"]}


def test_generic_sim2real_heldout_job_forwards_operator_acceptance(monkeypatch) -> None:
    env = _generic_heldout_job_env(monkeypatch, accepted=True)
    assert {var: env[var] for var in EULA_VARS} == dict.fromkeys(EULA_VARS, "YES")


def test_generic_sim2real_heldout_job_does_not_invent_acceptance(monkeypatch) -> None:
    env = _generic_heldout_job_env(monkeypatch, accepted=False)
    for var in EULA_VARS:
        assert var not in env


def test_isaac_preflight_refuses_before_remote_sim2real_provisioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.serverless_common.env import MissingIsaacEulaAcceptanceError
    from npa.workflows.sim2real import engine
    from npa.workflows.sim2real_loop import Sim2RealLoopConfig

    for var in EULA_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        engine.sys,
        "argv",
        ["-m", "npa.cli.main", "workbench", "sim2real", "run", "--run-id", "r"],
    )
    output_dir = tmp_path / "must-not-exist"
    config = Sim2RealLoopConfig(
        run_id="r",
        output_dir=output_dir,
        sim_backend="isaac",
        k8s_context="remote-context",
    )

    with pytest.raises(MissingIsaacEulaAcceptanceError) as caught:
        engine.run_preamble(config)

    message = str(caught.value)
    assert "No expensive action has begun" in message
    assert "NVIDIA Omniverse Licence Agreement" in message
    assert "NVIDIA Isaac Sim Additional Software and Materials Licence" in message
    assert "NVIDIA Software Licence Agreement" in message
    assert (
        "OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES "
        f"{engine.sys.executable} -m npa.cli.main workbench sim2real run --run-id r"
    ) in message
    assert not output_dir.exists()


def test_isaac_preflight_accepts_only_documented_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.serverless_common.env import (
        MissingIsaacEulaAcceptanceError,
        require_isaac_eula_acceptance,
    )

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "yes")
    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "TRUE")
    require_isaac_eula_acceptance(context="test", resume_command="npa test")

    monkeypatch.setenv("ISAACSIM_ACCEPT_EULA", "no")
    with pytest.raises(MissingIsaacEulaAcceptanceError, match="ISAACSIM_ACCEPT_EULA"):
        require_isaac_eula_acceptance(context="test", resume_command="npa test")
