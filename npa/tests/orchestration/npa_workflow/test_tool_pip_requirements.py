"""A toolRef may declare third-party CLIs it shells out to.

`npa workbench cosmos fetch` runs `huggingface-cli`. The retired `cosmos3-ea-fetch.yaml`
pip-installed `huggingface_hub[cli]` in its setup — one line that turned out to be the only
load-bearing part of its ~60-line preamble. The twin dropped it and the stage failed live with
``checkpoint download failed: [Errno 2] No such file or directory: 'huggingface-cli'``
(job 226), after `check-access` had already SUCCEEDED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.skypilot_render import (
    TOOL_REF_PIP_REQUIREMENTS,
    SkypilotRenderOptions,
    assert_no_unresolved_placeholders,
    render_pip_requirements_setup,
    render_skypilot_yaml,
    tool_pip_requirements,
)
from npa.orchestration.npa_workflow.spec import load_spec

SPECS = Path(__file__).resolve().parents[4] / "npa" / "workflows" / "workbench" / "npa-workflows"


def test_requirements_resolve_by_exact_ref_and_by_prefix() -> None:
    assert tool_pip_requirements("workbench.cosmos.fetch") == (
        ("huggingface-cli", "huggingface_hub[cli]>=0.23,<1.0"),
    )
    # An unrelated tool declares nothing.
    assert tool_pip_requirements("workbench.mjlab.eval") == ()
    assert tool_pip_requirements("") == ()


def test_install_is_conditional_on_the_executable_being_absent() -> None:
    """A purpose-built image that already ships the CLI must be left alone."""

    setup = render_pip_requirements_setup(tool_pip_requirements("workbench.cosmos.fetch"))

    assert "if ! command -v huggingface-cli >/dev/null 2>&1; then" in setup
    assert "-m pip install -q 'huggingface_hub[cli]>=0.23,<1.0'" in setup
    # Same PEP 668 fallbacks default_npa_setup uses: plain, --break-system-packages, --user.
    # Four mentions: the echo plus the three PEP 668 fallbacks (plain,
    # --break-system-packages, --user) that default_npa_setup also uses.
    assert setup.count("huggingface_hub[cli]>=0.23,<1.0") == 4


def test_no_requirements_renders_nothing() -> None:
    assert render_pip_requirements_setup(()) == ""


def test_every_declared_requirement_names_an_executable_and_a_spec() -> None:
    for tool_ref, requirements in TOOL_REF_PIP_REQUIREMENTS.items():
        assert requirements, tool_ref
        for executable, requirement in requirements:
            assert executable and " " not in executable, (tool_ref, executable)
            # A pip requirement, not a bare import name.
            assert any(marker in requirement for marker in "=<>[") or requirement.isidentifier()


def test_shipped_cosmos_spec_renders_the_installer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "cosmos-fetch.yaml")
    plan = build_plan(spec, run_id="pip-requirements-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="pip-requirements-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    stages = [doc for doc in docs if "run" in doc]
    assert len(stages) == 2
    for doc in stages:
        assert "huggingface_hub[cli]" in doc["setup"], doc["name"]
    assert_no_unresolved_placeholders(text)


def test_a_library_requirement_is_probed_by_import_not_by_command_v() -> None:
    """`huggingface_hub` has no binary, and the shim's interpreter is not a vendor venv.

    Live job 244: the LeRobot producer materialised its dataset with `huggingface_hub` and died
    with "huggingface_hub is required to download the example dataset" — the stage ran
    `/home/sky/miniconda3/bin/python3` (where npa is installed), not the image's
    `/opt/lerobot/venv`.
    """

    setup = render_pip_requirements_setup(tool_pip_requirements("workbench.lerobot.policy_train"))

    assert "-c 'import huggingface_hub' >/dev/null 2>&1; then" in setup
    assert "command -v" not in setup
    assert "-m pip install -q 'huggingface_hub>=0.23,<1.0'" in setup


def test_executable_and_module_probes_can_coexist() -> None:
    setup = render_pip_requirements_setup(
        (("huggingface-cli", "huggingface_hub[cli]"), ("python:numpy", "numpy>=1.24"))
    )

    assert "command -v huggingface-cli" in setup
    assert "-c 'import numpy'" in setup


def test_shipped_lerobot_spec_installs_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "tokenfactory-train-triage.yaml")
    plan = build_plan(spec, run_id="pip-requirements-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="pip-requirements-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    train = next(doc for doc in docs if doc.get("name", "").endswith("train-gpu"))
    assert "import huggingface_hub" in train["setup"]
    assert_no_unresolved_placeholders(text)


# ------------------------------------------------------------- vendor interpreters


def test_vendor_interpreter_resolves_by_prefix() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import tool_vendor_interpreters

    assert tool_vendor_interpreters("workbench.lerobot.policy_train") == (
        "/opt/lerobot/venv/bin/python",
    )
    assert tool_vendor_interpreters("workbench.mjlab.eval") == ()
    # Live job 267: npa went into /usr/bin/python3 and the capture stage died with
    # `No module named 'isaaclab'` — the simulator lives in the Omniverse kit environment.
    assert tool_vendor_interpreters("workbench.isaac_lab.capture_frames") == (
        "/isaac-sim/python.sh",
        "/isaac-sim/kit/python/bin/python3",
    )


def test_vendor_setup_installs_npa_there_and_records_it() -> None:
    """Live job 245: `LeRobot import failed: No module named 'lerobot'`.

    Setup installs npa into whatever `python3` resolves to — SkyPilot's miniconda — while the
    vendor image keeps LeRobot in `/opt/lerobot/venv`. The retired template avoided this with
    `source /opt/lerobot/venv/bin/activate`; the engine now installs npa into the vendor
    interpreter and records it, so the tool and the vendor library share one environment.
    """

    from npa.orchestration.npa_workflow.skypilot_render import (
        render_vendor_interpreter_setup,
        tool_vendor_interpreters,
    )

    setup = render_vendor_interpreter_setup(
        tool_vendor_interpreters("workbench.lerobot.policy_train")
    )

    assert "for npa_vendor_python in /opt/lerobot/venv/bin/python; do" in setup
    # --no-deps FIRST: resolving npa's requirements inside a vendor venv can bump torch and
    # break the vendor's own compiled extensions (live job 253, a torchcodec ABI mismatch).
    no_deps = setup.index("-m pip install -q --no-deps -e")
    # ... and a with-deps attempt only AFTER it, for vendor environments that carry none of
    # npa's dependencies (live job 268: Isaac's kit python, where --no-deps alone left
    # npa.workbench unimportable). Order is the whole safety property.
    with_deps = setup.index('-m pip install -q -e "$npa_vendor_src"')
    assert no_deps < with_deps
    # The second attempt is guarded by the probe, so it never runs when the first sufficed.
    assert setup.count("if ! \"$npa_vendor_python\" -c 'import npa.workbench'") == 2
    # Records it as THE stage interpreter, which is what the run shim reads.
    assert 'echo "$npa_vendor_python" > /tmp/npa-python' in setup
    # Probes a real subpackage: a vendor image may bake a PARTIAL npa on PYTHONPATH that makes
    # `import npa` pass while `import npa.workbench` fails (live job 250).
    assert "-c 'import npa.workbench'" in setup
    # Skips a candidate that is not present, rather than failing the stage.
    assert '[ -x "$npa_vendor_python" ] || continue' in setup
    # And when it gives up on a candidate it says why: job 268's bare warning blamed a
    # shadowing partial npa when the cause was missing dependencies.
    assert "\"$npa_vendor_python\" -c 'import npa.workbench' 2>&1 | tail -3 >&2" in setup
    assert "${" not in setup


def test_no_vendor_interpreter_renders_nothing() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import render_vendor_interpreter_setup

    assert render_vendor_interpreter_setup(()) == ""


def test_requirements_install_into_the_recorded_interpreter() -> None:
    """A library installed into miniconda is invisible to the vendor interpreter."""

    setup = render_pip_requirements_setup(tool_pip_requirements("workbench.lerobot.policy_train"))

    assert 'npa_req_python="$(cat /tmp/npa-python)"' in setup
    assert '"$npa_req_python" -c \'import huggingface_hub\'' in setup
    assert '"$npa_req_python" -m pip install -q' in setup


def test_lerobot_stage_switches_interpreter_before_installing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "tokenfactory-train-triage.yaml")
    plan = build_plan(spec, run_id="vendor-check")

    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="vendor-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )

    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    train = next(doc for doc in docs if doc.get("name", "").endswith("train-gpu"))
    setup = train["setup"]
    assert setup.index("npa_vendor_python") < setup.index("npa_req_python")
    assert_no_unresolved_placeholders(text)


def test_the_npa_console_script_is_looked_for_in_the_user_scheme_too() -> None:
    """`npa_pip_install` falls back to `--user` under PEP 668, which moves the script.

    Live job 260: the judge stage died with `bash: npa: command not found` on an image where
    that fallback fired, because the symlink step only consulted the default scripts dir.
    """

    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    setup = default_npa_setup()

    assert 'scheme=\"posix_user\"' in setup
    assert '"$HOME/.local/bin"' in setup
    assert "ln -sf" in setup


def test_the_npa_console_script_is_shimmed_to_the_recorded_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live job 284: `No such command 'cosmos2'. Did you mean 'cosmos'?`

    The cosmos2-transfer image bakes its own npa, whose console script is first on PATH. Setup
    saw `command -v npa` succeed and skipped installing; the overlay went into the vendor
    interpreter; the stage then ran a stale CLI against a fresh library. `python3` was already
    shimmed to the recorded interpreter, so `npa` has to be too — otherwise the two disagree
    about which install they mean, which is exactly the bug.
    """

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/prefix/npa")
    spec = load_spec(SPECS / "cosmos2-transfer.yaml")
    plan = build_plan(spec, run_id="npa-shim-check")
    text = render_skypilot_yaml(
        spec,
        plan,
        run_id="npa-shim-check",
        options=SkypilotRenderOptions(image_overrides={"*": ""}),
    )
    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    run_script = next(doc["run"] for doc in docs if doc.get("run"))

    assert "> /tmp/npa-shim/python3" in run_script
    assert "> /tmp/npa-shim/npa" in run_script
    # A baked npa source tree on PYTHONPATH shadows every install (live job 285:
    # PYTHONPATH=/opt/npa/src in the cosmos2-transfer image, holding an npa with no `cosmos2`),
    # so the recorded source goes in front of it.
    assert 'export PYTHONPATH="$npa_src_path:$PYTHONPATH"' in run_script
    assert "${" not in run_script, "the placeholder guard rejects braced expansions"
    assert "from npa.cli.main import app_entry" in run_script
    # Both shims come from the same recorded interpreter.
    assert run_script.count('"$npa_python"') >= 2


def test_the_source_overlay_installs_dependencies_when_the_cli_will_not_load() -> None:
    """Live job 309: `ModuleNotFoundError: No module named 'paramiko'` after a clean overlay.

    An image that installs npa with its own curated `--no-deps` list leaves the overlay short
    of whatever that list omitted. `--no-deps` is still the right FIRST attempt — the overlay
    is the same distribution the image already has, and resolving its requirements can move a
    pinned vendor stack (job 253's torch-ABI break) — but it cannot be the only one.
    """

    from npa.orchestration.npa_workflow.skypilot_render import default_npa_setup

    setup = default_npa_setup()

    first = setup.index("npa_pip_install -e /tmp/npa-src-overlay --no-deps")
    guard = setup.index("import npa.cli.main")
    second = setup.index("npa_pip_install -e /tmp/npa-src-overlay\n")
    assert first < guard < second, "the with-deps attempt must be guarded and come second"
    # `import npa` is not the right probe: it succeeded in job 309. The command tree is.
    assert "python3 -c 'import npa.cli.main'" in setup
