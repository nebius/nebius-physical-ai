"""Keep public model execution direct while preserving explicit runtime choices."""

import os
import subprocess

import pytest

from npa.orchestration.npa_workflow.interpreter import PlanStep
from npa.orchestration.npa_workflow.skypilot_render import (
    render_run_preamble_for_tool,
    secret_env_hints_for_plan,
)


TERMS_ENV = "NPA_OPENPI_ACCEPT_GEMMA_TERMS"


@pytest.mark.parametrize(
    ("tool_ref", "argv"),
    [
        ("workbench.openpi.direct", []),
        ("workbench.byof.repo", ["--solution-name", "openpi"]),
        ("workbench.byof.repo", ["pi05_droid_jointpos_polaris"]),
    ],
)
def test_public_checkpoint_execution_needs_no_repeated_acceptance(
    monkeypatch, tool_ref, argv
) -> None:
    monkeypatch.delenv(TERMS_ENV, raising=False)
    step = PlanStep(state="inference", tool_ref=tool_ref, argv=argv)
    assert secret_env_hints_for_plan([step]) == ()


@pytest.mark.parametrize("choice", ["", "NO", "FALSE", "YES", "invalid"])
def test_explicit_runtime_choice_is_hinted_once_without_exposing_value(
    monkeypatch, choice
) -> None:
    monkeypatch.setenv(TERMS_ENV, choice)
    steps = [
        PlanStep(state="direct", tool_ref="workbench.openpi.direct"),
        PlanStep(state="train", tool_ref="workbench.openpi.train"),
    ]
    assert secret_env_hints_for_plan(steps) == (TERMS_ENV,)


def test_openpi_choice_does_not_add_consent_to_unrelated_tools(monkeypatch) -> None:
    monkeypatch.setenv(TERMS_ENV, "NO")
    steps = [
        PlanStep(state="public", tool_ref="workbench.groot.download"),
        PlanStep(state="hosted", tool_ref="workbench.token_factory.generate"),
    ]
    assert secret_env_hints_for_plan(steps) == ("NEBIUS_TOKEN_FACTORY_KEY",)


def test_public_super_execution_needs_only_storage_credentials(monkeypatch) -> None:
    name = "NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE"
    monkeypatch.delenv(name, raising=False)
    steps = [PlanStep(state="benchmark", tool_ref="workbench.cosmos3.super_benchmark")]
    assert secret_env_hints_for_plan(steps) == ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    monkeypatch.setenv(name, "NO")
    assert secret_env_hints_for_plan(steps) == (name, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def test_super_stage_restores_runtime_before_invoking_benchmark() -> None:
    script = render_run_preamble_for_tool("workbench.cosmos3.super_benchmark", config={})
    bootstrap = "/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint --runtime /bin/true"
    export = 'export PATH="/opt/npa-cosmos3-serving/runtime/venv/bin:$PATH"'
    assert bootstrap in script and script.index(bootstrap) < script.index(export)
    assert "ACCEPT_NVIDIA_SOFTWARE_LICENSE=YES" not in script
    assert render_run_preamble_for_tool("workbench.cosmos3.text_to_image", config={}) == ""


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("bootstrap_exit", [0, 78])
def test_super_runtime_hook_preserves_old_wrapper_and_propagates_refusal(
    tmp_path, replacement, bootstrap_exit
) -> None:
    wrapper = tmp_path / "entrypoint"
    wrapper.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" > "$CALL_LOG"\n'
        f"exit {bootstrap_exit}\n"
    )
    wrapper.chmod(0o755)
    marker = tmp_path / "bootstrap-source-sha256s.txt"
    if replacement:
        marker.write_text("reviewed replacement bootstrap\n")
    script = render_run_preamble_for_tool("workbench.cosmos3.super_benchmark", config={})
    script = script.replace("/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint", str(wrapper))
    script = script.replace("/opt/npa-cosmos3-serving/bootstrap-source-sha256s.txt", str(marker))
    log = tmp_path / "call"
    result = subprocess.run(
        ["bash", "-ec", script + "echo workload-started\n"],
        env={**os.environ, "CALL_LOG": str(log)}, capture_output=True, text=True, check=False,
    )
    assert log.exists() is replacement
    if replacement:
        assert log.read_text().strip() == "--runtime /bin/true"
    assert result.returncode == (bootstrap_exit if replacement else 0)
    assert ("workload-started" in result.stdout) == (not replacement or bootstrap_exit == 0)
