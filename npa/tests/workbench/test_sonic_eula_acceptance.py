"""The operator's NVIDIA licence acceptance, on the way to the SONIC trainer.

`--runtime local` (from #238) runs the SONIC image's own `/entrypoint.sh train` when it is
present and falls back to a reference locomotion trainer when it is not. Six live jobs found
what stands between those two: the image's trainer **refuses** to download Isaac Sim / Isaac Lab
until NVIDIA's terms are accepted, and SkyPilot's pod does not inherit the image's docker ENV, so
the acceptance has to travel with the request (jobs 323 and 327, EVIDENCE §R47).

Without it, `local` does not fail — it quietly runs the reference trainer instead of the vendor
one, which is a worse outcome than an error.
"""

from __future__ import annotations

import inspect

import pytest

from npa.cli.workbench.sonic.train import _is_affirmative
from npa.workbench.sonic.train import NVIDIA_EULA_ENV, _run_entrypoint_trainer, train_local


@pytest.mark.parametrize("given", ["yes", "YES", " y ", "1", "true", "accept"])
def test_the_operator_can_say_yes(given: str) -> None:
    assert _is_affirmative(given)


@pytest.mark.parametrize("withheld", ["", "  ", "no", "n", "later", "maybe"])
def test_anything_else_is_not_acceptance(withheld: str) -> None:
    assert not _is_affirmative(withheld)


def test_both_variables_the_image_checks_are_named() -> None:
    """The entrypoint refuses unless BOTH are YES; naming one would be a silent half-fix."""

    assert set(NVIDIA_EULA_ENV) == {"OMNI_KIT_ACCEPT_EULA", "ISAACSIM_ACCEPT_EULA"}


def test_acceptance_reaches_the_entrypoint_trainer() -> None:
    for function in (train_local, _run_entrypoint_trainer):
        assert "accept_nvidia_eula" in inspect.signature(function).parameters, function.__name__


def test_acceptance_is_off_by_default_everywhere_it_appears() -> None:
    """Acceptance is the operator's to give, never the tool's to assume."""

    for function in (train_local, _run_entrypoint_trainer):
        assert inspect.signature(function).parameters["accept_nvidia_eula"].default is False


def test_the_cli_takes_a_value_so_a_spec_can_carry_it() -> None:
    """A toolRef argv is always flag-plus-value; a bare boolean flag cannot be expressed."""

    from npa.cli.workbench.sonic.train import train_cmd
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    parameter = inspect.signature(train_cmd).parameters["accept_nvidia_eula"]
    assert parameter.default.default == ""

    argv = [str(part) for part in TOOL_CATALOG["workbench.sonic.train"].argv_template]
    assert argv[argv.index("--accept-nvidia-eula") + 1] == "{{config.sonic_accept_nvidia_eula}}"


def test_no_shipped_spec_accepts_on_the_operators_behalf() -> None:
    from pathlib import Path

    import yaml

    specs = Path(__file__).resolve().parents[3] / "npa/workflows/workbench/npa-workflows"
    found = 0
    for path in specs.glob("sonic*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")).get("config", {})
        if "sonic_accept_nvidia_eula" in config:
            found += 1
            assert config["sonic_accept_nvidia_eula"] == "", path.name
    assert found, "expected the SONIC specs to expose the acceptance key"
