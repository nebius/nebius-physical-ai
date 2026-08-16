"""Run-scoped NVIDIA licence acceptance on the way to the SONIC trainer."""

from __future__ import annotations

import inspect

from npa.cli.workbench.sonic.train import train_cmd
from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG
from npa.workbench.sonic.train import (
    NVIDIA_EULA_ENV,
    _run_entrypoint_trainer,
    train_local,
)


def test_only_the_official_runtime_variable_is_forwarded() -> None:
    assert NVIDIA_EULA_ENV == ("ACCEPT_EULA",)


def test_acceptance_reaches_the_entrypoint_trainer() -> None:
    for function in (train_local, _run_entrypoint_trainer):
        assert "accept_eula" in inspect.signature(function).parameters, (
            function.__name__
        )


def test_acceptance_is_on_by_default_everywhere_it_appears() -> None:
    for function in (train_cmd, train_local, _run_entrypoint_trainer):
        parameter = inspect.signature(function).parameters["accept_eula"]
        default = getattr(parameter.default, "default", parameter.default)
        assert default is True


def test_workflow_catalog_does_not_manufacture_acceptance() -> None:
    argv = [str(part) for part in TOOL_CATALOG["workbench.sonic.train"].argv_template]
    assert "--accept-eula" not in argv
    assert "--accept-nvidia-eula" not in argv
