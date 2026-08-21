"""Submit should find the durable cache rather than be told about it."""

from __future__ import annotations

import json
import subprocess

import pytest

from npa.orchestration.npa_workflow import model_cache_preflight as preflight


def _fake_kubectl(monkeypatch: pytest.MonkeyPatch, *, phase: str, returncode: int = 0):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        calls.append(list(argv))
        payload = json.dumps({"status": {"phase": phase}}) if phase else ""
        return subprocess.CompletedProcess(argv, returncode, payload, "")

    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    return calls


def test_a_bound_claim_is_adopted_without_being_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_kubectl(monkeypatch, phase="Bound")
    environ: dict[str, str] = {}

    assert preflight.adopt_model_cache_claim(context="ctx", environ=environ) == (
        "npa-model-cache"
    )
    assert environ["NPA_MODEL_CACHE_PVC"] == "npa-model-cache"
    assert calls[0][:4] == ["/usr/bin/kubectl", "--context", "ctx", "get"]


def test_a_pending_claim_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Pending claim has no volume behind it.

    Adopting it would mount something that cannot bind and leave every pod in
    ContainerCreating -- worse than the ephemeral default it replaced.
    """

    _fake_kubectl(monkeypatch, phase="Pending")
    environ: dict[str, str] = {}

    assert preflight.adopt_model_cache_claim(context="ctx", environ=environ) == ""
    assert environ == {}


def test_a_cluster_with_no_claim_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_kubectl(monkeypatch, phase="", returncode=1)
    environ: dict[str, str] = {}

    assert preflight.adopt_model_cache_claim(context="ctx", environ=environ) == ""
    assert environ == {}


def test_a_cluster_that_cannot_answer_never_blocks_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")

    def explode(argv, **kwargs):  # noqa: ANN001, ANN003
        raise subprocess.TimeoutExpired(argv, 20)

    monkeypatch.setattr(preflight.subprocess, "run", explode)
    environ: dict[str, str] = {}

    assert preflight.adopt_model_cache_claim(context="ctx", environ=environ) == ""
    assert environ == {}


def test_without_kubectl_the_lookup_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    assert preflight.find_model_cache_claim(context="ctx") == ""


@pytest.mark.parametrize(
    "configured",
    [
        {"NPA_MODEL_CACHE_PVC": "operator-weights"},
        {"NPA_MODEL_CACHE_HOST_PATH": "/mnt/weights"},
        {"NPA_MODEL_CACHE_DIR": "/mnt/shared/weights"},
        {"NPA_MODEL_CACHE_DISABLED": "1"},
    ],
)
def test_an_operators_own_choice_is_never_overridden(
    monkeypatch: pytest.MonkeyPatch, configured: dict[str, str]
) -> None:
    calls = _fake_kubectl(monkeypatch, phase="Bound")
    environ = dict(configured)

    assert preflight.adopt_model_cache_claim(context="ctx", environ=environ) == ""
    assert environ == configured
    assert calls == []


def test_the_namespace_can_be_pointed_somewhere_other_than_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim only helps pods that can see it.

    An operator whose SkyPilot pods live outside `default` must be able to say so,
    or the lookup quietly finds nothing and the run pays for the download again.
    """

    calls = _fake_kubectl(monkeypatch, phase="Bound")
    monkeypatch.setenv("NPA_MODEL_CACHE_NAMESPACE", "workbench")
    environ: dict[str, str] = {}

    preflight.adopt_model_cache_claim(context="ctx", environ=environ)

    argv = calls[0]
    assert argv[argv.index("-n") + 1] == "workbench"


def test_an_explicit_namespace_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_kubectl(monkeypatch, phase="Bound")
    monkeypatch.setenv("NPA_MODEL_CACHE_NAMESPACE", "workbench")

    preflight.find_model_cache_claim(context="ctx", namespace="explicit")

    argv = calls[0]
    assert argv[argv.index("-n") + 1] == "explicit"
