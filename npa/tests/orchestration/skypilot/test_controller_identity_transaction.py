from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.cluster.identity import ClusterIdentityError
from npa.orchestration.skypilot import cleanup as controller
from npa.orchestration.skypilot.cleanup import CleanupResult
from npa import teardown_receipts


def _identity(tmp_path: Path):  # noqa: ANN001
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("current-context: verified-context\n", encoding="utf-8")
    return SimpleNamespace(
        project_alias="demo",
        project_id="project-demo",
        context="verified-context",
        cluster_id="cluster-demo",
        cluster_name="cluster-demo",
        kubeconfig=kubeconfig,
        cluster_absent=False,
        receipt_identity=lambda: {
            "project_alias": "demo",
            "project_id": "project-demo",
            "context": "verified-context",
            "cluster_id": "cluster-demo",
            "cluster_name": "cluster-demo",
            "cluster_absent": False,
        },
    )


@pytest.fixture()
def identity_fixture(monkeypatch, tmp_path: Path):  # noqa: ANN001
    identity = _identity(tmp_path)
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "teardown-receipts"))
    monkeypatch.setattr(
        "npa.cluster.identity.resolve_verified_cluster_identity",
        lambda **kwargs: identity,
    )
    monkeypatch.setattr(controller, "_nonterminal_job_ids", lambda **kwargs: [])
    return identity


def _row(name: str, *, context: str = "verified-context") -> dict[str, str]:
    return {"name": name, "cloud": "Kubernetes", "context": context}


def test_identity_failure_happens_before_remote_or_local_mutation(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.setenv("NPA_TEARDOWN_RECEIPT_DIR", str(tmp_path / "teardown-receipts"))
    monkeypatch.setattr(
        "npa.cluster.identity.resolve_verified_cluster_identity",
        lambda **kwargs: (_ for _ in ()).throw(
            ClusterIdentityError("context belongs to another project")
        ),
    )
    monkeypatch.setattr(
        controller,
        "_kubernetes_controller_pods",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("remote inspection")),
    )
    monkeypatch.setattr(
        controller,
        "_down_jobs_controller",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mutation")),
    )

    result = controller.cleanup_jobs_controller(
        project="demo", context="wrong-context", isolated_config_dir=tmp_path
    )

    assert not result.ok
    assert "another project" in result.errors[0]
    assert (
        teardown_receipts.latest_phase_states(project_alias="demo")["controller"][
            "terminal_state"
        ]
        == "verification_failed"
    )


@pytest.mark.parametrize(
    "error",
    [
        "Forbidden: RBAC denied list pods",
        "Unauthorized: authentication required",
        "connection refused",
        "request timed out",
    ],
)
def test_uncertain_remote_state_preserves_local_metadata(
    monkeypatch, identity_fixture, tmp_path: Path, error: str
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        controller, "_kubernetes_controller_pods", lambda **kwargs: ([], error)
    )
    local_calls: list[str] = []
    monkeypatch.setattr(
        controller,
        "_down_jobs_controller",
        lambda name, **kwargs: local_calls.append(name) or CleanupResult(),
    )

    result = controller.cleanup_jobs_controller(
        project="demo", context="verified-context", isolated_config_dir=tmp_path
    )

    assert local_calls == []
    assert not result.ok
    assert "preserved" in " ".join(result.errors)


def test_remote_absence_receipt_precedes_real_local_state_removal(
    monkeypatch, identity_fixture, tmp_path: Path
) -> None:  # noqa: ANN001
    name = "sky-jobs-controller-demo"
    monkeypatch.setattr(
        controller, "_kubernetes_controller_pods", lambda **kwargs: ([], "")
    )
    status_calls = {"count": 0}

    def status(**kwargs):  # noqa: ANN001
        status_calls["count"] += 1
        return ([_row(name)] if status_calls["count"] == 1 else [], "")

    monkeypatch.setattr(controller, "_jobs_controller_clusters", status)
    observed_states: list[str] = []

    def down(target, **kwargs):  # noqa: ANN001
        observed_states.append(
            teardown_receipts.latest_phase_states(project_alias="demo")["controller"][
                "terminal_state"
            ]
        )
        return CleanupResult(resources_removed=[target])

    monkeypatch.setattr(controller, "_down_jobs_controller", down)

    result = controller.cleanup_jobs_controller(
        project="demo", context="verified-context", isolated_config_dir=tmp_path
    )

    assert result.ok
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert observed_states == ["remote_absent_local_pending"]
    assert (
        teardown_receipts.latest_phase_states(project_alias="demo")["controller"][
            "terminal_state"
        ]
        == "verified_absent"
    )


def test_remote_delete_uses_cloned_state_then_verifies_then_mutates_real_state(
    monkeypatch, identity_fixture, tmp_path: Path
) -> None:  # noqa: ANN001
    name = "sky-jobs-controller-demo"
    other = "sky-jobs-controller-unrelated"
    monkeypatch.setattr(
        controller,
        "_kubernetes_controller_pods",
        lambda **kwargs: ([("default", f"{name}-ray-head", name)], ""),
    )
    status_calls = {"count": 0}

    def status(**kwargs):  # noqa: ANN001
        status_calls["count"] += 1
        if status_calls["count"] == 1:
            return [_row(name), _row(other, context="other-context")], ""
        return [_row(other, context="other-context")], ""

    monkeypatch.setattr(controller, "_jobs_controller_clusters", status)
    monkeypatch.setattr(
        controller, "_wait_for_controller_pods_absent", lambda *args, **kwargs: ([], "")
    )
    downs: list[tuple[str, Path | None, str]] = []

    def down(target, **kwargs):  # noqa: ANN001
        latest = teardown_receipts.latest_phase_states(project_alias="demo")[
            "controller"
        ]["terminal_state"]
        downs.append((target, kwargs.get("isolated_config_dir"), latest))
        return CleanupResult(resources_removed=[target])

    monkeypatch.setattr(controller, "_down_jobs_controller", down)

    result = controller.cleanup_jobs_controller(
        project="demo", context="verified-context", isolated_config_dir=tmp_path
    )

    assert result.ok
    assert result.verified is True
    assert result.remote_absence_verified is True
    assert [item[0] for item in downs] == [name, name]
    assert downs[0][1] != tmp_path
    assert downs[0][2] == "in_progress"
    assert downs[1][1] == tmp_path
    assert downs[1][2] == "remote_absent_local_pending"
    assert all(item[0] != other for item in downs)


def test_unrelated_stale_profile_is_never_selected(
    monkeypatch, identity_fixture, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        controller, "_kubernetes_controller_pods", lambda **kwargs: ([], "")
    )
    monkeypatch.setattr(
        controller,
        "_jobs_controller_clusters",
        lambda **kwargs: ([_row("sky-jobs-controller-stale", context="old")], ""),
    )
    downs: list[str] = []
    monkeypatch.setattr(
        controller,
        "_down_jobs_controller",
        lambda name, **kwargs: downs.append(name) or CleanupResult(),
    )

    result = controller.cleanup_jobs_controller(
        project="demo", context="verified-context", isolated_config_dir=tmp_path
    )

    assert result.ok
    assert downs == []


def test_context_prefix_collision_is_not_an_exact_identity_match() -> None:
    assert not controller._controller_belongs_to_context(
        _row("sky-jobs-controller-stale", context="verified-context-old"),
        "verified-context",
    )
    assert controller._controller_belongs_to_context(
        _row("sky-jobs-controller-exact", context="verified-context"),
        "verified-context",
    )


def test_receipt_failure_preserves_remote_and_local_state(
    monkeypatch, identity_fixture, tmp_path: Path
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        teardown_receipts,
        "record_teardown_event",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    remote_calls: list[str] = []
    monkeypatch.setattr(
        controller,
        "_kubernetes_controller_pods",
        lambda **kwargs: remote_calls.append("inspect") or ([], ""),
    )

    result = controller.cleanup_jobs_controller(
        project="demo", context="verified-context", isolated_config_dir=tmp_path
    )

    assert remote_calls == []
    assert not result.ok
    assert "no remote or local mutation" in result.errors[0]
