"""Prove namespace ownership, access scope, and private context handling."""

from __future__ import annotations

import json
import stat

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.workbench.namespace import app
from npa.clients.kube import KubectlResult
from npa.clients import kubernetes_namespace as resolver
from npa.workbench import namespaces


@pytest.mark.parametrize(
    "name",
    [
        "",
        "UPPER",
        "a.b",
        "-team",
        "team-",
        "a" * 64,
        "default",
        "kube-system",
        "skypilot-system",
    ],
)
def test_invalid_or_system_namespaces_are_rejected(name):
    with pytest.raises(ValueError):
        namespaces.namespace_manifests(name)


def test_access_is_scoped_and_cluster_discovery_is_read_only():
    manifests = namespaces.namespace_manifests(
        "team-a", users=("researcher",), groups=("researchers",)
    )
    role = next(item for item in manifests if item["kind"] == "ClusterRole")
    assert all(set(rule["verbs"]) <= {"get", "list", "watch"} for rule in role["rules"])
    assert not any(
        "pods" in rule["resources"] or "secrets" in rule["resources"]
        for rule in role["rules"]
    )
    for binding in (item for item in manifests if item["kind"] == "RoleBinding"):
        assert binding["metadata"]["namespace"] == "team-a"
        assert binding["roleRef"]["name"] == "edit"
    assert (
        len(
            {
                item["metadata"]["name"]
                for item in manifests
                if item["kind"] == "ClusterRole"
            }
        )
        == 1
    )
    other = namespaces.namespace_manifests("team-b")
    assert role["metadata"]["name"] != next(
        item["metadata"]["name"] for item in other if item["kind"] == "ClusterRole"
    )


@pytest.mark.parametrize(
    "subject", ["", " someone", "system:authenticated", "line\nbreak"]
)
def test_unsafe_subjects_are_rejected(subject):
    with pytest.raises(ValueError):
        namespaces.namespace_manifests("team-a", users=(subject,))


def test_dry_run_is_offline_and_complete(monkeypatch):
    monkeypatch.setattr(
        namespaces,
        "run_kubectl",
        lambda *args, **kwargs: pytest.fail("network during dry run"),
    )
    result = CliRunner().invoke(
        app, ["apply", "team-a", "--context", "cluster", "--user", "alice", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert len(payload["manifests"]) == 6


def test_foreign_objects_prevent_all_mutation(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return KubectlResult(0, json.dumps({"metadata": {"labels": {}}}))

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    with pytest.raises(ValueError, match="unmanaged Namespace"):
        namespaces.apply_namespace("team-a", context="cluster")
    assert all(args[0] == "get" for args in calls)


def test_reapply_reconciles_members_without_force_or_other_namespaces(monkeypatch):
    manifests = namespaces.namespace_manifests("team-a", users=("old-member",))
    objects = {(item["kind"], item["metadata"]["name"]): item for item in manifests}
    applied = []

    def run(args, **kwargs):
        assert kwargs["context"] == "cluster"
        if args[0] == "get":
            return KubectlResult(0, json.dumps(objects[(args[1], args[2])]))
        assert "--force-conflicts" not in args
        applied.extend(json.loads(kwargs["stdin"])["items"])
        return KubectlResult(0)

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    namespaces.apply_namespace("team-a", context="cluster", users=("new-member",))
    assert "old-member" not in json.dumps(applied)
    assert "new-member" in json.dumps(applied)
    assert all(
        item["metadata"].get("namespace", "team-a") == "team-a" for item in applied
    )


def test_private_context_preserves_identity_and_source(monkeypatch, tmp_path):
    document = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": "cluster",
        "contexts": [
            {"name": "cluster", "context": {"cluster": "cluster", "user": "own-user"}}
        ],
        "users": [{"name": "own-user", "user": {"token": "synthetic-test-token"}}],
    }
    original = json.dumps(document)

    def run(args, **kwargs):
        return KubectlResult(0, original if args[0] == "config" else "namespace/team-a")

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    output = tmp_path / "team"
    result = namespaces.write_namespace_context(
        "team-a", context="cluster", output_dir=output
    )
    selected = yaml.safe_load((output / "kubeconfig").read_text())
    assert selected["contexts"][0]["context"] == {
        "cluster": "cluster",
        "user": "own-user",
        "namespace": "team-a",
    }
    assert selected["users"] == document["users"]
    assert json.dumps(document) == original
    assert "synthetic-test-token" not in json.dumps(result)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "kubeconfig").stat().st_mode) == 0o600
    assert (
        yaml.safe_load((output / "sky.yaml").read_text())["kubernetes"][
            "remote_identity"
        ]
        == "npa-workbench"
    )
    with pytest.raises(FileExistsError):
        namespaces.write_namespace_context(
            "team-a", context="cluster", output_dir=output
        )


@pytest.mark.parametrize(
    "configured,expected", [(None, "default"), ("team-a", "team-a")]
)
def test_effective_namespace_matches_context(monkeypatch, configured, expected):
    settings = {"namespace": configured} if configured else {}
    payload = {"contexts": [{"name": "cluster", "context": settings}]}
    monkeypatch.setattr(
        resolver,
        "run_kubectl",
        lambda *args, **kwargs: KubectlResult(0, json.dumps(payload)),
    )
    assert resolver.context_namespace(context="cluster") == expected


@pytest.mark.parametrize(
    "result",
    [
        KubectlResult(1, stderr="credential-private"),
        KubectlResult(0, "{}"),
        KubectlResult(0, "not-json"),
    ],
)
def test_namespace_resolution_fails_closed_without_private_output(monkeypatch, result):
    monkeypatch.setattr(resolver, "run_kubectl", lambda *args, **kwargs: result)
    with pytest.raises(ValueError) as error:
        resolver.context_namespace(context="cluster")
    assert "credential-private" not in str(error.value)
