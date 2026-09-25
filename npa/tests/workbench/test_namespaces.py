"""Prove native namespace selection preserves existing access and private config."""

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


@pytest.mark.parametrize("name", ["", "UPPER", "a.b", "-team", "team-", "a" * 64])
def test_invalid_namespace_names_are_rejected(name):
    with pytest.raises(ValueError):
        namespaces.namespace_manifests(name)


def test_dry_run_is_offline_and_contains_only_the_namespace(monkeypatch):
    monkeypatch.setattr(
        namespaces,
        "run_kubectl",
        lambda *args, **kwargs: pytest.fail("network during dry run"),
    )
    result = CliRunner().invoke(
        app, ["apply", "team-a", "--context", "cluster", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert payload["manifests"] == [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "team-a"}}
    ]


@pytest.mark.parametrize("name", ["team-a", "default"])
def test_existing_namespace_is_reused_without_labels_or_access_changes(
    monkeypatch, name
):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return KubectlResult(0, f"namespace/{name}")

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    assert namespaces.apply_namespace(name, context="cluster")["status"] == "existing"
    assert len(calls) == 1 and calls[0][0] == "get"


def test_missing_namespace_is_created_without_rbac(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return KubectlResult(0)

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    result = namespaces.apply_namespace("team-a", context="cluster")
    assert result["status"] == "created"
    assert calls[1][0] == ["create", "-f", "-"]
    assert json.loads(calls[1][1]["stdin"])["kind"] == "Namespace"
    assert len(calls) == 2


def test_denied_namespace_read_never_attempts_creation(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return KubectlResult(1, stderr="private-credential-output")

    monkeypatch.setattr(namespaces, "run_kubectl", run)
    with pytest.raises(ValueError, match="check access") as error:
        namespaces.apply_namespace("team-a", context="cluster")
    assert "private-credential-output" not in str(error.value)
    assert len(calls) == 1


@pytest.fixture
def source_context(monkeypatch):
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
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.setattr(
        namespaces,
        "run_kubectl",
        lambda args, **kwargs: KubectlResult(
            0, original if args[0] == "config" else "namespace/team-a"
        ),
    )
    return document


@pytest.mark.parametrize("name", ["team-a", "default"])
def test_private_context_preserves_identity_and_source(source_context, tmp_path, name):
    original = json.dumps(source_context)
    output = tmp_path / "client"
    result = namespaces.write_namespace_context(
        name, context="cluster", output_dir=output
    )
    selected = yaml.safe_load((output / "kubeconfig").read_text())
    assert selected["contexts"][0]["context"] == {
        "cluster": "cluster",
        "user": "own-user",
        "namespace": name,
    }
    assert selected["users"] == source_context["users"]
    assert json.dumps(source_context) == original
    assert "synthetic-test-token" not in json.dumps(result)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in (output / "kubeconfig", output / "sky.yaml"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    sky = yaml.safe_load((output / "sky.yaml").read_text())
    assert sky["kubernetes"] == {"allowed_contexts": ["cluster"]}
    with pytest.raises(FileExistsError):
        namespaces.write_namespace_context(name, context="cluster", output_dir=output)


@pytest.mark.parametrize("selection", ["explicit", "environment", "default"])
def test_existing_skypilot_identity_and_settings_are_preserved(
    source_context, tmp_path, monkeypatch, selection
):
    source = tmp_path / "sky-source.yaml"
    if selection == "default":
        source = namespaces.Path.home() / ".sky/config.yaml"
        source.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "jobs": {"controller": {"resources": {"cpus": 2}}},
        "kubernetes": {
            "remote_identity": "existing-worker",
            "networking": "portforward",
        },
    }
    source.write_text(yaml.safe_dump(settings))
    if selection == "environment":
        monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", str(source))
    output = tmp_path / "client"
    namespaces.write_namespace_context(
        "team-a",
        context="cluster",
        output_dir=output,
        sky_config=source if selection == "explicit" else None,
    )
    result = yaml.safe_load((output / "sky.yaml").read_text())
    assert result["jobs"] == settings["jobs"]
    assert result["kubernetes"] == {
        **settings["kubernetes"],
        "allowed_contexts": ["cluster"],
    }
    assert yaml.safe_load(source.read_text()) == settings


@pytest.mark.parametrize("invalid", ["[private", "[one, two]", "kubernetes: []"])
def test_invalid_sky_config_fails_before_creating_output(
    source_context, tmp_path, invalid
):
    source = tmp_path / "invalid.yaml"
    source.write_text(invalid)
    with pytest.raises(ValueError) as error:
        namespaces.write_namespace_context(
            "team-a",
            context="cluster",
            output_dir=tmp_path / "client",
            sky_config=source,
        )
    assert "private" not in str(error.value)
    assert not (tmp_path / "client").exists()


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
        KubectlResult(
            0, json.dumps({"contexts": [{"name": "cluster", "context": None}]})
        ),
    ],
)
def test_namespace_resolution_fails_closed_without_private_output(monkeypatch, result):
    monkeypatch.setattr(resolver, "run_kubectl", lambda *args, **kwargs: result)
    with pytest.raises(ValueError) as error:
        resolver.context_namespace(context="cluster")
    assert "credential-private" not in str(error.value)
