from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from npa.cli.agent import _auth_secret_path
from npa.cli.agent_deployment import (
    AgentConfig,
    DeploymentIdentityError,
    agent_lifecycle_lock,
    assert_live_deployment,
    assert_remote_owner_if_present,
    assert_record_ownership,
    build_deployment_manifest,
    load_runtime_deployment,
    verify_remote_deployment,
)
from npa.cli import agent_deployment, agent_public
from npa.deploy import provisioner


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_public_agent_policy_is_deliberately_reexported_from_one_source() -> None:
    assert agent_public.AgentConfig is agent_deployment.AgentConfig
    assert agent_public.build_agent_urls is agent_deployment.build_agent_urls
    assert agent_public.record_public_https is agent_deployment.record_public_https
    assert agent_public.record_tls_verify is agent_deployment.record_tls_verify
    assert agent_public.record_customer_url is agent_deployment.record_customer_url


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-b", "codex/wan-pr261")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(
        repo, "remote", "add", "origin", "git@github.com:nebius/nebius-physical-ai.git"
    )
    (repo / "tracked.txt").write_text("exact source\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "source")
    return repo


def _manifest(source_repo: Path, *, name: str = "wan-pr261") -> dict[str, str]:
    return build_deployment_manifest(
        project_alias="project-a",
        name=name,
        workspace_label="Wan Workbench",
        repo_root=source_repo,
        bootstrap_timestamp="2026-08-10T00:00:00Z",
    )


def test_manifest_captures_exact_immutable_git_source(source_repo: Path) -> None:
    manifest = _manifest(source_repo)
    assert manifest["repository"] == "nebius/nebius-physical-ai"
    assert manifest["branch"] == "codex/wan-pr261"
    assert manifest["commit"] == _git(source_repo, "rev-parse", "HEAD")
    assert manifest["source_tree"] == _git(source_repo, "rev-parse", "HEAD^{tree}")
    assert manifest["short_commit"] == manifest["commit"][:12]
    assert manifest["workspace_label"] == "Wan Workbench"


def test_manifest_uses_ci_head_branch_in_detached_checkout(
    source_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git(source_repo, "checkout", "--detach", "HEAD")
    monkeypatch.setenv("GITHUB_HEAD_REF", "codex/wan-pr261")
    monkeypatch.setenv("GITHUB_REF_NAME", "261/merge")

    manifest = _manifest(source_repo)

    assert manifest["branch"] == "codex/wan-pr261"
    assert manifest["commit"] == _git(source_repo, "rev-parse", "HEAD")


def test_manifest_has_stable_detached_fallback_without_ci_metadata(
    source_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = _git(source_repo, "rev-parse", "HEAD")
    _git(source_repo, "checkout", "--detach", commit)
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)

    manifest = _manifest(source_repo)

    assert manifest["branch"] == f"detached@{commit[:12]}"
    assert manifest["deployment_id"] == _manifest(source_repo)["deployment_id"]


def test_dirty_checkout_cannot_claim_immutable_commit(source_repo: Path) -> None:
    (source_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    with pytest.raises(DeploymentIdentityError, match="checkout is dirty"):
        _manifest(source_repo)


def test_two_agent_names_have_distinct_ownership_and_storage_paths(
    source_repo: Path,
) -> None:
    first = _manifest(source_repo, name="wan-pr261")
    second = _manifest(source_repo, name="other-pr")
    assert first["deployment_id"] != second["deployment_id"]
    assert first["runtime_namespace"] != second["runtime_namespace"]
    assert provisioner.working_dir_path(
        "project-a", "wan-pr261"
    ) != provisioner.working_dir_path("project-a", "other-pr")
    assert _auth_secret_path("project-a", "wan-pr261") != _auth_secret_path(
        "project-a", "other-pr"
    )
    backend = provisioner._BACKEND_TF_TEMPLATE
    first_key = backend.format(
        bucket="b", project="project-a", name="wan-pr261", endpoint="e", region="r"
    )
    second_key = backend.format(
        bucket="b", project="project-a", name="other-pr", endpoint="e", region="r"
    )
    assert "project-a/wan-pr261/terraform.tfstate" in first_key
    assert "project-a/other-pr/terraform.tfstate" in second_key
    assert first_key != second_key


def test_record_owner_mismatch_and_legacy_record_fail_loudly(source_repo: Path) -> None:
    expected = _manifest(source_repo)
    with pytest.raises(DeploymentIdentityError, match="no immutable deployment owner"):
        assert_record_ownership({"public_ip": "203.0.113.1"}, expected)
    wrong = dict(expected)
    wrong["branch"] = "codex/other-pr"
    with pytest.raises(DeploymentIdentityError, match="owner mismatch.*branch"):
        assert_record_ownership({"deployment": wrong}, expected)


def test_live_commit_mismatch_fails_loudly(source_repo: Path) -> None:
    expected = _manifest(source_repo)
    actual = dict(expected)
    actual["commit"] = "f" * 40
    with pytest.raises(DeploymentIdentityError, match="identity mismatch.*commit"):
        assert_live_deployment(expected, actual)


def test_remote_verifier_rejects_wrong_runtime(source_repo: Path) -> None:
    expected = _manifest(source_repo)
    actual = dict(expected)
    actual["deployment_id"] = "npa-agent-contaminated"

    class FakeSsh:
        def run_or_raise(self, command: str, **_kwargs: object) -> tuple[int, str, str]:
            assert "--retry 30 --retry-connrefused" in command
            assert command.endswith("-fsS http://127.0.0.1:8787/deployment")
            return 0, json.dumps(actual), ""

    with pytest.raises(DeploymentIdentityError, match="deployment_id"):
        verify_remote_deployment(FakeSsh(), expected)


def test_existing_remote_owner_is_checked_before_bootstrap(source_repo: Path) -> None:
    expected = _manifest(source_repo)
    actual = dict(expected)
    actual["branch"] = "codex/other-pr"

    class FakeSsh:
        def run(self, command: str, **_kwargs: object) -> tuple[int, str, str]:
            assert command == "curl -fsS http://127.0.0.1:8787/deployment"
            return 0, json.dumps(actual), ""

    with pytest.raises(DeploymentIdentityError, match="owner mismatch.*branch"):
        assert_remote_owner_if_present(FakeSsh(), expected)


def test_backend_down_still_checks_persisted_manifest(source_repo: Path) -> None:
    expected = _manifest(source_repo)
    actual = dict(expected)
    actual["branch"] = "codex/other-pr"

    class FakeSsh:
        def run(self, command: str, **_kwargs: object) -> tuple[int, str, str]:
            if command.startswith("curl "):
                return 7, "", "backend stopped"
            assert "deployment.json" in command
            return 0, json.dumps(actual), ""

    with pytest.raises(DeploymentIdentityError, match="owner mismatch.*branch"):
        assert_remote_owner_if_present(FakeSsh(), expected)


def test_repository_manifest_redacts_remote_credentials(source_repo: Path) -> None:
    _git(
        source_repo,
        "remote",
        "set-url",
        "origin",
        "https://user:secret@example.com/org/repo.git?token=private",
    )
    manifest = _manifest(source_repo)
    assert manifest["repository"] == "example.com/org/repo.git"
    assert "user" not in manifest["repository"]
    assert "secret" not in manifest["repository"]
    assert "private" not in manifest["repository"]


def test_lifecycle_lock_serializes_same_namespace(tmp_path: Path) -> None:
    first_entered = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with agent_lifecycle_lock("project-a", "wan-pr261", lock_root=tmp_path):
            first_entered.set()
            release_first.wait()

    def second() -> None:
        first_entered.wait()
        second_attempting.set()
        with agent_lifecycle_lock("project-a", "wan-pr261", lock_root=tmp_path):
            second_entered.set()

    one = threading.Thread(target=first)
    two = threading.Thread(target=second)
    one.start()
    two.start()
    assert first_entered.wait(2)
    assert second_attempting.wait(2)
    assert not second_entered.is_set()
    release_first.set()
    one.join(2)
    two.join(2)
    assert not one.is_alive()
    assert not two.is_alive()
    assert second_entered.is_set()


def test_runtime_manifest_and_agent_record_preserve_provenance(
    source_repo: Path, tmp_path: Path
) -> None:
    expected = _manifest(source_repo)
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert load_runtime_deployment(path) == expected
    config = AgentConfig(
        project_alias="project-a",
        name="wan-pr261",
        project_id="project-id",
        tenant_id="tenant-id",
        region="us-central1",
        public_ip="203.0.113.1",
        instance_id="instance-id",
        agent_url="https://203.0.113.1/",
        rerun_url="https://203.0.113.1/rerun/",
        sim_viz_url="https://203.0.113.1/rerun/",
        sim_assets_url="https://203.0.113.1/assets/",
        cameras_api_url="https://203.0.113.1/assets/api/sim-assets/cameras",
        auth_user="npa",
        auth_secret_path="/private/auth.env",
        llm_provider="token_factory",
        llm_model="model",
        deployment=expected,
    )
    assert config.to_dict()["deployment"] == expected
    assert config.to_dict()["preload_stock_demo"] is True
