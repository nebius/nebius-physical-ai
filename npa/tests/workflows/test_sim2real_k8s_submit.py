"""Tests for canonical Sim2Real runbook direct-Kubernetes submission."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _operator():
    from npa.workflows.sim2real.monitor import OperatorConfig

    return OperatorConfig(
        bucket="unit-bucket",
        endpoint_url="https://storage.example.invalid",
        registry="registry.unit.test/team",
        k8s_context="unit-context",
    )


def _patch_operator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from npa.workflows.sim2real import k8s_submit

    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    monkeypatch.setattr(k8s_submit, "load_operator_config", _operator)
    monkeypatch.setattr(k8s_submit, "resolve_kubeconfig", lambda _context: kubeconfig)


def test_operator_config_prefers_canonical_registry_over_legacy_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real.monitor import load_operator_config

    config_dir = tmp_path / ".npa"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """container_registry: registry.canonical.test/team
storage:
  bucket: unit-bucket
  endpoint_url: https://storage.unit.test
  registry: registry.legacy.test/team
  k8s_context: unit-context
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)

    assert load_operator_config().registry == "registry.canonical.test/team"


def test_is_sim2real_runbook_accepts_only_committed_canonical_file(tmp_path: Path) -> None:
    from npa.orchestration.npa_workflow.detect import detect_submit_format
    from npa.workflows.sim2real.k8s_submit import is_sim2real_runbook
    from npa.workflows.sim2real.materialize import default_runbook_path

    runbook = default_runbook_path()
    lookalike = tmp_path / "sim2real" / "runbook-copy.yaml"
    lookalike.parent.mkdir()
    lookalike.write_text(runbook.read_text(encoding="utf-8"), encoding="utf-8")

    assert is_sim2real_runbook(runbook)
    assert detect_submit_format(runbook) == "sim2real_runbook"
    assert not is_sim2real_runbook(lookalike)
    assert detect_submit_format(lookalike) == "skypilot"


def test_plan_only_materializes_qualified_real_images_without_external_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    monkeypatch.setattr(
        k8s_submit,
        "_stage_orchestrator_source",
        lambda **_kwargs: pytest.fail("plan-only must not upload source"),
    )
    monkeypatch.setattr(
        k8s_submit,
        "_apply_manifest",
        lambda *_args, **_kwargs: pytest.fail("plan-only must not apply a Job"),
    )

    result = k8s_submit.submit_sim2real_staged_job(
        run_id="unit-plan",
        trigger_dataset_uri="s3://unit-bucket/trigger/",
        plan_only=True,
    )

    assert result.status == "planned"
    manifest = yaml.safe_load(Path(result.manifest_path).read_text(encoding="utf-8"))
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert manifest["metadata"]["name"] == "sim2real-unit-plan"
    assert container["image"].startswith("registry.unit.test/team/")
    assert env["NPA_SIM2REAL_SOURCE_TARBALL_URI"].startswith("s3://unit-bucket/")
    for key in k8s_submit._REQUIRED_REAL_IMAGE_ENVS:
        assert k8s_submit._registry_qualified(env[key]), (key, env[key])


def test_submit_stages_source_refreshes_all_images_and_applies_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit, registry_auth

    _patch_operator(monkeypatch, tmp_path)
    staged: list[dict[str, object]] = []
    refreshed: list[tuple[str, ...]] = []
    applied: list[Path] = []
    monkeypatch.setattr(
        k8s_submit,
        "_stage_orchestrator_source",
        lambda **kwargs: staged.append(kwargs) or "s3://unit-bucket/source/current.tgz",
    )
    monkeypatch.setattr(
        registry_auth,
        "ensure_registry_pull_secret_for_images",
        lambda *images, **_kwargs: refreshed.append(tuple(images)),
    )
    monkeypatch.setattr(
        k8s_submit,
        "_apply_manifest",
        lambda path, **_kwargs: applied.append(path),
    )

    result = k8s_submit.submit_sim2real_staged_job(
        run_id="unit-submit",
        trigger_dataset_uri="s3://unit-bucket/trigger/",
        env_overrides={
            "BYO_TRAINER_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_trainer",
            "BYO_POLICY_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_policy_rollout",
            "BYO_EVAL_COMMAND": "python3 -m npa.workflows.sim2real.byo_isaac_eval",
        },
    )

    assert result.status == "submitted"
    assert len(staged) == 1
    assert len(refreshed) == 1
    assert len(refreshed[0]) >= len(k8s_submit._REQUIRED_REAL_IMAGE_ENVS)
    assert applied == [Path(result.manifest_path)]


def test_unqualified_real_component_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from npa.workflows.sim2real import k8s_submit

    _patch_operator(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="AUGMENT_IMAGE must be a registry-qualified image"):
        k8s_submit.submit_sim2real_staged_job(
            run_id="unit-bad-image",
            env_overrides={"AUGMENT_IMAGE": "npa-cosmos2-transfer:latest"},
            plan_only=True,
        )


def test_workflow_var_aliases_route_to_runbook_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from npa.workflows.sim2real import k8s_submit

    captured: dict[str, object] = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return k8s_submit.Sim2RealSubmitResult(
            run_id="unit-vars",
            job_name="sim2real-unit-vars",
            k8s_context="unit-context",
            run_prefix_uri="s3://unit-bucket/custom/unit-vars/",
        )

    monkeypatch.setattr(k8s_submit, "submit_sim2real_staged_job", fake_submit)
    k8s_submit.submit_sim2real_from_workflow_vars(
        run_id="unit-vars",
        substitutions={
            "bucket": "unit-bucket",
            "prefix": "custom",
            "trigger_uri": "s3://unit-bucket/trigger/",
            "INNER_ITERATIONS": "3",
            "OUTER_ITERATIONS": "2",
        },
        plan_only=True,
    )

    assert captured["s3_bucket"] == "unit-bucket"
    assert captured["s3_prefix"] == "custom"
    assert captured["trigger_dataset_uri"] == "s3://unit-bucket/trigger/"
    assert captured["inner_iterations"] == 3
    assert captured["outer_iterations"] == 2
    assert captured["plan_only"] is True
