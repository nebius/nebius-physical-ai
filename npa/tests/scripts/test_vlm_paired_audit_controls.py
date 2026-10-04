"""Check frozen visual controls and protected scheduled audit wiring offline."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

from PIL import Image
import pytest
import yaml

from npa.live_verification import vlm_audit_controls
from npa.workbench.vlm_eval import _comparison_models
from npa.live_verification.vlm_audit_controls import (
    CONTROL_LABELS,
    PAIRED_MODELS,
    audit_controls,
    generated_paired_config,
)

ROOT = Path(__file__).resolve().parents[3]
PIXEL_HASHES = {
    "inside": "6ece94ec5403568dc114a1a79012d789f7dfb4183efb64a098c34cd4b73cba3d",
    "outside": "7d34e6d23d134403aed9dcbfdf619fd56aa086c8b1551bf54b09c0d70009031b",
    "blank": "cd1fb537c37464dc6bc2db970eb5416d81a8d9a72df3ecc98e44d3a083380fef",
}


def _module(path):
    spec = importlib.util.spec_from_file_location("audit_contract_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_controls_have_frozen_pixels_labels_and_shared_prompts(tmp_path):
    config = generated_paired_config(tmp_path / "controls")
    controls = audit_controls(config["cases"]["paired-judges"])
    assert controls.keys() == CONTROL_LABELS.keys()
    for name, control in controls.items():
        path = Path(control["request"]["input_path"])
        with Image.open(path) as image:
            assert image.size == (640, 480)
            assert hashlib.sha256(image.tobytes()).hexdigest() == PIXEL_HASHES[name]
        assert path.stat().st_mode & 0o777 == 0o600
        assert control["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert control["expectations"]["primary.result.passed"] is CONTROL_LABELS[name]
        assert (
            control["expectations"]["secondary.result.passed"] is CONTROL_LABELS[name]
        )
    assert len({case["request"]["task"] for case in controls.values()}) == 1
    assert len({case["request"]["rubric"] for case in controls.values()}) == 1
    assert len(set(PAIRED_MODELS)) == 2
    assert _comparison_models(*PAIRED_MODELS) == PAIRED_MODELS
    assert "moonshotai/Kimi-K3" not in PAIRED_MODELS


def test_generated_paired_controls_keep_exact_namespaced_prompts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vlm_audit_controls, "PREFERENCE_CONTROL_TASK", "unrelated task", raising=False
    )
    monkeypatch.setattr(
        vlm_audit_controls,
        "PREFERENCE_CONTROL_RUBRIC",
        "unrelated rubric",
        raising=False,
    )
    assert "CONTROL_TASK" not in vars(vlm_audit_controls)
    assert "CONTROL_RUBRIC" not in vars(vlm_audit_controls)
    config = generated_paired_config(tmp_path / "controls")
    controls = audit_controls(config["cases"]["paired-judges"])
    for control in controls.values():
        assert control["request"]["task"] == (
            "Describe the visible shapes and their positions. Judge whether a red square "
            "is fully inside the green rectangular outline."
        )
        assert control["request"]["rubric"] == (
            "Score 1 only if a red square is fully inside the green rectangular outline. "
            "Score 0 if the red square is outside, or either shape is absent. "
            "Judge only the submitted image. Return bare JSON without Markdown fences."
        )


def test_generated_config_selects_all_controls_and_private_outputs(
    monkeypatch, tmp_path
):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-key")
    config_path = runner._prepare_config(tmp_path, generated=True, audit_kind="paired")
    config = json.loads(config_path.read_text())
    controls = audit_controls(config["cases"]["paired-judges"])
    outputs = [case["request"]["output_path"] for case in controls.values()]
    assert len(set(outputs)) == 3
    assert all(Path(path).is_relative_to(tmp_path) for path in outputs)
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert "synthetic-key" not in config_path.read_text()
    monkeypatch.setenv(runner.CONFIG_ENV, str(config_path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    assert suite._audit_parameters() == [
        ("paired-judges", name) for name in CONTROL_LABELS
    ]


def test_combined_registry_collects_only_the_configured_generated_kind(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("NPA_VLM_AUDIT_LIVE_CONFIG", raising=False)
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    monkeypatch.setattr(suite, "AUDIT_CASES", ("paired-judges", "blinded-preference"))
    monkeypatch.setitem(
        vlm_audit_controls.AUDIT_CASES_BY_KIND, "preference", "blinded-preference"
    )
    config = generated_paired_config(tmp_path / "controls")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("NPA_VLM_AUDIT_LIVE_CONFIG", str(path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    assert suite._audit_parameters() == [
        ("paired-judges", name) for name in ("inside", "outside", "blank")
    ]


@pytest.mark.parametrize("kind", ["preference", "unknown", None, False])
def test_collection_rejects_mismatched_or_malformed_kind(monkeypatch, tmp_path, kind):
    monkeypatch.delenv("NPA_VLM_AUDIT_LIVE_CONFIG", raising=False)
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    config = generated_paired_config(tmp_path / "controls")
    config["audit_kind"] = kind
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("NPA_VLM_AUDIT_LIVE_CONFIG", str(path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    with pytest.raises(ValueError, match="invalid_audit_case_selection"):
        suite._audit_parameters()


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "reordered", "schema", "missing_kind"]
)
def test_generated_collection_rejects_changed_cardinality_or_order(tmp_path, mutation):
    config = generated_paired_config(tmp_path / "controls")
    controls = config["cases"]["paired-judges"]["controls"]
    if mutation == "missing":
        controls.pop("blank")
    elif mutation == "extra":
        controls["extra"] = controls["blank"]
    elif mutation == "reordered":
        controls["inside"] = controls.pop("inside")
    elif mutation == "missing_kind":
        config.pop("audit_kind")
    else:
        config["control_schema"] = "npa.preference_visual_controls.v1"
    with pytest.raises(ValueError, match="invalid_generated_audit_controls"):
        vlm_audit_controls.configured_audit_cases(
            config, available_cases=("paired-judges",), required_kind="paired"
        )


def test_generated_lane_does_not_fall_back_to_saved_key(monkeypatch, tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "")
    monkeypatch.setattr(
        runner, "_scheduled_paired_preflight", lambda: pytest.fail("network probe")
    )
    assert (
        runner.main(
            [
                "--generated-controls",
                "--audit-kind",
                "paired",
                "--evidence-dir",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["failure"] == "missing_audit_credential"
    assert receipt["audit_kind"] == "paired"
    assert not any(receipt["counts"].values())


def test_generated_preflight_uses_only_explicit_key_and_fixed_endpoint(monkeypatch):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-key")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_BASE_URL", "https://unused.invalid/v1/")

    class Client:
        def __init__(self, config):
            assert config.api_key == "synthetic-key"
            assert config.base_url == runner.DEFAULT_BASE_URL

        def list_models(self):
            return [PAIRED_MODELS[0]]

    monkeypatch.setattr(runner, "TokenFactoryClient", Client)
    with pytest.raises(ValueError, match="required_paired_model_not_advertised"):
        runner._scheduled_paired_preflight()


def test_scheduled_lane_preserves_branch_policy_and_uploads_only_receipts():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/token-factory-live.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["schedule"] == [{"cron": "17 6 * * *"}]
    job = workflow["jobs"]["token-factory-live"]
    assert job["environment"] == {"name": "token-factory-live", "deployment": False}
    assert "refs/heads/main" in job["if"]
    assert "refs/heads/" + triggers["push"]["branches"][0] in job["if"]
    step = next(
        row
        for row in job["steps"]
        if row.get("name") == "Run paired hosted visual contract controls"
    )
    assert step["env"] == {
        "NEBIUS_TOKEN_FACTORY_KEY": "${{ secrets.NEBIUS_TOKEN_FACTORY_KEY }}"
    }
    assert step["if"] == "${{ !cancelled() }}"
    assert (
        "vlm_audit_live_recheck.py --generated-controls --audit-kind paired"
        in step["run"]
    )
    upload = next(
        row for row in job["steps"] if "upload-artifact" in row.get("uses", "")
    )
    paths = upload["with"]["path"].splitlines()
    assert all(path.endswith("/receipt.json") and "*" not in path for path in paths)
    assert "${{ runner.temp }}/vlm-paired-audit/receipt.json" in paths
    assert not any("SSH" in key or "GPU" in key or "IAM" in key for key in step["env"])


def test_public_receipt_cannot_contain_private_identifier(tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="confidentiality"):
        runner._write_receipt(path, {"private": "u00" + "0" * 20})
    assert not path.exists()


def test_controls_are_repeatable_without_reusing_existing_files(tmp_path):
    first = generated_paired_config(tmp_path / "first")
    second = generated_paired_config(tmp_path / "second")
    first_controls = audit_controls(first["cases"]["paired-judges"])
    second_controls = audit_controls(second["cases"]["paired-judges"])
    for name in CONTROL_LABELS:
        assert (
            first_controls[name]["input_sha256"]
            == second_controls[name]["input_sha256"]
        )
    with pytest.raises(FileExistsError):
        generated_paired_config(tmp_path / "first")


def _private_judges(private):
    primary = {
        "model": private,
        "result": {
            "score": 1.0,
            "passed": True,
            "reasoning": private,
            "evidence": {
                "provider": {"returned_model": private, "raw_response": private}
            },
        },
    }
    secondary = {
        "model": private,
        "error": {
            "error_type": "transport_error",
            "message": private,
            "provider": {"raw_response": private},
        },
    }
    return primary, secondary


def test_public_outcomes_preserve_verdicts_without_provider_text(tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    private = "private-provider-content"
    primary, secondary = _private_judges(private)
    report = {
        "status": "judge_error",
        "passed": False,
        "escalation_required": True,
        "task": private,
        "primary": primary,
        "secondary": secondary,
    }
    directory = tmp_path / "paired-judges/0"
    directory.mkdir(parents=True)
    path = directory / runner.JUDGE_COMPARISON_RESULT_FILENAME
    path.write_text(json.dumps(report))
    path.chmod(0o600)
    summaries = runner._public_comparisons(tmp_path)
    assert private not in json.dumps(summaries)
    assert (
        summaries[0]["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert summaries[0]["escalation_required"] is True
    assert summaries[0]["primary"]["score"] == 1.0
    assert summaries[0]["primary"]["passed"] is True
    assert summaries[0]["secondary"]["error_type"] == "transport_error"
    secondary["error"]["error_type"] = private
    primary["result"]["score"] = float("inf")
    with pytest.raises(ValueError, match="invalid_audit_judge_error"):
        runner._public_judge(secondary)
    with pytest.raises(ValueError, match="score must be finite"):
        runner._public_judge(primary)


def test_generated_lane_rejects_silently_removed_controls(monkeypatch, tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-key")
    monkeypatch.setattr(runner, "_scheduled_paired_preflight", lambda: None)

    def execute(root, target, config):
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1,
                    "executed": 1,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "deselected": 0,
                    "xfail": False,
                }
            )
        )
        (target / "execution.json").chmod(0o600)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "run"
    assert (
        runner.main(
            [
                "--generated-controls",
                "--audit-kind",
                "paired",
                "--evidence-dir",
                str(target),
            ]
        )
        == 1
    )
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["counts"]["passed"] == 1
    assert receipt["passed"] is False


def test_public_outcomes_keep_original_indices_after_missing_controls(tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    primary, secondary = _private_judges("synthetic-model")
    for index in (1, 2, 10):
        directory = tmp_path / "paired-judges" / str(index)
        directory.mkdir(parents=True)
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).write_text(
            json.dumps(
                {
                    "status": "judge_error",
                    "passed": False,
                    "escalation_required": True,
                    "primary": primary,
                    "secondary": secondary,
                }
            )
        )
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).chmod(0o600)
    assert [row["control_index"] for row in runner._public_comparisons(tmp_path)] == [
        1,
        2,
        10,
    ]


def test_public_score_rejects_large_integer_without_float_overflow():
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    with pytest.raises(ValueError, match="score must be at most 1"):
        runner._public_judge({"result": {"passed": False, "score": 10**1000}})


@pytest.mark.parametrize("passed", [False, True])
def test_public_judge_preserves_literal_boolean(passed):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    outcome = {
        "model": "synthetic/model",
        "result": {
            "passed": passed,
            "score": 0.5,
            "evidence": {
                "provider": {
                    "returned_model": "synthetic/model",
                    "raw_response": "synthetic",
                }
            },
        },
    }
    assert runner._public_judge(outcome)["passed"] is passed


@pytest.fixture(autouse=True)
def _private_artifact_creation():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)
