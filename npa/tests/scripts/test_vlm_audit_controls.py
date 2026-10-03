"""Check frozen visual controls and protected scheduled audit wiring offline."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml

from npa.live_verification.vlm_audit_controls import (
    CONTROL_PAIRS,
    PREFERENCE_MODELS,
    audit_controls,
    configured_audit_cases,
    generated_preference_config,
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
    config = generated_preference_config(tmp_path / "controls")
    controls = audit_controls(config["cases"]["blinded-preference"])
    assert controls.keys() == CONTROL_PAIRS.keys()
    for name, control in controls.items():
        first, second, expected = CONTROL_PAIRS[name]
        for arm, image_name in (("baseline", first), ("candidate", second)):
            path = Path(control["request"][f"{arm}_path"])
            with Image.open(path) as image:
                assert image.size == (640, 480)
                assert (
                    hashlib.sha256(image.tobytes()).hexdigest()
                    == PIXEL_HASHES[image_name]
                )
            assert path.stat().st_mode & 0o777 == 0o600
            assert (
                control[f"{arm}_sha256"]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )
        assert control["expectations"]["mapped_preferences"] == [expected, expected]
        assert control["expectations"]["escalation_required"] is False
    assert len({case["request"]["task"] for case in controls.values()}) == 1
    assert len({case["request"]["rubric"] for case in controls.values()}) == 1
    assert PREFERENCE_MODELS == ("MiniMaxAI/MiniMax-M3",)


@pytest.mark.parametrize(
    "kind,factory,case",
    [
        ("paired", generated_paired_config, "paired-judges"),
        ("preference", generated_preference_config, "blinded-preference"),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "wrong-schema"])
def test_both_generated_families_require_exact_controls(
    tmp_path, kind, factory, case, mutation
):
    config = factory(tmp_path / "controls")
    available = ("paired-judges", "blinded-preference")
    assert configured_audit_cases(
        config, available_cases=available, required_kind=kind
    ) == (case,)
    controls = config["cases"][case]["controls"]
    first = next(iter(controls))
    if mutation == "missing":
        controls.pop(first)
    elif mutation == "extra":
        controls["extra"] = controls[first]
    elif mutation == "reordered":
        controls[first] = controls.pop(first)
    else:
        config["control_schema"] = "unknown"
    with pytest.raises(ValueError, match="invalid_generated_audit_controls"):
        configured_audit_cases(config, available_cases=available, required_kind=kind)


def test_preference_controls_keep_exact_kind_specific_instructions(tmp_path):
    config = generated_preference_config(tmp_path / "controls")
    for control in audit_controls(config["cases"]["blinded-preference"]).values():
        assert control["request"]["task"] == (
            "Compare how well the visible shapes satisfy the visual requirement."
        )
        assert control["request"]["rubric"] == (
            "Prefer the image with a red square fully inside the green rectangular outline. "
            "A square outside the outline does not satisfy the requirement. "
            "If the images are visually equivalent under this rule, return tie. "
            "Judge only visible shapes, not hidden physical state."
        )


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("count", [True, False, 1.0, "1", None, 0, -1])
def test_live_provider_proof_rejects_nonliteral_positive_counts(field, count):
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    raw = {
        "model": "vendor/vision",
        "choices": [{"finish_reason": "stop", "message": {}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    raw["usage"][field] = count
    body = json.dumps(raw)
    provider = SimpleNamespace(
        status_code=200,
        returned_model="vendor/vision",
        finish_reason="stop",
        provider_request_id="synthetic-request",
        raw_response=body,
        raw_response_sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match=field):
        suite._assert_provider_evidence(provider, "vendor/vision")


def test_generated_config_selects_all_controls_and_private_outputs(
    monkeypatch, tmp_path
):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-key")
    config_path = runner._prepare_config(
        tmp_path, generated=True, audit_kind="preference"
    )
    config = json.loads(config_path.read_text())
    controls = audit_controls(config["cases"]["blinded-preference"])
    outputs = [case["request"]["output_path"] for case in controls.values()]
    assert len(set(outputs)) == 3
    assert all(Path(path).is_relative_to(tmp_path) for path in outputs)
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert "synthetic-key" not in config_path.read_text()
    monkeypatch.setenv(runner.CONFIG_ENV, str(config_path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    assert suite._audit_parameters() == [
        ("blinded-preference", name) for name in CONTROL_PAIRS
    ]


@pytest.mark.parametrize(
    "kind,case", [("paired", "paired-judges"), ("preference", "blinded-preference")]
)
def test_collection_selects_only_configured_kind_in_prospective_registry(
    monkeypatch, tmp_path, kind, case
):
    from npa.live_verification import vlm_audit_controls

    monkeypatch.setattr(
        vlm_audit_controls,
        "AUDIT_CASES_BY_KIND",
        {"paired": "paired-judges", "preference": "blinded-preference"},
    )
    config = {
        "audit_kind": kind,
        "cases": {
            case: {"controls": {name: {} for name in ("first", "second", "third")}}
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("NPA_VLM_AUDIT_LIVE_CONFIG", str(path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    suite = _module("npa/tests/e2e/test_vlm_audits_live.py")
    assert suite._audit_parameters() == [
        (case, name) for name in ("first", "second", "third")
    ]


@pytest.mark.parametrize(
    "cases,kind",
    [
        ({"blinded-preference": {}}, "paired"),
        ({"blinded-preference": {}}, "unknown"),
        ({"blinded-preference": {}}, None),
        ({"blinded-preference": {}, "paired-judges": {}}, "preference"),
        ({"unknown": {}}, "preference"),
        ({}, "preference"),
    ],
)
def test_collection_rejects_wrong_unknown_or_mixed_kind(
    monkeypatch, tmp_path, cases, kind
):
    config = {"audit_kind": kind, "cases": cases}
    with pytest.raises(ValueError, match="invalid_audit_case_selection"):
        configured_audit_cases(
            config,
            available_cases=("paired-judges", "blinded-preference"),
            required_kind=kind,
        )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("NPA_VLM_AUDIT_LIVE_CONFIG", str(path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    with pytest.raises(ValueError, match="invalid_audit_case_selection"):
        _module("npa/tests/e2e/test_vlm_audits_live.py")


def test_generated_lane_does_not_fall_back_to_saved_key(monkeypatch, tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "")
    monkeypatch.setattr(
        runner, "_scheduled_preflight", lambda: pytest.fail("network probe")
    )
    assert (
        runner.main(
            [
                "--generated-controls",
                "--audit-kind",
                "preference",
                "--evidence-dir",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["failure"] == "missing_audit_credential"
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
            return []

    monkeypatch.setattr(runner, "TokenFactoryClient", Client)
    with pytest.raises(ValueError, match="required_preference_model_not_advertised"):
        runner._scheduled_preflight()


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
        if row.get("name") == "Run blinded hosted visual contract controls"
    )
    assert step["env"] == {
        "NEBIUS_TOKEN_FACTORY_KEY": "${{ secrets.NEBIUS_TOKEN_FACTORY_KEY }}"
    }
    assert step["if"] == "${{ !cancelled() }}"
    assert "vlm_audit_live_recheck.py --generated-controls" in step["run"]
    assert "--audit-kind preference" in step["run"]
    upload = next(
        row for row in job["steps"] if "upload-artifact" in row.get("uses", "")
    )
    paths = upload["with"]["path"].splitlines()
    assert all(path.endswith("/receipt.json") and "*" not in path for path in paths)
    assert "${{ runner.temp }}/vlm-preference-audit/receipt.json" in paths
    assert not any("SSH" in key or "GPU" in key or "IAM" in key for key in step["env"])


def test_public_receipt_cannot_contain_private_identifier(tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="confidentiality"):
        runner._write_receipt(path, {"private": "u00" + "0" * 20})
    assert not path.exists()


def test_controls_are_repeatable_without_reusing_existing_files(tmp_path):
    first = generated_preference_config(tmp_path / "first")
    second = generated_preference_config(tmp_path / "second")
    first_controls = audit_controls(first["cases"]["blinded-preference"])
    second_controls = audit_controls(second["cases"]["blinded-preference"])
    for name in CONTROL_PAIRS:
        assert (
            first_controls[name]["baseline_sha256"]
            == second_controls[name]["baseline_sha256"]
        )
    with pytest.raises(FileExistsError):
        generated_preference_config(tmp_path / "first")


def test_public_outcomes_preserve_verdicts_without_provider_text(tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    private = "private-provider-content"
    first = {
        "provider": {"returned_model": private, "raw_response": private},
        "verdict": {
            "preference": "A",
            "confidence": "high",
            "uncertainty": private,
            "observable_support": [private],
            "critical_defects": {"A": [private], "B": [private]},
        },
    }
    second = {"error": {"error_type": "transport_error", "message": private}}
    report = {
        "status": "judge_error",
        "escalation_required": True,
        "requests_counterbalanced": True,
        "task": private,
        "first_order": first,
        "reversed_order": second,
    }
    directory = tmp_path / "blinded-preference/0"
    directory.mkdir(parents=True)
    path = directory / runner.PREFERENCE_COMPARISON_RESULT_FILENAME
    path.write_text(json.dumps(report))
    summaries = runner._public_comparisons(tmp_path, ((0, path, {}),))
    assert private not in json.dumps(summaries)
    assert (
        summaries[0]["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert summaries[0]["escalation_required"] is True
    assert summaries[0]["first_order"]["preference"] == "A"
    assert summaries[0]["first_order"]["confidence"] == "high"
    assert summaries[0]["reversed_order"]["error_type"] == "transport_error"
    second["error"]["error_type"] = private
    first["verdict"]["preference"] = private
    first["verdict"]["confidence"] = private
    with pytest.raises(ValueError, match="Invalid audit error"):
        runner._public_order(second)
    with pytest.raises(ValueError, match="Invalid audit verdict"):
        runner._public_order(first)


def test_generated_lane_rejects_silently_removed_controls(monkeypatch, tmp_path):
    runner = _module("npa/scripts/vlm_audit_live_recheck.py")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-key")
    monkeypatch.setattr(runner, "_scheduled_preflight", lambda: None)

    def execute(root, target, config):
        (target / "pytest.xml").write_text(
            '<testsuites><testsuite><testcase name="one"/></testsuite></testsuites>'
        )
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1,
                    "executed": 1,
                    "deselected": 0,
                    "xfail": False,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "run"
    assert (
        runner.main(
            [
                "--generated-controls",
                "--audit-kind",
                "preference",
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
    for index in (1, 2, 10):
        directory = tmp_path / "blinded-preference" / str(index)
        directory.mkdir(parents=True)
        (directory / runner.PREFERENCE_COMPARISON_RESULT_FILENAME).write_text(
            json.dumps(
                {
                    "status": "judge_error",
                    "requests_counterbalanced": True,
                    "escalation_required": True,
                    "first_order": {"error": {"error_type": "transport_error"}},
                    "reversed_order": {"error": {"error_type": "transport_error"}},
                }
            )
        )
    reports = tuple(
        (
            index,
            tmp_path
            / "blinded-preference"
            / str(index)
            / runner.PREFERENCE_COMPARISON_RESULT_FILENAME,
            {},
        )
        for index in range(11)
    )
    summaries = runner._public_comparisons(tmp_path, reports)
    assert [row["control_index"] for row in summaries] == list(range(11))
    assert [row["control_index"] for row in summaries if "status" in row] == [
        1,
        2,
        10,
    ]


@pytest.fixture(autouse=True)
def _private_artifact_creation():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)
