"""Hermetic tests for fail-closed live verification and drift diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from npa.live_verification.token_factory_contract import (
    JSON_PROMPT,
    LIGHTNING,
    MINIMAX,
    model_reference,
    response_evidence,
    run_contract,
    structured_behavior,
)


def _runner():
    path = Path(__file__).resolve().parents[2] / "scripts/token_factory_live_recheck.py"
    spec = importlib.util.spec_from_file_location("live_recheck_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(model=LIGHTNING, content="4, because at most three draws are blue.", reasoning=""):
    return {
        "id": "synthetic-request-id",
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {
            "content": content, "reasoning_content": reasoning,
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": 2 if reasoning else 0}},
    }


class FakeProvider:
    def __init__(self, *, json_healthy=False, ignored_controls=False):
        self.calls = []
        self.json_healthy = json_healthy
        self.ignored_controls = ignored_controls

    def list_models(self):
        return [LIGHTNING, MINIMAX]

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        control = (kwargs.get("extra") or {}).get("chat_template_kwargs", {})
        thinking = control.get("enable_thinking") or control.get("thinking_mode") == "enabled"
        reasoning = "synthetic reasoning" if thinking and not self.ignored_controls else ""
        content = "4, because at most three draws are blue."
        if kwargs["messages"][0]["content"] == JSON_PROMPT:
            content = json.dumps({"score": 0.75, "success": True,
                                  "summary": "red square inside green outline"})
            if kwargs["response_format"] and not self.json_healthy:
                content = "{bad-prefix" + content
        return _response(kwargs["model"], content, reasoning)


def test_contract_actually_infers_defaults_and_observes_both_controls():
    client = FakeProvider()
    report = run_contract(client)
    assert report["passed"]
    assert len(client.calls) == 9
    assert {tuple((call.get("extra") or {}).get("chat_template_kwargs", {}).items())
            for call in client.calls} >= {
        (("enable_thinking", False),), (("enable_thinking", True),),
        (("thinking_mode", "disabled"),), (("thinking_mode", "enabled"),),
    }
    assert all("max_tokens" not in call for call in client.calls)


def test_json_mode_improvement_is_visible_drift_until_reviewed():
    report = run_contract(FakeProvider(json_healthy=True))
    drift = [check for check in report["checks"] if check.get("observed_json_behavior") == "healthy"]
    assert not report["passed"]
    assert {check["check"] for check in drift} == {"json_object", "json_schema", "prompted_json_workaround"}
    assert sum(not check["passed"] for check in drift) == 2
    assert run_contract(FakeProvider(json_healthy=True), expected_json_behavior="healthy")["passed"]


def test_ignored_thinking_controls_fail():
    report = run_contract(FakeProvider(ignored_controls=True))
    assert not report["passed"]
    assert sum("thinking_not_enabled" in check.get("errors", []) for check in report["checks"]) == 2


def test_missing_configured_model_fails_catalog_and_still_requests_it():
    client = FakeProvider()
    report = run_contract(client, additional_models=("synthetic/required-model",))
    assert not report["passed"]
    assert report["checks"][0]["missing_models"] == [model_reference("synthetic/required-model")]
    assert "synthetic/required-model" not in json.dumps(report)
    assert any(call["model"] == "synthetic/required-model" for call in client.calls)


@pytest.mark.parametrize("mutation,error", [
    (lambda r: r.update(model="unexpected-private-model"), "served_model_mismatch"),
    (lambda r: r["choices"][0].update(finish_reason="length"), "incomplete_visible_output"),
    (lambda r: r["choices"][0]["message"].update(content=""), "incomplete_visible_output"),
    (lambda r: r.update(usage={}), "missing_positive_usage"),
    (lambda r: r.update(id=None), "missing_request_identity"),
])
def test_response_evidence_rejects_incomplete_provider_contract(mutation, error):
    response = _response()
    mutation(response)
    report = response_evidence(response, LIGHTNING)
    assert error in report["errors"]
    serialized = json.dumps(report)
    assert "unexpected-private-model" not in serialized
    assert "synthetic-request-id" not in serialized


@pytest.mark.parametrize("value,expected", [
    ('{"score":.75}', "malformed_json"),
    ('```json\n{}\n```', "malformed_json"),
    ('{"score":true,"success":true,"summary":"red square inside green outline"}', "schema_invalid"),
    ('{"score":0.75,"success":true,"summary":"red square inside green outline"}', "healthy"),
])
def test_structured_classifier_never_repairs_original(value, expected):
    assert structured_behavior(value) == expected


def test_errors_omit_provider_body_endpoint_and_credential():
    client = FakeProvider()

    def failure(**kwargs):
        raise RuntimeError("private endpoint with synthetic credential value")

    client.chat_completion = failure
    report = run_contract(client)
    assert not report["passed"]
    assert "synthetic credential" not in json.dumps(report)
    assert all(check.get("error_type") == "RuntimeError" for check in report["checks"][1:])


def _completed_results():
    runner = _runner()
    results = runner.Results()
    results.collected = [suite + "::test_live" for suite in runner.SUITES]
    for nodeid in results.collected:
        results.pytest_runtest_logreport(SimpleNamespace(
            nodeid=nodeid, when="call", passed=True, failed=False, skipped=False,
            user_properties=[("provider_contract", {"passed": True})],
        ))
    return results


def test_counts_require_every_suite_executed_with_no_skips():
    results = _completed_results()
    assert results.complete(0)
    nodeid = results.collected[0]
    results.pytest_runtest_logreport(SimpleNamespace(
        nodeid=nodeid, when="teardown", passed=False, failed=False, skipped=True,
        user_properties=[],
    ))
    assert not results.complete(0)
    assert results.summary() == {
        "collected": 3, "executed": 3, "passed": 2, "failed": 0,
        "skipped": 1, "collection_errors": 0,
    }


def test_no_collection_or_missing_suite_cannot_pass():
    assert not _runner().Results().complete(0)
    results = _completed_results()
    results.collected.pop()
    assert not results.complete(0)


def test_pytest_rootdir_relative_suite_ids_and_prefix_impostors():
    results = _completed_results()
    results.collected = [node.removeprefix("npa/") for node in results.collected]
    assert results.complete(0)
    results.collected[0] = "unrelated/" + results.collected[0]
    assert not results.complete(0)


def test_required_job_missing_key_writes_failed_receipt_without_running_pytest(monkeypatch, tmp_path):
    runner = _runner()
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_KEY", raising=False)
    monkeypatch.setattr(runner.pytest, "main", lambda *a, **k: pytest.fail("pytest must not run"))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    assert runner.main(["--evidence-dir", str(tmp_path / "receipt")]) == 1
    target = tmp_path / "receipt/receipt.json"
    report = json.loads(target.read_text())
    assert not report["passed"] and not report["credential_present"]
    assert report["counts"]["executed"] == report["counts"]["skipped"] == 0
    assert "NEBIUS_TOKEN_FACTORY_KEY" in report["failure"]
    assert target.stat().st_mode & 0o777 == 0o600


def test_receipt_cannot_overwrite_prior_run(tmp_path):
    runner = _runner()
    path = tmp_path / "receipt.json"
    runner.write_receipt(path, {"passed": False})
    with pytest.raises(FileExistsError):
        runner.write_receipt(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": False}


def test_receipt_confidentiality_guard_runs_before_serialization(tmp_path):
    runner = _runner()
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="confidentiality guard"):
        runner.write_receipt(path, {"scope": "u00" + "0" * 20})
    assert not path.exists()


def test_pytest_exception_emits_failed_receipt_without_exception_body(monkeypatch, tmp_path):
    runner = _runner()
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "synthetic-test-key")
    monkeypatch.setenv("NPA_TF_RECHECK_SCOPE", "private-account-name")
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "a" * 40)
    monkeypatch.chdir(Path(__file__).resolve().parents[3])

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic-test-key private endpoint")

    monkeypatch.setattr(runner.pytest, "main", fail)
    assert runner.main(["--evidence-dir", str(tmp_path / "receipt")]) == 1
    saved = (tmp_path / "receipt/receipt.json").read_text()
    assert "synthetic-test-key" not in saved and "private-account-name" not in saved
    assert json.loads(saved)["error_type"] == "RuntimeError"


def test_direct_required_live_pytest_rejects_absent_key_before_collection(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    result = subprocess.run([
        sys.executable, "-m", "pytest", "npa/tests/e2e/test_token_factory_e2e.py",
        "--require-token-factory-live", "--collect-only", "-q",
    ], cwd=root, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Required live Token Factory mode needs NEBIUS_TOKEN_FACTORY_KEY" in result.stderr


@pytest.mark.parametrize("pollute_parent", [False, True], ids=["isolated", "polluted-parent"])
def test_actual_pytest_failure_diagnostics_exclude_private_exception_data(monkeypatch, tmp_path, pollute_parent):
    import textwrap

    root = Path(__file__).resolve().parents[3]
    if pollute_parent:
        # Full-suite collection already imports this real module under the same
        # basename as the synthetic fixture. In-process pytest then interrupts
        # collection with an import-file mismatch instead of exercising reports.
        path = root / "npa/tests/orchestration/skypilot/test_diagnostics.py"
        spec = importlib.util.spec_from_file_location("test_diagnostics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, "test_diagnostics", module)
        monkeypatch.setenv("PYTEST_ADDOPTS", "--invalid-parent-pytest-option")
        monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_parent_pytest_plugin")
        monkeypatch.setattr(pytest, "main", lambda *a, **k: pytest.fail("nested pytest must use its own interpreter"))

    source = textwrap.dedent("""\
        import pytest
        class PrivateProviderExceptionName(Exception):
            pass
        @pytest.fixture
        def broken_setup():
            raise ModuleNotFoundError("PRIVATE_PROVIDER_BODY setup")
        @pytest.fixture
        def broken_teardown():
            yield
            raise RuntimeError("PRIVATE_PROVIDER_BODY teardown")
        def test_call():
            assert False, "PRIVATE_PROVIDER_BODY assertion"
        def test_setup(broken_setup):
            pass
        def test_teardown(broken_teardown):
            pass
        def test_custom_exception():
            raise PrivateProviderExceptionName("PRIVATE_PROVIDER_BODY custom")
    """)
    (tmp_path / "test_diagnostics.py").write_text(source)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    worker = tmp_path / "run_diagnostics.py"
    worker.write_text(textwrap.dedent("""\
        import importlib.util
        from pathlib import Path
        import sys
        import pytest

        spec = importlib.util.spec_from_file_location("live_diagnostics_runner", sys.argv[1])
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.SUITES = ("test_diagnostics.py",)
        results = runner.Results()
        exit_code = pytest.main([
            "test_diagnostics.py", "-q", "--tb=short", "-c", "pytest.ini",
            "--confcutdir=.", "-p", "no:cacheprovider",
        ], plugins=[results])
        runner.write_receipt(Path("failure-receipt.json"), {
            "pytest_exit_code": int(exit_code), "counts": results.summary(),
            "tests": list(results.reports.values()),
        })
        raise SystemExit(exit_code)
    """))
    result = subprocess.run([
        sys.executable, str(worker), str(root / "npa/scripts/token_factory_live_recheck.py"),
    ], cwd=tmp_path, env={
        "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }, text=True, capture_output=True)
    receipt = tmp_path / "failure-receipt.json"
    assert result.returncode == 1 and receipt.is_file(), result.stdout + result.stderr
    saved = receipt.read_text()
    report = json.loads(saved)
    assert report["pytest_exit_code"] == 1
    assert report["counts"] == {
        "collected": 4, "executed": 3, "passed": 0, "failed": 4,
        "skipped": 0, "collection_errors": 0,
    }
    expected = {
        "test_call": ("call", "AssertionError", 'assert False, "PRIVATE_PROVIDER_BODY assertion"'),
        "test_setup": ("setup", "ModuleNotFoundError", 'raise ModuleNotFoundError("PRIVATE_PROVIDER_BODY setup")'),
        "test_teardown": ("teardown", "RuntimeError", 'raise RuntimeError("PRIVATE_PROVIDER_BODY teardown")'),
        "test_custom_exception": ("call", "other_exception", 'raise PrivateProviderExceptionName("PRIVATE_PROVIDER_BODY custom")'),
    }
    for row in report["tests"]:
        phase, exception_type, failing_line = expected[row["nodeid"].split("::")[-1]]
        assert row["diagnostics"] == [{
            "phase": phase, "exception_type": exception_type,
            "source": "test_diagnostics.py",
            "source_line": [line.strip() for line in source.splitlines()].index(failing_line) + 1,
        }]
    assert "PRIVATE_PROVIDER_BODY" not in saved
    assert "PrivateProviderExceptionName" not in saved
    assert str(tmp_path) not in saved


def test_workflow_limits_credentialed_code_to_reviewed_branches():
    import yaml

    root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load((root / ".github/workflows/token-factory-live.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch", "push"}
    assert triggers["schedule"] == [{"cron": "17 6 * * *"}]
    job = workflow["jobs"]["token-factory-live"]
    assert job["environment"] == "token-factory-live"
    assert "refs/heads/main" in job["if"]
    assert len(triggers["push"]["branches"]) == 1
    assert "refs/heads/" + triggers["push"]["branches"][0] in job["if"]
    assert workflow["permissions"] == {"contents": "read"}
    install = next(step for step in job["steps"] if step.get("name") == "Install isolated test runtime")
    assert "uvicorn websockets" in install["run"]
    uploads = [step for step in job["steps"] if "upload-artifact" in step.get("uses", "")]
    assert uploads[0]["with"]["path"].splitlines() == [
        "${{ runner.temp }}/token-factory-live/receipt.json",
        "${{ runner.temp }}/token-factory-missing-key/receipt.json",
    ]
    negative = next(step for step in job["steps"] if step.get("name") == "Prove a missing key fails closed")
    assert negative["env"] == {"NEBIUS_TOKEN_FACTORY_KEY": ""}
    assert 'receipt["pytest_exit_code"] == 2' in negative["run"]
    assert 'all(value == 0 for value in receipt["counts"].values())' in negative["run"]


def test_migration_agent_defaults_match_shared_client():
    """Detect drift in modules embedded into the separately deployed agent."""
    from npa.cli import agent_routing
    from npa.cli.agent import DEFAULT_LLM_MODEL, DEFAULT_LLM_MODELS
    from npa.clients.token_factory import DEFAULT_REASONER_MODEL, DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODEL

    assert agent_routing.CHEAP_MODEL == agent_routing.STANDARD_MODEL == DEFAULT_LLM_MODEL == DEFAULT_TEXT_MODEL
    assert agent_routing.VISION_MODEL == DEFAULT_VISION_MODEL
    assert agent_routing.REASONING_MODEL == DEFAULT_REASONER_MODEL
    assert tuple(DEFAULT_LLM_MODELS) == (DEFAULT_TEXT_MODEL, DEFAULT_REASONER_MODEL)
