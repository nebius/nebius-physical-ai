"""Test provenance lane gates without invoking a model or cloud resource."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest


@pytest.fixture
def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "provenance_runner_test", directory / "vlm_provenance_live_recheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "green").save(frame)
    values = {
        "input_path": str(frame),
        "output_path": str(tmp_path / "result.json"),
        "endpoint_url": "https://example.test/v1",
        "model": "test-model",
        "expected_served_model": "test-model",
        "task": "Describe the frame",
        "api_key_env": "TEST_VLM_KEY",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    path.chmod(0o600)
    monkeypatch.setenv("NPA_VLM_PROVENANCE_LIVE_CONFIG", str(path))
    monkeypatch.setenv("NPA_INTEGRATION_E2E", "1")
    monkeypatch.setenv("TEST_VLM_KEY", "synthetic-test-key")
    return values


@pytest.mark.parametrize(
    "variable", ["NPA_INTEGRATION_E2E", "NPA_VLM_PROVENANCE_LIVE_CONFIG"]
)
def test_missing_live_gate_fails_before_pytest(
    runner, config, monkeypatch, tmp_path, variable
):
    monkeypatch.delenv(variable)
    monkeypatch.setattr(
        runner, "_run_live_tests", lambda *args: pytest.fail("pytest reached")
    )
    target = tmp_path / "receipt"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["passed"] is False
    assert receipt["endpoint_preflight_passed"] is False
    assert receipt["counts"]["executed"] == 0
    assert "example.test" not in json.dumps(receipt)
    assert "synthetic-test-key" not in json.dumps(receipt)


def test_config_requires_owner_only_permissions(runner, config, monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only"):
        runner._load_config(Path.cwd())


@pytest.mark.parametrize(
    "invalid", ["existing-result", "no-frames", "remote-output", "public-output"]
)
def test_fixture_checks_fail_before_endpoint(runner, config, tmp_path, invalid):
    if invalid == "existing-result":
        Path(config["output_path"]).write_text("existing evidence")
    elif invalid == "no-frames":
        Path(config["input_path"]).unlink()
    elif invalid == "remote-output":
        config["output_path"] = "s3://synthetic-bucket/result.json"
    else:
        tmp_path.chmod(0o755)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner._check_local_artifacts(config, Path.cwd())


def test_endpoint_probe_uses_selected_credential_and_exact_model(
    runner, config, monkeypatch
):
    def get(url, **kwargs):
        assert url == "https://example.test/v1/models"
        assert kwargs["headers"] == {"Authorization": "Bearer synthetic-test-key"}
        assert kwargs["follow_redirects"] is False
        return httpx.Response(
            200,
            json={"data": [{"id": "test-model"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(runner.httpx, "get", get)
    runner._check_endpoint(config)
    config["expected_served_model"] = "other-model"
    with pytest.raises(ValueError, match="expected served model"):
        runner._check_endpoint(config)


@pytest.mark.parametrize(
    "status,payload", [(401, {}), (200, []), (200, {"data": None})]
)
def test_endpoint_rejection_or_malformed_inventory_cannot_pass(
    runner, config, monkeypatch, status, payload
):
    monkeypatch.setattr(
        runner.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            status, json=payload, request=httpx.Request("GET", url)
        ),
    )
    with pytest.raises((ValueError, httpx.HTTPStatusError)):
        runner._check_endpoint(config)


def test_missing_endpoint_key_stops_before_network(runner, config, monkeypatch):
    monkeypatch.delenv("TEST_VLM_KEY")
    monkeypatch.setattr(
        runner.httpx, "get", lambda *args, **kwargs: pytest.fail("network reached")
    )
    with pytest.raises(ValueError, match="credential"):
        runner._check_endpoint(config)


@pytest.mark.parametrize("outcome", ["empty", "skip", "pass"])
def test_readiness_only_or_skipped_inference_never_passes(
    runner, config, monkeypatch, tmp_path, outcome
):
    monkeypatch.setattr(runner, "_check_endpoint", lambda config: None)

    def run(target, results):
        assert results.suites == (runner.PROVENANCE_SUITE,)
        if outcome != "empty":
            node = (
                runner.PROVENANCE_SUITE
                + "::test_self_hosted_result_retains_served_model"
            )
            results.collected = [node]
            results.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid=node,
                    when="call" if outcome == "pass" else "setup",
                    passed=outcome == "pass",
                    failed=False,
                    skipped=outcome == "skip",
                    user_properties=[],
                )
            )
        return 0

    monkeypatch.setattr(runner, "_run_live_tests", run)
    target = tmp_path / "receipt"
    assert runner.main(["--evidence-dir", str(target)]) == (
        0 if outcome == "pass" else 1
    )
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["passed"] is (outcome == "pass")
    assert receipt["endpoint_preflight_passed"] is True
    assert (target / "receipt.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("fenced", [False, True])
def test_provenance_parser_tag_matches_retained_framing(fenced):
    path = Path(__file__).resolve().parents[1] / "e2e/test_vlm_served_model_live.py"
    spec = importlib.util.spec_from_file_location("served_model_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    content = '{"score":0.9,"success":true,"rationale":"test frame"}'
    if fenced:
        content = "```json\n" + content + "\n```"
    raw = json.dumps(
        {
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": content},
                }
            ],
        }
    )
    base = "npa_vlm_eval_compatible_json_v1"
    provider = {
        "raw_response": raw,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "status_code": 200,
        "returned_model": "test-model",
        "finish_reason": "stop",
        "parser_version": base + ("+markdown-fence-v1" if fenced else ""),
    }
    config = {"expected_served_model": "test-model"}
    module._assert_provider_evidence(provider, config)
    for invalid in (
        base + "+unsupported",
        base if fenced else base + "+markdown-fence-v1",
    ):
        provider["parser_version"] = invalid
        with pytest.raises(AssertionError):
            module._assert_provider_evidence(provider, config)
