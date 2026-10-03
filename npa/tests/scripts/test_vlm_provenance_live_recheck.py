"""Test provenance lane gates without invoking a model or cloud resource."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest


@pytest.fixture
def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.chdir(directory.parent.parent)
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


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.test",
        "https://example.test/v1",
        "https://example.test/v1/",
        "https://example.test/v1/chat/completions",
    ],
)
def test_endpoint_probe_uses_selected_credential_and_exact_model(
    runner, config, monkeypatch, endpoint
):
    config["endpoint_url"] = endpoint

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


@pytest.mark.parametrize("outcome", ["empty", "skip", "partial", "pass"])
def test_readiness_only_or_skipped_inference_never_passes(
    runner, config, monkeypatch, tmp_path, outcome
):
    monkeypatch.setattr(runner, "_check_endpoint", lambda config: None)

    def run(target, results):
        assert results.suites == (runner.PROVENANCE_SUITE,)
        if outcome == "empty":
            return 0
        nodes = (
            sorted(runner.REQUIRED_TESTS)
            if outcome == "pass"
            else [
                runner.PROVENANCE_SUITE
                + "::test_self_hosted_result_retains_served_model"
            ]
        )
        results.collected = nodes
        for node in nodes:
            results.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid=node,
                    when="setup" if outcome == "skip" else "call",
                    passed=outcome in {"pass", "partial"},
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


def test_sampling_requires_video_tools(runner, config, monkeypatch, tmp_path):
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="ffmpeg"):
        runner._check_sampling_inputs(config, tmp_path / "receipt")


@pytest.mark.parametrize("name", ["receipt.json", "pytest/custom.json"])
def test_sampling_preserves_custom_result_outside_runner_reserved_paths(
    runner, config, monkeypatch, tmp_path, name
):
    monkeypatch.setattr(runner.shutil, "which", lambda _: "/synthetic/tool")
    target = tmp_path / "receipt"
    config["output_path"] = str(target / name)
    with pytest.raises(ValueError, match="runner-reserved"):
        runner._check_sampling_inputs(config, target)


def test_sampling_receipt_binds_fixture_helper(runner):
    path = "npa/tests/e2e/vlm_sampling_live_helpers.py"
    assert path in runner._receipt(Path.cwd())["source_file_sha256"]


def test_existing_private_evidence_path_is_not_printed(runner, tmp_path, capsys):
    target = tmp_path / "private-evidence"
    target.mkdir()
    with pytest.raises(SystemExit):
        runner.main(["--evidence-dir", str(target)])
    output = capsys.readouterr()
    assert str(target) not in output.out + output.err


@pytest.mark.parametrize("package_prefix", [True, False])
def test_sampling_collection_accepts_both_supported_rootdirs(runner, package_prefix):
    nodes = sorted(runner.REQUIRED_TESTS)
    if not package_prefix:
        nodes = [node.removeprefix("npa/") for node in nodes]
    assert runner._sampling_tests_collected(nodes)
    assert not runner._sampling_tests_collected(nodes[:-1])
    assert not runner._sampling_tests_collected(nodes + ["unrelated::test_case"])


def test_sampling_gate_matches_actual_pytest_collection(runner, monkeypatch):
    monkeypatch.delenv("NPA_INTEGRATION_E2E", raising=False)
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    for variable in (
        "NPA_CI_SHARD_INDEX",
        "NPA_CI_TOTAL_SHARDS",
        "NPA_CI_TIMING_OUTPUT",
    ):
        monkeypatch.delenv(variable, raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            runner.PROVENANCE_SUITE,
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    nodes = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::test_self_hosted" in line
    ]
    assert runner._sampling_tests_collected(nodes)


@pytest.mark.parametrize("stage", ["_receipt", "write_receipt"])
def test_receipt_filesystem_errors_do_not_disclose_private_paths(
    runner, config, monkeypatch, tmp_path, capsys, stage
):
    def fail(*args, **kwargs):
        raise PermissionError(f"private location: {tmp_path}")

    monkeypatch.setattr(runner, stage, fail)
    monkeypatch.setattr(runner, "_verify", lambda *args: 2)
    assert runner.main(["--evidence-dir", str(tmp_path / "receipt")]) == 1
    output = capsys.readouterr()
    assert str(tmp_path) not in output.out + output.err
    assert json.loads(output.out)["passed"] is False


@pytest.mark.parametrize("boundary", ["evidence", "configuration", "input", "output"])
def test_real_symlink_loops_fail_without_disclosing_private_paths(
    runner, config, monkeypatch, tmp_path, capsys, boundary
):
    first, second = tmp_path / "private-loop-a", tmp_path / "private-loop-b"
    first.symlink_to(second)
    second.symlink_to(first)
    target = tmp_path / "receipt"
    if boundary == "evidence":
        with pytest.raises(SystemExit):
            runner.main(["--evidence-dir", str(first)])
    else:
        if boundary == "configuration":
            monkeypatch.setenv("NPA_VLM_PROVENANCE_LIVE_CONFIG", str(first))
        else:
            config[boundary + "_path"] = str(first)
            (tmp_path / "config.json").write_text(json.dumps(config))
        assert runner.main(["--evidence-dir", str(target)]) == 1
        assert not json.loads((target / "receipt.json").read_text())["passed"]
    output = capsys.readouterr()
    assert str(tmp_path) not in output.out + output.err
    assert "private-loop" not in output.out + output.err


def test_sampling_output_resolution_also_sanitizes_real_symlink_loop(
    runner, config, tmp_path
):
    loop = tmp_path / "private-output-loop"
    loop.symlink_to(loop)
    config["output_path"] = str(loop)
    with pytest.raises(ValueError, match="without symlink loops") as error:
        runner._check_sampling_inputs(config, tmp_path / "receipt")
    assert str(loop) not in str(error.value)


@pytest.mark.parametrize(
    "endpoint", ["http://example.test:invalid-port", "http://[broken"]
)
def test_real_malformed_endpoint_is_a_sanitized_failed_receipt(
    runner, config, tmp_path, capsys, endpoint
):
    config["endpoint_url"] = endpoint
    (tmp_path / "config.json").write_text(json.dumps(config))
    target = tmp_path / "receipt"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["phase"] == "endpoint" and not receipt["passed"]
    output = capsys.readouterr()
    assert endpoint not in output.out + output.err
    assert str(tmp_path) not in output.out + output.err


def test_real_git_failure_does_not_inherit_private_stderr(
    runner, config, monkeypatch, tmp_path, capfd
):
    private_git_path = tmp_path / "private-non-repository"
    monkeypatch.setenv("GIT_DIR", str(private_git_path))
    assert runner.main(["--evidence-dir", str(tmp_path / "receipt")]) == 1
    output = capfd.readouterr()
    assert str(private_git_path) not in output.out + output.err
    assert "fatal:" not in output.err
    assert json.loads(output.out)["passed"] is False
