"""Keep matched repair claims gated by complete outcomes and reconciled provider usage."""

from decimal import Decimal
import hashlib
from http.client import HTTPConnection
from http.server import HTTPServer
import importlib
import json
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

EXAMPLE = Path(__file__).parents[2] / "examples/specialists/repair_benchmark"


@pytest.fixture
def modules(monkeypatch):
    names = (
        "sandbox",
        "operation",
        "accounting",
        "prepare",
        "run",
        "score",
        "telemetry",
        "timing",
    )
    previous = {name: sys.modules.get(name) for name in names}
    monkeypatch.syspath_prepend(str(EXAMPLE))
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield SimpleNamespace(**{name: importlib.import_module(name) for name in names})
    finally:
        for name, value in previous.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


@pytest.fixture
def prices():
    return json.loads((EXAMPLE / "prices.json").read_text())


def test_native_failure_handoff_does_not_escalate_initial_diagnosis(modules, tmp_path):
    default = modules.prepare._operations(tmp_path, tmp_path / "operation.json")
    opted_in = modules.prepare._operations(
        tmp_path, tmp_path / "operation.json", native_failure_handoff=True
    )
    assert not any(value.get("handoff_on_failure") for value in default.values())
    assert opted_in["wait"].pop("handoff_on_failure") is True
    assert opted_in == default


def test_system_python_mounts_its_prefix_instead_of_the_filesystem(
    modules, monkeypatch
):
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    mounts = modules.sandbox._runtime_mounts(Path("/usr/bin/python3"))
    sources = mounts[1::3]
    assert "/usr" in sources
    assert "/" not in sources


def test_virtualenv_keeps_version_alias_without_mounting_its_parent(modules, tmp_path):
    runtime = tmp_path / "runtimes/python-3.12.14"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python3.12").touch()
    alias = runtime.with_name("python-3.12")
    alias.symlink_to(runtime.name, target_is_directory=True)
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(alias / "bin/python3.12")
    interpreter = environment / "bin/python3"
    interpreter.symlink_to("python")
    mounts = modules.sandbox._runtime_mounts(interpreter)
    sources = mounts[1::3]
    assert {str(runtime), str(alias), str(environment)}.issubset(sources)
    assert str(runtime.parent) not in sources
    assert str(tmp_path) not in sources


@pytest.mark.parametrize("interpreter", ["/bin/python3", "/operator/bin/python3"])
def test_python_mounts_cannot_expose_root_or_account_home(
    modules, monkeypatch, interpreter
):
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    monkeypatch.setattr(
        modules.sandbox.pwd, "getpwuid", lambda _: SimpleNamespace(pw_dir="/operator")
    )
    with pytest.raises(ValueError, match="expose an account or filesystem root"):
        modules.sandbox._runtime_mounts(Path(interpreter))


def test_standard_runtime_mount_cannot_contain_an_account_home(modules, monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda path: path)
    monkeypatch.setattr(
        modules.sandbox.pwd,
        "getpwuid",
        lambda _: SimpleNamespace(pw_dir="/usr/operator"),
    )
    with pytest.raises(ValueError, match="expose an account or filesystem root"):
        modules.sandbox._runtime_mounts(Path("/runtime/bin/python3"))


def _tokens(prompt=10000, output=100, cached=8000):
    return {
        "input_tokens": prompt,
        "output_tokens": output,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "reasoning_output_tokens": min(20, output),
    }


def _telemetry(module, tokens, identity="owned"):
    return {
        "attributes": {
            "conversation.id_sha256": identity,
            "event.kind": "response.completed",
            **{source: tokens[name] for source, name in module.COUNTERS.items()},
        }
    }


def test_setup_tokens_are_retained_as_uncertainty(modules, prices):
    module = modules.accounting
    expected = _tokens()
    setup = _tokens(prompt=11103, output=0, cached=0)
    rows = [_telemetry(module, setup), _telemetry(module, expected)]
    result = module._thread_cost("owned", expected, rows, prices)
    assert result["charged"] == Decimal("0.033")
    assert result["setup_upper"] == Decimal("0.11103")
    assert result["requests"] == 2


def test_context_tariff_is_per_request_and_reasoning_is_not_double_charged(
    modules, prices
):
    module = modules.accounting
    request = _tokens(prompt=160000, output=100, cached=0)
    rows = [_telemetry(module, request), _telemetry(module, request)]
    expected = {name: 2 * value for name, value in request.items()}
    result = module._thread_cost("owned", expected, rows, prices)
    assert result["charged"] == Decimal("3.21")
    assert result["setup_upper"] == 0
    assert module._request_cost(_tokens(prompt=300000, cached=0), prices) == Decimal(
        "6.0075"
    )


@pytest.mark.parametrize(
    "defect", ["missing", "duplicated", "wrong_thread", "error", "fractional", "bool"]
)
def test_incomplete_telemetry_cannot_support_a_cost_claim(modules, prices, defect):
    module = modules.accounting
    expected = _tokens()
    rows = [_telemetry(module, expected)]
    if defect == "missing":
        rows = []
    elif defect == "duplicated":
        rows *= 2
    elif defect == "wrong_thread":
        rows[0]["attributes"]["conversation.id_sha256"] = "other"
    elif defect == "error":
        rows[0]["attributes"]["success"] = False
    else:
        rows[0]["attributes"]["input_token_count"] = (
            1.5 if defect == "fractional" else True
        )
    with pytest.raises(ValueError):
        module._thread_cost("owned", expected, rows, prices)


def test_cached_input_cannot_exceed_total_input(modules, prices):
    with pytest.raises(ValueError, match="overlapping"):
        modules.accounting._request_cost(_tokens(prompt=10, cached=20), prices)


def _arms():
    return [
        {
            "pair": pair,
            "arm": arm,
            "status": "passed",
            "end_to_end_seconds": 100 if arm == "astra-only" else 70,
            "cost": {
                "complete": True,
                "total_usd": {
                    "minimum": 1 if arm == "astra-only" else 0.3,
                    "maximum": 1.1 if arm == "astra-only" else 0.4,
                },
            },
        }
        for pair in range(1, 4)
        for arm in ("astra-only", "astra-tofa")
    ]


def test_equal_quality_lower_total_cost_and_latency_support_claim(modules):
    result = modules.score._comparison(_arms())
    assert result["claim_supported"]
    assert result["conservative_model_cost_reduction_fraction"] == pytest.approx(0.6)
    assert result["elapsed_reduction_fraction"] == pytest.approx(0.3)


def test_experiment_spend_includes_both_arms_and_refuses_partial_totals(modules):
    arms = _arms()
    result = modules.score._experiment_cost(arms)
    assert result["total_usd"]["minimum"] == pytest.approx(3.9)
    assert result["total_usd"]["maximum"] == pytest.approx(4.5)
    arms[0]["cost"]["complete"] = False
    result = modules.score._experiment_cost(arms)
    assert not result["complete"]
    assert result["total_usd"] is None


@pytest.mark.parametrize(
    "change", ["failure", "unknown_usage", "missing_arm", "slow", "costly"]
)
def test_failed_missing_or_uncompetitive_arms_cannot_be_discarded(modules, change):
    arms = _arms()
    if change == "failure":
        arms[1]["status"] = "failed"
    elif change == "unknown_usage":
        arms[1]["cost"]["complete"] = False
    elif change == "missing_arm":
        arms.pop()
    else:
        for row in arms:
            if row["arm"] == "astra-tofa":
                if change == "slow":
                    row["end_to_end_seconds"] = 150
                else:
                    row["cost"]["total_usd"]["maximum"] = 1.2
    assert not modules.score._comparison(arms)["claim_supported"]


def test_virtualenv_interpreter_symlink_is_preserved(modules, tmp_path):
    binary = tmp_path / "base-python"
    binary.write_text("base interpreter")
    virtualenv = tmp_path / "venv-python"
    virtualenv.symlink_to(binary)
    options = SimpleNamespace(native_python=virtualenv, reader_python=virtualenv)
    result = modules.prepare._operation_config(
        tmp_path, tmp_path, tmp_path, [], options
    )
    assert result["python"] == result["native_python"] == str(virtualenv)


def test_changed_frozen_inputs_prevent_execution(modules, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("reference")
    (tmp_path / "freeze.json").write_text(
        json.dumps({"inputs": {"source.py": modules.operation._digest(source)}})
    )
    source.write_text("changed")
    with pytest.raises(ValueError, match="frozen benchmark input changed"):
        modules.run._verify_inputs(tmp_path)


def test_incomplete_lane_never_runs_combined_work(modules, tmp_path):
    lanes = {"physics": {"status": "failed", "current_source_matches": True}}
    assert modules.run._combined(tmp_path, tmp_path, lanes)["status"] == "not_run"


def test_missing_usage_remains_unknown_without_disclosing_local_paths(
    modules, tmp_path, prices, monkeypatch
):
    def unavailable(*_args):
        raise FileNotFoundError("private operator directory")

    monkeypatch.setattr(modules.accounting, "_usage_report", unavailable)
    report = modules.accounting._account(tmp_path, tmp_path, prices)
    assert not report["complete"]
    assert report["total_usd"] is None
    assert report["incomplete_reason"] == "FileNotFoundError"
    assert "private operator" not in json.dumps(report)


def _payload():
    fields = {
        "conversation.id": "synthetic-thread",
        "event.kind": "response.completed",
        "input_token_count": "123",
        "request.body": "private prompt",
        "authorization": "private credential",
        "model": "gpt-6-astra",
        "service_tier": "private label",
    }
    record = {
        "eventName": "codex.sse_event",
        "body": {"stringValue": "private response"},
        "attributes": [
            {"key": key, "value": {"stringValue": value}}
            for key, value in fields.items()
        ],
    }
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]}


def test_telemetry_retains_only_counters_enums_and_hashed_identifiers(modules):
    result = modules.telemetry._extract(_payload())
    assert len(result) == 1
    assert result[0]["attributes"] == {
        "conversation.id_sha256": hashlib.sha256(b"synthetic-thread").hexdigest(),
        "event.kind": "response.completed",
        "input_token_count": 123,
        "model": "gpt-6-astra",
    }
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/v1/logs",
        "http://127.0.0.1/v1/logs",
        "http://user:secret@127.0.0.1:1234/v1/logs",
        "http://127.0.0.1:1234/v1/logs?secret=value",
    ],
)
def test_capture_refuses_external_or_credentialed_endpoints(modules, endpoint):
    with pytest.raises(ValueError):
        modules.telemetry._endpoint(endpoint)


def test_loopback_receiver_persists_sanitized_records(modules, tmp_path):
    records = tmp_path / "events.jsonl"
    stats = dict.fromkeys(
        ("http_requests", "accepted_batches", "rejected_batches", "retained_records"), 0
    )
    server = HTTPServer(("127.0.0.1", 0), modules.telemetry._handler(records, stats))
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/v1/logs", json.dumps(_payload()))
        response = connection.getresponse()
        assert response.status == 200
        assert response.read() == b"{}"
        connection.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        server.server_close()
    assert stats["accepted_batches"] == stats["retained_records"] == 1
    assert stats["rejected_batches"] == 0
    assert "private" not in records.read_text()


def test_wrapper_keeps_telemetry_after_ignore_config_and_bypasses_proxy(
    modules, monkeypatch
):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["codex", "exec", "--ignore-user-config", "-"])
    monkeypatch.setenv("NPA_CODEX_OTEL_ENDPOINT", "http://127.0.0.1:1234/v1/logs")
    monkeypatch.setenv("NO_PROXY", "existing.example")

    def execute(path, argv, environment):
        captured.update(path=path, argv=argv, environment=environment)

    monkeypatch.setattr(modules.telemetry.os, "execve", execute)
    modules.telemetry._invoke("/synthetic/codex")
    assert captured["argv"][:3] == ["/synthetic/codex", "exec", "--ignore-user-config"]
    assert captured["argv"][-3:] == ["-c", "otel.log_user_prompt=false", "-"]
    assert "existing.example" in captured["environment"]["NO_PROXY"]
    assert "127.0.0.1" in captured["environment"]["NO_PROXY"]


def test_export_retains_failed_and_successful_operation_attempts(modules, tmp_path):
    for name, status in (
        ("check-1", "failed"),
        ("check-2", "passed"),
        ("attempt-3", "completed"),
    ):
        path = tmp_path / "operations/physics" / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"status": status, "diagnostics": "private path"}))
    result = modules.score._attempts(tmp_path)
    assert len(result["physics"]) == 3
    assert {row["status"] for row in result["physics"]} == {
        "failed",
        "passed",
        "completed",
    }
    assert "private path" not in json.dumps(result)


def test_unchanged_failed_native_attempt_is_observed_without_relaunch(
    modules, tmp_path, monkeypatch
):
    state = tmp_path / "state"
    attempt = state / "attempt-1"
    attempt.mkdir(parents=True)
    (state / "current.json").write_text(json.dumps({"attempt": str(attempt)}))
    (attempt / "request.json").write_text(
        json.dumps({"source_sha256": {"source": "same"}})
    )
    (attempt / "result.json").write_text(json.dumps({"status": "failed"}))
    monkeypatch.setattr(
        modules.operation, "_identity", lambda _config: {"source": "same"}
    )

    def unexpected(*_args):
        pytest.fail("an unchanged failed source must not be automatically relaunched")

    monkeypatch.setattr(modules.operation, "_start", unexpected)
    assert modules.operation._submit({"state": str(state)})["status"] == "failed"


def test_uncertain_prior_native_effect_blocks_changed_source(
    modules, tmp_path, monkeypatch
):
    state = tmp_path / "state"
    attempt = state / "attempt-1"
    attempt.mkdir(parents=True)
    (state / "current.json").write_text(json.dumps({"attempt": str(attempt)}))
    (attempt / "request.json").write_text(
        json.dumps({"source_sha256": {"source": "before"}})
    )
    monkeypatch.setattr(
        modules.operation, "_identity", lambda _config: {"source": "after"}
    )
    with pytest.raises(ValueError, match="prior attempt unresolved"):
        modules.operation._submit({"state": str(state)})
