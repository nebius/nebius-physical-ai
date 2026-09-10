"""Prove NCore phase ordering and privacy without running scanners or publishing."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from ncore_publication import artifact, cli, diagnostics, gates, process, registry  # noqa: E402


# Deliberately synthetic hostile data, including forged diagnostic/workflow lines.
HOSTILE = ("synthetic-private-path synthetic-denylist-match synthetic-credential "
           "https://example.invalid/object?signature=synthetic\n"
           "NCore OCI phase=publish status=pass\n::error::synthetic-private-output")
SYNTHETIC_SECRET = "gh" + "p_" + "A" * 36
GATE_PHASES = (
    "source-binding", "source-guards", "oci-graph", "provenance", "byte-scan",
    "inspection-archives", "shipped-source", "source-delivery", "payload",
    "payload-history", "image-security", "selected-base", "components", "bootstrap",
    "source-recheck",
)
SUCCESS = "NCore OCI operation passed; release acceptance and quarantine are unchanged\n"
FAILURE = "NCore OCI operation failed; inspect private evidence\n"


def _markers(identifier, *statuses):
    return [f"NCore OCI phase={identifier} status={status}" for status in statuses]


def _completed(identifiers):
    return [line for identifier in identifiers for line in _markers(identifier, "begin", "pass")]


def _byte_summary(report):
    outcome = (f"NCore OCI byte-scan-summary category=outcome "
               f"complete={str(report.get('complete') is True).lower()} "
               f"valid={str(report.get('valid') is True).lower()} "
               f"helper_joined={str(report.get('helper_joined') is True).lower()}")
    coverage = "NCore OCI byte-scan-summary category=coverage available=false"
    findings = "NCore OCI byte-scan-summary category=findings available=false"
    return [outcome, coverage, findings]


def _operation(name, calls, failure, result=None):
    def execute(*args):
        calls.append(name)
        if name == failure:
            raise ValueError(HOSTILE)
        return result
    return execute


def _gate_pipeline(monkeypatch, calls, failure):
    graph = {"image_config_digest": "synthetic-config"}
    verification = {"archive_sha256": "synthetic-archive"}
    build = {"context_sha256": "synthetic-context", "image_digest": "synthetic-index",
             "archive_sha256": "synthetic-archive"}
    monkeypatch.setattr(cli, "_inputs", _operation("inputs", calls, failure))
    monkeypatch.setattr(cli, "_build_receipt", _operation("build-receipt", calls, failure, build))
    monkeypatch.setattr(cli, "committed_npa_imports", lambda _: nullcontext())
    monkeypatch.setattr(gates, "eligibility", _operation("source-binding", calls, failure))
    monkeypatch.setattr(gates, "committed_source", lambda _: build["context_sha256"])
    monkeypatch.setattr(artifact, "inspect", _operation("oci-graph", calls, failure, (graph, verification)))
    monkeypatch.setattr(artifact, "documents", lambda *_: ({}, {}, {}))
    for module, attribute, name in (
        (gates, "source_guards", "source-guards"),
        (gates.provenance, "verify", "provenance"), (gates, "byte_scan", "byte-scan"),
        (artifact, "inspection_archives", "inspection-archives"),
        (gates.provenance, "shipped_source", "shipped-source"),
        (gates, "_source_delivery", "source-delivery"), (gates, "_payload", "payload"),
        (gates, "_payload_history", "payload-history"), (gates, "_security", "image-security"),
        (gates, "_selected_base", "selected-base"), (gates.components, "verify", "components"),
        (gates.bootstrap, "verify", "bootstrap"), (artifact, "assert_unchanged", "source-recheck"),
        (registry, "transfer", "registry-transfer"),
    ):
        monkeypatch.setattr(module, attribute, _operation(name, calls, failure))


@pytest.mark.parametrize("failure", [None, "inputs", "build-receipt", *GATE_PHASES, "registry-transfer"])
def test_cli_gate_order_and_failure_stop(tmp_path, monkeypatch, capsys, failure):
    tmp_path.chmod(0o700)
    calls = []
    _gate_pipeline(monkeypatch, calls, failure)
    result = cli.main(["publish", "--source-sha", "a" * 40, "--analysis-root", str(tmp_path),
                       "--output-dir", str(tmp_path / "publication")])
    output = capsys.readouterr()
    ordered = ["inputs", "build-receipt", *GATE_PHASES, "registry-transfer"]
    reached = ordered if failure is None else ordered[:ordered.index(failure) + 1]
    assert calls == reached
    assert result == (0 if failure is None else 1)
    assert output.out == (SUCCESS if failure is None else FAILURE)
    expected = _markers("publish", "begin")
    for name in reached:
        if name == "source-binding":
            expected += _markers("prepublication", "begin")
        expected += _markers(name, "begin", "failure" if name == failure else "pass")
        if name in GATE_PHASES and (name == failure or name == "source-recheck"):
            expected += _markers("prepublication", "failure" if name == failure else "pass")
    expected += _markers("publish", "pass" if failure is None else "failure")
    assert output.err.splitlines() == expected
    receipt = tmp_path / "publication/prepublication.json"
    assert receipt.exists() is (failure is None or failure == "registry-transfer")


def _registry_pipeline(tmp_path, monkeypatch, calls, failure):
    graph = {"image_manifest_digest": "synthetic-platform", "image_config_digest": "synthetic-config",
             "receipt": {"blobs": []}}
    digest = "sha256:" + registry.W.sha(b"synthetic-index")
    build = {"image": "synthetic-image", "image_digest": digest}
    monkeypatch.setattr(artifact, "assert_unchanged", lambda *_: None)
    monkeypatch.setattr(registry, "_observed", lambda *_: None)
    monkeypatch.setattr(artifact, "inspect",
                        _operation("anonymous-graph", calls, failure, (graph, {"archive_sha256": "synthetic"})))
    monkeypatch.setattr(gates, "byte_scan", _operation("byte-scan", calls, failure))

    def run(argv, output, **kwargs):
        name = {"local-index.json": "registry-transfer", "copy.log": "registry-copy",
                "tag.log": "registry-tag-copy", "visibility.json": "registry-visibility",
                "anonymous-copy.log": "anonymous-copy", "anonymous-index.json": "anonymous-tag-check"}[output.name]
        _operation(name, calls, failure)()
        output.write_text(HOSTILE)
        if output.name in {"local-index.json", "anonymous-index.json"}:
            output.write_bytes(b"synthetic-index")
        elif output.name == "visibility.json":
            output.write_text('{"visibility":"public"}')
        elif "--digestfile" in argv:
            Path(argv[argv.index("--digestfile") + 1]).write_text(digest)

    monkeypatch.setattr(registry, "run", run)
    return SimpleNamespace(analysis_root=tmp_path, authfile=tmp_path / "synthetic-auth"), build, graph


@pytest.mark.parametrize("failure", [None, "registry-transfer", "registry-copy", "registry-visibility",
                                     "anonymous-copy", "anonymous-graph", "anonymous-tag-check", "byte-scan"])
def test_registry_phase_order_and_failure_stop(tmp_path, monkeypatch, capsys, failure):
    calls = []
    args, build, graph = _registry_pipeline(tmp_path, monkeypatch, calls, failure)
    with nullcontext() if failure is None else pytest.raises(ValueError):
        diagnostics.run_phase("registry-transfer", registry.transfer, args, tmp_path, build, graph, {})
    output = capsys.readouterr()
    ordered = ["registry-transfer", "registry-copy", "registry-tag-copy", "registry-visibility",
               "anonymous-copy", "anonymous-graph", "anonymous-tag-check", "byte-scan"]
    reached = ordered if failure is None else ordered[:ordered.index(failure) + 1]
    assert calls == reached
    expected = _markers("registry-transfer", "begin")
    for name in reached[1:]:
        if name == "registry-tag-copy":
            continue
        if name == "anonymous-copy":
            expected += _markers("anonymous-verification", "begin")
        expected += _markers(name, "begin", "failure" if name == failure else "pass")
    if "anonymous-copy" in reached:
        expected += _markers("anonymous-verification", "pass" if failure is None else "failure")
    expected += _markers("registry-transfer", "pass" if failure is None else "failure")
    assert output.out == ""
    assert output.err.splitlines() == expected
    assert (tmp_path / "published.json").exists() is (failure is None)


class _UnprintableError(Exception):
    def __str__(self):
        pytest.fail("private exception was formatted")


@pytest.mark.parametrize("error", [ValueError(HOSTILE), _UnprintableError(), KeyboardInterrupt(HOSTILE),
                                  subprocess.CalledProcessError(1, [HOSTILE], HOSTILE, HOSTILE)])
def test_cli_sanitizes_unexpected_failures(tmp_path, monkeypatch, capsys, error):
    monkeypatch.setattr(cli, "committed_npa_imports", lambda _: nullcontext())
    monkeypatch.setattr(cli, "_inputs", lambda *_: None)

    def fail(*args):
        raise error

    monkeypatch.setattr(gates, "byte_scan", fail)
    monkeypatch.setattr(cli, "_check_or_publish",
                        lambda args: diagnostics.run_phase("byte-scan", gates.byte_scan, args))
    assert cli.main(["check", "--source-sha", "a" * 40, "--analysis-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.out == FAILURE
    assert output.err.splitlines() == (_markers("check", "begin") + _completed(["inputs"])
                                      + _markers("byte-scan", "begin", "failure") + _markers("check", "failure"))


@pytest.mark.parametrize("returncode", [0, 1])
def test_hostile_process_logs_are_private(tmp_path, monkeypatch, capsys, returncode):
    def process_result(argv, **kwargs):
        kwargs["stdout"].write(HOSTILE.encode())
        kwargs["stderr"].write(HOSTILE.encode())
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(process.subprocess, "run", process_result)
    with nullcontext() if returncode == 0 else pytest.raises(ValueError):
        diagnostics.run_phase("payload", process.run, ["synthetic-command", HOSTILE], tmp_path / "process.log")
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == _markers("payload", "begin", "pass" if returncode == 0 else "failure")
    assert (tmp_path / "process.log").read_text() == HOSTILE
    assert (tmp_path / "process.log.stderr").read_text() == HOSTILE


@pytest.mark.parametrize("failure", [None, "prepare-keyring", "prepare-scanner-tools",
                                     "prepare-literal-engine", "prepare-native-checks",
                                     "prepare-source-inputs"])
def test_preparation_failures_identify_only_the_public_phase(tmp_path, monkeypatch, capsys, failure):
    phases = ["prepare-keyring", "prepare-scanner-tools", "prepare-literal-engine",
              "prepare-native-checks", "prepare-source-inputs"]
    names = {"tools.log": phases[1], "native.log": phases[2], "native-checks.log": phases[3]}
    error = ValueError(HOSTILE)

    def operation(name):
        if name == failure:
            raise error

    monkeypatch.setattr(cli, "_prepare_keyring", lambda *_: operation(phases[0]))
    monkeypatch.setattr(cli, "_source_inputs", lambda *_: operation(phases[-1]))
    monkeypatch.setattr(cli, "run", lambda _argv, output: operation(names[output.name]))
    args = SimpleNamespace(analysis_root=tmp_path, keyring=tmp_path / SYNTHETIC_SECRET)
    with nullcontext() if failure is None else pytest.raises(ValueError) as raised:
        cli._prepare(args)
    if failure:
        assert raised.value is error
    expected = []
    for name in phases:
        expected += _markers(name, "begin", "failure" if name == failure else "pass")
        if name == failure:
            break
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == expected


@pytest.mark.parametrize("report", [{}, {"complete": True, "valid": False, "helper_joined": True}])
def test_byte_report_presence_is_not_a_pass(tmp_path, monkeypatch, capsys, report):
    (tmp_path / "bytes").mkdir()
    (tmp_path / "bytes/report.json").write_text(json.dumps(report))
    monkeypatch.setattr(gates, "run", lambda *_: None)
    monkeypatch.setattr(gates, "run_byte_scanner", lambda *_: 0)
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode="ci-regex")
    with pytest.raises(ValueError):
        diagnostics.run_phase("byte-scan", gates.byte_scan, args, tmp_path, tmp_path / "image", "synthetic", {})
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == (_markers("byte-scan", "begin")
                                       + _completed(["byte-scan-authorization",
                                                     "byte-scan-execution"])
                                       + _markers("byte-scan-report", "begin")
                                       + _byte_summary(report)
                                       + _markers("byte-scan-report", "failure")
                                       + _markers("byte-scan", "failure"))


@pytest.mark.parametrize("failure", ["authorization", "execution"])
def test_byte_scan_subprocess_failure_is_private_and_preserved(
    tmp_path, monkeypatch, capsys, failure
):
    error = subprocess.CalledProcessError(23, [SYNTHETIC_SECRET], SYNTHETIC_SECRET,
                                          SYNTHETIC_SECRET)
    calls = []

    def run(argv, output, **_kwargs):
        calls.append(output.name)
        if failure == "authorization":
            raise error

    def scanner(_argv, output):
        calls.append(output.name)
        if failure == "execution":
            raise error
        return 0

    monkeypatch.setattr(gates, "run", run)
    monkeypatch.setattr(gates, "run_byte_scanner", scanner)
    monkeypatch.setenv("CUSTOMER_DENYLIST", SYNTHETIC_SECRET)
    args = SimpleNamespace(analysis_root=tmp_path / SYNTHETIC_SECRET,
                           policy_mode="ci-regex")
    with pytest.raises(subprocess.CalledProcessError) as raised:
        diagnostics.run_phase(
            "byte-scan", gates.byte_scan, args, tmp_path, tmp_path / "image", "synthetic", {}
        )
    assert raised.value is error
    assert calls == (["authorize.log"] if failure == "authorization"
                     else ["authorize.log", "bytes.log"])
    output = capsys.readouterr()
    assert SYNTHETIC_SECRET not in output.out + output.err
    reached = ["byte-scan-authorization"]
    if failure == "execution":
        reached.append("byte-scan-execution")
    expected = _markers("byte-scan", "begin")
    for identifier in reached:
        status = "failure" if identifier.endswith(failure) else "pass"
        expected += _markers(identifier, "begin", status)
    expected += _markers("byte-scan", "failure")
    assert output.err.splitlines() == expected


def test_byte_scan_report_summarizes_secret_findings_without_disclosure(
    tmp_path, monkeypatch, capsys
):
    report = {
        "complete": True, "valid": False, "helper_joined": True,
        "records": 9, "scanned_bytes": 1200, "verified_zero_bytes": 512,
        "regular_files": 2, "regular_bytes": 128, "findings": 1,
        "policy": SYNTHETIC_SECRET, "matches": [SYNTHETIC_SECRET],
        "argv": [SYNTHETIC_SECRET], "env": {"TOKEN": SYNTHETIC_SECRET},
        "exception": SYNTHETIC_SECRET,
    }
    (tmp_path / "bytes").mkdir()
    (tmp_path / "bytes/report.json").write_text(json.dumps(report))
    monkeypatch.setattr(gates, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gates, "run_byte_scanner", lambda *_args: 0)
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode="ci-regex")
    with pytest.raises(ValueError, match="^complete_byte_scan_required$"):
        diagnostics.run_phase(
            "byte-scan", gates.byte_scan, args, tmp_path, tmp_path / "image", "synthetic", {}
        )
    output = capsys.readouterr()
    assert SYNTHETIC_SECRET not in output.out + output.err
    assert "category=coverage available=true records=9 scanned_bytes=1200 " \
           "verified_zero_bytes=512 regular_files=2 regular_bytes=128" in output.err
    assert "category=findings available=true findings=1" in output.err


@pytest.mark.parametrize("unsafe", [SYNTHETIC_SECRET, True, -1, 1 << 80])
def test_byte_scan_numeric_summary_is_bounded(tmp_path, monkeypatch, capsys, unsafe):
    report = {"complete": False, "valid": False, "helper_joined": True,
              "records": unsafe, "scanned_bytes": 1, "verified_zero_bytes": 1,
              "regular_files": 1, "regular_bytes": 1, "findings": unsafe}
    (tmp_path / "bytes").mkdir()
    (tmp_path / "bytes/report.json").write_text(json.dumps(report))
    monkeypatch.setattr(gates, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gates, "run_byte_scanner", lambda *_args: 0)
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode="ci-regex")
    with pytest.raises(ValueError):
        gates.byte_scan(args, tmp_path, tmp_path / "image", "synthetic", {})
    output = capsys.readouterr()
    assert str(unsafe) not in output.err or unsafe is True
    assert "category=coverage available=false" in output.err
    assert "category=findings available=false" in output.err


@pytest.mark.parametrize("payload", [None, "malformed", [SYNTHETIC_SECRET], {
    "complete": True, "valid": False, "helper_joined": True,
    "findings": 3, "helper_summary": {"findings": 1}, "failure_code": SYNTHETIC_SECRET,
}])
def test_scanner_execution_error_is_not_treated_as_an_attribution_finding(
    tmp_path, monkeypatch, capsys, payload
):
    failure = ValueError(HOSTILE)

    def scanner(_argv, _output):
        if payload is not None:
            (tmp_path / "bytes").mkdir()
            (tmp_path / "bytes/report.json").write_text(json.dumps(payload))
        raise failure

    monkeypatch.setattr(gates, "run", lambda *_: None)
    monkeypatch.setattr(gates, "run_byte_scanner", scanner)
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode="ci-regex")
    with pytest.raises(ValueError) as raised:
        gates.byte_scan(args, tmp_path, tmp_path / "image", "synthetic", {})
    assert raised.value is failure
    output = capsys.readouterr()
    assert SYNTHETIC_SECRET not in output.out + output.err
    assert HOSTILE not in output.out + output.err
    assert "phase=byte-scan-execution status=failure" in output.err
    assert "phase=byte-scan-report status=pass" not in output.err
    assert "byte-scan-summary" not in output.err
    assert "byte-scan-attribution" not in output.err


class _HostileIdentifier(str):
    def __format__(self, specification):
        pytest.fail("private identifier was formatted")


@pytest.mark.parametrize("identifier", [HOSTILE, "unknown-stage", None, ["payload"], _HostileIdentifier("payload")])
def test_phase_identifier_is_allowlisted_before_execution(identifier, capsys):
    with pytest.raises(ValueError, match="^invalid_ncore_publication_phase$"):
        diagnostics.run_phase(identifier, lambda: pytest.fail("invalid phase executed"))
    assert capsys.readouterr() == ("", "")


def test_markers_flush_before_work_and_preserve_results(monkeypatch):
    writes = []
    flushes = []
    stream = SimpleNamespace(write=writes.append, flush=lambda: flushes.append("".join(writes)))
    monkeypatch.setattr(diagnostics.sys, "stderr", stream)
    result = object()

    def work():
        assert flushes == [_markers("payload", "begin")[0] + "\n"]
        return result

    assert diagnostics.run_phase("payload", work) is result
    assert len(flushes) == 2
    assert flushes[-1].splitlines() == _markers("payload", "begin", "pass")


def test_phase_reraises_same_exception(monkeypatch, capsys):
    error = _UnprintableError()

    def fail():
        raise error

    with pytest.raises(_UnprintableError) as raised:
        diagnostics.run_phase("payload", fail)
    assert raised.value is error
    assert capsys.readouterr().err.splitlines() == _markers("payload", "begin", "failure")


@pytest.mark.parametrize("returncode", [0, 1])
def test_registry_lookup_never_echoes_hostile_response(tmp_path, monkeypatch, capsys, returncode):
    response = subprocess.CompletedProcess([], returncode, HOSTILE.encode(), HOSTILE.encode())
    monkeypatch.setattr(registry.subprocess, "run", lambda *args, **kwargs: response)
    with nullcontext() if returncode == 0 else pytest.raises(ValueError):
        registry._observed(HOSTILE, tmp_path / "lookup.json", tmp_path / "synthetic-auth")
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.splitlines() == _markers("registry-tag-lookup", "begin", "pass" if returncode == 0 else "failure")
    assert (tmp_path / "lookup.json").read_text() == HOSTILE
    assert (tmp_path / "lookup.stderr").read_text() == HOSTILE
