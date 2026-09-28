"""Calibrate and run every frozen matched arm, including combined-patch native verification."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from operation import (
    _diagnose,
    _digest,
    _identity,
    _observe,
    _submit,
    _verify,
    _wait,
    _write,
)
from sandbox import TARGETS


def _read(path):
    return json.loads(path.read_text())


def _verify_inputs(root, edited=()):
    frozen = _read(root / "freeze.json")
    for name, expected in frozen["inputs"].items():
        if name not in edited and _digest(root / name) != expected:
            raise ValueError("frozen benchmark input changed: " + name)
    return frozen


def _await_native(config):
    result = _submit(config)
    while result["status"] in {"running", "submitted"}:
        result = _wait(config)
    if result["status"] == "completed":
        return _verify(config)
    return result


def _calibrate(root):
    _verify_inputs(root)
    output = root / "calibration/result.json"
    if output.exists():
        raise ValueError("calibration already exists; inspect its retained result")
    reference = _read(root / "calibration/reference.json")
    result = {"status": "failed", "reference": _diagnose(reference), "historical": {}}
    if result["reference"]["status"] != "passed":
        _write(output, result)
        raise ValueError("reference regression calibration failed")
    for path in sorted((root / "pairs/pair-1/astra-only/configs").glob("*.json")):
        config = _read(path)
        config["state"] = str(root / "calibration" / path.stem)
        check = _diagnose(config)
        result["historical"][path.stem] = check
        if check["returncode"] != 1 or not re.search(
            r"\d+ failed", check["diagnostics"]
        ):
            _write(output, result)
            raise ValueError(
                "historical source did not produce ordinary regression failures"
            )
    result["native_reference"] = _await_native(reference)
    if result["native_reference"]["status"] == "completed":
        result["status"] = "passed"
    _write(output, result)
    return result


def _capture(options):
    url = urlsplit(options.otel_endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path != "/v1/logs"
    ):
        raise ValueError("OTEL capture must use a credential-free loopback endpoint")
    wrapper = options.codex_wrapper.resolve() / "codex"
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise ValueError("an executable operator-owned codex wrapper is required")
    return {
        "wrapper": str(wrapper),
        "wrapper_sha256": _digest(wrapper),
        "endpoint": options.otel_endpoint,
        "records": str(options.telemetry_records.resolve()),
    }


def _environment(root, capture):
    return {
        **os.environ,
        "PYTHONPATH": str(root / "source"),
        "PATH": str(Path(capture["wrapper"]).parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
        "NPA_CODEX_OTEL_ENDPOINT": capture["endpoint"],
    }


def _host_state():
    return {
        "logical_cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
        "observed_epoch": time.time(),
    }


def _command(root, directory, arm, protocol):
    return [
        sys.executable,
        str(root / "coordinator/experiment.py"),
        "--team-config",
        str(directory / "team.json"),
        "--prompt",
        str(root / "prompt.txt"),
        "--output",
        str(directory / "run"),
        "--arm",
        arm,
        "--coordination",
        protocol["coordination"],
        "--effort",
        protocol["astra_effort"],
        "--live",
    ]


def _lane_results(directory):
    results = {}
    for path in sorted((directory / "configs").glob("*.json")):
        config = _read(path)
        state = _observe(config)
        if state["status"] not in {"completed", "failed", "not_submitted"}:
            raise ValueError(
                "unresolved native effect; sequence requires reconciliation"
            )
        state["current_source_matches"] = state.get("source_sha256") == _identity(
            config
        )
        results[path.stem] = state
    return results


def _combined_config(root, directory):
    profile_configs = [
        _read(path) for path in sorted((directory / "configs").glob("*.json"))
    ]
    workspace = directory / "combined/workspace"
    for config in profile_configs:
        for name in config["targets"]:
            source = Path(config["workspace"]) / "npa/src" / name
            destination = workspace / "npa/src" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    result = {
        **profile_configs[0],
        "workspace": str(workspace),
        "targets": list(TARGETS),
        "state": str(directory / "combined/operations"),
        "matrix": str(root / "driver/multimodel-scenes.json"),
    }
    _write(directory / "combined/config.json", result)
    return result


def _combined(root, directory, lanes):
    if not all(
        item["status"] == "completed" and item["current_source_matches"]
        for item in lanes.values()
    ):
        return {"status": "not_run", "reason": "incomplete or stale lane artifacts"}
    config = _combined_config(root, directory)
    checks = _diagnose(config)
    if checks["status"] != "passed":
        return {"status": "failed", "regressions": checks}
    return {"regressions": checks, **_await_native(config)}


def _arm(root, directory, arm, protocol, capture):
    if (directory / "run").exists() or (directory / "state").exists():
        raise ValueError("arm has existing execution state; refusing automatic replay")
    started = time.monotonic()
    record = {"arm": arm, "started_epoch": time.time(), "host_before": _host_state()}
    _write(directory / "launch-intent.json", record)
    command = _command(root, directory, arm, protocol)
    with (
        (directory / "launcher.stdout").open("x") as out,
        (directory / "launcher.stderr").open("x") as err,
    ):
        process = subprocess.run(
            command, cwd=root, env=_environment(root, capture), stdout=out, stderr=err
        )
    record["coordinator_exit_code"] = process.returncode
    record["agent_tool_seconds"] = time.monotonic() - started
    record["lanes"] = _lane_results(directory)
    record["combined"] = _combined(root, directory, record["lanes"])
    record["end_to_end_seconds"] = time.monotonic() - started
    record["status"] = (
        "passed"
        if process.returncode == 0 and record["combined"]["status"] == "completed"
        else "failed"
    )
    record.update(ended_epoch=time.time(), host_after=_host_state())
    _write(directory / "outcome.json", record)
    shutil.copyfile(capture["records"], directory / "telemetry-snapshot.jsonl")
    return record


def _edited_sources(root, directory):
    team = _read(directory / "team.json")
    return {
        str((Path(profile["workspace"]) / name).relative_to(root))
        for profile in team["profiles"]
        for name in profile["write_paths"]
    }


def _run(root, options):
    if not options.live:
        raise ValueError("--live is required for paid inference")
    frozen, capture = _verify_inputs(root), _capture(options)
    if _read(root / "calibration/result.json")["status"] != "passed":
        raise ValueError("passing historical/reference calibration is required")
    path = root / "execution.json"
    if path.exists():
        raise ValueError(
            "sequence already launched; retain and inspect existing evidence"
        )
    result = {
        "status": "running",
        "freeze_sha256": _digest(root / "freeze.json"),
        "capture": capture,
        "prices": _read(options.prices),
        "arms": [],
    }
    _write(path, result)
    _sequence(root, frozen, capture, result, path)
    result["status"] = "finished"
    _write(path, result)
    return result


def _sequence(root, frozen, capture, result, path):
    edited = set()
    for number, order in enumerate(frozen["protocol"]["pairs"], 1):
        for arm in order:
            _verify_inputs(root, edited)
            if _digest(Path(capture["wrapper"])) != capture["wrapper_sha256"]:
                raise ValueError("instrumentation wrapper changed")
            directory = root / "pairs" / f"pair-{number}" / arm
            record = _arm(root, directory, arm, frozen["protocol"], capture)
            result["arms"].append({"pair": number, **record})
            edited.update(_edited_sources(root, directory))
            _write(path, result)
            print(
                json.dumps(
                    {
                        "pair": number,
                        "arm": arm,
                        "status": record["status"],
                        "seconds": record["end_to_end_seconds"],
                    }
                ),
                flush=True,
            )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("calibrate", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codex-wrapper", type=Path)
    parser.add_argument("--otel-endpoint")
    parser.add_argument("--telemetry-records", type=Path)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--live", action="store_true")
    options = parser.parse_args()
    os.umask(0o077)
    result = (
        _calibrate(options.root.resolve())
        if options.action == "calibrate"
        else _run(options.root.resolve(), options)
    )
    print(json.dumps({"status": result["status"]}))
    raise SystemExit(0 if result["status"] in {"passed", "finished"} else 1)


if __name__ == "__main__":
    _main()
