"""Retain source-bound repair attempts and independently verify their real artifacts."""

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from sandbox import TARGETS, _command


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _sources(config):
    result = {}
    for relative in TARGETS:
        root = (
            Path(config["workspace"]) / "npa/src"
            if relative in config["targets"]
            else Path(config["source"])
        )
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("source scope violation")
        result[relative] = path
    return result


def _identity(config):
    return {name: _digest(path) for name, path in _sources(config).items()}


def _snapshot(config, root):
    initial = _identity(config)
    for relative, path in _sources(config).items():
        ast.parse(path.read_text())
        target = root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    if _identity(config) != initial:
        raise ValueError("source changed during snapshot")
    return initial


def _diagnose(config):
    root = Path(config["state"]) / f"check-{time.time_ns()}"
    (root / "output").mkdir(parents=True)
    hashes = _snapshot(config, root)
    result = subprocess.run(
        _command(config, root, "tests"), text=True, capture_output=True
    )
    (root / "stdout.txt").write_text(result.stdout)
    (root / "stderr.txt").write_text(result.stderr)
    value = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "source_sha256": hashes,
        "diagnostics": result.stdout[-24000:] + result.stderr[-2000:],
    }
    _write(root / "result.json", value)
    return value


def _current(config):
    path = Path(config["state"]) / "current.json"
    return Path(json.loads(path.read_text())["attempt"]) if path.exists() else None


def _observe(config):
    attempt = _current(config)
    if attempt is None:
        return {"status": "not_submitted"}
    if (attempt / "result.json").exists():
        return {
            "attempt": attempt.name,
            **json.loads((attempt / "result.json").read_text()),
        }
    path = attempt / "started.json"
    if not path.exists():
        path = attempt / "process.json"
    if not path.exists():
        return {
            "status": "uncertain",
            "reason": "launch intent without process identity",
        }
    process = json.loads(path.read_text())
    try:
        fields = (
            Path(f"/proc/{process['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        )
        if fields[0] == "Z" or fields[19] != process["start_ticks"]:
            raise ValueError("process not running")
    except (OSError, ValueError):
        return {
            "status": "uncertain",
            "reason": "unresolved process; no automatic replay",
        }
    return {"status": "running", "attempt": attempt.name}


def _start(config, state):
    attempt = state / f"attempt-{time.time_ns()}"
    (attempt / "output").mkdir(parents=True)
    hashes = _snapshot(config, attempt)
    _write(attempt / "request.json", {"config": config, "source_sha256": hashes})
    _write(state / "current.json", {"attempt": str(attempt)})
    _write(attempt / "launch-intent.json", {"created_epoch": time.time()})
    with (attempt / "runner.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, __file__, "--worker", str(attempt)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    ticks = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    _write(attempt / "process.json", {"pid": child.pid, "start_ticks": ticks})
    return {"status": "submitted", "attempt": attempt.name, "source_sha256": hashes}


def _submit(config):
    state = Path(config["state"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior, observed = _current(config), _observe(config)
        if prior:
            if json.loads((prior / "request.json").read_text())[
                "source_sha256"
            ] == _identity(config):
                return observed
            if observed["status"] not in {"completed", "failed"}:
                raise ValueError("prior attempt unresolved")
        return _start(config, state)


def _verify_native(config, attempt, hashes):
    expected = {
        Path(key).name: value
        for key, value in hashes.items()
        if "token_factory/" in key
    }
    _write(attempt / "expected-source.json", expected)
    environment = {
        **os.environ,
        "PYTHONPATH": config["source"],
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    argv = [
        config["python"],
        str(Path(config["driver"]) / "verify.py"),
        "--input",
        str(attempt / "output/run"),
        "--matrix",
        config["matrix"],
        "--expected-source",
        str(attempt / "expected-source.json"),
        "--native-python",
        config["native_python"],
        "--output",
        str(attempt / "verification"),
    ]
    with (attempt / "verification.log").open("x") as log:
        checked = subprocess.run(
            argv, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
    if checked.returncode:
        raise ValueError("independent replay/native reader failed; inspect logs")
    return json.loads((attempt / "verification/result.json").read_text())


def _worker(attempt):
    request = json.loads((attempt / "request.json").read_text())
    config, started = request["config"], time.monotonic()
    ticks = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    _write(attempt / "started.json", {"pid": os.getpid(), "start_ticks": ticks})
    result = {"status": "failed", "source_sha256": request["source_sha256"]}
    try:
        with (
            (attempt / "stdout.txt").open("x") as out,
            (attempt / "stderr.txt").open("x") as err,
        ):
            process = subprocess.run(
                _command(config, attempt, "native"), stdout=out, stderr=err
            )
        result["workload_returncode"] = process.returncode
        if process.returncode:
            raise ValueError("native workflow failed; inspect logs")
        result["verification"] = _verify_native(
            config, attempt, request["source_sha256"]
        )
        result["status"] = "completed"
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
    finally:
        result["seconds"] = time.monotonic() - started
        _write(attempt / "result.json", result)


def _wait(config):
    started = time.monotonic()
    while True:
        value = _observe(config)
        if value["status"] != "running" or time.monotonic() - started >= 50:
            return value
        time.sleep(0.3)


def _verify(config):
    state = _observe(config)
    if state["status"] != "completed":
        raise ValueError("native completion and independent verification required")
    if state["source_sha256"] != _identity(config):
        raise ValueError("current source differs from verified attempt")
    checks = _diagnose(config)
    if checks["status"] != "passed":
        return checks
    return {
        **state,
        "regression_checks": "passed",
        "diagnostics": checks["diagnostics"],
    }


def _logs(config):
    root = _current(config)
    names = ("stdout.txt", "stderr.txt", "verification.log", "runner.log")
    return {
        "status": _observe(config)["status"],
        "logs": {
            name: (root / name).read_text()[-16000:]
            for name in names
            if root and (root / name).exists()
        },
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "operation",
        nargs="?",
        choices=("diagnose", "submit", "status", "wait", "logs", "verify"),
    )
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker:
        _worker(args.worker)
        return
    handlers = {
        "diagnose": _diagnose,
        "submit": _submit,
        "status": _observe,
        "wait": _wait,
        "logs": _logs,
        "verify": _verify,
    }
    try:
        value = handlers[args.operation](json.loads(args.config.read_text()))
    except Exception as error:
        value = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    print(json.dumps(value))
    raise SystemExit(1 if value["status"] in {"failed", "uncertain"} else 0)


if __name__ == "__main__":
    _main()
