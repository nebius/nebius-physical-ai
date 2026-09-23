"""Operate and drain a private pool of single-job Nebius CPU runners."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import time
import uuid

from ci_cpu_runner_cloud import (
    _check_capacity,
    _create_worker,
    _delete_worker,
    _github,
    _load,
    _nebius,
    _nebius_items,
    _owned,
    _owned_worker,
    _pages,
    _save,
    _verify_project,
)


def _config(root: Path) -> dict:
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Runner state must be outside the repository")
    if root.stat().st_mode & 0o077:
        raise ValueError("Runner state must be private: chmod 700 the state directory")
    config = _load(root / "config.json")
    for field in ("owner", "label"):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,62}", config[field]):
            raise ValueError(f"Invalid pool {field}")
    if len(config["owner"]) > 32:
        raise ValueError("Pool ownership label must fit worker hostnames")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", config["repository"]):
        raise ValueError("Invalid GitHub repository")
    if config["variable"] not in {"NPA_CI_PRIORITY_RUNNER", "NPA_CI_TEST_RUNNER"}:
        raise ValueError("Only the existing CI routing variables may be managed")
    if not isinstance(config["workers"], int) or config["workers"] < 1:
        raise ValueError("Worker capacity must be a positive integer")
    if not re.fullmatch(r"[0-9]+vcpu-[0-9]+gb", config["preset"]):
        raise ValueError("A CPU-only preset is required")
    for directory in ("workers", "retired"):
        (root / directory).mkdir(mode=0o700, exist_ok=True)
    return config


def _records(root: Path) -> list[dict]:
    return [_load(path) for path in sorted((root / "workers").glob("*.json"))]


def _runners(root: Path, config: dict) -> dict[int, dict]:
    return {
        runner["id"]: runner
        for runner in _pages(root, config, "actions/runners", "runners")
    }


def _active_runs(root: Path, config: dict) -> set[int]:
    runs = set()
    for status in ("queued", "in_progress", "waiting", "pending", "requested"):
        runs.update(
            run["id"]
            for run in _pages(
                root, config, f"actions/runs?status={status}", "workflow_runs"
            )
        )
    return runs


def _routing(root: Path, config: dict):
    return _github(root, config, f"actions/variables/{config['variable']}", absent=True)


def _record_routing(root: Path, config: dict, current) -> None:
    receipt = root / "routing.json"
    if not receipt.exists():
        _save(
            receipt,
            {
                "previous": current,
                "variable": config["variable"],
                "label": config["label"],
            },
        )
        return
    saved = _load(receipt)
    if (saved["variable"], saved["label"]) != (config["variable"], config["label"]):
        raise ValueError("Routing receipt belongs to a different pool")
    previous = saved["previous"]["value"] if saved["previous"] else None
    value = current["value"] if current else None
    if value not in {previous, config["label"]}:
        raise RuntimeError(
            "Routing changed outside this pool; inspect its saved receipt"
        )


def _enable(root: Path, config: dict) -> None:
    _verify_project(root, config)
    if not _running(root) or (root / "drain.json").exists():
        raise RuntimeError("Start the controller before enabling runner routing")
    runners = _runners(root, config)
    online = [
        r
        for r in _records(root)
        if runners.get(r.get("runner_id"), {}).get("status") == "online"
    ]
    if len(online) < config["workers"]:
        raise RuntimeError("Every configured worker must be online before routing CI")
    current = _routing(root, config)
    _record_routing(root, config, current)
    endpoint = (
        f"actions/variables/{config['variable']}" if current else "actions/variables"
    )
    _github(
        root,
        config,
        endpoint,
        method="PATCH" if current else "POST",
        payload={
            "name": config["variable"],
            "value": config["label"],
        },
    )
    print("CI routing enabled for the verified CPU pool.")


def _disable(root: Path, config: dict) -> None:
    receipt_path = root / "routing.json"
    current = _routing(root, config)
    if not current or current["value"] != config["label"]:
        print("CI routing already points away from this pool.")
        return
    if not receipt_path.exists():
        raise RuntimeError("Cannot restore routing without its original-value receipt")
    previous = _load(receipt_path)["previous"]
    endpoint = f"actions/variables/{config['variable']}"
    if previous:
        _github(
            root,
            config,
            endpoint,
            method="PATCH",
            payload={"name": config["variable"], "value": previous["value"]},
        )
    else:
        _github(root, config, endpoint, method="DELETE")
    print("Previous CI routing restored; in-flight jobs remain supported.")


def _begin_drain(root: Path, config: dict) -> None:
    if (root / "drain.json").exists():
        return
    current = _routing(root, config)
    if not (root / "routing.json").exists() and (
        not current or current["value"] != config["label"]
    ):
        _save(root / "drain.json", {"runs": []})
        return
    runs = _active_runs(root, config)
    _disable(root, config)
    runs.update(_active_runs(root, config))
    _save(root / "drain.json", {"runs": sorted(runs)})


def _drained(root: Path, config: dict, runners: dict) -> bool:
    path = root / "drain.json"
    if not path.exists():
        return False
    remaining = []
    for run_id in _load(path)["runs"]:
        run = _github(root, config, f"actions/runs/{run_id}")
        if run["status"] != "completed":
            remaining.append(run_id)
    _save(path, {"runs": remaining})
    owned = [runners.get(record.get("runner_id"), {}) for record in _records(root)]
    return not remaining and not any(runner.get("busy") for runner in owned)


def _reconcile(root: Path, config: dict, runners: dict) -> None:
    finished = []
    for record in _records(root):
        if record["phase"] == "creating":
            _recover_worker(root, config, record, runners)
        elif record["runner_id"] not in runners:
            finished.append(record)
        elif runners[record["runner_id"]]["status"] == "offline":
            instance = _owned_worker(root, config, record)
            if instance is None or instance["status"]["state"] == "STOPPED":
                finished.append(record)
    _retire_workers(root, config, finished)


def _retire_workers(root: Path, config: dict, records: list[dict]) -> None:
    if records:
        with ThreadPoolExecutor(max_workers=len(records)) as executor:
            list(
                executor.map(
                    lambda record: _delete_worker(root, config, record), records
                )
            )


def _recover_worker(root: Path, config: dict, record: dict, runners: dict) -> None:
    if "runner_id" not in record:
        matches = [r for r in runners.values() if r["name"] == record["name"]]
        for runner in matches:
            if runner["busy"]:
                raise RuntimeError(
                    "Incomplete runner registration is unexpectedly busy"
                )
            _github(root, config, f"actions/runners/{runner['id']}", method="DELETE")
    _create_worker(root, config, record["name"])


def _fill(root: Path, config: dict) -> None:
    if len(_records(root)) < config["workers"]:
        _verify_image_and_network(root, config)
        _check_capacity(root, config, config["workers"] - len(_records(root)))
    names = []
    for _ in range(config["workers"] - len(_records(root))):
        name = f"npa-ci-{config['owner']}-{uuid.uuid4().hex[:12]}"
        _save(root / "workers" / f"{name}.json", {"name": name, "phase": "creating"})
        names.append(name)
    if names:
        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            list(executor.map(lambda name: _create_worker(root, config, name), names))


def _verify_firewall(root: Path, config: dict) -> None:
    rules = _nebius_items(
        root, config, "vpc security-rule", config["worker_security_group_id"]
    )
    deny_all = False
    for rule in rules:
        spec = rule["spec"]
        if "ingress" not in spec:
            continue
        if spec["access"] != "DENY":
            raise ValueError("Worker firewall must not allow inbound connections")
        if rule.get("status", {}).get("state") != "READY":
            raise ValueError("Worker firewall is not ready")
        deny_all |= spec["protocol"] == "ANY" and spec["ingress"].get(
            "source_cidrs"
        ) == ["0.0.0.0/0"]
    if not deny_all:
        raise ValueError("Worker firewall is missing its deny-all ingress rule")


def _verify_image_and_network(root: Path, config: dict) -> None:
    image = _nebius(root, config, "compute image", "get", {"id": config["image_id"]})
    _owned(image, config)
    group = _nebius(
        root,
        config,
        "vpc security-group",
        "get",
        {"id": config["worker_security_group_id"]},
    )
    _owned(group, config)
    if group["spec"]["network_id"] != config["network_id"]:
        raise ValueError("Worker firewall belongs to a different network")
    _verify_firewall(root, config)
    subnet = _nebius(root, config, "vpc subnet", "get", {"id": config["subnet_id"]})
    if (
        subnet["metadata"]["parent_id"] != config["project_id"]
        or subnet["spec"]["network_id"] != config["network_id"]
    ):
        raise ValueError("Worker subnet belongs to a different project or network")


def _serve(root: Path, config: dict) -> None:
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _verify_project(root, config)
        while True:
            runners = _runners(root, config)
            if _drained(root, config, runners):
                _retire_workers(root, config, _records(root))
                _save(root / "status.json", {"phase": "stopped", "workers": 0})
                return
            _reconcile(root, config, runners)
            _fill(root, config)
            _save(
                root / "status.json",
                {
                    "phase": "draining"
                    if (root / "drain.json").exists()
                    else "running",
                    "workers": len(_records(root)),
                },
            )
            time.sleep(20)


def _running(root: Path) -> bool:
    with (root / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _launch(root: Path, config: dict) -> None:
    if _running(root):
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "serve",
        "--state-dir",
        str(root),
    ]
    if sys.platform == "darwin":
        _launch_agent(root, config, command)
    else:
        with (root / "controller.log").open("a") as log:
            subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    print("Local runner controller started; keep this operator machine available.")


def _launch_agent(root: Path, config: dict, command: list[str]) -> None:
    label = f"com.nebius.npa.ci-runners.{config['owner']}"
    path = root / "controller.plist"
    spec = {
        "Label": label,
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", "-s", *command],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
        "EnvironmentVariables": {"PATH": os.environ["PATH"], "PYTHONUNBUFFERED": "1"},
    }
    path.write_bytes(plistlib.dumps(spec))
    path.chmod(0o600)
    domain = f"gui/{os.getuid()}"
    loaded = subprocess.run(
        ["launchctl", "print", f"{domain}/{label}"], capture_output=True, check=False
    )
    command = (
        ["launchctl", "kickstart", f"{domain}/{label}"]
        if loaded.returncode == 0
        else ["launchctl", "bootstrap", domain, str(path)]
    )
    subprocess.run(command, check=True, capture_output=True)


def _status(root: Path, config: dict) -> None:
    runners = _runners(root, config)
    records = _records(root)
    owned = [runners.get(record.get("runner_id"), {}) for record in records]
    current = _routing(root, config)
    print(
        json.dumps(
            {
                "controller_running": _running(root),
                "draining": (root / "drain.json").exists(),
                "routing_enabled": bool(
                    current and current["value"] == config["label"]
                ),
                "capacity": config["workers"],
                "tracked_workers": len(records),
                "online": sum(runner.get("status") == "online" for runner in owned),
                "busy": sum(bool(runner.get("busy")) for runner in owned),
                "retired": len(list((root / "retired").glob("*.json"))),
            },
            indent=2,
        )
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["up", "serve", "status", "enable", "disable", "down"]
    )
    parser.add_argument(
        "--state-dir", type=Path, default=Path.home() / ".npa/ci-runners/default"
    )
    args = parser.parse_args()
    root = args.state_dir.expanduser().resolve()
    config = _config(root)
    if args.command == "up":
        if (root / "drain.json").exists() and _running(root):
            raise RuntimeError(
                "Wait for the requested drain to finish before starting again"
            )
        (root / "drain.json").unlink(missing_ok=True)
        _launch(root, config)
    elif args.command == "down":
        if not _records(root) and not _running(root):
            _disable(root, config)
            print("No workers remain.")
            return
        _begin_drain(root, config)
        _launch(root, config)
        print(
            "Drain requested. Existing workflows finish before workers and their disks are deleted."
        )
    else:
        {"serve": _serve, "status": _status, "enable": _enable, "disable": _disable}[
            args.command
        ](root, config)


if __name__ == "__main__":
    try:
        _main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
