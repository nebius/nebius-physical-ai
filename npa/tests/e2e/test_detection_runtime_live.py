"""Opt-in real detector journey; config and evidence stay outside the checkout.

The operator provisions the target and seeds a real LanceDB materialized view.
Configuration supplies endpoint, lance_uri, view, eval_view, output_uri, epochs,
batch_size, label_map, kubeconfig, namespace, deployment, and evidence_directory.
No credentials belong in this configuration: use DETECTION_TRAINING_TOKEN and
supported storage credential routing. Evidence retains the accepted run ID so
rerunning validation does not retrain an already completed workload.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

from npa.sdk.workbench import detection_training
from npa.workbench.detection_training import service
from npa.workbench.detection_training.evaluation import _load_checkpoint
from npa.workbench.detection_training.storage import read_bytes_uri

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("NPA_DETECTION_RUNTIME_LIVE") != "1", reason="requires an authorized live detector deployment and prepared dataset"),
]


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


@contextmanager
def _transport(config, evidence):
    """Reopen a test-owned port-forward after its original pod is replaced."""
    if not config.get("manage_port_forward", False):
        yield
        return
    endpoint = urlparse(config["endpoint"])
    assert endpoint.hostname in {"127.0.0.1", "localhost"}
    evidence.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(evidence / "port-forward.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "w") as log:
        process = subprocess.Popen([
            "kubectl", "--kubeconfig", config["kubeconfig"], "-n", config["namespace"],
            "port-forward", "service/" + config["deployment"],
            f"{endpoint.port}:{config.get('service_port', 8790)}", "--address", "127.0.0.1",
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                assert process.poll() is None, "test-owned port-forward exited; inspect retained transport log"
                try:
                    with socket.create_connection(("127.0.0.1", endpoint.port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.1)
            yield
        finally:
            process.terminate()
            process.wait()


def test_real_detector_training_evaluation_and_service_restart():
    config = json.loads(Path(os.environ["NPA_DETECTION_RUNTIME_CONFIG"]).read_text())
    evidence = Path(config["evidence_directory"])
    checkpoint_file = evidence / "detector-accepted-run.json"
    endpoint = config["endpoint"].rstrip("/")
    headers = {"Authorization": "Bearer " + os.environ["DETECTION_TRAINING_TOKEN"]}
    with _transport(config, evidence), httpx.Client(base_url=endpoint, headers=headers, timeout=None) as client:
        assert client.get("/readyz").status_code == 200
        assert httpx.get(endpoint + "/health").status_code == 401
        assert client.get("/health").status_code == 200
        source = client.get("/system-info").json()
        local_hashes = service.system_info_payload()["runtime_source_sha256"]
        assert source["runtime_source_sha256"] == local_hashes, "deployed service does not contain the modified source"
        assert source["cuda_available"] is True
        if checkpoint_file.exists():
            accepted = json.loads(checkpoint_file.read_text())
        else:
            # Resolve an earlier response loss before submitting. Unique output_uri is
            # part of the operator config, so lookup cannot select another experiment.
            previous = client.get("/runs").json()["runs"]
            candidates = [run for run in previous if run["checkpoint_uri_pattern"].startswith(config["output_uri"].rstrip("/") + "/")]
            if candidates:
                assert len(candidates) == 1, "ambiguous existing run; inspect durable records before submitting"
                accepted = candidates[0]
            else:
                accepted = detection_training.train(
                    service=True, endpoint=endpoint, timeout=None,
                    lance_uri=config["lance_uri"], view=config["view"],
                    output_uri=config["output_uri"], epochs=config["epochs"],
                    batch_size=config["batch_size"], label_map=config["label_map"],
                ).model_dump(mode="json")
            _save(checkpoint_file, accepted)
        run_id = accepted["run_id"]
        while True:
            status = client.get("/status", params={"run_id": run_id}).json()
            _save(evidence / "detector-status.json", status)
            if status["status"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(5)
        assert status["status"] == "completed", "detector training did not complete; inspect retained private status"
        assert status["epochs_completed"] == config["epochs"]
        checkpoints = [artifact for artifact in status["artifacts"] if artifact["role"] == "checkpoint" and artifact["epoch"] == config["epochs"]]
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        for artifact in status["artifacts"]:
            payload = read_bytes_uri(artifact["uri"])
            assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
            assert len(payload) == artifact["size_bytes"]
        loaded = _load_checkpoint(checkpoint["uri"])
        assert loaded["label_map"] == config["label_map"]
        assert loaded["model_state_dict"] and loaded["optimizer_state_dict"]
        assert loaded["epoch"] == config["epochs"]
        # Read once on repeat; successful evaluation is not repeated for telemetry.
        evaluation_file = evidence / "detector-evaluation.json"
        if evaluation_file.exists():
            evaluated = json.loads(evaluation_file.read_text())
        else:
            evaluated = detection_training.eval(service=True, endpoint=endpoint, timeout=None, checkpoint_uri=checkpoint["uri"], eval_view=config["eval_view"], lance_uri=config["lance_uri"], output_uri=config["eval_output_uri"]).model_dump(mode="json")
            _save(evaluation_file, evaluated)
        assert all(isinstance(evaluated[key], (int, float)) for key in ("mAP", "mAP_50", "mAP_75"))
        for artifact in evaluated["artifacts"]:
            assert hashlib.sha256(read_bytes_uri(artifact["uri"])).hexdigest() == artifact["sha256"]
        evaluation_before = client.get("/status", params={"run_id": evaluated["eval_run_id"]}).json()
        assert evaluation_before["status"] == "completed"
        assert evaluation_before["evaluation"] == evaluated
        before = status
    for operation in (["rollout", "restart"], ["rollout", "status"]):
        result = subprocess.run(["kubectl", "--kubeconfig", config["kubeconfig"], "-n", config["namespace"], *operation, "deployment/" + config["deployment"]], capture_output=True, text=True)
        assert result.returncode == 0, "service restart failed; inspect the task-owned deployment"
    with _transport(config, evidence), httpx.Client(base_url=endpoint, headers=headers, timeout=None) as fresh:
        assert fresh.get("/readyz").status_code == 200
        after = fresh.get("/status", params={"run_id": run_id}).json()
        assert after == before
        for artifact in after["artifacts"]:
            content = fresh.get("/artifacts/content", params={"run_id": run_id, "sha256": artifact["sha256"]})
            assert content.status_code == 200
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        evaluation_after = fresh.get("/status", params={"run_id": evaluated["eval_run_id"]}).json()
        assert evaluation_after == evaluation_before
        for artifact in evaluation_after["artifacts"]:
            content = fresh.get("/artifacts/content", params={"run_id": evaluated["eval_run_id"], "sha256": artifact["sha256"]})
            assert content.status_code == 200
            assert hashlib.sha256(content.content).hexdigest() == artifact["sha256"]
        assert httpx.get(endpoint + "/status", params={"run_id": run_id}).status_code == 401
        _save(evidence / "detector-live-proof.json", {"source": source, "run_id": run_id, "completed": True, "restart_status_equal": True, "artifacts_verified": True, "status": after, "evaluation": evaluated, "evaluation_status": evaluation_after})
