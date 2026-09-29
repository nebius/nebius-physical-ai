"""Submit operator-owned Slurm scripts and validate their durable result contract."""

from __future__ import annotations

import signal
import subprocess
import sys
import uuid
from pathlib import PurePosixPath
from typing import Any

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from .contracts import approved_checkpoint, checkpoint, digest


def _execute(settings: dict[str, Any], command: list[str]) -> int:
    if settings["transport"] == "local":
        return subprocess.run(command, capture_output=True, check=False).returncode
    if settings["transport"] != "soperator":
        raise ValueError("transport must be local or soperator")
    return _pod_execute(settings, ["chroot", "/mnt/jail", *command])


def _pod_execute(settings, command):
    from kubernetes import client, config
    from kubernetes.stream import stream

    configuration = client.Configuration()
    if settings.get("context"):
        config.load_kube_config(
            context=settings["context"], client_configuration=configuration
        )
    else:
        config.load_incluster_config(client_configuration=configuration)
    with client.ApiClient(configuration) as api_client:
        core = client.CoreV1Api(api_client)
        connection = stream(
            core.connect_get_namespaced_pod_exec,
            settings["login_pod"],
            settings["namespace"],
            container=settings["login_container"],
            command=command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        try:
            while connection.is_open():
                connection.update(timeout=1)
                connection.read_stdout()
                connection.read_stderr()
            if connection.returncode is None:
                raise RuntimeError(
                    "Slurm client connection closed without an exit status"
                )
            return connection.returncode
        finally:
            connection.close()


def _terminate(signum, frame):
    raise KeyboardInterrupt("Slurm stage interrupted")


def _submit(settings: dict[str, Any], stage: str, request_uri: str) -> None:
    if settings["transport"] == "reference-local":
        # This explicit reference mode exercises the same result contract without Slurm.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "npa.workflows.policy_training.reference",
                "--request-uri",
                request_uri,
            ],
            check=True,
            capture_output=True,
        )
        return
    _submit_slurm(settings, stage, request_uri)


def _submit_slurm(settings: dict[str, Any], stage: str, request_uri: str) -> None:
    script = settings["scripts"][stage]
    if not isinstance(script, str) or not PurePosixPath(script).is_absolute():
        raise ValueError("batch script must be an absolute worker-readable path")
    name = "npa-policy-" + uuid.uuid4().hex
    command = [
        "sbatch",
        "--wait",
        "--parsable",
        f"--job-name={name}",
        script,
        "--request-uri",
        request_uri,
    ]
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        returncode = _execute(settings, command)
        if returncode:
            raise RuntimeError(f"Slurm batch failed with exit code {returncode}")
    except BaseException:
        # A lost client connection does not cancel a submitted batch job.
        if _execute(settings, ["scancel", f"--name={name}"]):
            raise RuntimeError(
                "Slurm batch failed and its cancellation request failed"
            ) from None
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def _request(stage, partition, split_uri, input_uri, run_id, output_uri):
    splits = read_json_uri(split_uri)
    selected = splits["partitions"][partition]
    if digest(read_json_uri(selected["uri"])) != selected["sha256"]:
        raise ValueError("partition content changed after splitting")
    request = {
        "schema": "npa.policy.batch-request.v1",
        "run_id": run_id,
        "stage": stage,
        "partition": partition,
        "dataset": selected,
        "result_uri": output_uri,
    }
    if input_uri:
        request["checkpoint"] = (
            approved_checkpoint(input_uri)
            if stage in {"finetune", "deploy"}
            else checkpoint(read_json_uri(input_uri))
        )
    elif stage != "pretrain":
        raise ValueError("stage requires an input checkpoint")
    return request


def batch(
    settings_uri, stage, partition, split_uri, input_uri, output_uri, run_id, iteration
):
    """Run one Slurm job and publish only its validated result.

    Args:
        settings_uri: Private transport settings and absolute script paths.
        stage: Pretrain, evaluate-pretrain, finetune, evaluate-finetune, or deploy.
        partition: Required data partition for this stage.
        split_uri: Immutable split index.
        input_uri: Prior training result or approved gate, when required.
        output_uri: Validated stage result location.
        run_id: Workflow execution identity.
        iteration: Enclosing training-loop iteration, starting at one.
    Returns:
        None.
    Raises:
        ValueError: Partition or output provenance is invalid.
        RuntimeError: Slurm reports a failed batch job.
    """
    request, request_uri = _prepare_request(
        stage, partition, split_uri, input_uri, output_uri, run_id, int(iteration)
    )
    write_json_uri(request_uri, request)
    _submit(read_json_uri(settings_uri), stage, request_uri)
    result = read_json_uri(request["result_uri"])
    _validate_result(request, result)
    write_json_uri(output_uri, {**result, "request": request})


def _prepare_request(
    stage, partition, split_uri, input_uri, output_uri, run_id, iteration
):
    expected = {
        "pretrain": "train",
        "finetune": "train",
        "deploy": "holdout_2",
        "evaluate-pretrain": "holdout_1",
        "evaluate-finetune": "holdout_2",
    }
    if expected.get(stage) != partition:
        raise ValueError("stage cannot consume that data partition")
    if iteration < 1:
        raise ValueError("iteration must be positive")
    attempt = output_uri.rsplit("/", 1)[0] + "/attempts/" + uuid.uuid4().hex
    request = _request(
        stage, partition, split_uri, input_uri, run_id, attempt + "/result.json"
    )
    request["iteration"] = iteration
    if iteration > 1 and stage in {"pretrain", "finetune"}:
        request["checkpoint"] = _resume_checkpoint(output_uri, request)
    return request, attempt + "/request.json"


def _resume_checkpoint(output_uri, request):
    root, attempt_number, filename = output_uri.rsplit("/", 2)
    iteration = request["iteration"]
    if attempt_number != str(iteration):
        raise ValueError("training outputs must use an iteration-number directory")
    previous = read_json_uri(f"{root}/{iteration - 1}/{filename}")
    prior_request = previous["request"]
    _validate_result(prior_request, previous)
    for key in ("run_id", "stage", "dataset"):
        if prior_request[key] != request[key]:
            raise ValueError(
                "resume candidate belongs to a different run, stage or dataset"
            )
    if prior_request["iteration"] != iteration - 1:
        raise ValueError("resume candidate belongs to a different iteration")
    return checkpoint(previous)


def _validate_result(request, result):
    if result.get("schema") != "npa.policy.batch-result.v1":
        raise ValueError("batch result has an invalid schema")
    if result.get("request_sha256") != digest(request):
        raise ValueError("batch result does not match the current request")
    if result.get("status") != "completed":
        raise ValueError("batch result did not complete")
    actual = checkpoint(result)
    if request["stage"] not in {"pretrain", "finetune"}:
        if actual != request["checkpoint"]:
            raise ValueError("evaluation or deployment used the wrong checkpoint")
    if request["stage"] == "deploy" and not result.get("policy_test_report_uri"):
        raise ValueError("deployment requires a policy test report")
