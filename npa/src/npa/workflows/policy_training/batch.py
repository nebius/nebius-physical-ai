"""Submit operator-owned Slurm scripts and validate their durable result contract."""

from __future__ import annotations

import uuid

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from .contracts import approved_checkpoint, checkpoint, digest, _validate_engine
from .selection import select_training_data
from .slurm import _reserve, _submit_slurm


def _submit(settings, stage, request_uri):
    if settings.get("transport") not in {"local", "soperator"}:
        raise ValueError("production transport must be local or soperator")
    _submit_slurm(settings, stage, request_uri)


def _request(
    stage, partition, split_uri, input_uri, run_id, output_uri, *, _reference=False
):
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
            approved_checkpoint(input_uri, _reference=_reference)
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
    _batch(
        settings_uri,
        stage,
        partition,
        split_uri,
        input_uri,
        output_uri,
        run_id,
        iteration,
    )


def _batch(
    settings_uri,
    stage,
    partition,
    split_uri,
    input_uri,
    output_uri,
    run_id,
    iteration,
    *,
    _reference=False,
):
    _validate_partition(stage, partition)
    settings = read_json_uri(settings_uri)
    request, request_uri = _prepare_request(
        stage,
        partition,
        split_uri,
        input_uri,
        output_uri,
        run_id,
        int(iteration),
        settings,
        _reference=_reference,
    )
    _complete_request(settings, stage, request, request_uri, output_uri, _reference)


def _complete_request(settings, stage, request, request_uri, output_uri, _reference):
    if _reference:
        from .reference import run

        write_json_uri(request_uri, request)
        run(request_uri)
    else:
        if not _reserve(request_uri, request) and read_json_uri(request_uri) != request:
            raise ValueError("saved batch request changed; start a new run")
        _submit(settings, stage, request_uri)
    result = read_json_uri(request["result_uri"])
    _validate_result(request, result, _reference=_reference)
    write_json_uri(output_uri, {**result, "request": request})


def _prepare_request(
    stage,
    partition,
    split_uri,
    input_uri,
    output_uri,
    run_id,
    iteration,
    settings=None,
    *,
    _reference=False,
):
    _validate_partition(stage, partition)
    if iteration < 1:
        raise ValueError("iteration must be positive")
    attempt = (
        output_uri.rsplit("/", 1)[0]
        + "/attempts/"
        + uuid.uuid5(uuid.NAMESPACE_URL, output_uri).hex
    )
    request = _request(
        stage,
        partition,
        split_uri,
        input_uri,
        run_id,
        attempt + "/result.json",
        _reference=_reference,
    )
    request["iteration"] = iteration
    select_training_data(request, settings or {})
    if iteration > 1 and stage in {"pretrain", "finetune"}:
        request["checkpoint"] = _resume_checkpoint(
            output_uri, request, _reference=_reference
        )
    return request, attempt + "/request.json"


def _validate_partition(stage, partition):
    expected = {
        "pretrain": "train",
        "finetune": "train",
        "deploy": "holdout_2",
        "evaluate-pretrain": "holdout_1",
        "evaluate-finetune": "holdout_2",
    }
    if expected.get(stage) != partition:
        raise ValueError("stage cannot consume that data partition")


def _resume_checkpoint(output_uri, request, *, _reference=False):
    root, attempt_number, filename = output_uri.rsplit("/", 2)
    iteration = request["iteration"]
    if attempt_number != str(iteration):
        raise ValueError("training outputs must use an iteration-number directory")
    previous = read_json_uri(f"{root}/{iteration - 1}/{filename}")
    prior_request = previous["request"]
    _validate_result(prior_request, previous, _reference=_reference)
    for key in ("run_id", "stage"):
        if prior_request[key] != request[key]:
            raise ValueError(
                "resume candidate belongs to a different run, stage or dataset"
            )
    if prior_request["dataset"]["sha256"] != request["dataset"]["sha256"]:
        raise ValueError("resume candidate belongs to a different training selection")
    if prior_request["iteration"] != iteration - 1:
        raise ValueError("resume candidate belongs to a different iteration")
    return checkpoint(previous)


def _validate_result(request, result, *, _reference=False):
    if result.get("schema") != "npa.policy.batch-result.v1":
        raise ValueError("batch result has an invalid schema")
    if result.get("request_sha256") != digest(request):
        raise ValueError("batch result does not match the current request")
    if result.get("status") != "completed":
        raise ValueError("batch result did not complete")
    _validate_engine(result, reference=_reference)
    actual = checkpoint(result)
    if request["stage"] in {"pretrain", "finetune"} and request.get("checkpoint"):
        if actual["sha256"] == request["checkpoint"]["sha256"]:
            raise ValueError("training returned unchanged checkpoint bytes")
    if request["stage"] not in {"pretrain", "finetune"}:
        if actual != request["checkpoint"]:
            raise ValueError("evaluation or deployment used the wrong checkpoint")
    if request["stage"] == "deploy" and not result.get("policy_test_report_uri"):
        raise ValueError("deployment requires a policy test report")
