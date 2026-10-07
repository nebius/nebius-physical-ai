"""Remove a completed reference run's owned namespace after verifying collected weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .public_vla_launch import IMAGE


def _verify_collection(output):
    completed = json.loads((output / "completed.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    if (
        completed.get("training_executed") is not True
        or completed.get("evaluation_executed") is not True
    ):
        raise ValueError("collection has no completed training and evaluation proof")
    weights = output / "exported-policy/policy/model.safetensors"
    with weights.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != selection.get("checkpoint_sha256"):
        raise ValueError("collected checkpoint checksum does not match selection")
    for relative in (
        "report/index.html",
        "report/evidence.json",
        "exported-policy/manifest.json",
    ):
        if not (output / relative).is_file():
            raise ValueError("collection is missing a required deliverable")


def _client(receipt):
    from kubernetes import client, config
    from .public_vla_collect import _kubectl

    _kubectl(receipt)
    options = dict(zip(receipt["kubectl"][1::2], receipt["kubectl"][2::2], strict=True))
    configuration = client.Configuration()
    config.load_kube_config(
        config_file=options.get("--kubeconfig"),
        context=options.get("--context"),
        client_configuration=configuration,
    )
    return client.ApiClient(configuration)


def _delete_owned_namespace(api, receipt, output):
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    uid = receipt.get("namespace_uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError(
            "receipt lacks namespace ownership UID; automated deletion refused"
        )
    core = client.CoreV1Api(api)
    try:
        namespace = core.read_namespace(receipt["namespace"])
    except ApiException as error:
        if error.status != 404:
            raise
        _record_cleanup(output, uid)
        return
    if (
        namespace.metadata.uid != uid
        or (namespace.metadata.labels or {}).get("npa.nebius.ai/public-vla-run")
        != receipt["namespace"]
    ):
        raise ValueError("namespace ownership changed; deletion refused")
    _require_completed_job(client.BatchV1Api(api), receipt)
    body = client.V1DeleteOptions(preconditions=client.V1Preconditions(uid=uid))
    core.delete_namespace(receipt["namespace"], body=body)
    while True:
        try:
            current = core.read_namespace(receipt["namespace"])
        except ApiException as error:
            if error.status != 404:
                raise
            break
        if current.metadata.uid != uid:
            raise RuntimeError("namespace name was reused; cleanup receipt withheld")
        time.sleep(2)
    _record_cleanup(output, uid)


def _record_cleanup(output, uid):
    (output / "cleanup.json").write_text(
        json.dumps({"namespace_uid": uid, "namespace_deleted": True})
    )


def _require_completed_job(api, receipt):
    jobs = api.list_namespaced_job(receipt["namespace"]).items
    if len(jobs) != 1 or jobs[0].metadata.name != "pipeline":
        raise ValueError("unexpected namespace workloads; deletion refused")
    job = jobs[0]
    conditions = {item.type: item.status for item in job.status.conditions or []}
    if job.status.active or conditions.get("Complete") != "True":
        raise ValueError("training is not complete; cancel workloads before teardown")
    if job.spec.template.spec.containers[0].image != IMAGE:
        raise ValueError("training job image changed; deletion refused")


def _cleanup_run(receipt, output):
    _verify_collection(output)
    with _client(receipt) as api:
        _delete_owned_namespace(api, receipt, output)


def main() -> None:
    """Delete only a completed, collected reference run's owned namespace.

    Args:
        None; reads a local launch receipt and collection directory from argv.
    Returns:
        None; saves a local cleanup receipt after verified namespace absence.
    Raises:
        ValueError: Ownership, completion or artifact verification fails.
        ApiException: Kubernetes cannot verify or delete the namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    _cleanup_run(json.loads(args.input_path.read_text()), args.output_path)


if __name__ == "__main__":
    main()
