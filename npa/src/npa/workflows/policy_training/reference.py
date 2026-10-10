"""Execute the generated reaching workload through the production batch contract."""

from __future__ import annotations

import argparse

from npa.workbench.dataset.storage import read_json_uri, write_json_uri
from .contracts import digest
from .reference_model import episodes, evaluate, load_weights, train


def run(request_uri):
    """Train, evaluate, or test a generated reference policy using a batch request.

    Args:
        request_uri: Local or S3 request emitted by the pipeline's batch adapter.
    Returns:
        None; writes a completed result and real checkpoint or trajectory artifacts.
    Raises:
        ValueError: Request, dataset, checkpoint, or stage is invalid.
    """
    request = read_json_uri(request_uri)
    if request.get("schema") != "npa.policy.batch-request.v1":
        raise ValueError("invalid reference request")
    records = episodes(request)
    weights = load_weights(request.get("checkpoint"))
    result = {
        "schema": "npa.policy.batch-result.v1",
        "status": "completed",
        "request_sha256": digest(request),
        "engine": "numpy-planar-reference",
    }
    if request["stage"] in {"pretrain", "finetune"}:
        result["checkpoint"], result["training_loss"] = train(request, records, weights)
    elif request["stage"] in {"evaluate-pretrain", "evaluate-finetune", "deploy"}:
        result["checkpoint"] = request["checkpoint"]
        result["systems"], trajectories = evaluate(request, records, weights)
        report_uri = request["result_uri"].rsplit("/", 1)[0] + "/rollouts.json"
        write_json_uri(
            report_uri, {"engine": result["engine"], "systems": trajectories}
        )
        result["rollouts_uri"] = report_uri
        if request["stage"] == "deploy":
            result["policy_test_report_uri"] = report_uri
    else:
        raise ValueError("unknown reference stage")
    write_json_uri(request["result_uri"], result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-uri", required=True)
    run(parser.parse_args().request_uri)
