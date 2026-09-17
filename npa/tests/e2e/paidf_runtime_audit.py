"""Verify fresh PAIDF wave completion and exact automatic partial-launch recovery chains."""

from collections import defaultdict
import hashlib
import json
import re
from urllib.parse import urlsplit


_CAUSES = {"kubernetes_transport", "kubernetes_rate_limit", "kubernetes_server"}
_CHECKS = {"exact_image_pull", "credentials_access", "accelerator_resolution",
           "per_node_gpu_shape", "gang_capacity"}


def assert_completed_fresh_runtime(client, bucket, prefix, runtime):
    """Verify completed logical waves without discarding failed attempt evidence.

    Args:
        client: Selected project's read-only object-store client.
        bucket: Bucket holding this run's durable records.
        prefix: Exact run prefix, including its trailing slash.
        runtime: Decoded final runtime record.
    Returns:
        None.
    Raises:
        AssertionError: Completion, freshness, or a recovery chain is unproven.
    """
    assert runtime["status"] == "succeeded" and runtime["waves"]
    assert prefix.endswith("/" + runtime["run_id"] + "/")
    groups = defaultdict(list)
    for wave in runtime["waves"]:
        assert wave["replayed"] is False and wave["adopted"] is False
        assert wave.get("recovery_resumed", False) is False
        assert wave["job_id"] and wave["job_name"] and wave["logical_launch_id"]
        groups[wave["key"]].append(wave)
    events = _reservation_events(client, bucket, prefix)
    consumed = set()
    for attempts in groups.values():
        _assert_chain(attempts, runtime["run_id"], events, consumed)
        if attempts[-1].get("partial_launch"):
            _assert_successful_partial(client, bucket, prefix, attempts[-1], runtime["run_id"])
    assert consumed == set(events), "Unmatched recovery reservation in completed run"


def _reservation_events(client, bucket, prefix):
    result = {}
    root = prefix + "npa-workflow/supervisor/attempts/"
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root):
        for item in page.get("Contents", []):
            key = item["Key"]
            if "/recovery_reserved-" not in key:
                continue
            with client.get_object(Bucket=bucket, Key=key)["Body"] as body:
                raw = body.read()
            digest = hashlib.sha256(raw).hexdigest()
            assert key.endswith("/recovery_reserved-" + digest + ".json"), "Reservation hash mismatch"
            event = json.loads(raw)
            parent = event["attempt_identity"]["logical_attempt_id"]
            assert key == root + parent + "/recovery_reserved-" + digest + ".json"
            assert parent not in result, "Duplicate or conflicting reservation"
            assert event["phase"] == "recovery_reserved"
            assert event["schema_version"] == "npa.workflow.supervisor.v1"
            result[parent] = event
    return result


def _identity(wave, run_id, *, provider=True):
    immutable = wave["immutable_identity"]
    assert all(re.fullmatch("[0-9a-f]{64}", immutable[name])
               for name in ("workflow_sha256", "source_sha256", "image_digest"))
    return dict(
        runtime="skypilot", run_id=run_id, attempt=wave["attempt"],
        logical_attempt_id=wave["logical_launch_id"],
        provider_job_id=wave["job_id"] if provider else "",
        provider_job_name=wave["job_name"], checkpoint_prefix="", **immutable,
    )


def _assert_chain(attempts, run_id, events, consumed):
    assert [wave["attempt"] for wave in attempts] == list(range(1, len(attempts) + 1))
    assert len({wave["logical_launch_id"] for wave in attempts}) == len(attempts)
    assert len({wave["job_id"] for wave in attempts}) == len(attempts)
    assert attempts[-1]["status"] == "succeeded" and attempts[-1]["sky_status"] == "SUCCEEDED"
    for index, wave in enumerate(attempts):
        assert wave["states"] == attempts[0]["states"] and wave["outputs"] == attempts[0]["outputs"]
        assert wave["immutable_identity"] == attempts[0]["immutable_identity"]
        assert wave["infrastructure_recovery"]["used"] == index
        _identity(wave, run_id)
        if index == 0:
            assert not wave.get("recovery_reservation")
        if index < len(attempts) - 1:
            assert wave["logical_launch_id"] in events, "Missing exact parent reservation"
            event = events[wave["logical_launch_id"]]
            _assert_recovery(wave, attempts[index + 1], run_id, event)
            consumed.add(wave["logical_launch_id"])
    if len(attempts) == 1:
        assert not attempts[0].get("recovery_reservation")


def _assert_recovery(parent, successor, run_id, event):
    proof = parent["partial_launch"]
    reservation = proof["recovery_reservation"]
    assert parent["status"] == "failed" and parent["sky_status"] == "CANCELLED"
    assert parent["cancellation"]["state"] == "verified" and not parent["cancellation"]["error"]
    assert not parent["observations"] and parent["launch_sequence"] > 0
    assert proof["schema"] == 1 and proof["cause"] in _CAUSES
    assert parent["error_category"] == proof["cause"]
    assert proof["workload_observable_at_failure"] is False
    assert proof["outputs_absent_before_cancel"]
    assert proof["wave_key"] == reservation["wave_key"] == parent["key"]
    assert proof["parent"] == reservation["parent"] == event["attempt_identity"] == _identity(parent, run_id)
    assert reservation["successor"] == event["new_attempt_identity"] == _identity(successor, run_id, provider=False)
    assert successor["recovery_reservation"] == reservation
    assert reservation["used"] == parent["infrastructure_recovery"]["used"] + 1
    assert event["infrastructure_recovery_policy"]["used"] + 1 == reservation["used"]
    assert event["infrastructure_recovery_policy"]["limit"] == parent["infrastructure_recovery"]["limit"]
    assert reservation["used"] <= event["infrastructure_recovery_policy"]["limit"]
    assert event["classification"] == "transient_infrastructure"
    assert event["recovery"]["action"] == parent["recovery_decision"] == "relaunch_incomplete_wave"
    assert event["recovery"]["relaunch_allowed"] is True
    observation = event["observation"]
    assert observation["state"] == "cancelled" and observation["reason_code"] == proof["cause"].upper()
    assert observation["exact_identity"] is True and observation["workload_observable"] is True
    _assert_recovery_gates(parent, run_id, event)


def _assert_recovery_gates(parent, run_id, event):
    outputs, preflight = event["outputs"], event["preflight"]
    declared = [item["uri"].rstrip("/") + "/" if item.get("kind") == "directory" else item["uri"]
                for item in parent["outputs"]]
    assert declared and outputs["declared"] == outputs["missing"] == declared
    assert outputs["status"] == "absent" and not outputs["valid"] and not outputs["error"]
    assert preflight["relaunch_ready"] is True
    assert preflight["checks"]["credentials_access"] == "<redacted>"
    assert all(preflight["checks"][check] == "pass" for check in _CHECKS - {"credentials_access"})
    assert preflight["observed_at"] and preflight["scope"] == dict(
        source="default_sdk_recovery_preflight", run_id=run_id, wave_key=parent["key"],
        attempt=parent["attempt"], rendered_wave_sha256=parent["partial_launch"]["rendered_wave_sha256"],
    )
    assert re.fullmatch("[0-9a-f]{64}", preflight["scope"]["rendered_wave_sha256"])



def _assert_successful_partial(client, bucket, prefix, wave, run_id):
    proof = wave["partial_launch"]
    assert wave["recovery_decision"] == "reuse_completed_wave"
    assert wave["launch_sequence"] > 0 and not wave["observations"]
    assert proof["schema"] == 1 and proof["cause"] in _CAUSES
    assert wave["error_category"] == proof["cause"]
    assert proof["parent"] == _identity(wave, run_id) and proof["wave_key"] == wave["key"]
    assert proof["workload_observable_at_failure"] is False
    assert re.fullmatch("[0-9a-f]{64}", proof["rendered_wave_sha256"])
    assert not proof.get("recovery_reservation"), "Successful attempt reserved another launch"
    assert not wave["cancellation"]["error"]
    if wave["cancellation"]["state"] == "verified":
        assert proof["outputs_absent_before_cancel"]
    else:
        assert wave["cancellation"]["state"] == "not_applicable"
        assert not proof.get("outputs_absent_before_cancel")
    _assert_terminal_receipt(client, bucket, prefix, wave, run_id)
    assert wave["outputs"], "Successful partial launch lacks declared outputs"
    for output in wave["outputs"]:
        _assert_published_output(client, bucket, prefix, output)


def _assert_terminal_receipt(client, bucket, prefix, wave, run_id):
    root = prefix + "npa-workflow/supervisor/attempts/" + wave["logical_launch_id"] + "/"
    receipts = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=root):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.startswith(root + "attempt_terminal-"):
                continue
            with client.get_object(Bucket=bucket, Key=key)["Body"] as body:
                raw = body.read()
            assert key == root + "attempt_terminal-" + hashlib.sha256(raw).hexdigest() + ".json"
            receipts.append(json.loads(raw))
    assert len(receipts) == 1, "Missing or conflicting exact terminal receipt"
    receipt = receipts[0]
    expected = _identity(wave, run_id)
    expected.pop("checkpoint_prefix")
    assert receipt["schema_version"] == "npa.workflow.supervisor.v1"
    assert receipt["phase"] == "attempt_terminal" and receipt["attempt_identity"] == expected
    assert receipt["classification"] == wave["partial_launch"]["cause"]
    assert receipt["recovery"]["action"] == "reuse_completed_wave"
    fields = ("key", "states", "attempt", "job_id", "job_name", "logical_launch_id", "status", "sky_status",
              "partial_launch", "recovery_reservation", "recovery_resumed", "cancellation", "observations",
              "outputs", "immutable_identity", "infrastructure_recovery", "replayed", "adopted",
              "error_category", "recovery_decision", "launch_sequence", "scheduler_fence_sequence")
    assert all(receipt["attempt"][field] == wave[field] for field in fields), "Terminal receipt differs from runtime"


def _assert_published_output(client, bucket, prefix, output):
    selected = urlsplit(output["uri"])
    key = selected.path.lstrip("/")
    assert selected.scheme == "s3" and selected.netloc == bucket
    assert not selected.query and not selected.fragment
    assert key.startswith(prefix) and all(part not in {".", ".."} for part in key.split("/"))
    kind = output.get("kind") or "file"
    assert kind in {"file", "directory"}
    if kind == "file" and not output["uri"].endswith("/"):
        assert client.head_object(Bucket=bucket, Key=key)["ContentLength"] > 0, "Declared file is empty"
        return
    directory = key.rstrip("/") + "/"
    nonempty = False
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=directory):
        for item in page.get("Contents", []):
            assert item["Key"].startswith(directory)
            nonempty = nonempty or item.get("Size", 0) > 0
    assert nonempty, "Declared output directory is missing or empty"
