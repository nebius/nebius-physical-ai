"""Pin one fresh managed-policy lifecycle around every unstarted campaign case."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from npa.clients.storage import StoragePreconditionFailed
from npa.workflows.behavior_challenge import campaign, campaign_runner
from npa.workflows.behavior_challenge.case_store import CaseStore


class _MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.revision = 0

    def read_bytes_with_etag(self, uri):
        return self.objects.get(uri)

    def put_bytes_conditional(
        self, payload, uri, *, if_match="", if_none_match=False, **_
    ):
        current = self.objects.get(uri)
        if (if_none_match and current) or (
            if_match and (not current or current[1] != if_match)
        ):
            raise StoragePreconditionFailed("conditional conflict")
        self.revision += 1
        etag = str(self.revision)
        self.objects[uri] = payload, etag
        return etag


@pytest.fixture
def panel():
    registry = ["picking_up_trash"] + [f"task_{index}" for index in range(99)]
    artifact = {"sha256": "a" * 64, "bytes": 1}
    identity = campaign.freeze_policy_identity(
        "fixture", {"checkpoint": artifact, "serving": artifact}
    )
    return campaign.declare_panel(identity, registry, registry[:1], "development")


def _store(panel, suffix="state"):
    return CaseStore(
        _MemoryStorage(),
        f"s3://example-bucket/{suffix}",
        panel["panel_id"],
    )


def _stub_artifact_publication(monkeypatch, events=None):
    def inspect(_, case):
        if events is not None:
            events.append("inspect")
        return {"case_id": case["case_id"]}

    def bind(_, record):
        if events is not None:
            events.append("bind")
        return record

    def publish(*_):
        if events is not None:
            events.append("publish")

    monkeypatch.setattr(campaign_runner, "inspect_rollout", inspect)
    monkeypatch.setattr(campaign_runner, "bind_inspected_rollout", bind)
    monkeypatch.setattr(campaign_runner, "_record_originals", publish)


def test_policy_is_ready_while_claimed_and_stops_before_inspection(
    panel, tmp_path, monkeypatch
):
    partition = campaign.partition_panel(panel, len(panel["cases"]))
    store = _store(panel)
    case = panel["cases"][0]
    events = []
    _stub_artifact_publication(monkeypatch, events)

    @contextmanager
    def prepare(selected, _output):
        assert selected == case
        assert store.read(case).record["state"] == "claimed"
        events.append("ready")
        yield
        assert store.read(case).record["state"] == "started"
        events.append("stopped")

    def execute(selected, _output):
        assert selected == case
        assert store.read(case).record["state"] == "started"
        events.append("execute")

    campaign_runner.run_partition(
        panel,
        partition,
        0,
        store,
        tmp_path,
        execute,
        prepare_case=prepare,
    )

    assert events == ["ready", "execute", "stopped", "inspect", "bind", "publish"]


def test_policy_startup_failure_leaves_a_reclaimable_prestart_case(panel, tmp_path):
    partition = campaign.partition_panel(panel, len(panel["cases"]))
    store = _store(panel)
    case = panel["cases"][0]

    @contextmanager
    def unavailable(*_):
        raise RuntimeError("policy startup failed")
        yield

    def execute(*_):
        raise AssertionError("evaluator ran before policy readiness")

    with pytest.raises(RuntimeError, match="policy startup failed"):
        campaign_runner.run_partition(
            panel,
            partition,
            0,
            store,
            tmp_path,
            execute,
            prepare_case=unavailable,
        )

    original = store.read(case)
    assert original.record["state"] == "claimed"
    replacement = store.claim(case, "replacement-worker")
    assert replacement.record["state"] == "claimed"
    assert (
        replacement.record["previous_claims"][0]["claim_id"]
        == (original.record["claim_id"])
    )


def test_each_case_gets_the_same_fresh_rng_across_partition_shapes(
    panel, tmp_path, monkeypatch
):
    _stub_artifact_publication(monkeypatch)

    def evaluate(worker_count):
        partition = campaign.partition_panel(panel, worker_count)
        store = _store(panel, f"state-{worker_count}")
        observed = {}
        active = None

        @contextmanager
        def prepare(case, _output):
            nonlocal active
            active = {"case_id": case["case_id"], "rng_ordinal": 0}
            yield
            active = None

        def execute(case, _output):
            assert active is not None and active["case_id"] == case["case_id"]
            observed[case["case_id"]] = active["rng_ordinal"]
            active["rng_ordinal"] += 1

        for worker_index in range(worker_count):
            campaign_runner.run_partition(
                panel,
                partition,
                worker_index,
                store,
                tmp_path / str(worker_count),
                execute,
                prepare_case=prepare,
            )
        return observed

    serial = evaluate(1)
    parallel = evaluate(3)
    assert serial == parallel
    assert set(serial.values()) == {0}
    assert len(serial) == len(panel["cases"])


def test_started_and_completed_recovery_skips_policy_preparation(
    panel, tmp_path, monkeypatch
):
    partition = campaign.partition_panel(panel, 1)
    versions = {}
    for index, case in enumerate(panel["cases"]):
        versions[case["case_id"]] = SimpleNamespace(
            record={
                "state": "started" if index % 2 == 0 else "complete",
                "case": case,
                "claim_id": f"claim-{index}",
            }
        )
    store = SimpleNamespace(
        panel_id=panel["panel_id"],
        read=lambda case: versions[case["case_id"]],
    )
    recovered = []

    def recover(_, version, _output, _panel):
        recovered.append(version.record["case"]["case_id"])
        return {"case_id": version.record["case"]["case_id"]}

    @contextmanager
    def prepare(*_):
        raise AssertionError("recovery started a fresh policy")
        yield

    monkeypatch.setattr(campaign_runner, "_recover_persistent_case", recover)
    records = campaign_runner.run_partition(
        panel,
        partition,
        0,
        store,
        tmp_path,
        lambda *_: (_ for _ in ()).throw(AssertionError("recovery evaluated")),
        prepare_case=prepare,
    )

    assert recovered == [case["case_id"] for case in panel["cases"]]
    assert records == [{"case_id": value} for value in recovered]
