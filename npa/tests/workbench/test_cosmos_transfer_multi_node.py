"""Multi-node augment: shard the sampled combos across a gang-scheduled block.

``resources.gpu.num_nodes: N`` makes SkyPilot gang-schedule N identical pods for
the one augment task and run the SAME command in each. Without a shard every node
would render every variant, so these tests pin the two properties that make the
block real work rather than N copies of it:

* each rank renders only its stride of the combos, on its own GPUs; and
* rank 0 joins the per-node shard manifests into the one ``manifest.json`` the
  downstream stages read, or fails naming the ranks that never reported.

No GPU or Cosmos runtime is touched: inference and S3 are both fakes.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import cosmos2
from npa.clients.storage import StorageError, StoragePreconditionFailed
from npa.workbench.cosmos import transfer as tx

runner = CliRunner()
ATTEMPT = "wave-attempt-1"


@pytest.fixture(autouse=True)
def _scheduler_fence_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_WORKFLOW_FENCE_SEQUENCE", "1")
    monkeypatch.setenv("NPA_WORKFLOW_FENCE_ATTEMPT", "1")


class FakeStorage:
    """In-memory stand-in for the S3 objects a gang's nodes exchange."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.version = 0

    def _next_etag(self) -> str:
        self.version += 1
        return f'"etag-{self.version}"'

    def upload_file(self, local: str, uri: str) -> str:
        self.objects[uri] = Path(local).read_bytes()
        self.etags[uri] = self._next_etag()
        return uri

    def read_bytes_with_etag(self, uri: str):
        if uri not in self.objects:
            return None
        return self.objects[uri], self.etags[uri]

    def put_bytes_conditional(
        self,
        payload: bytes,
        uri: str,
        *,
        if_match: str = "",
        if_none_match: bool = False,
        content_type: str = "application/octet-stream",
    ) -> str:
        del content_type
        if if_none_match:
            if uri in self.objects:
                raise StoragePreconditionFailed(uri)
        elif not if_match or self.etags.get(uri) != if_match:
            raise StoragePreconditionFailed(uri)
        self.objects[uri] = payload
        self.etags[uri] = self._next_etag()
        return self.etags[uri]

    def download_path(self, uri: str, local_path: str) -> str:
        if uri not in self.objects:
            return local_path
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[uri])
        return str(dest)


def _clip(index: int, *, run_id: str = "run1", attempt_id: str = ATTEMPT) -> dict:
    base = f"s3://bkt/{run_id}/cosmos_augmented/_attempts/{attempt_id}"
    return {
        "clip": f"aug-{run_id}-{index}",
        "variant_index": index,
        "augmented_video_uri": f"{base}/aug-{run_id}-{index}/augmented_video.mp4",
        "frames_uri": f"{base}/aug-{run_id}-{index}/",
        "frames": [{"frame_id": "frame-00000", "uri": "s3://bkt/f.png"}],
        "frame_count": 1,
        "variables": {"prompt": f"scene {index}"},
        "video_bytes": 1000,
        "input_conditioned": True,
        "control": "edge",
    }


def _attempt_clip(index: int, output_uri: str, attempt_id: str) -> dict:
    clip = _clip(index)
    attempt_uri = tx.attempt_output_uri_for(output_uri, attempt_id)
    clip["augmented_video_uri"] = (
        f"{attempt_uri}/aug-run1-{index}/augmented_video.mp4"
    )
    clip["frames_uri"] = f"{attempt_uri}/aug-run1-{index}/"
    return clip


def _write_shard(*args, attempt_id: str = ATTEMPT, **kwargs):
    clips = []
    for source in args[0]:
        clip = dict(source)
        for field in ("augmented_video_uri", "frames_uri"):
            value = str(clip.get(field) or "")
            if "/_attempts/" in value:
                prefix, rest = value.split("/_attempts/", 1)
                _old, _separator, suffix = rest.partition("/")
                clip[field] = f"{prefix}/_attempts/{attempt_id}/{suffix}"
        clips.append(clip)
    kwargs.setdefault("scheduler_fence_sequence", 1)
    kwargs.setdefault("scheduler_fence_attempt", 1)
    kwargs.setdefault("scheduler_launch_id", "test-launch")
    kwargs.setdefault("logical_wave_id", "test-wave")
    kwargs.setdefault("publication_generation", 1)
    return tx.write_shard_manifest(clips, *args[1:], attempt_id=attempt_id, **kwargs)


def _seed_claim(
    storage: FakeStorage,
    output_uri: str,
    *,
    attempt_id: str = ATTEMPT,
    generation: int = 1,
    run_id: str = "run1",
    node_count: int = 2,
) -> str:
    claim = {
        "schema": tx.TRANSFER_MANIFEST_SCHEMA,
        "mode": tx.TRANSFER_MANIFEST_MODE,
        "status": tx.PUBLICATION_CLAIM_STATUS,
        "run_id": run_id,
        "attempt_id": attempt_id,
        tx.PUBLICATION_GENERATION_FIELD: generation,
        "node_count": node_count,
        "logical_wave_id": "test-wave",
        "membership_digest": "test-members",
        "scheduler_fence_sequence": generation,
        "scheduler_fence_attempt": 1,
        "scheduler_launch_id": "test-launch",
    }
    canonical = tx.transfer_manifest_uri_for(output_uri)
    existing = storage.read_bytes_with_etag(canonical)
    return storage.put_bytes_conditional(
        (json.dumps(claim, sort_keys=True) + "\n").encode(),
        canonical,
        if_match=existing[1] if existing else "",
        if_none_match=existing is None,
        content_type="application/json",
    )


def _merge_shards(*args, attempt_id: str = ATTEMPT, **kwargs):
    storage = kwargs["storage_client"]
    output_uri = args[0]
    if "publication_claim_etag" not in kwargs:
        kwargs["publication_claim_etag"] = _seed_claim(
            storage,
            output_uri,
            attempt_id=attempt_id,
            run_id=kwargs.get("run_id", ""),
            node_count=kwargs.get("node_count", 1),
        )
        kwargs["publication_generation"] = 1
    return tx.merge_shard_manifests(*args, attempt_id=attempt_id, **kwargs)


def test_shard_indices_stride_so_every_node_gets_a_balanced_share() -> None:
    # 5 combos over 2 nodes: striding keeps the nodes within one variant of each
    # other; contiguous blocks would give one node 3 and the other 2 in sequence.
    assert cosmos2._shard_indices(5, rank=0, nodes=2) == [0, 2, 4]
    assert cosmos2._shard_indices(5, rank=1, nodes=2) == [1, 3]
    # Every variant is rendered exactly once across the gang.
    covered = [i for rank in range(3) for i in cosmos2._shard_indices(7, rank=rank, nodes=3)]
    assert sorted(covered) == list(range(7))
    # More nodes than variants: the surplus ranks render nothing.
    assert cosmos2._shard_indices(2, rank=3, nodes=4) == []
    # Single node keeps the whole list, in order.
    assert cosmos2._shard_indices(3, rank=0, nodes=1) == [0, 1, 2]


def test_gang_shard_reads_skypilot_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.delenv("NPA_COSMOS_NODE_RANK", raising=False)
    monkeypatch.delenv("SKYPILOT_NUM_NODES", raising=False)
    monkeypatch.delenv("SKYPILOT_NODE_RANK", raising=False)
    # No gang: a single-node augment, which is the shipped default.
    assert cosmos2._gang_shard() == (0, 1)

    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "4")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "4")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "2")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2\n10.0.0.3\n10.0.0.4")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "42")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "7")
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "loop-2-wave-1")
    assert cosmos2._gang_shard() == (2, 4)

    # A local gang is reproducible only with an explicit shared attempt id.
    for name in (
        "SKYPILOT_NUM_NODES",
        "SKYPILOT_NODE_RANK",
        "SKYPILOT_NODE_IPS",
        "SKYPILOT_INTERNAL_JOB_ID",
        "SKYPILOT_MANAGED_JOB_ID",
        "NPA_WORKFLOW_ATTEMPT_ID",
        "NPA_WORKFLOW_FENCE_SEQUENCE",
        "NPA_WORKFLOW_FENCE_ATTEMPT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "1")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)
    assert cosmos2._gang_shard() == (1, 2)


def test_gang_identity_cross_checks_renderer_and_skypilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "3")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2\n10.0.0.3")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "42")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "7")
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "wave")
    with pytest.raises(cosmos2.typer.BadParameter, match="contradictory"):
        cosmos2._gang_environment()


def test_managed_job_evidence_without_authoritative_count_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "7")

    with pytest.raises(cosmos2.typer.BadParameter, match="authoritative"):
        cosmos2._gang_environment()


def test_ordinary_single_node_skypilot_task_does_not_require_cosmos_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sky exports these for all tasks; only a real gang needs NPA's contract."""

    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "42")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "7")

    assert cosmos2._gang_contract_required() is False

    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    assert cosmos2._gang_contract_required() is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NPA_COSMOS_NODE_RANK", "1"),
        ("NPA_COSMOS_ATTEMPT_ID", ATTEMPT),
    ],
)
def test_partial_local_identity_without_authoritative_count_fails_closed(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.setenv(name, value)

    assert cosmos2._gang_contract_required() is True
    with pytest.raises(cosmos2.typer.BadParameter, match="authoritative"):
        cosmos2._gang_environment()


def test_partial_single_node_sky_identity_requires_full_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.delenv("SKYPILOT_NODE_IPS", raising=False)

    assert cosmos2._gang_contract_required() is True
    with pytest.raises(cosmos2.typer.BadParameter, match="authoritative"):
        cosmos2._gang_environment()


def test_direct_cli_sky_gang_without_authoritative_count_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_COSMOS_NODE_COUNT", raising=False)
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "42")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "7")
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "wave")

    def unexpected_runtime_probe() -> bool:
        raise AssertionError("runtime must not be probed before gang identity")

    monkeypatch.setattr(tx, "cosmos_transfer_available", unexpected_runtime_probe)
    result = runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/run1/input/",
            "--output-uri",
            "s3://bkt/run1/cosmos_augmented/",
            "--run-id",
            "run1",
            "--configs-uri",
            "s3://bkt/run1/configs/",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "authoritative" in result.output


@pytest.mark.parametrize(
    ("configs_uri", "output_uri", "match"),
    [
        (
            "",
            "s3://bkt/run1/cosmos_augmented/",
            "requires a non-empty --configs-uri",
        ),
        (
            "/tmp/configs/",
            "/tmp/output",
            "requires an s3:// --output-uri",
        ),
    ],
)
def test_direct_cli_rejects_a_gang_without_active_durable_sharding(
    monkeypatch: pytest.MonkeyPatch,
    configs_uri: str,
    output_uri: str,
    match: str,
) -> None:
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "0")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)

    def unexpected_runtime_probe() -> bool:
        raise AssertionError("runtime must not be probed before gang preflight")

    monkeypatch.setattr(tx, "cosmos_transfer_available", unexpected_runtime_probe)
    argv = [
        "workbench",
        "cosmos2",
        "transfer",
        "--input-uri",
        "s3://bkt/run1/input/",
        "--output-uri",
        output_uri,
        "--run-id",
        "run1",
        "--execute",
    ]
    if configs_uri:
        argv.extend(("--configs-uri", configs_uri))

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert match in result.output


def test_local_multi_node_execution_fails_before_an_unfenced_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "0")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)
    with pytest.raises(cosmos2.typer.BadParameter, match="publication fence"):
        cosmos2._gang_identity(output_uri="s3://bkt/out/", run_id="run1")


def test_managed_single_node_claims_the_same_scheduler_publication_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "1")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "1")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "101")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "9")
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "loop-1")

    identity = cosmos2._gang_identity(
        output_uri="s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
    )

    assert identity[:2] == (0, 1)
    assert len(identity[2]) == 64
    assert identity[3]
    assert identity[4] == 1
    with pytest.raises(RuntimeError, match="stale|duplicates"):
        cosmos2._gang_identity(
            output_uri="s3://bkt/run1/cosmos_augmented/",
            run_id="run1",
            storage_client=storage,
        )


def test_attempt_identity_is_shared_and_only_outer_retry_may_advance_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    shared: dict[str, object] = {}

    def exchange(**kwargs):
        if kwargs["offered"] is not None:
            shared.clear()
            shared.update(kwargs["offered"])
        return dict(shared)

    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("SKYPILOT_NUM_NODES", "2")
    monkeypatch.setenv("SKYPILOT_NODE_IPS", "10.0.0.1\n10.0.0.2")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "101")
    monkeypatch.setenv("SKYPILOT_MANAGED_JOB_ID", "9")
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "loop-iteration-2")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    rank0 = cosmos2._gang_identity(
        output_uri="s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
        rendezvous=exchange,
    )
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "1")
    rank1 = cosmos2._gang_identity(
        output_uri="s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
        rendezvous=exchange,
    )
    assert rank0[2] == rank1[2]

    # SkyPilot_TASK_ID is deliberately irrelevant.  SkyPilot documents that it
    # stays constant across managed-job recovery, and changing it here cannot
    # manufacture a new shard identity.
    monkeypatch.setenv("SKYPILOT_TASK_ID", "not-part-of-the-identity")
    task_id_changed = cosmos2._gang_identity(
        output_uri="s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
        rendezvous=exchange,
    )
    assert task_id_changed[2] == rank1[2]

    # Stock SkyPilot recovery retains the scheduler token. A replacement rank 0
    # must fail before GPU work instead of using arrival order to steal a newer
    # canonical generation.
    monkeypatch.setenv("SKYPILOT_TASK_ID", "constant-across-recovery")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "0")
    monkeypatch.setenv("SKYPILOT_INTERNAL_JOB_ID", "102")
    with pytest.raises(RuntimeError, match="stale|duplicates"):
        cosmos2._gang_identity(
            output_uri="s3://bkt/run1/cosmos_augmented/",
            run_id="run1",
            storage_client=storage,
            rendezvous=exchange,
        )

    # Once that managed job is terminal, the durable NPA runtime explicitly
    # retries the wave with a higher ordered token shared by the whole new gang.
    monkeypatch.setenv("NPA_WORKFLOW_ATTEMPT_ID", "loop-iteration-2-npa-retry-2")
    monkeypatch.setenv("NPA_WORKFLOW_FENCE_ATTEMPT", "2")
    recovered = cosmos2._gang_identity(
        output_uri="s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        storage_client=storage,
        rendezvous=exchange,
    )
    assert recovered[2] != rank1[2]
    assert recovered[4] == rank0[4] + 1


def test_rank_zero_rendezvous_shares_one_generation_over_scheduler_membership() -> None:
    offered = {
        "attempt_id": "a" * 64,
        "publication_generation": 7,
        "logical_wave_id": "loop-2",
        "membership_digest": "members",
        "internal_job_id": "42",
        "node_count": 2,
    }
    leader = cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=2,
        node_ips=["127.0.0.1", "127.0.0.2"],
        logical_wave_id="loop-2",
        membership_digest="members",
        internal_job_id="42",
        offered=offered,
    )
    follower = cosmos2._sky_gang_rendezvous(
        rank=1,
        node_count=2,
        node_ips=["127.0.0.1", "127.0.0.2"],
        logical_wave_id="loop-2",
        membership_digest="members",
        internal_job_id="42",
        offered=None,
    )
    assert leader == follower == offered


def test_rendezvous_retries_a_rank_when_the_first_response_is_not_acknowledged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = {
        "attempt_id": "f" * 64,
        "publication_generation": 9,
        "logical_wave_id": "loop-3",
        "membership_digest": "members",
        "internal_job_id": "44",
        "node_count": 3,
    }
    claimant_nonce = "1" * 64
    monkeypatch.setattr(cosmos2.secrets, "token_hex", lambda _size: claimant_nonce)
    node_ips = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]
    cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-3",
        membership_digest="members",
        internal_job_id="44",
        offered=offered,
    )
    request = {
        "protocol": "npa.cosmos.gang-attempt/v1",
        "rank": 1,
        "claimant_nonce": claimant_nonce,
        "node_count": 3,
        "logical_wave_id": "loop-3",
        "membership_digest": "members",
        "internal_job_id": "44",
    }
    port = cosmos2._rendezvous_port("loop-3", "members")
    with cosmos2.socket.create_connection(
        (node_ips[0], port), timeout=5.0, source_address=(node_ips[1], 0)
    ) as connection:
        connection.sendall(
            json.dumps(request, sort_keys=True).encode("utf-8") + b"\n"
        )
        discarded = cosmos2._recv_json_line(connection)
        assert discarded == {
            "protocol": "npa.cosmos.gang-attempt/v1",
            **offered,
        }
        # Simulate the response being lost above the transport after sendall()
        # succeeded: close without the application acknowledgement.

    assert cosmos2._sky_gang_rendezvous(
        rank=1,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-3",
        membership_digest="members",
        internal_job_id="44",
        offered=None,
    ) == offered
    assert cosmos2._sky_gang_rendezvous(
        rank=2,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-3",
        membership_digest="members",
        internal_job_id="44",
        offered=None,
    ) == offered


def test_hung_ack_does_not_block_a_later_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = {
        "attempt_id": "e" * 64,
        "publication_generation": 10,
        "logical_wave_id": "loop-hung-ack",
        "membership_digest": "members",
        "internal_job_id": "45",
        "node_count": 3,
    }
    claimant_nonce = "2" * 64
    monkeypatch.setattr(cosmos2.secrets, "token_hex", lambda _size: claimant_nonce)
    node_ips = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]
    cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-hung-ack",
        membership_digest="members",
        internal_job_id="45",
        offered=offered,
    )
    request = {
        "protocol": "npa.cosmos.gang-attempt/v1",
        "rank": 1,
        "claimant_nonce": claimant_nonce,
        "node_count": 3,
        "logical_wave_id": "loop-hung-ack",
        "membership_digest": "members",
        "internal_job_id": "45",
    }
    port = cosmos2._rendezvous_port("loop-hung-ack", "members")
    held = cosmos2.socket.create_connection(
        (node_ips[0], port), timeout=5.0, source_address=(node_ips[1], 0)
    )
    try:
        held.sendall(json.dumps(request, sort_keys=True).encode("utf-8") + b"\n")
        response = cosmos2._recv_json_line(held)
        assert response == {
            "protocol": "npa.cosmos.gang-attempt/v1",
            **offered,
        }

        # Rank 1 deliberately keeps the post-response socket open without an ACK.
        # A per-connection handler must still let rank 2 receive and acknowledge.
        assert cosmos2._sky_gang_rendezvous(
            rank=2,
            node_count=3,
            node_ips=node_ips,
            logical_wave_id="loop-hung-ack",
            membership_digest="members",
            internal_job_id="45",
            offered=None,
        ) == offered
    finally:
        held.close()

    # The unacknowledged rank remains retryable and lets the server finish cleanly.
    assert cosmos2._sky_gang_rendezvous(
        rank=1,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-hung-ack",
        membership_digest="members",
        internal_job_id="45",
        offered=None,
    ) == offered


def test_concurrent_duplicate_rank_has_exactly_one_committed_member() -> None:
    offered = {
        "attempt_id": "d" * 64,
        "publication_generation": 11,
        "logical_wave_id": "loop-duplicate-rank",
        "membership_digest": "members",
        "internal_job_id": "46",
        "node_count": 3,
    }
    node_ips = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]
    cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-duplicate-rank",
        membership_digest="members",
        internal_job_id="46",
        offered=offered,
    )
    request = {
        "protocol": "npa.cosmos.gang-attempt/v1",
        "rank": 1,
        "node_count": 3,
        "logical_wave_id": "loop-duplicate-rank",
        "membership_digest": "members",
        "internal_job_id": "46",
    }
    port = cosmos2._rendezvous_port("loop-duplicate-rank", "members")
    barrier = threading.Barrier(3)
    responses: list[tuple[socket.socket, dict[str, object], str]] = []
    response_lock = threading.Lock()

    def request_generation(claimant_nonce: str) -> None:
        connection = cosmos2.socket.create_connection(
            (node_ips[0], port), timeout=5.0, source_address=(node_ips[1], 0)
        )
        barrier.wait()
        connection.sendall(
            json.dumps(
                {**request, "claimant_nonce": claimant_nonce}, sort_keys=True
            ).encode("utf-8")
            + b"\n"
        )
        response = cosmos2._recv_json_line(connection)
        with response_lock:
            responses.append((connection, response, claimant_nonce))

    contenders = [
        threading.Thread(target=request_generation, args=(nonce,))
        for nonce in ("3" * 64, "4" * 64)
    ]
    for contender in contenders:
        contender.start()
    barrier.wait()
    for contender in contenders:
        contender.join(timeout=5.0)
        assert not contender.is_alive()

    winners = [item for item in responses if not item[1].get("error")]
    losers = [item for item in responses if item[1].get("error")]
    assert len(winners) == 1
    assert len(losers) == 1
    winner_connection, winner_response, winner_nonce = winners[0]
    assert winner_response == {
        "protocol": "npa.cosmos.gang-attempt/v1",
        **offered,
    }
    winner_connection.sendall(
        json.dumps(
            {
                "protocol": "npa.cosmos.gang-attempt/v1",
                "rank": 1,
                "claimant_nonce": winner_nonce,
                "acknowledged": True,
            },
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    assert cosmos2._recv_json_line(winner_connection) == {
        "protocol": "npa.cosmos.gang-attempt/v1",
        "rank": 1,
        "claimant_nonce": winner_nonce,
        "committed": True,
    }
    winner_connection.sendall(
        json.dumps(
            {
                "protocol": "npa.cosmos.gang-attempt/v1",
                "rank": 1,
                "claimant_nonce": winner_nonce,
                "commitment_received": True,
            },
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    for connection, _response, _nonce in responses:
        connection.close()

    assert cosmos2._sky_gang_rendezvous(
        rank=2,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-duplicate-rank",
        membership_digest="members",
        internal_job_id="46",
        offered=None,
    ) == offered


def test_lost_commit_confirmation_is_idempotent_for_the_same_claimant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offered = {
        "attempt_id": "c" * 64,
        "publication_generation": 12,
        "logical_wave_id": "loop-lost-commit",
        "membership_digest": "members",
        "internal_job_id": "47",
        "node_count": 2,
    }
    node_ips = ["127.0.0.1", "127.0.0.2"]
    claimant_nonce = "5" * 64
    monkeypatch.setattr(cosmos2.secrets, "token_hex", lambda _size: claimant_nonce)
    cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=2,
        node_ips=node_ips,
        logical_wave_id="loop-lost-commit",
        membership_digest="members",
        internal_job_id="47",
        offered=offered,
    )
    request = {
        "protocol": "npa.cosmos.gang-attempt/v1",
        "rank": 1,
        "claimant_nonce": claimant_nonce,
        "node_count": 2,
        "logical_wave_id": "loop-lost-commit",
        "membership_digest": "members",
        "internal_job_id": "47",
    }
    port = cosmos2._rendezvous_port("loop-lost-commit", "members")
    with cosmos2.socket.create_connection(
        (node_ips[0], port), timeout=5.0, source_address=(node_ips[1], 0)
    ) as connection:
        connection.sendall(
            json.dumps(request, sort_keys=True).encode("utf-8") + b"\n"
        )
        assert cosmos2._recv_json_line(connection) == {
            "protocol": "npa.cosmos.gang-attempt/v1",
            **offered,
        }
        connection.sendall(
            json.dumps(
                {
                    "protocol": "npa.cosmos.gang-attempt/v1",
                    "rank": 1,
                    "claimant_nonce": claimant_nonce,
                    "acknowledged": True,
                },
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        discarded = cosmos2._recv_json_line(connection)
        assert discarded["committed"] is True
        # Discard the commitment confirmation and close without its receipt.

    assert cosmos2._sky_gang_rendezvous(
        rank=1,
        node_count=2,
        node_ips=node_ips,
        logical_wave_id="loop-lost-commit",
        membership_digest="members",
        internal_job_id="47",
        offered=None,
    ) == offered


def test_recovery_rendezvous_rejects_stale_launch_and_duplicate_rank() -> None:
    offered = {
        "attempt_id": "b" * 64,
        "publication_generation": 8,
        "logical_wave_id": "loop-2",
        "membership_digest": "replacement-members",
        "internal_job_id": "new-job-43",
        "node_count": 3,
    }
    node_ips = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]
    cosmos2._sky_gang_rendezvous(
        rank=0,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-2",
        membership_digest="replacement-members",
        internal_job_id="new-job-43",
        offered=offered,
    )

    # An escaped process from the prior managed-job launch still owns rank 1's
    # IP, but its scheduler-issued internal job id cannot join the replacement.
    with pytest.raises(cosmos2.typer.BadParameter, match="contradictory"):
        cosmos2._sky_gang_rendezvous(
            rank=1,
            node_count=3,
            node_ips=node_ips,
            logical_wave_id="loop-2",
            membership_digest="replacement-members",
            internal_job_id="old-job-42",
            offered=None,
        )
    assert cosmos2._sky_gang_rendezvous(
        rank=1,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-2",
        membership_digest="replacement-members",
        internal_job_id="new-job-43",
        offered=None,
    ) == offered
    with pytest.raises(cosmos2.typer.BadParameter, match="contradictory"):
        cosmos2._sky_gang_rendezvous(
            rank=1,
            node_count=3,
            node_ips=node_ips,
            logical_wave_id="loop-2",
            membership_digest="replacement-members",
            internal_job_id="new-job-43",
            offered=None,
        )
    assert cosmos2._sky_gang_rendezvous(
        rank=2,
        node_count=3,
        node_ips=node_ips,
        logical_wave_id="loop-2",
        membership_digest="replacement-members",
        internal_job_id="new-job-43",
        offered=None,
    ) == offered


def test_identity_rendezvous_has_only_an_explicit_opt_in_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_COSMOS_IDENTITY_TIMEOUT_S", "0")
    with pytest.raises(cosmos2.typer.BadParameter, match="timed out.*rank 1"):
        cosmos2._sky_gang_rendezvous(
            rank=1,
            node_count=2,
            node_ips=["127.0.0.1", "127.0.0.2"],
            logical_wave_id="absent-leader",
            membership_digest="members",
            internal_job_id="42",
            offered=None,
        )


def test_live_validation_faults_are_exactly_run_rank_generation_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_SCOPE", "task-run")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_DELAY_S", "2.5")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_DELAY_RANK", "1")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_DELAY_GENERATION", "2")
    monkeypatch.setattr(cosmos2.time, "sleep", slept.append)
    cosmos2._apply_validation_fault(
        run_id="task-run", rank=1, generation=2, phase="before-render"
    )
    cosmos2._apply_validation_fault(
        run_id="task-run", rank=0, generation=2, phase="before-render"
    )
    assert slept == [2.5]

    monkeypatch.setenv("NPA_COSMOS_VALIDATION_FAIL_PHASE", "after-shard")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_FAIL_RANK", "0")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_FAIL_GENERATION", "1")
    with pytest.raises(RuntimeError, match="task-scoped.*generation=1"):
        cosmos2._apply_validation_fault(
            run_id="task-run", rank=0, generation=1, phase="after-shard"
        )


def test_live_validation_fault_refuses_an_unscoped_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_DELAY_S", "1")
    monkeypatch.setenv("NPA_COSMOS_VALIDATION_SCOPE", "some-other-run")
    with pytest.raises(cosmos2.typer.BadParameter, match="exact non-empty --run-id"):
        cosmos2._apply_validation_fault(
            run_id="task-run", rank=1, generation=2, phase="before-render"
        )


@pytest.mark.parametrize(
    ("rank", "nodes"),
    [("4", "4"), ("-1", "2"), ("x", "2"), ("0", "0")],
)
def test_gang_shard_fails_closed_on_an_inconsistent_identity(
    monkeypatch: pytest.MonkeyPatch, rank: str, nodes: str
) -> None:
    """Collapsing to one node would duplicate every variant on every GPU."""

    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", nodes)
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", rank)
    with pytest.raises(cosmos2.typer.BadParameter, match="multi-node augment identity"):
        cosmos2._gang_shard()


def test_shard_manifests_merge_into_the_single_node_run_manifest() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    # Rank 1 finishes first; the merge must still restore sampled combo order.
    _write_shard(
        [_clip(1), _clip(3)],
        output_uri,
        run_id="run1",
        rank=1,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )
    _write_shard(
        [_clip(0), _clip(2)],
        output_uri,
        run_id="run1",
        rank=0,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )

    manifest = _merge_shards(
        output_uri, run_id="run1", node_count=2, storage_client=storage
    )

    assert manifest["schema"] == tx.TRANSFER_MANIFEST_SCHEMA
    assert manifest["clips"] == [
        "aug-run1-0",
        "aug-run1-1",
        "aug-run1-2",
        "aug-run1-3",
    ]
    assert manifest["variant_count"] == 4
    assert manifest["multiply_mode"] == "multi-variant"
    assert manifest["node_count"] == 2
    # Concurrent renders across the whole block: 2 GPUs on each of 2 nodes.
    assert manifest["variant_parallelism"] == 4
    assert [s["rank"] for s in manifest["shards"]] == [0, 1]
    # The join is durable: it lands on the same key a single-node run writes.
    written = json.loads(storage.objects[tx.transfer_manifest_uri_for(output_uri)])
    assert written["clips"] == manifest["clips"]
    assert (
        tx.shard_manifest_uri_for(output_uri, 1, attempt_id=ATTEMPT)
        == "s3://bkt/run1/cosmos_augmented/_attempts/wave-attempt-1/manifest-rank-1.json"
    )


def test_shard_manifests_and_clips_are_scoped_to_the_attempt_prefix() -> None:
    """Late recovery writes cannot touch the current attempt's object keys."""

    uri = tx.shard_manifest_uri_for(
        "s3://bkt/run1/cosmos_augmented/", 3, attempt_id=ATTEMPT
    )
    relative = uri.split("cosmos_augmented/", 1)[1]
    assert relative == "_attempts/wave-attempt-1/manifest-rank-3.json"
    assert tx.attempt_output_uri_for(
        "s3://bkt/run1/cosmos_augmented/", ATTEMPT
    ).endswith("/_attempts/wave-attempt-1")


def test_merge_waits_for_a_slow_rank_then_joins() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    waits: list[float] = []

    def late_arrival(seconds: float) -> None:
        waits.append(seconds)
        _write_shard(
            [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
            variant_parallelism=1, variant_total=2, storage_client=storage,
        )

    manifest = _merge_shards(
        output_uri,
        run_id="run1",
        node_count=2,
        storage_client=storage,
        poll_interval_s=0.5,
        sleep=late_arrival,
    )

    assert waits == [0.5]
    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1"]


def test_merge_ignores_a_stale_shard_until_the_current_rank_overwrites_it() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    _write_shard(
        [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
        attempt_id="prior-loop-same-run",
    )

    def current_arrival(_seconds: float) -> None:
        _write_shard(
            [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
            variant_parallelism=1, variant_total=2, storage_client=storage,
        )

    manifest = _merge_shards(
        output_uri,
        run_id="run1",
        node_count=2,
        storage_client=storage,
        sleep=current_arrival,
    )

    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1"]
    assert manifest["attempt_id"] == ATTEMPT
    assert {item["attempt_id"] for item in manifest["shards"]} == {ATTEMPT}


def test_merge_rejects_prior_recovery_attempt_with_identical_stable_fields() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
        attempt_id="recovered-launch",
    )
    _write_shard(
        [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
        attempt_id="failed-launch",
    )
    waits = 0

    def recovered_rank(_seconds: float) -> None:
        nonlocal waits
        waits += 1
        _write_shard(
            [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
            variant_parallelism=1, variant_total=2, storage_client=storage,
            attempt_id="recovered-launch",
        )

    manifest = _merge_shards(
        output_uri, run_id="run1", node_count=2, storage_client=storage,
        attempt_id="recovered-launch", sleep=recovered_rank,
    )
    assert waits == 1
    assert manifest["attempt_id"] == "recovered-launch"


def test_second_loop_waits_for_delayed_current_rank_not_late_prior_rank() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    first_id, first_etag, first_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="grade-loop-iteration-1",
        node_count=2,
        membership_digest="same-stable-membership",
        scheduler_fence_sequence=1,
        scheduler_fence_attempt=1,
        scheduler_launch_id="loop-1-job",
        storage_client=storage,
        nonce_factory=lambda: "a" * 32,
    )
    for rank in (0, 1):
        _write_shard(
            [_attempt_clip(rank, output_uri, first_id)],
            output_uri,
            run_id="run1",
            rank=rank,
            node_count=2,
            variant_total=2,
            storage_client=storage,
            attempt_id=first_id,
            scheduler_fence_sequence=1,
            scheduler_fence_attempt=1,
            scheduler_launch_id="loop-1-job",
            logical_wave_id="grade-loop-iteration-1",
            publication_generation=first_generation,
        )
    tx.merge_shard_manifests(
        output_uri,
        run_id="run1",
        node_count=2,
        attempt_id=first_id,
        publication_claim_etag=first_etag,
        publication_generation=first_generation,
        storage_client=storage,
    )

    second_id, second_etag, second_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="grade-loop-iteration-2",
        node_count=2,
        membership_digest="same-stable-membership",
        scheduler_fence_sequence=2,
        scheduler_fence_attempt=1,
        scheduler_launch_id="loop-2-job",
        storage_client=storage,
        nonce_factory=lambda: "b" * 32,
    )
    _write_shard(
        [_attempt_clip(0, output_uri, second_id)],
        output_uri,
        run_id="run1",
        rank=0,
        node_count=2,
        variant_total=2,
        storage_client=storage,
        attempt_id=second_id,
        scheduler_fence_sequence=2,
        scheduler_fence_attempt=1,
        scheduler_launch_id="loop-2-job",
        logical_wave_id="grade-loop-iteration-2",
        publication_generation=second_generation,
    )
    waits = 0

    def delayed_rank_one(_seconds: float) -> None:
        nonlocal waits
        waits += 1
        if waits == 1:
            # A delayed write from iteration 1 lands successfully, but only at
            # iteration 1's private key. The current join must keep waiting.
            _write_shard(
                [_attempt_clip(1, output_uri, first_id)],
                output_uri,
                run_id="run1",
                rank=1,
                node_count=2,
                variant_total=2,
                storage_client=storage,
                attempt_id=first_id,
                scheduler_fence_sequence=1,
                scheduler_fence_attempt=1,
                scheduler_launch_id="loop-1-job",
                logical_wave_id="grade-loop-iteration-1",
                publication_generation=first_generation,
            )
            current = json.loads(
                storage.objects[tx.transfer_manifest_uri_for(output_uri)]
            )
            assert current["status"] == tx.PUBLICATION_CLAIM_STATUS
            assert current["attempt_id"] == second_id
            return
        _write_shard(
            [_attempt_clip(1, output_uri, second_id)],
            output_uri,
            run_id="run1",
            rank=1,
            node_count=2,
            variant_total=2,
            storage_client=storage,
            attempt_id=second_id,
            scheduler_fence_sequence=2,
            scheduler_fence_attempt=1,
            scheduler_launch_id="loop-2-job",
            logical_wave_id="grade-loop-iteration-2",
            publication_generation=second_generation,
        )

    manifest = tx.merge_shard_manifests(
        output_uri,
        run_id="run1",
        node_count=2,
        attempt_id=second_id,
        publication_claim_etag=second_etag,
        publication_generation=second_generation,
        storage_client=storage,
        sleep=delayed_rank_one,
    )
    assert waits == 2
    assert manifest["attempt_id"] == second_id
    assert all(f"/_attempts/{second_id}/" in uri for uri in manifest["augmented_videos"])


def test_recovery_generation_fences_late_prior_finalization_and_writes() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    old_id, old_etag, old_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="same-managed-task-and-wave",
        node_count=2,
        membership_digest="same-stable-membership",
        scheduler_fence_sequence=3,
        scheduler_fence_attempt=1,
        scheduler_launch_id="failed-managed-job",
        storage_client=storage,
        nonce_factory=lambda: "c" * 32,
    )
    new_id, new_etag, new_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="same-managed-task-and-wave",
        node_count=2,
        membership_digest="same-stable-membership",
        scheduler_fence_sequence=3,
        scheduler_fence_attempt=2,
        scheduler_launch_id="npa-runtime-retry",
        storage_client=storage,
        nonce_factory=lambda: "d" * 32,
    )
    assert new_generation == old_generation + 1
    assert new_id != old_id

    # An escaped old leader arriving after the scheduler-authorized retry cannot
    # manufacture "generation + 1" and take over the canonical fence.
    with pytest.raises(RuntimeError, match="stale|duplicates"):
        tx.claim_run_publication(
            output_uri,
            run_id="run1",
            logical_wave_id="same-managed-task-and-wave",
            node_count=2,
            membership_digest="late-old-membership",
            scheduler_fence_sequence=3,
            scheduler_fence_attempt=1,
            scheduler_launch_id="late-old-managed-job",
            storage_client=storage,
            nonce_factory=lambda: "e" * 32,
        )
    current = json.loads(storage.objects[tx.transfer_manifest_uri_for(output_uri)])
    assert current["attempt_id"] == new_id

    late_old = [_attempt_clip(0, output_uri, old_id), _attempt_clip(1, output_uri, old_id)]
    _write_shard(
        [late_old[1]],
        output_uri,
        run_id="run1",
        rank=1,
        node_count=2,
        variant_total=2,
        storage_client=storage,
        attempt_id=old_id,
        scheduler_fence_sequence=3,
        scheduler_fence_attempt=1,
        scheduler_launch_id="failed-managed-job",
        logical_wave_id="same-managed-task-and-wave",
        publication_generation=old_generation,
    )
    with pytest.raises(RuntimeError, match="fence|superseded"):
        tx.write_run_manifest(
            late_old,
            output_uri,
            run_id="run1",
            node_count=2,
            attempt_id=old_id,
            publication_claim_etag=old_etag,
            publication_generation=old_generation,
            storage_client=storage,
        )

    for rank in (0, 1):
        _write_shard(
            [_attempt_clip(rank, output_uri, new_id)],
            output_uri,
            run_id="run1",
            rank=rank,
            node_count=2,
            variant_total=2,
            storage_client=storage,
            attempt_id=new_id,
            scheduler_fence_sequence=3,
            scheduler_fence_attempt=2,
            scheduler_launch_id="npa-runtime-retry",
            logical_wave_id="same-managed-task-and-wave",
            publication_generation=new_generation,
        )
    manifest = tx.merge_shard_manifests(
        output_uri,
        run_id="run1",
        node_count=2,
        attempt_id=new_id,
        publication_claim_etag=new_etag,
        publication_generation=new_generation,
        storage_client=storage,
    )
    assert manifest["status"] == tx.TRANSFER_MANIFEST_STATUS
    assert manifest["attempt_id"] == new_id


def test_superseded_leader_exits_unbounded_join_while_rank_is_missing() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    old_id, old_etag, old_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="managed-recovery-wave",
        node_count=2,
        membership_digest="old-members",
        scheduler_fence_sequence=4,
        scheduler_fence_attempt=1,
        scheduler_launch_id="old-launch",
        storage_client=storage,
        nonce_factory=lambda: "1" * 32,
    )
    _write_shard(
        [_attempt_clip(0, output_uri, old_id)],
        output_uri,
        run_id="run1",
        rank=0,
        node_count=2,
        variant_total=2,
        storage_client=storage,
        attempt_id=old_id,
        scheduler_fence_sequence=4,
        scheduler_fence_attempt=1,
        scheduler_launch_id="old-launch",
        logical_wave_id="managed-recovery-wave",
        publication_generation=old_generation,
    )
    sleeps = 0

    def supersede(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        tx.claim_run_publication(
            output_uri,
            run_id="run1",
            logical_wave_id="managed-recovery-wave",
            node_count=2,
            membership_digest="replacement-members",
            scheduler_fence_sequence=4,
            scheduler_fence_attempt=2,
            scheduler_launch_id="replacement-launch",
            storage_client=storage,
            nonce_factory=lambda: "2" * 32,
        )

    with pytest.raises(RuntimeError, match="superseded|authoritative"):
        tx.merge_shard_manifests(
            output_uri,
            run_id="run1",
            node_count=2,
            attempt_id=old_id,
            publication_claim_etag=old_etag,
            publication_generation=old_generation,
            storage_client=storage,
            sleep=supersede,
            # Deliberately no timeout: the canonical recovery fence itself must
            # evict an escaped old leader waiting for rank 1.
        )
    assert sleeps == 1


def test_publication_claim_refuses_an_unexamined_foreign_manifest() -> None:
    storage = FakeStorage()
    uri = "s3://bkt/run1/cosmos_augmented/"
    canonical = tx.transfer_manifest_uri_for(uri)
    foreign = {
        "schema": tx.TRANSFER_MANIFEST_SCHEMA,
        "mode": tx.TRANSFER_MANIFEST_MODE,
        "status": tx.TRANSFER_MANIFEST_STATUS,
        "run_id": "some-other-run",
    }
    storage.objects[canonical] = json.dumps(foreign).encode()
    storage.etags[canonical] = storage._next_etag()
    with pytest.raises(StorageError, match="unreadable"):
        tx.claim_run_publication(
            uri,
            run_id="run1",
            logical_wave_id="wave",
            node_count=2,
            membership_digest="members",
            scheduler_fence_sequence=1,
            scheduler_fence_attempt=1,
            scheduler_launch_id="job",
            storage_client=storage,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "attempt_id",
        "publication_generation",
        "logical_publication",
        "logical_wave_id",
        "membership_digest",
        "scheduler_fence_sequence",
        "scheduler_fence_attempt",
        "scheduler_launch_id",
    ],
)
def test_committed_attempt_manifest_requires_the_complete_publication_identity(
    missing: str,
) -> None:
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    document = {
        "schema": tx.TRANSFER_MANIFEST_SCHEMA,
        "mode": tx.TRANSFER_MANIFEST_MODE,
        "status": tx.TRANSFER_MANIFEST_STATUS,
        "run_id": "run1",
        "node_count": 1,
        "attempt_id": ATTEMPT,
        "publication_generation": 2,
        "logical_publication": "conditional",
        "logical_wave_id": "grade-loop-2",
        "membership_digest": "single-member",
        "scheduler_fence_sequence": 3,
        "scheduler_fence_attempt": 1,
        "scheduler_launch_id": "job-3",
        "variant_count": 1,
        "variants": [
            {
                "clip": "aug-run1-0",
                "variant_index": 0,
                "augmented_video_uri": (
                    f"{output_uri}_attempts/{ATTEMPT}/"
                    "aug-run1-0/augmented_video.mp4"
                ),
            }
        ],
    }
    document.pop(missing)

    with pytest.raises(ValueError, match="publication identity"):
        tx.validate_committed_run_manifest(document, output_uri)


def test_claim_refuses_a_malformed_scheduler_owned_single_node_manifest() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    canonical = tx.transfer_manifest_uri_for(output_uri)
    malformed = {
        "schema": tx.TRANSFER_MANIFEST_SCHEMA,
        "mode": tx.TRANSFER_MANIFEST_MODE,
        "status": tx.TRANSFER_MANIFEST_STATUS,
        "run_id": "run1",
        "node_count": 1,
        "logical_publication": "conditional",
        "publication_generation": 1,
        "variant_count": 0,
        "variants": [],
    }
    storage.objects[canonical] = json.dumps(malformed).encode()
    storage.etags[canonical] = storage._next_etag()

    with pytest.raises(StorageError, match="unexamined publication fence"):
        tx.claim_run_publication(
            output_uri,
            run_id="run1",
            logical_wave_id="loop-2",
            node_count=1,
            membership_digest="single-member",
            scheduler_fence_sequence=2,
            scheduler_fence_attempt=1,
            scheduler_launch_id="job-2",
            storage_client=storage,
        )


def test_shard_rejects_a_descriptor_outside_its_attempt_prefix() -> None:
    storage = FakeStorage()
    clip = _clip(0)
    clip["augmented_video_uri"] = (
        "s3://bkt/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4"
    )
    with pytest.raises(ValueError, match="attempt-scoped"):
        tx.write_shard_manifest(
            [clip],
            "s3://bkt/run1/cosmos_augmented/",
            run_id="run1",
            rank=0,
            node_count=2,
            variant_total=1,
            attempt_id=ATTEMPT,
            scheduler_fence_sequence=1,
            scheduler_fence_attempt=1,
            scheduler_launch_id="test-launch",
            logical_wave_id="test-wave",
            publication_generation=1,
            storage_client=storage,
        )


def test_merge_refuses_duplicate_or_missing_global_variant_indices() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=1, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )

    with pytest.raises(RuntimeError, match="cover every variant exactly once"):
        _merge_shards(
            output_uri, run_id="run1", node_count=2, storage_client=storage
        )

    partial = json.loads(storage.objects[tx.transfer_manifest_uri_for(output_uri)])
    assert partial["status"] == tx.PUBLICATION_CLAIM_STATUS


def test_merge_fails_naming_the_ranks_that_never_reported() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=3,
        variant_parallelism=1, variant_total=3, storage_client=storage,
    )

    with pytest.raises(RuntimeError, match=r"rank\(s\) \[1, 2\]"):
        _merge_shards(
            output_uri,
            run_id="run1",
            node_count=3,
            storage_client=storage,
            timeout_s=0,
            sleep=lambda _s: None,
        )

    # A partial manifest is never published: an understated fan-out would look
    # like a successful smaller run to every downstream stage.
    partial = json.loads(storage.objects[tx.transfer_manifest_uri_for(output_uri)])
    assert partial["status"] == tx.PUBLICATION_CLAIM_STATUS


def test_the_join_keeps_waiting_rather_than_timing_out_a_slow_sibling() -> None:
    """A sibling's remaining work is however long its diffusions take.

    Any default deadline short enough to be useful would fail runs that were
    about to succeed. Operators can choose an explicit observable deadline for
    a live-but-hung sibling.
    """

    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    waits: list[float] = []

    def very_late_arrival(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) < 40:
            return
        _write_shard(
            [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
            variant_parallelism=1, variant_total=2, storage_client=storage,
        )

    manifest = _merge_shards(
        output_uri,
        run_id="run1",
        node_count=2,
        storage_client=storage,
        poll_interval_s=0.01,
        sleep=very_late_arrival,
    )

    assert len(waits) == 40
    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1"]


def test_an_operator_can_ask_for_a_join_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    monkeypatch.setenv("NPA_COSMOS_SHARD_JOIN_TIMEOUT_S", "0")

    diagnostics: list[str] = []
    with pytest.raises(RuntimeError, match=r"rank\(s\) \[1\].*attempt"):
        _merge_shards(
            output_uri,
            run_id="run1",
            node_count=2,
            storage_client=storage,
            sleep=lambda _s: None,
            progress=diagnostics.append,
        )
    assert any("missing_ranks=[1]" in item for item in diagnostics)
    assert any("timeout=0s" in item for item in diagnostics)


def test_shard_join_propagates_storage_permission_failures() -> None:
    class DeniedStorage(FakeStorage):
        def read_bytes_with_etag(self, uri: str):
            if "manifest-rank-1.json" in uri:
                raise PermissionError("provider denied GetObject")
            return super().read_bytes_with_etag(uri)

    storage = DeniedStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    _write_shard(
        [_clip(0)],
        output_uri,
        run_id="run1",
        rank=0,
        node_count=2,
        variant_total=2,
        storage_client=storage,
    )
    with pytest.raises(PermissionError, match="provider denied"):
        _merge_shards(
            output_uri,
            run_id="run1",
            node_count=2,
            storage_client=storage,
        )


@pytest.mark.parametrize("value", ["nope", "-1", "nan", "inf"])
def test_invalid_join_timeout_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("NPA_COSMOS_SHARD_JOIN_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="finite|non-negative|number"):
        _merge_shards(
            "s3://bkt/run1/cosmos_augmented/",
            run_id="run1",
            node_count=2,
            storage_client=FakeStorage(),
        )


def _multiply_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, storage: FakeStorage
) -> list[dict]:
    """Wire the multiply path to fake inference/publish; return rendered variants."""

    rendered: list[dict] = []
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "manifest.json").write_text(
        json.dumps(
            {
                "augmentations": [
                    {"prompt": f"scene {i}", "lighting": f"l{i}"} for i in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tx, "cosmos_transfer_available", lambda: True)
    monkeypatch.setattr(cosmos2, "_materialize_input_clip", lambda *_a, **_k: "/tmp/in.mp4")
    monkeypatch.setattr(cosmos2, "_variant_parallelism", lambda n: max(1, n))

    def fake_run(**kwargs):
        rendered.append(kwargs)
        return {
            "video_path": "/tmp/out.mp4",
            "video_bytes": 1000,
            "spec": "spec.json",
            "input_conditioned": True,
            "input_video": "/tmp/in.mp4",
            "control": "edge",
        }

    def fake_publish(transfer, output_uri, **kwargs):
        index = int(kwargs["variant_index"])
        clip = _clip(index)
        clip_name = str(clip["clip"])
        base = output_uri.rstrip("/")
        clip["augmented_video_uri"] = f"{base}/{clip_name}/augmented_video.mp4"
        clip["frames_uri"] = f"{base}/{clip_name}/"
        return clip

    monkeypatch.setattr(tx, "run_cosmos_transfer", fake_run)
    monkeypatch.setattr(tx, "publish_transfer_clip", fake_publish)
    from npa.clients.storage import StorageClient

    monkeypatch.setattr(StorageClient, "from_environment", lambda: storage)
    return rendered


def _invoke_multiply(configs_dir: Path) -> object:
    return runner.invoke(
        app,
        [
            "workbench",
            "cosmos2",
            "transfer",
            "--input-uri",
            "s3://bkt/run1/input/",
            "--output-uri",
            "s3://bkt/run1/cosmos_augmented/",
            "--run-id",
            "run1",
            "--configs-uri",
            str(configs_dir) + "/",
            "--condition-on-input",
            "--execute",
        ],
    )


def test_a_worker_renders_only_its_stride_and_publishes_a_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "1")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)
    monkeypatch.setattr(
        cosmos2,
        "_gang_identity",
        lambda **_kwargs: (
            1,
            2,
            ATTEMPT,
            "",
            1,
            {
                "logical_wave_id": "test-wave",
                "scheduler_fence_sequence": 1,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "test-launch",
            },
        ),
    )

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    # Rank 1 of 2 renders variants 1 and 3 -- not all four.
    assert [call["run_id"] for call in rendered] == ["run1-v1", "run1-v3"]
    # Its GPU pins start at 0: the device index is node-local, not the global one.
    assert sorted(call["cuda_visible_devices"] for call in rendered) == ["0", "1"]
    shard = json.loads(
        storage.objects[
            "s3://bkt/run1/cosmos_augmented/_attempts/"
            "wave-attempt-1/manifest-rank-1.json"
        ]
    )
    assert shard["schema"] == tx.SHARD_MANIFEST_SCHEMA
    assert shard["clips"] == ["aug-run1-1", "aug-run1-3"]
    assert shard["variant_total"] == 4
    # A worker must not publish the run manifest; rank 0 owns that key.
    assert "s3://bkt/run1/cosmos_augmented/manifest.json" not in storage.objects
    payload = json.loads(result.output)
    assert payload["node_rank"] == 1
    assert payload["node_count"] == 2
    assert payload["shard_variant_count"] == 2
    assert payload["augmentation_variables"] == [
        {"lighting": "l1", "prompt": "scene 1"},
        {"lighting": "l3", "prompt": "scene 3"},
    ]
    assert payload["prompts"] == ["scene 1", "scene 3"]
    assert payload["prompt"] == "scene 1"


def test_rank_zero_merges_the_gang_into_one_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    # The other node already finished and left its shard behind.
    _write_shard(
        [_clip(1), _clip(3)],
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        rank=1,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )
    claim_etag = _seed_claim(
        storage, "s3://bkt/run1/cosmos_augmented/", node_count=2
    )
    monkeypatch.setattr(
        cosmos2,
        "_gang_identity",
        lambda **_kwargs: (
            0,
            2,
            ATTEMPT,
            claim_etag,
            1,
            {
                "logical_wave_id": "test-wave",
                "scheduler_fence_sequence": 1,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "test-launch",
            },
        ),
    )
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "0")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    assert [call["run_id"] for call in rendered] == ["run1-v0", "run1-v2"]
    manifest = json.loads(
        storage.objects["s3://bkt/run1/cosmos_augmented/manifest.json"]
    )
    assert manifest["clips"] == [
        "aug-run1-0",
        "aug-run1-1",
        "aug-run1-2",
        "aug-run1-3",
    ]
    assert manifest["variant_count"] == 4
    assert manifest["node_count"] == 2
    payload = json.loads(result.output)
    assert payload["variant_count"] == 4
    assert payload["shard_variant_count"] == 2


def test_an_explicit_local_surplus_rank_reports_an_empty_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "5")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "4")
    monkeypatch.setenv("NPA_COSMOS_ATTEMPT_ID", ATTEMPT)
    monkeypatch.setattr(
        cosmos2,
        "_gang_identity",
        lambda **_kwargs: (
            4,
            5,
            ATTEMPT,
            "",
            1,
            {
                "logical_wave_id": "test-wave",
                "scheduler_fence_sequence": 1,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "test-launch",
            },
        ),
    )

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    assert rendered == []
    payload = json.loads(result.output)
    assert payload["shard_variant_count"] == 0
    assert payload["augmentation_variables"] == []
    assert payload["prompts"] == []
    assert payload["prompt"] == ""


def test_single_node_augment_writes_no_shard_and_keeps_todays_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path must be byte-for-byte the same artifact set as before."""

    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    for name in (
        "NPA_COSMOS_NODE_COUNT",
        "NPA_COSMOS_NODE_RANK",
        "NPA_COSMOS_ATTEMPT_ID",
        "SKYPILOT_NUM_NODES",
        "SKYPILOT_NODE_RANK",
    ):
        monkeypatch.delenv(name, raising=False)

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    assert len(rendered) == 4
    assert not [uri for uri in storage.objects if "manifest-rank-" in uri]
    manifest = json.loads(
        storage.objects["s3://bkt/run1/cosmos_augmented/manifest.json"]
    )
    assert manifest["variant_count"] == 4
    assert manifest["node_count"] == 1
    assert "shards" not in manifest


def test_scheduler_single_node_uses_attempt_prefix_and_conditional_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FakeStorage()
    _multiply_cli(monkeypatch, tmp_path, storage)
    claim_etag = _seed_claim(
        storage,
        "s3://bkt/run1/cosmos_augmented/",
        node_count=1,
    )
    monkeypatch.setattr(
        cosmos2,
        "_gang_identity",
        lambda **_kwargs: (
            0,
            1,
            ATTEMPT,
            claim_etag,
            1,
            {
                "logical_wave_id": "test-wave",
                "scheduler_fence_sequence": 1,
                "scheduler_fence_attempt": 1,
                "scheduler_launch_id": "test-launch",
            },
        ),
    )

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        storage.objects["s3://bkt/run1/cosmos_augmented/manifest.json"]
    )
    assert manifest["attempt_id"] == ATTEMPT
    assert manifest["scheduler_fence_sequence"] == 1
    assert all(
        uri.startswith(
            "s3://bkt/run1/cosmos_augmented/_attempts/wave-attempt-1/"
        )
        for uri in manifest["augmented_videos"]
    )


def test_late_single_node_finalization_is_fenced_by_the_next_loop() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    old_id, old_etag, old_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="loop-1",
        node_count=1,
        membership_digest="single-old",
        scheduler_fence_sequence=1,
        scheduler_fence_attempt=1,
        scheduler_launch_id="old-launch",
        storage_client=storage,
        nonce_factory=lambda: "old-single-node-nonce",
    )
    new_id, _new_etag, _new_generation = tx.claim_run_publication(
        output_uri,
        run_id="run1",
        logical_wave_id="loop-2",
        node_count=1,
        membership_digest="single-new",
        scheduler_fence_sequence=2,
        scheduler_fence_attempt=1,
        scheduler_launch_id="new-launch",
        storage_client=storage,
        nonce_factory=lambda: "new-single-node-nonce",
    )

    with pytest.raises(RuntimeError, match="superseded|fence"):
        tx.write_run_manifest(
            [_attempt_clip(0, output_uri, old_id)],
            output_uri,
            run_id="run1",
            storage_client=storage,
            node_count=1,
            attempt_id=old_id,
            publication_claim_etag=old_etag,
            publication_generation=old_generation,
        )
    canonical = json.loads(storage.objects[tx.transfer_manifest_uri_for(output_uri)])
    assert canonical["attempt_id"] == new_id
    assert canonical["status"] == tx.PUBLICATION_CLAIM_STATUS
