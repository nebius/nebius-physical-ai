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
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import cosmos2
from npa.workbench.cosmos import transfer as tx

runner = CliRunner()


class FakeStorage:
    """In-memory stand-in for the S3 objects a gang's nodes exchange."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, local: str, uri: str) -> str:
        self.objects[uri] = Path(local).read_bytes()
        return uri

    def download_path(self, uri: str, local_path: str) -> str:
        if uri not in self.objects:
            return local_path
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[uri])
        return str(dest)


def _clip(index: int, *, run_id: str = "run1") -> dict:
    return {
        "clip": f"aug-{run_id}-{index}",
        "variant_index": index,
        "augmented_video_uri": f"s3://bkt/{run_id}/cosmos_augmented/aug-{run_id}-{index}/augmented_video.mp4",
        "frames_uri": f"s3://bkt/{run_id}/cosmos_augmented/aug-{run_id}-{index}/",
        "frames": [{"frame_id": "frame-00000", "uri": "s3://bkt/f.png"}],
        "frame_count": 1,
        "variables": {"prompt": f"scene {index}"},
        "video_bytes": 1000,
        "input_conditioned": True,
        "control": "edge",
    }


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

    monkeypatch.setenv("SKYPILOT_NUM_NODES", "4")
    monkeypatch.setenv("SKYPILOT_NODE_RANK", "2")
    assert cosmos2._gang_shard() == (2, 4)

    # An NPA override wins, so the shard is reproducible off-cluster.
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "1")
    assert cosmos2._gang_shard() == (1, 2)


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
    tx.write_shard_manifest(
        [_clip(1), _clip(3)],
        output_uri,
        run_id="run1",
        rank=1,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )
    tx.write_shard_manifest(
        [_clip(0), _clip(2)],
        output_uri,
        run_id="run1",
        rank=0,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )

    manifest = tx.merge_shard_manifests(
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
        tx.shard_manifest_uri_for(output_uri, 1)
        == "s3://bkt/run1/cosmos_augmented/manifest-rank-1.json"
    )


def test_shard_manifests_are_files_not_a_subdirectory_of_the_augment_prefix() -> None:
    """Every consumer counts a subdir of the augment prefix as a scenario variant."""

    uri = tx.shard_manifest_uri_for("s3://bkt/run1/cosmos_augmented/", 3)
    relative = uri.split("cosmos_augmented/", 1)[1]
    assert "/" not in relative

    from npa.workflows import data_factory_stages as dfs

    keys = [
        "physical-ai-data-factory/run1/cosmos_augmented/manifest.json",
        "physical-ai-data-factory/run1/cosmos_augmented/manifest-rank-0.json",
        "physical-ai-data-factory/run1/cosmos_augmented/manifest-rank-1.json",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-1/augmented_video.mp4",
    ]
    _, prefix = dfs._split(
        "s3://bkt/physical-ai-data-factory/run1/cosmos_augmented/"
    )
    rels = [k[len(prefix):] for k in keys if k.startswith(prefix)]
    clips = sorted({r.split("/", 1)[0] for r in rels if "/" in r})
    assert clips == ["aug-run1-0", "aug-run1-1"]


def test_merge_waits_for_a_slow_rank_then_joins() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    tx.write_shard_manifest(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=2,
        variant_parallelism=1, variant_total=2, storage_client=storage,
    )
    waits: list[float] = []

    def late_arrival(seconds: float) -> None:
        waits.append(seconds)
        tx.write_shard_manifest(
            [_clip(1)], output_uri, run_id="run1", rank=1, node_count=2,
            variant_parallelism=1, variant_total=2, storage_client=storage,
        )

    manifest = tx.merge_shard_manifests(
        output_uri,
        run_id="run1",
        node_count=2,
        storage_client=storage,
        poll_interval_s=0.5,
        sleep=late_arrival,
    )

    assert waits == [0.5]
    assert manifest["clips"] == ["aug-run1-0", "aug-run1-1"]


def test_merge_fails_naming_the_ranks_that_never_reported() -> None:
    storage = FakeStorage()
    output_uri = "s3://bkt/run1/cosmos_augmented/"
    tx.write_shard_manifest(
        [_clip(0)], output_uri, run_id="run1", rank=0, node_count=3,
        variant_parallelism=1, variant_total=3, storage_client=storage,
    )

    with pytest.raises(RuntimeError, match=r"rank\(s\) \[1, 2\]"):
        tx.merge_shard_manifests(
            output_uri,
            run_id="run1",
            node_count=3,
            storage_client=storage,
            timeout_s=0,
            sleep=lambda _s: None,
        )

    # A partial manifest is never published: an understated fan-out would look
    # like a successful smaller run to every downstream stage.
    assert tx.transfer_manifest_uri_for(output_uri) not in storage.objects


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
        return _clip(index)

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

    result = _invoke_multiply(tmp_path / "configs")

    assert result.exit_code == 0, result.output
    # Rank 1 of 2 renders variants 1 and 3 -- not all four.
    assert [call["run_id"] for call in rendered] == ["run1-v1", "run1-v3"]
    # Its GPU pins start at 0: the device index is node-local, not the global one.
    assert sorted(call["cuda_visible_devices"] for call in rendered) == ["0", "1"]
    shard = json.loads(
        storage.objects["s3://bkt/run1/cosmos_augmented/manifest-rank-1.json"]
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


def test_rank_zero_merges_the_gang_into_one_run_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    # The other node already finished and left its shard behind.
    tx.write_shard_manifest(
        [_clip(1), _clip(3)],
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        rank=1,
        node_count=2,
        variant_parallelism=2,
        variant_total=4,
        storage_client=storage,
    )
    monkeypatch.setenv("NPA_COSMOS_NODE_COUNT", "2")
    monkeypatch.setenv("NPA_COSMOS_NODE_RANK", "0")

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


def test_single_node_augment_writes_no_shard_and_keeps_todays_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default path must be byte-for-byte the same artifact set as before."""

    storage = FakeStorage()
    rendered = _multiply_cli(monkeypatch, tmp_path, storage)
    for name in ("NPA_COSMOS_NODE_COUNT", "NPA_COSMOS_NODE_RANK", "SKYPILOT_NUM_NODES", "SKYPILOT_NODE_RANK"):
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
