"""Contract tests for the private LingBot-VA LIBERO-Long workflow."""

from __future__ import annotations

import json
from itertools import accumulate
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec, validate_spec
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    build_skypilot_task_docs,
)
from npa.solutions import lingbot_va as L


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/lingbot-va-libero-long.yaml"
DOCKERFILE = ROOT / "npa/docker/workbench/lingbot-va/Dockerfile"


class _PinnedLeRobotTensor:
    """Small CPU-only carrier for the pinned LeRobot index function's output."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return list(self._values)


class _PinnedLeRobotTorch:
    """The only ``torch`` surface used by ``get_episode_data_index``."""

    @staticmethod
    def LongTensor(values: list[int]) -> _PinnedLeRobotTensor:
        return _PinnedLeRobotTensor(values)


torch = _PinnedLeRobotTorch()


def _pinned_v033_get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, _PinnedLeRobotTensor]:
    """Execute the exact v0.3.3 cumulative-index body against fixture records.

    This is the body of ``lerobot.datasets.utils.get_episode_data_index`` at
    Hugging Face LeRobot ``b883328e6c95681ca90a18b102e4ae5e1f91e2bf`` (Apache-2.0),
    with only the local function name and test-only tensor annotation changed.
    The production image still installs and invokes LeRobot itself.  Keeping the
    body here makes the native positional-index contract deterministic without
    requiring a CUDA Torch installation in the repository test environment.
    """
    episode_lengths = {
        ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()
    }
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def _raw_dataset(root: Path) -> Path:
    """Create sparse raw v2.1 episode ids like the real interleaved source."""
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "fps": 10,
        "features": {
            "observation.images.image": {"dtype": "video"},
            "observation.images.image2": {"dtype": "video"},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info) + "\n")
    tasks = [
        {"task_index": task_id, "task": L.LIBERO_LONG_TASKS[task_id]}
        if task_id < 10
        else {"task_index": task_id, "task": f"short task {task_id}"}
        for task_id in range(40)
    ]
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(task) + "\n" for task in tasks)
    )
    episodes: list[dict[str, object]] = []
    stats: list[dict[str, object]] = []
    source_index = 0
    for task_id in range(10):
        for _ in range(2):
            record = {
                "episode_index": source_index,
                "length": 4,
                "tasks": [tasks[task_id]["task"]],
            }
            episodes.append(record)
            stats.append({"episode_index": source_index, "stats": {}})
            source = root / f"data/chunk-000/episode_{source_index:06d}.parquet"
            source.parent.mkdir(parents=True, exist_ok=True)
            actions = [
                [float(frame + channel + source_index) for channel in range(7)]
                for frame in range(4)
            ]
            pq.write_table(
                pa.table(
                    {
                        "action": pa.array(actions, type=pa.list_(pa.float32(), 7)),
                        "episode_index": pa.array([source_index] * 4, type=pa.int64()),
                        "index": pa.array(
                            list(range(source_index * 4, source_index * 4 + 4)),
                            type=pa.int64(),
                        ),
                    }
                ),
                source,
            )
            source_index += 13
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episodes)
    )
    (root / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in stats)
    )
    (root / "README.md").write_text("---\nlicense: cc-by-4.0\n---\n")
    (root / L.RAW_SOURCE_MANIFEST).write_text(
        json.dumps(
            {
                "complete": True,
                "dataset_id": L.RAW_DATASET_ID,
                "dataset_revision": L.RAW_DATASET_REF,
                "dataset_license": L.RAW_DATASET_LICENSE,
                "dataset_format": L.RAW_DATASET_FORMAT,
                "selected_task_ids": list(L.LIBERO_LONG_TASK_IDS),
                "files": {"meta/info.json": {"sha256": "test-only"}},
            }
        )
        + "\n"
    )
    return root


def _fake_native_latents(
    dataset: Path, records: list[dict[str, object]], chunk_size: int, _: Path
) -> None:
    """Unit-test seam: production preparation always calls the native GPU encoder."""
    (dataset / "empty_emb.pt").write_bytes(b"test-only embedding")
    for record in records:
        index = int(record["episode_index"])
        length = int(record["length"])
        for camera in L.CAMERAS:
            video = dataset / f"videos/chunk-000/{camera}/episode_{index:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"test-only derived video:{camera}:{index}".encode())
            latent = (
                dataset
                / f"latents/chunk-000/{camera}/episode_{index:06d}_0_{length}.pth"
            )
            latent.parent.mkdir(parents=True, exist_ok=True)
            latent.write_bytes(f"test-only native latent:{camera}:{index}".encode())


def test_prepare_selects_cc_by_raw_long_data_and_reindexes_sparse_episodes(
    tmp_path: Path, monkeypatch
) -> None:
    source = _raw_dataset(tmp_path / "source")
    prepared_dataset = tmp_path / "prepared-dataset"
    destination = tmp_path / "prepared.json"
    monkeypatch.setattr(L, "_snapshot_base_checkpoint", lambda path: path)
    monkeypatch.setattr(L, "_materialize_native_latents", _fake_native_latents)

    result = L.prepare(
        f"file://{source}", f"file://{prepared_dataset}", f"file://{destination}"
    )

    persisted = json.loads(destination.read_text())
    assert result["prepared_manifest_uri"] == f"file://{destination}"
    assert persisted["source_dataset_id"] == "HuggingFaceVLA/libero"
    assert persisted["source_dataset_revision"] == L.RAW_DATASET_REF
    assert persisted["source_dataset_license"] == "CC-BY-4.0"
    assert len(persisted["source_staging_manifest_sha256"]) == 64
    assert persisted["selected_libero_long_task_ids"] == list(range(10))
    assert persisted["attention"] == {"training": "flex", "inference": "torch"}
    assert L.LIBERO_BENCHMARK == "libero_10"
    assert persisted["action_contract"]["used_action_channel_ids"] == list(range(7))
    assert persisted["action_contract"]["q01"][7:] == [0.0] * 23
    assert persisted["action_contract"]["q99"][7:] == [0.0] * 23
    assert set(persisted["train_episode_indices"]).isdisjoint(
        persisted["heldout_episode_indices"]
    )
    assert sorted(
        entry["prepared_episode_index"] for entry in persisted["source_episode_mapping"]
    ) == list(range(20))
    assert [
        entry["source_episode_index"] for entry in persisted["source_episode_mapping"]
    ] == list(range(0, 260, 13))
    copied = pq.read_table(prepared_dataset / "data/chunk-000/episode_000001.parquet")
    assert copied.column("episode_index").to_pylist() == [1] * 4
    assert copied.column("index").to_pylist() == [4, 5, 6, 7]
    assert (prepared_dataset / "videos/chunk-000" / L.CAMERAS[0]).is_dir()


def test_training_subset_physically_reindexes_parquet_videos_and_latents(
    tmp_path: Path, monkeypatch
) -> None:
    source = _raw_dataset(tmp_path / "source")
    prepared_dataset = tmp_path / "prepared-dataset"
    destination = tmp_path / "prepared.json"
    monkeypatch.setattr(L, "_snapshot_base_checkpoint", lambda path: path)
    monkeypatch.setattr(L, "_materialize_native_latents", _fake_native_latents)
    prepared = L.prepare(
        f"file://{source}", f"file://{prepared_dataset}", f"file://{destination}"
    )

    L._materialize_training_subset(
        prepared_dataset, list(prepared["train_episode_indices"])
    )

    records = L._episode_records(prepared_dataset)
    assert [record["episode_index"] for record in records] == list(range(10))
    assert [record["prepared_episode_index"] for record in records] == list(
        prepared["train_episode_indices"]
    )
    assert not {record["prepared_episode_index"] for record in records} & set(
        prepared["heldout_episode_indices"]
    )
    info = json.loads((prepared_dataset / "meta/info.json").read_text())
    assert info["total_episodes"] == 10
    assert info["total_frames"] == 40
    # This executes the exact pinned LeRobot 0.3.3 cumulative-index behavior
    # that LingBot's native loader uses.  The first selected record after the
    # excluded episode and the final selected record must resolve contiguous
    # frame offsets, rather than their original sparse prepared episode IDs.
    native_metas = {record["episode_index"]: record for record in records}
    native_index = _pinned_v033_get_episode_data_index(native_metas)
    assert native_index["from"].tolist() == list(range(0, 40, 4))
    assert records[1]["prepared_episode_index"] == 2
    assert native_index["from"].tolist()[records[1]["episode_index"]] == 4
    assert records[-1]["prepared_episode_index"] == 18
    assert native_index["from"].tolist()[records[-1]["episode_index"]] == 36
    assert {
        record["prepared_episode_index"] for record in native_metas.values()
    }.isdisjoint(prepared["heldout_episode_indices"])
    assert all(
        record["action_config"][0]["action_text"] == record["tasks"][0]
        for record in native_metas.values()
    )
    for native_index, record in enumerate(records):
        prepared_index = record["prepared_episode_index"]
        assert record["episode_index"] == native_index
        assert record["action_config"] == [
            {
                "start_frame": 0,
                "end_frame": 4,
                "action_text": record["tasks"][0],
            }
        ]
        table = pq.read_table(
            prepared_dataset / f"data/chunk-000/episode_{native_index:06d}.parquet"
        )
        # These are the exact rows the pinned LeRobot cumulative frame index
        # resolves for an interleaved train/held-out source: a selected record
        # after each omitted episode and the final selected record must retain
        # its source action values while gaining contiguous native offsets.
        assert table.column("episode_index").to_pylist() == [native_index] * 4
        assert table.column("index").to_pylist() == list(
            range(native_index * 4, native_index * 4 + 4)
        )
        assert (
            table.column("index").to_pylist()[0]
            == _pinned_v033_get_episode_data_index(native_metas)["from"].tolist()[
                native_index
            ]
        )
        assert table.column("action").to_pylist() == [
            [
                float(frame + channel + record["source_episode_index"])
                for channel in range(7)
            ]
            for frame in range(4)
        ]
        for camera in L.CAMERAS:
            video = (
                prepared_dataset
                / f"videos/chunk-000/{camera}/episode_{native_index:06d}.mp4"
            )
            latent = (
                prepared_dataset
                / f"latents/chunk-000/{camera}/episode_{native_index:06d}_0_4.pth"
            )
            assert video.read_bytes() == (
                f"test-only derived video:{camera}:{prepared_index}".encode()
            )
            assert latent.read_bytes() == (
                f"test-only native latent:{camera}:{prepared_index}".encode()
            )
    assert not list(prepared_dataset.glob(".*-before-training-subset"))


def test_workflow_is_five_connected_native_stages_with_exact_artifact_handoffs() -> (
    None
):
    spec = load_spec(WORKFLOW)
    validate_spec(spec)
    plan = build_plan(spec, run_id="lingbot-va-contract")
    payload = yaml.safe_load(WORKFLOW.read_text())

    assert list(spec.states) == [
        "prepare",
        "posttrain",
        "rollout",
        "evaluate",
        "visualize",
    ]
    assert len(plan.steps) == 5
    assert payload["resources"]["train"]["accelerators"] == "RTXPRO6000:8"
    assert payload["resources"]["gpu"]["accelerators"] == "RTXPRO6000:1"
    assert (
        "REPLACE_WITH_VERIFIED_CANDIDATE_DIGEST" in payload["config"]["candidate_image"]
    )
    prepare = payload["states"]["prepare"]
    assert prepare["run"]["argv"][-2:] == [
        "--output-uri",
        "{{config.prepared_uri}}",
    ]
    assert "--prepared-dataset-uri" in prepare["run"]["argv"]
    assert (
        prepare["outputs"][0]["schema"]
        == "npa.lingbot_va.lerobot_wan_latent_dataset.v1"
    )
    for state in payload["states"].values():
        argv = state["run"]["argv"]
        assert argv[:5] == [
            "lingbot-va-runtime",
            "exec",
            "python",
            "-m",
            "npa.solutions.lingbot_va",
        ]
        assert "echo" not in argv
        assert "sleep" not in argv
    assert (
        payload["states"]["posttrain"]["inputs"][0]["uri"]
        == payload["states"]["prepare"]["outputs"][1]["uri"]
    )
    assert (
        payload["states"]["rollout"]["inputs"][1]["uri"]
        == payload["states"]["posttrain"]["outputs"][0]["uri"]
    )
    assert (
        payload["states"]["evaluate"]["inputs"][0]["uri"]
        == payload["states"]["rollout"]["outputs"][1]["uri"]
    )
    assert (
        payload["states"]["visualize"]["inputs"][2]["uri"]
        == payload["states"]["evaluate"]["outputs"][0]["uri"]
    )


def test_pinned_candidate_receives_staged_npa_source_for_worker_bootstrap(
    monkeypatch,
) -> None:
    """A source-only candidate still needs the generic NPA worker bootstrap.

    The image deliberately carries only the LingBot runtime adapter, while the
    SkyPilot setup shell provides the generic ``npa`` command from the exact
    operator-staged checkout.  Without this environment handoff, setup stops
    before the first genuine prepare stage with ``npa CLI not found``.
    """
    spec = load_spec(WORKFLOW)
    source_uri = "s3://fixture-bucket/npa-src/npa/" + "a" * 64
    candidate_image = (
        "registry.example.invalid/operator/npa-lingbot-va@sha256:" + "b" * 64
    )
    spec.config["candidate_image"] = candidate_image
    monkeypatch.setenv("NPA_SRC_S3_URI", source_uri)
    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)

    plan = build_plan(spec, run_id="lingbot-va-staged-source")
    docs = build_skypilot_task_docs(
        spec,
        plan.steps,
        run_id="lingbot-va-staged-source",
        options=SkypilotRenderOptions(
            registry="registry.example.invalid/operator/validated"
        ),
    )

    assert len(docs) == 5
    for doc in docs:
        assert doc["resources"]["image_id"] == f"docker:{candidate_image}"
        assert doc["envs"]["NPA_SRC_S3_URI"] == source_uri
        assert "NPA_SRC_OVERLAY" not in doc["envs"]
        assert "npa CLI not found" in doc["setup"]


def test_source_only_image_pins_the_distinct_cuda_contract_without_extra_acceptance() -> (
    None
):
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    requirements = DOCKERFILE.parent.joinpath("runtime-requirements.txt").read_text(
        encoding="utf-8"
    )
    runtime_script = DOCKERFILE.parent.joinpath("lingbot_va_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert L.SOURCE_REF in dockerfile
    assert L.POSTTRAIN_MODEL_REF in dockerfile
    assert "torch==2.9.1+cu126" in requirements
    assert "torchvision==0.24.1+cu126" in requirements
    assert "torchaudio==2.9.1+cu126" in requirements
    assert "diffusers==0.38.0" in requirements
    assert "transformers==5.10.0" in requirements
    assert "accelerate==1.14.0" in requirements
    assert "Pillow==12.3.0" in requirements
    assert "datasets==5.0.1" in requirements
    assert "deepdiff==8.6.2" in requirements
    assert "Flask==3.1.3" in requirements
    assert "pyarrow==25.0.1" in requirements
    assert "lerobot==0.3.3" in runtime_script
    assert "--no-deps" in runtime_script
    assert "torch==2.9.1+cu126" in runtime_script
    assert "ACCEPT_" not in dockerfile
    assert "robbyant/libero-long-lerobot" not in dockerfile
    assert "npa-wan2-2@sha256:" in dockerfile
    assert "install -d -m 0755 /opt/npa-native/npa/solutions" in dockerfile
    assert "/usr/share/doc/npa-lingbot-va" in dockerfile


def test_visualization_binds_the_recording_to_the_workflow_run() -> None:
    """Require the factual RRD to retain its exact workflow run identity."""
    source = Path(L.__file__).read_text(encoding="utf-8")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert "NPA_WORKFLOW_RUN_ID" in source
    assert "recording_id=run_id" in source
    assert workflow["states"]["visualize"]["outputs"][0]["schema"] == (
        "application/vnd.rerun.rrd"
    )
