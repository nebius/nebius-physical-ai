"""Unit tests for the real Physical AI Data Factory stage functions."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import typer

from npa.workflows import data_factory_stages as dfs


def _stub_real_fiftyone(
    report: dict, augment_uri: str, keys: list[str], dedup_threshold: float | str
) -> dict:
    del augment_uri, keys, dedup_threshold
    return {
        **report,
        "curation_engine": "fiftyone-brain",
        "curated_kept": report["augmented_clips"],
        "curated_dropped": 0,
    }


def _completed_curator_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "engine": "cosmos-curator-stages",
                "curated_uri": "s3://b/p/curated/",
                "clip_count": 1,
            }
        )
    )
    return path


def _mock_committed_manifest(
    monkeypatch: pytest.MonkeyPatch, keys: list[str], *, bucket: str = "b"
) -> None:
    """Make listed canonical test objects carry the real committed contract."""

    original = dfs._download_json
    def load(uri: str):
        if "cosmos_augmented/" in uri and uri.endswith("/manifest.json"):
            object_key = uri.split(f"s3://{bucket}/", 1)[-1]
            manifest_prefix = object_key.rsplit("/", 1)[0] + "/"
            videos = sorted(
                key
                for key in keys
                if key.startswith(manifest_prefix)
                and key.endswith("/augmented_video.mp4")
            )
            variants = [
                {
                    "clip": key.rsplit("/", 2)[-2],
                    "augmented_video_uri": f"s3://{bucket}/{key}",
                }
                for key in videos
            ]
            return {
                "schema": "npa.cosmos2.transfer.v1",
                "mode": "cosmos_transfer2.5_gpu",
                "status": "executed",
                "node_count": 1,
                "variant_count": len(variants),
                "variants": variants,
            }
        return original(uri)

    monkeypatch.setattr(dfs, "_download_json", load)


def test_attempt_keys_without_canonical_manifest_fail_closed() -> None:
    keys = ["run/cosmos_augmented/_attempts/orphan/clip/augmented_video.mp4"]
    with pytest.raises(RuntimeError, match="without a canonical manifest"):
        dfs._committed_augment_manifest(
            "s3://b/run/cosmos_augmented/", listed_keys=keys
        )


def test_generate_configs_writes_real_manifest(tmp_path: Path) -> None:
    out = tmp_path / "configs" / "manifest.json"
    result = dfs.generate_configs(str(out), n_augmentations=3, seed="run-x")
    assert result["n_augmentations"] == 3
    assert len(result["augmentations"]) == 3
    for combo in result["augmentations"]:
        # Appearance vars work for a replaceable physical-scene input.
        assert combo["color_grade"] in dfs.APPEARANCE_VARIABLES["color_grade"]
        assert combo["lighting"] in dfs.APPEARANCE_VARIABLES["lighting"]
        # Each combo carries the prompt that actually conditions the augmentation.
        assert combo["color_grade"] in combo["prompt"]
        assert "input-conditioned" in combo["prompt"]
    assert out.is_file()
    assert json.loads(out.read_text())["schema"] == "npa.data_factory.configs.v1"


def test_generate_configs_propagates_augmentation_subject_into_real_prompts(
    tmp_path: Path,
) -> None:
    result = dfs.generate_configs(
        str(tmp_path / "subject") + "/",
        n_augmentations=2,
        seed="run-subject",
        augment_subject="warehouse picking robot clips",
    )
    assert result["scene"] == "warehouse picking robot clips"
    assert all(
        "warehouse picking robot clips" in item["prompt"]
        for item in result["augmentations"]
    )


def test_prompt_from_combo_is_appearance_only() -> None:
    combo = {
        "color_grade": "warm",
        "surface_finish": "matte",
        "lighting": "dim evening light",
        "background": "plain wall",
    }
    prompt = dfs.prompt_from_combo(combo)
    assert "warm" in prompt
    assert "matte" in prompt
    assert "non-identity-bearing backdrop" in prompt
    assert "Preserve the exact foreground objects" in prompt
    assert "cloth" not in prompt


def test_generate_configs_is_deterministic_by_seed(tmp_path: Path) -> None:
    a = dfs.generate_configs(str(tmp_path / "a.json"), n_augmentations=2, seed="s")
    b = dfs.generate_configs(str(tmp_path / "b.json"), n_augmentations=2, seed="s")
    assert a["augmentations"] == b["augmentations"]


def test_generate_configs_fans_out_coherent_profiles_and_distinct_seeds(
    tmp_path: Path,
) -> None:
    result = dfs.generate_configs(
        str(tmp_path / "profiles.json"),
        n_augmentations=4,
        seed="quality-search",
    )

    candidates = result["augmentations"]
    assert len(
        {
            tuple(candidate[key] for key in dfs.APPEARANCE_VARIABLES)
            for candidate in candidates
        }
    ) == 4
    assert len({candidate["inference_seed"] for candidate in candidates}) == 4
    assert all(
        0 <= candidate["inference_seed"] < 2**31 for candidate in candidates
    )


def test_generate_configs_supports_a_shared_controlled_comparison_seed(
    tmp_path: Path,
) -> None:
    baseline = dfs.generate_configs(
        str(tmp_path / "baseline.json"),
        n_augmentations=3,
        seed="baseline-run",
        augmentation_seed="controlled-comparison-v1",
    )
    component = dfs.generate_configs(
        str(tmp_path / "component.json"),
        n_augmentations=3,
        seed="component-run",
        augmentation_seed="controlled-comparison-v1",
    )

    assert baseline["augmentations"] == component["augmentations"]
    assert baseline["augmentation_seed"] == "controlled-comparison-v1"


def test_generate_configs_derives_first_search_candidate_from_passing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "status": "completed",
        "augment_uri": "s3://b/prior/cosmos_augmented/iteration-1/",
        "clips": [
            {
                "clip_id": "prior-best",
                "score": 0.91,
                "passed": True,
                "attribute_verification": {
                    "passed": True,
                    "passed_checks": 4,
                    "total_checks": 4,
                },
                "hallucination": {"passed": True},
            }
        ],
    }
    variables = {
        "color_grade": "neutral balanced color",
        "surface_finish": "natural low-gloss materials",
        "lighting": "soft diffuse daylight",
        "background": "stable low-detail surroundings",
    }
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda _uri: ["prior/grade/iteration-1/ranking/cosmos_evaluator.json"],
    )
    monkeypatch.setattr(dfs, "_read_json_key", lambda _bucket, _key: report)
    monkeypatch.setattr(
        dfs,
        "_committed_augment_manifest",
        lambda _uri: {
            "variants": [
                {
                    "clip": "prior-best",
                    "augmented_video_uri": (
                        "s3://b/prior/cosmos_augmented/iteration-1/"
                        "prior-best/augmented_video.mp4"
                    ),
                }
            ]
        },
    )
    monkeypatch.setattr(
        dfs,
        "_download_json",
        lambda _uri: {
            "variables": variables,
            "inference_seed": 42,
            "effective_control_weight": 0.9,
            "effective_guidance": 2.0,
        },
    )

    manifest = dfs.generate_configs(
        str(tmp_path / "configs.json"),
        n_augmentations=8,
        seed="new-run",
        quality_anchor_uri="s3://b/prior/",
    )

    assert manifest["n_augmentations"] == 8
    assert manifest["augmentations"][0]["inference_seed"] == 42
    assert {
        key: manifest["augmentations"][0][key] for key in dfs.APPEARANCE_VARIABLES
    } == variables
    assert manifest["quality_anchor"]["clip_id"] == "prior-best"
    assert manifest["quality_anchor"]["score"] == 0.91


@pytest.mark.parametrize(
    ("passing", "expected_selected"),
    [(True, ["candidate-a"]), (False, [])],
    ids=["one-independent-pass", "zero-selection-fails-closed"],
)
def test_candidate_selection_is_additive_and_preserves_complete_ranking_pool(
    monkeypatch: pytest.MonkeyPatch,
    passing: bool,
    expected_selected: list[str],
) -> None:
    augment_uri = "s3://b/run/cosmos_augmented/iteration-1/"
    selection_uri = "s3://b/run/selection/iteration-1/"
    source_rows = [
        {"key": "run/cosmos_augmented/iteration-1/manifest.json", "size": 100, "etag": "m"},
        {"key": "run/cosmos_augmented/iteration-1/candidate-a/augmented_video.mp4", "size": 200, "etag": "a"},
        {"key": "run/cosmos_augmented/iteration-1/candidate-a/metadata.json", "size": 50, "etag": "am"},
        {"key": "run/cosmos_augmented/iteration-1/candidate-b/augmented_video.mp4", "size": 210, "etag": "b"},
        {"key": "run/cosmos_augmented/iteration-1/candidate-b/metadata.json", "size": 55, "etag": "bm"},
    ]
    destination_rows: list[dict[str, object]] = []
    manifest = {
        "schema": "npa.cosmos2.transfer.v1",
        "mode": "cosmos_transfer2.5_gpu",
        "status": "executed",
        "node_count": 1,
        "variant_count": 2,
        "variants": [
            {
                "clip": clip,
                "variant_index": index,
                "augmented_video_uri": (
                    f"{augment_uri}{clip}/augmented_video.mp4"
                ),
                "control_uris": {},
            }
            for index, clip in enumerate(("candidate-a", "candidate-b"))
        ],
    }
    ranking = {
        "status": "completed",
        "clips": [
            {
                "clip_id": "candidate-a",
                "status": "completed",
                "score": 0.9,
                "passed": passing,
                "input_conditioned": True,
                "attribute_verification": {
                    "passed": passing,
                    "total_checks": 4,
                    "passed_checks": 4 if passing else 3,
                },
                "hallucination": {"passed": True},
            },
            {
                "clip_id": "candidate-b",
                "status": "completed",
                "score": 0.6,
                "passed": False,
                "input_conditioned": True,
                "attribute_verification": {
                    "passed": False,
                    "total_checks": 4,
                    "passed_checks": 3,
                },
                "hallucination": {"passed": True},
            },
        ],
    }

    def inventory(uri: str) -> list[dict]:
        if uri == augment_uri:
            return [dict(row) for row in source_rows]
        if uri == selection_uri:
            return [dict(row) for row in destination_rows]
        raise AssertionError(uri)

    class FakeS3:
        def copy_object(self, *, Bucket: str, CopySource: dict, Key: str) -> None:
            assert Bucket == "b"
            source = next(row for row in source_rows if row["key"] == CopySource["Key"])
            destination_rows.append({**source, "key": Key})

    def upload(payload: dict, uri: str) -> str:
        if uri == f"{selection_uri}manifest.json":
            destination_rows.append(
                {"key": "run/selection/iteration-1/manifest.json", "size": 100, "etag": "sm"}
            )
        return uri

    monkeypatch.setattr(dfs, "_inventory_rows", inventory)
    monkeypatch.setattr(dfs, "_committed_augment_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(dfs, "_download_json", lambda _uri: ranking)
    monkeypatch.setattr(dfs, "_s3_client", lambda: FakeS3())
    monkeypatch.setattr(dfs, "_upload_json", upload)

    result = dfs.select_hard_passing_candidates(
        augment_uri,
        "s3://b/run/grade/iteration-1/ranking/",
        selection_uri,
        "s3://b/run/selection/iteration-1/selection.json",
        0.75,
    )

    assert result["selected_clip_ids"] == expected_selected
    assert result["ranking_pool_unchanged_after_selection"] is True
    assert source_rows == inventory(augment_uri)
    assert any(row["key"].endswith("manifest.json") for row in destination_rows)
    if expected_selected:
        assert any("candidate-a/augmented_video.mp4" in row["key"] for row in destination_rows)
    assert not any("candidate-b/" in row["key"] for row in destination_rows)


def test_rejected_review_fields_are_truthful_and_never_promotion_eligible() -> None:
    candidate = dfs._review_candidate_from_evaluation(
        iteration=2,
        clip="candidate-z",
        media_key="run/candidate-z/augmented_video.mp4",
        evaluation={
            "score": 0.8,
            "passed": True,
            "attribute_verification": {
                "checks": [{"variable": "lighting", "passed": True}]
            },
            "hallucination": {"passed": True},
        },
        run_disposition="rejected",
    )

    assert candidate["candidate_id"] == "iteration-2/candidate-z"
    assert candidate["candidate_passed"] is True
    assert candidate["promotion_eligible"] is False
    assert candidate["hallucination_status"] == "passed"


def test_terminal_review_preservation_ignores_only_declared_outputs_and_ledger() -> None:
    before = [
        {"key": "run/candidate.mp4", "size": 11, "etag": "source"},
        {"key": "run/npa-workflow/runtime.json", "size": 20, "etag": "old"},
    ]
    after = [
        {"key": "run/candidate.mp4", "size": 11, "etag": "source"},
        {"key": "run/npa-workflow/runtime.json", "size": 21, "etag": "new"},
        {"key": "run/review/dataset/samples.json", "size": 30, "etag": "data"},
        {"key": "run/review/report.json", "size": 40, "etag": "report"},
    ]

    preserved = dfs._assert_terminal_review_source_preserved(
        before,
        after,
        dataset_prefix="run/review/dataset/",
        report_key="run/review/report.json",
        workflow_prefix="run/npa-workflow/",
    )

    assert preserved == [before[0]]
    changed = [dict(row) for row in after]
    changed[0]["etag"] = "changed"
    with pytest.raises(RuntimeError, match="changed source inventory"):
        dfs._assert_terminal_review_source_preserved(
            before,
            changed,
            dataset_prefix="run/review/dataset/",
            report_key="run/review/report.json",
            workflow_prefix="run/npa-workflow/",
        )
    with pytest.raises(RuntimeError, match="removed workflow evidence"):
        dfs._assert_terminal_review_source_preserved(
            before,
            [row for row in after if "npa-workflow" not in row["key"]],
            dataset_prefix="run/review/dataset/",
            report_key="run/review/report.json",
            workflow_prefix="run/npa-workflow/",
        )


def test_terminal_review_archive_resumes_exact_objects_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    (archive / "data").mkdir(parents=True)
    (archive / "metadata.json").write_text('{"name":"review"}\n')
    (archive / "data" / "candidate.mp4").write_bytes(b"video-bytes")
    objects: dict[str, bytes] = {}
    put_count = 0

    class Body(io.BytesIO):
        def close(self) -> None:
            super().close()

    class FakeS3:
        def put_object(self, *, Bucket: str, Key: str, Body, **_kwargs) -> None:
            nonlocal put_count
            assert Bucket == "b"
            if Key in objects:
                raise AssertionError("publisher attempted to overwrite an object")
            objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)
            put_count += 1

        def get_object(self, *, Bucket: str, Key: str) -> dict:
            assert Bucket == "b"
            return {"Body": Body(objects[Key])}

    client = FakeS3()

    def inventory(uri: str) -> list[dict]:
        assert uri == "s3://b/run/review/dataset/"
        return [
            {"key": key, "size": len(value), "etag": f"etag-{index}"}
            for index, (key, value) in enumerate(sorted(objects.items()))
            if key.startswith("run/review/dataset/")
        ]

    monkeypatch.setattr(dfs, "_s3_client", lambda: client)
    monkeypatch.setattr(dfs, "_inventory_rows", inventory)

    uri, rows = dfs._publish_terminal_review_directory_once(
        archive, "s3://b/run/review/dataset/"
    )
    first_put_count = put_count
    resumed_uri, resumed_rows = dfs._publish_terminal_review_directory_once(
        archive, "s3://b/run/review/dataset/"
    )

    assert uri == resumed_uri == "s3://b/run/review/dataset/"
    assert rows == resumed_rows
    assert first_put_count == 2
    assert put_count == first_put_count

    (archive / "metadata.json").write_text('{"name":"different"}\n')
    with pytest.raises(RuntimeError, match="mismatched archive object"):
        dfs._publish_terminal_review_directory_once(
            archive, "s3://b/run/review/dataset/"
        )


def test_terminal_review_validates_existing_portable_dataset_semantically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "candidate_id": "iteration-1/candidate-a",
        "iteration": 1,
        "clip_id": "candidate-a",
        "media_key": "run/cosmos/candidate-a.mp4",
        "score": 0.7,
        "candidate_passed": False,
        "hard_checks_passed": False,
        "promotion_eligible": False,
        "failed_attributes": ["lighting"],
        "attribute_results": [{"attribute": "lighting", "passed": False}],
        "hallucination_status": "passed",
        "hard_check_results": {"hallucination": {"passed": True}},
    }
    prefix = "run/review/dataset/"
    objects = {
        prefix + "metadata.json": json.dumps(
            {
                "version": "1.0",
                "info": {
                    "schema": "npa.paidf.fiftyone-terminal-review/v1",
                    "dataset_name": "review-dataset",
                    "quality_disposition": "rejected",
                    "review_only": True,
                },
                "sample_fields": [{"name": "candidate_id"}],
            }
        ).encode(),
        prefix + "samples.json": json.dumps(
            {
                "samples": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "iteration": 1,
                        "clip_id": "candidate-a",
                        "filepath": "data/candidate-a.mp4",
                        "score": 0.7,
                        "candidate_passed": False,
                        "hard_checks_passed": False,
                        "promotion_eligible": False,
                        "quality_disposition": "rejected",
                        "failed_attributes": ["lighting"],
                        "hallucination_status": "passed",
                        "attribute_results_json": json.dumps(
                            candidate["attribute_results"], sort_keys=True
                        ),
                        "hard_check_results_json": json.dumps(
                            candidate["hard_check_results"], sort_keys=True
                        ),
                    }
                ]
            }
        ).encode(),
        prefix + "frames.json": b'{"frames":[]}',
        prefix + "data/candidate-a.mp4": b"same-media",
        candidate["media_key"]: b"same-media",
    }

    class Body(io.BytesIO):
        pass

    class FakeS3:
        def get_object(self, *, Bucket: str, Key: str) -> dict:
            assert Bucket == "b"
            return {"Body": Body(objects[Key])}

    monkeypatch.setattr(dfs, "_s3_client", lambda: FakeS3())
    archive_rows = [
        {"key": key, "size": len(value), "etag": key}
        for key, value in objects.items()
        if key.startswith(prefix)
    ]

    metadata = dfs._terminal_review_archive_metadata(
        dataset_uri="s3://b/run/review/dataset/",
        archive_rows=archive_rows,
        candidates=[candidate],
        dataset_name="review-dataset",
        run_disposition="rejected",
    )

    assert metadata["candidate_count"] == 1
    assert metadata["promotion_eligible_count"] == 0
    assert metadata["review_only"] is True
    objects[prefix + "data/candidate-a.mp4"] = b"different"
    with pytest.raises(RuntimeError, match="media differs"):
        dfs._terminal_review_archive_metadata(
            dataset_uri="s3://b/run/review/dataset/",
            archive_rows=archive_rows,
            candidates=[candidate],
            dataset_name="review-dataset",
            run_disposition="rejected",
        )


def test_candidate_selection_does_not_trust_a_bare_passed_summary() -> None:
    assert not dfs._independently_hard_passing_candidate(
        {"status": "completed", "score": 0.99, "passed": True}, 0.75
    )
    assert not dfs._independently_hard_passing_candidate(
        {
            "status": "completed",
            "score": 0.99,
            "passed": True,
            "input_conditioned": True,
            "attribute_verification": {
                "passed": True,
                "total_checks": 4,
                "passed_checks": 4,
            },
            "hallucination": {"passed": False},
        },
        0.75,
    )


def test_prepare_refinement_uses_baseline_then_adapts_failed_retry(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = tmp_path / "grade" / "decision.json"

    baseline = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    assert baseline["attempt"] == 0
    assert baseline["adapted_from_prior_evaluation"] is False
    assert baseline["settings"] == {"control_weight": 1.0, "guidance": 3}
    assert (tmp_path / "configs" / "refinement-attempt-00.json").is_file()
    assert (tmp_path / "configs" / "refinement-attempt-00.commit.json").is_file()

    (grade / "cosmos_evaluator.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.4,
                "passed": False,
                "clips": [
                    {
                        "passed": False,
                        "appearance_fidelity": {"passed": False},
                        "hallucination": {"passed": True},
                    }
                ],
            }
        )
    )
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "loop_back"
    )
    retry = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    assert retry["attempt"] == 1
    assert retry["adapted_from_prior_evaluation"] is True
    assert retry["settings"] == {"control_weight": 1.0, "guidance": 2}
    assert retry["failed_checks"] == ["appearance_fidelity"]
    assert (tmp_path / "configs" / "refinement-attempt-01.json").is_file()


def test_prepare_refinement_records_exact_failed_attribute_names(tmp_path: Path) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    dfs.prepare_refinement(str(grade), str(refinement), decision_uri=str(decision))
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.7,
                "passed": False,
                "clips": [
                    {
                        "passed": False,
                        "attribute_verification": {
                            "passed": False,
                            "checks": [
                                {"variable": "lighting", "passed": False},
                                {"variable": "background", "passed": True},
                            ],
                        },
                        "hallucination": {"passed": True},
                    }
                ],
            }
        )
    )
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "loop_back"
    )

    retry = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )

    assert retry["failed_checks"] == ["attribute_verification"]
    assert retry["failed_attributes"] == ["lighting"]


def test_prepare_refinement_uses_ranking_failures_when_holdout_is_empty(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    dfs.prepare_refinement(
        str(grade),
        str(refinement),
        decision_uri=str(decision),
        loop_iteration=1,
    )
    first_grade = grade / "iteration-1"
    ranking = first_grade / "ranking" / "cosmos_evaluator.json"
    ranking.parent.mkdir(parents=True)
    ranking.write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.8,
                "passed": False,
                "clips": [
                    {
                        "passed": False,
                        "attribute_verification": {
                            "passed": False,
                            "checks": [
                                {"variable": "surface_finish", "passed": False},
                                {"variable": "lighting", "passed": True},
                            ],
                        },
                        "temporal_consistency": {"passed": False},
                        "hallucination": {"passed": True},
                    }
                ],
            }
        )
    )
    final_report = first_grade / "cosmos_evaluator.json"
    final_report.write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.0,
                "passed": False,
                "clips": [],
            }
        )
    )
    assert (
        dfs.grade_gate(
            str(first_grade),
            str(first_grade / "decision.json"),
            0.75,
            str(refinement),
        )
        == "loop_back"
    )

    retry = dfs.prepare_refinement(
        str(grade),
        str(refinement),
        decision_uri=str(decision),
        loop_iteration=2,
    )

    assert retry["failed_checks"] == [
        "attribute_verification",
        "temporal_consistency",
    ]
    assert retry["failed_attributes"] == ["surface_finish"]
    assert retry["prior_evaluator_report_uri"] == str(final_report)
    assert retry["adaptation_evaluator_report_uri"] == str(ranking)
    assert retry["adaptation_evaluator_report_sha256"] == dfs._payload_sha256(
        json.loads(ranking.read_text())
    )


def test_prepare_refinement_reads_preceding_append_only_iteration(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    baseline = dfs.prepare_refinement(
        str(grade),
        str(refinement),
        decision_uri=str(decision),
        loop_iteration=1,
    )
    assert baseline["attempt"] == 0

    first_grade = grade / "iteration-1"
    first_grade.mkdir(parents=True)
    (first_grade / "cosmos_evaluator.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.4,
                "passed": False,
                "clips": [
                    {
                        "passed": False,
                        "appearance_fidelity": {"passed": False},
                        "hallucination": {"passed": True},
                    }
                ],
            }
        )
    )
    assert (
        dfs.grade_gate(
            str(first_grade),
            str(first_grade / "decision.json"),
            0.75,
            str(refinement),
        )
        == "loop_back"
    )

    retry = dfs.prepare_refinement(
        str(grade),
        str(refinement),
        decision_uri=str(decision),
        loop_iteration=2,
    )

    assert retry["attempt"] == 1
    assert retry["failed_checks"] == ["appearance_fidelity"]
    assert (first_grade / "cosmos_evaluator.json").is_file()
    assert (first_grade / "decision.json").is_file()
    assert not (grade / "cosmos_evaluator.json").exists()
    assert not decision.exists()


def test_prepare_refinement_replays_a_committed_adapted_attempt_idempotently(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "completed", "score": 0.4, "passed": False})
    )
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "loop_back"
    )
    retry = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    history = tmp_path / "configs" / "refinement-attempt-01.json"
    marker = tmp_path / "configs" / "refinement-attempt-01.commit.json"
    before = (refinement.read_bytes(), history.read_bytes(), marker.read_bytes())

    repeated = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )

    assert repeated == retry
    assert (refinement.read_bytes(), history.read_bytes(), marker.read_bytes()) == before
    assert not (tmp_path / "configs" / "refinement-attempt-02.json").exists()


def test_prepare_refinement_can_record_a_non_adaptive_policy(tmp_path: Path) -> None:
    scores = tmp_path / "grade"
    result = dfs.prepare_refinement(
        str(scores),
        str(tmp_path / "refinement.json"),
        enabled="false",
        base_control_weight="0.8",
        base_guidance="2.0",
    )
    assert result["adaptive"] is False
    assert result["adapted_from_prior_evaluation"] is False
    assert result["settings"] == {"control_weight": 0.8, "guidance": 2.0}


@pytest.mark.parametrize(
    "report",
    [
        {"status": "completed", "score": 0.74, "passed": True},
        {"status": "degraded", "score": 0.99, "passed": True},
    ],
    ids=["passed-below-score", "passed-non-completed"],
)
def test_prepare_refinement_adapts_exactly_when_quality_gate_retries(
    tmp_path: Path, report: dict
) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    (grade / "cosmos_evaluator.json").write_text(json.dumps(report))
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "loop_back"
    )

    retry = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )

    assert retry["attempt"] == 1
    assert retry["adapted_from_prior_evaluation"] is True
    assert retry["settings"] == {"control_weight": 1.0, "guidance": 2}


def test_prepare_refinement_changes_every_retry_then_fails_closed_at_saturation(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    current = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    effective_pairs = [
        (current["settings"]["control_weight"], current["settings"]["guidance"])
    ]

    for evaluation_cycle in range(1, 3):
        (grade / "cosmos_evaluator.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "score": 0.1,
                    "passed": False,
                    "evaluation_cycle": evaluation_cycle,
                }
            )
        )
        assert (
            dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
            == "loop_back"
        )
        current = dfs.prepare_refinement(
            str(grade), str(refinement), decision_uri=str(decision)
        )
        pair = (
            current["settings"]["control_weight"],
            current["settings"]["guidance"],
        )
        assert pair != effective_pairs[-1]
        assert current["attempt"] == evaluation_cycle
        effective_pairs.append(pair)

    pointer_before = refinement.read_bytes()
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "score": 0.1,
                "passed": False,
                "evaluation_cycle": 3,
            }
        )
    )
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "loop_back"
    )
    with pytest.raises(dfs.RefinementStateError, match="schedule is exhausted"):
        dfs.prepare_refinement(
            str(grade), str(refinement), decision_uri=str(decision)
        )
    assert refinement.read_bytes() == pointer_before
    assert not (tmp_path / "configs" / "refinement-attempt-03.json").exists()


def test_prepare_refinement_does_not_create_a_retry_after_promotion(
    tmp_path: Path,
) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    refinement = tmp_path / "configs" / "refinement.json"
    decision = grade / "decision.json"
    baseline = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps({"status": "completed", "score": 0.9, "passed": True})
    )
    assert (
        dfs.grade_gate(str(grade), str(decision), 0.75, str(refinement))
        == "promote_checkpoint"
    )

    promoted = dfs.prepare_refinement(
        str(grade), str(refinement), decision_uri=str(decision)
    )

    assert promoted == baseline
    assert not (tmp_path / "configs" / "refinement-attempt-01.json").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_control_weight": "1.1"}, "between 0 and 1"),
        ({"max_control_weight": "1.1"}, "between base_control_weight and 1"),
        ({"base_guidance": "2.5"}, "non-negative integer"),
        ({"guidance_step": "0.5"}, "non-negative integer"),
    ],
)
def test_prepare_refinement_rejects_values_cosmos_cannot_load(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dfs.prepare_refinement(
            str(tmp_path / "grade"),
            str(tmp_path / "refinement.json"),
            **kwargs,
        )


def test_prepare_refinement_propagates_non_not_found_reads_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refinement = tmp_path / "private-object-name.json"
    original = dfs._download_json

    def fail_read(uri: str):
        if uri == str(refinement):
            raise PermissionError(f"private provider detail for {uri}")
        return original(uri)

    monkeypatch.setattr(dfs, "_download_json", fail_read)

    with pytest.raises(dfs.RefinementStateError) as captured:
        dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))

    assert "read failed (PermissionError)" in str(captured.value)
    assert "private-object-name" not in str(captured.value)


def test_prepare_refinement_fails_closed_on_malformed_prior_state(
    tmp_path: Path,
) -> None:
    refinement = tmp_path / "private-object-name.json"
    refinement.write_text("not-json")

    with pytest.raises(dfs.RefinementStateError) as captured:
        dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))

    message = str(captured.value)
    assert "read failed (JSONDecodeError)" in message
    assert "private-object-name" not in message


def test_prepare_refinement_is_idempotent_only_with_commit_proof(
    tmp_path: Path,
) -> None:
    refinement = tmp_path / "configs" / "refinement.json"
    baseline = dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))
    history = tmp_path / "configs" / "refinement-attempt-00.json"
    marker = tmp_path / "configs" / "refinement-attempt-00.commit.json"
    before_history = history.read_bytes()
    before_marker = marker.read_bytes()

    repeated = dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))

    assert repeated == baseline
    assert history.read_bytes() == before_history
    assert marker.read_bytes() == before_marker


def test_prepare_refinement_repairs_exact_history_after_marker_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refinement = tmp_path / "configs" / "refinement.json"
    original = dfs._put_immutable_json
    failed_once = False

    def fail_marker(payload: dict, uri: str, *, label: str) -> str:
        nonlocal failed_once
        if label == "refinement commit marker" and not failed_once:
            failed_once = True
            raise dfs.RefinementStateError("injected marker failure")
        return original(payload, uri, label=label)

    monkeypatch.setattr(dfs, "_put_immutable_json", fail_marker)
    with pytest.raises(dfs.RefinementStateError, match="injected marker failure"):
        dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))
    history = tmp_path / "configs" / "refinement-attempt-00.json"
    assert history.is_file()
    assert not refinement.exists()

    recovered = dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))

    assert recovered["attempt"] == 0
    assert (tmp_path / "configs" / "refinement-attempt-00.commit.json").is_file()


def test_prepare_refinement_never_overwrites_conflicting_attempt_history(
    tmp_path: Path,
) -> None:
    refinement = tmp_path / "configs" / "refinement.json"
    history = tmp_path / "configs" / "refinement-attempt-00.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps({"schema": "conflicting"}))
    before = history.read_bytes()

    with pytest.raises(dfs.RefinementStateError, match="conflicting immutable"):
        dfs.prepare_refinement(str(tmp_path / "grade"), str(refinement))

    assert history.read_bytes() == before
    assert not refinement.exists()


def test_grade_gate_promotes_above_threshold(tmp_path: Path) -> None:
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(
        json.dumps({"status": "completed", "score": 0.8, "passed": True})
    )
    decision_path = tmp_path / "decision.json"
    decision = dfs.grade_gate(str(scores), str(decision_path), threshold=0.5)
    assert decision == "promote_checkpoint"
    assert json.loads(decision_path.read_text())["decision"] == "promote_checkpoint"


def test_grade_gate_loops_below_threshold(tmp_path: Path, monkeypatch) -> None:
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(json.dumps({"score": 0.1}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "decision.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_accepts_string_threshold(tmp_path: Path, monkeypatch) -> None:
    """The blueprint interpolates a quoted config.grade_threshold; grade_gate must
    cast a str threshold (and fall back to 0.5 on a non-numeric value)."""
    scores = tmp_path / "vlm_eval_stub.json"
    scores.write_text(
        json.dumps({"status": "completed", "score": 0.6, "passed": True})
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    # "0.5" (str) -> 0.6 >= 0.5 -> promote.
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold="0.5")
        == "promote_checkpoint"
    )
    # non-numeric -> fallback 0.5 -> 0.6 >= 0.5 -> promote.
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold="bogus")
        == "promote_checkpoint"
    )


@pytest.mark.parametrize(
    "report",
    [
        {"score": "n/a"},  # non-numeric score
        {"score": None},
        {"score": {"overall": 0.9}},  # nested where a number is expected
        ["not", "an", "object"],  # JSON root is not a report at all
    ],
    ids=["non-numeric", "null", "nested", "not-an-object"],
)
def test_grade_gate_loops_back_on_a_malformed_report(
    tmp_path: Path, monkeypatch, report
) -> None:
    """A gate exists to make a decision, so a malformed score must not abort the loop.

    The report downloads cleanly here — only its ``score`` is unusable — so the
    download's own error handling never sees it.
    """

    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(json.dumps(report))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_will_not_promote_a_degraded_report(
    tmp_path: Path, monkeypatch
) -> None:
    """A high score the evaluator itself flagged as degraded must not promote.

    The evaluator marks a run degraded when it lost object storage part-way, so the
    score reflects the clips it managed to read rather than the batch.
    """

    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(json.dumps({"score": 0.95, "status": "degraded"}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.5)
        == "loop_back"
    )


def test_grade_gate_will_not_promote_when_a_hard_check_failed(
    tmp_path: Path, monkeypatch
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    scores.write_text(
        json.dumps({"score": 0.95, "status": "completed", "passed": False})
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(scores), str(tmp_path / "d.json"), threshold=0.75)
        == "loop_back"
    )


def test_quality_disposition_accepts_only_a_complete_hard_check_pass(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": 0.81, "status": "completed", "passed": True})
    )
    result = dfs.enforce_quality_disposition(
        str(scores), str(disposition), threshold="0.75"
    )
    assert result["quality_status"] == "accepted"
    assert json.loads(disposition.read_text())["quality_status"] == "accepted"


@pytest.mark.parametrize(
    ("report", "expected_status", "expected_decision"),
    [
        (
            {"score": 0.81, "status": "completed", "passed": True},
            "accepted",
            "promote_checkpoint",
        ),
        (
            {"score": 0.74, "status": "completed", "passed": True},
            "rejected",
            "loop_back",
        ),
    ],
)
def test_write_quality_disposition_routes_without_raising(
    tmp_path: Path,
    monkeypatch,
    report: dict,
    expected_status: str,
    expected_decision: str,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(json.dumps(report))
    decisions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: decisions.append((uri, decision)),
    )

    result = dfs.write_quality_disposition(
        str(scores),
        str(disposition),
        "s3://example/grade/decision.json",
        threshold=0.75,
    )

    assert result["quality_status"] == expected_status
    assert result["decision"] == expected_decision
    persisted = json.loads(disposition.read_text())
    assert persisted["quality_status"] == expected_status
    assert persisted["decision"] == expected_decision
    assert decisions == [("s3://example/grade/decision.json", expected_decision)]


def test_quality_disposition_persists_rejection_before_failing(tmp_path: Path) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": 0.9, "status": "completed", "passed": False})
    )
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert payload["hard_checks_passed"] is False


@pytest.mark.parametrize(
    "report_contents",
    [None, "not-json", '["valid", "json", "but-not-an-object"]'],
    ids=["missing", "unparseable", "non-object"],
)
def test_quality_disposition_is_written_for_every_unreadable_report(
    tmp_path: Path, report_contents: str | None
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    if report_contents is not None:
        scores.write_text(report_contents)
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert payload["evaluator_status"] == "missing"
    assert any("unavailable or malformed" in reason for reason in payload["reasons"])


def test_quality_disposition_rejects_a_non_numeric_score_and_persists_reason(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(
        json.dumps({"score": "n/a", "status": "completed", "passed": True})
    )
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["score"] == 0.0
    assert "evaluator score is not numeric" in payload["reasons"]


def test_quality_disposition_resolves_a_report_prefix(tmp_path: Path) -> None:
    grade = tmp_path / "grade"
    grade.mkdir()
    (grade / "cosmos_evaluator.json").write_text(
        json.dumps({"score": 0.9, "status": "completed", "passed": True})
    )
    disposition = tmp_path / "quality_disposition.json"
    result = dfs.enforce_quality_disposition(
        str(grade), str(disposition), threshold=0.75
    )
    assert result["quality_status"] == "accepted"
    assert result["evaluator_report_uri"].endswith("grade/cosmos_evaluator.json")


def test_quality_disposition_rejects_a_degraded_report(tmp_path: Path) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(json.dumps({"score": 1.0, "status": "degraded", "passed": True}))
    with pytest.raises(RuntimeError, match="quality rejected"):
        dfs.enforce_quality_disposition(str(scores), str(disposition), threshold=0.75)
    payload = json.loads(disposition.read_text())
    assert payload["quality_status"] == "rejected"
    assert "evaluator status is degraded" in payload["reasons"]


@pytest.mark.parametrize(
    ("report", "expected_status", "expected_decision"),
    [
        (
            {"score": 0.9, "status": "completed", "passed": True},
            "accepted",
            "promote_checkpoint",
        ),
        (
            {"score": 0.9, "status": "incomplete", "passed": True},
            "rejected",
            "loop_back",
        ),
        (
            {"score": 0.9, "status": "completed", "passed": False},
            "rejected",
            "loop_back",
        ),
    ],
)
def test_dynamic_quality_disposition_persists_strict_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: dict,
    expected_status: str,
    expected_decision: str,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    scores.write_text(json.dumps(report), encoding="utf-8")
    decisions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: decisions.append((uri, decision)),
    )

    result = dfs.write_quality_disposition(
        str(scores),
        str(disposition),
        "s3://example-bucket/run/decision.json",
        threshold=0.75,
    )

    assert result["quality_status"] == expected_status
    assert result["decision"] == expected_decision
    assert decisions == [("s3://example-bucket/run/decision.json", expected_decision)]
    persisted = json.loads(disposition.read_text())
    assert persisted["quality_status"] == expected_status
    assert persisted["decision"] == expected_decision


@pytest.mark.parametrize("contents", [None, "not-json", "[]"])
def test_dynamic_quality_disposition_rejects_unavailable_or_malformed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str | None,
) -> None:
    scores = tmp_path / "cosmos_evaluator.json"
    disposition = tmp_path / "quality_disposition.json"
    if contents is not None:
        scores.write_text(contents, encoding="utf-8")
    decisions: list[str] = []
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda _uri, decision: decisions.append(decision),
    )

    result = dfs.write_quality_disposition(
        str(scores), str(disposition), str(tmp_path / "decision.json"), 0.75
    )

    assert result["quality_status"] == "rejected"
    assert result["decision"] == "loop_back"
    assert result["reasons"]
    assert decisions == ["loop_back"]


def test_grade_gate_malformed_authoritative_report_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """A present but malformed newest report must not promote from stale data."""

    (tmp_path / "cosmos_evaluator.json").write_text(json.dumps({"score": "n/a"}))
    (tmp_path / "vlm_eval_stub.json").write_text(json.dumps({"score": 0.9}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(tmp_path), str(tmp_path / "d.json"), threshold=0.5)
        == "loop_back"
    )


def test_download_json_missing_exact_file_does_not_substitute(
    tmp_path: Path, monkeypatch
) -> None:
    """When the requested .json is missing and download falls back to the prefix
    dir, _download_json must raise, not silently return a different JSON."""
    import pytest

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / "decision.json").write_text(json.dumps({"decision": "loop_back"}))

    class _FakeStorage:
        def download_file(self, uri, dest):  # noqa: ARG002
            raise FileNotFoundError("exact object absent")

    monkeypatch.setattr(dfs, "_storage", lambda: _FakeStorage())
    with pytest.raises(FileNotFoundError):
        dfs._download_json("s3://bucket/grade/vlm_eval_stub.json")


def test_grade_gate_missing_eval_loops_not_reads_decision(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing eval result must loop_back, never mis-read decision.json as score."""
    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    # A promote decision.json is present but the eval result is absent.
    (prefix_dir / "decision.json").write_text(
        json.dumps({"decision": "promote_checkpoint"})
    )

    class _FakeStorage:
        def download_file(self, uri, dest):  # noqa: ARG002
            raise FileNotFoundError("exact object absent")

        def upload_file(self, source, uri):  # noqa: ARG002
            return uri

    monkeypatch.setattr(dfs, "_storage", lambda: _FakeStorage())
    assert (
        dfs.grade_gate("s3://bucket/grade/", "s3://bucket/grade/decision.json", 0.5)
        == "loop_back"
    )


def test_grade_gate_reads_the_cosmos_evaluator_report(
    tmp_path: Path, monkeypatch
) -> None:
    """The gate must threshold on the Cosmos Evaluator score the evaluate stage writes."""
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / RESULT_FILENAME).write_text(
        json.dumps({"status": "completed", "score": 0.9, "passed": True})
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(prefix_dir), str(tmp_path / "decision.json"), 0.5)
        == "promote_checkpoint"
    )


def test_grade_gate_falls_back_to_the_older_vlm_eval_report(
    tmp_path: Path, monkeypatch
) -> None:
    """Runs started before the evaluate stage existed must still grade."""
    from npa.workbench.vlm_eval import RESULT_FILENAME

    prefix_dir = tmp_path / "grade"
    prefix_dir.mkdir()
    (prefix_dir / RESULT_FILENAME).write_text(json.dumps({"score": 0.2}))
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.decisions.write_decision",
        lambda uri, decision: None,
    )
    assert (
        dfs.grade_gate(str(prefix_dir), str(tmp_path / "decision.json"), 0.5)
        == "loop_back"
    )


def test_curate_merges_the_cosmos_curator_report(tmp_path: Path, monkeypatch) -> None:
    """The FiftyOne review report must carry the curator stage's summary."""
    curator_report = tmp_path / "cosmos_curator.json"
    curator_report.write_text(
        json.dumps(
            {
                "schema": "npa.cosmos_curate.curation.v1",
                "status": "completed",
                "engine": "cosmos-curator-stages",
                "curated_uri": "s3://b/p/curation/cosmos_curator/",
                "clip_count": 6,
                "filtered_count": 1,
                "variant_count": 2,
                "total_duration_s": 18.0,
                "motion_filter": "score-only",
            }
        )
    )
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-0/augmented_video.mp4",
        "p/cosmos_augmented/aug-1/augmented_video.mp4",
    ]
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda uri: keys,
    )
    _mock_committed_manifest(monkeypatch, keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    report = dfs.curate(
        "s3://b/p/cosmos_augmented/",
        str(tmp_path / "report.json"),
        curator_report_uri=str(curator_report),
    )
    assert report["cosmos_curator"]["engine"] == "cosmos-curator-stages"
    assert report["cosmos_curator"]["clip_count"] == 6
    assert report["cosmos_curator"]["filtered_count"] == 1
    # The review stage's own findings are untouched by the merge.
    assert report["schema"] == "npa.fiftyone.curation.v1"
    assert report["multiply"]["mode"] == "multi-variant"


def test_curate_fails_closed_when_curator_report_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dfs, "_list_keys", lambda uri: ["p/cosmos_augmented/aug-0/augmented_video.mp4"]
    )
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        dfs.curate(
            "s3://b/p/cosmos_augmented/",
            str(tmp_path / "report.json"),
            curator_report_uri=str(tmp_path / "absent.json"),
        )


def test_curate_fails_closed_when_no_curator_report_is_passed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dfs, "_list_keys", lambda uri: ["p/cosmos_augmented/aug-0/augmented_video.mp4"]
    )
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    with pytest.raises(RuntimeError, match="report URI is required"):
        dfs.curate("s3://b/p/cosmos_augmented/", str(tmp_path / "report.json"))


def test_curate_counts_augmented_set(tmp_path: Path, monkeypatch) -> None:
    # Per-clip layout as emitted by publish_transfer_to_s3 (subdirs + top-level
    # manifest.json which must NOT be counted as a clip).
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-run/augmented_video.mp4",
        "p/cosmos_augmented/aug-run/frame-00000.png",
        "p/cosmos_augmented/aug-run/frame-00001.png",
        "p/cosmos_augmented/aug-run/metadata.json",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    written = {}
    monkeypatch.setattr(
        dfs,
        "_upload_json",
        lambda payload, uri: written.update(payload=payload, uri=uri) or uri,
    )
    curator_report = _completed_curator_report(tmp_path / "curator.json")
    report = dfs.curate(
        "s3://b/p/cosmos_augmented/",
        "s3://b/p/curation/report.json",
        curator_report_uri=str(curator_report),
    )
    assert report["video_count"] == 1
    assert report["frame_count"] == 2
    assert set(report["clip_ids"]) == {"aug-run"}
    assert "manifest.json" not in report["clip_ids"]
    assert report["status"] == "curated"
    # Single-variant limitation surfaced in the machine-readable report.
    assert report["multiply"]["mode"] == "single-variant"
    assert report["curation_engine"] == "fiftyone-brain"


def test_curate_fails_closed_when_fiftyone_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    from npa.workflows import data_factory_curate as dfc

    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda uri: ["p/cosmos_augmented/aug-0/augmented_video.mp4"],
    )
    monkeypatch.setattr(
        dfc,
        "run_curation",
        lambda **kwargs: (_ for _ in ()).throw(dfc.FiftyoneUnavailable("absent")),
    )
    with pytest.raises(dfc.FiftyoneUnavailable, match="absent"):
        dfs.curate(
            "s3://b/p/cosmos_augmented/",
            "s3://b/p/curation/report.json",
            curator_report_uri=str(
                _completed_curator_report(tmp_path / "curator.json")
            ),
        )


def test_generate_configs_feeds_first_augmentation_to_transfer(tmp_path: Path) -> None:
    """The sampled config manifest must be consumable by the augment stage."""
    from npa.cli.workbench.cosmos2 import _first_augmentation

    configs_uri = str(tmp_path / "configs") + "/"
    manifest = dfs.generate_configs(configs_uri, "3", seed="run-xyz")
    assert manifest["n_augmentations"] == 3

    combo = _first_augmentation(configs_uri)
    assert combo == manifest["augmentations"][0]
    assert set(combo) == {
        "lighting",
        "background",
        "color_grade",
        "surface_finish",
        "inference_seed",
        "prompt",
    }
    # The prompt is what the augment stage feeds into Cosmos Transfer.
    assert combo["prompt"]


def test_generate_configs_uses_leisaac_task_lineage_for_conditioning(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "leisaac-lineage.json").write_text(
        json.dumps(
            {
                "schema": "npa.leisaac.paidf-input.v1",
                "source": {
                    "task": "LeIsaac-SO101-LiftCube-v0",
                    "dataset_uri": "s3://bucket/dataset/versions/v1",
                    "episode_index": 0,
                },
            }
        )
    )
    manifest = dfs.generate_configs(str(tmp_path / "configs") + "/", 1, seed="lift")
    combo = manifest["augmentations"][0]
    assert set(combo) == {
        "lighting",
        "background",
        "color_grade",
        "surface_finish",
        "inference_seed",
        "prompt",
    }
    assert "red-cube lift motion" in combo["prompt"]
    assert "Preserve the exact foreground objects" in combo["prompt"]
    assert manifest["source_leisaac"]["episode_index"] == 0


def test_generate_configs_non_numeric_count_falls_back(tmp_path: Path) -> None:
    manifest = dfs.generate_configs(str(tmp_path / "c") + "/", "not-a-number", seed="s")
    assert manifest["n_augmentations"] == 2


def _png(path: Path) -> Path:
    import pytest

    Image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), (20, 40, 60)).save(path)
    return path


def test_publish_transfer_layout_interoperates_with_curate_and_viz(
    tmp_path: Path, monkeypatch
) -> None:
    """The real producer's S3 layout must flow through curate + build_run_rrd."""
    import pytest

    pytest.importorskip("rerun")
    from npa.workbench.cosmos import transfer as tx
    from npa.workflows.data_factory_viz import build_run_rrd

    video = tmp_path / "out.mp4"
    video.write_bytes(b"x" * 200_000)

    # Mock frame extraction (no cosmos venv here); write real PNGs into dest.
    def fake_extract(vp, dest, max_frames=8):
        return [_png(Path(dest) / f"frame-{i:05d}.png") for i in range(3)]

    monkeypatch.setattr(tx, "extract_frames", fake_extract)

    # Fake storage: mirror uploaded keys into a local tree so we can (a) collect
    # bucket-relative keys for curate, and (b) run build_run_rrd against the tree.
    mirror = tmp_path / "mirror"
    recorded: list[str] = []

    class FakeStorage:
        def upload_file(self, local: str, uri: str) -> str:
            key = uri.replace("s3://bkt/", "")
            recorded.append(key)
            out = mirror / key
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(Path(local).read_bytes())
            return uri

    manifest = tx.publish_transfer_to_s3(
        {"video_path": str(video), "video_bytes": 200_000, "spec": "s"},
        "s3://bkt/run1/cosmos_augmented/",
        run_id="run1",
        variables={"weather": "rainy", "time_of_day": "night"},
        storage_client=FakeStorage(),
    )
    assert manifest["frame_count"] == 3

    # (a) curate must parse the produced layout correctly.
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: recorded)
    _mock_committed_manifest(monkeypatch, recorded, bucket="bkt")
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    report = dfs.curate(
        "s3://bkt/run1/cosmos_augmented/",
        "s3://bkt/run1/curation/report.json",
        curator_report_uri=str(
            _completed_curator_report(tmp_path / "curator-interoperability.json")
        ),
    )
    assert report["clip_ids"] == ["aug-run1"], report["clip_ids"]
    assert report["video_count"] == 1
    assert report["frame_count"] == 3
    assert "manifest.json" not in report["clip_ids"]

    # (b) build_run_rrd must consume the same per-clip layout (frames + metadata).
    out_rrd = tmp_path / "reports" / "sim2real.rrd"
    result = build_run_rrd(str(mirror / "run1"), str(out_rrd))
    assert result["frames_logged"] >= 3
    assert out_rrd.is_file()


def test_curate_reports_multi_variant_for_multiple_clips(
    tmp_path: Path, monkeypatch
) -> None:
    """N augmented clip dirs (multiply) must be counted and reported multi-variant."""
    keys = [
        "p/cosmos_augmented/manifest.json",
        "p/cosmos_augmented/aug-run-0/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-0/frame-00000.png",
        "p/cosmos_augmented/aug-run-0/metadata.json",
        "p/cosmos_augmented/aug-run-1/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-1/frame-00000.png",
        "p/cosmos_augmented/aug-run-1/metadata.json",
        "p/cosmos_augmented/aug-run-2/augmented_video.mp4",
        "p/cosmos_augmented/aug-run-2/metadata.json",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)
    report = dfs.curate(
        "s3://b/p/cosmos_augmented/",
        "s3://b/p/curation/report.json",
        curator_report_uri=str(
            _completed_curator_report(tmp_path / "curator-multi.json")
        ),
    )
    assert report["augmented_clips"] == 3
    assert set(report["clip_ids"]) == {"aug-run-0", "aug-run-1", "aug-run-2"}
    assert report["video_count"] == 3
    assert report["multiply"]["mode"] == "multi-variant"
    assert report["multiply"]["variant_count"] == 3


def test_finalize_reports_multi_variant_from_clip_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    keys = [
        "physical-ai-data-factory/run1/input/video_0.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/manifest.json",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-0/augmented_video.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/aug-run1-1/augmented_video.mp4",
        "physical-ai-data-factory/run1/reports/sim2real.rrd",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys, bucket="b")
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.finalize(
        "s3://b/physical-ai-data-factory/run1/", "s3://b/.../final.json"
    )
    assert report["multiply_mode"] == "multi-variant"
    assert report["variant_count"] == 2


def test_finalize_counts_only_latest_append_only_augmentation_iteration(
    monkeypatch,
) -> None:
    keys = [
        "physical-ai-data-factory/run1/cosmos_augmented/iteration-1/manifest.json",
        "physical-ai-data-factory/run1/cosmos_augmented/iteration-1/old/augmented_video.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/iteration-2/manifest.json",
        "physical-ai-data-factory/run1/cosmos_augmented/iteration-2/new-a/augmented_video.mp4",
        "physical-ai-data-factory/run1/cosmos_augmented/iteration-2/new-b/augmented_video.mp4",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    _mock_committed_manifest(monkeypatch, keys, bucket="b")
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)

    report = dfs.finalize(
        "s3://b/physical-ai-data-factory/run1/", "s3://b/run1/final.json"
    )

    assert report["multiply_mode"] == "multi-variant"
    assert report["variant_count"] == 2


def test_all_augmentations_reads_every_combo(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _all_augmentations, _first_augmentation

    configs_uri = str(tmp_path / "configs") + "/"
    manifest = dfs.generate_configs(configs_uri, "4", seed="multi")
    combos = _all_augmentations(configs_uri)
    assert len(combos) == 4
    assert combos == manifest["augmentations"]
    assert _first_augmentation(configs_uri) == combos[0]


def test_all_augmentations_missing_manifest_fails_closed(tmp_path: Path) -> None:
    from npa.cli.workbench.cosmos2 import _all_augmentations

    with pytest.raises(
        typer.BadParameter,
        match="could not read the configured augmentation manifest",
    ):
        _all_augmentations(str(tmp_path / "nope") + "/")


def test_finalize_aggregates_stage_artifacts(tmp_path: Path, monkeypatch) -> None:
    keys = [
        "physical-ai-data-factory/run1/input/video_0.mp4",
        "physical-ai-data-factory/run1/labeled_original/captions.json",
        "physical-ai-data-factory/run1/reports/sim2real.rrd",
    ]
    monkeypatch.setattr(dfs, "_list_keys", lambda uri: keys)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    report = dfs.finalize(
        "s3://b/physical-ai-data-factory/run1/",
        "s3://b/physical-ai-data-factory/run1/reports/final.json",
    )
    assert report["artifact_count"] == 3
    assert report["has_rrd"] is True
    assert report["stages"]["input"] == 1
    assert report["multiply_mode"] == "single-variant"


def test_curation_and_final_reports_carry_input_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    source = {
        "schema_version": "npa.paidf.input-provenance.v1",
        "source_kind": "upstream_sample",
        "input_origin": "actual_capture",
        "input_origin_label": "Upstream real sample",
        "sha256": "a" * 64,
        "staged_canonical_s3_uri": "s3://b/physical-ai-data-factory/run/input/",
        "derivation": {"kind": "normalized_conditioning_clip"},
    }
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda _uri: [
            "physical-ai-data-factory/run/cosmos_augmented/aug-run/frame-00000.png"
        ],
    )
    curator_report = _completed_curator_report(tmp_path / "curator.json")
    monkeypatch.setattr(
        dfs,
        "_download_json",
        lambda uri: (
            source
            if str(uri).endswith("/input/provenance.json")
            else json.loads(Path(uri).read_text())
        ),
    )
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    monkeypatch.setattr(dfs, "_enrich_with_fiftyone_curation", _stub_real_fiftyone)

    curated = dfs.curate(
        "s3://b/physical-ai-data-factory/run/cosmos_augmented/",
        "s3://b/physical-ai-data-factory/run/curation/report.json",
        curator_report_uri=str(curator_report),
    )
    final = dfs.finalize(
        "s3://b/physical-ai-data-factory/run/",
        "s3://b/physical-ai-data-factory/run/reports/final.json",
    )

    assert curated["input_source"] == source
    assert [group["name"] for group in curated["dataset_groups"]] == [
        "source",
        "conditioning",
        "augmented",
    ]
    assert final["input_source"] == source


def test_is_truthy_matches_common_values() -> None:
    for v in ("1", "true", "TRUE", "yes", "on", True):
        assert dfs._is_truthy(v) is True
    for v in ("", "0", "false", "no", None, "off"):
        assert dfs._is_truthy(v) is False


class _FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload_file(self, local: str, dest: str) -> str:
        assert Path(local).is_file()  # a real PNG was produced
        self.uploads.append(dest)
        return dest


def test_seed_default_input_frames_writes_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(dfs, "_list_keys", lambda _uri: [])
    fake = _FakeStorage()
    monkeypatch.setattr(dfs, "_storage", lambda: fake)

    written = dfs._seed_default_input_frames(
        "s3://b/physical-ai-data-factory/run/input/", count=3, seed="x"
    )

    assert written == 3
    assert len(fake.uploads) == 3
    assert all(dest.endswith(".png") for dest in fake.uploads)
    assert fake.uploads[0].endswith("input/frame_0000.png")


def test_seed_default_input_frames_skips_when_images_exist(monkeypatch) -> None:
    monkeypatch.setattr(
        dfs,
        "_list_keys",
        lambda _uri: ["physical-ai-data-factory/run/input/frame_0000.png"],
    )
    fake = _FakeStorage()
    monkeypatch.setattr(dfs, "_storage", lambda: fake)

    written = dfs._seed_default_input_frames(
        "s3://b/physical-ai-data-factory/run/input/", seed="x"
    )

    assert written == 0
    assert fake.uploads == []


def test_seed_default_input_frames_noop_without_uri() -> None:
    assert dfs._seed_default_input_frames("", seed="x") == 0


def test_generate_configs_seeds_default_input_when_flag_set(
    tmp_path: Path, monkeypatch
) -> None:
    calls: dict[str, str] = {}

    def fake_seed(input_uri: str, seed: str = "") -> int:
        calls["input_uri"] = input_uri
        calls["seed"] = seed
        return 8

    monkeypatch.setattr(dfs, "_seed_default_input_frames", fake_seed)
    monkeypatch.setattr(dfs, "_upload_json", lambda payload, uri: uri)
    result = dfs.generate_configs(
        str(tmp_path / "c.json"),
        n_augmentations=1,
        seed="run-x",
        input_uri="s3://b/physical-ai-data-factory/run-x/input/",
        seed_default_input="true",
    )
    assert result["seeded_default_input_frames"] == 8
    assert result["input_source"]["kind"] == "npa_seeded_fixture"
    assert result["input_source"]["staged_canonical_s3_uri"] == calls["input_uri"]
    assert result["input_source"]["frame_count"] == 8
    assert calls["input_uri"] == "s3://b/physical-ai-data-factory/run-x/input/"
    assert calls["seed"] == "run-x"


def test_generate_configs_no_seed_when_flag_false(tmp_path: Path, monkeypatch) -> None:
    def boom(*_a, **_k) -> int:  # pragma: no cover - must not run
        raise AssertionError("must not seed default input when the flag is falsy")

    monkeypatch.setattr(dfs, "_seed_default_input_frames", boom)
    result = dfs.generate_configs(
        str(tmp_path / "c.json"),
        n_augmentations=1,
        seed="s",
        input_uri="s3://b/input/",
        seed_default_input="false",
    )
    assert result["seeded_default_input_frames"] == 0
    assert result["input_source"]["kind"] == "operator_provided"


def test_generate_configs_records_the_seeded_count_in_the_written_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """The count has to be in the artifact, not just stdout.

    Regression: the manifest was uploaded before the seeding block ran, so
    configs/manifest.json never said whether the run used synthetic frames.
    """
    monkeypatch.setattr(dfs, "_seed_default_input_frames", lambda input_uri, seed="": 5)
    real_upload = dfs._upload_json
    monkeypatch.setattr(
        dfs,
        "_upload_json",
        lambda payload, uri: (
            uri if uri.startswith("s3://") else real_upload(payload, uri)
        ),
    )
    configs = tmp_path / "c.json"

    dfs.generate_configs(
        str(configs),
        n_augmentations=1,
        seed="run-x",
        input_uri="s3://b/physical-ai-data-factory/run-x/input/",
        seed_default_input="true",
    )

    written = json.loads(configs.read_text(encoding="utf-8"))
    assert written["seeded_default_input_frames"] == 5
    assert written["input_source"]["kind"] == "npa_seeded_fixture"


def test_generate_configs_fails_when_requested_seeding_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """Requested-but-failed seeding must not defer the failure to annotate-original."""

    def boom(*_a, **_k) -> int:
        raise RuntimeError("Pillow is not installed")

    monkeypatch.setattr(dfs, "_seed_default_input_frames", boom)

    with pytest.raises(RuntimeError, match="seed_default_input was requested"):
        dfs.generate_configs(
            str(tmp_path / "c.json"),
            n_augmentations=1,
            seed="s",
            input_uri="s3://b/input/",
            seed_default_input="true",
        )


def test_generate_configs_fixture_refuses_existing_user_media(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dfs, "_seed_default_input_frames", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        dfs,
        "_download_json",
        lambda _uri: {
            "schema_version": "npa.paidf.input-provenance.v1",
            "source_kind": "user_supplied",
            "input_origin_label": "User-supplied input",
        },
    )

    with pytest.raises(RuntimeError, match="refusing to silently reuse or overwrite"):
        dfs.generate_configs(
            str(tmp_path / "c.json"),
            n_augmentations=1,
            seed="s",
            input_uri="s3://b/input/",
            seed_fixture="true",
        )
