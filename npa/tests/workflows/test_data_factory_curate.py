"""Unit tests for the pure FiftyOne-curation logic (no FiftyOne required)."""

from __future__ import annotations

from npa.workflows import data_factory_curate as dfc


def test_select_curated_keeps_most_unique_representative() -> None:
    ids = ["a", "b", "c", "d"]
    uniqueness = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.4}
    # a and b are near-duplicates; a is more unique so it wins.
    selection = dfc.select_curated(ids, uniqueness, [("a", "b")])
    assert "a" in selection["kept"]
    assert "b" not in selection["kept"]
    dropped_ids = {d["id"] for d in selection["dropped"]}
    assert dropped_ids == {"b"}
    dropped_b = next(d for d in selection["dropped"] if d["id"] == "b")
    assert dropped_b["reason"] == "near_duplicate"
    assert dropped_b["representative"] == "a"
    assert selection["kept_count"] == 3
    assert selection["dropped_count"] == 1
    assert selection["near_duplicate_count"] == 1


def test_select_curated_clusters_transitive_duplicates() -> None:
    ids = ["a", "b", "c", "d"]
    uniqueness = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.7}
    # a-b and b-c form one transitive cluster {a,b,c}; b (0.9) is the rep.
    selection = dfc.select_curated(ids, uniqueness, [("a", "b"), ("b", "c")])
    clusters = selection["near_duplicate_clusters"]
    assert len(clusters) == 1
    assert clusters[0]["representative"] == "b"
    assert clusters[0]["members"] == ["a", "b", "c"]
    assert "b" in selection["kept"]
    assert "d" in selection["kept"]  # untouched singleton
    assert {d["id"] for d in selection["dropped"]} == {"a", "c"}


def test_select_curated_flags_redundant_low_uniqueness() -> None:
    ids = ["a", "b", "c", "d"]
    uniqueness = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.01}
    selection = dfc.select_curated(ids, uniqueness, [], redundant_quantile=0.15)
    # No duplicates -> all kept, but the lowest-uniqueness sample is flagged.
    assert selection["kept"] == ["a", "b", "c", "d"]
    assert selection["dropped_count"] == 0
    assert "d" in selection["redundant"]
    assert "a" not in selection["redundant"]


def test_select_curated_deterministic_tiebreak() -> None:
    ids = ["z", "y"]
    uniqueness = {"z": 0.5, "y": 0.5}
    selection = dfc.select_curated(ids, uniqueness, [("z", "y")])
    # Equal uniqueness -> stable id tiebreak keeps "y" (sorts first).
    assert selection["kept"] == ["y"]
    assert [d["id"] for d in selection["dropped"]] == ["z"]


def test_uniqueness_summary_empty_and_populated() -> None:
    assert dfc.uniqueness_summary({}) == {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    summ = dfc.uniqueness_summary({"a": 0.2, "b": 0.8})
    assert summ["count"] == 2
    assert summ["min"] == 0.2
    assert summ["max"] == 0.8
    assert summ["mean"] == 0.5


def test_uniqueness_summary_ignores_non_finite() -> None:
    summ = dfc.uniqueness_summary({"a": 0.4, "b": float("nan"), "c": 0.6})
    assert summ["count"] == 2
    assert summ["min"] == 0.4
    assert summ["max"] == 0.6
    assert summ["mean"] == 0.5


def test_merge_records_uniqueness_method() -> None:
    selection = dfc.select_curated(["a"], {"a": 0.5}, [])
    report = dfc.merge_curation_into_report(
        {"schema": "npa.fiftyone.curation.v1"},
        engine=dfc.CURATION_ENGINE_FIFTYONE,
        fiftyone_version="1.15.0",
        embedding_kind="rgb16-hist8",
        dedup_threshold=0.1,
        uniqueness={"a": 0.5},
        selection=selection,
        visualization=None,
        fields=[],
        uniqueness_method="embedding-fallback",
    )
    assert report["fiftyone"]["brain"]["uniqueness_method"] == "embedding-fallback"
    assert report["fiftyone"]["brain"]["visualization_method"] == ""


def test_merge_curation_into_report_preserves_v1_and_adds_block() -> None:
    base = {
        "schema": "npa.fiftyone.curation.v1",
        "status": "curated",
        "clip_ids": ["a", "b"],
        "multiply": {"mode": "multi-variant", "variant_count": 2},
    }
    selection = dfc.select_curated(["a", "b"], {"a": 0.9, "b": 0.2}, [])
    report = dfc.merge_curation_into_report(
        base,
        engine=dfc.CURATION_ENGINE_FIFTYONE,
        fiftyone_version="1.15.0",
        embedding_kind="rgb16-hist8",
        dedup_threshold=0.1,
        uniqueness={"a": 0.9, "b": 0.2},
        selection=selection,
        visualization=[{"id": "a", "point": [0.1, 0.2]}],
        fields=["cloth_color", "lighting"],
    )
    # v1 fields preserved.
    assert report["schema"] == "npa.fiftyone.curation.v1"
    assert report["multiply"]["mode"] == "multi-variant"
    # New engine + block added.
    assert report["curation_engine"] == "fiftyone-brain"
    assert report["curated_kept"] == 2
    fo = report["fiftyone"]
    assert fo["fiftyone_version"] == "1.15.0"
    assert fo["fields"] == ["cloth_color", "lighting"]
    assert fo["brain"]["uniqueness"]["count"] == 2
    assert fo["brain"]["visualization_method"] == "pca"
    assert fo["samples"]["a"]["kept"] is True
    assert fo["samples"]["a"]["uniqueness"] == 0.9


def test_augmented_representatives_picks_first_frame_and_meta() -> None:
    keys = [
        "p/aug/manifest.json",
        "p/aug/c1/frame-00001.png",
        "p/aug/c1/frame-00000.png",
        "p/aug/c1/metadata.json",
        "p/aug/c2/frame-00000.png",
    ]
    reps = dfc._augmented_representatives(keys, "p/aug/")
    assert set(reps) == {"c1", "c2"}
    assert reps["c1"]["frame_key"] == "p/aug/c1/frame-00000.png"
    assert reps["c1"]["meta_key"] == "p/aug/c1/metadata.json"
    assert reps["c2"]["meta_key"] == ""


def test_run_curation_raises_when_fiftyone_absent() -> None:
    import pytest

    # In the unit-test env FiftyOne is not installed, so run_curation must raise
    # FiftyoneUnavailable (callers then fall back to the report-only path).
    with pytest.raises(dfc.FiftyoneUnavailable):
        dfc.run_curation(
            keys=["p/aug/c1/frame-0.png"],
            augment_prefix="p/aug/",
            base_report={"schema": "npa.fiftyone.curation.v1"},
            download_key=lambda key, dest: dest,
            read_json=lambda key: None,
            workdir="/tmp/does-not-matter",
        )
