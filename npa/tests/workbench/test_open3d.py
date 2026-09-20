"""Open3D contract and validator coverage; no native library, no network.

These tests exist because the validators in `artifacts.py` are the only thing
standing between a plausible-looking journal and a wrong reconstruction shipped as
a result. So each case here is a specific way a registration stage can fail while
still writing well-formed JSON: a scaled matrix instead of a rotation, more
downsampled points than input points, a fitness of zero next to a non-empty
correspondence set, a pose graph whose anchor drifted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from npa.workbench.open3d import runtime
from npa.workbench.open3d.artifacts import (
    Open3dError,
    canonical,
    read_journal,
    sha256_bytes,
    summarize,
    validate_distance_bands,
    validate_mesh,
    validate_overlay_matches_crop,
    validate_pair,
    validate_pose_graph,
    validate_registration_result,
    validate_result,
    validate_rigid_transform,
    validate_support,
)
from npa.workbench.open3d.schemas import (
    DEMO_DATA_RELEASE,
    SOURCE_VERSION,
    Fragment,
    ReconstructRequest,
    RegistrationManifest,
    StageDemoRequest,
    fragment_id_for,
)

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _rotation(degrees: float, shift: float = 0.0) -> list[list[float]]:
    angle = math.radians(degrees)
    return [
        [math.cos(angle), -math.sin(angle), 0.0, shift],
        [math.sin(angle), math.cos(angle), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _result(fitness: float = 0.7, rmse: float = 0.01, count: int = 500) -> dict:
    return {
        "fitness": fitness,
        "inlier_rmse": rmse,
        "correspondence_count": count,
    }


def _row(pair_id: str = "a__b", **overrides) -> dict:
    row = {
        "pair_id": pair_id,
        "source_id": "a",
        "target_id": "b",
        "source_points": 1000,
        "target_points": 900,
        "source_downsampled_points": 300,
        "target_downsampled_points": 280,
        "fpfh_dimension": 33,
        "global_registration": _result(0.6, 0.02, 400),
        "icp": _result(0.7, 0.01, 500),
        "transformation": _rotation(5.0, 0.1),
        "aligned_points": 1000,
        "aligned_sha256": "b" * 64,
        "wall_seconds": 1.5,
    }
    row.update(overrides)
    return row


def _fragment(name: str, digest: str = "a" * 64) -> dict:
    return {
        "id": name,
        "uri": f"s3://example-bucket/scans/{name}.ply",
        "sha256": digest,
        "bytes": 2048,
    }


# ------------------------------------------------------------------ transforms


def test_a_real_rotation_is_accepted() -> None:
    validate_rigid_transform(_rotation(30.0, 1.25), what="transform")


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(
            [[2.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]],
            id="uniform-scale",
        ),
        pytest.param(
            [[1.0, 0.4, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1]],
            id="shear",
        ),
        pytest.param(
            [[-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1]],
            id="reflection-det-minus-one",
        ),
        pytest.param(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0]],
            id="projective-bottom-row",
        ),
        pytest.param(
            [
                [float("nan"), 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1],
            ],
            id="not-finite",
        ),
        pytest.param([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], id="wrong-shape"),
    ],
)
def test_transforms_that_are_not_rigid_are_rejected(matrix) -> None:
    with pytest.raises(Open3dError):
        validate_rigid_transform(matrix, what="transform")


def test_booleans_are_not_accepted_as_matrix_numbers() -> None:
    """`True == 1` in Python, so a bool array would otherwise pass as identity."""

    with pytest.raises(Open3dError):
        validate_rigid_transform(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ],
            what="transform",
        )


# --------------------------------------------------------- registration results


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_result(fitness=1.4), id="fitness-above-one"),
        pytest.param(_result(fitness=-0.1), id="negative-fitness"),
        pytest.param(_result(rmse=-0.01), id="negative-rmse"),
        pytest.param(_result(count=0), id="no-correspondences"),
        pytest.param(_result(fitness=0.0, count=10), id="zero-fitness-with-matches"),
        pytest.param({"fitness": 0.5, "inlier_rmse": 0.01}, id="missing-field"),
    ],
)
def test_impossible_registration_results_are_rejected(result) -> None:
    with pytest.raises(Open3dError):
        validate_registration_result(result, what="result")


# ------------------------------------------------------------------- journal


def test_a_complete_pair_row_is_accepted() -> None:
    validate_pair(_row())


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"fpfh_dimension": 32}, id="not-33-bin-fpfh"),
        pytest.param(
            {"source_downsampled_points": 1001}, id="downsample-grew-the-cloud"
        ),
        pytest.param({"source_downsampled_points": 0}, id="downsample-emptied-cloud"),
        pytest.param({"source_points": 0}, id="empty-source"),
        pytest.param({"aligned_points": 999}, id="aligned-lost-a-point"),
        pytest.param({"aligned_sha256": "short"}, id="no-digest"),
        pytest.param({"wall_seconds": -1.0}, id="negative-duration"),
        pytest.param({"transformation": IDENTITY[:3]}, id="bad-transform"),
    ],
)
def test_incoherent_pair_rows_are_rejected(overrides) -> None:
    with pytest.raises(Open3dError):
        validate_pair(_row(**overrides))


def test_journal_rejects_duplicate_pair_identity(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n")
    with pytest.raises(Open3dError, match="duplicate pair identity"):
        read_journal(path)


def test_journal_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n\n")
    with pytest.raises(Open3dError, match="no fragment pairs"):
        read_journal(path)


def test_journal_rejects_unparseable_lines(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(Open3dError, match="invalid registration journal"):
        read_journal(path)


def test_summary_counts_only_pairs_icp_actually_improved() -> None:
    improved = _row(
        "a__b", global_registration=_result(rmse=0.05), icp=_result(rmse=0.01)
    )
    worsened = _row(
        "b__c", global_registration=_result(rmse=0.01), icp=_result(rmse=0.05)
    )
    summary = summarize([improved, worsened])
    assert summary["pair_count"] == 2
    assert summary["pairs_refined_by_icp"] == 1
    assert summary["input_points"] == 2000


def test_result_must_match_the_journal_it_describes() -> None:
    rows = [_row()]
    report = {
        "schema_version": "npa.open3d.result.v1",
        "engine": "open3d",
        "source_version": SOURCE_VERSION,
        "run_id": "run-1",
        "kind": "register",
        "summary": summarize(rows),
    }
    validate_result(report, rows, run_id="run-1", kind="register")

    # A summary that claims more pairs than the journal holds is the exact failure
    # a reviewer cannot see by reading either file alone.
    inflated = dict(report, summary=dict(report["summary"], pair_count=9))
    with pytest.raises(Open3dError):
        validate_result(inflated, rows, run_id="run-1", kind="register")
    with pytest.raises(Open3dError):
        validate_result(report, rows, run_id="other-run", kind="register")
    with pytest.raises(Open3dError):
        validate_result(report, rows, run_id="run-1", kind="multiway")
    with pytest.raises(Open3dError):
        validate_result(
            dict(report, source_version="0.19.0"), rows, run_id="run-1", kind="register"
        )


# ----------------------------------------------------------------- pose graph


def _graph(node_count: int = 3) -> dict:
    return {
        "nodes": [
            {
                "fragment_id": f"f{index}",
                "pose": IDENTITY if index == 0 else _rotation(3.0 * index),
            }
            for index in range(node_count)
        ],
        "edges": [
            {
                "source_node_id": index,
                "target_node_id": index + 1,
                "transformation": _rotation(2.0),
                "uncertain": False,
            }
            for index in range(node_count - 1)
        ],
    }


def test_a_well_formed_pose_graph_is_accepted() -> None:
    validate_pose_graph(_graph(), fragment_count=3)


def test_pose_graph_anchor_must_stay_at_the_identity() -> None:
    graph = _graph()
    graph["nodes"][0]["pose"] = _rotation(1.0)
    with pytest.raises(Open3dError, match="identity anchor"):
        validate_pose_graph(graph, fragment_count=3)


def test_pose_graph_must_hold_one_node_per_fragment() -> None:
    with pytest.raises(Open3dError, match="one node per fragment"):
        validate_pose_graph(_graph(2), fragment_count=3)


def test_pose_graph_must_connect_every_fragment() -> None:
    graph = _graph()
    graph["edges"] = graph["edges"][:1]
    with pytest.raises(Open3dError, match="connect every fragment"):
        validate_pose_graph(graph, fragment_count=3)


def test_pose_graph_edge_ids_must_be_in_range() -> None:
    graph = _graph()
    graph["edges"][0]["target_node_id"] = 7
    with pytest.raises(Open3dError, match="out of range"):
        validate_pose_graph(graph, fragment_count=3)


# ----------------------------------------------------------------------- mesh


def _mesh(**overrides) -> dict:
    mesh = {
        "vertex_count": 1200,
        "triangle_count": 2300,
        "surface_area": 14.5,
        "is_edge_manifold": False,
        "is_vertex_manifold": True,
        "is_watertight": False,
        "extent": [1.5, 2.0, 0.9],
        "sha256": "c" * 64,
    }
    mesh.update(overrides)
    return mesh


def test_a_partial_scan_surface_is_accepted_even_when_open() -> None:
    """Indoor scans reconstruct to open surfaces; that is a fact, not a failure."""

    validate_mesh(_mesh())


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"surface_area": 0.0}, id="vertices-but-no-area"),
        pytest.param({"triangle_count": 0}, id="no-triangles"),
        pytest.param({"extent": [1.0, 0.0, 1.0]}, id="degenerate-bounding-box"),
        pytest.param({"extent": [1.0, 2.0]}, id="two-dimensional-extent"),
        pytest.param({"is_watertight": "yes"}, id="stringly-typed-flag"),
        pytest.param({"sha256": ""}, id="no-digest"),
    ],
)
def test_meshes_that_are_not_surfaces_are_rejected(overrides) -> None:
    with pytest.raises(Open3dError):
        validate_mesh(_mesh(**overrides))


# ------------------------------------------------------- reconstruction support


def _support_report(**overrides) -> dict:
    """A coherent reconstruction report: half the area cropped, coverage intact."""

    report = {
        "support_distance_factor": 1.0,
        "support_before_crop": {
            "voxel_size": 0.05,
            "unsupported_area_fraction": 0.527,
            "unsupported_area": 11.72,
            "max_vertex_distance_to_sample": 0.905,
            "median_vertex_distance_to_sample": 0.0218,
            "p95_vertex_distance_to_sample": 0.218,
        },
        "support": {
            "voxel_size": 0.05,
            "unsupported_area_fraction": 0.0,
            "unsupported_area": 0.0,
            "max_vertex_distance_to_sample": 0.0499,
            "median_vertex_distance_to_sample": 0.0203,
            "p95_vertex_distance_to_sample": 0.0416,
        },
        "coverage": {
            "samples": 6340,
            "fraction_within_voxel": 0.9924,
            "fraction_within_two_voxels": 0.9996,
            "sample_to_surface_rmse": 0.0108,
            "sample_to_surface_p95": 0.0223,
        },
    }
    report.update(overrides)
    return report


def test_a_crop_that_removes_unsupported_area_is_accepted() -> None:
    """Poisson closes a partial scan; removing that shell is the intended result."""

    validate_support(_support_report(), voxel_size=0.05)


def test_an_uncropped_reconstruction_is_accepted_with_its_shell_declared() -> None:
    """Factor 0 publishes the closed surface, so its unsupported area must stand."""

    report = _support_report(support_distance_factor=0.0)
    report["support"] = dict(report["support_before_crop"])
    validate_support(report, voxel_size=0.05)


def test_cropping_cannot_report_more_unsupported_area_than_it_started_with() -> None:
    """Removing triangles can only shrink the unsupported area; arithmetic says so."""

    report = _support_report()
    report["support"]["unsupported_area_fraction"] = 0.6
    with pytest.raises(Open3dError, match="more unsupported area"):
        validate_support(report, voxel_size=0.05)


def test_surface_left_beyond_the_support_limit_is_rejected() -> None:
    """The reported limit and the reported distances have to agree."""

    report = _support_report()
    report["support"]["max_vertex_distance_to_sample"] = 0.4
    with pytest.raises(Open3dError, match="farther from the scan"):
        validate_support(report, voxel_size=0.05)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.pop("support"), id="no-support-measurement"),
        pytest.param(lambda r: r.pop("coverage"), id="no-coverage-measurement"),
        pytest.param(lambda r: r.pop("support_before_crop"), id="no-baseline"),
        pytest.param(
            lambda r: r["coverage"].update(samples=0), id="coverage-measured-nothing"
        ),
        pytest.param(
            lambda r: r["coverage"].update(samples=True), id="sample-count-is-a-bool"
        ),
        pytest.param(
            lambda r: r["coverage"].update(fraction_within_voxel=1.4),
            id="coverage-over-one",
        ),
        pytest.param(
            lambda r: r["coverage"].update(sample_to_surface_rmse=-0.01),
            id="negative-rmse",
        ),
        pytest.param(
            lambda r: r["coverage"].update(sample_to_surface_rmse=float("nan")),
            id="rmse-is-nan",
        ),
        pytest.param(
            lambda r: r["support"].update(unsupported_area_fraction=-0.1),
            id="negative-unsupported-fraction",
        ),
        pytest.param(
            lambda r: r.update(support_distance_factor=-1.0), id="negative-factor"
        ),
    ],
)
def test_incoherent_support_reports_are_rejected(mutate) -> None:
    report = _support_report()
    mutate(report)
    with pytest.raises(Open3dError):
        validate_support(report, voxel_size=0.05)


def test_support_report_must_be_an_object() -> None:
    with pytest.raises(Open3dError):
        validate_support(["support"], voxel_size=0.05)


def test_support_distance_factor_defaults_to_one_voxel() -> None:
    """The sampling scale, not a tuned number: past a voxel there is no sample."""

    assert (
        ReconstructRequest(
            input_path="s3://bucket/graph",
            output_path="s3://bucket/surface",
            run_id="run",
        ).support_distance_factor
        == 1.0
    )


def test_support_distance_factor_can_be_disabled_but_not_negative() -> None:
    kwargs = {
        "input_path": "s3://bucket/graph",
        "output_path": "s3://bucket/surface",
        "run_id": "run",
    }
    assert (
        ReconstructRequest(
            support_distance_factor=0.0, **kwargs
        ).support_distance_factor
        == 0.0
    )
    with pytest.raises(ValidationError):
        ReconstructRequest(support_distance_factor=-0.5, **kwargs)


# -------------------------------------------------------------------- schemas


def test_fragment_uri_must_be_a_readable_point_cloud() -> None:
    with pytest.raises(ValueError, match="fragment uri must end in"):
        Fragment.model_validate(
            {"id": "a", "uri": "s3://b/a.npy", "sha256": "a" * 64, "bytes": 10}
        )


def test_manifest_rejects_repeated_fragments() -> None:
    same_id = [_fragment("a"), dict(_fragment("b"), id="a")]
    with pytest.raises(ValueError, match="ids must be unique"):
        RegistrationManifest.model_validate({"fragments": same_id, "voxel_size": 0.05})
    same_uri = [_fragment("a"), dict(_fragment("b"), uri=_fragment("a")["uri"])]
    with pytest.raises(ValueError, match="uris must be unique"):
        RegistrationManifest.model_validate({"fragments": same_uri, "voxel_size": 0.05})


def test_manifest_needs_two_fragments_to_register_anything() -> None:
    with pytest.raises(ValueError):
        RegistrationManifest.model_validate(
            {"fragments": [_fragment("a")], "voxel_size": 0.05}
        )


@pytest.mark.parametrize("voxel", [0.0, -1.0, 11.0])
def test_manifest_voxel_size_must_be_a_usable_scale(voxel) -> None:
    with pytest.raises(ValueError):
        RegistrationManifest.model_validate(
            {"fragments": [_fragment("a"), _fragment("b")], "voxel_size": voxel}
        )


def test_manifest_rejects_an_unreviewed_demo_release() -> None:
    with pytest.raises(ValueError, match="demo data release"):
        RegistrationManifest.model_validate(
            {
                "fragments": [_fragment("a"), _fragment("b")],
                "voxel_size": 0.05,
                "demo_data_release": "20990101-data",
            }
        )
    RegistrationManifest.model_validate(
        {
            "fragments": [_fragment("a"), _fragment("b")],
            "voxel_size": 0.05,
            "demo_data_release": DEMO_DATA_RELEASE,
        }
    )


def test_requests_reject_unknown_keys() -> None:
    """A typo'd flag must fail loudly rather than being silently ignored."""

    with pytest.raises(ValueError):
        StageDemoRequest.model_validate(
            {"output_path": "s3://b/out", "voxelsize": 0.05}
        )


def test_reconstruct_bounds_the_poisson_lattice() -> None:
    with pytest.raises(ValueError):
        ReconstructRequest.model_validate(
            {
                "input_path": "s3://b/in",
                "output_path": "s3://b/out",
                "run_id": "r",
                "poisson_depth": 14,
            }
        )
    with pytest.raises(ValueError):
        ReconstructRequest.model_validate(
            {
                "input_path": "s3://b/in",
                "output_path": "s3://b/out",
                "run_id": "r",
                "density_quantile": 0.9,
            }
        )


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/scans/cloud_bin_0.pcd", "cloud_bin_0"),
        ("s3://bucket/scans/Scan 01.PLY", "Scan_01"),
        ("/local/path/room-a.ply", "room-a"),
    ],
)
def test_fragment_ids_come_from_the_uri(uri, expected) -> None:
    assert fragment_id_for(uri) == expected


def test_fragment_id_refuses_a_nameless_uri() -> None:
    with pytest.raises(ValueError):
        fragment_id_for("s3://bucket/__.ply")


# -------------------------------------------------------------------- runtime


def test_publish_fails_when_storage_reads_back_different_bytes(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "write_bytes_uri", lambda uri, payload: None)
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: b"truncated")
    with pytest.raises(Open3dError, match="read-after-write"):
        runtime._publish("s3://example-bucket/out/result.json", b"the real payload")


def test_publish_accepts_bytes_that_survive_the_round_trip(monkeypatch) -> None:
    stored: dict[str, bytes] = {}
    monkeypatch.setattr(
        runtime, "write_bytes_uri", lambda uri, payload: stored.setdefault(uri, payload)
    )
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: stored[uri])
    runtime._publish("s3://example-bucket/out/result.json", b"payload")
    assert stored["s3://example-bucket/out/result.json"] == b"payload"


def test_download_refuses_a_scan_that_changed_since_prepare(
    monkeypatch, tmp_path
) -> None:
    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"first")),
                _fragment("b", sha256_bytes(b"second")),
            ],
            "voxel_size": 0.05,
        }
    )
    calls: list[str] = []

    def rewritten(uri: str) -> bytes:
        calls.append(uri)
        return b"rewritten"

    monkeypatch.setattr(runtime, "read_bytes_uri", rewritten)
    with pytest.raises(Open3dError, match="the manifest and the stored scan disagree"):
        runtime._download_fragments(manifest, tmp_path)
    # A store that really has diverged must stop, not retry forever.
    assert len(calls) == runtime.FRAGMENT_READ_ATTEMPTS


def test_download_retries_a_truncated_read_rather_than_losing_the_stage(
    monkeypatch, tmp_path
) -> None:
    """A live `register` stage died here while `multiway` read the same scans.

    The stream ended early; the stored object was fine. Retrying is what the
    fragment digest makes safe, so the stage should survive it.
    """

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"scan-a")),
                _fragment("b", sha256_bytes(b"scan-b")),
            ],
            "voxel_size": 0.05,
        }
    )
    reads: list[str] = []

    def truncate_the_first_read(uri: str) -> bytes:
        reads.append(uri)
        if len(reads) == 1:
            raise OSError("IncompleteRead(5848412 bytes read, 276537 more expected)")
        return b"scan-a" if uri.endswith("a.ply") else b"scan-b"

    monkeypatch.setattr(runtime, "read_bytes_uri", truncate_the_first_read)
    paths = runtime._download_fragments(manifest, tmp_path)
    assert len(reads) == 3  # a truncated, a again, then b
    assert Path(paths["a"]).read_bytes() == b"scan-a"
    assert Path(paths["b"]).read_bytes() == b"scan-b"


def test_download_retries_bytes_that_arrive_short_without_raising(
    monkeypatch, tmp_path
) -> None:
    """Silently short bytes are the dangerous case: no exception, wrong geometry.

    Registering a truncated point cloud would produce a plausible transform over
    the wrong input, so the digest must reject it and the retry must recover.
    """

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"the whole cloud")),
                _fragment("b", sha256_bytes(b"the other cloud")),
            ],
            "voxel_size": 0.05,
        }
    )
    responses = [b"the whole", b"the whole cloud", b"the other cloud"]
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: responses.pop(0))
    paths = runtime._download_fragments(manifest, tmp_path)
    assert responses == []
    assert Path(paths["a"]).read_bytes() == b"the whole cloud"
    assert Path(paths["b"]).read_bytes() == b"the other cloud"


def test_download_names_every_failed_attempt_in_the_error(
    monkeypatch, tmp_path
) -> None:
    """An operator triaging a flaky store needs to see what each attempt did."""

    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"scan-a")),
                _fragment("b", sha256_bytes(b"scan-b")),
            ],
            "voxel_size": 0.05,
        }
    )
    responses: list[bytes | Exception] = [
        OSError("IncompleteRead(10 bytes read, 5 more expected)"),
        b"short",
        OSError("connection reset by peer"),
    ]

    def flaky(uri: str) -> bytes:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(runtime, "read_bytes_uri", flaky)
    with pytest.raises(Open3dError) as failure:
        runtime._download_fragments(manifest, tmp_path)
    message = str(failure.value)
    assert "attempt 1: OSError: IncompleteRead" in message
    assert "attempt 2: digest mismatch (5 bytes read)" in message
    assert "attempt 3: OSError: connection reset by peer" in message


def test_download_keeps_the_readable_suffix(monkeypatch, tmp_path) -> None:
    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"cloud")),
                _fragment("b", sha256_bytes(b"cloud")),
            ],
            "voxel_size": 0.05,
        }
    )
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: b"cloud")
    paths = runtime._download_fragments(manifest, tmp_path)
    assert {Path(path).name for path in paths.values()} == {"a.ply", "b.ply"}


def test_expected_pairs_differ_between_consecutive_and_full_graphs() -> None:
    manifest = RegistrationManifest.model_validate(
        {
            "fragments": [_fragment("a"), _fragment("b"), _fragment("c")],
            "voxel_size": 0.05,
        }
    )
    assert runtime._expected_pairs("register", manifest) == ["a__b", "b__c"]
    assert runtime._expected_pairs("multiway", manifest) == ["a__b", "a__c", "b__c"]


def test_reconstruct_refuses_a_pairwise_prefix(monkeypatch) -> None:
    """`register` writes no fused cloud, so pointing reconstruct at it is a setup bug."""

    monkeypatch.setattr(runtime, "validate_read_path", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "validate_write_path", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "authorize_uri", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: canonical({"kind": "register"})
    )
    with pytest.raises(Open3dError, match="consumes a multiway output prefix"):
        runtime.reconstruct(
            ReconstructRequest(
                input_path="s3://example-bucket/pairs",
                output_path="s3://example-bucket/surface",
                run_id="r",
            )
        )


def test_visualize_refuses_a_prefix_without_a_reconstruction(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "validate_read_path", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "validate_write_path", lambda *a, **k: None)
    monkeypatch.setattr(runtime, "authorize_uri", lambda *a, **k: None)
    monkeypatch.setattr(
        runtime,
        "read_bytes_uri",
        lambda uri: canonical({"schema_version": "npa.open3d.result.v1"}),
    )
    with pytest.raises(Open3dError, match="reconstruct output prefix"):
        runtime.visualize(
            runtime.RunRequest(
                input_path="s3://example-bucket/graph",
                output_path="s3://example-bucket/reports",
                run_id="r",
            )
        )


def test_manifest_read_error_names_the_command_that_writes_it(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "read_bytes_uri", lambda uri: b"<html>404</html>")
    with pytest.raises(Open3dError, match="open3d prepare"):
        runtime._load_manifest("s3://example-bucket/prepared")


def test_manifest_read_retries_a_truncated_stream(monkeypatch) -> None:
    """Fault injection in the built image found this gap the unit mocks missed.

    A truncated read of the manifest ended a `register` stage before it started;
    only the fragment reads were being retried.
    """

    good = json.dumps(
        {
            "fragments": [
                _fragment("a", sha256_bytes(b"scan-a")),
                _fragment("b", sha256_bytes(b"scan-b")),
            ],
            "voxel_size": 0.05,
        }
    ).encode()
    responses: list[bytes | Exception] = [
        OSError("IncompleteRead(5848412 bytes read, 276537 more expected)"),
        good[: len(good) // 2],  # truncated JSON does not close
        good,
    ]

    def flaky(uri: str) -> bytes:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(runtime, "read_bytes_uri", flaky)
    manifest = runtime._load_manifest("s3://example-bucket/prepared")
    assert responses == []
    assert [fragment.id for fragment in manifest.fragments] == ["a", "b"]


def test_manifest_read_stops_after_bounded_attempts(monkeypatch) -> None:
    reads: list[str] = []

    def always_bad(uri: str) -> bytes:
        reads.append(uri)
        return b"{"

    monkeypatch.setattr(runtime, "read_bytes_uri", always_bad)
    with pytest.raises(Open3dError, match="after 3 attempts"):
        runtime._load_manifest("s3://example-bucket/prepared")
    assert len(reads) == runtime.FRAGMENT_READ_ATTEMPTS


def test_runner_failure_surfaces_the_upstream_log_tail(tmp_path, monkeypatch) -> None:
    """Triage should not need the pod back to learn why the stage died."""

    def fake_run(command, cwd, stdout, stderr, check):
        stdout.write(b"open3d: KDTree radius must be positive\n")

        class Completed:
            returncode = 3

        return Completed()

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    with pytest.raises(Open3dError, match="KDTree radius must be positive"):
        runtime._run_runner("register", {}, tmp_path, "run-1")


def _visualize_payload(monkeypatch, tmp_path, *, factor, cropped) -> dict:
    """Drive `visualize` far enough to capture the payload it hands the runner."""

    manifest = {
        "fragments": [
            _fragment("a", sha256_bytes(b"scan-a")),
            _fragment("b", sha256_bytes(b"scan-b")),
        ],
        "voxel_size": 0.05,
    }
    documents = {
        "s3://example-bucket/mesh/result.json": {
            "schema_version": "npa.open3d.reconstruction.v1",
            "registration_path": "s3://example-bucket/graph",
            "support_distance_factor": factor,
            "unsupported_vertices_removed": cropped,
        },
        "s3://example-bucket/graph/result.json": {
            "input_path": "s3://example-bucket/prepared"
        },
        "s3://example-bucket/graph/pose_graph.json": {"nodes": []},
        "s3://example-bucket/prepared/manifest.json": manifest,
    }

    def read(uri: str) -> bytes:
        if uri in documents:
            return canonical(documents[uri])
        if uri.endswith("a.ply"):
            return b"scan-a"
        if uri.endswith("b.ply"):
            return b"scan-b"
        return b"artifact-bytes"

    captured: dict = {}

    def capture(kind, payload, root, run_id):
        captured.update(payload)
        raise Open3dError("stop after the payload is built")

    for name in ("validate_read_path", "validate_write_path", "authorize_uri"):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)
    monkeypatch.setattr(runtime, "read_bytes_uri", read)
    monkeypatch.setattr(runtime, "_run_runner", capture)
    with pytest.raises(Open3dError, match="stop after the payload"):
        runtime.visualize(
            runtime.RunRequest(
                input_path="s3://example-bucket/mesh",
                output_path="s3://example-bucket/reports",
                run_id="r",
            )
        )
    return captured


def test_visualize_tells_the_runner_which_threshold_the_crop_used(
    monkeypatch, tmp_path
) -> None:
    """The overlay must redraw the crop that ran, not the default.

    Without the factor the recording would paint surface between one and two
    voxels of a sample as removed at factor 2.0, while `mesh.ply` kept it: the
    right view wrong and every published number still right.
    """

    payload = _visualize_payload(monkeypatch, tmp_path, factor=2.0, cropped=3677)
    assert payload["support_distance_factor"] == 2.0
    assert payload["unsupported_vertices_removed"] == 3677


def test_visualize_passes_a_disabled_crop_through_as_disabled(
    monkeypatch, tmp_path
) -> None:
    """`mesh_uncropped.ply` is written unconditionally, so its path proves nothing.

    A factor-0 run must not draw a "removed" overlay for surface it kept.
    """

    payload = _visualize_payload(monkeypatch, tmp_path, factor=0.0, cropped=0)
    assert payload["support_distance_factor"] == 0.0
    assert payload["unsupported_vertices_removed"] == 0
    assert payload["uncropped_mesh_path"]


def _crop_reports(*, shown, factor=1.0, cropped=3677, kept=28208, full=36660):
    return (
        {"unsupported_triangles_shown": shown},
        {
            "support_distance_factor": factor,
            "unsupported_vertices_removed": cropped,
            "mesh": {"triangle_count": kept},
            "mesh_uncropped": {"triangle_count": full},
        },
    )


def test_overlay_must_show_exactly_the_triangles_the_crop_removed() -> None:
    recording, reconstruction = _crop_reports(shown=36660 - 28208)
    validate_overlay_matches_crop(recording, reconstruction)


def test_an_overlay_at_a_different_threshold_is_caught_by_arithmetic() -> None:
    """The failure this exists for: right numbers, wrong view.

    A bare-voxel threshold against a factor-2.0 crop draws more triangles than the
    crop removed, and nothing else in the report changes.
    """

    recording, reconstruction = _crop_reports(shown=12000, factor=2.0)
    with pytest.raises(Open3dError, match="not showing the crop that ran"):
        validate_overlay_matches_crop(recording, reconstruction)


def test_a_run_that_cropped_nothing_must_draw_nothing() -> None:
    recording, reconstruction = _crop_reports(shown=8452, factor=0.0, cropped=0)
    with pytest.raises(Open3dError, match="cropped nothing"):
        validate_overlay_matches_crop(recording, reconstruction)
    recording, reconstruction = _crop_reports(shown=0, factor=0.0, cropped=0)
    validate_overlay_matches_crop(recording, reconstruction)


def test_overlay_check_requires_the_counts_it_compares() -> None:
    with pytest.raises(Open3dError, match="how many unsupported triangles"):
        validate_overlay_matches_crop({}, {"support_distance_factor": 1.0})
    with pytest.raises(Open3dError, match="support_distance_factor"):
        validate_overlay_matches_crop({"unsupported_triangles_shown": 0}, {})


def _support_block(*, unsupported, beyond_1_5, beyond_3):
    return {
        "voxel_size": 0.05,
        "unsupported_area_fraction": unsupported,
        "unsupported_area_beyond_1_5_voxels": beyond_1_5,
        "unsupported_area_beyond_3_voxels": beyond_3,
        "max_vertex_distance_to_sample": 0.9,
        "median_vertex_distance_to_sample": 0.01,
        "p95_vertex_distance_to_sample": 0.2,
    }


def test_nested_distance_bands_must_decrease() -> None:
    validate_distance_bands(
        _support_block(unsupported=0.527324, beyond_1_5=0.484153, beyond_3=0.384809)
    )
    with pytest.raises(Open3dError, match="the further band is a subset"):
        validate_distance_bands(
            _support_block(unsupported=0.527324, beyond_1_5=0.3, beyond_3=0.4)
        )


def test_a_report_written_before_the_bands_existed_still_validates() -> None:
    """`validate` re-verifies already-published reports, which predate these keys."""

    older = _support_block(unsupported=0.527324, beyond_1_5=0.0, beyond_3=0.0)
    del older["unsupported_area_beyond_1_5_voxels"]
    del older["unsupported_area_beyond_3_voxels"]
    validate_distance_bands(older)


def test_a_band_that_is_not_a_fraction_is_rejected() -> None:
    for bad in (1.4, -0.1, "0.2", True, None):
        block = _support_block(unsupported=0.2, beyond_1_5=0.1, beyond_3=0.05)
        block["unsupported_area_beyond_3_voxels"] = bad
        with pytest.raises(Open3dError, match="must be a fraction"):
            validate_distance_bands(block)


def test_an_extrapolated_shell_and_near_threshold_surface_read_differently() -> None:
    """The distinction coverage cannot make, from the numbers that can.

    Both profiles below are measured. The demo scans leave 0.3848 of all surface area
    further than three voxels from any sample, which sample spacing cannot explain. A
    watertight mesh sampled at roughly one voxel spacing leaves exactly none there, and
    no vertex further than one voxel from ground truth — yet still reports 0.2055
    unsupported, a fifth of its area, from discretization alone.

    A headline fraction of 0.2055 on correct geometry against 0.5273 on an invented
    shell is a difference of degree that a threshold could not safely split. The share
    past three voxels, 0.0 against 0.7297, is the separation this reads instead.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(14736)

    shell = _crop_justification(
        _support_block(unsupported=0.527324, beyond_1_5=0.484153, beyond_3=0.384809),
        3677,
        _Mesh(),
    )
    assert shell["removed_surface_reads_as"] == "extrapolated shell"
    assert "invented surface" in shell["note"]

    discretization = _crop_justification(
        _support_block(unsupported=0.205530, beyond_1_5=0.007524, beyond_3=0.0),
        44230,
        _Mesh(),
    )
    assert discretization["removed_surface_reads_as"] == "near-threshold surface"
    assert "coverage cannot rule that out" in discretization["note"]
    assert "1.5" in discretization["note"]


def test_crop_justification_is_silent_when_nothing_was_unsupported() -> None:
    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(100)

    result = _crop_justification(
        _support_block(unsupported=0.0, beyond_1_5=0.0, beyond_3=0.0), 0, _Mesh()
    )
    assert result["unsupported_area_share_beyond_3_voxels"] == 0.0
    assert result["vertex_fraction_removed"] == 0.0


def test_a_real_scan_sits_at_the_sensitivity_floor_not_on_an_arbitrary_line() -> None:
    """Why the real scan lands where it does, which is not a coin toss.

    A real scan of a solid object with modest real holes measured 0.1012 past three voxels
    as a share of its unsupported area, against a boundary of 0.1. That looked like an
    awkward coincidence until the review lane produced a sensitivity curve: the share is
    monotone in the invented fraction, and the detector's floor sits near 9 percent
    invented area. A scan of a solid object with modest holes belongs at that floor.

    Pinned because nothing gates on the reading, and a later change must not quietly start
    treating a value this close to the floor as decisive.
    """

    from npa.workbench.open3d.runner import (
        FABRICATION_AREA_SHARE,
        _crop_justification,
    )

    class _Mesh:
        vertices = range(100000)

    eagle = _crop_justification(
        _support_block(unsupported=0.1249, beyond_1_5=0.0642, beyond_3=0.0126),
        1000,
        _Mesh(),
    )
    share = eagle["unsupported_area_share_beyond_3_voxels"]
    assert abs(share - 0.1009) < 0.001
    assert abs(share - FABRICATION_AREA_SHARE) / FABRICATION_AREA_SHARE < 0.02
    # Sitting on the boundary, the artifact must say so rather than assert a verdict.
    assert eagle["removed_surface_reads_as"] == "undecided"


def test_the_case_where_the_headline_metric_overstates_fabrication_200_fold() -> None:
    """The strongest measured case for reading the bands instead of the fraction.

    A watertight mesh observed at 0.708 of a voxel reports 0.5659 unsupported area — more
    than half its surface — while only 0.0027 of its area is actually further than one
    voxel from the true surface. The headline number overstates fabrication by a factor
    of roughly 200. The bands put 0.0003 past three voxels, and the reading stays
    near-threshold, which is correct.

    Figures from evidence/open3d/band-validity-sweep.json.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(50000)

    reading = _crop_justification(
        _support_block(unsupported=0.5659, beyond_1_5=0.0180, beyond_3=0.0003),
        0,
        _Mesh(),
    )
    assert reading["removed_surface_reads_as"] == "near-threshold surface"
    assert reading["unsupported_area_share_beyond_3_voxels"] < 0.001


def test_the_undecided_band_covers_most_of_the_sensitivity_floor() -> None:
    """Where the reading speaks, and where it now declines to.

    The review lane's polar-cap curve, with the invented fraction known exactly from the
    cap angle. Before the undecided band this field gave a confident verdict at every one
    of these points, including the two sitting on its own sensitivity floor, where its
    verdict was a coin toss. Now it declines there.

    What remains confidently wrong is narrower and worth naming: a share under 0.05, which
    on this curve is about 3 percent invented area, still reads `near-threshold surface`.
    That is why the note for that branch calls itself a lower bound rather than a clean
    bill of health. The field cannot be made to see fabrication below its floor; it can
    only be stopped from denying it.

    Figures: review lane `program/review/evidence/verify_602_boundary.py`.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(100000)

    def reads(unsupported: float, beyond_3: float) -> dict:
        return _crop_justification(
            _support_block(
                unsupported=unsupported, beyond_1_5=beyond_3 * 2, beyond_3=beyond_3
            ),
            0,
            _Mesh(),
        )

    # invented fraction -> (unsupported, past 3v) -> expected reading
    curve = [
        (0.000, 0.75126, 0.00973, "near-threshold surface"),
        (0.030, 0.76106, 0.03149, "near-threshold surface"),
        (0.067, 0.77010, 0.06375, "undecided"),
        (0.090, 0.77602, 0.08187, "undecided"),
        (0.117, 0.78417, 0.11002, "undecided"),
        (0.250, 0.81944, 0.23530, "extrapolated shell"),
        (0.500, 0.87790, 0.48395, "extrapolated shell"),
    ]
    for invented, unsupported, beyond_3, expected in curve:
        result = reads(unsupported, beyond_3)
        assert result["removed_surface_reads_as"] == expected, (
            f"{invented:.1%} invented area read as "
            f"{result['removed_surface_reads_as']!r}, expected {expected!r}"
        )

    # The one remaining confident miss names itself as a lower bound.
    missed = reads(0.76106, 0.03149)
    assert "lower bound and not a clean bill of health" in missed["note"]

    # The undecided note must send the reader to the bands, not to a verdict.
    undecided = reads(0.77602, 0.08187)
    assert "unsupported_area_beyond_3_voxels" in undecided["note"]
    assert "no reading here worth acting on" in undecided["note"]


def test_the_floor_is_setup_dependent_so_no_test_pins_it_to_a_number() -> None:
    """What survived varying geometry and sampling, and what did not.

    Across two geometries and two sampling schemes the floor ranged from 0.011 to 0.037
    invented area, against 0.09 in an independent setup at a different resolution ratio.
    Nearly an order of magnitude, so a test asserting a floor value would be asserting a
    property of one setup.

    What held in every case is what this pins instead: a perfect reconstruction never reads
    as a shell, whatever its headline fraction. That number ranged from exactly 0.0 under
    Poisson-disk sampling to 0.7513 under uniform random, and the reading was correct at
    both ends — which is the whole argument for reading the bands rather than the fraction.

    Figures: evidence/open3d/floor-across-geometry-and-sampling.json.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(10000)

    for unsupported, beyond_3 in (
        (0.0, 0.0),  # icosphere and armadillo, Poisson-disk
        (0.1393, 0.0),  # armadillo, uniform random
        (0.16362554224511913, 0.0),  # icosphere, uniform random
        (0.75126, 0.00973),  # review lane's icosphere at its own resolution
    ):
        perfect = _crop_justification(
            _support_block(
                unsupported=unsupported, beyond_1_5=beyond_3, beyond_3=beyond_3
            ),
            0,
            _Mesh(),
        )
        assert perfect["removed_surface_reads_as"] == "near-threshold surface", (
            f"a zero-error reconstruction reporting {unsupported} unsupported must not "
            "read as an extrapolated shell"
        )
