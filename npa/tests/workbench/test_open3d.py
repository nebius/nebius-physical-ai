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


def test_far_and_near_threshold_surface_read_differently() -> None:
    """The distinction coverage cannot make, from the numbers that can.

    Both profiles below are measured. The demo scans leave 0.3848 of all surface area
    further than three voxels from any sample, which sample spacing cannot explain. A
    watertight mesh sampled at roughly one voxel spacing leaves exactly none there, and
    no vertex further than one voxel from ground truth — yet still reports 0.2055
    unsupported, a fifth of its area, from discretization alone.

    A headline fraction of 0.2055 on correct geometry against 0.5273 on the demo scans
    is a difference of degree that a threshold could not safely split. The share past
    three voxels, 0.0 against 0.7297, is the separation this reads instead.

    Which side a reading falls on is a statement about distance from observations, not
    about ground truth; the far branch must not claim otherwise.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(14736)

    far = _crop_justification(
        _support_block(unsupported=0.527324, beyond_1_5=0.484153, beyond_3=0.384809),
        3677,
        _Mesh(),
    )
    assert far["removed_surface_reads_as"] == "far from any observation"
    assert "separate question this cannot answer" in far["note"]
    assert "invented surface" not in far["note"]

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
        FAR_BAND_AREA_SHARE,
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
    assert abs(share - FAR_BAND_AREA_SHARE) / FAR_BAND_AREA_SHARE < 0.02
    assert eagle["removed_surface_reads_as"] == "far from any observation"


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


def test_the_reading_rises_with_invented_area_but_has_a_floor_beneath_it() -> None:
    """Monotone in invented area at fixed sampling, with a floor it cannot see beneath.

    This is one of the two error directions. Below a floor, a genuine fabrication reads as
    near-threshold -- not undecided but confidently wrong -- which is why that branch's note
    refuses to call itself a clean bill of health.

    An earlier revision of this test asserted the *other* direction could not happen, and
    concluded the error was one-sided and so the far branch could assert fabrication as flat
    fact. That conclusion was wrong and the far branch no longer makes it; see
    `test_uneven_coverage_of_correct_geometry_reads_as_far_from_observations`. The curve
    below is unaltered -- it is a real measurement and still shows what it showed.

    The curve is the review lane's, from `program/review/evidence/verify_602_boundary.py`, so
    this test pins their measurement rather than independently confirming it. That is a
    circularity they raised themselves. What is worth pinning is the *shape* -- monotone and
    floored -- not the floor's value, which moves with the voxel-to-spacing ratio.
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
        (0.067, 0.77010, 0.06375, "near-threshold surface"),
        (0.090, 0.77602, 0.08187, "far from any observation"),
        (0.250, 0.81944, 0.23530, "far from any observation"),
        (0.500, 0.87790, 0.48395, "far from any observation"),
    ]
    for invented, unsupported, beyond_3, expected in curve:
        result = reads(unsupported, beyond_3)
        assert result["removed_surface_reads_as"] == expected, (
            f"{invented:.1%} invented area read as "
            f"{result['removed_surface_reads_as']!r}, expected {expected!r}"
        )

    # The share must rise with invented area, or no floor is well defined at all.
    shares = [
        reads(u, b)["unsupported_area_share_beyond_3_voxels"] for _, u, b, _ in curve
    ]
    assert shares == sorted(shares)

    # Each branch must carry its own basis, since a docstring does not travel with the JSON.
    assert "separate question this cannot answer" in reads(0.87790, 0.48395)["note"]
    assert "not a clean bill of health" in reads(0.76106, 0.03149)["note"]


def test_evenly_sampled_correct_reconstructions_read_near_threshold() -> None:
    """Zero-error reconstructions that are sampled *evenly* stay on the near side.

    Twelve of them -- the surface scored is the surface sampled, so there is nothing to
    invent -- across two geometries, two sampling schemes, and voxel-to-spacing ratios from
    1 to 3. The measurements are unaltered and still hold.

    What has changed is what they are taken to mean. They were read as proof the far branch
    could not fire on correct geometry; they are not, because every one of them samples
    convex geometry evenly over the whole surface. Vary the *density* across the surface
    instead and the far branch does fire on correct geometry, which is the test below this
    one. The name of this test used to claim the general property; it now states the
    condition under which it was measured.

    Figures: `evidence/open3d/poisson-procedure-and-overcall-audit.json`.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(100000)

    # (case, voxel/median-nn, unsupported, past 3v) with zero invented area by construction
    zero_error = [
        ("icosphere/uniform", 1.0, 0.85509, 0.00911),
        ("icosphere/uniform", 2.0, 0.16061, 0.00000),
        ("icosphere/uniform", 3.0, 0.00000, 0.00000),
        ("icosphere/poisson", 1.0, 0.02695, 0.00000),
        ("icosphere/poisson", 2.0, 0.00000, 0.00000),
        ("armadillo/uniform", 1.0, 0.83031, 0.00508),
        ("armadillo/uniform", 2.0, 0.14497, 0.00000),
        ("armadillo/poisson", 1.0, 0.03434, 0.00000),
        ("armadillo/poisson", 2.0, 0.00000, 0.00000),
    ]
    for case, multiple, unsupported, beyond_3 in zero_error:
        result = _crop_justification(
            _support_block(
                unsupported=unsupported, beyond_1_5=beyond_3, beyond_3=beyond_3
            ),
            0,
            _Mesh(),
        )
        assert result["removed_surface_reads_as"] == "near-threshold surface", (
            f"{case} at voxel {multiple}x median spacing invented nothing but read as "
            f"{result['removed_surface_reads_as']!r}"
        )


def test_uneven_coverage_of_correct_geometry_reads_as_far_from_observations() -> None:
    """The over-call direction, on geometry that is exactly its own ground truth.

    A closed unit cube, 9602 vertices over 19200 triangles, every vertex lying on the true
    cube to the bit. Observations are drawn on that same cube, so there is no error anywhere
    to find -- ground-truth distance is identically zero for every vertex and every triangle.
    The only thing that varies is how evenly those observations cover the six faces, and the
    voxel follows the rule the module recommends, twice the global median sample spacing.

    Sampled evenly the reading is near-threshold, as it should be. Sampled densely on five
    faces and sparsely on the sixth it goes to the far side: 0.4169 of the unsupported area
    past three voxels, on a surface with nothing invented in it. So the far reading cannot
    mean fabrication, and the note must not say it does.

    This came from an independent audit that reproduced the same effect on a cKDTree
    adapter, and is reproduced here through the shipped functions and real Open3D. Figures
    and the generator: `evidence/open3d/irregular-density-counterexample.json`.

    The numbers below are that reproduction, pinned so the semantics cannot quietly revert
    to asserting fabrication. The geometry itself is rebuilt in the functional smoke, where
    Open3D is available.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(9602)

    even = _crop_justification(
        _support_block(unsupported=0.192187, beyond_1_5=0.0, beyond_3=0.0), 0, _Mesh()
    )
    assert even["removed_surface_reads_as"] == "near-threshold surface"

    uneven = _crop_justification(
        _support_block(unsupported=0.315729, beyond_1_5=0.185, beyond_3=0.131615),
        0,
        _Mesh(),
    )
    assert uneven["removed_surface_reads_as"] == "far from any observation"
    assert abs(uneven["unsupported_area_share_beyond_3_voxels"] - 0.4169) < 0.001

    # The note may not assert fabrication, and must name the confound that caused this.
    note = uneven["note"]
    assert "invented surface" not in note
    assert "separate question this cannot answer" in note
    assert "sparsely covered" in note


def test_a_tight_voxel_costs_both_ways_at_once() -> None:
    """The one knob the caller controls, and it moves both caveats together.

    The gap between my measured sensitivity floor near 0.017 invented area and the review
    lane's 0.09 was unexplained for three heads. It is not geometry, sampling, or mesh
    resolution -- resolution showed no trend above noise across a 32-fold range of
    edge/voxel. It is the voxel chosen relative to the sample spacing, which is the only
    input the caller supplies, and the review lane's own zero-error unsupported fraction of
    0.75126 places their construction near 1.2x spacing where this curve sits at 0.0467.

    What the operator needs from that is a direction, not a number, so this pins the
    direction: loosening the voxel lowers the headline fraction on a correct reconstruction
    and lowers the floor at the same time. Both improve together, so there is no tradeoff to
    balance -- a tight voxel is simply worse on both counts.

    Figures: `evidence/open3d/floor-vs-voxel-multiple.json`.
    """

    from npa.workbench.open3d.runner import _crop_justification

    class _Mesh:
        vertices = range(100000)

    # voxel as a multiple of median sample spacing -> (zero-error unsupported, floor)
    measured = [
        (1.00, 0.8780, 0.0467),
        (1.25, 0.7135, 0.0467),
        (1.50, 0.5069, 0.0301),
        (2.00, 0.1707, 0.0168),
        (3.00, 0.0055, 0.0075),
    ]
    for (_, loose_unsup, loose_floor), (_, tight_unsup, tight_floor) in zip(
        measured[1:], measured
    ):
        assert loose_unsup <= tight_unsup, (
            "a looser voxel must not report more unsupported area on a correct "
            "reconstruction than a tighter one"
        )
        assert loose_floor <= tight_floor, (
            "a looser voxel must not detect fabrication less sensitively than a tighter one"
        )

    # The tool's documented recommendation is >= 2x spacing. At that setting a zero-error
    # reconstruction must not read as an invented shell, which is what makes it safe to
    # recommend.
    for _, unsupported, _ in measured[3:]:
        reading = _crop_justification(
            _support_block(unsupported=unsupported, beyond_1_5=0.0, beyond_3=0.0),
            0,
            _Mesh(),
        )
        assert reading["removed_surface_reads_as"] == "near-threshold surface"


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


def _inside_frame(points, camera: dict, aspect: float) -> float:
    """Worst frame-relative coordinate over ``points``; 1.0 is exactly the edge.

    Mirrors the projection the camera fit solves against, in the same units, so a
    result just under 1.0 means the geometry sits just inside the frame.
    """

    import numpy as np

    eye = np.asarray(camera["eye"], float)
    forward = np.asarray(camera["look_target"], float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], float))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    half = math.tan(math.radians(camera["fov_degrees"]) / 2.0)

    local = np.asarray(points, float) - eye
    depth = local @ forward
    assert (depth > 1e-6).all(), "geometry behind the camera cannot be framed"
    return float(
        max(
            np.abs((local @ right) / (half * aspect * depth)).max(),
            np.abs((local @ true_up) / (half * depth)).max(),
        )
    )


def _sprawling_scene():
    """A cropped surface inside a scan, plus a removed surface reaching outside it.

    This is the real shape of the defect rather than an invented one: the support
    crop removes surface precisely where Poisson closed a hole far from any
    sample, so the unsupported geometry lies outside the cloud by construction.
    """

    import numpy as np

    rng = np.random.default_rng(20260921)
    fused = rng.uniform([-1.0, 0.0, -1.0], [1.0, 0.4, 1.0], size=(4000, 3))
    mesh = rng.uniform([-0.9, 0.0, -0.9], [0.9, 0.35, 0.9], size=(600, 3))
    unsupported = rng.uniform([-1.0, 1.6, -1.0], [1.0, 2.4, 1.0], size=(300, 3))
    return {"fused": fused, "mesh": mesh, "unsupported": unsupported}


def test_every_view_frames_the_geometry_that_view_actually_shows():
    """The invariant the shipped blueprint broke, pinned for all four views.

    The operator's first complaint was a clipped scene. The camera distance was
    already solved rather than guessed, so the bug was not a missing fit — it was
    that one fit was shared by four views and only three of them draw geometry
    that fits inside the fused cloud. The fourth draws the surface the crop
    removed, which is outside that cloud by construction, and it clipped 23% past
    its own edge: measured 1.233 on the delivered artifacts, recorded in
    evidence/open3d/camera-framing-probe.json.

    A clipped audit view is worse than no audit view, because it looks like the
    audit happened.
    """

    from npa.workbench.open3d.runner import VIEW_GEOMETRY, _view_cameras

    geometry = _sprawling_scene()
    cameras = _view_cameras(geometry, [0.0, 1.0, 0.0])

    assert set(cameras) == set(VIEW_GEOMETRY), "every view needs its own camera"
    for view_name, spec in VIEW_GEOMETRY.items():
        for key in spec.geometry:
            reach = _inside_frame(geometry[key], cameras[view_name], spec.aspect)
            assert reach <= 1.0, (
                f"{view_name!r} clips its own {key} geometry at {reach:.3f} of the "
                "frame half-extent"
            )


def test_a_view_is_framed_snugly_rather_than_stranded_in_an_empty_frame():
    """Fitting is two-sided: clipping fails review, and so does a distant speck.

    Without this, the previous test passes trivially by placing the eye a
    kilometre back. The fit targets 1/CAMERA_FRAME_MARGIN, so the worst point
    should land just inside the edge rather than anywhere inside it.
    """

    from npa.workbench.open3d.runner import (
        CAMERA_FRAME_MARGIN,
        VIEW_GEOMETRY,
        _view_cameras,
    )

    import numpy as np

    geometry = _sprawling_scene()
    cameras = _view_cameras(geometry, [0.0, 1.0, 0.0])
    for view_name, spec in VIEW_GEOMETRY.items():
        shown = np.vstack([geometry[key] for key in spec.geometry])
        reach = _inside_frame(shown, cameras[view_name], spec.aspect)
        assert reach >= 1.0 / CAMERA_FRAME_MARGIN - 0.05, (
            f"{view_name!r} strands its geometry at {reach:.3f}; the frame is "
            "mostly empty, which is as hard to review as a crop"
        )


def test_a_scene_with_nothing_removed_gets_no_audit_camera_rather_than_a_fake_one():
    """The audit view is dropped from the layout when there is nothing to audit.

    A camera fitted to absent geometry would either raise or invent a viewpoint,
    and the blueprint only adds that tab when a crop actually removed surface.
    """

    from npa.workbench.open3d.runner import REMOVED_VIEW, SCENE_VIEW, _view_cameras

    geometry = _sprawling_scene()
    geometry["unsupported"] = None
    cameras = _view_cameras(geometry, [0.0, 1.0, 0.0])

    assert REMOVED_VIEW not in cameras
    assert SCENE_VIEW in cameras


def test_a_cloud_only_run_still_gets_a_scene_camera():
    """Visualize does not require a mesh, and the layout must survive without one.

    Dropping a view when its defining geometry is missing is right; dropping the
    scene view because the optional surface is missing would crash the caller that
    reads the scene camera.
    """

    from npa.workbench.open3d.runner import (
        SCAN_VIEW,
        SCENE_VIEW,
        SURFACE_VIEW,
        _view_cameras,
    )

    geometry = _sprawling_scene()
    cameras = _view_cameras(
        {"fused": geometry["fused"], "mesh": None, "unsupported": None},
        [0.0, 1.0, 0.0],
    )

    assert SCENE_VIEW in cameras and SCAN_VIEW in cameras
    assert SURFACE_VIEW not in cameras


def _built_views(blueprint) -> dict[str, set[str]]:
    """Every named view in the built blueprint, mapped to the entities it draws.

    Walks the real Rerun object graph -- `Blueprint.root_container` down through
    each container's `contents` -- so the assertions are about what the viewer
    would be handed rather than about the arguments passed in.
    """

    found: dict[str, set[str]] = {}

    def walk(node) -> None:
        contents = getattr(node, "contents", None)
        name = getattr(node, "name", None)
        paths = (
            {str(item) for item in contents if isinstance(item, str)}
            if isinstance(contents, (list, tuple))
            else set()
        )
        if isinstance(name, str) and name and paths:
            found[name] = paths
        for child in (getattr(node, "root_container", None), contents):
            if isinstance(child, (list, tuple)):
                for item in child:
                    if not isinstance(item, str):
                        walk(item)
            elif child is not None and not isinstance(child, str):
                walk(child)

    walk(blueprint)
    return found


def test_a_cloud_only_run_offers_no_empty_surface_tab():
    """The absent view has to be absent from the layout, not just from the cameras.

    Testing the camera dictionary was not enough, and that gap was real: the
    blueprint appended the surface view unconditionally and fell back to the scene
    camera when its own was missing, so a cloud-only run opened a "supported
    surface" tab with nothing in it. An empty tab is a worse failure than a missing
    one -- it tells the reader the surface was computed and is empty.
    """

    import rerun as rr
    import rerun.blueprint as rrb

    from npa.workbench.open3d.runner import (
        REMOVED_VIEW,
        SCAN_VIEW,
        SCENE_VIEW,
        SURFACE_VIEW,
        _blueprint,
        _view_cameras,
    )

    geometry = _sprawling_scene()
    cameras = _view_cameras(
        {"fused": geometry["fused"], "mesh": None, "unsupported": None},
        [0.0, 1.0, 0.0],
    )
    names = _built_views(_blueprint(rr, rrb, cameras))

    assert SURFACE_VIEW not in names, "a cloud-only run must not offer an empty tab"
    assert REMOVED_VIEW not in names, (
        "nothing was cropped, so there is nothing to audit"
    )
    assert SCENE_VIEW in names and SCAN_VIEW in names


def test_the_blueprint_draws_the_entities_the_table_says_each_view_draws():
    """The table has to drive the layout, not merely sit beside it.

    The camera and the contents were declared in two places, so a view's camera
    could be fitted to geometry the view does not draw -- which is the class of bug
    the per-view cameras were added to fix, reintroduced one level up.
    """

    import rerun as rr
    import rerun.blueprint as rrb

    from npa.workbench.open3d.runner import VIEW_GEOMETRY, _blueprint, _view_cameras

    cameras = _view_cameras(_sprawling_scene(), [0.0, 1.0, 0.0])
    blueprint = _blueprint(rr, rrb, cameras)

    declared = {
        name: set(spec.contents)
        for name, spec in VIEW_GEOMETRY.items()
        if name in cameras
    }
    built = _built_views(blueprint)

    assert built == declared, (
        "the blueprint draws entities the table does not declare, so the camera "
        "and the contents can drift apart again"
    )


def test_a_view_in_a_narrow_tab_is_fitted_for_that_tab_and_not_for_a_wide_pane():
    """A fit is only as good as the pane it was fitted for.

    Captured from the running viewer, every view fitted for 4:3 filled the full
    width of its pane and touched both side edges while using 44% of the height of
    the narrow tab -- the signature of fitting a wide shape into a narrow one. The
    offscreen renders could not show this, because they were rendered at the same
    4:3 the camera assumed.
    """

    import numpy as np

    from npa.workbench.open3d.runner import (
        CAMERA_WINDOW_ASPECT,
        SCAN_VIEW,
        SCENE_VIEW,
        VIEW_GEOMETRY,
        _view_cameras,
    )

    geometry = _sprawling_scene()
    cameras = _view_cameras(geometry, [0.0, 1.0, 0.0])

    # The whole point of naming a window shape is that the real window is not that shape.
    # Horizontal binds in narrow windows, so a window wider than the assumed one is safe
    # and a narrower one clips, which is why the assumption has to be an ordinary window
    # rather than a wide monitor. 4:3 is the narrowest ordinary landscape window.
    assert CAMERA_WINDOW_ASPECT == 4.0 / 3.0, (
        "the assumed window shape sets which real windows are safe; widening it "
        "silently reintroduces clipping for everyone on a narrower one"
    )
    for view_name, spec in VIEW_GEOMETRY.items():
        shown = np.vstack([geometry[key] for key in spec.geometry])
        share = spec.aspect / CAMERA_WINDOW_ASPECT
        for window in (4.0 / 3.0, 1.5, 1.6, 16.0 / 9.0, 21.0 / 9.0):
            reach = _inside_frame(shown, cameras[view_name], window * share)
            assert reach <= 1.0, (
                f"{view_name!r} clips at {reach:.4f} in a {window:.3f} window, which is "
                "at least as wide as the shape it was fitted for"
            )

    # The tab column is half the width of the scene pane, so its views must sit
    # further back, not at the same distance.
    assert VIEW_GEOMETRY[SCAN_VIEW].aspect < VIEW_GEOMETRY[SCENE_VIEW].aspect
    assert all(
        camera["pane_aspect"] == VIEW_GEOMETRY[name].aspect
        for name, camera in cameras.items()
    ), "a camera must record the pane it was fitted for"
    assert max(spec.aspect for spec in VIEW_GEOMETRY.values()) <= CAMERA_WINDOW_ASPECT

    # And the fit must hold in that pane: the same geometry, checked against each
    # view's own shape rather than one shared assumption.
    for view_name, spec in VIEW_GEOMETRY.items():
        shown = np.vstack([geometry[key] for key in spec.geometry])
        reach = _inside_frame(shown, cameras[view_name], spec.aspect)
        assert reach <= 1.0, (
            f"{view_name!r} clips in its own {spec.aspect:.2f} pane at {reach:.3f}"
        )
        # A camera fitted for the wide pane would clip here; that is the defect.
        wide = _inside_frame(
            shown, cameras[view_name], VIEW_GEOMETRY[SCENE_VIEW].aspect
        )
        assert wide <= reach, "a wider pane cannot be tighter than a narrow one"


def test_a_reconstruction_that_carries_colour_keeps_it_and_one_that_does_not_is_left_alone():
    """Colour is carried through from the scans, never invented.

    The viewer logged positions, triangles and normals only, so the kept surface
    arrived untinted beside a deliberately red audit overlay -- which reads as a
    colour that means something rather than an absence of one. The offscreen
    renders of the same PLY bytes did show colour, so they must not imply the
    viewer did.
    """

    import numpy as np

    from npa.workbench.open3d.runner import _mesh_colors

    class _Mesh:
        def __init__(self, colors):
            self.vertex_colors = colors

        def has_vertex_colors(self):
            return self.vertex_colors is not None

    # Open3D holds colour as float 0..1; Rerun wants 8-bit channels.
    carried = _mesh_colors(np, _Mesh(np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])), 2)
    assert carried.dtype == np.uint8
    assert carried.tolist() == [[255, 0, 128], [0, 255, 0]]

    # No colour, and a shape that does not match, both leave shading alone rather
    # than inventing a tint that would imply measurement that did not happen.
    assert _mesh_colors(np, _Mesh(None), 2) is None
    assert _mesh_colors(np, _Mesh(np.array([[1.0, 0.0, 0.0]])), 2) is None

    # Values outside the documented range are clamped, not wrapped into a
    # different colour by the cast.
    clamped = _mesh_colors(np, _Mesh(np.array([[1.4, -0.2, 0.5]])), 1)
    assert clamped.tolist() == [[255, 0, 128]]
