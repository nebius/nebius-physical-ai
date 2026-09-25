"""`/index` and `/query`: the dataset-of-record's half of the LanceDB contract.

`npa.workbench.dataset.integrations` has always POSTed these two paths, and the wrapper has
always exposed `/tables/{name}` and `/query-table`. Two halves written against different APIs
that never met — because until the service could be deployed where a stage can reach it, nobody
ever made the call. Live job 313 finally did, and got
`Client error '404 Not Found' for url '…/query'` (EVIDENCE §R41).
"""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from npa.workbench.lancedb.server import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LANCEDB_STORAGE_PATH", str(tmp_path / "lance"))
    monkeypatch.setenv("LANCEDB_AUTH_MODE", "none")
    return TestClient(create_app())


def _records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "clip-1",
            "location": "san-francisco",
            "frames": 120,
            "night": True,
        },
        {"record_id": "clip-2", "location": "berlin", "frames": 90, "night": False},
    ]


def _table_schema(*, vector_size: int = 2) -> dict[str, object]:
    return {
        "fields": [
            {"name": "id", "type": "string", "nullable": False},
            {
                "name": "vector",
                "type": {
                    "name": "fixed_size_list",
                    "item_type": "float32",
                    "list_size": vector_size,
                },
                "nullable": False,
            },
        ]
    }


def _open_created_table(tmp_path: Path, name: str):
    return lancedb.connect(tmp_path / "lance").open_table(name)


def _created_table_names(tmp_path: Path) -> set[str]:
    return set(lancedb.connect(tmp_path / "lance").list_tables().tables)


def test_create_table_with_schema_stores_zero_rows(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post("/tables/empty_vectors", json={"schema": _table_schema()})

    assert response.status_code == 200, response.text
    assert response.json()["rows"] == 0
    table = _open_created_table(tmp_path, "empty_vectors")
    assert table.count_rows() == 0
    assert table.schema.equals(
        pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), 2), nullable=False),
            ]
        ),
        check_metadata=False,
    )


@pytest.mark.parametrize("payload", [{}, {"schema": {}}, {"schema": {"fields": []}}])
def test_create_table_requires_rows_or_usable_schema(
    client: TestClient, tmp_path: Path, payload: dict[str, object]
) -> None:
    response = client.post("/tables/rejected", json=payload)

    assert response.status_code == 400, response.text
    assert "rejected" not in _created_table_names(tmp_path)


def test_create_table_schema_controls_stored_rows(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/vectors",
        json={
            "schema": _table_schema(),
            "rows": [{"id": "row-1", "vector": [1.0, 2.0]}],
        },
    )

    assert response.status_code == 200, response.text
    table = _open_created_table(tmp_path, "vectors")
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 2)
    assert not table.schema.field("id").nullable
    assert table.to_arrow().to_pylist() == [{"id": "row-1", "vector": [1.0, 2.0]}]


def test_create_table_infers_the_schema_lancedb_stores(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/inferred_vectors",
        json={
            "rows": [
                {"id": "a", "vector": [0.1, 0.2, 0.3, 0.4]},
                {"id": "b", "vector": [0.9, 0.8, 0.7, 0.6]},
            ],
            "mode": "overwrite",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["rows"] == 2
    table = _open_created_table(tmp_path, "inferred_vectors")
    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)


def test_create_table_rejects_inconsistent_inferred_vectors_before_mutation(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/inconsistent_vectors",
        json={"rows": [{"vector": [1.0, 2.0]}, {"vector": [1.0]}]},
    )

    assert response.status_code == 400, response.text
    assert "consistent positive size" in response.json()["detail"]
    assert "inconsistent_vectors" not in _created_table_names(tmp_path)


def test_create_table_preserves_fields_introduced_by_later_rows(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/events",
        json={"rows": [{"id": "one"}, {"id": "two", "score": 0.5}]},
    )

    assert response.status_code == 200, response.text
    assert _open_created_table(tmp_path, "events").to_arrow().to_pylist() == [
        {"id": "one", "score": None},
        {"id": "two", "score": 0.5},
    ]


def test_create_table_rejects_incompatible_rows_before_mutation(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/bad_vectors",
        json={
            "schema": _table_schema(),
            "rows": [{"id": "row-1", "vector": [1.0]}],
        },
    )

    assert response.status_code == 400, response.text
    assert "bad_vectors" not in _created_table_names(tmp_path)


def test_create_table_overwrite_and_append_preserve_schema(
    client: TestClient, tmp_path: Path
) -> None:
    schema = _table_schema()
    client.post(
        "/tables/vectors",
        json={"schema": schema, "rows": [{"id": "old", "vector": [0.0, 0.0]}]},
    )
    overwritten = client.post(
        "/tables/vectors",
        json={
            "schema": schema,
            "rows": [{"id": "new", "vector": [1.0, 1.0]}],
            "mode": "overwrite",
        },
    )
    appended = client.post(
        "/tables/vectors",
        json={
            "schema": schema,
            "rows": [{"id": "next", "vector": [2.0, 2.0]}],
            "mode": "append",
        },
    )

    assert overwritten.status_code == 200, overwritten.text
    assert appended.status_code == 200, appended.text
    assert appended.json()["rows"] == 1
    table = _open_created_table(tmp_path, "vectors")
    assert table.count_rows() == 2
    assert [row["id"] for row in table.to_arrow().to_pylist()] == ["new", "next"]


def test_create_table_create_overwrite_and_append_infer_compatible_rows(
    client: TestClient, tmp_path: Path
) -> None:
    created = client.post("/tables/events", json={"rows": [{"id": "old", "value": 1}]})
    overwritten = client.post(
        "/tables/events",
        json={"rows": [{"id": "new", "value": 2}], "mode": "overwrite"},
    )
    appended = client.post(
        "/tables/events",
        json={"rows": [{"id": "next", "value": 3}], "mode": "append"},
    )

    assert created.status_code == 200, created.text
    assert overwritten.status_code == 200, overwritten.text
    assert appended.status_code == 200, appended.text
    assert _open_created_table(tmp_path, "events").count_rows() == 2


def test_append_schema_mismatch_fails_without_mutation(
    client: TestClient, tmp_path: Path
) -> None:
    client.post(
        "/tables/vectors",
        json={
            "schema": _table_schema(),
            "rows": [{"id": "row-1", "vector": [1.0, 2.0]}],
        },
    )

    response = client.post(
        "/tables/vectors",
        json={
            "schema": _table_schema(vector_size=3),
            "rows": [{"id": "row-2", "vector": [1.0, 2.0, 3.0]}],
            "mode": "append",
        },
    )

    assert response.status_code == 400, response.text
    assert _open_created_table(tmp_path, "vectors").count_rows() == 1


def test_create_table_rejects_server_side_s3_import(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/tables/s3_vectors",
        json={"input_path": "s3://example/vectors.json", "schema": _table_schema()},
    )

    assert response.status_code == 400, response.text
    assert "not implemented" in response.json()["detail"]
    assert "s3_vectors" not in _created_table_names(tmp_path)


def test_index_creates_the_table_on_first_write(client: TestClient) -> None:
    response = client.post("/index", json={"table": "dataset", "records": _records()})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "created"
    assert body["rows"] == 2
    assert body["inserted"] == 2
    assert body["reused"] == 0
    assert body["table"] == "dataset"


def test_readiness_is_public_while_health_remains_authenticated(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            storage_path=str(tmp_path / "secure-lance"),
            auth_mode="token",
            token="s3cr3t",
        )
    )

    assert client.get("/readyz").json() == {"status": "ok"}
    assert client.get("/health").status_code == 401
    assert (
        client.get("/health", headers={"Authorization": "Bearer s3cr3t"}).status_code
        == 200
    )


def test_index_appends_on_a_second_write(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/index",
        json={
            "table": "dataset",
            "records": [
                {
                    "record_id": "clip-3",
                    "location": "berlin",
                    "frames": 30,
                    "night": False,
                }
            ],
        },
    )

    assert response.json()["status"] == "appended"
    assert response.json()["inserted"] == 1
    assert response.json()["reused"] == 0
    listed = client.post("/query", json={"table": "dataset", "limit": 100}).json()
    assert listed["count"] == 3


def test_index_replays_identical_records_without_duplicate_rows(
    client: TestClient,
) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post("/index", json={"table": "dataset", "records": _records()})

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "replayed"
    assert response.json()["inserted"] == 0
    assert response.json()["reused"] == 2
    listed = client.post("/query", json={"table": "dataset", "limit": 100}).json()
    assert listed["count"] == 2
    assert [record["record_id"] for record in listed["records"]] == [
        "clip-1",
        "clip-2",
    ]
    berlin = client.post(
        "/query", json={"table": "dataset", "filter": {"location": "berlin"}}
    ).json()
    assert berlin["count"] == 1
    assert berlin["records"][0]["record_id"] == "clip-2"


def test_index_mixes_novel_and_replayed_records_atomically(client: TestClient) -> None:
    original = _records()[0]
    client.post("/index", json={"table": "dataset", "records": [original]})
    novel = {
        "record_id": "clip-2",
        "location": "berlin",
        "frames": 90,
        "night": False,
    }

    response = client.post(
        "/index", json={"table": "dataset", "records": [original, novel]}
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "appended"
    assert response.json()["inserted"] == 1
    assert response.json()["reused"] == 1
    assert client.post("/query", json={"table": "dataset"}).json()["count"] == 2


def test_index_rejects_conflicting_content_before_appending(
    client: TestClient,
) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})
    novel = {
        "record_id": "clip-3",
        "location": "oslo",
        "frames": 10,
        "night": False,
    }
    conflicting = {**_records()[0], "frames": 121}

    response = client.post(
        "/index", json={"table": "dataset", "records": [novel, conflicting]}
    )

    assert response.status_code == 409, response.text
    assert "clip-1" in response.json()["detail"]
    listed = client.post("/query", json={"table": "dataset", "limit": 100}).json()
    assert listed["count"] == 2
    assert {record["record_id"] for record in listed["records"]} == {
        "clip-1",
        "clip-2",
    }


def test_index_deduplicates_identical_records_within_one_request(
    client: TestClient,
) -> None:
    record = {
        "record_id": "nested",
        "location": "oslo",
        "frames": 10,
        "night": False,
        "metadata": {"weather": "snow", "sensors": ["camera", "lidar"]},
    }
    reordered = {
        "metadata": {"sensors": ["camera", "lidar"], "weather": "snow"},
        "night": False,
        "frames": 10,
        "location": "oslo",
        "record_id": "nested",
    }

    response = client.post(
        "/index", json={"table": "dataset", "records": [record, reordered]}
    )

    assert response.status_code == 200, response.text
    assert response.json()["inserted"] == 1
    assert response.json()["reused"] == 1
    replay = client.post("/index", json={"table": "dataset", "records": [reordered]})
    assert replay.status_code == 200, replay.text
    assert replay.json()["status"] == "replayed"
    assert replay.json()["inserted"] == 0
    assert replay.json()["reused"] == 1
    assert client.post("/query", json={"table": "dataset"}).json()["count"] == 1


def test_index_rejects_conflicting_request_duplicates_before_creation(
    client: TestClient, tmp_path: Path
) -> None:
    record = _records()[0]

    response = client.post(
        "/index",
        json={"table": "dataset", "records": [record, {**record, "frames": 121}]},
    )

    assert response.status_code == 409, response.text
    assert "dataset" not in _created_table_names(tmp_path)


def test_index_rejects_an_empty_payload(client: TestClient) -> None:
    assert (
        client.post("/index", json={"table": "dataset", "records": []}).status_code
        == 400
    )


def test_query_filters_by_equality_facet(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/query",
        json={"table": "dataset", "filter": {"location": "berlin"}, "limit": 10},
    )

    body = response.json()
    assert body["count"] == 1
    assert body["records"][0]["record_id"] == "clip-2"


def test_query_handles_a_boolean_facet(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    body = client.post(
        "/query", json={"table": "dataset", "filter": {"night": True}}
    ).json()

    assert [record["record_id"] for record in body["records"]] == ["clip-1"]


def test_query_on_an_unregistered_table_is_empty_not_an_error(
    client: TestClient,
) -> None:
    """A curation step may legitimately query before anything has been indexed."""

    body = client.post("/query", json={"table": "never-written", "filter": {}}).json()

    assert body == {"table": "never-written", "records": [], "count": 0}


def test_query_escapes_a_value_containing_a_quote(client: TestClient) -> None:
    client.post(
        "/index",
        json={
            "table": "dataset",
            "records": [
                {
                    "record_id": "x",
                    "location": "o'hare",
                    "frames": 1,
                    "night": False,
                }
            ],
        },
    )

    body = client.post(
        "/query", json={"table": "dataset", "filter": {"location": "o'hare"}}
    ).json()

    assert body["count"] == 1


def test_query_rejects_a_field_name_that_is_not_a_plain_identifier(
    client: TestClient,
) -> None:
    """A facet API has no operators; accepting arbitrary SQL would make this an injection point."""

    # Index first: a query against an unknown table returns empty before any predicate is built,
    # which would make this pass for the wrong reason.
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/query", json={"table": "dataset", "filter": {"1=1 OR x": "y"}}
    )

    assert response.status_code == 400


def test_query_rejects_an_absurd_limit(client: TestClient) -> None:
    assert (
        client.post("/query", json={"table": "dataset", "limit": 0}).status_code == 400
    )
    assert (
        client.post("/query", json={"table": "dataset", "limit": 10_001}).status_code
        == 400
    )


def test_the_paths_match_what_the_dataset_integration_posts() -> None:
    """Pin the contract itself, so the two halves cannot drift apart again."""

    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "npa/src/npa/workbench/dataset/integrations.py").read_text(
        encoding="utf-8"
    )

    assert "_post(" in source and "/index" in source
    assert "_post(" in source and "/query" in source
    server = (repo_root / "npa/src/npa/workbench/lancedb/server.py").read_text(
        encoding="utf-8"
    )
    assert '@app.post("/index")' in server
    assert '@app.post("/query")' in server


# ------------------------------------------------- the caller's half of the contract


def test_unset_facets_are_not_sent_as_equality_predicates() -> None:
    """Live: a query returned 0 records from a table that held three matching rows.

    The caller builds a predicate with every facet it knows about, set or not. Sent verbatim
    that asks the index for `modality = '' AND min_quality = 'None'`, which matches nothing.
    """

    from npa.workbench.dataset.integrations import equality_facets

    assert equality_facets(
        {
            "event": "cut_in",
            "location": "san_francisco",
            "modality": "",
            "quality_metric": "completeness",
            "min_quality": None,
        }
    ) == {"event": "cut_in", "location": "san_francisco"}


def test_a_quality_threshold_is_applied_to_the_rows_not_pushed_into_the_facet_api(
    monkeypatch,
) -> None:
    from npa.workbench.dataset import integrations

    sent: dict[str, object] = {}

    def fake_post(endpoint, path, *, payload, token_env, timeout):
        sent.update(payload)
        return {
            "records": [
                {"record_id": "a", "completeness": 0.9},
                {"record_id": "b", "completeness": 0.2},
            ]
        }

    monkeypatch.setattr(integrations, "_post", fake_post)

    records = integrations.query_lancedb(
        lancedb_endpoint="http://svc:8686",
        filter_predicate={
            "event": "cut_in",
            "quality_metric": "completeness",
            "min_quality": 0.5,
        },
        limit=10,
        table="fleet-dataset",
    )

    # A threshold has no equality form, so it never reaches the endpoint.
    assert sent["filter"] == {"event": "cut_in"}
    assert sent["table"] == "fleet-dataset"
    assert [record["record_id"] for record in records] == ["a"]
