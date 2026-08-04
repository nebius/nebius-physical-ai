"""`/index` and `/query`: the dataset-of-record's half of the LanceDB contract.

`npa.workbench.dataset.integrations` has always POSTed these two paths, and the wrapper has
always exposed `/tables/{name}` and `/query-table`. Two halves written against different APIs
that never met — because until the service could be deployed where a stage can reach it, nobody
ever made the call. Live job 313 finally did, and got
`Client error '404 Not Found' for url '…/query'` (EVIDENCE §R41).
"""

from __future__ import annotations

from pathlib import Path

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
        {"id": "clip-1", "location": "san-francisco", "frames": 120, "night": True},
        {"id": "clip-2", "location": "berlin", "frames": 90, "night": False},
    ]


def test_index_creates_the_table_on_first_write(client: TestClient) -> None:
    response = client.post("/index", json={"table": "dataset", "records": _records()})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "created"
    assert body["rows"] == 2
    assert body["table"] == "dataset"


def test_index_appends_on_a_second_write(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/index",
        json={"table": "dataset", "records": [{"id": "clip-3", "location": "berlin", "frames": 30, "night": False}]},
    )

    assert response.json()["status"] == "appended"
    listed = client.post("/query", json={"table": "dataset", "limit": 100}).json()
    assert listed["count"] == 3


def test_index_rejects_an_empty_payload(client: TestClient) -> None:
    assert client.post("/index", json={"table": "dataset", "records": []}).status_code == 400


def test_query_filters_by_equality_facet(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/query", json={"table": "dataset", "filter": {"location": "berlin"}, "limit": 10}
    )

    body = response.json()
    assert body["count"] == 1
    assert body["records"][0]["id"] == "clip-2"


def test_query_handles_a_boolean_facet(client: TestClient) -> None:
    client.post("/index", json={"table": "dataset", "records": _records()})

    body = client.post("/query", json={"table": "dataset", "filter": {"night": True}}).json()

    assert [record["id"] for record in body["records"]] == ["clip-1"]


def test_query_on_an_unregistered_table_is_empty_not_an_error(client: TestClient) -> None:
    """A curation step may legitimately query before anything has been indexed."""

    body = client.post("/query", json={"table": "never-written", "filter": {}}).json()

    assert body == {"table": "never-written", "records": [], "count": 0}


def test_query_escapes_a_value_containing_a_quote(client: TestClient) -> None:
    client.post(
        "/index",
        json={"table": "dataset", "records": [{"id": "x", "location": "o'hare", "frames": 1, "night": False}]},
    )

    body = client.post("/query", json={"table": "dataset", "filter": {"location": "o'hare"}}).json()

    assert body["count"] == 1


def test_query_rejects_a_field_name_that_is_not_a_plain_identifier(client: TestClient) -> None:
    """A facet API has no operators; accepting arbitrary SQL would make this an injection point."""

    # Index first: a query against an unknown table returns empty before any predicate is built,
    # which would make this pass for the wrong reason.
    client.post("/index", json={"table": "dataset", "records": _records()})

    response = client.post(
        "/query", json={"table": "dataset", "filter": {"1=1 OR x": "y"}}
    )

    assert response.status_code == 400


def test_query_rejects_an_absurd_limit(client: TestClient) -> None:
    assert client.post("/query", json={"table": "dataset", "limit": 0}).status_code == 400
    assert client.post("/query", json={"table": "dataset", "limit": 10_001}).status_code == 400


def test_the_paths_match_what_the_dataset_integration_posts() -> None:
    """Pin the contract itself, so the two halves cannot drift apart again."""

    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "npa/src/npa/workbench/dataset/integrations.py").read_text(
        encoding="utf-8"
    )

    assert '_post(lancedb_endpoint, "/index"' in source
    assert '_post(lancedb_endpoint, "/query"' in source
    server = (repo_root / "npa/src/npa/workbench/lancedb/server.py").read_text(encoding="utf-8")
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


def test_a_quality_threshold_is_applied_to_the_rows_not_pushed_into_the_facet_api(monkeypatch) -> None:
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
        filter_predicate={"event": "cut_in", "quality_metric": "completeness", "min_quality": 0.5},
        limit=10,
        table="fleet-dataset",
    )

    # A threshold has no equality form, so it never reaches the endpoint.
    assert sent["filter"] == {"event": "cut_in"}
    assert sent["table"] == "fleet-dataset"
    assert [record["record_id"] for record in records] == ["a"]
