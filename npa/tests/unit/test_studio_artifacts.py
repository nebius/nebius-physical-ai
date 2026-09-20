"""Verify Studio storage discovery coverage, attribution and credential routing."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

from botocore.exceptions import ClientError
import pytest

from npa import studio, studio_artifacts as search


class Pages:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        for page in self.pages:
            if isinstance(page, Exception):
                raise page
            yield page


class Client:
    def __init__(self, buckets, objects, metadata=None):
        self.meta = SimpleNamespace(endpoint_url="https://storage.example.test")
        self.buckets = Pages(buckets)
        self.objects = Pages(objects)
        self.metadata = metadata or {}
        self.head_calls = []

    def get_paginator(self, operation):
        return self.buckets if operation == "list_buckets" else self.objects

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        if isinstance(self.metadata, Exception):
            raise self.metadata
        return {"Metadata": self.metadata}


def obj(key):
    return {
        "Key": key,
        "Size": 20,
        "ETag": "multipart-etag-2",
        "LastModified": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }


def denied():
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "secret must not be emitted"}},
        "List",
    )


@pytest.fixture
def storage(monkeypatch):
    client = Client(
        [{"Buckets": [{"Name": "test-assets"}]}],
        [{"Contents": [obj("runs/movie.mp4")]}],
    )
    monkeypatch.setattr(search, "s3_client_for_project", lambda project: client)
    monkeypatch.setattr(
        search,
        "resolve_project_storage",
        lambda *a, **k: SimpleNamespace(checkpoint_bucket=""),
    )
    return client


def invoke(capsys, *arguments):
    code = search.run_search(list(arguments))
    return code, json.loads(capsys.readouterr().out)


def test_search_paginates_buckets_and_objects_without_a_known_type_gate(
    storage, capsys
):
    storage.buckets.pages.append({"Buckets": [{"Name": "more-assets"}]})
    storage.objects.pages.append({"Contents": [obj("new-format.unknown")]})
    code, result = invoke(capsys)
    assert code == 0 and result["complete"]
    assert result["count"] == 4
    assert {r["render_hint"] for r in result["artifacts"]} == {"video", "download"}
    assert len(storage.objects.calls) == 2
    assert not storage.head_calls
    assert all(r["provenance"] == {"status": "not_read"} for r in result["artifacts"])


def test_partial_page_failure_keeps_results_and_redacts_provider_message(
    storage, capsys
):
    storage.objects.pages.append(denied())
    code, result = invoke(capsys)
    assert code == 1 and not result["complete"]
    assert result["count"] == 1
    assert result["sources"][0]["error"]["code"] == "AccessDenied"
    assert "secret" not in json.dumps(result)


def test_explicit_bucket_works_when_bucket_enumeration_is_denied(storage, capsys):
    storage.buckets.pages = [denied()]
    code, result = invoke(capsys, "--bucket", "shared-assets")
    assert code == 0 and result["scope"] == "explicit_buckets"
    assert storage.objects.calls == [{"Bucket": "shared-assets", "Prefix": ""}]
    assert not storage.buckets.calls


def test_denied_enumeration_searches_configured_bucket_but_reports_partial(
    storage, capsys, monkeypatch
):
    storage.buckets.pages = [denied()]
    monkeypatch.setattr(
        search,
        "resolve_project_storage",
        lambda *a, **k: SimpleNamespace(checkpoint_bucket="saved-assets"),
    )
    code, result = invoke(capsys)
    assert code == 1 and result["count"] == 1
    assert not result["discovery"][0]["complete"]
    assert result["sources"][0]["complete"]


def test_metadata_is_declared_provenance_not_filename_inference(storage, capsys):
    storage.objects.pages = [{"Contents": [obj("cosmos/fake-name.mp4")]}]
    storage.metadata = {
        "npa-tool": "actual-tool",
        "npa-gpu": "B200",
        "secret": "not output",
    }
    _, result = invoke(capsys, "--read-metadata")
    provenance = result["artifacts"][0]["provenance"]
    assert provenance == {
        "status": "declared",
        "basis": "s3_object_metadata",
        "declared": {"tool": "actual-tool", "gpu": "B200"},
    }
    assert "sha256" not in provenance["declared"]


def test_metadata_names_are_case_insensitive(storage, capsys):
    storage.metadata = {
        "Tool": "cosmos3-nano",
        "Tool-Evidence": "run-manifest",
        "SHA256": "declared-hash",
    }
    _, result = invoke(capsys, "--read-metadata")
    assert result["artifacts"][0]["provenance"]["declared"] == {
        "tool": "cosmos3-nano",
        "tool_evidence": "run-manifest",
        "sha256": "declared-hash",
    }


def test_missing_or_denied_metadata_stays_unknown(storage, capsys):
    _, result = invoke(capsys, "--read-metadata")
    assert result["artifacts"][0]["provenance"]["status"] == "unknown"
    storage.metadata = denied()
    _, result = invoke(capsys, "--read-metadata")
    assert result["artifacts"][0]["provenance"]["status"] == "unavailable"


def test_key_kind_prefix_and_since_filters(storage, capsys):
    storage.objects.pages = [
        {"Contents": [obj("Nano/a.mp4"), obj("Nano/a.json"), obj("other.mp4")]}
    ]
    _, result = invoke(
        capsys,
        "--query",
        "nano",
        "--kind",
        "video",
        "--prefix",
        "runs/",
        "--since",
        "2026-01-01T00:00:00Z",
    )
    assert result["count"] == 1
    assert storage.objects.calls[0]["Prefix"] == "runs/"
    _, result = invoke(capsys, "--since", "2026-01-03T00:00:00Z")
    assert result["count"] == 0


def test_each_project_uses_its_own_credentials_and_preserves_source_tuple(
    storage, capsys, monkeypatch
):
    selected = []
    monkeypatch.setattr(search, "list_projects", lambda: {"one": {}, "two": {}})
    monkeypatch.setattr(
        search,
        "s3_client_for_project",
        lambda project: selected.append(project) or storage,
    )
    _, result = invoke(capsys, "--all-projects")
    assert selected == ["one", "two"]
    assert {r["project"] for r in result["artifacts"]} == {"one", "two"}
    assert all(r["endpoint"] and r["bucket"] and r["key"] for r in result["artifacts"])


def test_object_uri_escapes_special_characters(storage, capsys):
    storage.objects.pages = [{"Contents": [obj("runs/a #?.mp4")]}]
    _, result = invoke(capsys)
    assert result["artifacts"][0]["s3_uri"] == "s3://test-assets/runs/a%20%23%3F.mp4"


def test_search_does_not_require_registry_or_renderer(storage, capsys, tmp_path):
    code = studio.run(
        ["--registry", str(tmp_path / "absent.json"), "search", "--kind", "video"]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1


def test_since_requires_timezone():
    with pytest.raises(SystemExit):
        search.run_search(["--since", "2026-01-01"])


def test_configured_bucket_uri_is_normalized(storage, capsys, monkeypatch):
    storage.buckets.pages = [denied()]
    monkeypatch.setattr(
        search,
        "resolve_project_storage",
        lambda *a, **k: SimpleNamespace(checkpoint_bucket="s3://saved-assets/prefix/"),
    )
    _, result = invoke(capsys)
    assert result["sources"][0]["bucket"] == "saved-assets"
    assert storage.objects.calls[0]["Bucket"] == "saved-assets"


def test_tenant_search_uses_discovered_region_and_reports_denied_sources(
    storage, capsys, monkeypatch
):
    from npa import studio_sources

    targets = [
        {
            "project": "one",
            "resource_project_id": "test-project-one",
            "bucket": "test-assets",
            "endpoint": "https://region.example.test",
        },
        {
            "project": "two",
            "resource_project_id": "test-project-two",
            "bucket": "other-assets",
            "endpoint": "https://region.example.test",
        },
    ]
    monkeypatch.setattr(studio_sources, "tenant_sources", lambda project: (targets, []))
    calls = []

    def client(project, *, endpoint_url):
        calls.append((project, endpoint_url))
        if project == "two":
            raise denied()
        return storage

    monkeypatch.setattr(search, "s3_client_for_project", client)
    code, result = invoke(capsys, "--discover-tenant", "--project", "one")
    assert code == 1 and not result["complete"]
    assert result["count"] == 1
    assert result["artifacts"][0]["resource_project_id"] == "test-project-one"
    assert calls == [
        ("one", "https://region.example.test"),
        ("two", "https://region.example.test"),
    ]
    assert result["sources"][1]["error"]["code"] == "AccessDenied"


def test_tenant_inventory_failure_cannot_look_like_empty_success(capsys, monkeypatch):
    from npa import studio_sources

    monkeypatch.setattr(
        studio_sources,
        "tenant_sources",
        lambda project: (
            [],
            [{"operation": "list_tenant_projects", "code": "NebiusError"}],
        ),
    )
    code, result = invoke(capsys, "--discover-tenant")
    assert code == 1 and result["count"] == 0


def test_tenant_inventory_preserves_credential_ownership_and_paginates(monkeypatch):
    from npa import studio_sources

    calls = []
    monkeypatch.setattr(
        studio_sources,
        "resolve_environment",
        lambda project: SimpleNamespace(tenant_id="test-tenant"),
    )
    monkeypatch.setattr(
        studio_sources,
        "list_projects",
        lambda: {
            "selected": {"project_id": "test-one", "tenant_id": "test-tenant"},
            "other": {"project_id": "test-two", "tenant_id": "test-tenant"},
        },
    )

    def inventory(arguments):
        calls.append(arguments)
        if arguments[0] == "iam":
            return {
                "items": [
                    {"metadata": {"id": identity}, "spec": {"region": "us-central1"}}
                    for identity in ("test-one", "test-two", "test-unknown")
                ]
            }
        return {"items": [{"metadata": {"name": "test-assets"}}]}

    monkeypatch.setattr(studio_sources, "_run_json", inventory)
    sources, errors = studio_sources.tenant_sources("selected")
    assert not errors
    assert [source["project"] for source in sources] == [
        "selected",
        "other",
        "selected",
    ]
    assert all(call[-1] == "--all" for call in calls)


def test_tenant_inventory_rejects_incomplete_pages(monkeypatch):
    from npa import studio_sources

    monkeypatch.setattr(
        studio_sources,
        "_run_json",
        lambda arguments: {"items": [], "next_page_token": "next"},
    )
    with pytest.raises(ValueError, match="incomplete"):
        studio_sources._items(["iam", "project", "list"])


def test_provider_empty_protobuf_list_is_empty_and_unknown_payload_is_rejected(
    monkeypatch,
):
    from npa import studio_sources

    monkeypatch.setattr(studio_sources, "_run_json", lambda arguments: {})
    assert studio_sources._items(["storage", "bucket", "list"]) == []
    monkeypatch.setattr(
        studio_sources, "_run_json", lambda arguments: {"unexpected": True}
    )
    with pytest.raises(ValueError, match="items"):
        studio_sources._items(["storage", "bucket", "list"])
