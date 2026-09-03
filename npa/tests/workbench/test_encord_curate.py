"""`encord curate`: filter specs, the pinned preset shape, server-side selection."""

from __future__ import annotations


import pytest

from encord_fakes import (
    CURATE_RECEIPT_URI,
    ENVIRON,
    FakeCollection,
    FakeFolder,
    FakeItem,
    FakeStorage,
    FakeUserClient,
    fake_uuid,
)
from npa.workbench.encord.curate import (
    build_filter_preset_json,
    curate_receipt_uri_for,
    parse_filter_specs,
    preset_name_for,
    run_curate,
)
from npa.workbench.encord.schemas import EncordToolError


def _curate_kwargs(storage: FakeStorage, client: FakeUserClient, **overrides):
    kwargs = dict(
        folder="src",
        filters=["width:1:100000"],
        collection="keepers-new",
        output_path=CURATE_RECEIPT_URI,
        user_client=client,
        storage_client=storage,
        environ=dict(ENVIRON),
    )
    kwargs.update(overrides)
    return kwargs


def _curate_client(**client_kwargs) -> tuple[FakeUserClient, FakeFolder]:
    """A client whose 'src' folder holds one item (curate fails fast on empty)."""

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(fake_uuid(30), "seed.png", None)]
    return FakeUserClient(folders=[folder], **client_kwargs), folder


@pytest.fixture(autouse=False)
def _fast_polling(monkeypatch: pytest.MonkeyPatch):
    import npa.workbench.encord.curate as curate_module

    monkeypatch.setattr(curate_module, "POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(curate_module, "REISSUE_INTERVAL_SECONDS", 0.0)


def test_curate_receipt_uri_helper() -> None:
    assert curate_receipt_uri_for("s3://b/p/") == "s3://b/p/curate_receipt.json"
    assert curate_receipt_uri_for("s3://b/p/r.json") == "s3://b/p/r.json"


def test_preset_name_is_run_scoped_and_never_collides_ad_hoc() -> None:
    assert preset_name_for("run-1") == "npa-curate-run-1"
    first, second = preset_name_for(""), preset_name_for("   ")
    assert first.startswith("npa-curate-adhoc-") and second.startswith("npa-curate-adhoc-")
    assert first != second  # two ad-hoc curates must not race on one title


def test_parse_filter_specs_repeatable_and_comma_separated() -> None:
    parsed = parse_filter_specs(["brightness:0.2:0.8,sharpness:0.3:1", "width:32:4096"])
    assert [(f.metric, f.min, f.max) for f in parsed] == [
        ("brightness", 0.2, 0.8),
        ("sharpness", 0.3, 1.0),
        ("width", 32.0, 4096.0),
    ]
    assert [f.encord_metric for f in parsed] == [
        "metric_brightness",
        "metric_sharpness",
        "metric_width",
    ]
    # Computed vs intrinsic drives the zero-selection diagnostic.
    assert [f.computed for f in parsed] == [True, True, False]


@pytest.mark.parametrize(
    ("specs", "match"),
    [
        ([], "At least one --filter"),
        (["  ", ""], "At least one --filter"),
        (["blur:0:1"], "Unknown filter metric 'blur'"),
        (["brightness:0.2"], "expected metric:min:max"),
        (["brightness:a:b"], "must be numbers"),
        (["brightness:0.9:0.1"], "min exceeds max"),
    ],
)
def test_parse_filter_specs_fails_closed(specs: list[str], match: str) -> None:
    with pytest.raises(EncordToolError, match=match):
        parse_filter_specs(specs)


def test_build_filter_preset_json_pins_the_live_verified_shape() -> None:
    payload = build_filter_preset_json(parse_filter_specs(["brightness:0.2:0.8"]))
    assert payload == {
        "global_filters": {
            "filters": [
                {
                    "include": True,
                    "values": [0.2, 0.8],
                    "domain": "data",
                    "metric": "metric_brightness",
                    "type": "metric",
                }
            ]
        }
    }


def test_run_curate_happy_path_creates_collection_and_preset(_fast_polling) -> None:
    client, folder = _curate_client()
    client.pending_curate_items = [
        FakeItem(fake_uuid(31), "a.png", None),
        FakeItem(fake_uuid(32), "b.png", None),
    ]
    storage = FakeStorage()
    receipt = run_curate(**_curate_kwargs(storage, client, workflow_run="run-1"))

    assert receipt.status == "done"
    assert receipt.items_selected == 2
    assert receipt.items_total == 1  # the folder held one item when evaluated
    assert receipt.collection_created is True
    assert receipt.collection_name == "keepers-new"
    assert receipt.folder_uuid == str(folder.uuid)
    assert receipt.preset_name == "npa-curate-run-1"
    # The preset payload sent to Encord is the pinned shape, verbatim.
    (preset,) = client.created_presets
    assert preset.filter_preset_json["global_filters"]["filters"][0]["metric"] == "metric_width"
    (collection,) = client.created_collections
    assert collection.top_level_folder_uuid == str(folder.uuid)
    assert collection.preset_calls == [str(preset.uuid)]
    # The run-scoped preset is transient scaffolding, deleted once evaluated.
    assert client.deleted_presets == [str(preset.uuid)]
    assert receipt.preset_deleted is True
    # Curate never creates folders.
    assert client.created_folders == []
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["schema"] == "npa.encord.curate_receipt.v1"
    assert payload["items_selected"] == 2 and payload["items_total"] == 1


def test_run_curate_planned_receipt_lands_before_the_first_mutation(
    _fast_polling,
) -> None:
    """The write-ahead receipt must exist before Encord is touched."""

    storage = FakeStorage()

    class MutationChecksReceipt(FakeUserClient):
        def create_collection(self, **kwargs):
            planned = storage.written(CURATE_RECEIPT_URI)
            assert planned["status"] == "planned"
            assert planned["preset_name"] == "npa-curate-run-2"
            assert planned["filters"][0]["metric"] == "width"
            return super().create_collection(**kwargs)

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(fake_uuid(30), "seed.png", None)]
    client = MutationChecksReceipt(folders=[folder])
    client.pending_curate_items = [FakeItem(fake_uuid(31), "a.png", None)]
    receipt = run_curate(**_curate_kwargs(storage, client, workflow_run="run-2"))
    assert receipt.status == "done"
    assert [uri for _, uri in storage.uploads] == [CURATE_RECEIPT_URI, CURATE_RECEIPT_URI]


def test_run_curate_reuses_existing_collection(_fast_polling) -> None:
    existing = FakeCollection([])
    existing.name = "keepers-new"
    existing.pending = [FakeItem(fake_uuid(33), "c.png", None)]
    client, _ = _curate_client(collection=existing)
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client))
    assert receipt.collection_created is False
    assert receipt.collection_uuid == str(existing.uuid)
    assert receipt.items_selected == 1
    assert client.created_collections == []


def test_run_curate_refuses_a_populated_collection(_fast_polling) -> None:
    """A stale selection would read as this run's; fail closed instead."""

    existing = FakeCollection([FakeItem(fake_uuid(36), "old.png", None)])
    existing.name = "keepers-new"
    client, _ = _curate_client(collection=existing)
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="already holds items"):
        run_curate(**_curate_kwargs(storage, client))
    assert client.created_presets == []
    assert existing.preset_calls == []
    assert storage.written(CURATE_RECEIPT_URI)["status"] == "failed"


def test_run_curate_reissues_evaluation_until_indexing_catches_up(_fast_polling) -> None:
    # Observed live: add_preset_items evaluates once, and items pushed moments
    # earlier are not metric-indexed yet — the re-issue loop must recover.
    client, _ = _curate_client()
    client.pending_curate_items = [FakeItem(fake_uuid(35), "late.png", None)]
    client.curate_reveal_after_calls = 3
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client, poll_seconds=5.0))
    assert receipt.items_selected == 1
    (collection,) = client.created_collections
    assert len(collection.preset_calls) >= 3


def test_run_curate_zero_selection_fails_closed_with_receipt(_fast_polling) -> None:
    client, _ = _curate_client()
    client.pending_curate_items = []
    storage = FakeStorage()
    with pytest.raises(EncordToolError) as excinfo:
        run_curate(
            **_curate_kwargs(
                storage,
                client,
                filters=["brightness:0.2:0.8"],
                poll_seconds=0.05,
            )
        )
    message = str(excinfo.value)
    assert "selected 0 items" in message
    # The diagnostic names the computed-metric cause when one is in play.
    assert "quality metrics have been computed" in message
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "empty"
    assert payload["items_selected"] == 0 and payload["items_total"] == 1
    # The transient preset is gone even though the run failed.
    assert payload["preset_deleted"] is True and len(client.deleted_presets) == 1


def test_run_curate_deletes_the_preset_when_evaluation_raises() -> None:
    """A crash after create_preset must not leave the transient preset behind."""

    class ExplodingCollection(FakeCollection):
        def add_preset_items(self, filter_preset) -> None:
            raise RuntimeError("502 from Encord")

    class ExplodingClient(FakeUserClient):
        def create_collection(self, *, top_level_folder_uuid, name, description=""):
            collection = ExplodingCollection([])
            collection.name = name
            self.created_collections.append(collection)
            return collection

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(fake_uuid(30), "seed.png", None)]
    client = ExplodingClient(folders=[folder])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="502 from Encord"):
        run_curate(**_curate_kwargs(storage, client))
    (preset,) = client.created_presets
    assert client.deleted_presets == [str(preset.uuid)]
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed" and payload["preset_deleted"] is True
    assert payload["preset_uuid"] == str(preset.uuid)


def test_run_curate_records_a_failed_preset_delete(_fast_polling) -> None:
    class StickyPresets(FakeUserClient):
        def delete_preset(self, preset_uuid):
            raise RuntimeError("403 preset delete")

    folder = FakeFolder(name="src")
    folder.folder_items = [FakeItem(fake_uuid(30), "seed.png", None)]
    client = StickyPresets(folders=[folder])
    client.pending_curate_items = [FakeItem(fake_uuid(31), "a.png", None)]
    receipt = run_curate(**_curate_kwargs(FakeStorage(), client))
    assert receipt.status == "done"
    assert receipt.preset_deleted is False  # cleanup by prefix is owed


def test_run_curate_empty_folder_fails_fast_before_any_scaffolding() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="src")])  # no folder items
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="contains no storage items"):
        run_curate(**_curate_kwargs(storage, client))
    assert client.created_collections == []
    assert client.created_presets == []
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed"
    assert payload["items_total"] == 0
    assert "contains no storage items" in payload["error"]


def test_run_curate_unknown_metric_fails_before_any_encord_call() -> None:
    client = FakeUserClient(folders=[FakeFolder(name="src")])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="Unknown filter metric"):
        run_curate(**_curate_kwargs(storage, client, filters=["blur:0:1"]))
    assert client.created_presets == []
    assert client.created_collections == []
    assert storage.objects == {}  # not even a planned receipt


def test_run_curate_missing_folder_writes_receipt_then_raises() -> None:
    client = FakeUserClient(folders=[])
    storage = FakeStorage()
    with pytest.raises(EncordToolError, match="No Encord storage folder"):
        run_curate(**_curate_kwargs(storage, client, folder="absent"))
    payload = storage.written(CURATE_RECEIPT_URI)
    assert payload["status"] == "failed"
    assert "No Encord storage folder" in payload["error"]
    assert client.created_folders == []


