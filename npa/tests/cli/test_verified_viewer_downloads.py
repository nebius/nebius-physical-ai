"""Generated viewer cache controls; transport fixtures are not policy artifacts."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from npa.workflows.sim2real import publication

from .test_stage14_twentieth_agent_controls import _import_rendered_backend


def _publication_objects(good):
    root = "s3://demo-bucket/runs/run-a"
    generation = "a" * 64
    objects, targets = {}, []
    for suffix, data in (
        ("reports/sim2real-report.json", b"{}"),
        ("reports/sim2real.rrd", good),
        ("components/stage_14.json", b"{}"),
    ):
        immutable = (
            f"{root}/components/history/stage_14/{hashlib.sha256(data).hexdigest()}.json"
            if suffix.startswith("components/")
            else f"{root}/reports/generations/{generation}/{Path(suffix).name}"
        )
        objects[immutable.removeprefix("s3://demo-bucket/")] = data
        targets.append(
            {
                "uri": f"{root}/{suffix}",
                "state": "present",
                "immutable_uri": immutable,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    targets.append({"uri": f"{root}/reports/sim2real.mcap", "state": "absent"})
    objects["runs/run-a/reports/sim2real.rrd"] = good
    journal = publication._journal_bytes(
        transaction_id=generation,
        attempt_id="b" * 32,
        state="committed",
        objects=targets,
    )
    return objects, journal


class _ViewerStore:
    def __init__(self, context):
        self.context = context
        self.download_payload = context.good

    def get_object(self, *, Bucket, Key, **conditions):
        assert Bucket == "demo-bucket"
        if Key.endswith(".sim2real-publication.json"):
            if not self.context.journaled:
                raise KeyError(Key)
            data = self.context.journal
        else:
            data = self.context.objects[Key]
        if "IfMatch" in conditions:
            assert conditions["IfMatch"] == hashlib.sha256(data).hexdigest()
        return {"Body": io.BytesIO(data)}

    def download_file(self, bucket, object_key, destination):
        assert bucket == "demo-bucket" and object_key == self.context.artifact.key
        Path(destination).write_bytes(self.download_payload)

    def head_object(self, *, Bucket, Key):
        data = self.context.objects[Key]
        return {"ContentLength": len(data), "ETag": hashlib.sha256(data).hexdigest()}


def _viewer_context(module, tmp_path, journaled):
    good = b"explicit verified transport fixture"
    objects, journal = _publication_objects(good)
    key = (
        f"runs/run-a/reports/generations/{'a' * 64}/sim2real.rrd"
        if journaled
        else "runs/run-a/reports/sim2real.rrd"
    )
    artifact = module.Artifact(
        run_id="run-a",
        key=key,
        s3_uri=f"s3://demo-bucket/{key}",
        size=len(good),
        last_modified="",
        render="rerun",
        inline=False,
        namespace="runs",
        relative_key=key.removeprefix("runs/run-a/"),
        source_etag=hashlib.sha256(good).hexdigest(),
    )
    context = SimpleNamespace(
        module=module,
        good=good,
        hostile=b"explicit rejected transport fixture",
        objects=objects,
        journal=journal,
        journaled=journaled,
        artifact=artifact,
        directory=tmp_path / "verified-cache",
    )
    context.store = _ViewerStore(context)
    return context


@pytest.fixture(params=[False, True])
def viewer_context(request, tmp_path, monkeypatch):
    name = f"npa_rendered_verified_viewer_{request.param}"
    module = _import_rendered_backend(monkeypatch, tmp_path, module_name=name)
    try:
        yield _viewer_context(module, tmp_path, request.param)
    finally:
        sys.modules.pop(name, None)


def _download(context, store=None):
    return context.module._download_verified_viewer_artifact(
        store or context.store,
        "demo-bucket",
        context.artifact,
        recordings_dir=context.directory,
    )


def test_rejected_viewer_download_preserves_exposed_verified_bytes(viewer_context):
    context = viewer_context
    path = _download(context)
    assert path.read_bytes() == context.good
    assert hashlib.sha256(context.good).hexdigest() in path.name
    context.store.download_payload = context.hostile
    with pytest.raises((context.module.PublicationConflict, HTTPException)):
        _download(context)
    assert path.read_bytes() == context.good
    assert list(context.directory.iterdir()) == [path.parent]
    context.store.download_payload = context.good
    second = _download(context)
    assert second != path and second.read_bytes() == path.read_bytes()
    context.module.release_verified_recording(second)
    assert path.read_bytes() == context.good


def test_concurrent_rejected_download_cannot_corrupt_verified_cache(viewer_context):
    context = viewer_context
    path = _download(context)
    rejected = _ViewerStore(context)
    rejected.download_payload = context.hostile
    with ThreadPoolExecutor(max_workers=2) as executor:
        accepted = executor.submit(_download, context, _ViewerStore(context))
        hostile = executor.submit(_download, context, rejected)
        second = accepted.result()
        assert second != path and second.read_bytes() == path.read_bytes()
        with pytest.raises((context.module.PublicationConflict, HTTPException)):
            hostile.result()
    assert path.read_bytes() == context.good
    context.module.release_verified_recording(second)
    assert list(context.directory.iterdir()) == [path.parent]


def _install_route_inventory(context, monkeypatch):
    module = context.module
    inventory = [context.artifact]
    if context.journaled:
        for key, data in context.objects.items():
            if key == context.artifact.key or key.endswith("/reports/sim2real.rrd"):
                continue
            inventory.append(
                module.Artifact(
                    run_id="run-a",
                    key=key,
                    s3_uri=f"s3://demo-bucket/{key}",
                    size=len(data),
                    last_modified="",
                    render="json",
                    inline=True,
                    namespace="runs",
                    relative_key=key.removeprefix("runs/run-a/"),
                    source_etag=hashlib.sha256(data).hexdigest(),
                )
            )
    monkeypatch.setattr(module, "RECORDINGS_DIR", context.directory)
    monkeypatch.setattr(
        module, "_agent_artifact_s3_client", lambda: (context.store, {})
    )
    monkeypatch.setattr(module, "_load_session_run_if_known", lambda **_: None)
    monkeypatch.setattr(module, "_begin_agent_artifact_access", lambda: object())
    monkeypatch.setattr(module, "_end_agent_artifact_access", lambda: None)
    monkeypatch.setattr(
        module,
        "_authorize_exact_run_ref_source",
        lambda **_: ("demo-bucket", "fixture-project", "runs"),
    )
    monkeypatch.setattr(
        module,
        "_load_selected_run_artifacts",
        lambda **_: ("demo-bucket", "fixture-project", "runs", inventory),
    )
    monkeypatch.setattr(
        module,
        "_resolved_artifact_for_content",
        lambda *_args, **_: ("run-a", "demo-bucket", context.artifact),
    )
    monkeypatch.setattr(module, "_load_state", lambda: {})
    monkeypatch.setattr(
        module,
        "_apply_loaded_artifact",
        lambda **_: pytest.fail("rejected bytes must never reach viewer publication"),
    )


@pytest.mark.parametrize("route", ["run", "artifact"])
def test_both_load_routes_reject_before_viewer_publication(
    viewer_context, monkeypatch, route
):
    context = viewer_context
    path = _download(context)
    _install_route_inventory(context, monkeypatch)
    context.store.download_payload = context.hostile
    body = {
        "run_id": "run-a",
        "run_ref": context.module.encode_run_ref("demo-bucket", "runs", "run-a"),
        "resource_bucket": "demo-bucket",
        "project_id": "fixture-project",
        "resolved_prefix": "runs",
        "source_selected": True,
        "key": context.artifact.key,
    }
    handler = (
        context.module.sim_viz_load_run
        if route == "run"
        else context.module.sim_viz_load_artifact
    )
    with pytest.raises(HTTPException) as rejected:
        handler(body)
    assert rejected.value.status_code == 409
    assert path.read_bytes() == context.good


@pytest.mark.parametrize("route", ["run", "artifact"])
@pytest.mark.parametrize("fail_apply", [False, True])
def test_load_route_releases_only_its_input_after_apply(
    viewer_context, monkeypatch, route, fail_apply
):
    context = viewer_context
    held = _download(context)
    _install_route_inventory(context, monkeypatch)
    consumed = []
    published = context.directory / "published.rrd"

    def apply(**kwargs):
        path = kwargs["local_path"]
        consumed.append(path)
        assert path != held and path.read_bytes() == context.good
        published.write_bytes(path.read_bytes())
        if fail_apply:
            raise HTTPException(status_code=422, detail="fixture apply failure")
        return {}

    monkeypatch.setattr(context.module, "_apply_loaded_artifact", apply)
    monkeypatch.setattr(context.module, "_sim_viz_load_response", lambda *_a, **_k: {})
    body = {
        "run_id": "run-a",
        "run_ref": context.module.encode_run_ref("demo-bucket", "runs", "run-a"),
        "resource_bucket": "demo-bucket",
        "project_id": "fixture-project",
        "resolved_prefix": "runs",
        "source_selected": True,
        "key": context.artifact.key,
    }
    handler = (
        context.module.sim_viz_load_run
        if route == "run"
        else context.module.sim_viz_load_artifact
    )
    if fail_apply:
        with pytest.raises(HTTPException):
            handler(body)
    else:
        assert handler(body)["ok"]
    assert len(consumed) == 1 and not consumed[0].exists()
    assert held.read_bytes() == context.good
    assert published.read_bytes() == context.good
