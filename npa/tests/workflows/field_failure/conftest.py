"""Provide explicitly synthetic navigation evidence using isolated in-memory S3."""

import io
import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from npa.clients.storage import StorageClient
from npa.workflows.field_failure import artifacts
from npa.workflows.field_failure.stages import run_stage


class _Store:
    def __init__(self):
        self.objects = {}
        self.lock = threading.Lock()

    def get_object(self, *, Bucket, Key):
        uri = f"s3://{Bucket}/{Key}"
        if uri not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[uri])}

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        uri = f"s3://{Bucket}/{Key}"
        assert ContentType == "application/json" and IfNoneMatch == "*"
        with self.lock:
            if uri in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                )
            self.objects[uri] = Body
        return {"ETag": artifacts._digest(Body)}

    def asset(self, uri, content):
        data = content if isinstance(content, bytes) else artifacts._encode(content)
        self.objects[uri] = data
        return {"uri": uri, "sha256": artifacts._digest(data)}

    def read(self, uri):
        return json.loads(self.objects[uri])


def _fixture_scenario(store, scenario, group, name):
    return {
        "scenario_id": scenario,
        "group_id": group,
        "asset": store.asset(
            "s3://fixture/input/" + name, ("synthetic " + name).encode()
        ),
    }


def _fixture_metric(name, direction, improvement, maximum):
    return {
        "name": name,
        "direction": direction,
        "minimum_improvement": improvement,
        "maximum_regression": 0.0,
        "minimum": 0.0,
        "maximum": maximum,
    }


def _fixture_adapters(code_sha):
    return {
        name: {
            "entrypoint": "synthetic_navigation_fixture:" + name,
            "source_sha256": code_sha,
            "runtime_image": "registry.example.invalid/navigation@sha256:" + "a" * 64,
        }
        for name in ["reconstruct", "train", "evaluate"]
    }


def _bundle(store, code_sha):
    held = _fixture_scenario(store, "unseen", "held-site", "held.usd")
    held["seeds"] = [7, 11]
    return {
        "schema_version": "npa.field-failure.bundle.v1",
        "task": "navigation",
        "baseline": {
            "policy_id": "existing",
            "checkpoint": store.asset(
                "s3://fixture/input/base.pt", b"synthetic baseline"
            ),
        },
        "baseline_training_groups": ["prior-training-site"],
        "captures": [
            _fixture_scenario(store, "failure", "training-site", "capture.mcap")
        ],
        "held_out": [held],
        "protocol": store.asset("s3://fixture/input/protocol.json", {"fixture": True}),
        "adapters": _fixture_adapters(code_sha),
        "primary_metric": "progress",
        "metrics": [
            _fixture_metric("progress", "higher", 0.1, 1.0),
            _fixture_metric("collisions", "lower", 0.0, 100.0),
        ],
    }


def _reconstruct(store, request):
    prefix = request["output_prefix"]
    return {
        "schema_version": "npa.field-failure.reconstruction.v1",
        "bundle_sha256": request["bundle_sha256"],
        "adapter": request["adapter"],
        "attempt_id": request["attempt_id"],
        "inputs_sha256": request["inputs_sha256"],
        "scenes": [
            {
                "scenario_id": c["scenario_id"],
                "group_id": c["group_id"],
                "capture_sha256": c["asset"]["sha256"],
                "asset": store.asset(
                    prefix + c["scenario_id"] + ".usd", b"synthetic reconstructed scene"
                ),
            }
            for c in request["captures"]
        ],
        "evidence": store.asset(
            prefix + "conversion.json", {"fixture": "reconstruction"}
        ),
    }


def _train(store, request):
    prefix = request["output_prefix"]
    return {
        "schema_version": "npa.field-failure.training.v1",
        "bundle_sha256": request["bundle_sha256"],
        "adapter": request["adapter"],
        "attempt_id": request["attempt_id"],
        "inputs_sha256": request["inputs_sha256"],
        "reconstruction_sha256": request["reconstruction_sha256"],
        "initial_checkpoint_sha256": request["baseline"]["checkpoint"]["sha256"],
        "candidate": {
            "policy_id": "candidate",
            "checkpoint": store.asset(prefix + "candidate.pt", b"synthetic candidate"),
        },
        "training_scenario_ids": [s["scenario_id"] for s in request["scenes"]],
        "training_group_ids": sorted({s["group_id"] for s in request["scenes"]}),
        "training_scene_sha256": sorted(
            {s["asset"]["sha256"] for s in request["scenes"]}
        ),
        "evidence": store.asset(prefix + "learning.json", {"fixture": "training"}),
    }


def _episode(store, request, scenario, seed):
    candidate = request["policy"]["policy_id"] == "candidate"
    key = f"{scenario['scenario_id']}-{seed}"
    measurements = {
        "scenario_id": scenario["scenario_id"],
        "scene_sha256": scenario["asset"]["sha256"],
        "seed": seed,
        "status": "completed",
        "termination": "success",
        "steps": 8,
        "metrics": {
            "progress": 0.8 if candidate else 0.5,
            "collisions": 0.0 if candidate else 1.0,
        },
    }
    trace = {
        **measurements,
        "schema_version": "npa.field-failure.episode.v1",
        "bundle_sha256": request["bundle_sha256"],
        "checkpoint_sha256": request["policy"]["checkpoint"]["sha256"],
        "protocol_sha256": request["protocol"]["sha256"],
        "trajectory": store.asset(
            request["output_prefix"] + key + ".mcap",
            f"synthetic trajectory {candidate} {seed}".encode(),
        ),
    }
    return {
        **measurements,
        "evidence": store.asset(request["output_prefix"] + key + ".json", trace),
    }


def _evaluate(store, request):
    return {
        "schema_version": "npa.field-failure.evaluation.v1",
        "bundle_sha256": request["bundle_sha256"],
        "protocol_sha256": request["protocol"]["sha256"],
        "adapter": request["adapter"],
        "attempt_id": request["attempt_id"],
        "inputs_sha256": request["inputs_sha256"],
        "policy": request["policy"],
        "episodes": [
            _episode(store, request, s, seed)
            for s in request["held_out"]
            for seed in s["seeds"]
        ],
    }


def _install_fixture_adapter(monkeypatch, tmp_path, store, requests):
    module_name = "synthetic_navigation_fixture"
    source = tmp_path / (module_name + ".py")
    source.write_text(
        "\n".join(
            f"def {name}(request):\n    from synthetic_navigation_runtime import {name} as call\n    return call(request)"
            for name in ["reconstruct", "train", "evaluate", "substitute"]
        )
    )
    runtime = ModuleType("synthetic_navigation_runtime")
    for name, implementation in [
        ("reconstruct", _reconstruct),
        ("train", _train),
        ("evaluate", _evaluate),
    ]:

        def call(request, implementation=implementation, name=name):
            requests.append((name, request))
            return implementation(store, request)

        setattr(runtime, name, call)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    return source, runtime


@pytest.fixture
def navigation(monkeypatch, tmp_path):
    store, requests = _Store(), []
    storage = StorageClient.__new__(StorageClient)
    storage._s3 = store
    monkeypatch.setattr(artifacts, "_storage", lambda: storage)
    source, runtime = _install_fixture_adapter(monkeypatch, tmp_path, store, requests)
    bundle = _bundle(store, artifacts._digest(source.read_bytes()))
    bundle_uri = "s3://fixture/input/bundle.json"
    pin = store.asset(bundle_uri, bundle)["sha256"]
    root = "s3://fixture/runs/fixture-run"

    def run(stage, adapter=None, image=None):
        chosen = "evaluate" if stage.endswith("evaluate") else stage
        identity = bundle["adapters"].get(chosen, {})
        run_stage(
            stage,
            bundle_uri,
            pin,
            root,
            "fixture-run",
            adapter if adapter is not None else identity.get("entrypoint", ""),
            image if image is not None else identity.get("runtime_image", ""),
        )

    yield SimpleNamespace(
        store=store,
        bundle=bundle,
        bundle_uri=bundle_uri,
        pin=pin,
        root=root,
        run=run,
        source=source,
        runtime=runtime,
        requests=requests,
    )
    sys.modules.pop("synthetic_navigation_fixture", None)


@pytest.fixture
def evaluated(navigation):
    for stage in [
        "validate",
        "reconstruct",
        "train",
        "baseline-evaluate",
        "candidate-evaluate",
    ]:
        navigation.run(stage)
    return navigation
