"""Exercise public observation through generic reconstruction with synthetic native boundaries."""

import copy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from admission_fixture import _native_report, build_observation, set_outcomes
from npa.clients.storage import StorageClient, StoragePreconditionFailed
from npa.workflows.field_failure import artifacts, reference_demo as demo
from npa.workflows.field_failure import reference_demo_publication as publication
from npa.workflows.field_failure.native_artifacts import _archive
from npa.workflows.field_failure.reference_demo_inputs import reference_metrics
from npa.workflows.navigation.artifacts import materialize, publish, write_json
from npa.workflows.navigation.contract import read_recipe


class MemoryStorage:
    def __init__(self, objects):
        self.objects = objects
        self.s3 = self

    def get_object(self, *, Bucket, Key):
        from botocore.exceptions import ClientError

        uri = "s3://" + Bucket + "/" + Key
        if uri not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[uri])}

    def put_bytes_conditional(self, data, uri, *, if_none_match, **kwargs):
        assert if_none_match is True
        if uri in self.objects:
            raise StoragePreconditionFailed("fixture already exists")
        self.objects[uri] = data

    def read_bytes_with_etag(self, uri):
        data = self.objects.get(uri)
        return None if data is None else (data, artifacts._digest(data))

    def download_directory(self, uri, target):
        for name, data in list(self.objects.items()):
            if name.startswith(uri.rstrip("/") + "/"):
                path = Path(target) / name[len(uri.rstrip("/")) + 1 :]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


@pytest.fixture
def baseline_publication(tmp_path, monkeypatch):
    from npa.workflows.field_failure.reference_demo_inputs import prepare_warehouse

    storage = MemoryStorage({})
    monkeypatch.setattr(StorageClient, "from_environment", lambda: storage)
    image = "registry.example.invalid/native@sha256:" + "a" * 64
    source = tmp_path / "baseline-input"
    prepare_warehouse(source, image=image, count=4, iterations=1500, steps=300)
    (source / "final-cases.json").unlink()
    args = SimpleNamespace(
        stage="baseline",
        output_root="s3://fixture/public/baseline",
        navigation_image=image,
    )
    prefix = args.output_root + "/baseline-input"
    publish(source, prefix)
    monkeypatch.setenv("NPA_TASK_IMAGE", image)
    return SimpleNamespace(args=args, storage=storage, source=source, prefix=prefix)


def test_baseline_consumes_completed_s3_attempt_before_native_training(
    baseline_publication, monkeypatch, tmp_path
):
    from npa.workflows.navigation import stages

    fixture, calls = baseline_publication, []
    expected_recipe = (fixture.source / "recipe.json").read_bytes()
    expected_scene = (fixture.source / "scene.usdz").read_bytes()
    assert fixture.prefix + "/recipe.json" not in fixture.storage.objects

    def native(stage, source, output):
        calls.append(stage)
        assert (source / "recipe.json").read_bytes() == expected_recipe
        assert (source / "scene.usdz").read_bytes() == expected_scene
        assert not (source / "final-cases.json").exists()
        recipe = read_recipe(source)
        assert recipe.iterations == 1500 and recipe.episode_steps == 300
        (output / "policy.pt").write_bytes(b"synthetic native boundary checkpoint")
        report = {"fixture": "native training boundary", "iterations": 1500}
        write_json(output / "training.json", report)
        return report

    monkeypatch.setattr(stages, "_native", native)
    report = demo.run_reference_stage(fixture.args)
    assert calls == ["train"] and report["iterations"] == 1500
    prepared = materialize(
        fixture.args.output_root + "/baseline-prepared", tmp_path / "prepared"
    )
    trained = materialize(
        fixture.args.output_root + "/baseline-training", tmp_path / "trained"
    )
    assert (prepared / "recipe.json").read_bytes() == expected_recipe
    assert (trained / "recipe.json").read_bytes() == expected_recipe
    assert (trained / "scene.usdz").read_bytes() == expected_scene


@pytest.mark.parametrize(
    "fault",
    ["missing-completion", "missing-claim", "claim-mismatch", "recipe", "scene"],
)
def test_baseline_rejects_incomplete_or_tampered_input_before_native_training(
    baseline_publication, monkeypatch, fault
):
    from npa.workflows.navigation import stages

    fixture, calls = baseline_publication, []
    objects, prefix = fixture.storage.objects, fixture.prefix
    completion = json.loads(objects[prefix + "/completion.json"])
    if fault.startswith("missing-"):
        del objects[prefix + "/" + fault.removeprefix("missing-") + ".json"]
    elif fault == "claim-mismatch":
        changed = {**completion, "attempt": "attempts/" + "f" * 32}
        objects[prefix + "/claim.json"] = json.dumps(changed).encode()
    else:
        attempt = prefix + "/" + completion["attempt"]
        name = "recipe.json" if fault == "recipe" else "scene.usdz"
        objects[attempt + "/" + name] += b"\nchanged"
    monkeypatch.setattr(stages, "_native", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="incomplete|immutable|original local"):
        demo.run_reference_stage(fixture.args)
    assert calls == []
    assert not any("/baseline-prepared/" in name for name in objects)
    assert not any("/baseline-training/" in name for name in objects)


@pytest.fixture
def pipeline(tmp_path, monkeypatch, navigation):
    observed = build_observation(tmp_path, monkeypatch)
    store = navigation.store
    storage = MemoryStorage(store.objects)
    monkeypatch.setattr(StorageClient, "from_environment", lambda: storage)
    monkeypatch.setattr(artifacts, "_storage", lambda: storage)
    monkeypatch.setattr(publication, "_storage", lambda: storage)
    args = SimpleNamespace(output_root="s3://fixture/public/demo", run_id="demo")
    plan = _plan_inputs(tmp_path, args, observed, navigation)
    artifacts._publish(args.output_root + "/reference-plan.json", plan)
    _baseline_output(tmp_path, args, observed)
    calls = _native_boundary(monkeypatch, plan)
    return SimpleNamespace(
        args=args,
        plan=plan,
        store=store,
        calls=calls,
        requests=navigation.requests,
        storage=storage,
    )


def _plan_inputs(tmp_path, args, observed, navigation):
    store, recipe = navigation.store, observed.recipe.model_dump()
    plan = copy.deepcopy(observed.plan)
    source = tmp_path / "observation-input"
    source.mkdir()
    shutil.copyfile(observed.root / recipe["scene_file"], source / recipe["scene_file"])
    recipe["initial_checkpoint"] = None
    write_json(source / "recipe.json", recipe)
    archive = tmp_path / "observation-input.tar"
    _archive(source, archive)
    plan["failure_observation"]["input"] = store.asset(
        args.output_root + "/inputs/observation.tar", archive.read_bytes()
    )
    plan["protocol"] = store.asset(
        args.output_root + "/inputs/protocol.json",
        {
            "schema_version": "npa.field-failure.native-protocol.v1",
            "navigation_image": recipe["image"],
            **{
                key: recipe[key]
                for key in (
                    "task",
                    "adapter_module",
                    "adapter_sha256",
                    "source_bundle_sha256",
                )
            },
        },
    )
    plan["capture"] = navigation.bundle["captures"][0]["asset"]
    plan["final"] = navigation.bundle["held_out"][0]["asset"]
    plan["final_seeds"] = navigation.bundle["held_out"][0]["seeds"]
    plan["adapters"] = copy.deepcopy(navigation.bundle["adapters"])
    for adapter in plan["adapters"].values():
        adapter["runtime_image"] = recipe["image"]
    plan["metrics"] = reference_metrics(recipe["episode_steps"])
    return plan


def _baseline_output(tmp_path, args, observed):
    output = tmp_path / "trained-baseline"
    output.mkdir()
    shutil.copyfile(observed.root / "baseline.pt", output / "policy.pt")
    write_json(
        output / "training.json",
        {
            "iterations": observed.recipe.iterations,
            "runtime": {"robot_population": 2},
            "policy_parameter_delta_l2": 1.0,
            "isolation": {"passed": True},
            "checkpoint_sha256": observed.recipe.initial_checkpoint.sha256,
        },
    )
    publish(output, args.output_root + "/baseline-training")


def _native_boundary(monkeypatch, plan):
    from npa.workflows.navigation import stages

    calls = []

    def native(stage, source, output):
        assert stage == "evaluate-checkpoint"
        calls.append(stage)
        shutil.copytree(source, output, dirs_exist_ok=True)
        recipe = read_recipe(source)
        office = plan["cohorts"]["regions"]["training"]["office"]
        report = _native_report(output, recipe, office, monkeypatch)
        set_outcomes(
            SimpleNamespace(root=output, recipe=recipe, report=report), success=False
        )
        return report

    monkeypatch.setattr(stages, "_native", native)
    return calls


def _run(pipeline, stage):
    return demo.run_reference_stage(SimpleNamespace(**vars(pipeline.args), stage=stage))


def test_observe_admit_seal_and_reconstruct_bind_the_same_measured_failures(pipeline):
    _run(pipeline, "observe-failures")
    admission = _run(pipeline, "admit-capture")
    reference = _run(pipeline, "seal")
    assert reference["admission"]["sha256"] == artifacts._digest(
        artifacts._encode(admission)
    )
    assert pipeline.calls == ["evaluate-checkpoint"]
    _run(pipeline, "validate")
    _run(pipeline, "reconstruct")
    stage, request = pipeline.requests[-1]
    assert stage == "reconstruct"
    assert request["captures"][0]["asset"] == admission["capture"]
    record = pipeline.store.read(
        pipeline.args.output_root + "/loop/demo/reconstruction.json"
    )
    assert record["scenes"][0]["capture_sha256"] == admission["capture"]["sha256"]
    admission["failures"] = []
    pipeline.store.asset(reference["admission"]["uri"], admission)
    with pytest.raises(ValueError, match="SHA-256"):
        _run(pipeline, "train")
    assert all(stage != "train" for stage, _ in pipeline.requests)


def test_completed_observation_retry_reverifies_without_another_gpu_call(pipeline):
    before = _run(pipeline, "observe-failures")
    assert _run(pipeline, "observe-failures") == before
    assert pipeline.calls == ["evaluate-checkpoint"]
    prefix = pipeline.args.output_root + "/failure-observation"
    del pipeline.store.objects[prefix + "/completion.json"]
    with pytest.raises(ValueError, match="incomplete immutable attempt"):
        _run(pipeline, "observe-failures")
    assert pipeline.calls == ["evaluate-checkpoint"]


def test_retained_observation_bytes_cannot_change_during_retry(pipeline):
    _run(pipeline, "observe-failures")
    prefix = pipeline.args.output_root + "/failure-observation"
    completed = pipeline.store.read(prefix + "/completion.json")
    uri = prefix + "/" + completed["attempt"] + "/trajectory.json"
    pipeline.store.objects[uri] += b" "
    with pytest.raises(ValueError, match="published bytes differ"):
        _run(pipeline, "observe-failures")
    assert pipeline.calls == ["evaluate-checkpoint"]
