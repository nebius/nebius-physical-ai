"""Synthetic author tests for the external harness; these are not GPU evidence."""

from pathlib import Path

import pytest
from npa.orchestration.npa_workflow.run_state import RuntimeRunState

import derived_controls as controls


def synthetic_wave(step, sequence, spec, options, source):
    return controls.runtime.WaveAttempt(
        key=controls.runtime.wave_key([step], group="", sequence_number=sequence),
        states=[step.state],
        kind="serial",
        job_id=f"synthetic-{sequence}",
        status="succeeded",
        sky_status="SUCCEEDED",
        outputs=step.outputs,
        workflow_sha256=controls.runtime._workflow_identity(spec),
        source_sha256=source,
        image_digest=controls.runtime._image_identity(options),
    ).to_dict()


def object_inventory(objects):
    return [
        {
            "relative": p.relative_to(objects).as_posix(),
            "sha256": controls.digest(p.read_bytes()),
            "bytes": len(p.read_bytes()),
        }
        for p in objects.rglob("*")
        if p.is_file()
    ]


@pytest.fixture
def synthetic_bundle(tmp_path):
    spec_path = Path(controls.__file__).with_name("selector-fixture.yaml")
    spec = controls.load_spec(spec_path)
    spec.config["prefix"] = "synthetic"
    options = controls.selector_options()
    run_id, source = "synthetic-author-check", "c" * 64
    steps = controls.build_plan(spec, run_id=run_id).steps
    state = RuntimeRunState(
        workflow=spec.name,
        run_id=run_id,
        api_version=spec.api_version,
        status="succeeded",
        plan_fingerprint=controls.runtime.plan_fingerprint(spec, run_id=run_id),
        waves=[
            synthetic_wave(step, i, spec, options, source)
            for i, step in enumerate(steps, 1)
        ],
    )
    objects = tmp_path / "baseline"
    controls.write_json(objects / "npa-workflow/runtime.json", state.to_dict())
    for step in steps:
        name = step.outputs[0]["uri"].removeprefix("s3://example-bucket/synthetic/")
        controls.write_json(objects / name, {"synthetic": True})
    inventory = object_inventory(objects)
    return {
        "spec": spec,
        "options": options,
        "run_id": run_id,
        "source_uri": "s3://example-bucket/synthetic-source/" + source,
        "bucket": "example-bucket",
        "prefix": "synthetic",
        "object_root": objects,
        "entries": controls.verified_objects(objects, inventory),
    }


def test_synthetic_author_exercises_all_nine_derived_paths(synthetic_bundle, tmp_path):
    results = controls.run_bundle(synthetic_bundle, tmp_path / "copied-cases")
    assert len(results) == 9
    assert all(item["status"] == "passed" for item in results)
    assert all(not any(item["provider_calls"].values()) for item in results)
    assert all(
        item["output_checks"] == 0
        for item in results
        if "sha256" in item["case"] or "image_digest" in item["case"]
    )


@pytest.mark.parametrize("fault", ["body", "size", "extra", "symlink"])
def test_inventory_rejects_modified_readback(synthetic_bundle, tmp_path, fault):
    root = synthetic_bundle["object_root"]
    entries = list(synthetic_bundle["entries"].values())
    path = root / entries[0]["relative"]
    if fault == "body":
        path.write_bytes(b"corrupt")
    elif fault == "size":
        entries[0] = {**entries[0], "bytes": -1}
    elif fault == "extra":
        (root / "unlisted").write_text("unverified")
    else:
        (root / "outside").symlink_to(tmp_path)
    with pytest.raises(controls.EvidenceFailure):
        controls.verified_objects(root, entries)


@pytest.mark.parametrize("relative", ["../outside", "/absolute"])
def test_copy_paths_reject_scope_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        controls.bounded_path(tmp_path, relative)


def test_provider_trap_records_forbidden_calls():
    trap = controls.ProviderTrap()
    with pytest.raises(RuntimeError, match="forbidden provider boundary"):
        trap.seam("launch")()
    assert trap.calls == {"launch": 1}
    with pytest.raises(AssertionError):
        trap.assert_unused()


def test_actual_toolref_renderer_mapping_and_reorder():
    result = controls.check_selector_rendering(
        Path(controls.__file__).with_name("selector-fixture.yaml")
    )
    assert result["distinct_images"] == 2
    assert result["reordered_mapping_unchanged"]
    assert result["swapped_mapping_changes_identity"]
