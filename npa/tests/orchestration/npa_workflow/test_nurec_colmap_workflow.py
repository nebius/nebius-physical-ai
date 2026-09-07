"""COLMAP workflow handoffs and real-input live-matrix wiring (offline)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow.interpreter import build_plan
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "workflows/testing/nurec-colmap-reconstruct.yaml"
NRE_IMAGE = (
    "nvcr.io/nvidia/nre/nre-ga@sha256:"
    "97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284"
)


def test_nurec_consumer_fetches_immutable_ncore_reader() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_setup_for_tool,
    )

    setup = render_setup_for_tool(
        "workbench.nurec.rig_from_sequence",
        config={},
        options=SkypilotRenderOptions(),
    )
    assert "nvidia_ncore-19.5.1-py3-none-any.whl#sha256=" in setup
    assert "a753f81470ba1b35567cbca26794a7f9ceefe04ec306b962a52dd18dc988fe29" in setup
    assert "'nvidia-ncore'" not in setup


@pytest.fixture
def helpers(monkeypatch):
    path = ROOT / "npa/tests/e2e/npa_workflow_live_helpers.py"
    spec = importlib.util.spec_from_file_location("colmap_live_helpers", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for variable in (
        "NPA_E2E_NUREC_COLMAP_ARCHIVE",
        "NPA_E2E_FORCE_ACCELERATORS",
        "NPA_E2E_ACCELERATOR_REMAP",
        "NPA_WORKFLOW_GPU_ACCELERATOR",
    ):
        monkeypatch.delenv(variable, raising=False)
    return module


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_conversion_hands_exact_portable_sequence_to_existing_nre():
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="colmap-test")
    assert [step.state for step in plan.steps] == [
        "convert",
        "reconstruct",
        "render",
        "visualize",
        "finalize",
    ]
    convert, reconstruct, render, visualize, finalize = plan.steps
    from npa.workbench.ncore_staging import (
        DEFAULT_COLMAP_CACHE_DIR,
        DEFAULT_COLMAP_SCRATCH_DIR,
    )

    assert _flag(convert.argv, "--cache-dir") == str(DEFAULT_COLMAP_CACHE_DIR)
    assert _flag(convert.argv, "--scratch-dir") == str(DEFAULT_COLMAP_SCRATCH_DIR)
    assert convert.argv[:4] == ["npa", "workbench", "nurec", "convert-colmap"]
    output = _flag(convert.argv, "--output-path")
    assert output.endswith("/ncore/sequence/")
    assert _flag(reconstruct.argv, "--ncore-uri") == output
    assert {item["uri"] for item in convert.outputs} == {
        output + "sequence.json",
        output + "conversion.json",
    }
    assert reconstruct.inputs == [{"uri": output, "schema": ""}]
    assert _flag(convert.argv, "--dataset-root") == "struktur28"
    assert _flag(convert.argv, "--rig-mode") == "derive"
    assert _flag(reconstruct.argv, "--poses-component-group") == "npa_rig"
    assert _flag(reconstruct.argv, "--max-epochs") == "0"
    assert (
        _flag(reconstruct.argv, "--config-name")
        == "configs/experimental/3dgut/3dgut_colmap.yaml"
    )
    assert _flag(render.argv, "--artifact-uri") == _flag(
        reconstruct.argv, "--output-uri"
    )
    assert visualize.tool_ref == "workbench.nurec.visualize"
    assert finalize.tool_ref == "workbench.nurec.finalize"


def test_cpu_converter_and_proprietary_rtx_stages_are_separate():
    spec = yaml.safe_load(SPEC.read_text())
    states, resources = spec["states"], spec["resources"]
    assert "accelerators" not in resources[states["convert"]["resources"]]
    for state in ("reconstruct", "render"):
        profile = resources[states[state]["resources"]]
        assert profile["accelerators"] == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        assert profile["image"] == NRE_IMAGE
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC.name)
    assert case.tier == "gpu" and not case.plan_only and not case.rotation_skip
    assert case.max_wait_seconds == 0
    assert "HF_TOKEN" not in case.secret_envs
    assert case.image_tool == ""
    assert dict(case.image_overrides)["workbench.nurec.convert_colmap"] == "ncore"


def test_render_keeps_converter_digest_and_nre_runtime_in_their_own_stages(
    helpers, monkeypatch
):
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://unit-bucket/npa-source")
    spec = load_spec(SPEC)
    converter = "registry.example/npa-ncore@sha256:" + "a" * 64
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="render-colmap"),
        run_id="render-colmap",
        options=SkypilotRenderOptions(
            registry="registry.example",
            image_overrides={"workbench.nurec.convert_colmap": converter},
            materialize_registry_secrets=False,
        ),
    )
    tasks = [doc for doc in yaml.safe_load_all(rendered) if doc and "run" in doc]
    assert len(tasks) == 5
    assert tasks[0]["resources"]["image_id"] == "docker:" + converter
    assert "accelerators" not in tasks[0]["resources"]
    for task in tasks[1:3]:
        assert task["resources"]["image_id"] == "docker:" + NRE_IMAGE
        assert (
            task["resources"]["accelerators"]
            == "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        )
    assert "npa-rerun-viewer" in tasks[3]["resources"]["image_id"]


@pytest.mark.parametrize("local", [False, True])
def test_live_seed_uploads_complete_verified_archive_and_attribution(
    helpers, monkeypatch, tmp_path, local
):
    body = b"complete synthetic archive bytes for the source-staging unit seam"
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(helpers, "NUREC_COLMAP_SHA256", digest)
    downloads, uploads, sidecars = [], [], []

    def download(path):
        downloads.append(path)
        path.write_bytes(body)

    class S3:
        def upload_file(self, filename, bucket, key):
            uploads.append((Path(filename).read_bytes(), bucket, key))

        def put_object(self, **kwargs):
            sidecars.append(kwargs)

    monkeypatch.setattr(helpers, "_download_nurec_colmap_archive", download)
    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", lambda *a, **k: S3()
    )
    if local:
        archive = tmp_path / "source.zip"
        archive.write_bytes(body)
        monkeypatch.setenv("NPA_E2E_NUREC_COLMAP_ARCHIVE", str(archive))
    helpers.seed_live_workflow_inputs(
        spec_name=SPEC.name, bucket="unit-bucket", run_id="seed-run"
    )
    assert len(downloads) == (0 if local else 1)
    assert uploads == [
        (
            body,
            "unit-bucket",
            "npa-workflow-e2e/seed-run/nurec-colmap-reconstruct/source/struktur28_colmap.zip",
        )
    ]
    attribution = json.loads(sidecars[0]["Body"])
    assert attribution["sha256"] == digest
    assert attribution["revision"] == helpers.NUREC_COLMAP_REVISION
    assert attribution["license"] == "CC-BY-4.0"
    assert attribution["source_counts"] == {
        "images": 518,
        "cameras": 3,
        "points": 163453,
    }
    materialized = helpers.materialize_live_spec(
        tmp_path, SPEC.name, bucket="unit-bucket", run_id="seed-run"
    )
    config = yaml.safe_load(materialized.read_text())["config"]
    uri = (
        config["colmap_input_uri"]
        .replace("{{config.bucket}}", config["bucket"])
        .replace("{{config.prefix}}", config["prefix"])
    )
    assert uri == "s3://unit-bucket/" + uploads[0][2]


def test_wrong_source_hash_fails_before_any_upload(helpers, monkeypatch, tmp_path):
    source = tmp_path / "wrong.zip"
    source.write_bytes(b"truncated or different source")
    monkeypatch.setenv("NPA_E2E_NUREC_COLMAP_ARCHIVE", str(source))
    with pytest.raises(
        pytest.fail.Exception, match="differs from the pinned complete dataset"
    ):
        helpers._seed_nurec_colmap_source(object(), bucket="unit-bucket", prefix="unit")


def test_download_uses_immutable_public_colmap_object(helpers, monkeypatch, tmp_path):
    urls = []

    def download(url, output, *, allowed_hosts, redirect_hosts):
        urls.append(url)
        assert allowed_hosts == frozenset({"huggingface.co"})
        assert "cas-bridge.xethub.hf.co" in redirect_hosts
        output.write(b"entire public object")

    monkeypatch.setattr("npa._public_https.download_public_https", download)
    output = tmp_path / "source.zip"
    helpers._download_nurec_colmap_archive(output)
    assert urls == [
        "https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP/resolve/"
        "2521064a3af6ab1c1caa2ba1b01ddde7eecded69/colmap/struktur28_colmap.zip"
    ]
    assert output.read_bytes() == b"entire public object"


@pytest.mark.parametrize(
    "variable",
    [
        "NPA_E2E_FORCE_ACCELERATORS",
        "NPA_E2E_ACCELERATOR_REMAP",
        "NPA_WORKFLOW_GPU_ACCELERATOR",
    ],
)
def test_live_matrix_refuses_generic_gpu_remapping(
    helpers, monkeypatch, tmp_path, variable
):
    monkeypatch.setenv(variable, "B200:1")
    with pytest.raises(pytest.fail.Exception, match=variable):
        helpers.materialize_live_spec(
            tmp_path, SPEC.name, bucket="unit-bucket", run_id="unit"
        )


@pytest.mark.parametrize("failure", ["", "subset", "corrupt_shard", "missing_rrd"])
def test_live_output_verifier_rejects_subset_or_corrupt_conversion(
    helpers, monkeypatch, failure
):
    root = "npa-workflow-e2e/unit/nurec-colmap-reconstruct/"
    sequence = root + "ncore/sequence/"
    meta = {"version": "v4", "component_stores": [{"path": "camera.zarr.itar"}]}
    files = {
        "sequence.json": json.dumps(meta).encode(),
        "camera.zarr.itar": b"verified shard",
        "npa-rig.json": b"{}",
    }
    counts = {"images": 518, "poses": 518, "cameras": 3, "points": 163453}
    report = {
        "status": "ok",
        "engine": "nvidia-ncore-colmap",
        "converter": {"revision": "59c698d206da92b406a4f72619fce3b3a2c64bfd"},
        "source": {
            "archive_sha256": helpers.NUREC_COLMAP_SHA256,
            "counts": counts,
            "origin_points_filtered": 0,
        },
        "options": {"dataset_root": "struktur28"},
        "poses_component_group": "npa_rig",
        "counts": counts,
        "members": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }
    if failure == "subset":
        counts.update(images=59, poses=59, cameras=2)
    if failure == "corrupt_shard":
        files["camera.zarr.itar"] = b"corrupt shard!"
    files["conversion.json"] = json.dumps(report).encode()
    objects = {sequence + name: data for name, data in files.items()}
    objects[root + "source/attribution.json"] = json.dumps(
        {"revision": helpers.NUREC_COLMAP_REVISION, "license": "CC-BY-4.0"}
    ).encode()
    objects[root + "reports/final.json"] = json.dumps(
        {"has_usdz": True, "has_novel_views": True, "has_rrd": failure != "missing_rrd"}
    ).encode()

    class S3:
        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(objects[Key])}

        def head_object(self, **kwargs):
            return {"ContentLength": 100}

    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", lambda *a, **k: S3()
    )
    if failure:
        with pytest.raises(AssertionError):
            helpers.assert_nurec_colmap_live_outputs(
                bucket="unit-bucket", run_id="unit"
            )
    else:
        helpers.assert_nurec_colmap_live_outputs(bucket="unit-bucket", run_id="unit")
