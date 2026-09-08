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


@pytest.mark.parametrize("failure", ["", "subset", "corrupt_shard"])
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

    class S3:
        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(objects[Key])}

        def head_object(self, **kwargs):
            return {"ContentLength": 100}

    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", lambda *a, **k: S3()
    )
    reached = []
    monkeypatch.setattr(
        helpers, "_download_nurec_proof", lambda *args: reached.append("download")
    )
    monkeypatch.setattr(
        helpers,
        "_assert_nurec_downstream_proof",
        lambda *args, **kwargs: reached.append("decode"),
    )
    if failure:
        with pytest.raises(AssertionError):
            helpers.assert_nurec_colmap_live_outputs(
                bucket="unit-bucket", run_id="unit"
            )
    else:
        helpers.assert_nurec_colmap_live_outputs(bucket="unit-bucket", run_id="unit")
    assert reached == ([] if failure else ["download", "decode"])


# These are synthetic format fixtures, not GPU reconstruction acceptance evidence.
def _write_synthetic_usdz(path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdUtils = pytest.importorskip("pxr.UsdUtils")
    layer = path.with_suffix(".usda")
    stage = Usd.Stage.CreateNew(str(layer))
    scene = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(scene.GetPrim())
    points = UsdGeom.Points.Define(stage, "/Scene/SyntheticPoints")
    points.CreatePointsAttr([(0, 0, 0), (1, 0, 0)])
    stage.GetRootLayer().Save()
    assert UsdUtils.CreateNewUsdzPackage(str(layer), str(path))


def _write_proof_document(root, relative, payload):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_synthetic_conversion_report(root, helpers):
    _write_proof_document(
        root,
        "ncore/sequence/conversion.json",
        {
            "schema_version": 1,
            "status": "ok",
            "engine": "nvidia-ncore-colmap",
            "converter": {"revision": "59c698d206da92b406a4f72619fce3b3a2c64bfd"},
            "counts": {"images": 518, "poses": 518, "cameras": 3, "points": 163453},
            "source": {"archive_sha256": helpers.NUREC_COLMAP_SHA256},
            "poses_component_group": "npa_rig",
        },
    )


def _write_synthetic_nurec_lineage(root, helpers):
    _write_proof_document(
        root,
        "source/attribution.json",
        {
            "revision": helpers.NUREC_COLMAP_REVISION,
            "sha256": helpers.NUREC_COLMAP_SHA256,
            "license": "CC-BY-4.0",
            "dataset": "synthetic-fixture",
            "selected_capture": "struktur28",
        },
    )
    _write_synthetic_conversion_report(root, helpers)
    _write_proof_document(
        root,
        "ncore/sequence/npa-rig.json",
        {
            "status": "ok",
            "reference_camera": "camera2",
            "pose_count": 2,
            "cameras": ["camera1", "camera2"],
            "poses_component_group": "npa_rig",
        },
    )
    _write_proof_document(
        root,
        "reports/final.json",
        {
            "has_usdz": True,
            "has_novel_views": True,
            "has_rrd": True,
        },
    )


@pytest.fixture
def downstream_run(tmp_path, helpers):
    from PIL import Image

    root = tmp_path / "nurec-colmap-reconstruct"
    _write_synthetic_nurec_lineage(root, helpers)
    (root / "reconstruction").mkdir()
    (root / "reconstruction/parsed.yaml").write_text(
        "dataset:\n  camera_ids: [camera1, camera2]\n"
    )
    (root / "reconstruction/metrics.yaml").write_text(
        yaml.safe_dump(
            {
                "aggregated_metrics": {
                    name: {"aggregation_method": "mean", "value": value}
                    for name, value in {
                        "test/psnr": 22.6,
                        "test/ssim": 0.83,
                        "test/lpips": 0.19,
                    }.items()
                }
            }
        )
    )
    for camera in ("camera1", "camera2"):
        directory = root / "novel_views" / camera
        directory.mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (32, 24), (30 + index * 20, 40, 60)).save(
                directory / f"{index:06}.png"
            )
    return root


def _write_proof_rrd(root):
    from npa.workflows.data_factory_viz import build_run_rrd

    build_run_rrd(
        str(root), str(root / "reports/sim2real.rrd"), app_id="neural-reconstruction"
    )


def test_actual_usdz_aggregated_metrics_media_and_rrd_pass(helpers, downstream_run):
    _write_synthetic_usdz(downstream_run / "reconstruction/last.usdz")
    _write_proof_rrd(downstream_run)
    helpers._assert_nurec_downstream_proof(
        downstream_run, recording_id=downstream_run.name
    )


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"test/psnr": True, "test/ssim": 0.8, "test/lpips": 0.2},
        {"test/psnr": float("nan"), "test/ssim": 0.8, "test/lpips": 0.2},
        {"test/psnr": 23, "test/ssim": float("inf"), "test/lpips": 0.2},
        {"test/psnr": 0, "test/ssim": 0.8, "test/lpips": 0.2},
        {"test/psnr": 23, "test/ssim": 1.1, "test/lpips": 0.2},
        {"test/psnr": 23, "test/ssim": 0.8, "test/lpips": -1},
    ],
)
def test_live_metrics_reject_missing_bool_nonfinite_and_invalid_values(
    helpers, tmp_path, metrics
):
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump(metrics))
    with pytest.raises(AssertionError):
        helpers._assert_nurec_quality_metrics(path)


@pytest.mark.parametrize(
    "text", ["test: [invalid", "[]", "test/psnr: 23\ntest/ssim: 0.8\n"]
)
def test_live_metrics_reject_malformed_or_incomplete_yaml(helpers, tmp_path, text):
    path = tmp_path / "metrics.yaml"
    path.write_text(text)
    with pytest.raises((AssertionError, yaml.YAMLError)):
        helpers._assert_nurec_quality_metrics(path)


def test_live_metrics_support_flat_nre_format(helpers, tmp_path):
    path = tmp_path / "metrics.yaml"
    path.write_text("test:\n  psnr: 23\n  ssim: 0.8\n  lpips: 0.2\n")
    assert helpers._assert_nurec_quality_metrics(path)["test/psnr"] == 23


@pytest.mark.parametrize(
    "failure", ["corrupt", "no_layer", "invalid_layer", "empty_scene"]
)
def test_live_usdz_rejects_invalid_package_or_empty_scene(helpers, tmp_path, failure):
    import zipfile

    Tf = pytest.importorskip("pxr.Tf")
    path = tmp_path / "last.usdz"
    if failure == "corrupt":
        path.write_bytes(b"not a USDZ")
    elif failure == "no_layer":
        with zipfile.ZipFile(path, "w") as package:
            package.writestr("report.json", "{}")
    elif failure == "invalid_layer":
        with zipfile.ZipFile(path, "w") as package:
            package.writestr("scene.usda", "not a USD layer")
    else:
        Usd = pytest.importorskip("pxr.Usd")
        UsdUtils = pytest.importorskip("pxr.UsdUtils")
        layer = tmp_path / "empty.usda"
        stage = Usd.Stage.CreateNew(str(layer))
        stage.GetRootLayer().Save()
        assert UsdUtils.CreateNewUsdzPackage(str(layer), str(path))
    with pytest.raises((AssertionError, zipfile.BadZipFile, Tf.ErrorException)):
        helpers._assert_nurec_usdz(path)


@pytest.mark.parametrize("failure", ["image", "video", "no_frames"])
def test_live_novel_views_require_decodable_images_and_video(
    helpers, downstream_run, failure
):
    import av
    from PIL import UnidentifiedImageError

    root = downstream_run / "novel_views"
    if failure == "image":
        (root / "camera1/000000.png").write_bytes(b"not an image")
    elif failure == "video":
        (root / "camera1.mp4").write_bytes(b"not a video")
    else:
        for frame in root.rglob("*.png"):
            frame.unlink()
    with pytest.raises(
        (AssertionError, UnidentifiedImageError, av.error.InvalidDataError)
    ):
        helpers._assert_nurec_novel_media(downstream_run)


def test_live_novel_video_is_fully_decoded(helpers, downstream_run):
    import base64

    (downstream_run / "novel_views/camera1.mp4").write_bytes(
        base64.b64decode(helpers._CONDITIONED_COSMOS_MP4_B64)
    )
    assert helpers._assert_nurec_novel_media(downstream_run) == {
        "camera1": {0, 1},
        "camera2": {0, 1},
    }


@pytest.mark.parametrize(
    "failure",
    ["corrupt", "wrong_run", "lineage_changed", "metrics_changed", "frame_absent"],
)
def test_live_rrd_requires_identity_lineage_metrics_and_actual_frame_rows(
    helpers, downstream_run, failure
):
    _write_proof_rrd(downstream_run)
    recording_id = downstream_run.name
    if failure == "corrupt":
        (downstream_run / "reports/sim2real.rrd").write_bytes(b"not a recording")
    elif failure == "wrong_run":
        recording_id = "another-run"
    elif failure == "lineage_changed":
        path = downstream_run / "ncore/sequence/npa-rig.json"
        payload = json.loads(path.read_text())
        payload["reference_camera"] = "camera1"
        path.write_text(json.dumps(payload))
    elif failure == "metrics_changed":
        (downstream_run / "reconstruction/metrics.yaml").write_text(
            "test: {psnr: 24, ssim: 0.9, lpips: 0.2}"
        )
    else:
        (downstream_run / "novel_views/camera1/000001.png").unlink()
    frames = helpers._assert_nurec_novel_media(downstream_run)
    with pytest.raises(AssertionError):
        helpers._assert_nurec_rrd(downstream_run, recording_id, frames)


def _publish_synthetic_conversion(root, helpers):
    sequence = root / "ncore/sequence"
    _write_proof_document(
        root,
        "ncore/sequence/sequence.json",
        {
            "version": "v4",
            "component_stores": [{"path": "camera.zarr.itar"}],
        },
    )
    (sequence / "camera.zarr.itar").write_bytes(
        b"synthetic shard for hash verification only"
    )
    path = sequence / "conversion.json"
    report = json.loads(path.read_text())
    report["source"].update(counts=report["counts"], origin_points_filtered=0)
    report["options"] = {"dataset_root": "struktur28"}
    report["members"] = [
        {
            "path": name,
            "bytes": (sequence / name).stat().st_size,
            "sha256": hashlib.sha256((sequence / name).read_bytes()).hexdigest(),
        }
        for name in ("sequence.json", "camera.zarr.itar", "npa-rig.json")
    ]
    path.write_text(json.dumps(report))


@pytest.fixture
def published_proof(helpers, downstream_run, monkeypatch):
    _write_synthetic_usdz(downstream_run / "reconstruction/last.usdz")
    _publish_synthetic_conversion(downstream_run, helpers)
    _write_proof_rrd(downstream_run)
    root = "npa-workflow-e2e/unit/nurec-colmap-reconstruct/"
    bodies = {
        root + path.relative_to(downstream_run).as_posix(): path.read_bytes()
        for path in downstream_run.rglob("*")
        if path.is_file()
    }
    downloads = []

    class S3:
        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(bodies[Key])}

        def get_paginator(self, name):
            return self

        def paginate(self, *, Bucket, Prefix):
            yield {
                "Contents": [{"Key": key} for key in bodies if key.startswith(Prefix)]
            }

        def download_file(self, bucket, key, filename):
            downloads.append(key)
            Path(filename).write_bytes(bodies[key])

    monkeypatch.setattr(
        "npa.clients.project_credentials.s3_client_for_project", lambda *a, **k: S3()
    )
    return root, bodies, downloads


def test_live_entrypoint_reads_published_bodies_through_real_decoders(
    helpers, published_proof
):
    root, _, downloads = published_proof
    helpers.assert_nurec_colmap_live_outputs(bucket="unit-bucket", run_id="unit")
    for relative in (
        "reconstruction/last.usdz",
        "reconstruction/metrics.yaml",
        "reports/sim2real.rrd",
        "novel_views/camera1/000001.png",
        "novel_views/camera2/000001.png",
    ):
        assert root + relative in downloads


@pytest.mark.parametrize(
    "relative",
    [
        "reconstruction/last.usdz",
        "reconstruction/metrics.yaml",
        "reconstruction/parsed.yaml",
        "reports/sim2real.rrd",
    ],
)
def test_success_booleans_cannot_replace_missing_artifact_bodies(
    helpers, published_proof, relative
):
    root, bodies, _ = published_proof
    del bodies[root + relative]
    assert all(json.loads(bodies[root + "reports/final.json"]).values())
    with pytest.raises(KeyError):
        helpers.assert_nurec_colmap_live_outputs(bucket="unit-bucket", run_id="unit")


def test_live_rrd_rejects_text_only_novel_view_markers(helpers, downstream_run):
    import rerun as rr

    from npa.workflows.data_factory_viz import _load_stage_docs

    record = rr.RecordingStream(
        "neural-reconstruction", recording_id=downstream_run.name
    )
    record.save(str(downstream_run / "reports/sim2real.rrd"))
    for entity, text in _load_stage_docs(downstream_run).items():
        record.log(entity, rr.TextDocument(text), static=True)
    record.log(
        "novel_view/camera1",
        rr.TextDocument("frame and EncodedImage:blob are only text"),
    )
    record.disconnect()
    frames = helpers._assert_nurec_novel_media(downstream_run)
    with pytest.raises(AssertionError, match="no image data"):
        helpers._assert_nurec_rrd(downstream_run, downstream_run.name, frames)


def test_live_rrd_rejects_decodable_stale_frame_pixels(helpers, downstream_run):
    from PIL import Image

    _write_proof_rrd(downstream_run)
    # Preserve IDs, dimensions, file formats, source lineage and metrics while
    # changing only a published render after the RRD captured the old pixels.
    Image.new("RGB", (32, 24), (220, 10, 20)).save(
        downstream_run / "novel_views/camera1/000001.png"
    )
    frames = helpers._assert_nurec_novel_media(downstream_run)
    with pytest.raises(AssertionError, match="image bytes differ"):
        helpers._assert_nurec_rrd(downstream_run, downstream_run.name, frames)


@pytest.mark.parametrize(
    "entity",
    [
        "/provenance/source",
        "/provenance/conversion",
        "/provenance/rig",
        "/novel_view/camera2",
    ],
)
def test_live_rrd_rejects_missing_decoded_entities(helpers, downstream_run, entity):
    from rerun.recording import Recording, load_recording

    _write_proof_rrd(downstream_run)
    path = downstream_run / "reports/sim2real.rrd"
    chunks = [
        chunk
        for chunk in load_recording(path).chunks()
        if str(chunk.entity_path) != entity
    ]
    Recording.from_chunks(chunks, "neural-reconstruction", downstream_run.name).save(
        path
    )
    frames = helpers._assert_nurec_novel_media(downstream_run)
    with pytest.raises(AssertionError):
        helpers._assert_nurec_rrd(downstream_run, downstream_run.name, frames)
