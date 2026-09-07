"""NCore packaging boundaries, immutable routing and source extraction guards."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from npa.deploy import images
from npa.orchestration.npa_workflow.skypilot_render import (
    SkypilotRenderOptions,
    render_setup_for_tool,
    resolve_task_image,
    tool_image_key,
)
from npa.smoke.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]
PACKAGING = ROOT / "npa/docker/workbench/ncore"


def _stager():
    spec = importlib.util.spec_from_file_location(
        "ncore_stager", PACKAGING / "stage_upstream.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ncore_is_public_eligible_but_has_no_automatic_release() -> None:
    assert images.CONTAINER_IMAGE_NAMES["ncore"] == "npa-ncore"
    assert images.is_publicly_redistributable("ncore")
    assert "ncore" in images.UNVALIDATED_PUBLICATION_TOOLS
    assert "ncore" not in images.publicly_publishable_tools()
    with pytest.raises(ValueError, match="no accepted release"):
        images.container_image_for_tool("ncore")
    sha = "a" * 40
    assert images.development_image_for_tool("ncore", git_sha=sha).endswith(
        f"/npa-ncore:dev-{sha}"
    )
    ref = "local.invalid/npa-ncore:dev-" + sha
    build = images.build_and_push_command(ref)
    assert "--push" not in build
    assert f"--source-sha {sha} --image {ref}" in build
    assert images.build_and_push_command("local.invalid/npa-ncore:unbuilt") == ""


def test_conversion_has_pinned_cpu_setup_without_vendor_reinstallation() -> None:
    assert tool_image_key("workbench.nurec.convert_colmap") == "ncore"
    setup = render_setup_for_tool(
        "workbench.nurec.convert_colmap", config={}, options=SkypilotRenderOptions()
    )
    assert "/opt/venv/bin/python /opt/ncore/bin/verify-packaging.py" in setup
    assert "/tmp/npa-python" in setup
    for floating_install in (
        "pip install",
        "nvidia-ncore",
        "apt-get",
        "NPA_SRC_OVERLAY",
    ):
        assert floating_install not in setup
    assert tool_image_key("workbench.nurec.visualize") == "rerun-viewer"
    digest = "local.invalid/npa-ncore@sha256:" + "b" * 64
    assert (
        resolve_task_image(
            "workbench.nurec.convert_colmap",
            {"cpus": 2},
            options=SkypilotRenderOptions(
                image_overrides={"workbench.nurec.convert_colmap": digest}
            ),
        )
        == digest
    )


def test_golden_is_honest_about_its_evidence() -> None:
    spec = load_manifest()["ncore"]
    assert spec.golden_eval.kind == "build-import"
    assert spec.golden_eval.gpu == "none"
    assert spec.golden_eval.status == "needs-image-update"


def test_final_path_exposes_skypilot_bootstrap_service() -> None:
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    final_path = dockerfile.split("ENV PATH=", 1)[1].split()[0].split(":")
    assert "/usr/sbin" in final_path
    assert "/sbin" in final_path


@pytest.mark.parametrize(
    "argv",
    [
        ["--output-dir", "a directory", "colmap-v4", "--include-3d-points"],
        ["--root-dir", "$(touch never)", "--", "*", "semi;colon", ""],
    ],
)
def test_native_launcher_preserves_argv_and_exit_status(
    tmp_path: Path, argv: list[str]
) -> None:
    target = tmp_path / "capture argv"
    target.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\nexit 17\n')
    target.chmod(0o755)
    launcher = tmp_path / "launcher"
    launcher.write_text(
        (PACKAGING / "colmap-convert")
        .read_text()
        .replace("/opt/venv/bin/python", f'"{target}"')
    )
    result = subprocess.run(
        ["bash", str(launcher), *argv], capture_output=True, check=False
    )
    assert result.returncode == 17
    assert result.stdout.decode().split("\0")[:-1] == [
        "-I",
        "-B",
        "-m",
        "npa.workflows.ncore_runtime",
        "converter",
        *argv,
    ]


def test_archive_hash_mismatch_fails_before_extraction(tmp_path: Path) -> None:
    (tmp_path / "ncore.tar.gz").write_bytes(b"untrusted source")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _stager().stage({"ncore": {"sha256": "0" * 64}}, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("relative", ["../escape.py", "ncore/../../escape.py"])
def test_archive_paths_cannot_escape_source_root(tmp_path: Path, relative: str) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo("ncore-" + "a" * 40 + "/" + relative)
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    payload = data.getvalue()
    (tmp_path / "ncore.tar.gz").write_bytes(payload)
    with pytest.raises(ValueError, match="unsafe archive path"):
        _stager().stage(
            {
                "ncore": {
                    "revision": "a" * 40,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            },
            tmp_path / "out",
            tmp_path,
        )


def test_source_selection_excludes_media_models_tests_and_other_converters() -> None:
    selected = _stager().selected
    for path in (
        "ncore/data/v4/__init__.py",
        "ncore/impl/data/v4/compat.py",
        "tools/debug.py",
        "tools/data_converter/cli.py",
        "tools/data_converter/colmap/converter.py",
        "LICENSE",
        "deps/pycolmap/fix-python3-map.patch",
    ):
        assert selected(path, "ncore")
    for path in (
        "ncore/impl/data/util_test.py",
        "ncore/tests/data.py",
        "data/capture.jpg",
        "ncore/weights.pt",
        "ncore/sample.png",
        "tools/data_converter/waymo/converter.py",
        ".git/config",
        "docs/example.usdz",
    ):
        assert not selected(path, "ncore")
    assert selected("LICENSE.txt", "pycolmap")


def test_pins_preserve_trueprice_reader_and_upstream_patch() -> None:
    lock = json.loads((PACKAGING / "source-lock.json").read_text())
    assert lock["ncore"]["revision"] == "59c698d206da92b406a4f72619fce3b3a2c64bfd"
    assert lock["ncore"]["target"] == "//tools/data_converter/colmap:convert"
    assert lock["pycolmap"]["revision"] == "fe7a7c45df803b6c391777e349f0d8d65d39d777"
    assert lock["pycolmap"]["patch"] == "deps/pycolmap/fix-python3-map.patch"
    for filename in ("runtime-requirements.lock", "build-requirements.lock"):
        text = (PACKAGING / filename).read_text()
        assert (
            "pycolmap==" not in text and "torch==" not in text and "nvidia-" not in text
        )


def test_builder_exports_committed_context_without_publication() -> None:
    text = (PACKAGING / "build.sh").read_text()
    assert 'archive "$SOURCE_SHA" npa/src/npa npa/docker/workbench/ncore' in text
    assert '--build-arg "SOURCE_SHA=$SOURCE_SHA"' in text
    assert "--push" not in text
    assert "--stage-catalog" not in text


def test_reader_sentinel_keeps_unsigned_max_on_numpy2(tmp_path: Path) -> None:
    import numpy as np

    source = tmp_path / "scene_manager.py"
    source.write_text(
        "import numpy as np\nclass SceneManager:\n    INVALID_POINT3D = np.uint64(-1)\n"
    )
    _stager().patch_numpy_sentinel(source)
    namespace = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert namespace["SceneManager"].INVALID_POINT3D == np.iinfo(np.uint64).max
    assert "np.uint64(-1)" not in source.read_text()
    with pytest.raises(ValueError, match="sentinel source changed"):
        _stager().patch_numpy_sentinel(source)


DOWNSAMPLE_SOURCE = (
    "                    cameras[ncore_camera_id] = ColmapCamera(\n"
    "                        camera_id=ncore_camera_id,\n"
    "                        colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],\n"
    "                        image_path=parent_dir / image_root,\n"
)

MASK_SOURCE = """                if mask_path is not None:
                    mask_array = np.array(PILImage.open(str(mask_path)).convert("L"), dtype=np.uint8)
                    generic_data["mask"] = mask_array
                    masks_found += 1
"""


def test_mask_patch_is_narrow_and_asserts_source_drift(tmp_path):
    source = tmp_path / "converter.py"
    source.write_text(DOWNSAMPLE_SOURCE + MASK_SOURCE)
    _stager().patch_downsample_masks(source)
    assert source.read_text().startswith(DOWNSAMPLE_SOURCE)
    assert "PILImage.Resampling.NEAREST" in source.read_text()
    assert "source_image.size != source_size" in source.read_text()
    for changed in (
        source.read_text(),
        MASK_SOURCE * 2,
        MASK_SOURCE.replace('"L"', '"RGB"'),
    ):
        source.write_text(changed)
        with pytest.raises(ValueError, match="downsample mask source changed"):
            _stager().patch_downsample_masks(source)
        assert source.read_text() == changed


@pytest.mark.parametrize("count", [0, 1, 2])
def test_text_reader_patch_rejects_source_drift_without_writing(tmp_path, count):
    source = tmp_path / "scene_manager.py"
    changed = (
        "    def _load_cameras_txt(self, input_file):\n        pass\n\n    #-----\n"
    ) * count
    source.write_text(changed)
    with pytest.raises(ValueError, match="text reader source changed"):
        _stager().patch_colmap_text_readers(source)
    assert source.read_text() == changed


def test_downsample_patch_is_narrow_and_asserts_source_drift(tmp_path):
    source = tmp_path / "converter.py"
    original_camera = "colmap_camera=self.scene_manager.cameras[imdata[k].camera_id],\n"
    source.write_text(original_camera + DOWNSAMPLE_SOURCE)
    _stager().patch_downsample_camera(source)
    assert source.read_text().startswith(original_camera)
    assert (
        "colmap_camera=camera,  # NPA: use the current downsample camera"
        in source.read_text()
    )
    with pytest.raises(ValueError, match="downsample.*source changed"):
        _stager().patch_downsample_camera(source)
    source.write_text(DOWNSAMPLE_SOURCE * 2)
    with pytest.raises(ValueError, match="downsample.*source changed"):
        _stager().patch_downsample_camera(source)
    assert source.read_text() == DOWNSAMPLE_SOURCE * 2


def test_staging_records_postpatch_converter_inventory(monkeypatch, tmp_path):
    module = _stager()
    lock = {
        "ncore": {"revision": "a" * 40},
        "pycolmap": {
            "revision": "b" * 40,
            "patch": "deps/pycolmap/fix-python3-map.patch",
        },
    }
    contents = {
        "ncore": {
            "tools/data_converter/colmap/converter.py": DOWNSAMPLE_SOURCE + MASK_SOURCE,
            "deps/pycolmap/fix-python3-map.patch": "synthetic patch boundary",
        },
        "pycolmap": {"pycolmap/scene_manager.py": "INVALID_POINT3D = np.uint64(-1)\n"},
    }
    for component, files in contents.items():
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, text in files.items():
                member = tarfile.TarInfo(
                    f"{component}-{lock[component]['revision']}/{name}"
                )
                member.size = len(text.encode())
                archive.addfile(member, io.BytesIO(text.encode()))
        payload = buffer.getvalue()
        (tmp_path / f"{component}.tar.gz").write_bytes(payload)
        lock[component]["sha256"] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)

    # This synthetic archive isolates stage ordering/inventory from the separately
    # exercised real reader patch (the archive has no complete reader methods).
    def patch_text_reader(path):
        assert "np.uint64(-1)" not in path.read_text()
        path.write_text(path.read_text() + "# text reader patched\n")

    monkeypatch.setattr(module, "patch_colmap_text_readers", patch_text_reader)
    output = tmp_path / "staged"
    module.stage(lock, output, tmp_path)
    relative = "ncore/tools/data_converter/colmap/converter.py"
    patched = (output / relative).read_bytes()
    assert b"colmap_camera=camera," in patched
    assert b"PILImage.Resampling.NEAREST" in patched
    inventory = json.loads((output / "source-inventory.json").read_text())
    assert inventory["files"][relative] == hashlib.sha256(patched).hexdigest()
    assert (
        inventory["files"][relative]
        != hashlib.sha256((DOWNSAMPLE_SOURCE + MASK_SOURCE).encode()).hexdigest()
    )
    reader = "pycolmap/pycolmap/scene_manager.py"
    assert (output / reader).read_text().endswith("# text reader patched\n")
    assert (
        inventory["files"][reader]
        == hashlib.sha256((output / reader).read_bytes()).hexdigest()
    )
