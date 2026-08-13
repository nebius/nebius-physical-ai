"""Offline tests for the Omniverse-payload scanner's classifier.

The scanner (``npa/scripts/scan_image_omniverse_payload.py``) is the mechanical proof
behind reclassifying the four Isaac images as publicly redistributable: it inspects a
BUILT image's filesystem and layer history rather than trusting a Dockerfile. Running it
needs a registry and multi-GB pulls, so the part that can regress silently - the
classifier - is tested here against synthetic listings, and the live scan runs in
Phase 7 / CI.

Both directions matter, and the tricky direction is the second one: the images
deliberately keep a ``/isaac-sim/python.sh`` shim (~30 call sites already invoke Isaac
through that path, and pods override ENTRYPOINT so the shim is the only reliable
bootstrap trigger), so "grep finds nothing" was never available as a proof. The scanner
has to distinguish Kit payload from our own 40-line shell script.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCANNER = REPO_ROOT / "npa" / "scripts" / "scan_image_omniverse_payload.py"


def _load_scanner():
    """Import the scanner by path (it is a script, not a package module).

    It must be registered in ``sys.modules`` before ``exec_module``: ``@dataclass``
    resolves annotations through ``sys.modules[cls.__module__]``, which is ``None``
    for a module loaded from a spec but never registered.
    """
    spec = importlib.util.spec_from_file_location("npa_payload_scanner", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scanner = _load_scanner()


# Real Kit payload paths, taken from the layout observed in an actual pip Isaac Sim 5.1
# install on an RTX PRO 6000 pod.
PAYLOAD_PATHS = [
    "opt/venv/lib/python3.11/site-packages/isaacsim/kit/kernel/py/omni/ext/_impl/_internal.py",
    "opt/venv/lib/python3.11/site-packages/isaacsim/extscache/omni.graph-1.141.2+69cbf6ad.lx64.r.cp311/omni/graph/core/__init__.py",
    "opt/venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.utils/isaacsim/core/utils/numpy/rotations.py",
    "opt/venv/lib/python3.11/site-packages/isaaclab/apps/isaaclab.python.headless.kit",
    "opt/venv/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/app/app_launcher.py",
    "isaac-sim/kit/libcarb.so",
    "isaac-sim/exts/omni.isaac.core/config/extension.toml",
    "isaac-sim/extscache/omni.kit.window.viewport-1.0.0/PACKAGE-LICENSES/LICENSE",
    "usr/lib/libomni.usd.so",
    "opt/nvidia/omniverse/kit/kernel/plugins/carb.dll",
    "isaac-sim/assets/Isaac/Robots/Franka/franka.usd",
]

# Paths the re-architected images legitimately DO ship.
ALLOWED_PATHS = [
    "isaac-sim/python.sh",
    "isaac-sim/",
    "opt/npa/bin/isaac-python",
    "opt/npa/bin/isaac-bootstrap",
    "opt/npa/docker/workbench/common/isaac_bootstrap.sh",
    "opt/npa/docker/workbench/common/isaac_python.sh",
    "opt/npa/docker/workbench/common/isaac-nvidia-wheels.txt",
    "opt/npa/docker/workbench/common/isaac-oss-deps.txt",
    "opt/npa/docker/workbench/isaac-lab/smoke_functional.py",
    "opt/npa/docker/workbench/isaac-lab/smoke_env.py",
    "opt/isaac-cache/",
    "opt/isaac-cache/v/",
    "workspace/isaaclab/",
    "opt/isaac-lab/",
    # Ordinary, unrelated image content must not trip anything.
    "usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "opt/venv/lib/python3.11/site-packages/torch/__init__.py",
    "opt/sonic/gear_sonic/__init__.py",
    "opt/groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
    "usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0",
]


@pytest.mark.parametrize("path", PAYLOAD_PATHS)
def test_scanner_flags_real_kit_payload(path: str) -> None:
    why = scanner.classify_path(path)
    assert why, f"Kit payload not detected: {path}"


@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_scanner_allows_what_the_images_actually_ship(path: str) -> None:
    why = scanner.classify_path(path)
    assert why is None, (
        f"legitimate path wrongly flagged as Kit payload: {path} ({why})"
    )


def test_the_shim_is_allowed_but_a_kit_tree_at_the_same_root_is_not() -> None:
    """The crux of the design: /isaac-sim holds our shim and nothing else.

    A naive `tar -tf | grep isaac-sim` cannot tell these apart, which is why the scanner
    pairs payload signatures with an explicit allowlist instead.
    """
    assert scanner.classify_path("isaac-sim/python.sh") is None
    assert scanner.classify_path("isaac-sim/kit/kernel/plugins/carb.dll")
    assert scanner.classify_path("isaac-sim/exts/omni.isaac.core/extension.toml")


def test_allowlist_is_small_and_explicit() -> None:
    """An unexpected path must fail closed, so the allowlist must not be broad.

    A prefix like `opt/` or a bare `isaac` substring would let real payload through.
    """
    assert len(scanner.ALLOWED_EXACT) <= 6
    for prefix in scanner.ALLOWED_PREFIXES:
        assert prefix.startswith("opt/npa/docker/workbench/"), prefix
        assert prefix.endswith("/"), f"{prefix} must be a directory prefix"
    # The allowlist must not admit a Kit tree hidden under an allowed prefix.
    assert (
        scanner.classify_path("opt/npa/docker/workbench/common/isaacsim/kit/libcarb.so")
        is None
    )
    # ... which is acceptable only because that prefix is ours and contains no payload;
    # assert the payload signatures themselves still fire outside it.
    assert scanner.classify_path("opt/other/isaacsim/kit/libcarb.so")


HISTORY_BAKING = [
    'RUN pip install --no-cache-dir "isaacsim==5.1.0.0"',
    'RUN /isaac-sim/python.sh -m pip install --no-deps "isaaclab==2.3.2.post1"',
    "FROM nvcr.io/nvidia/isaac-lab:2.3.2",
    "RUN /opt/npa/bin/isaac-bootstrap ensure",
    "RUN isaac_bootstrap.sh warm",
    "ENV OMNI_KIT_ACCEPT_EULA=YES",
    "ENV ISAACSIM_ACCEPT_EULA=YES PRIVACY_CONSENT=Y",
]

HISTORY_FINE = [
    "RUN /opt/npa/docker/workbench/common/install_isaac_runtime_base.sh",
    "RUN /opt/npa/bin/isaac-bootstrap status",
    "COPY docker/workbench/common /opt/npa/docker/workbench/common",
    "ENV ISAAC_LAB_PYTHON=/isaac-sim/python.sh",
    "ENV NPA_ISAAC_CACHE_DIR=/opt/isaac-cache",
    "ENV ISAAC_SIM_VERSION=5.1.0.0 ISAAC_LAB_VERSION=2.3.2.post1",
    "RUN pip install -r /opt/npa/docker/workbench/common/isaac-oss-deps.txt",
    "FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3",
]


@pytest.mark.parametrize("command", HISTORY_BAKING)
def test_scanner_flags_build_layers_that_installed_isaac(command: str) -> None:
    """The image's own recorded history is checked too, so an image built from a
    Dockerfile nobody reviewed is still caught."""
    assert scanner.classify_history(command), f"baking layer not detected: {command}"


@pytest.mark.parametrize("command", HISTORY_FINE)
def test_scanner_allows_runtime_fetch_build_layers(command: str) -> None:
    why = scanner.classify_history(command)
    assert why is None, f"legitimate layer wrongly flagged: {command} ({why})"


def test_report_verdict_and_exit_semantics() -> None:
    report = scanner.ScanReport(image="example:tag", source="registry")
    assert report.clean
    assert report.to_dict()["verdict"] == "clean"
    assert report.to_dict()["scan_complete"] is True

    report.payload_hits.append({"path": "isaac-sim/kit/libcarb.so", "why": "carb"})
    assert not report.clean
    assert report.to_dict()["verdict"] == "omniverse-payload-detected"

    history_only = scanner.ScanReport(image="example:tag", source="registry")
    history_only.history_hits.append(
        {"command": "RUN pip install isaacsim", "why": "x"}
    )
    assert not history_only.clean, "a baking layer alone must fail the scan"


def test_scanner_is_executable_and_self_documenting() -> None:
    assert SCANNER.is_file()
    text = SCANNER.read_text(encoding="utf-8")
    # The reason it cannot simply grep for "isaac" is the single most likely thing for a
    # future reader to try to "simplify"; keep the rationale in the file.
    assert "python.sh" in text and "allowlist" in text.lower()


def test_oci_layout_tarball_scans_root_level_blob_layers(tmp_path: Path) -> None:
    """Docker's containerd image store saves layers as ``blobs/sha256/*``.

    The root-level form must be opened as a nested tar, not counted as one opaque
    outer member and incorrectly declared clean.
    """
    layer_stream = io.BytesIO()
    with tarfile.open(fileobj=layer_stream, mode="w") as layer:
        payload = b"kit"
        member = tarfile.TarInfo("isaac-sim/kit/libcarb.so")
        member.size = len(payload)
        layer.addfile(member, io.BytesIO(payload))
    outer_path = tmp_path / "image.tar"
    with tarfile.open(outer_path, mode="w") as outer:
        payload = layer_stream.getvalue()
        member = tarfile.TarInfo("blobs/sha256/exact-layer")
        member.size = len(payload)
        outer.addfile(member, io.BytesIO(payload))

    paths = list(scanner._iter_tarball(outer_path))

    assert "isaac-sim/kit/libcarb.so" in paths
    assert scanner.classify_path(paths[0])


# --------------------------------------------------------------------------------------
# Gated model weights are a separate licence axis from Omniverse Kit, and the workbench
# rule is the same: never baked. The distinction that matters is a git-LFS POINTER (a
# ~130-byte reference the operator resolves with their own token - the compliant
# arrangement) versus a real tensor. This scanner sees only a tar listing, so it reports
# weight-shaped paths for a human rather than guessing; the authoritative content check
# runs inside the image build, where the bytes exist.
# --------------------------------------------------------------------------------------


def _sonic_dockerfile() -> str:
    return (
        REPO_ROOT / "npa" / "docker" / "workbench" / "sonic" / "Dockerfile"
    ).read_text(encoding="utf-8")


def _instructions_only(dockerfile_text: str) -> str:
    """Drop comment lines. These Dockerfiles document what they deliberately do NOT do, so
    prose naming a removed instruction must not read as that instruction being present."""
    return "\n".join(
        line
        for line in dockerfile_text.splitlines()
        if not line.lstrip().startswith("#")
    )


def test_weight_shaped_paths_are_reported_not_flagged_as_kit_payload() -> None:
    """A .pt file is not Omniverse Kit, so it must not fail the Kit verdict."""
    for path in (
        "opt/sonic/gear_sonic/trl/utils/smplx/body_model/smpl_coco17_J_regressor.pt",
        "opt/sonic/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/"
        "GR00T-WholeBodyControl-Walk.onnx",
    ):
        assert scanner.classify_path(path) is None, path
        assert path.endswith(scanner.WEIGHT_SUFFIXES), path


def test_report_lists_weight_shaped_paths_without_failing() -> None:
    report = scanner.ScanReport(image="example:tag", source="registry")
    report.weight_shaped_paths.append("opt/sonic/x/policy.onnx")
    assert report.clean, (
        "weight-shaped paths are informational, not a Kit-payload failure"
    )
    assert report.to_dict()["weight_shaped_paths"] == ["opt/sonic/x/policy.onnx"]


def test_sonic_build_checks_weights_by_content_not_extension() -> None:
    """Pin the content-based check in sonic's Dockerfile.

    Discovered the hard way: an extension-only check reported eight 'baked weights' in
    npa-sonic that were all ~130-byte git-LFS pointers, which are exactly what SHOULD be
    in the image. Failing on the suffix would have been a false positive that pushed
    someone towards weakening the guard; failing to notice a real tensor would be far
    worse. So the build reports pointers and fails only on real payload.
    """
    dockerfile = _sonic_dockerfile()
    instructions = _instructions_only(dockerfile)
    assert "version https://git-lfs.github.com/spec/v1" in dockerfile, (
        "the weight check must recognise git-LFS pointers by their magic string"
    )
    assert "NPA_SONIC_LFS_POINTERS_ONLY" in dockerfile
    assert (
        "real model weights baked into the image (not LFS pointers, not an "
        in dockerfile
    )
    # And smudging must be disabled, which is what actually keeps the tensors out.
    assert "GIT_LFS_SKIP_SMUDGE=1" in dockerfile, (
        "`git lfs install --system` makes a plain `git checkout` download every tracked "
        "object, so without GIT_LFS_SKIP_SMUDGE=1 the image bakes gated weights"
    )
    # Instructions only: the surrounding comment explains the old `git lfs pull ... ||
    # true` line, so matching raw text would trip on the very prose that documents the fix.
    assert "git lfs pull" not in instructions, (
        "no `git lfs pull` should remain: any pull re-materialises the tensors this "
        "image must not ship"
    )


def test_sonic_does_not_bake_the_gated_groot_policies() -> None:
    """Regression pin for a real, previously-invisible gated-weights leak.

    npa-sonic was shipping decoupled_wbc/.../g1/policy/GR00T-WholeBodyControl-{Walk,
    Balance}.onnx -- from a directory containing a file named "NVIDIA Open Model License"
    -- plus six SMPL-X regressors, because `git lfs install --system` makes `git checkout`
    smudge automatically. Entirely separate from the Omniverse Kit problem, and nothing
    checked for it until the build-time weight assertion was added.
    """
    instructions = _instructions_only(_sonic_dockerfile())
    smudge_line = next(
        line for line in instructions.splitlines() if "GIT_LFS_SKIP_SMUDGE" in line
    )
    clone_index = instructions.index("git clone")
    assert instructions.index(smudge_line) < clone_index, (
        "GIT_LFS_SKIP_SMUDGE must be exported BEFORE the clone/checkout, or the objects "
        "are already downloaded by the time it takes effect"
    )


def test_the_one_allowlisted_source_asset_is_named_and_size_bounded() -> None:
    """The single non-LFS weight-shaped file in gear_sonic is a reviewed exception.

    `coco_aug_dict.pth` is committed directly to git (not LFS) at ~1.8 KiB. The repo's
    dual licence separates Apache-2.0 SOURCE from NVIDIA-Open-Model-License WEIGHTS, and a
    sub-2-KiB keypoint-augmentation lookup table in the source tree is source. That is a
    judgement call, so it must stay a named allowlist entry with a size bound - not a
    pattern that would also admit a real checkpoint - and it must remain visible in the
    diff for a reviewer to object to.
    """
    dockerfile = _sonic_dockerfile()
    assert "ALLOWED_SOURCE_ASSETS" in dockerfile
    assert "gear_sonic/trl/utils/smplx/body_model/coco_aug_dict.pth" in dockerfile
    assert "ALLOWED_SOURCE_ASSET_MAX_BYTES" in dockerfile, (
        "the exception must be size-bounded"
    )
    instructions = _instructions_only(dockerfile)
    # A wildcard or suffix-wide exemption would defeat the whole check.
    for forbidden in ("*.pth", "*.pt", "smplx/**", "suffix in ALLOWED"):
        assert forbidden not in instructions, (
            f"the allowlist must name exact paths, found {forbidden!r}"
        )


def test_history_matching_ignores_comments_inside_heredocs() -> None:
    """A comment in an inlined script must not read as a build-time Isaac install.

    buildkit records the whole RUN, heredocs included, so sonic's own explanatory line
    "# happens on GPU (isaac-bootstrap verify / the golden eval)" landed in the image
    history and made the scanner report a bake against a clean image. A false positive
    here is worse than useless: it would block a legitimate reclassification, and the
    tempting fix is to loosen the pattern until it stops firing.
    """
    command = (
        "RUN python - <<PY\n"
        "    # See install_isaac_runtime_base.sh: a driverless builder reports an empty\n"
        "    # arch list, so the per-device check happens on GPU (isaac-bootstrap verify\n"
        "    # / the golden eval).\n"
        "    print('ok')\n"
        "PY"
    )
    assert scanner.classify_history(command) is None, "prose must not count as a bake"

    # ... but a real invocation on an instruction line still must.
    assert scanner.classify_history(
        command + "\nRUN /opt/npa/bin/isaac-bootstrap ensure"
    )


def test_sonic_excludes_the_robocasa_omniverse_asset_tree() -> None:
    """The third independent finding, pinned so it cannot come back.

    Omniverse 3D assets are a separate licence question from the Kit SDK and from model
    weights, and they are what kept sonic restricted after the wheels were dealt with:
    decoupled_wbc/dexmg drags in RoboCasa's asset library, including
    .../robocasa/models/assets/objects/omniverse/*.{obj,mtl,png} -- NVIDIA Omniverse meshes,
    materials and albedo textures, i.e. exactly the "models and textures" the Isaac Sim
    licence covers. Found by scanning the built image; invisible in the Dockerfile.
    """
    dockerfile = _sonic_dockerfile()
    instructions = _instructions_only(dockerfile)
    assert '"!/decoupled_wbc/dexmg/**"' in instructions, (
        "the sparse checkout must exclude decoupled_wbc/dexmg, which carries the RoboCasa "
        "Omniverse asset library"
    )
    # And the build must prove the absence itself, not rely on someone remembering to scan.
    assert "FATAL: NVIDIA Omniverse assets baked into the image" in dockerfile
    assert "no Omniverse assets" in dockerfile


def test_scanner_flags_omniverse_asset_paths() -> None:
    """The scanner must catch these too - it is what found them in the first place."""
    for path in (
        "opt/sonic/decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/objects/"
        "omniverse/locomanip/cardbox_c1/meshes/cardbox_c1.obj",
        "opt/sonic/decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/objects/"
        "omniverse/locomanip/cardbox_c1/textures/T_Cardbox_C2_Albedo.png",
    ):
        assert scanner.classify_path(path), path


# --------------------------------------------------------------------------------------
# history-only mode (the fast pre-publish gate)
# --------------------------------------------------------------------------------------


def _fake_registry(
    monkeypatch, *, history: list[str], entries: list[str]
) -> dict[str, int]:
    """Stub the two registry readers and count which ones actually get called."""
    calls = {"history": 0, "export": 0}

    def fake_history(image: str):
        calls["history"] += 1
        return history, "sha256:" + "0" * 64

    def fake_export(image: str):
        calls["export"] += 1
        yield from entries

    monkeypatch.setattr(scanner, "_image_history", fake_history)
    monkeypatch.setattr(scanner, "_iter_crane_export", fake_export)
    return calls


def test_history_only_does_not_stream_the_filesystem(monkeypatch) -> None:
    """The whole point: it must read the config blob and NOT the ~69 GB of layers."""
    calls = _fake_registry(monkeypatch, history=["ENV FOO=1"], entries=["opt/x"])

    report = scanner.scan("reg/npa-isaac-lab:t", None, history_only=True)

    assert calls["history"] == 1
    assert calls["export"] == 0, "history-only must not stream the image filesystem"
    assert report.entries_scanned == 0
    assert report.history_only is True
    assert report.clean


def test_history_only_still_catches_a_baked_install(monkeypatch) -> None:
    """A gate that cannot fail is not a gate. This is the case it exists for."""
    _fake_registry(
        monkeypatch,
        history=['RUN /bin/bash -lc pip install --no-deps "isaaclab==2.3.2.post1"'],
        entries=[],
    )

    report = scanner.scan("reg/npa-isaac-lab:t", None, history_only=True)

    assert not report.clean
    assert report.history_hits


def test_history_only_is_recorded_in_the_report(monkeypatch) -> None:
    """A history-only 'clean' is a weaker claim than a full-scan 'clean'.

    It says the build ran no Isaac install, not that the image ships no Isaac bytes -- a
    COPY from a vendor stage would pass it. The flag has to survive into the JSON so a
    consumer cannot quietly cite a gate result as the redistribution proof.
    """
    _fake_registry(monkeypatch, history=[], entries=[])

    gated = scanner.scan("reg/i:t", None, history_only=True).to_dict()
    full = scanner.scan("reg/i:t", None).to_dict()

    assert gated["history_only"] is True
    assert full["history_only"] is False
    assert gated["verdict"] == full["verdict"] == "clean"


def test_full_scan_remains_the_default(monkeypatch) -> None:
    """Nobody should get the weaker check by accident."""
    calls = _fake_registry(monkeypatch, history=[], entries=["opt/x", "opt/y"])

    report = scanner.scan("reg/i:t", None)

    assert calls["export"] == 1
    assert report.entries_scanned == 2
    assert report.history_only is False


def _empty_tar_stream() -> io.BytesIO:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w"):
        pass
    payload.seek(0)
    return payload


def test_registry_export_failure_is_fatal_after_a_valid_partial_tar(
    monkeypatch,
) -> None:
    """A truncated export must never become a clean redistribution report."""

    class FailedExport:
        stdout = _empty_tar_stream()

        @staticmethod
        def wait() -> int:
            return 17

    monkeypatch.setattr(scanner, "_require", lambda _tool: "/usr/bin/crane")
    monkeypatch.setattr(
        scanner.subprocess, "Popen", lambda *_args, **_kwargs: FailedExport()
    )

    with pytest.raises(subprocess.CalledProcessError, match="exit status 17"):
        list(scanner._iter_crane_export("registry.example/image:tag"))


def test_registry_config_failure_is_fatal(monkeypatch) -> None:
    monkeypatch.setattr(scanner, "_require", lambda _tool: "/usr/bin/crane")
    monkeypatch.setattr(
        scanner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=9, stdout=""),
    )

    with pytest.raises(subprocess.CalledProcessError, match="exit status 9"):
        scanner._image_history("registry.example/image:tag")


def test_registry_digest_failure_is_fatal(monkeypatch) -> None:
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout=json.dumps({"history": []})),
            SimpleNamespace(returncode=11, stdout=""),
        )
    )
    monkeypatch.setattr(scanner, "_require", lambda _tool: "/usr/bin/crane")
    monkeypatch.setattr(
        scanner.subprocess, "run", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(subprocess.CalledProcessError, match="exit status 11"):
        scanner._image_history("registry.example/image:tag")


def test_registry_digest_must_be_a_sha256(monkeypatch) -> None:
    responses = iter(
        (
            SimpleNamespace(returncode=0, stdout=json.dumps({"history": []})),
            SimpleNamespace(returncode=0, stdout="not-a-digest\n"),
        )
    )
    monkeypatch.setattr(scanner, "_require", lambda _tool: "/usr/bin/crane")
    monkeypatch.setattr(
        scanner.subprocess, "run", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(RuntimeError, match="invalid linux/amd64 image digest"):
        scanner._image_history("registry.example/image:tag")
