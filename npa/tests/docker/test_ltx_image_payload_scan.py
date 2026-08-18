"""The npa-ltx2 payload scan is mutation-tested in both directions.

The image's redistribution claim is "contains no LTX-2.5 and no CUDA". A scanner
that only ever passes proves nothing, so every historical way of baking the
payload must be detected, and the legitimate runtime-fetch plumbing — which
necessarily *mentions* ltx_pipelines, Lightricks, and the acceptance variables —
must not trip it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


walker = _load("scan_image_wan_payload")
scanner = _load("scan_image_ltx_payload")


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _docker_save(
    path: Path, *, layers: list[dict[str, bytes]], config: dict
) -> Path:
    layer_archives: list[tuple[str, bytes]] = []
    for index, members in enumerate(layers):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as layer:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                layer.addfile(info, io.BytesIO(payload))
        layer_archives.append((f"layer-{index}/layer.tar", stream.getvalue()))
    manifest = [{"Config": "config.json", "Layers": [name for name, _ in layer_archives]}]
    members = {
        "manifest.json": json.dumps(manifest).encode(),
        "config.json": json.dumps(config).encode(),
        **dict(layer_archives),
    }
    return _tar(path, members)


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


# A minimally realistic clean image: our scripts, the copied licensing module,
# uv, huggingface_hub, and the notices. Everything here legitimately names LTX.
CLEAN_ROOTFS = {
    "usr/local/bin/ltx-runtime": (
        b"#!/usr/bin/env bash\n"
        b"SOURCE_REPO=https://github.com/Lightricks/LTX-2.git\n"
        b"uv sync --extra natten\n"
        b"hf download Lightricks/LTX-2.5\n"
    ),
    # What the image actually ships. It carried a copied licensing module and an
    # in-image gate script until the declaration was removed; modelling files the
    # image no longer contains would mean the "a clean image passes" case below
    # was asserting against a fiction.
    "opt/npa/ltx2/video_check.py": b"FLAT_FRAME_TOLERANCE = 1.0\n",
    "opt/npa/ltx2/validate_video.py": b"import video_check\n",
    "usr/share/doc/npa-ltx2/REDISTRIBUTION.md": b"# npa-ltx2\nltx-core, ltx_pipelines\n",
    "usr/local/lib/python3.12/site-packages/uv/__init__.py": b"",
    "usr/local/lib/python3.12/site-packages/huggingface_hub/__init__.py": b"",
}

CLEAN_CONFIG = {
    "history": [
        {"created_by": "RUN pip install uv==0.9.8 huggingface-hub[cli]==0.36.0"},
        {"created_by": "COPY docker/workbench/ltx2/ltx_runtime.sh /usr/local/bin/"},
        {"created_by": "RUN ltx-runtime health"},
        {"created_by": "RUN su ubuntu -s /bin/bash -c 'ltx-runtime assert-refusal'"},
        {"created_by": "COPY src/npa/workbench/ltx2/video_check.py /opt/npa/ltx2/"},
    ],
    "config": {
        "Env": [
            "NPA_LTX_SOURCE_REF=fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca",
            "NPA_LTX_MODEL_CACHE=/workspace/model-cache/ltx-2.5",
        ]
    },
}


class TestCleanImagePasses:
    def test_a_correctly_built_image_has_no_findings(self, tmp_path: Path) -> None:
        findings = scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), CLEAN_CONFIG)
        assert findings == [], _kinds(findings)

    def test_mentioning_the_vendor_in_bootstrap_plumbing_is_allowed(
        self, tmp_path: Path
    ) -> None:
        """Runtime-fetch scripts must name what they fetch; that is not baking."""

        rootfs = dict(CLEAN_ROOTFS)
        rootfs["opt/npa/ltx2/smoke.sh"] = (
            b"find / -name 'ltx_core' -o -name 'ltx_pipelines'\n"
            b"NPA_LTX_ACCEPT_COMMUNITY_LICENSE= ltx-runtime ensure\n"
        )
        assert scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG) == []

    def test_docker_save_scans_every_layer_and_history(self, tmp_path: Path) -> None:
        assert scanner.docker_save_material is scanner.walker.docker_save_material
        archive = _docker_save(
            tmp_path / "image.tar",
            layers=[CLEAN_ROOTFS, {"workspace/.wh.deleted": b""}],
            config=CLEAN_CONFIG,
        )
        (tmp_path / "layers").mkdir(exist_ok=True)
        layers, config = scanner.docker_save_material(archive, tmp_path / "layers")
        assert scanner.scan_tars(layers, config) == []

    def test_docker_save_detects_payload_deleted_by_a_later_layer(
        self, tmp_path: Path
    ) -> None:
        archive = _docker_save(
            tmp_path / "image.tar",
            layers=[
                {"opt/models/ltx-2.5-secret.safetensors": b"weights"},
                {"opt/models/.wh.ltx-2.5-secret.safetensors": b""},
            ],
            config=CLEAN_CONFIG,
        )
        layer_dir = tmp_path / "layers"
        layer_dir.mkdir()
        layers, config = scanner.docker_save_material(archive, layer_dir)
        assert "ltx_weight_file" in _kinds(scanner.scan_tars(layers, config))

    @pytest.mark.parametrize(
        "created_by",
        [
            "RUN ltx-runtime health",
            "RUN ltx-runtime status",
            "RUN ltx-runtime version",
            "RUN ltx-runtime terms",
            # The refusal proof must be runnable at build; it is the evidence the
            # mechanism works, and it cannot download.
            "RUN ltx-runtime assert-refusal",
        ],
    )
    def test_non_downloading_modes_at_build_are_allowed(
        self, tmp_path: Path, created_by: str
    ) -> None:
        config = {"history": [{"created_by": created_by}]}
        findings = scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        assert findings == [], _kinds(findings)


class TestBakedLtxPayloadIsDetected:
    @pytest.mark.parametrize(
        "member",
        [
            "usr/local/lib/python3.12/site-packages/ltx_core/__init__.py",
            "usr/local/lib/python3.12/site-packages/ltx_pipelines/distilled.py",
            "usr/local/lib/python3.12/site-packages/ltx_trainer/train.py",
            "opt/venv/lib/python3.12/site-packages/ltx_core-1.2.0.dist-info/METADATA",
            "usr/lib/python3/dist-packages/ltx_pipelines/__init__.py",
        ],
    )
    def test_an_installed_ltx_distribution_fails(
        self, tmp_path: Path, member: str
    ) -> None:
        """The LTX-specific trap: the code is licensed material, not a wrapper."""

        rootfs = dict(CLEAN_ROOTFS) | {member: b"x"}
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)
        assert "ltx_python_distribution" in _kinds(findings)

    def test_a_baked_source_tree_fails(self, tmp_path: Path) -> None:
        rootfs = dict(CLEAN_ROOTFS) | {
            "opt/byof/packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py": b"x"
        }
        assert _kinds(scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)) & {
            "ltx_source_tree",
            "ltx_python_distribution",
        }

    @pytest.mark.parametrize(
        "member",
        [
            "workspace/model-cache/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
            "opt/models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
            "opt/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
            "opt/models/diffusion_pytorch_model.bin",
        ],
    )
    def test_baked_weights_fail(self, tmp_path: Path, member: str) -> None:
        rootfs = dict(CLEAN_ROOTFS) | {member: b"weights"}
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)
        assert _kinds(findings) & {"ltx_weight_file", "checkpoint_or_weight"}

    @pytest.mark.parametrize(
        "member",
        [
            "usr/local/lib/python3.12/site-packages/torch/__init__.py",
            "usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.13",
            "usr/local/lib/python3.12/site-packages/nvidia_cudnn_cu13.dist-info/METADATA",
            "usr/lib/x86_64-linux-gnu/libcudart.so.13",
        ],
    )
    def test_baked_cuda_or_torch_fails(self, tmp_path: Path, member: str) -> None:
        rootfs = dict(CLEAN_ROOTFS) | {member: b"x"}
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)
        assert _kinds(findings) & {
            "torch_distribution",
            "nvidia_python_distribution",
            "cuda_library",
        }

    def test_a_populated_runtime_cache_fails(self, tmp_path: Path) -> None:
        """A warmed cache in a layer is the runtime fetch performed at build."""

        rootfs = dict(CLEAN_ROOTFS) | {
            "root/.cache/uv/archive-v0/ltx.whl": b"x",
        }
        assert "package_cache" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)
        )


class TestBakedAcceptanceIsDetected:
    @pytest.mark.parametrize(
        "created_by",
        [
            "ENV NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES",
            "ARG NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS=YES",
            "RUN NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES ltx-runtime ensure",
        ],
    )
    def test_pre_granted_acceptance_fails(
        self, tmp_path: Path, created_by: str
    ) -> None:
        """Baking the answer removes the operator from the licensing decision."""

        config = {"history": [{"created_by": created_by}]}
        assert "baked_ltx_acceptance" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )

    def test_a_baked_env_acceptance_fails(self, tmp_path: Path) -> None:
        config = {"config": {"Env": ["NPA_LTX_ACCEPT_COMMUNITY_LICENSE=YES"]}}
        assert "baked_ltx_acceptance" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )

    @pytest.mark.parametrize(
        "created_by",
        [
            "ENV NPA_LTX_ENTITY_CLASS=community",
            "ENV NPA_LTX_USE_CLASS=non-commercial",
            "ARG NPA_LTX_COMMERCIAL_AGREEMENT_REF=CUA-1",
        ],
    )
    def test_a_baked_declaration_fails(self, tmp_path: Path, created_by: str) -> None:
        """Answering the revenue-threshold question for the operator is the breach."""

        config = {"history": [{"created_by": created_by}]}
        assert "baked_ltx_declaration" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )


class TestBuildTimeFetchIsDetected:
    @pytest.mark.parametrize(
        "created_by",
        [
            "RUN ltx-runtime ensure",
            "RUN ltx-runtime warm",
            "RUN ltx-runtime fetch-weights",
            "RUN ltx-runtime exec python -c 'import ltx_core'",
        ],
    )
    def test_running_the_bootstrap_at_build_fails(
        self, tmp_path: Path, created_by: str
    ) -> None:
        config = {"history": [{"created_by": created_by}]}
        assert "runtime_bootstrap_at_build" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )

    @pytest.mark.parametrize(
        "created_by",
        [
            "RUN uv pip install ltx-core ltx-pipelines",
            "RUN pip install git+https://github.com/Lightricks/LTX-2.git",
            "RUN uv sync --extra natten && echo done",
        ],
    )
    def test_installing_ltx_at_build_fails(
        self, tmp_path: Path, created_by: str
    ) -> None:
        config = {"history": [{"created_by": created_by}]}
        findings = scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        assert _kinds(findings) & {"ltx_install_at_build", "runtime_bootstrap_at_build"}

    def test_downloading_weights_at_build_fails(self, tmp_path: Path) -> None:
        config = {"history": [{"created_by": "RUN hf download Lightricks/LTX-2.5"}]}
        assert "hf_download_at_build" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )

    def test_a_cuda_base_image_fails(self, tmp_path: Path) -> None:
        config = {"history": [{"created_by": "FROM nvidia/cuda:13.2.0-runtime"}]}
        assert "cuda_base" in _kinds(
            scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), config)
        )


class TestTheAuditedBaseImageBytes:
    """The base image's own OpenSSH/ffmpeg binaries are audited by identity.

    `openssh-server` and `ffmpeg` are installed on purpose — the SkyPilot
    bootstrap contract needs the first and the output validator decodes with the
    second — and their binaries contain key-format literals and a CUDA/NVENC ELF
    reference. The first real scan of a real build failed on exactly those,
    because this scanner passed empty allowlists on a wrong assumption about its
    own image.

    What makes the fix an audit rather than a hole is that the entries are
    path *and* exact SHA-256, so substituted bytes at an audited path still fail.
    """

    def test_the_allowlists_are_wired_up_and_not_empty(self) -> None:
        assert scanner.AUDITED_SECRET_LITERAL_FILE_SHA256
        assert scanner.AUDITED_LITERAL_LIBRARY_SHA256
        assert "usr/sbin/sshd" in scanner.AUDITED_SECRET_LITERAL_FILE_SHA256

    def test_substituted_bytes_at_an_audited_path_are_caught(
        self, tmp_path: Path
    ) -> None:
        """A path alone must not be enough, or this would be a blanket exemption."""

        rootfs = dict(CLEAN_ROOTFS) | {
            "usr/sbin/sshd": b"-----BEGIN OPENSSH PRIVATE KEY-----\nnot the real sshd\n"
        }
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)

        assert "audited_literal_byte_drift" in _kinds(findings)

    def test_the_exact_audited_bytes_are_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: matching bytes must actually pass, or the scan is unusable.

        The real sshd is not available here, so this pins a stand-in through the
        same code path the real hashes take.
        """

        payload = b"-----BEGIN OPENSSH PRIVATE KEY-----\nstand-in for the real sshd\n"
        monkeypatch.setitem(
            scanner.AUDITED_SECRET_LITERAL_FILE_SHA256,
            "usr/sbin/sshd",
            hashlib.sha256(payload).hexdigest(),
        )
        rootfs = dict(CLEAN_ROOTFS) | {"usr/sbin/sshd": payload}

        assert scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG) == []

    def test_an_unaudited_secret_still_fails(self, tmp_path: Path) -> None:
        """Auditing the base must not have opened the door to a real leaked key."""

        rootfs = dict(CLEAN_ROOTFS) | {
            "opt/npa/ltx2/id_ed25519": b"-----BEGIN OPENSSH PRIVATE KEY-----\nleaked\n"
        }
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)

        assert "credential_content" in _kinds(findings)

    def test_a_real_cuda_library_still_fails_despite_the_audited_libav(
        self, tmp_path: Path
    ) -> None:
        rootfs = dict(CLEAN_ROOTFS) | {"usr/lib/x86_64-linux-gnu/libcudart.so.13": b"x"}
        findings = scanner.scan(_tar(tmp_path / "r.tar", rootfs), CLEAN_CONFIG)

        assert "cuda_library" in _kinds(findings)


class TestSharedWalkerPolicyIsRestored:
    def test_the_wan_policy_survives_an_ltx_scan(self, tmp_path: Path) -> None:
        """Borrowing the walker must not leave the Wan scanner reconfigured."""

        before = (walker.FORBIDDEN_PATHS, walker.FORBIDDEN_HISTORY)
        scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), CLEAN_CONFIG)
        assert (walker.FORBIDDEN_PATHS, walker.FORBIDDEN_HISTORY) == before

    def test_the_policy_is_restored_even_when_a_scan_raises(self) -> None:
        before = (walker.FORBIDDEN_PATHS, walker.FORBIDDEN_HISTORY)
        with pytest.raises(RuntimeError):
            with walker.payload_policy(forbidden_paths=(), forbidden_history=()):
                raise RuntimeError("boom")
        assert (walker.FORBIDDEN_PATHS, walker.FORBIDDEN_HISTORY) == before

    def test_the_wan_scanner_still_detects_its_own_payload_afterwards(
        self, tmp_path: Path
    ) -> None:
        scanner.scan(_tar(tmp_path / "r.tar", CLEAN_ROOTFS), CLEAN_CONFIG)
        rootfs = {
            "opt/wan-base/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12": b"x"
        }
        findings = walker.scan(_tar(tmp_path / "w.tar", rootfs), {})
        assert "nvidia_python_distribution" in _kinds(findings)
