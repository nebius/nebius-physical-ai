"""Keep Genesis dependency correction inside the original installation layer."""

from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
GENESIS = ROOT / "npa/docker/workbench/genesis"
VARIANTS = (
    ("Dockerfile", "0.25.2", "/opt/genesis/venv/lib/python3.12/site-packages"),
    ("Dockerfile.sm120", "0.26.0", "/opt/genesis/venv/lib/python3.11/site-packages"),
)
CORRECTION = "/opt/genesis/remove-scikit-image-recipe.py"


def _instructions(text):
    return [
        (match.group(1), match.group(2))
        for line in text.replace("\\\n", " ").splitlines()
        if (match := re.match(r"^([A-Z]+)\s+(.*)$", line))
    ]


def _assert_installation_contract(text, version, site_packages):
    instructions = _instructions(text)
    install_index, installation = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert any(
        kind == "ENV" and re.search(r"\bPIP_NO_CACHE_DIR=1\b", body)
        for kind, body in instructions[:install_index]
    )
    assert any(
        kind == "COPY"
        and "docker/workbench/curobo/remove_scikit_image_recipe.py" in body
        and CORRECTION in body
        for kind, body in instructions[:install_index]
    )
    assert f'"scikit-image=={version}"' in installation
    assert f"python {CORRECTION}" in installation
    assert f"--site-packages {site_packages}" in installation
    assert "--capability genesis" in installation
    assert installation.rfind("pip install") < installation.index(
        f"python {CORRECTION}"
    )
    assert "dependency-source-correction.json" in installation


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_genesis_ffmpeg_contract_avoids_bundled_wheel_executables(
    dockerfile, version, site_packages
):
    text = (GENESIS / dockerfile).read_text()
    instructions = _instructions(text)
    index, installation = next(
        (index, body)
        for index, (kind, body) in enumerate(instructions)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert any(
        kind == "ENV" and "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in body
        for kind, body in instructions[:index]
    )
    assert "pip install --no-binary imageio-ffmpeg" in installation
    assert '"imageio-ffmpeg==0.6.0"' in installation
    assert f"find {site_packages.rsplit('/lib/', 1)[0]}" in installation
    assert "imageio_ffmpeg/binaries/ffmpeg*' -print -quit" in installation
    assert 'imageio_ffmpeg.get_ffmpeg_exe() == "/usr/bin/ffmpeg"' in installation


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_genesis_recipes_select_their_supported_python_and_bind_source(
    dockerfile, version, site_packages
):
    text = (GENESIS / dockerfile).read_text()
    assert site_packages in text
    assert "GENESIS_VENV=/opt/genesis/venv" in text
    assert "NPA_BAKED_PYTHON=/opt/genesis/venv/bin/python" in text
    assert "PATH=/opt/genesis/venv/bin:/opt/npa/venv/bin:$PATH" in text
    expected_venv = (
        "RUN python3.12 -m venv /opt/genesis/venv"
        if dockerfile == "Dockerfile"
        else 'RUN python3.11 -m venv "${GENESIS_VENV}"'
    )
    assert expected_venv in text
    assert '"pillow>=13.0.0,<14.0"' in text
    assert "ARG NPA_SOURCE_SHA" in text
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in text
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in text
    assert 'test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40' in text


@pytest.mark.parametrize("dockerfile,_,__", VARIANTS)
def test_genesis_recipes_enforce_the_fsspec_security_floor(dockerfile, _, __):
    """Every Genesis variant must reject a closure below the strict floor."""

    text = (GENESIS / dockerfile).read_text()
    installation = next(
        body
        for kind, body in _instructions(text)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert '"fsspec==2026.6.0"' in installation
    assert "pip check" in installation


def test_canonical_genesis_recipe_uses_a_pinned_cuda_base_and_output_attestation():
    text = (GENESIS / "Dockerfile").read_text()
    assert "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:" in text
    assert "FROM ${BASE_IMAGE}" not in text
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in text


def test_sm120_genesis_recipe_keeps_its_source_attested_parent_contract():
    text = (GENESIS / "Dockerfile.sm120").read_text()
    assert re.search(r"^ARG BASE_IMAGE$", text, re.MULTILINE)
    assert "FROM ${BASE_IMAGE}" in text


def test_canonical_genesis_builder_binds_the_checkout_source_to_the_image(tmp_path):
    """The canonical build passes and verifies only the checked-out source SHA."""

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = image ]; then printf '%s\\n' '" + source_sha + "'; exit 0; fi\n"
        'if [ "$1" = build ]; then exit 0; fi\n'
        "exit 99\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(GENESIS / "build.sh"), "--tag", "dev-" + source_sha],
        cwd=ROOT,
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "npa-genesis:dev-" + source_sha


def test_sm120_builder_refuses_the_unqualified_secure_closure_before_docker(
    tmp_path: Path,
) -> None:
    """The unrepaired additive SM120 path remains closed before Docker starts."""

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    invoked = tmp_path / "docker-invoked"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = image ]; then printf '%s\\n' '{source_sha}'; exit 0; fi\n"
        f"touch {invoked}\nexit 99\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(GENESIS / "build_sm120.sh"), "--base-image", "local-base"],
        cwd=ROOT,
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert (
        "additive SM120 Genesis image build is unavailable before any Docker build starts"
        in result.stderr
    )
    assert "Pillow>=13.0.0,<14.0" in result.stderr
    assert "fsspec==2026.6.0" in result.stderr
    assert not invoked.exists()


@pytest.mark.parametrize(
    ("builder", "extra_args"),
    (("build.sh", ()), ("build_sm120.sh", ("--base-image", "local-base"))),
)
def test_genesis_builders_reject_push_without_registry_before_docker(
    tmp_path: Path, builder: str, extra_args: tuple[str, ...]
) -> None:
    """CLI misuse must not be obscured by parent inspection or closure refusal."""

    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    invoked = tmp_path / "docker-invoked"
    docker = tmp_path / "docker"
    docker.write_text(f"#!/bin/sh\ntouch {invoked}\nexit 99\n")
    docker.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(GENESIS / builder),
            *extra_args,
            "--push",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "NPA_SOURCE_SHA": source_sha,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--push requires --registry" in result.stderr
    assert not invoked.exists()


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_genesis_corrects_reviewed_source_before_the_installation_layer_commits(
    dockerfile, version, site_packages
):
    _assert_installation_contract(
        (GENESIS / dockerfile).read_text(), version, site_packages
    )


@pytest.mark.parametrize("dockerfile,_,__", VARIANTS)
def test_genesis_build_time_sanitizers_are_readable_by_the_runtime_user(
    dockerfile, _, __
):
    text = (GENESIS / dockerfile).read_text()
    for source, destination in (
        (
            "docker/workbench/curobo/remove_scikit_image_recipe.py",
            "/opt/genesis/remove-scikit-image-recipe.py",
        ),
        (
            "docker/workbench/genesis/prepare_public_runtime.py",
            "/opt/genesis/prepare-public-runtime.py",
        ),
        (
            "docker/workbench/genesis/prepare_public_policies.py",
            "/opt/genesis/prepare-public-policies.py",
        ),
        (
            "docker/workbench/common/envgen_compat/tetgen.py",
            "/opt/genesis/compat/tetgen.py",
        ),
    ):
        assert f"COPY --chmod=0644 {source}" in text
        assert destination in text


@pytest.mark.parametrize("dockerfile,_,site_packages", VARIANTS)
def test_genesis_recipes_apply_the_same_public_act_runtime_boundary(
    dockerfile, _, site_packages
):
    """Architecture variants must not bypass reviewed public-runtime restrictions."""

    text = (GENESIS / dockerfile).read_text()
    installation = next(
        body
        for kind, body in _instructions(text)
        if kind == "RUN" and '"genesis-world==' in body
    )
    assert "NPA_GENESIS_SUPPORTED_STUDENT_POLICIES=act" in text
    assert f"--site-packages {site_packages} --check-only" in installation
    assert "prepare-public-runtime.py" in installation
    assert "prepare-public-policies.py" in installation
    assert "pip uninstall -y tetgen diffusers wandb torchcodec" in installation
    assert "public-runtime-dependencies.json" in installation
    assert "public-policy-source.json" in installation
    assert "npa-genesis-compat.pth" in installation


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_correction_in_a_child_layer_is_rejected(dockerfile, version, site_packages):
    text = (GENESIS / dockerfile).read_text()
    delayed = text.replace(
        f"    && python {CORRECTION}", f"\nRUN python {CORRECTION}"
    ).replace(
        f"    && /opt/genesis/venv/bin/python {CORRECTION}",
        f"\nRUN /opt/genesis/venv/bin/python {CORRECTION}",
    )
    assert delayed != text
    with pytest.raises(AssertionError):
        _assert_installation_contract(delayed, version, site_packages)


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_dependency_wheel_caches_cannot_survive_the_correction(
    dockerfile, version, site_packages
):
    text = (
        (GENESIS / dockerfile)
        .read_text()
        .replace("PIP_NO_CACHE_DIR=1", "PIP_NO_CACHE_DIR=0")
    )
    with pytest.raises(AssertionError):
        _assert_installation_contract(text, version, site_packages)


@pytest.mark.parametrize("dockerfile,version,site_packages", VARIANTS)
def test_a_different_dependency_version_is_rejected(dockerfile, version, site_packages):
    other = "0.26.0" if version == "0.25.2" else "0.25.2"
    text = (
        (GENESIS / dockerfile)
        .read_text()
        .replace(f'"scikit-image=={version}"', f'"scikit-image=={other}"')
    )
    with pytest.raises(AssertionError):
        _assert_installation_contract(text, version, site_packages)
