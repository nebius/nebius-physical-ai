"""Static contract for the combined RoboCasa + LeRobot ACT evaluation runtime."""

from pathlib import Path


DOCKERFILE = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "workbench"
    / "robocasa"
    / "Dockerfile"
)
BUILD_SCRIPT = DOCKERFILE.with_name("build.sh")
PINNED_CUDA_BASE = (
    "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
    "@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"
)


def test_robocasa_keeps_known_good_gymnasium_and_policy_only_lerobot() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert '"gymnasium==0.29.1"' in text
    assert 'pip install --no-cache-dir --no-deps "lerobot==0.5.1"' in text
    assert '"av>=15.0.0,<16.0.0"' in text
    assert '"diffusers>=0.27.2,<0.36.0"' in text
    assert '"pyserial>=3.5,<4.0"' in text
    assert "from lerobot.policies.act.modeling_act import ACTPolicy" in text
    assert "from lerobot.policies.factory import make_pre_post_processors" in text
    assert '"draccus==0.10.0"' in text
    assert '"einops>=0.8.0,<0.9.0"' in text
    assert "${ROBOCASA_REPO_URL} /opt/robocasa/source" in text
    assert "-e /opt/robocasa/source" in text


def test_robocasa_runtime_is_non_root_without_passwordless_sudo() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    user_lines = [
        line.strip() for line in text.splitlines() if line.startswith("USER ")
    ]
    assert user_lines[-1] == "USER ubuntu"
    assert "NOPASSWD" not in text
    assert "openssh-server" not in text
    assert "rsync sudo" not in text


def test_robocasa_image_binds_committed_source_revision() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert f"ARG BASE_IMAGE={PINNED_CUDA_BASE}" in dockerfile
    assert PINNED_CUDA_BASE in build_script
    assert "ARG NPA_SOURCE_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert 'test "$(printf %s "${NPA_SOURCE_SHA}" | wc -c)" -eq 40' in dockerfile
    assert "COPY src/npa/clients/storage.py /app/npa/clients/storage.py" in dockerfile
    assert (
        "COPY src/npa/cli/path_contract.py /app/npa/cli/path_contract.py" in dockerfile
    )
    assert "from npa.workbench.robocasa.service import app" in dockerfile
    assert 'rev-parse HEAD)" != "${NPA_SOURCE_SHA}"' in build_script
    assert "status --porcelain --untracked-files=no -- ." in build_script
