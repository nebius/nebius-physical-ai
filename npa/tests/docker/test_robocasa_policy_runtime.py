"""Static contract for the combined RoboCasa + LeRobot ACT evaluation runtime."""

import shlex
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
    assert '"opencv-python>=4.9,<4.14"' in text
    assert '"opencv-python-headless>=' not in text
    assert "opencv == ['opencv-python']" in text
    assert "${ROBOCASA_REPO_URL} /opt/robocasa/source" in text
    assert "-e /opt/robocasa/source" in text


def test_robocasa_act_runtime_stays_within_lerobot_dependency_bounds() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert '"torch==2.9.0"' in text
    assert '"torchvision==0.24.0"' in text
    assert '"torch==2.12.1"' not in text
    assert '"torchvision==0.27.1"' not in text
    assert "req.specifier.contains(version(req.name), prereleases=True)" in text


def test_robocasa_policy_runtime_has_one_cuda_wheel_family() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    dependency_install = text.split("# Install RoboCasa source", maxsplit=1)[0]
    assert dependency_install.count("python -m pip install --no-cache-dir \\\n") == 1
    assert (
        '${ROBOSUITE_COMMIT}" \\\n    && python -m pip install --no-cache-dir \\'
        not in dependency_install
    )
    assert dependency_install.count('"torch==2.9.0"') == 1
    assert dependency_install.count('"torchvision==0.24.0"') == 1
    assert '"torch==2.9.0" "torchvision==0.24.0" \\\n' in dependency_install
    assert 'pip install --no-cache-dir --upgrade \\\n        "torch' not in text
    assert "forbidden_names = {" in text
    assert "'cuda-toolkit'" in text
    assert "'nvidia-cudnn-cu13'" in text
    assert "torch.__version__ == '2.9.0+cu128'" in text
    assert "torch.version.cuda == '12.8'" in text


def test_robocasa_public_runtime_excludes_restricted_optional_payloads() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    dependency_install = text.split("# Install RoboCasa source", maxsplit=1)[0]

    assert "IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg" in text
    assert "        ffmpeg fuse netcat-openbsd" in text
    assert "imageio_ffmpeg-0.6.0.tar.gz#sha256=" in dependency_install
    assert '"imageio[ffmpeg]"' not in dependency_install
    assert "*/imageio_ffmpeg/binaries/ffmpeg*" in dependency_install
    assert "imageio_ffmpeg.get_ffmpeg_exe() == '/usr/bin/ffmpeg'" in dependency_install
    assert "imageio.mimsave(video," in dependency_install
    assert "itertools.islice(reader, 3)" in dependency_install
    assert "len(frames) == 2" in dependency_install
    assert dependency_install.count("render_dataset_with_omniverse.py") >= 2
    assert (
        "*/robosuite/scripts/__pycache__/render_dataset_with_omniverse*.pyc"
        in dependency_install
    )


def test_robocasa_system_install_layer_removes_builder_resolver_state() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = next(
        index for index, line in enumerate(lines) if line.startswith("RUN apt-get update")
    )
    end = start
    while lines[end].rstrip().endswith("\\"):
        end += 1
    install_run = "".join(lines[start : end + 1]).replace("\\\n", " ")
    commands = [command.strip() for command in install_run.split("&&")]

    install = next(
        index for index, command in enumerate(commands) if "apt-get install" in command
    )
    venv = commands.index("python3.12 -m venv /opt/robocasa/venv")
    cleanup = next(
        index
        for index, command in enumerate(commands)
        if (tokens := shlex.split(command))
        and tokens[0] == "rm"
        and "/run/systemd/resolve" in tokens
    )
    absent_path = commands.index("test ! -e /run/systemd/resolve")
    absent_symlink = commands.index("test ! -L /run/systemd/resolve")

    cleanup_tokens = shlex.split(commands[cleanup])
    assert "/var/lib/apt/lists/*" in cleanup_tokens
    assert install < venv < cleanup < absent_path < absent_symlink
    assert absent_symlink == len(commands) - 1


def test_robocasa_runtime_is_non_root_without_passwordless_sudo() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    user_lines = [
        line.strip() for line in text.splitlines() if line.startswith("USER ")
    ]
    assert user_lines[-1] == "USER ubuntu"
    assert "NOPASSWD" not in text
    assert "openssh-server" not in text
    assert "rsync sudo" not in text
    assert "chown -R ubuntu:ubuntu /opt/robocasa/source/robocasa/models/assets" in text
    assert "chown -R ubuntu:ubuntu /app " not in text
    assert "chown -R ubuntu:ubuntu /opt/robocasa/source\n" not in text
    assert "chown -R ubuntu:ubuntu /app /opt/robocasa\n" not in text
    copy_end = text.index(
        "COPY src/npa/smoke/test_robocasa_functional.py /app/smoke_functional.py"
    )
    normalize_permissions = text.index(
        "RUN chmod -R u=rwX,go=rX /app/npa /app/smoke_functional.py"
    )
    runtime_import = text.index(
        'RUN python -c "from npa.workbench.robocasa.service import app;'
    )
    assert copy_end < normalize_permissions < runtime_import


def test_robocasa_image_binds_committed_source_revision() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert f"ARG BASE_IMAGE={PINNED_CUDA_BASE}" in dockerfile
    assert PINNED_CUDA_BASE in build_script
    assert "ARG NPA_SOURCE_SHA" in dockerfile
    assert 'org.opencontainers.image.revision="${NPA_SOURCE_SHA}"' in dockerfile
    assert 'org.opencontainers.image.base.name="${BASE_IMAGE}"' in dockerfile
    assert (
        'org.opencontainers.image.base.digest="'
        "sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"
        '"' in dockerfile
    )
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA=1" in dockerfile
    assert "FROM --platform=" not in dockerfile
    assert "docker build \\\n  --platform linux/amd64 \\" in build_script
    assert "\n  --provenance=false \\" in build_script
    assert "grep -Eq '^[0-9a-f]{40}$'" in dockerfile
    assert "COPY src/npa/clients/storage.py /app/npa/clients/storage.py" in dockerfile
    assert (
        "COPY src/npa/cli/path_contract.py /app/npa/cli/path_contract.py" in dockerfile
    )
    assert "from npa.workbench.robocasa.service import app" in dockerfile
    assert 'rev-parse HEAD)" != "${NPA_SOURCE_SHA}"' in build_script
    assert "status --porcelain=v1 --untracked-files=all -- ." in build_script


def test_robocasa_candidate_version_is_consistent() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "ARG ROBOCASA_VERSION=0.1.1" in dockerfile
    assert 'ROBOCASA_VERSION="${ROBOCASA_VERSION:-0.1.1}"' in build_script
    assert "npa.cuda_architectures" not in dockerfile
