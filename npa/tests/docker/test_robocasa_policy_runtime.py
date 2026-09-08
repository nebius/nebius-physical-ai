"""Static contract for the combined RoboCasa + LeRobot ACT evaluation runtime."""

from pathlib import Path


DOCKERFILE = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "workbench"
    / "robocasa"
    / "Dockerfile"
)


def test_robocasa_keeps_known_good_gymnasium_and_policy_only_lerobot() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    lock = DOCKERFILE.with_name("requirements.lock").read_text()
    policy = DOCKERFILE.with_name("policy-requirements.lock").read_text()

    assert "gymnasium==0.29.1" in lock
    assert "lerobot==0.5.1" in policy
    assert "bbd11021023fde0947b6d1ff1c52fe91c86a28ab09a96359892f3ef7e8866862" in policy
    assert "--no-deps --require-hashes" in text
    assert "av==17.1.0" in lock
    assert "diffusers==" in lock
    assert "pyserial==" in lock
    assert "from lerobot.policies.act.modeling_act import ACTPolicy" in text
    assert "from lerobot.policies.factory import make_pre_post_processors" in text
    assert "draccus==0.10.0" in lock
    assert "einops==" in lock
    assert "8f3c96ec8d1bfcd8126cad2bca887da98d30e997" in text
    assert "1893328b5222ac0443287e593c161c696c77e3f8018f6d9f6bd900871d2caad3" in text
    assert "-e /opt/robocasa/source" in text
    assert "COPY src/npa /opt/npa-src/src/npa" in text
    assert "from npa.workbench.robocasa.service import create_app" in text
    assert "touch /app/npa/__init__.py" not in text


def test_robocasa_runtime_is_non_root_without_passwordless_sudo() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    user_lines = [line.strip() for line in text.splitlines() if line.startswith("USER ")]
    assert user_lines[-1] == "USER ubuntu"
    assert "NOPASSWD" not in text
    assert "openssh-server" not in text
    assert "rsync sudo" not in text
    assert "ROBOCASA_AUTH_MODE=token" in text


def test_robocasa_keeps_runtime_only_cudnn_and_source_asset_notices() -> None:
    import hashlib

    text = DOCKERFILE.read_text()
    assert "13.0.2-base-ubuntu24.04@sha256:" in text
    assert "cudnn-devel" not in text
    assert "--require-hashes" in text
    installation = next(part for part in text.split("\nRUN ") if part.startswith("pip install"))
    assert "python /opt/filter_cudnn_runtime.py" in installation
    expected = {
        "LICENSE-APACHE-2.0": "a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9",
        "LICENSE-SAWYER": "dd4ff820f89332d392f36508110c85c7b46de7787f121c2b0dc3e87f9d7f60fb",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(DOCKERFILE.with_name(name).read_bytes()).hexdigest() == digest
        assert f"docker/workbench/robocasa/{name}" in text
    notices = DOCKERFILE.with_name("ASSET-NOTICES.md").read_text()
    assert "Creative Commons Attribution 4.0" in notices
    assert "Rethink Robotics Inc." in " ".join(notices.split())
