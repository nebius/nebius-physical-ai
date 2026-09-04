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

    user_lines = [line.strip() for line in text.splitlines() if line.startswith("USER ")]
    assert user_lines[-1] == "USER ubuntu"
    assert "NOPASSWD" not in text
    assert "openssh-server" not in text
    assert "rsync sudo" not in text
