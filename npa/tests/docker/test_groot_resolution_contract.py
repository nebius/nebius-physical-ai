"""Keep the GR00T image resolver aligned with its x86_64 build target."""

from pathlib import Path


DOCKERFILE = Path(__file__).resolve().parents[2] / "docker/workbench/groot/Dockerfile"


def test_groot_uv_resolution_is_limited_to_the_published_linux_x86_target():
    text = DOCKERFILE.read_text(encoding="utf-8")
    patch = 'required-environments = [\\n"'
    assert patch in text
    assert "platform_machine == 'x86_64'" in text
    assert "platform_machine == 'aarch64'" in text
    assert text.index(patch) < text.index("uv sync --python 3.12")
