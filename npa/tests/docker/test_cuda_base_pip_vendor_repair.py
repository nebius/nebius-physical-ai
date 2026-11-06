"""Require the CUDA-image base pip repair that removes vendored vulnerable certifi."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECIPES = {
    "genesis": ROOT / "npa/docker/workbench/genesis/Dockerfile",
    "lerobot": ROOT / "npa/docker/workbench/lerobot/Dockerfile",
}


def test_cuda_recipes_replace_the_runtime_venv_pip_vendor_closure() -> None:
    """Each affected CUDA image must retain the verified pip derivative in its venv."""
    for name, recipe in RECIPES.items():
        text = recipe.read_text(encoding="utf-8")

        assert "source=docker/workbench/common/secure_pip" in text, name
        assert "target=/opt/secure-pip-builder,ro" in text, name
        assert "python3.12 /opt/secure-pip-builder/build.py" in text, name
        assert f"/opt/{name}/venv/bin/python -m pip install" in text, name
        assert "--no-index --no-deps --force-reinstall" in text, name
        assert 'pip.__version__ == "26.2.1+npa.1"' in text, name
        assert 'certifi.__version__ == "2026.10.19"' in text, name
        assert f"/usr/share/doc/npa-{name}/secure-pip-build.json" in text, name
        assert "/usr/local/bin/python3.11" not in text, name
