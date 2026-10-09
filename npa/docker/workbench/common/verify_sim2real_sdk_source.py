"""Verify the source and image-independent SDK behavior used by thin children."""

from pathlib import Path
import argparse

import npa
from npa.workflows.sim2real.artifact_config import build_artifact_config_from_env
from npa.workflows.sim2real.diagnostic_config import build_diagnostic_config_from_env


def verify(source_root: Path) -> None:
    """Require the copied SDK and usable artifact/diagnostic configuration.

    Args:
        source_root: Directory containing the child image's complete NPA source.
    Returns:
        None.
    Raises:
        RuntimeError: Import resolution or image-independent settings drifted.
    """

    expected = (source_root / "npa/__init__.py").resolve()
    if Path(npa.__file__).resolve() != expected:
        raise RuntimeError("Thin policy image imported a different SDK source")
    artifact = build_artifact_config_from_env(
        run_id="sdk-build-contract", s3_bucket="sdk-build-contract"
    )
    diagnostic = build_diagnostic_config_from_env(
        run_id="sdk-build-contract",
        s3_bucket="sdk-build-contract",
        k8s_namespace="default",
    )
    if artifact.s3_bucket != diagnostic.s3_bucket:
        raise RuntimeError("Artifact and diagnostic storage settings diverged")
    print("SIM2REAL_CHILD_SDK_SOURCE_AND_IMAGE_FREE_CONFIG_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    verify(parser.parse_args().source_root)
