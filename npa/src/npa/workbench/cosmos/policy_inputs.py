"""Fetch pinned policy inputs inside the native environment, without importing NPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Download immutable inputs using the native runtime's Hugging Face client.

    Args:
        None; revisions and the private result path come from CLI arguments.
    Returns:
        None; writes paths only after all downloads succeed.
    Raises:
        Exception: Native Hub access or local publication failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vae-revision", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    dataset = Path(snapshot_download("nvidia/LIBERO_LeRobot_v3", repo_type="dataset",
                   revision=args.dataset_revision, allow_patterns=["libero_10/**"])) / "libero_10"
    model = snapshot_download("nvidia/Cosmos3-Nano", revision=args.model_revision)
    vae = hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth", revision=args.vae_revision)
    args.output_path.write_text(json.dumps({"dataset": str(dataset), "model": model, "vae": vae}))


if __name__ == "__main__":
    main()
