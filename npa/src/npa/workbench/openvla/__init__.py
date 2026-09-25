"""npa.workbench.openvla - OpenVLA / OpenVLA-OFT workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper

train = make_cli_wrapper(
    "npa.cli.workbench.openvla",
    "train_cmd",
    "Fine-tune OpenVLA with the OpenVLA-OFT LoRA recipe.",
)
serve = make_cli_wrapper(
    "npa.cli.workbench.openvla",
    "serve_cmd",
    "Serve an OpenVLA checkpoint over HTTP.",
)
eval = make_cli_wrapper(
    "npa.cli.workbench.openvla",
    "eval_cmd",
    "Evaluate an OpenVLA checkpoint.",
)

__all__ = [
    "train",
    "serve",
    "eval",
]
