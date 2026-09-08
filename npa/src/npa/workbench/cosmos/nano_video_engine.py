"""Launch the pinned Omni CLI with explicit video-only Cosmos model settings."""

from __future__ import annotations

import sys
from typing import Any


def model_config() -> dict[str, bool]:
    """Return the model settings required by the staged video checkpoint.

    Args:
        None.
    Returns:
        Fresh settings disabling unstaged audio and guardrail components.
    Raises:
        None.
    """
    return {"sound_gen": False, "guardrails": False}


def create_command(argv: list[str]) -> tuple[Any, Any]:
    """Parse and validate the upstream serving command without starting it.

    Args:
        argv: Upstream arguments beginning with the ``serve`` subcommand.
    Returns:
        The upstream command and its explicitly tracked argument namespace.
    Raises:
        SystemExit: The upstream parser rejects an argument.
        ValueError: The upstream command rejects its configuration.
    """
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    cli_env_setup()
    parser = TrackingArgumentParser(description="NPA Cosmos3-Nano diffusion server")
    command = OmniServeCommand()
    command.subparser_init(parser.add_subparsers(required=True, dest="subparser"))
    args = parser.parse_args(argv)
    # Omni 0.28 removed stage_args, and its unregistered diffusion fallback
    # ignores deploy-file overrides. Model settings must reach that fallback
    # as explicit arguments; the upstream CLI has no model-config option yet.
    args.model_config = model_config()
    args.explicit_keys = args.explicit_keys | {"model_config"}
    command.validate(args)
    return command, args


def main() -> None:
    """Run the upstream service after applying the video checkpoint contract.

    Args:
        None; arguments are read from the process command line.
    Returns:
        None.
    Raises:
        Exception: Upstream parsing, validation, or serving fails.
    """
    command, args = create_command(sys.argv[1:])
    command.cmd(args)


if __name__ == "__main__":
    main()
