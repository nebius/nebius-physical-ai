"""Compose conditioned video generation, paired visual review, and dataset lineage."""


def build_parser():
    """Return the workflow stage parser.

    Args:
        None.
    Returns:
        The stage argument parser.
    Raises:
        None.
    """
    from npa.workflows.video_sweep.__main__ import build_parser as stage_parser

    return stage_parser()
