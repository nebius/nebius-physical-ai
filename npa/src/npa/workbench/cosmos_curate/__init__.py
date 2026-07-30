"""NVIDIA Cosmos Curator workbench tool.

Wraps the open-source NVIDIA Cosmos Curator
(https://github.com/nvidia-cosmos/cosmos-curate, Apache-2.0) so the Physical AI
Data Factory blueprint curates its augmented video with the curation system
NVIDIA built for physical-AI video datasets.

Upstream's curation stages are ordinary Python objects driven by a Ray pipeline
in the curator container. The clipping, transcoding, motion-scoring, and
metadata-writing stages need neither Ray nor a GPU, so :mod:`.pipeline` drives
them directly and gets upstream's canonical output layout — ``clips/``,
``metas/v0/``, ``processed_videos/`` — from upstream's own code. When the full
container is available, :func:`split_pipeline_argv` builds the documented
``video-pipeline split`` command that adds the GPU stages (TransNetV2 shot
detection, aesthetic filtering, embeddings, VLM captioning); both write the same
layout, and :mod:`.report` reads either.

:func:`curate_augmented` is the blueprint's entry point: stage the run's
augmented variants down, curate them, publish the curator's output, and summarize
it as an ``npa.cosmos_curate.curation.v1`` report.
"""

from __future__ import annotations

from npa.workbench.cosmos_curate.pipeline import (
    ENGINE_IN_PROCESS,
    ENGINE_PIPELINE_CLI,
    CuratorRunResult,
    curate_videos,
    discover_videos,
    split_pipeline_argv,
)
from npa.workbench.cosmos_curate.report import (
    RESULT_FILENAME,
    CuratedClip,
    CurationReport,
    curate_augmented,
    ingest_output,
    result_uri_for,
    write_report,
)
from npa.workbench.cosmos_curate.upstream import (
    UPSTREAM_LICENSE,
    UPSTREAM_REPO,
    CosmosCurateError,
    CosmosCurateUnavailable,
    CuratorAvailability,
    probe_availability,
    upstream_source_dir,
)

__all__ = [
    "ENGINE_IN_PROCESS",
    "ENGINE_PIPELINE_CLI",
    "RESULT_FILENAME",
    "UPSTREAM_LICENSE",
    "UPSTREAM_REPO",
    "CosmosCurateError",
    "CosmosCurateUnavailable",
    "CuratedClip",
    "CurationReport",
    "CuratorAvailability",
    "CuratorRunResult",
    "curate_augmented",
    "curate_videos",
    "discover_videos",
    "ingest_output",
    "probe_availability",
    "result_uri_for",
    "split_pipeline_argv",
    "upstream_source_dir",
    "write_report",
]
