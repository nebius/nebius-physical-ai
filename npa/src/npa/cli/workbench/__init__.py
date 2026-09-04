"""npa workbench — parent group for all workbench tools."""

from __future__ import annotations

import os

import typer

from npa.clients.credentials import load_credentials

_LIGHT_IMPORT = os.environ.get("NPA_SKIP_EAGER_IMPORTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_LIGHT_TOOL = os.environ.get("NPA_LIGHT_WORKBENCH_TOOL", "").strip().lower()


def _groot_light_app() -> typer.Typer:
    """Build the dependency-minimal workbench surface baked into the GR00T image."""

    from npa.cli.groot import app as groot_app

    light = typer.Typer(
        name="workbench",
        help="Physical AI workbench tools.",
        no_args_is_help=True,
    )

    @light.callback()
    def main() -> None:
        """Physical AI workbench tools."""

        load_credentials(
            warn=lambda msg: typer.echo(msg, err=True),
            export_to_environment=True,
        )

    light.add_typer(groot_app, name="groot")
    return light


def _full_app() -> typer.Typer:
    """Build the complete workstation command tree on ordinary clients."""

    from npa.cli.cosmos import app as cosmos_app
    from npa.cli.fiftyone import app as fiftyone_app
    from npa.cli.genesis import app as genesis_app
    from npa.cli.groot import app as groot_app
    from npa.cli.isaac_lab import app as isaac_lab_app
    from npa.cli.nurec import app as nurec_app
    from npa.cli.workbench.alpamayo2_super import app as alpamayo2_super_app
    from npa.cli.workbench.byof import app as byof_app
    from npa.cli.workbench.cosmos2 import app as cosmos2_app
    from npa.cli.workbench.cosmos3 import app as cosmos3_app
    from npa.cli.workbench.cosmos_curate import app as cosmos_curate_app
    from npa.cli.workbench.cosmos_evaluator import app as cosmos_evaluator_app
    from npa.cli.workbench.data import app as data_app
    from npa.cli.workbench.dataset import app as dataset_app
    from npa.cli.workbench.detection_training import app as detection_training_app
    from npa.cli.workbench.foxglove import app as foxglove_app
    from npa.cli.workbench.golden_eval import app as golden_eval_app
    from npa.cli.workbench.health import app as health_app
    from npa.cli.workbench.insights import app as insights_app
    from npa.cli.workbench.lancedb import app as lancedb_app
    from npa.cli.workbench.leisaac import app as leisaac_app
    from npa.cli.workbench.lerobot import app as lerobot_app
    from npa.cli.workbench.lichtblick import app as lichtblick_app
    from npa.cli.workbench.ltx2 import app as ltx2_app
    from npa.cli.workbench.mjlab import app as mjlab_app
    from npa.cli.workbench.robocasa import app as robocasa_app
    from npa.cli.workbench.scenario_gen import app as scenario_gen_app
    from npa.cli.workbench.sim2real import app as sim2real_app
    from npa.cli.workbench.sim2real_envgen import app as sim2real_envgen_app
    from npa.cli.workbench.sonic import app as sonic_app
    from npa.cli.workbench.token_factory import app as token_factory_app
    from npa.cli.workbench.vlm_eval import app as vlm_eval_app
    from npa.cli.workbench.workflow import app as workflow_app
    full = typer.Typer(
        name="workbench",
        help="Physical AI workbench tools.",
        no_args_is_help=True,
    )

    @full.callback()
    def main() -> None:
        """Physical AI workbench tools."""

        load_credentials(
            warn=lambda msg: typer.echo(msg, err=True),
            export_to_environment=True,
        )

    full.add_typer(lerobot_app, name="lerobot")
    full.add_typer(cosmos_app, name="cosmos")
    full.add_typer(cosmos2_app, name="cosmos2")
    full.add_typer(cosmos3_app, name="cosmos3")
    full.add_typer(cosmos_curate_app, name="cosmos-curate")
    full.add_typer(cosmos_evaluator_app, name="cosmos-evaluator")
    full.add_typer(fiftyone_app, name="fiftyone")
    full.add_typer(foxglove_app, name="foxglove")
    full.add_typer(genesis_app, name="genesis")
    full.add_typer(groot_app, name="groot")
    full.add_typer(isaac_lab_app, name="isaac-lab")
    full.add_typer(leisaac_app, name="leisaac")
    full.add_typer(nurec_app, name="nurec")
    full.add_typer(sonic_app, name="sonic")
    full.add_typer(mjlab_app, name="mjlab")
    full.add_typer(robocasa_app, name="robocasa")
    full.add_typer(lichtblick_app, name="lichtblick")
    full.add_typer(ltx2_app, name="ltx2")
    full.add_typer(alpamayo2_super_app, name="alpamayo2-super")
    full.add_typer(lancedb_app, name="lancedb")
    full.add_typer(detection_training_app, name="detection-training")
    full.add_typer(scenario_gen_app, name="scenario-gen")
    full.add_typer(dataset_app, name="dataset")
    full.add_typer(insights_app, name="insights")
    full.add_typer(vlm_eval_app, name="vlm-eval")
    full.add_typer(token_factory_app, name="token-factory")
    full.add_typer(byof_app, name="byof")
    full.add_typer(workflow_app, name="workflow")
    full.add_typer(health_app, name="health")
    full.add_typer(sim2real_app, name="sim2real", hidden=True)
    # Internal typed surface for npa.workflow toolRefs. Keep it out of Workbench
    # help: the public Sim2Real command family remains intentionally retired.
    full.add_typer(sim2real_envgen_app, name="sim2real-envgen", hidden=True)
    full.add_typer(golden_eval_app, name="golden-eval")
    # Backward-compatible S3 bridge; not advertised in workbench --help.
    full.add_typer(data_app, name="data", hidden=True)
    return full


def _rerun_viewer_light_app() -> typer.Typer:
    """Build the dependency-minimal nurec surface for the Rerun viewer image.

    The npa-rerun-viewer image bakes the light flag, but the workflow stages it
    exists for (``workbench.nurec.visualize`` / ``workbench.nurec.finalize``)
    live under ``npa workbench nurec`` — without this registration those stages
    fail with "No such command 'nurec'" even though the image's setup installed
    the nurec runtime deps. The nurec CLI chain imports only stdlib + typer, so
    it is safe on the dependency-minimal surface.
    """

    from npa.cli.nurec import app as nurec_app

    light = typer.Typer(
        name="workbench",
        help="Physical AI workbench tools.",
        no_args_is_help=True,
    )

    @light.callback()
    def main() -> None:
        """Physical AI workbench tools."""

        load_credentials(
            warn=lambda msg: typer.echo(msg, err=True),
            export_to_environment=True,
        )

    light.add_typer(nurec_app, name="nurec")
    return light


if _LIGHT_IMPORT:
    # Capability images need only this one command and deliberately omit the
    # unrelated platform SDK dependency tree. Preserve the historical Cosmos2
    # surface unless an image explicitly declares another narrow capability.
    if _LIGHT_TOOL == "groot":
        app = _groot_light_app()
    elif _LIGHT_TOOL == "rerun-viewer":
        app = _rerun_viewer_light_app()
    else:
        from npa.cli.workbench.cosmos2 import app
else:
    app = _full_app()
