"""npa eval-harness — standardized policy A/B evaluation.

Runs policies on registered manipulation tasks for N episodes, applies a
success judge per episode, and writes JSON + Markdown reports with success
rates, Wilson confidence intervals, and (for ``compare``) paired-seed A/B
comparisons with bootstrap confidence intervals.  Runs locally by default.
"""

from __future__ import annotations

import typer
from rich.console import Console

from npa.cli.path_contract import PathContractError, validate_write_path

app = typer.Typer(
    name="eval-harness",
    help="Standardized policy evaluation: single-policy runs and paired A/B comparisons.",
    no_args_is_help=True,
)

console = Console(stderr=True)


@app.command("run")
def run_cmd(
    task: str = typer.Option(
        ..., "--task", help="Registered task, e.g. 'mujoco_manip.peg_insertion'."
    ),
    policy: str = typer.Option(
        ...,
        "--policy",
        help="Policy spec: 'scripted:<expert|noisy|random>' or 'python:<module>:<callable>'.",
    ),
    episodes: int = typer.Option(
        10, "--episodes", help="Number of evaluation episodes."
    ),
    seed: int = typer.Option(
        0, "--seed", help="Base random seed (episode i uses seed+i)."
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="S3 output prefix for report.json + report.md (--output-uri is a compatibility alias).",
    ),
    judge: str = typer.Option(
        "heuristic", "--judge", help="Success judge: 'heuristic' or 'vlm'."
    ),
    max_steps: int = typer.Option(
        500, "--max-steps", help="Maximum control steps per episode."
    ),
    vlm_endpoint_url: str = typer.Option(
        "",
        "--vlm-endpoint-url",
        help="VLM endpoint URL for --judge vlm (API key via $VLM_EVAL_API_KEY).",
    ),
) -> None:
    """Evaluate one policy on a task and write JSON + Markdown reports."""
    from npa.workflows.byof.eval_harness_pipeline import run as stage_run

    try:
        output_path = validate_write_path(
            output_path,
            tool="eval-harness run",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        raise typer.BadParameter(str(exc), param_hint="--output-path") from exc
    result = stage_run(
        task=task,
        policy=policy,
        output_path=output_path,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )
    summary = result["summary"]
    console.print(
        f"[green]done:[/green] {task} x {policy}: "
        f"success_rate={summary['success_rate']:.3f} "
        f"({summary['successes']}/{summary['episodes']}) "
        f"-> {result['artifacts']['report_markdown']}"
    )


@app.command("compare")
def compare_cmd(
    task: str = typer.Option(
        ..., "--task", help="Registered task, e.g. 'mujoco_manip.peg_insertion'."
    ),
    policy_a: str = typer.Option(
        ..., "--policy-a", help="Policy A spec ('scripted:...' or 'python:...')."
    ),
    policy_b: str = typer.Option(
        ..., "--policy-b", help="Policy B spec ('scripted:...' or 'python:...')."
    ),
    episodes: int = typer.Option(
        10, "--episodes", help="Number of paired episodes per policy."
    ),
    seed: int = typer.Option(
        0, "--seed", help="Base random seed (episode i uses seed+i for both)."
    ),
    output_path: str = typer.Option(
        ...,
        "--output-path",
        "--output-uri",
        help="S3 output prefix for compare.json + compare.md (--output-uri is a compatibility alias).",
    ),
    judge: str = typer.Option(
        "heuristic", "--judge", help="Success judge: 'heuristic' or 'vlm'."
    ),
    max_steps: int = typer.Option(
        500, "--max-steps", help="Maximum control steps per episode."
    ),
    vlm_endpoint_url: str = typer.Option(
        "",
        "--vlm-endpoint-url",
        help="VLM endpoint URL for --judge vlm (API key via $VLM_EVAL_API_KEY).",
    ),
) -> None:
    """A/B compare two policies with paired episode seeds."""
    from npa.workflows.byof.eval_harness_pipeline import compare as stage_compare

    try:
        output_path = validate_write_path(
            output_path,
            tool="eval-harness compare",
            option="--output-path",
            required=True,
        )
    except PathContractError as exc:
        raise typer.BadParameter(str(exc), param_hint="--output-path") from exc
    result = stage_compare(
        task=task,
        policy_a=policy_a,
        policy_b=policy_b,
        output_path=output_path,
        episodes=episodes,
        seed=seed,
        judge=judge,
        max_steps=max_steps,
        vlm_endpoint_url=vlm_endpoint_url,
    )
    summary = result["summary"]
    ci = summary["bootstrap_95"]
    console.print(
        f"[green]done:[/green] {task}: A-B diff={summary['success_rate_diff']:+.3f} "
        f"95% CI [{ci['lo']:+.3f}, {ci['hi']:+.3f}] "
        f"-> {result['artifacts']['compare_markdown']}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
