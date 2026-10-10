"""Render a network-free evaluator compatibility check for baked Sim2Real images."""

from __future__ import annotations

import textwrap


def render_evaluator_probe(model: str) -> str:
    """Check the selected evaluator in the image before the first GPU wave.

    Args:
        model: Exact operator-selected hosted model ID.

    Returns:
        Python source executed by the image's baked NPA interpreter.

    Raises:
        None.
    """
    return textwrap.dedent(f"""\
        try:
            from npa.workbench.cosmos.reason import hosted_rollout_model_family
            from npa.workflows.sim2real.stage9_evaluator import (
                EvaluatorContractError, validate_hosted_evaluator,
            )
            family = hosted_rollout_model_family({model!r})
            from npa.workflows.sim2real.hosted_evaluation import evaluate_rollouts
            from npa.workflows.sim2real.workflow_stage import build_parser
            options = build_parser().parse_args([
                '--stage', '8', '--root-uri', 's3://probe/run', '--run-id', 'probe',
                '--evaluation-concurrency', '8', '--evaluation-max-frames', '0',
            ])
            if (not callable(evaluate_rollouts) or
                    options.evaluation_concurrency != 8 or
                    options.evaluation_max_frames != 0):
                raise RuntimeError('image lacks concurrent durable rollout evaluation')
            try:
                validate_hosted_evaluator(
                    stage7={{}}, evaluator={{}}, stage8_record={{}},
                    expected_model={model!r}, expected_source_sha=actual,
                    evaluator_uri='', outer_iteration=0, inner_iteration=0,
                )
            except EvaluatorContractError:
                pass
            else:
                raise RuntimeError('evaluator accepted an empty Stage 8 boundary')
        except (ImportError, ValueError, RuntimeError, TypeError) as exc:
            raise SystemExit(
                'Sim2Real evaluator image is incompatible with the selected model '
                + {model!r} + ': ' + str(exc) + '. Select a coherent image set '
                'with the current Stage 8/9 evaluator contract and use a new run ID.'
            ) from exc
        print('baked Sim2Real evaluator verified', {model!r}, family)
        """)
