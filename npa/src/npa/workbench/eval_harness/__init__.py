"""npa.workbench.eval_harness - standardized policy A/B evaluation.

Runs policies on registered manipulation tasks for N episodes, applies a
success judge per episode (heuristic environment signal, or a real
VLM-as-judge client over :mod:`npa.workbench.vlm_eval`), and writes JSON +
Markdown reports with success rates, Wilson confidence intervals, and --
for A/B runs -- paired-seed comparisons with bootstrap confidence intervals.

The workbench-first SDK surface re-exports the pipeline stages (``run``,
``compare``) alongside the runner, metrics, and judge API; the CLI is a thin
client.

Note on import order below: the ``compare`` submodule is imported before the
pipeline stage functions are rebound, because importing a submodule binds its
name on the package -- the pipeline ``run``/``compare`` stage functions must
win that rebinding (``wb.run is pipe.run``).
"""

from __future__ import annotations

from npa.workbench.eval_harness import metrics
from npa.workbench.eval_harness.compare import CompareReport, compare_policies
from npa.workbench.eval_harness.env import ManipEnv
from npa.workbench.eval_harness.judges import (
    HeuristicJudge,
    JudgeResult,
    VLMJudge,
    make_judge,
)
from npa.workbench.eval_harness.policies import make_policy_factory
from npa.workbench.eval_harness.reports import (
    render_compare_markdown,
    render_run_markdown,
    write_compare_report,
    write_run_report,
)
from npa.workbench.eval_harness.runner import (
    EpisodeRecord,
    RunReport,
    run_policy,
)
from npa.workbench.eval_harness.tasks import (
    get_task,
    known_tasks,
    register_task,
)
from npa.workflows.byof.eval_harness_pipeline import (
    EvalHarnessPipelineError,
    compare,
    run,
)

__all__ = [
    "EvalHarnessPipelineError",
    "run",
    "compare",
    "metrics",
    "CompareReport",
    "compare_policies",
    "ManipEnv",
    "HeuristicJudge",
    "JudgeResult",
    "VLMJudge",
    "make_judge",
    "make_policy_factory",
    "render_compare_markdown",
    "render_run_markdown",
    "write_compare_report",
    "write_run_report",
    "EpisodeRecord",
    "RunReport",
    "run_policy",
    "get_task",
    "known_tasks",
    "register_task",
]
