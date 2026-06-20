# NPA workflow specifications (apiVersion: npa.workflow/v0.0.1)

**Start here:** [docs/workbench/npa-workflow-guide.md](../../../docs/workbench/npa-workflow-guide.md)

Consumption paths (same workflow, three entrypoints):

- **YAML:** this directory
- **CLI:** `npa workbench workflow validate-spec|plan-spec|run-spec`
- **SDK:** `npa.orchestration.npa_workflow.load_spec / build_plan / run_workflow`

Golden examples:

| Spec | Purpose |
| --- | --- |
| `vlm-eval-single.yaml` | Single-tool minimal |
| `tokenfactory-rollout-judge.yaml` | Serial two-tool |
| `sim2real-vlm-rl.yaml` | Fixed + dynamic loops |

See also `docs/workbench/npa-workflow-tool-catalog.md` for `toolRef` and token rules.
