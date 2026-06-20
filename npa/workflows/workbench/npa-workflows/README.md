# NPA workflow specifications (apiVersion: npa.workflow/v0.0.1)
#
# Consumption paths (same workflow, three entrypoints):
#   YAML: npa workbench workflow validate-spec|plan-spec|run-spec <this-file>
#   CLI:  same commands above
#   SDK:  npa.orchestration.npa_workflow.load_spec / build_plan / run_workflow
#
# SkyPilot submits each planned step as a GPU pod; the spec defines shape and tool
# commands — not sim2real-specific Python orchestrators.
#
# See docs/workbench/npa-workflow-tool-catalog.md for toolRef and token rules.
