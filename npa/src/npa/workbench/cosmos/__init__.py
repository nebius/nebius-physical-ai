"""npa.workbench.cosmos - NVIDIA Cosmos workbench commands."""

from __future__ import annotations

from npa._sdk import make_cli_wrapper
from npa.workbench.cosmos.cosmos3 import (
    Cosmos3AccessConfig as Cosmos3AccessConfig,
    Cosmos3AccessError as Cosmos3AccessError,
    Cosmos3CheckResult as Cosmos3CheckResult,
    Cosmos3FetchResult as Cosmos3FetchResult,
    Cosmos3ServeConfig as Cosmos3ServeConfig,
    build_cosmos3_inference_args,
    check_cosmos3_access,
    fetch_cosmos3_artifacts,
)
from npa.workbench.cosmos.workflows import (
    COSMOS_AUGMENT_YAML,
    COSMOS_ATTRIBUTION,
    COSMOS_REASON_YAML,
    CosmosSkyLaunchResult,
    build_cosmos_augment_env,
    build_cosmos_reason_env,
    launch_cosmos_sky_workflow,
)

ensure_ingress = make_cli_wrapper(
    "npa.cli.cosmos", "ensure_ingress_cmd", "Ensure ingress for a Cosmos workbench."
)
register_byovm = make_cli_wrapper(
    "npa.cli.cosmos", "register_byovm_cmd", "Register a BYOVM Cosmos workbench."
)
check = make_cli_wrapper(
    "npa.cli.cosmos", "check_cmd", "Check Cosmos3 source and HF access."
)
fetch = make_cli_wrapper(
    "npa.cli.cosmos", "fetch_cmd", "Fetch Cosmos3 source and HF artifacts."
)
list = make_cli_wrapper("npa.cli.cosmos", "list_cmd", "List Cosmos workbenches.")
deploy = make_cli_wrapper("npa.cli.cosmos", "deploy_cmd", "Deploy a Cosmos workbench.")
autoscale = make_cli_wrapper(
    "npa.cli.cosmos", "autoscale_cmd", "Configure Cosmos serverless autoscaling."
)
serve = make_cli_wrapper("npa.cli.cosmos", "serve_cmd", "Serve a Cosmos model.")
finetune = make_cli_wrapper("npa.cli.cosmos", "finetune_cmd", "Run Cosmos finetuning.")
optimize = make_cli_wrapper(
    "npa.cli.cosmos", "optimize_cmd", "Run Cosmos optimization."
)
infer = make_cli_wrapper("npa.cli.cosmos", "infer_cmd", "Run Cosmos inference.")
augment = make_cli_wrapper(
    "npa.cli.cosmos",
    "augment_cmd",
    "Run Cosmos controlled-generation augmentation.",
)
reason = make_cli_wrapper(
    "npa.cli.cosmos",
    "reason_cmd",
    "Run Cosmos reasoning/VLM evaluation.",
)
status = make_cli_wrapper("npa.cli.cosmos", "status_cmd", "Show Cosmos status.")
system_info = make_cli_wrapper(
    "npa.cli.cosmos", "system_info_cmd", "Show Cosmos system information."
)

__all__ = [
    "check_cosmos3_access",
    "fetch_cosmos3_artifacts",
    "build_cosmos3_inference_args",
    "CosmosSkyLaunchResult",
    "build_cosmos_augment_env",
    "build_cosmos_reason_env",
    "launch_cosmos_sky_workflow",
    "ensure_ingress",
    "register_byovm",
    "check",
    "fetch",
    "list",
    "deploy",
    "autoscale",
    "serve",
    "finetune",
    "optimize",
    "infer",
    "augment",
    "reason",
    "status",
    "system_info",
]
