"""Gating tests for the shipped `neural-reconstruction` capability.

Mirrors ``test_cosmos3_agent_skills_are_discoverable_and_well_formed``: the skill
must be indexed, attributed, and point only at real entrypoints, and the SkyPilot
workflow must keep the properties the capability depends on (RT-core GPU routing,
real stage commands, the Rerun recording the agent displays).
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills"
SKILL_INDEX = SKILL_ROOT / "index.yaml"
SKILL_PATH = SKILL_ROOT / "workflows" / "neural-reconstruction" / "SKILL.md"
WORKFLOW = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot" / "nurec-reconstruct.yaml"
SPEC = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "nurec-reconstruct.yaml"
)

#: GPUs with no RT cores. Reconstruction and rasterization are RT-core work, so a
#: reference to any of these in the workflow is a routing bug.
NON_RT_CORE_GPUS = ("H100", "H200", "A100", "B200", "GH200")


def _workflow() -> dict:
    documents = [doc for doc in yaml.safe_load_all(WORKFLOW.read_text(encoding="utf-8")) if doc]
    assert len(documents) == 1, "the NuRec workflow is a single SkyPilot task"
    return documents[0]


# ---------------------------------------------------------------------------------
# skill discoverability + attribution
# ---------------------------------------------------------------------------------
def test_neural_reconstruction_skill_is_discoverable_and_well_formed() -> None:
    index = yaml.safe_load(SKILL_INDEX.read_text())
    entries = {entry["name"]: entry for entry in index["skills"]}

    assert "neural-reconstruction" in entries
    entry = entries["neural-reconstruction"]
    assert entry["category"] == "workflows"
    assert entry["when_to_use"].strip()
    assert (REPO_ROOT / entry["path"]).exists()

    text = SKILL_PATH.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
    assert frontmatter["name"] == "neural-reconstruction"
    assert frontmatter["description"].strip()

    assert "Source And Attribution" in text
    assert "NVIDIA CORPORATION & AFFILIATES" in text
    assert "NOTICE-NVIDIA-SKILLS" in text


def test_notice_file_covers_the_new_skill() -> None:
    index = yaml.safe_load(SKILL_INDEX.read_text())
    notice = next(
        entry
        for entry in index["licenses"]
        if entry["path"] == "skills/NOTICE-NVIDIA-SKILLS"
    )

    assert "neural-reconstruction" in notice["applies_to"]
    body = (SKILL_ROOT / "NOTICE-NVIDIA-SKILLS").read_text(encoding="utf-8")
    assert "NVIDIA CORPORATION & AFFILIATES" in body
    assert "nurec-skills" in body
    assert "NVIDIA/ncore" in body


def test_skill_smoke_entries_point_at_files_that_exist() -> None:
    index = yaml.safe_load(SKILL_INDEX.read_text())
    entry = next(item for item in index["skills"] if item["name"] == "neural-reconstruction")

    kinds = {smoke["type"] for smoke in entry["smoke"]}
    assert {"file_exists", "cli_help", "workflow_yaml", "npa_workflow_yaml"} <= kinds
    for smoke in entry["smoke"]:
        for relative in smoke.get("paths", []) or ([smoke["path"]] if "path" in smoke else []):
            assert (REPO_ROOT / relative).exists(), relative


def test_skill_documents_the_rig_pose_failure_and_its_fix() -> None:
    # The single most likely way a user gets stuck is an NCore sequence NRE
    # refuses to load; the skill has to name the error and the remedy.
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "Rig-to-world poses are currently required" in text
    assert "npa_rig" in text
    assert "poses_component_group" in text


def test_skill_states_the_rt_core_routing_constraint() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "RT-core" in text or "RT cores" in text
    assert "L40S" in text
    assert "H100" in text  # named explicitly as forbidden


def test_skill_records_the_ngc_entitlement_reality() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")

    # A standard NGC key cannot pull the non-GA repository.
    assert "402" in text
    assert "nre-ga" in text


# ---------------------------------------------------------------------------------
# SkyPilot workflow contract
# ---------------------------------------------------------------------------------
def test_workflow_name_matches_its_filename() -> None:
    assert _workflow()["name"] == WORKFLOW.stem


def test_workflow_routes_at_an_rt_core_gpu() -> None:
    resources = _workflow()["resources"]

    accelerators = str(resources["accelerators"])
    assert "RTXPRO" in accelerators.upper() or "L40S" in accelerators.upper()
    assert resources["cloud"] == "kubernetes"


def test_workflow_never_references_a_gpu_without_rt_cores() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )

    for gpu in NON_RT_CORE_GPUS:
        assert gpu not in body, f"{gpu} has no RT cores; NuRec must not be routed at it"


def test_workflow_rejects_a_disk_size_kubernetes_will_not_accept() -> None:
    assert "disk_size" not in _workflow()["resources"]


def test_workflow_uses_the_pullable_ga_container() -> None:
    envs = _workflow()["envs"]

    assert envs["NPA_NUREC_IMAGE"].startswith("nvcr.io/nvidia/nre/nre-ga:")
    assert _workflow()["resources"]["image_id"] == "docker:${NPA_NUREC_IMAGE}"


def test_workflow_supplies_the_sudo_shim_the_vendor_image_lacks() -> None:
    # Without this, SkyPilot's runtime setup exits 127 on `sudo` and the cluster
    # never becomes usable even though provisioning succeeds.
    spec = _workflow()["config"]["kubernetes"]["pod_config"]["spec"]

    init = {container["name"]: container for container in spec["initContainers"]}
    assert "npa-sudo-shim" in init
    script = "\n".join(str(part) for part in init["npa-sudo-shim"]["command"])
    assert "/shim/sudo" in script
    assert 'exec "$@"' in script

    ray_node = next(c for c in spec["containers"] if c["name"] == "ray-node")
    mounts = {mount["name"]: mount["mountPath"] for mount in ray_node["volumeMounts"]}
    # /usr/local/sbin is on the image's PATH and empty.
    assert mounts["npa-sudo-shim"] == "/usr/local/sbin"


def test_workflow_enlarges_dev_shm_for_nre() -> None:
    spec = _workflow()["config"]["kubernetes"]["pod_config"]["spec"]

    ray_node = next(c for c in spec["containers"] if c["name"] == "ray-node")
    mounts = {mount["name"]: mount["mountPath"] for mount in ray_node["volumeMounts"]}
    assert mounts["dshm"] == "/dev/shm"

    dshm = next(volume for volume in spec["volumes"] if volume["name"] == "dshm")
    assert dshm["emptyDir"]["medium"] == "Memory"


def test_workflow_pulls_from_ngc_with_the_cluster_pull_secret() -> None:
    spec = _workflow()["config"]["kubernetes"]["pod_config"]["spec"]

    secrets = {item["name"] for item in spec["imagePullSecrets"]}
    assert "ngc-nvcr-imagepullsecret" in secrets


def test_workflow_keeps_every_cache_under_tmp() -> None:
    envs = _workflow()["envs"]

    for key in ("NPA_NUREC_CACHE", "NPA_NUREC_OUT", "NPA_NUREC_RENDER_DIR"):
        assert envs[key].startswith("/tmp/"), key


def test_workflow_runs_only_real_nurec_entrypoints() -> None:
    run = _workflow()["run"]

    for verb in ("check", "fetch", "reconstruct", "render", "visualize", "finalize"):
        assert f"npa workbench nurec {verb}" in run, verb
    # No echo/manifest stub standing in for a stage.
    assert "contract_ready" not in run


def test_workflow_produces_the_artifact_the_agent_prefers() -> None:
    run = _workflow()["run"]

    assert "reports/sim2real.rrd" in run
    assert "reports/final.json" in run


def test_workflow_renders_novel_views_not_training_views() -> None:
    envs = _workflow()["envs"]
    run = _workflow()["run"]

    # A zero offset would silently reproduce the training views.
    offset = envs["NPA_NUREC_RIG_TRANSLATION_OFFSET"]
    assert any(float(part) != 0.0 for part in offset.split(","))
    assert "--rig-translation-offset" in run


def test_workflow_env_placeholders_are_substitutable_not_shadowed() -> None:
    """Run-specific env values must stay ``${...}`` placeholders.

    ``--var`` substitution is textual, so a literal default here would shadow it
    and the stage would run with an empty value (observed live: an empty
    NPA_SRC_S3_URI failed setup with `Invalid bucket name ""`).
    """
    envs = _workflow()["envs"]

    for key in (
        "NPA_NUREC_RUN_ID",
        "NPA_NUREC_RUN_URI",
        "NPA_SRC_S3_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NGC_API_KEY",
    ):
        assert envs[key] == f"${{{key}}}", key


def test_workflow_contains_no_committed_secret_or_bucket() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "nvapi-" not in text
    assert "hf_" not in text.replace("hf_token", "").replace("HF_TOKEN", "")
    # No hardcoded bucket: the destination is a placeholder.
    assert "s3://lerobot" not in text


# ---------------------------------------------------------------------------------
# declarative twin
# ---------------------------------------------------------------------------------
def test_declarative_spec_is_an_npa_workflow_with_real_toolrefs() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    assert spec["apiVersion"] == "npa.workflow/v0.0.1"
    assert spec["metadata"]["name"] == SPEC.stem
    tool_refs = [state["toolRef"] for state in spec["states"].values() if state.get("toolRef")]
    assert tool_refs, "every NuRec stage runs a real tool"
    for tool_ref in tool_refs:
        assert tool_ref.startswith("workbench.nurec."), tool_ref
        assert tool_ref in TOOL_CATALOG, tool_ref


def test_declarative_spec_validates() -> None:
    from npa.orchestration.npa_workflow import load_spec, validate_spec

    validate_spec(load_spec(SPEC))


def test_declarative_spec_routes_the_gpu_stages_at_rt_cores() -> None:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    accelerators = str(spec["resources"]["gpu"]["accelerators"]).upper()
    assert "RTXPRO" in accelerators or "L40S" in accelerators
    for gpu in NON_RT_CORE_GPUS:
        assert gpu not in accelerators


def test_declarative_spec_ends_at_the_rerun_recording_and_report() -> None:
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))

    # Stage outputs reference config keys, so resolve one level of templating.
    outputs = [
        output["uri"]
        for state in spec["states"].values()
        for output in state.get("outputs", []) or []
    ]
    assert "{{config.rrd_uri}}" in outputs
    assert spec["config"]["rrd_uri"].endswith("/reports/sim2real.rrd")
    assert "{{config.final_report_uri}}" in outputs
    assert spec["config"]["final_report_uri"].endswith("/reports/final.json")
    assert spec["states"]["finalize"].get("terminal") is True


def test_catalog_entries_call_the_real_cli_flags() -> None:
    """Every catalog flag must be an actual option of the real CLI command."""
    from npa.cli.main import app as main_app
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    node = typer.main.get_command(main_app)
    workbench = node.commands["workbench"]
    nurec = workbench.commands["nurec"]

    checked = 0
    for name, entry in TOOL_CATALOG.items():
        if not name.startswith("workbench.nurec."):
            continue
        argv = entry.argv_template
        assert argv[:3] == ["npa", "workbench", "nurec"], name
        verb = argv[3]
        assert verb in nurec.commands, f"{name} -> unknown verb {verb}"
        options: set[str] = set()
        for param in nurec.commands[verb].params:
            options.update(getattr(param, "opts", []) or [])
            options.update(getattr(param, "secondary_opts", []) or [])
        for token in argv[4:]:
            if token.startswith("--"):
                assert token in options, f"{name}: {token} is not a real option of {verb}"
        checked += 1
    assert checked >= 6, "expected every nurec verb to be catalogued"
