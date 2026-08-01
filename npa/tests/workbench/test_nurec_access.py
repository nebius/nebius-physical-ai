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
    # The router skill whose picker table / mix-ups / secret-handling guidance the
    # SKILL adapts. Pinned, because an unpinned citation cannot be audited later.
    assert "NVIDIA/skills" in body
    assert "physical-ai-neural-reconstruction" in body
    assert UPSTREAM_ROUTER_PIN in body


#: Upstream commit the adapted router content was taken from. Cited in both the
#: NOTICE and the SKILL so a reader can diff against exactly that revision.
UPSTREAM_ROUTER_PIN = "0122ea0"


def test_skill_cites_the_router_upstream_it_adapts() -> None:
    """Adapted upstream content must be attributed in the SKILL body itself.

    The NOTICE covers licensing; this asserts a reader of the skill can see where
    the routing table and mix-ups came from without opening another file.
    """
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "physical-ai-neural-reconstruction" in text
    assert UPSTREAM_ROUTER_PIN in text
    assert "Apache-2.0" in text
    # The adapted sections are actually present.
    assert "## Which Capability Answers This?" in text
    assert "## Easy Mix-Ups" in text
    assert "## Verifying Secrets Safely" in text
    assert "## Troubleshooting" in text


def test_skill_warns_about_the_token_echoing_antipattern() -> None:
    """The upstream secrets guidance is only useful if the wrong form is shown.

    ``${VAR:-no}`` falls back only when the variable is EMPTY, so the common
    "is it set?" one-liner prints the token itself.
    """
    text = SKILL_PATH.read_text(encoding="utf-8")

    assert "${HF_TOKEN:+yes}${HF_TOKEN:-no}" in text
    assert "rotate" in text.lower()
    # And the safe alternative is given, not just the warning.
    assert "hf auth whoami" in text
    assert "${#HF_TOKEN}" in text


def test_routing_table_marks_unimplemented_capabilities_as_upstream() -> None:
    """Rows the workbench cannot do must not read like workbench features.

    The router skill lists many NuRec capabilities; advertising one we have no
    verb for is exactly the "real components" failure mode this repo guards
    against elsewhere.
    """
    text = SKILL_PATH.read_text(encoding="utf-8")
    table = text.split("## Which Capability Answers This?", 1)[1].split("## Easy Mix-Ups", 1)[0]

    for unimplemented in ("serve-grpc", "asset-harvester", "render-grpc"):
        row = next(ln for ln in table.splitlines() if unimplemented in ln)
        assert "Upstream" in row, f"{unimplemented} must be marked upstream: {row}"

    # Everything advertised as a workbench verb is a real CLI command.
    for verb in ("check", "fetch", "reconstruct", "render", "visualize"):
        assert f"npa workbench nurec {verb}" in table


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


# ---------------------------------------------------------------------------------
# live-infra registration (checked in the normal suite, not gated behind e2e)
# ---------------------------------------------------------------------------------
def test_nurec_case_is_registered_in_the_live_submit_matrix() -> None:
    """The gpu-tier matrix entry must exist even where a live run is not runnable."""
    from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX

    case = next(
        (item for item in SUBMIT_LIVE_MATRIX if item.spec == "nurec-reconstruct.yaml"), None
    )
    assert case is not None, "nurec-reconstruct.yaml is missing from SUBMIT_LIVE_MATRIX"
    assert case.tier == "gpu"
    assert "NGC_API_KEY" in case.secret_envs, "the nre-ga container needs an NGC key"
    assert "HF_TOKEN" in case.secret_envs, "the PhysicalAI capture needs an HF token"
    # A ~14 GB image pull plus 30k training steps needs more than the tier default.
    assert case.max_wait_seconds >= 3600


def test_live_e2e_test_exists_and_asserts_the_definition_of_done() -> None:
    live = (
        REPO_ROOT / "npa" / "tests" / "e2e" / "test_nurec_reconstruct_live_e2e.py"
    )
    assert live.is_file()
    text = live.read_text(encoding="utf-8")

    # Markers so the case is selectable with the rest of the GPU/SkyPilot tier.
    assert "pytest.mark.gpu" in text
    assert "pytest.mark.e2e_skypilot" in text
    # And it must assert the artifacts the capability promises.
    for suffix in ("reports/sim2real.rrd", "reconstruction/last.usdz", "novel_views"):
        assert suffix in text, suffix
    assert "recording_has_run_entities" in text
    assert "is_stock_demo_recording" in text


# ---------------------------------------------------------------------------------
# SDK surface (the third tier: CLI <-> SDK <-> YAML)
# ---------------------------------------------------------------------------------
def test_sdk_module_exposes_every_cli_verb() -> None:
    """`npa.sdk.workbench.nurec` must cover the CLI one-for-one.

    The repo's three-access contract is CLI + SDK + YAML; a verb that exists only
    on the CLI is invisible to SDK consumers.
    """
    from npa.cli.main import app as main_app
    from npa.sdk.workbench import nurec as sdk

    node = typer.main.get_command(main_app)
    cli_verbs = set(node.commands["workbench"].commands["nurec"].commands)

    assert set(sdk.__all__) == cli_verbs, (
        f"SDK/CLI drift: only-CLI={sorted(cli_verbs - set(sdk.__all__))}, "
        f"only-SDK={sorted(set(sdk.__all__) - cli_verbs)}"
    )
    for verb in sorted(cli_verbs):
        wrapper = getattr(sdk, verb)
        assert wrapper.__npa_cli_module__ == "npa.cli.nurec", verb
        assert wrapper.__npa_cli_callback__ == f"{verb}_cmd", verb


def test_sdk_namespace_registers_the_tool() -> None:
    import npa.sdk.workbench as workbench

    assert "nurec" in workbench.__all__
    assert hasattr(workbench, "nurec")


def test_workbench_package_also_exposes_the_pure_api() -> None:
    """The tool package re-exports the framework-free API for direct use."""
    from npa.workbench import nurec

    for name in (
        "NurecConfig",
        "check_nurec_access",
        "fetch_nurec_dataset",
        "reconstruct_scene",
        "render_novel_views",
        "build_nre_train_args",
        "build_nre_render_args",
        "has_rt_cores",
    ):
        assert name in nurec.__all__, name
        assert hasattr(nurec, name), name


# ---------------------------------------------------------------------------------
# declarative spec: cross-pod handoff + pod accommodations
# ---------------------------------------------------------------------------------
# Each state of an npa.workflow spec runs in its OWN pod, so anything the previous
# stage left in /tmp is gone. These tests pin the two consequences.
def _spec() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def test_spec_hands_the_ncore_sequence_between_pods() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    fetch = TOOL_CATALOG["workbench.nurec.fetch"].argv_template
    reconstruct = TOOL_CATALOG["workbench.nurec.reconstruct"].argv_template

    # Publishing is what makes the sequence reachable from another pod.
    assert "--publish-sequence" in fetch
    assert "--ncore-uri" in reconstruct
    handoff = reconstruct[reconstruct.index("--ncore-uri") + 1]
    assert handoff == "{{config.ncore_sequence_uri}}"
    assert _spec()["config"]["ncore_sequence_uri"].endswith("/ncore/sequence/")


def test_spec_hands_the_trained_usdz_between_pods() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    reconstruct = TOOL_CATALOG["workbench.nurec.reconstruct"].argv_template
    render = TOOL_CATALOG["workbench.nurec.render"].argv_template

    published = reconstruct[reconstruct.index("--output-uri") + 1]
    consumed = render[render.index("--artifact-uri") + 1]
    assert published == consumed == "{{config.reconstruction_uri}}"


def test_spec_gpu_profile_supplies_the_sudo_shim_and_large_shm() -> None:
    kubernetes = _spec()["resources"]["gpu"]["kubernetes"]

    assert kubernetes["provision_timeout"] >= 1800
    spec = kubernetes["pod_config"]["spec"]
    init = {c["name"]: c for c in spec["initContainers"]}
    assert "npa-sudo-shim" in init
    script = "\n".join(str(part) for part in init["npa-sudo-shim"]["command"])
    assert "/shim/sudo" in script
    assert 'exec "$@"' in script
    # A literal backslash-n, not a real newline: printf has to interpret it.
    assert "\\n" in script

    ray_node = next(c for c in spec["containers"] if c["name"] == "ray-node")
    mounts = {m["name"]: m["mountPath"] for m in ray_node["volumeMounts"]}
    assert mounts["npa-sudo-shim"] == "/usr/local/sbin"
    assert mounts["dshm"] == "/dev/shm"
    dshm = next(v for v in spec["volumes"] if v["name"] == "dshm")
    assert dshm["emptyDir"]["medium"] == "Memory"


def test_spec_init_container_image_matches_the_runtime_image() -> None:
    """The shim runs in the same image, so there is no second registry dependency."""
    gpu = _spec()["resources"]["gpu"]
    init = gpu["kubernetes"]["pod_config"]["spec"]["initContainers"][0]

    assert init["image"] == gpu["image"]


def test_renderer_lifts_the_pod_config_onto_gpu_tasks_only() -> None:
    """pod_config must reach the GPU tasks and stay off the CPU ones."""
    from npa.orchestration.npa_workflow.skypilot_render import normalize_task_config

    spec = _spec()
    gpu = normalize_task_config(spec["resources"]["gpu"])
    cpu = normalize_task_config(spec["resources"]["cpu"])

    assert gpu["kubernetes"]["pod_config"]["spec"]["initContainers"]
    assert gpu["kubernetes"]["provision_timeout"] >= 1800
    assert cpu == {}


def test_renderer_task_config_ignores_unknown_kubernetes_fields() -> None:
    """A spec must not be able to smuggle arbitrary cluster config into a task."""
    from npa.orchestration.npa_workflow.skypilot_render import normalize_task_config

    out = normalize_task_config(
        {"kubernetes": {"pod_config": {"spec": {}}, "remote_identity": "evil-sa"}}
    )

    assert "remote_identity" not in out["kubernetes"]


def test_renderer_stages_npa_source_into_a_vendor_image(monkeypatch) -> None:
    """A pinned VENDOR image still needs npa staged in.

    The NuRec runtime is NVIDIA's NRE container, which has never heard of npa.
    The renderer previously withheld NPA_SRC_S3_URI whenever an image was pinned
    (assuming a baked workbench image), so every stage died in setup with
    "npa CLI not found; set NPA_SRC_S3_URI or use a workbench image" -- observed
    live on job 228. Setup's install path is guarded by `command -v npa`, so
    propagating the URI is a no-op for images that really do bake npa in.
    """
    from npa.orchestration.npa_workflow import build_plan, load_spec
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        build_skypilot_task_doc,
    )

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/test")
    monkeypatch.delenv("NPA_SRC_OVERLAY", raising=False)

    spec = load_spec(SPEC)
    plan = build_plan(spec)
    step = next(s for s in plan.steps if s.state == "reconstruct")
    doc = build_skypilot_task_doc(
        spec, step, run_id="test-run", options=SkypilotRenderOptions()
    )

    # The vendor image is pinned...
    assert doc["resources"]["image_id"].startswith("docker:nvcr.io/nvidia/nre/nre-ga")
    # ...and the source is staged anyway, so setup can install npa WITH deps.
    assert doc["envs"]["NPA_SRC_S3_URI"] == "s3://example-bucket/npa-src/test"
    # The --no-deps overlay path stays opt-in and must NOT be triggered here.
    assert "NPA_SRC_OVERLAY" not in doc["envs"]


def test_renderer_overlay_stays_opt_in(monkeypatch) -> None:
    from npa.orchestration.npa_workflow import build_plan, load_spec
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        build_skypilot_task_doc,
    )

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/test")
    monkeypatch.setenv("NPA_SRC_OVERLAY", "1")

    spec = load_spec(SPEC)
    plan = build_plan(spec)
    step = next(s for s in plan.steps if s.state == "reconstruct")
    doc = build_skypilot_task_doc(
        spec, step, run_id="test-run", options=SkypilotRenderOptions()
    )

    assert doc["envs"]["NPA_SRC_OVERLAY"] == "1"


def test_spec_cpu_stages_do_not_request_a_gpu(monkeypatch) -> None:
    """visualize/finalize are CPU work; holding an RT-core GPU for them is waste."""
    from npa.orchestration.npa_workflow import build_plan, load_spec
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        build_skypilot_task_doc,
    )

    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://example-bucket/npa-src/test")
    spec = load_spec(SPEC)
    plan = build_plan(spec)

    for name in ("visualize", "finalize"):
        step = next(s for s in plan.steps if s.state == name)
        doc = build_skypilot_task_doc(
            spec, step, run_id="test-run", options=SkypilotRenderOptions()
        )
        assert "accelerators" not in doc["resources"], name
        assert "config" not in doc, f"{name} should not carry GPU pod_config"


def test_renderer_installs_the_nurec_runtime_deps_the_vendor_image_lacks() -> None:
    """The NRE container carries none of the tool's runtime dependencies.

    Live failures walked through them one at a time: `huggingface-cli` missing
    (job 230 fetch), then nvidia-ncore for the rig derivation, rerun-sdk for the
    recording (only an optional `viz` extra of npa), and ffmpeg for
    `nre render --export-video`.
    """
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_setup_for_tool,
    )

    setup = render_setup_for_tool(
        "workbench.nurec.fetch", config={}, options=SkypilotRenderOptions()
    )

    assert "huggingface_hub" in setup
    assert "nvidia-ncore" in setup
    assert "rerun-sdk==" in setup
    assert "ffmpeg" in setup
    # Installed into the interpreter npa itself went into, so a second npa-less
    # python winning on PATH cannot silently break the stage.
    assert "/tmp/npa-python" in setup
    # PEP 668 fallbacks, because the image is Ubuntu 24.04.
    assert "--break-system-packages" in setup
    # And the install is verified rather than assumed.
    assert "import ncore, rerun" in setup


def test_renderer_nurec_rerun_pin_matches_the_packaged_extra() -> None:
    """The setup pin and npa's `viz` extra must not drift apart.

    Read as text rather than parsed: tomllib is 3.11+ and the repo still supports
    3.10 (npa declares tomli only as a <3.11 marker dependency).
    """
    from npa.orchestration.npa_workflow.skypilot_render import NUREC_RERUN_PIN

    pyproject = (REPO_ROOT / "npa" / "pyproject.toml").read_text(encoding="utf-8")
    viz_line = next(
        line for line in pyproject.splitlines() if line.strip().startswith("viz = [")
    )

    assert NUREC_RERUN_PIN in viz_line, f"{NUREC_RERUN_PIN} not in: {viz_line}"


def test_renderer_does_not_add_nurec_deps_to_other_tools() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_setup_for_tool,
    )

    setup = render_setup_for_tool(
        "workbench.vlm_eval.run", config={}, options=SkypilotRenderOptions()
    )

    assert "nvidia-ncore" not in setup


def test_spec_ncore_sequence_uri_matches_what_fetch_publishes() -> None:
    """`reconstruct --ncore-uri` must point exactly where `fetch` published.

    fetch writes to `<ncore_uri>sequence/` (that suffix is hard-coded in
    fetch_cmd). Nothing in the catalog forces the spec's `ncore_sequence_uri` to
    agree, and a mismatch fails SILENTLY in a different pod -- reconstruct
    materializes an empty prefix and reports "no NCore sequence meta-file found".
    """
    config = _spec()["config"]

    assert config["ncore_sequence_uri"] == config["ncore_uri"] + "sequence/"


def test_spec_reconstruct_reads_the_uri_the_fetch_stage_writes() -> None:
    from npa.orchestration.npa_workflow.catalog import TOOL_CATALOG

    fetch = TOOL_CATALOG["workbench.nurec.fetch"].argv_template
    reconstruct = TOOL_CATALOG["workbench.nurec.reconstruct"].argv_template
    config = _spec()["config"]

    published_root = fetch[fetch.index("--output-uri") + 1]
    consumed = reconstruct[reconstruct.index("--ncore-uri") + 1]

    # Resolve both one level through the spec config and compare the real paths.
    resolved_published = config[published_root.strip("{} ").removeprefix("config.")]
    resolved_consumed = config[consumed.strip("{} ").removeprefix("config.")]
    assert resolved_consumed == resolved_published + "sequence/"


def test_staging_the_source_is_inert_for_an_image_that_already_bakes_npa() -> None:
    """Propagating NPA_SRC_S3_URI to every pinned image must not change baked ones.

    The renderer now injects the URI for ANY pinned image (it previously required
    NPA_SRC_OVERLAY=1). That is safe only because the in-pod install is guarded on
    `command -v npa`, so an image that already ships npa skips it entirely. This
    pins that guard -- it is the whole reason the change is non-breaking.
    """
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        default_npa_setup,
        render_setup_for_tool,
    )

    setup = default_npa_setup()
    assert 'if ! command -v npa >/dev/null 2>&1; then' in setup
    # The baked-image path is tried before any S3 sync.
    assert setup.index("/opt/nebius-physical-ai/npa") < setup.index("NPA_SRC_S3_URI")

    # And an unrelated tool's setup is unchanged by the NuRec additions.
    other = render_setup_for_tool(
        "workbench.token_factory.caption", config={}, options=SkypilotRenderOptions()
    )
    assert "nvidia-ncore" not in other


def test_declarative_spec_has_more_than_two_gpu_stages() -> None:
    """Guards the premise of the live declarative e2e: it is genuinely multi-step.

    The live run lives in ``npa/tests/e2e/test_nurec_reconstruct_live_e2e.py`` and
    is gated behind real GPU infra. This assertion is deliberately here instead,
    where it runs in the ordinary suite, so that collapsing the spec into a single
    stage fails immediately rather than silently weakening an e2e nobody runs.
    """
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    states = spec["states"]

    assert len(states) >= 2, states
    gpu_states = [name for name, s in states.items() if s.get("resources") == "gpu"]
    assert len(gpu_states) >= 2, f"expected multiple GPU stages, got {gpu_states}"
    # And the chain is real: each stage after the first declares what it needs.
    assert any(s.get("needs") for s in states.values())


GUIDE = REPO_ROOT / "docs" / "workbench" / "guides" / "neural-reconstruction.md"


def test_guide_exists_and_is_indexed() -> None:
    assert GUIDE.exists()
    index = (GUIDE.parent / "README.md").read_text(encoding="utf-8")
    assert "neural-reconstruction.md" in index, "guide must be listed in the guides index"


def test_guide_only_documents_commands_that_exist() -> None:
    """Every `npa ...` command line in the guide and the spec header must be real.

    A copy-paste guide whose commands 404 is worse than no guide. This caught
    `npa workbench artifacts list-runs`, which does not exist.
    """
    import re

    from typer.main import get_command

    from npa.cli.main import app

    root = get_command(app)

    def resolve(parts: list[str]) -> bool:
        cmd = root
        for part in parts:
            get_sub = getattr(cmd, "get_command", None)
            if get_sub is None:
                return False
            cmd = get_sub(None, part)  # type: ignore[arg-type]
            if cmd is None:
                return False
        return True

    sources = {
        "guide": GUIDE.read_text(encoding="utf-8"),
        "spec header": SPEC.read_text(encoding="utf-8"),
    }
    # Only lines that INVOKE npa (optionally inside a YAML comment), never prose
    # that merely mentions the word.
    invocation = re.compile(r"^\s*(?:#\s*)?npa\s+(.*)$")
    checked = 0
    for label, text in sources.items():
        for line in text.splitlines():
            match = invocation.match(line)
            if not match:
                continue
            parts: list[str] = []
            for token in match.group(1).split():
                if token.startswith("-") or token in {"\\", "|"}:
                    break
                parts.append(token)
            if not parts:
                continue
            assert resolve(parts), f"{label}: `npa {' '.join(parts)}` is not a real command"
            checked += 1
    assert checked > 5, f"expected several npa commands, found {checked}"


def test_guide_references_a_committed_image() -> None:
    """The guide leads with a rendered-output image; it must actually be in-repo."""
    import re

    text = GUIDE.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert refs, "guide should show the rendered output"
    for ref in refs:
        assert (GUIDE.parent / ref).resolve().exists(), ref
