"""Guardrail: the raw SkyPilot task catalog only ever shrinks.

`npa.workflow/v0.0.1` specs are becoming the only workflow authoring surface. The
raw SkyPilot task templates under ``npa/src/npa/workflows/skypilot/`` are being
retired one verified port at a time, so this guardrail pins the exact remaining
set. Two properties matter to a reviewer:

* **No re-additions.** A new raw SkyPilot task YAML cannot appear without editing
  this list, which forces the question "why is this not an npa.workflow spec?".
* **A machine-checked tally.** Each retirement PR shows the count going down in a
  single readable diff, instead of a prose claim in a PR body.

Deleting an entry from ``REMAINING`` is the *last* step of a retirement: the twin
spec must already have a live run recorded in ``EVIDENCE.md``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKYPILOT_DIR = REPO_ROOT / "npa" / "src" / "npa" / "workflows" / "skypilot"

#: Raw SkyPilot task templates still shipped, with the reason each one survives.
#: Retirement tally: started at 36.
REMAINING: dict[str, str] = {
    # --- loaded and launched by a shipped runner script ---
    # --- referenced by a CLI/SDK path pointer or shipped data ---
    # --- no npa.workflow twin authored yet ---
    # --- twin exists but is NOT live-verified yet ---
    # --- arrived AFTER this sweep, via #234 ---
    "cosmos3-generate.yaml": (
        "added by #235, also while this retirement was in flight, and also caught here on "
        "merge. Its npa.workflow twin is npa-workflows/cosmos3-generate.yaml; retiring the raw "
        "one needs a live run of that twin, which belongs with #235. See EVIDENCE.md \u00a7R49"
    ),
    "nurec-reconstruct.yaml": (
        "added by #234 while this retirement was in flight, and caught here on merge rather "
        "than in review, which is what this guardrail is for. It is NOT simply a twinned "
        "template: the npa.workflow spec of the same name runs each state in its OWN pod and "
        "hands artifacts over through S3, while this one is a SINGLE-POD task whose stages "
        "share /tmp. The spec has a live-matrix case; whether the single-pod variant should "
        "survive alongside it is #234's call, so it is listed rather than deleted. See "
        "EVIDENCE.md \u00a7R49"
    ),
    # --- RETIRED here: twin live-verified, see EVIDENCE.md -----------------------
    # cosmos3-reason.yaml     job 182            npa-wf-gpu-cosmos3-reason-af7ded35
    # isaac-lab-rl-sweep.yaml jobs 185/186/187   npa-wf-multi-isaac-lab-rl-sweep-c4b86dc5
    # sonic-export.yaml       job 192            npa-wf-gpu-sonic-export-cb60c5ab
    # sonic-eval.yaml         job 198            npa-wf-gpu-sonic-eval-bb3b9c72
    # sonic-export-eval.yaml  job 197            npa-wf-multi-sonic-export-eval-2f5e979e
    #
    # Phase 2a (pointer-only CLI callers repointed first):
    # token-factory-caption.yaml       job 199  npa-wf-cpu-token-factory-caption-1dbebbb4
    # vlm-eval-token-factory.yaml      job 200  npa-wf-cpu-vlm-eval-token-factory-736df0b1
    # token-factory-cosmos-reason.yaml job 201  npa-wf-cpu-token-factory-cosmos-reason-d9669c7f
    # token-factory-generate.yaml      job 202  npa-wf-cpu-token-factory-generate-94815797
    # mjlab-eval.yaml                  job 203  npa-wf-gpu-mjlab-eval-32c1efb5
    #
    # Phase 2b:
    # retargeting.yaml                 job 204  npa-wf-cpu-retargeting-b8e5bc8b
    #   (was FAILING before this change for lack of motion data - EVIDENCE §6.1)
    #
    # Phase 3b:
    # vlm-eval.yaml           job 219  npa-wf-gpu-vlm-eval-single-25906482
    # vlm-eval-benchmark.yaml job 220  npa-wf-gpu-vlm-eval-benchmark-e47bc877
    #   (both needed the renderer to START the vLLM server the spec asks for)
    # sim-to-real-loop.yaml   job 218  npa-wf-gpu-vlm-eval-loop-88da76ad
    #   retired via the NEW `npa workbench vlm-eval loop` capability, not via the staged
    #   engine: nothing else produced task_success_report.json. See EVIDENCE.md §R18-R20.
    #
    # Phase 3b (continued):
    # scenario-gen-adversarial.yaml  job 213  npa-wf-cpu-scenario-gen-smoke-bc5ed74b
    #   twin = scenario-gen-smoke.yaml, which runs the SAME two CLI commands. The template's
    #   Isaac Lab image + 200000 adversary steps selected no different code path: the RL
    #   adversary is a Python-API seam with no CLI flag. See EVIDENCE.md §R25.
    #
    # sim2real-envgen-split.yaml  jobs 223/224  npa-wf-multi-sim2real-envgen-shards-79c2cb1c
    #   twin = sim2real-envgen-shards.yaml. The template read its shard index from a
    #   Kubernetes Job completion index; the spec declares the fan-out as a parallel group.
    #   max_concurrent_observed: 2, and the barrier's split-manifest saw all 64 envs.
    #   See EVIDENCE.md \u00a7R27.
    #
    # cosmos3-ea-fetch.yaml  job 227  npa-wf-cpu-cosmos-fetch-ebbcc897
    #   twin = cosmos-fetch.yaml. check-access reported source_repo reachable / hf_model
    #   reachable, fetch-artifacts reported checkpoint downloaded. Run with public
    #   substitutes for the gated Cosmos3 assets; identical code path. See EVIDENCE.md
    #   \u00a7R28.
    #
    # RELOCATED (not retired) to npa/src/npa/burst/examples/: a single-task input to
    # `npa burst submit-yaml`, not a workflow. Same reasoning as the BYOF profiles
    # (DESIGN.md \u00a7R10). See npa/tests/guardrails/test_burst_examples.py.
    # isaac-lab-cosmos-sdg-burst-smoke.yaml
    #
    # tokenfactory-train-triage.yaml  job 256  npa-wf-multi-tokenfactory-train-triage-6732d78a
    #   train-gpu SUCCEEDED on npa-lerobot:0.6.0-k8s-runtime (real 206 MB checkpoint), then the
    #   CPU triage stage wrote a report from that run's artifacts. Six live iterations, five
    #   engine gaps and one broken vendor image on the way. See EVIDENCE.md \u00a7R32-R33.
    #
    # tokenfactory-rollout-judge.yaml  job 261
    #   npa-wf-multi-tokenfactory-rollout-judge-combo-d4798e41. Twin is the NEW
    #   tokenfactory-rollout-judge-combo.yaml, not the older same-named spec, which is a
    #   different workflow (EVIDENCE.md \u00a7R23). Two real MP4 episodes rendered on GPU, then
    #   scored from CPU by the hosted backend. See EVIDENCE.md \u00a7R34.
    #
    # tokenfactory-scene-to-rollout-judge.yaml  job 262
    #   npa-wf-multi-tokenfactory-scene-to-rollout-judge-c9b64b65, all three stages SUCCEEDED on
    #   the first attempt, and the judge's task literally contained the reasoner's analysis.
    #   See EVIDENCE.md \u00a7R35.
    #
    # sim2real-actions.yaml  job npa-wf-multi-sim2real-envgen-shards-d5c752f1
    #   Absorbed as the envgen spec's fourth stage; its actions-summary.json records
    #   input_train_uri == the split stage's own output. See EVIDENCE.md \u00a7R36.
    #
    # isaac-franka-capture-reason.yaml  job 283
    #   npa-wf-multi-isaac-franka-capture-reason-d8eca4b3: Isaac rendered the Franka and cube on
    #   a GPU, a hosted reasoner planned from those frames on CPU, and the plan describes what
    #   the frames actually show. Four defects on the way. See EVIDENCE.md \u00a7R37.
    #
    # cosmos2-transfer.yaml  job 288
    #   The template held a GPU to print `"status": "contract_ready"`. Its twin ran the real
    #   Cosmos-Transfer2.5 model for 14m23s and published a 3.9 MB augmented clip plus the
    #   manifest that says how it was made. See EVIDENCE.md \u00a7R38.
    #
    # sim-to-real-pipeline.yaml / sim-to-real-trigger.yaml  (no live run, deliberately)
    #   Retired rather than ported: the pipeline stage ran
    #   `python -m npa.workflows.sim_to_real real-loop`, and that module raises a
    #   DeprecationWarning pointing at the staged sim2real engine. A twin would have made the
    #   new surface the home of a legacy path. The watch capability is NOT deprecated and was
    #   ported: npa.workflows.sim_to_real_trigger now submits npa-workflows/sim2real-vlm-rl.yaml.
    #   See EVIDENCE.md \u00a7R40.
    #
    # dataset-ingest-curate.yaml  job 317
    #   npa-wf-cpu-dataset-ingest-curate-754816f0, all five stages SUCCEEDED including
    #   `register`, which read 12 records back out of the in-cluster LanceDB service that
    #   `ingest` had just written. See EVIDENCE.md \u00a7R41.
    #
    # cosmos3-text-to-image-inference.yaml  job 320
    #   npa-wf-gpu-cosmos3-text-to-image-4286b9f4 SUCCEEDED in 5m37s: Cosmos3-Nano generated a
    #   960x960 image from the prompt and published it with its manifest. Eleven jobs and eight
    #   defects from the bash block it replaced. See EVIDENCE.md \u00a7R43.
    #
    # bdd100k-pipeline.yaml  job 326
    #   npa-wf-multi-bdd100k-pipeline-763b2bdf: all ELEVEN stages SUCCEEDED against both
    #   in-cluster services — LanceDB and detection-training — with three real GPU training
    #   runs and three evals. See EVIDENCE.md \u00a7R46.
    #
    # Phase 2c: RELOCATED (not deleted) to npa/src/npa/workflows/byof/profiles/ —
    # they are BYOF resource profiles reached through byof.yaml's toolRef, not
    # workflow templates. See that directory's README.md and
    # npa/tests/guardrails/test_byof_profiles.py.
    # byof-container-smoke-rtxpro.yaml, byof-datagen-rtxpro-smoke.yaml,
    # isaac-lab-rl-train.yaml, isaac-lab-rl-train-rtxpro.yaml,
    # isaac-lab-rl-train-rtxpro-smoke.yaml
}


def test_remaining_skypilot_templates_match_the_pinned_tally() -> None:
    on_disk = {path.name for path in SKYPILOT_DIR.glob("*.yaml")}
    pinned = set(REMAINING)

    added = sorted(on_disk - pinned)
    assert not added, (
        "new raw SkyPilot task YAML(s) appeared in the retiring catalog: "
        f"{added}. Author an npa.workflow/v0.0.1 spec under "
        "npa/workflows/workbench/npa-workflows/ instead; if a raw template is "
        "genuinely required, add it to REMAINING with a reason."
    )
    removed = sorted(pinned - on_disk)
    assert not removed, (
        f"REMAINING lists templates that are already deleted: {removed}. "
        "Drop them from the list in the same change that deletes the files."
    )


def test_every_remaining_template_states_why_it_survives() -> None:
    unexplained = sorted(name for name, reason in REMAINING.items() if not reason.strip())
    assert not unexplained, f"REMAINING entries need a reason: {unexplained}"


#: Workbench CLI modules that advertise a workflow file through a module constant.
#: These are printed by `<tool> workflow` / `<tool> status`, so a retired template
#: silently turns the advertised path into a 404 for the operator who copies it.
CLI_WORKFLOW_PATH_MODULES = (
    "npa.cli.workbench.mjlab",
    "npa.cli.workbench.retargeting",
    "npa.cli.workbench.token_factory",
    "npa.cli.workbench.vlm_eval",
)


def test_cli_advertised_workflow_paths_exist() -> None:
    """Every `*_WORKFLOW_PATH` a CLI prints must be a real file."""

    from importlib import import_module
    from pathlib import Path as _Path

    missing: list[str] = []
    checked = 0
    for module_name in CLI_WORKFLOW_PATH_MODULES:
        module = import_module(module_name)
        for attr in dir(module):
            if not attr.endswith("WORKFLOW_PATH"):
                continue
            value = getattr(module, attr)
            if not isinstance(value, _Path):
                continue
            checked += 1
            if not (REPO_ROOT / value).is_file():
                missing.append(f"{module_name}.{attr} -> {value}")
    assert checked >= 8, f"expected to check several CLI workflow paths, saw {checked}"
    assert not missing, "CLI modules advertise workflow files that do not exist: " + ", ".join(
        missing
    )


#: The skill that lists the reference templates for an operator to start from. Its list
#: went stale the moment Phase 2 deleted six templates, which is exactly the drift a
#: reader would trust and be misled by.
REFERENCE_SKILL = (
    REPO_ROOT / "skills" / "workflows" / "workbench-reference-workflows" / "SKILL.md"
)


def test_reference_skill_lists_exactly_the_remaining_templates() -> None:
    """The skill's "Current Reference YAMLs" section must match the directory."""

    import re

    text = REFERENCE_SKILL.read_text(encoding="utf-8")
    start = text.index("## Current Reference YAMLs")
    section = text[start : text.index("## Retired Templates", start)]
    listed = set(re.findall(r"`([a-z0-9][a-z0-9.-]*\.yaml)`", section))

    assert listed == set(REMAINING), (
        "skills/workflows/workbench-reference-workflows/SKILL.md advertises a different set "
        f"of templates than the catalog holds. Only in the skill: {sorted(listed - set(REMAINING))}. "
        f"Only on disk: {sorted(set(REMAINING) - listed)}."
    )


def test_retirement_tally_is_monotonic() -> None:
    """The catalog started at 36 templates; it may only get smaller."""

    assert len(REMAINING) <= 36, (
        f"the SkyPilot catalog grew to {len(REMAINING)} templates; it is being retired"
    )
