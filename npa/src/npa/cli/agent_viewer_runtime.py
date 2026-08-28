"""Runtime helpers for stage labels and serialized viewer artifact loads.

This source is embedded into the generated agent backend. Keeping the shared
viewer publication transaction here prevents the deploy CLI from regrowing its
monolithic backend template while preserving one auditable critical section.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a recording without materializing the whole artifact in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

# NPA_EMBED_STANDALONE_START
# The rendered backend supplies these globals. Explicit sentinels keep this
# module importable for direct helper tests without hiding undefined names from
# static analysis.
if __name__ == "npa.cli.agent_viewer_runtime":
    (
        DEFAULT_SIM_VIZ,
        GROOT_TRAINING_CAMERA_LABEL,
        LICHTBLICK_RECORDING_HTTP_PATH,
        MCAP_RECORDING_PATH,
        NEURAL_RECONSTRUCTION_CAMERA_LABEL,
        NEURAL_RECONSTRUCTION_PREVIEW_ENTITY,
        NEURAL_RECONSTRUCTION_VIEWER_NOTE,
        RECORDING_PATH,
        RRD_PATH,
        _ARTIFACT_LOAD_LOCK,
        _copy_artifact_preview,
        clear_cross_run_mcap_state,
        _is_data_factory_recording,
        _is_sim2real_pipeline_recording,
        _lichtblick_iframe_url,
        _lichtblick_recording_url,
        _load_state,
        _now_iso,
        _publish_foxglove_recording,
        _publish_mcap_recording,
        _publish_rrd_recording,
        _record_sim_viz_run,
        _rerun_iframe_url,
        _restart_rerun_serve,
        _save_state,
        _sim_viz_load_response,
        _sim2real_pipeline_camera_label,
        _wait_rerun_web_viewer_healthy,
        is_groot_training_recording,
        is_neural_reconstruction_recording,
    ) = (None,) * 30
# NPA_EMBED_STANDALONE_END


def _artifact_stage_key(key: str, run_id: str) -> str:
    """Derive the same run-relative stage key used by the browser."""
    scoped = str(key or "")
    marker = "/" + str(run_id) + "/"
    idx = scoped.find(marker)
    if idx >= 0:
        scoped = scoped[idx + len(marker) :]
    elif run_id and scoped.startswith(str(run_id) + "/"):
        scoped = scoped[len(str(run_id)) + 1 :]
    parts = [part for part in scoped.split("/") if part]
    first = parts[0] if parts else "artifacts"
    if first == "reports":
        return "reports"
    if first == "eval" and len(parts) > 1:
        return "eval/" + parts[1]
    if first in ("actions", "vlm_eval", "training_signal", "envs") and len(parts) > 1:
        return first + "/" + parts[1]
    return first


_STAGE_DESCRIPTIONS = {
    "input": (
        "Input — the run's source clip(s) and frames that the pipeline augments "
        "(the base footage the augment stage conditions on)."
    ),
    "configs": (
        "Config Generation — samples appearance-only scenario combos "
        "(lighting/background/surface/color) into manifest.json. Each combo drives "
        "one Cosmos Transfer inference, so N combos become N scenario variants in "
        "the multiply fan-out."
    ),
    "labeled_original": (
        "Understand & Annotate — Token Factory VLM dense captions of the SOURCE "
        "frames (captions.json). These are descriptive labels, NOT the quality "
        "gate (see the grade stage for the attribute-verify / hallucination check)."
    ),
    "cosmos_augmented": (
        "Augment & Multiply — real Cosmos Transfer 2.5 GPU output: one augmented "
        "variant per sampled scenario, each under aug-<clip>/ with "
        "augmented_video.mp4 + frames + metadata.json (the sampled appearance). "
        "manifest.json records variant_count / multiply_mode / variant_parallelism."
    ),
    "grade": (
        "Evaluate & Validate — the VLM attribute-verification / hallucination check "
        "(vlm_eval_stub.json: score / threshold / model) plus the quality-gate "
        "decision.json (promote_checkpoint vs loop_back). This IS the eval, not a "
        "caption."
    ),
    "labeled_augmented": (
        "Pseudo-Label Augmented — Token Factory VLM captions of the AUGMENTED clips "
        "(captions.json) so the amplified dataset ships fully labeled."
    ),
    "curation": (
        "Curation — a real dataset report over the augmented + graded set "
        "(clip/frame counts, per-clip coverage, multiply mode) for FiftyOne / "
        "Voxel51 review."
    ),
    "reports": (
        "Visualize & Finalize — the embedded Rerun recording (sim2real.rrd) and the "
        "aggregate final report (final.json) summarizing the whole run."
    ),
    "eval/heldout": "Held-out evaluation — simulation eval report for the trained checkpoint.",
    "actions/train": "Policy rollouts — action rollouts collected on the training envs.",
    "vlm_eval/train": "VLM eval — VLM scoring of the training rollouts.",
    "outer_loop": "Decision / outer loop — the promote_checkpoint vs loop_back gate decision.",
}

_STAGE_LABELS = {
    "input": "Input",
    "configs": "Configs",
    "labeled_original": "Labeled original",
    "cosmos_augmented": "Cosmos augmented",
    "grade": "Grade",
    "labeled_augmented": "Labeled augmented",
    "curation": "Curation",
    "reports": "Reports / visualization",
    "eval/heldout": "Held-out eval",
    "actions/train": "Policy rollouts",
    "vlm_eval/train": "VLM eval",
    "outer_loop": "Decision / outer loop",
}


def _stage_label(stage_key: str) -> str:
    key = str(stage_key or "").strip()
    if key in _STAGE_LABELS:
        return _STAGE_LABELS[key]
    cleaned = (
        key.replace("_", " ").replace("/", " / ").replace("-", " ").strip()
        or "Artifacts"
    )
    return cleaned[:1].upper() + cleaned[1:]


def _stage_description(stage_key: str, label: str, count: int) -> str:
    key = str(stage_key or "").strip()
    description = _STAGE_DESCRIPTIONS.get(key)
    if description:
        return description
    return f"{label} — {count} artifact(s) discovered under `{key or 'run'}`."


def _load_session_run_if_known(
    *, body: dict, run_id: str, requested_camera: str = ""
) -> dict | None:
    """Load an unqualified session-owned run without scanning artifact storage."""
    if any(
        body.get(key)
        for key in (
            "run_ref",
            "rrd_uri",
            "prefix",
            "resource_bucket",
            "project_id",
            "resolved_prefix",
            "source_selected",
        )
    ):
        return None
    state = _load_state()
    runs = state.get("sim_viz_runs")
    runs = runs if isinstance(runs, dict) else {}
    selected = runs.get(run_id)
    current = state.get("sim_viz")
    if (
        isinstance(current, dict)
        and str(current.get("run_id") or "").strip() == run_id
        and (
            str(current.get("source_type") or "").strip() == "artifact_storage"
            or any(
                str(current.get(key) or "").strip()
                for key in (
                    "artifact_run_ref",
                    "bucket",
                    "resolved_prefix",
                    "canonical_mcap_s3_uri",
                )
            )
        )
    ):
        # Rerun self-heal also keeps a basename history alias so its local RRD
        # can be recovered. That alias is not a session-owned replacement for
        # an active source-qualified run. Let normal artifact resolution handle
        # the unqualified request so it preserves the run's canonical MCAP.
        return None
    sim2real_runs = state.get("sim2real_runs")
    sim2real_runs = sim2real_runs if isinstance(sim2real_runs, dict) else {}
    if isinstance(selected, dict):
        source_type = str(selected.get("source_type") or "").strip()
        if source_type == "artifact_storage" or any(
            str(selected.get(key) or "").strip()
            for key in ("artifact_run_ref", "bucket", "resolved_prefix")
        ):
            return None
        selected = dict(selected)
    elif run_id in sim2real_runs:
        selected = {"run_id": run_id}
    else:
        return None
    if requested_camera:
        selected["camera"] = requested_camera
    stage = str(body.get("stage") or "").strip()
    if stage:
        selected["stage"] = stage
    mode = str(body.get("mode") or "").strip().lower()
    if mode in {"static", "live"}:
        selected["mode"] = mode
    selected["run_id"] = run_id
    selected["rrd_updated_at"] = _now_iso()
    state["sim_viz"] = selected
    _record_sim_viz_run(state, selected)
    _save_state(state)
    return {
        "ok": True,
        "sim_viz": _sim_viz_load_response(state, selected, run_id=run_id),
    }


def _serialized_artifact_apply(func):
    """Serialize publication and refresh caller state inside the same lock."""

    def wrapped(*args, **kwargs):
        with _ARTIFACT_LOAD_LOCK:
            supplied_state = kwargs.get("state")
            if isinstance(supplied_state, dict):
                current_state = _load_state()
                if current_state is not supplied_state:
                    supplied_state.clear()
                    supplied_state.update(current_state)
            return func(*args, **kwargs)

    return wrapped


@_serialized_artifact_apply
def _apply_loaded_artifact(
    *,
    state: dict,
    run_id: str,
    key: str,
    s3_uri: str,
    render: str,
    local_path: Path,
    source_identity: tuple[str, str, str] = ("", "", ""),
    run_ref: str = "",
    requested_camera: str = "",
    artifact_contract: dict | None = None,
    source_fingerprint: str = "",
    source_size_bytes: int = 0,
    source_last_modified: str = "",
) -> dict:
    now = _now_iso()
    sim_viz = dict(DEFAULT_SIM_VIZ)
    current = state.get("sim_viz")
    if isinstance(current, dict):
        sim_viz.update(current)
    # Never let a previous RRD's binding survive a later media load.
    sim_viz.pop("served_recording_sha256", None)
    sim_viz.pop("served_recording_size_bytes", None)
    clear_cross_run_mcap_state(sim_viz, run_id)
    camera = str(sim_viz.get("camera") or "workspace")
    contract = artifact_contract if isinstance(artifact_contract, dict) else {}
    contract_matches = (
        contract.get("matches") if isinstance(contract.get("matches"), dict) else {}
    )
    learning_paths = {
        str(path)
        for semantic in ("rrd", "mcap")
        for path in contract_matches.get(semantic) or []
    }
    is_learning = contract.get("authoritative") is True and any(
        str(key).endswith("/" + path) or str(key) == path for path in learning_paths
    )
    contract_camera = str(contract.get("primary_camera") or "").strip()
    # Keep the data-factory exclusion on one line: npa/tests/cli/test_agent.py
    # guards that exact expression as source text.
    if (
        render == "rerun"
        and _is_sim2real_pipeline_recording(key)
        and not _is_data_factory_recording(key)
        and not is_neural_reconstruction_recording(key)
    ):
        camera = _sim2real_pipeline_camera_label(camera)
    elif render == "rerun" and is_learning:
        camera = contract_camera
    elif render == "rerun" and is_groot_training_recording(key):
        # A training-telemetry recording must not inherit a previous policy
        # rollout's held-out camera label or preview entity.
        camera = GROOT_TRAINING_CAMERA_LABEL
    elif render == "rerun" and is_neural_reconstruction_recording(key):
        # A NuRec run following Sim2Real must not inherit "heldout-sim".
        camera = NEURAL_RECONSTRUCTION_CAMERA_LABEL
    if requested_camera:
        if _is_sim2real_pipeline_recording(key):
            camera = _sim2real_pipeline_camera_label(requested_camera)
        elif is_learning:
            if requested_camera != contract_camera:
                raise ValueError(
                    "requested camera differs from validated GR00T provenance"
                )
            camera = contract_camera
        elif is_groot_training_recording(key):
            camera = GROOT_TRAINING_CAMERA_LABEL
        else:
            camera = requested_camera
    resource_bucket, project_id, resolved_prefix = source_identity
    sim_viz.update(
        {
            "run_id": run_id,
            "source_type": "artifact_storage",
            "source_label": "S3 artifacts",
            "stage": "artifact-loaded",
            "rrd_updated_at": now,
            "artifact_uri": s3_uri,
            "artifact_key": key,
            "artifact_render": render,
            "artifact_run_ref": str(run_ref or ""),
            "mode": "static",
            "camera": camera,
            "artifact_contract": contract if is_learning else {},
            "artifact_contract_authoritative": bool(is_learning),
            "evaluation_kind": str(contract.get("evaluation_kind") or "")
            if is_learning
            else "",
            "closed_loop": bool(contract.get("closed_loop")) if is_learning else False,
            "bucket": str(resource_bucket or "").strip(),
            "project_id": str(project_id or "").strip(),
            "resolved_prefix": str(resolved_prefix or "").strip(),
            "artifact_source_fingerprint": str(source_fingerprint or "").strip(),
            "artifact_source_size_bytes": max(0, int(source_size_bytes or 0)),
            "artifact_source_last_modified": str(source_last_modified or "").strip(),
        }
    )
    if render == "rerun":
        capability_path = _publish_rrd_recording(local_path)
        # The systemd Rerun service opens RRD_PATH while nginx serves
        # RECORDING_PATH. Keep both atomically on the selected real artifact.
        if RRD_PATH.parent.is_dir():
            rrd_tmp = RRD_PATH.with_suffix(".rrd.tmp")
            shutil.copy2(local_path, rrd_tmp)
            rrd_tmp.replace(RRD_PATH)
        sim_viz["served_recording_sha256"] = _sha256_file(RECORDING_PATH)
        sim_viz["served_recording_size_bytes"] = RECORDING_PATH.stat().st_size
        restarted = _restart_rerun_serve(force=True)
        rerun_ready = _wait_rerun_web_viewer_healthy() if restarted else False
        sim_viz["rrd_uri"] = f"file://{RECORDING_PATH}"
        sim_viz["artifact_preview_url"] = capability_path
        sim_viz["artifact_download_url"] = "/api/sim-viz/rrd-blob"
        sim_viz["rerun_iframe_url"] = _rerun_iframe_url(
            str(sim_viz.get("camera") or "workspace"), recording_path=capability_path
        )
        sim_viz["rerun_ready"] = bool(capability_path) and rerun_ready
        if is_learning:
            sim_viz["preview_entity"] = f"heldout/camera/{camera}"
            sim_viz["visualization_note"] = (
                "Offline held-out GR00T policy evaluation loaded (not a rollout). "
                f"The validated primary camera is {camera}. The Rerun recording aligns "
                "persisted held-out frames with expert and baseline/post-training "
                "predictions, action error, finite training loss, and provenance."
            )
        elif is_groot_training_recording(key):
            sim_viz["preview_entity"] = GROOT_TRAINING_CAMERA_LABEL
            sim_viz["visualization_note"] = (
                "GR00T training telemetry loaded. Entities contain representative "
                "frames decoded from the run's real LeRobot dataset, validated "
                "training metrics, safe logs, and provenance. Frame time is "
                "dataset/synthetic-fps, not robot capture time; this is not a "
                "policy rollout evaluation."
            )
        elif _is_data_factory_recording(key):
            sim_viz["preview_entity"] = "augmented"
            sim_viz["visualization_note"] = (
                "Physical AI Data Factory recording loaded. Entities: input/<clip> "
                "(source frames), augmented/<clip> (Cosmos Transfer 2.5 output; the "
                "static text label shows the sampled appearance variables), and "
                "captions/ (Token Factory VLM pseudo-labels). Scrub the frame "
                "timeline to compare original vs augmented."
            )
        elif is_neural_reconstruction_recording(key):
            sim_viz["preview_entity"] = NEURAL_RECONSTRUCTION_PREVIEW_ENTITY
            sim_viz["visualization_note"] = NEURAL_RECONSTRUCTION_VIEWER_NOTE
        elif _is_sim2real_pipeline_recording(key):
            sim_viz["preview_entity"] = "camera"
            sim_viz["visualization_note"] = (
                "Pipeline Sim2Real recording loaded. The primary Rerun view is the "
                "held-out simulation camera stream; any 3D Franka/world entities are "
                "reference proxy context, not custom hardware footage."
            )
    elif render == "mcap":
        # Publish the original recording for Foxglove-family readers and reserve
        # the fixed Lichtblick slot for genuine MCAP files.
        is_mcap = key.lower().endswith(".mcap")
        published = _publish_foxglove_recording(local_path, key)
        sim_viz["rrd_uri"] = ""
        sim_viz["rerun_iframe_url"] = "/rerun/"
        sim_viz["rerun_ready"] = False
        sim_viz["preview_entity"] = ""
        sim_viz["foxglove_url"] = published
        sim_viz["foxglove_ready"] = bool(published)
        sim_viz["mcap_updated_at"] = now
        if is_mcap:
            _publish_mcap_recording(local_path)
            mcap_url = _lichtblick_recording_url()
            start_time_ns = 0
            end_time_ns = 0
            try:
                from npa.workbench.foxglove.inspect import summarize_mcap

                mcap_info = summarize_mcap(local_path)
                start_time_ns = int(mcap_info.start_time_ns)
                end_time_ns = int(mcap_info.end_time_ns)
            except (ImportError, OSError, RuntimeError, ValueError):
                # Timing is an initial-seek optimization; the validated download
                # and embedded viewer remain available when inspection fails.
                pass
            sim_viz["mcap_uri"] = f"file://{MCAP_RECORDING_PATH}"
            sim_viz["artifact_preview_url"] = LICHTBLICK_RECORDING_HTTP_PATH
            sim_viz["artifact_download_url"] = LICHTBLICK_RECORDING_HTTP_PATH
            sim_viz["lichtblick_iframe_url"] = _lichtblick_iframe_url(
                mcap_url=mcap_url,
                mcap_size=MCAP_RECORDING_PATH.stat().st_size,
                primary_camera=camera if is_learning else "",
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
            )
            sim_viz["lichtblick_ready"] = MCAP_RECORDING_PATH.is_file()
            if is_learning:
                sim_viz["visualization_note"] = (
                    "Offline held-out GR00T policy-evaluation MCAP loaded (not a rollout). "
                    f"The validated primary camera topic is /camera/{camera}; aligned "
                    "expert/model actions, errors, held-out metrics, loss, and provenance "
                    "use explicit dataset-index/optimizer-step time domains."
                )
            elif is_groot_training_recording(key):
                sim_viz["visualization_note"] = (
                    "GR00T training telemetry MCAP loaded in the embedded Lichtblick "
                    "viewer. It contains real dataset frames, safe training logs, and "
                    "factual metrics on dataset/synthetic-fps time; it is not a policy "
                    "rollout or robot-capture recording. The same file is also published "
                    "on a CORS + byte-range path for Foxglove-compatible clients."
                )
            else:
                sim_viz["visualization_note"] = (
                    "MCAP recording loaded: it plays in the embedded Lichtblick "
                    "(Foxglove-compatible, OSS) viewer — rollout camera, VLM critiques and "
                    "reward/advantage signals — and the same file is published on a CORS + "
                    "byte-range path for the official Foxglove app."
                )
        else:
            sim_viz["lichtblick_ready"] = False
            sim_viz["artifact_preview_url"] = published or _copy_artifact_preview(
                local_path, key
            )
            sim_viz["artifact_download_url"] = sim_viz["artifact_preview_url"]
            sim_viz["visualization_note"] = (
                f"Recording loaded ({Path(key).suffix.lower() or 'unknown'}). Foxglove-family "
                "viewers read it directly; the Lichtblick slot is reserved for .mcap."
            )
    else:
        preview_url = _copy_artifact_preview(local_path, key)
        sim_viz["artifact_preview_url"] = preview_url
        sim_viz["artifact_download_url"] = preview_url
        sim_viz["rrd_uri"] = ""
        sim_viz["rerun_iframe_url"] = "/rerun/"
        sim_viz["rerun_ready"] = False
        sim_viz["preview_entity"] = ""
        sim_viz["foxglove_ready"] = False
        sim_viz["foxglove_url"] = ""
        sim_viz["visualization_note"] = (
            f"Loaded {render} artifact preview. Use the Video/Image/Data viewer tabs."
        )
    sim_viz["active_run_id"] = run_id
    state["sim_viz"] = sim_viz
    _record_sim_viz_run(state, sim_viz)
    _save_state(state)
    return sim_viz
