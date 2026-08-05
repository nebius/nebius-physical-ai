"""Run provenance for the Physical AI Data Factory blueprint.

Given a run's real S3 artifact keys (and, optionally, a JSON reader), derive WHICH
pipeline stages produced the data and WHICH components/models made each — so the
agent's "Describe this" can explain where a visual comes from in the workflow and
what produced it, grounded in the artifacts that actually exist (not a guess).

Pure and dependency-light so it is unit-testable and safe to embed in the agent
backend. ``read_json(key) -> dict | None`` is optional; when provided it enriches
entries with the ACTUAL engine/model recorded in the artifacts (e.g. real Cosmos
Transfer 2.5 on GPU vs a CPU appearance stand-in).
"""

from __future__ import annotations

from typing import Any, Callable

# Canonical stage order + the component that produces each stage's artifacts.
# runtime is where the compute actually runs (grounds the "which needs a GPU" story).
_STAGE_COMPONENTS: list[tuple[str, dict[str, str]]] = [
    ("configs", {"stage": "Config generation", "component": "Appearance-variable sampler", "runtime": "CPU"}),
    ("input", {"stage": "Source + conditioning", "component": "Verified run input and derived conditioning", "runtime": "operator-side prepare"}),
    ("labeled_original", {"stage": "Annotate originals", "component": "Token Factory VLM", "runtime": "hosted GPU (Token Factory)"}),
    ("cosmos_augmented", {"stage": "Augment", "component": "Cosmos Transfer 2.5", "runtime": "GPU (Nebius K8s)"}),
    ("grade", {"stage": "Attribute verify + quality gate", "component": "Token Factory vlm_eval + CPU gate", "runtime": "hosted GPU (Token Factory) + CPU"}),
    ("labeled_augmented", {"stage": "Pseudo-label augmented", "component": "Token Factory VLM", "runtime": "hosted GPU (Token Factory)"}),
    ("curation", {"stage": "Curation", "component": "Dataset curation report", "runtime": "CPU"}),
    ("reports", {"stage": "Visualize + finalize", "component": "Rerun recording + aggregate report", "runtime": "CPU"}),
]


def _stage_of(key: str, run_id: str) -> str:
    scoped = str(key or "")
    marker = "/" + str(run_id or "") + "/"
    idx = scoped.find(marker)
    if idx >= 0:
        scoped = scoped[idx + len(marker):]
    elif run_id and scoped.startswith(str(run_id) + "/"):
        scoped = scoped[len(str(run_id)) + 1:]
    parts = [p for p in scoped.split("/") if p]
    return parts[0] if parts else ""


def build_run_provenance(
    keys: list[str],
    *,
    run_id: str = "",
    read_json: Callable[[str], dict | None] | None = None,
) -> dict[str, Any]:
    """Return {run_id, components:[...], summary} describing where the run's data
    came from and which components produced it."""
    present: set[str] = set()
    counts: dict[str, int] = {}
    stage_key_sample: dict[str, str] = {}
    for key in keys:
        stage = _stage_of(key, run_id)
        if not stage:
            continue
        present.add(stage)
        counts[stage] = counts.get(stage, 0) + 1
        stage_key_sample.setdefault(stage, key)

    def _read(key: str) -> dict:
        if not read_json:
            return {}
        try:
            data = read_json(key)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - provenance is best-effort context
            return {}

    components: list[dict[str, Any]] = []
    for stage, base in _STAGE_COMPONENTS:
        if stage not in present:
            continue
        entry = dict(base)
        entry["stage_key"] = stage
        entry["artifact_count"] = counts.get(stage, 0)
        if stage == "cosmos_augmented":
            mode = _augment_mode(keys, run_id, _read)
            if mode == "cosmos_transfer2.5_gpu":
                entry["engine"] = "cosmos_transfer_2.5_gpu"
                entry["detail"] = "real Cosmos Transfer 2.5 diffusion on GPU"
                entry["model"] = "nvidia/Cosmos-Transfer2.5-2B"
            elif mode:
                entry["engine"] = mode
                entry["detail"] = "CPU appearance-transform stand-in (GPU Cosmos Transfer is the heavy variant)"
            else:
                entry["model"] = "nvidia/Cosmos-Transfer2.5-2B"
        elif stage in {"labeled_original", "labeled_augmented"}:
            cap = _read(stage_key_sample.get(stage, "").rsplit("/", 1)[0] + "/captions.json")
            model = str(cap.get("model") or "")
            if model:
                entry["model"] = model
        elif stage == "grade":
            ev = _read(_grade_result_key(keys, run_id))
            model = str(ev.get("model") or "")
            if model:
                entry["model"] = model
        elif stage == "curation":
            rep = _read(_stage_json_key(keys, run_id, "curation", "report.json"))
            engine = str(rep.get("curation_engine") or "")
            if engine == "fiftyone-brain":
                fo = rep.get("fiftyone") if isinstance(rep.get("fiftyone"), dict) else {}
                entry["component"] = "Real FiftyOne Brain curation"
                entry["engine"] = "fiftyone_brain"
                entry["runtime"] = "CPU (npa-fiftyone image)"
                kept = rep.get("curated_kept")
                dropped = rep.get("curated_dropped")
                bits = ["uniqueness + near-duplicate detection + PCA visualization (real FiftyOne Brain)"]
                if kept is not None:
                    bits.append(f"{kept} kept" + (f", {dropped} dropped" if dropped else ""))
                entry["detail"] = "; ".join(bits)
                version = str(fo.get("fiftyone_version") or "") if isinstance(fo, dict) else ""
                if version:
                    entry["model"] = f"fiftyone {version}"
            elif engine == "report-only":
                entry["engine"] = "report_only"
                entry["detail"] = (
                    "artifact counts only — FiftyOne Brain did not run for this "
                    "historical or fallback result"
                )
        components.append(entry)

    summary = "; ".join(
        f"{c['stage']} — {c['component']}"
        + (f" ({c['model']})" if c.get("model") else "")
        + f" [{c['runtime']}]"
        for c in components
    )
    origin = build_run_origin(keys, run_id=run_id, read_json=read_json)
    return {
        "run_id": run_id,
        "components": components,
        "summary": summary,
        "origin": origin,
    }


# Stage whose artifacts may contain source input, and visual-producing stages in
# pipeline order. Derived conditioning and fixture frames are classified below.
_ORIGINAL_INPUT_STAGES: tuple[str, ...] = ("input",)
_VISUAL_STAGE_ORDER: tuple[str, ...] = ("input", "labeled_original", "cosmos_augmented")
_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".webm", ".avi", ".mkv")


def _artifact_kind(key: str) -> str:
    low = str(key or "").lower()
    if low.endswith(_IMAGE_EXTS):
        return "image"
    if low.endswith(_VIDEO_EXTS):
        return "video"
    if low.endswith(".json"):
        return "metadata"
    return "file"


def _keys_for_stage(keys: list[str], run_id: str, stage: str) -> list[str]:
    return sorted(k for k in keys if _stage_of(k, run_id) == stage)


def _visual_keys_for_stage(keys: list[str], run_id: str, stage: str) -> list[str]:
    return [k for k in _keys_for_stage(keys, run_id, stage) if _artifact_kind(k) in {"image", "video"}]


def _stage_json_key(keys: list[str], run_id: str, stage: str, suffix: str) -> str:
    for key in _keys_for_stage(keys, run_id, stage):
        if key.endswith(suffix):
            return key
    return ""


def build_run_origin(
    keys: list[str],
    *,
    run_id: str = "",
    read_json: Callable[[str], dict | None] | None = None,
) -> dict[str, Any]:
    """Resolve WHERE a run's original input came from, grounded in real artifacts.

    Answers "what was the original input image/frame?" without guessing:

    - If the run stored source frames/clips under ``input/``, those keys are source
      inputs; normalized conditioning artifacts are reported separately.
    - Otherwise it reports that no original RGB input was persisted, names the
      earliest stored visuals (e.g. the Cosmos Transfer 2.5 *augmented* frames,
      which are augment OUTPUTS), the augment engine/model, the config-driven
      appearance variables, and what the VLM actually labeled (captions
      ``input_path``) — so the agent states the truth instead of hedging.

    Returns ``{run_id, original_present, original_inputs, earliest_visual,
    augment, config_variables, labeled_from, label_model, summary}``.
    """

    def _read(key: str) -> dict:
        if not read_json or not key:
            return {}
        try:
            data = read_json(key)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - origin is best-effort context
            return {}

    provenance_key = _stage_json_key(keys, run_id, "input", "provenance.json")
    input_source = _read(provenance_key)
    config_variables: dict[str, Any] | None = None
    cfg_key = _stage_json_key(keys, run_id, "configs", "manifest.json")
    cfg = _read(cfg_key)
    variables = cfg.get("variables")
    if isinstance(variables, dict) and variables:
        config_variables = variables
    source = cfg.get("input_source")
    if not input_source and isinstance(source, dict):
        input_source = dict(source)
    elif not input_source and int(cfg.get("seeded_default_input_frames") or 0) > 0:
        input_source = {
            "source_kind": "synthetic_fixture",
            "input_origin_label": "Synthetic seeded fixture",
            "frame_count": int(cfg.get("seeded_default_input_frames") or 0),
        }
    source_kind = str(input_source.get("source_kind") or input_source.get("kind") or "")
    source_kind = {
        "npa_seeded_fixture": "synthetic_fixture",
        "operator_provided": "user_supplied",
        "stored_run_input": "user_supplied",
    }.get(source_kind, source_kind)

    original_inputs: list[dict[str, str]] = []
    derived_conditioning: list[dict[str, str]] = []
    for stage in _ORIGINAL_INPUT_STAGES:
        for key in _visual_keys_for_stage(keys, run_id, stage):
            leaf = key.rsplit("/", 1)[-1].lower()
            if leaf == "conditioning.mp4" or leaf.startswith("conditioning-frame-"):
                derived_conditioning.append(
                    {"key": key, "stage": stage, "kind": _artifact_kind(key)}
                )
            elif source_kind != "synthetic_fixture":
                original_inputs.append(
                    {"key": key, "stage": stage, "kind": _artifact_kind(key)}
                )

    # Augment engine/model (real GPU Cosmos Transfer vs CPU stand-in).
    augment: dict[str, str] | None = None
    if any(_stage_of(k, run_id) == "cosmos_augmented" for k in keys):
        mode = _augment_mode(keys, run_id, _read)
        if mode == "cosmos_transfer2.5_gpu":
            augment = {
                "engine": "cosmos_transfer_2.5_gpu",
                "model": "nvidia/Cosmos-Transfer2.5-2B",
                "detail": "real Cosmos Transfer 2.5 diffusion on GPU",
            }
        elif mode:
            augment = {"engine": mode, "model": "", "detail": "CPU appearance-transform stand-in"}
        else:
            augment = {"engine": "", "model": "nvidia/Cosmos-Transfer2.5-2B", "detail": ""}

    # What the VLM actually pseudo-labeled (captions input_path) — proves whether
    # there was a separate original set or the augmented frames were labeled.
    labeled_from = ""
    label_model = ""
    cap_key = _stage_json_key(keys, run_id, "labeled_augmented", "captions.json")
    if cap_key:
        cap = _read(cap_key)
        labeled_from = str(cap.get("input_path") or "")
        label_model = str(cap.get("model") or "")

    # Config-driven appearance variables (grounds "generated from config, not an image").
    # Earliest stored visual stage (for the no-original fallback).
    earliest_visual: dict[str, Any] | None = None
    present_components = {stage: base for stage, base in _STAGE_COMPONENTS}
    for stage in _VISUAL_STAGE_ORDER:
        vkeys = _visual_keys_for_stage(keys, run_id, stage)
        if vkeys:
            base = present_components.get(stage, {})
            earliest_visual = {
                "stage": base.get("stage", stage),
                "component": base.get("component", ""),
                "runtime": base.get("runtime", ""),
                "keys": vkeys[:8],
                "count": len(vkeys),
            }
            break

    original_present = bool(original_inputs)
    summary = _origin_summary(
        run_id=run_id,
        original_present=original_present,
        original_inputs=original_inputs,
        earliest_visual=earliest_visual,
        augment=augment,
        config_variables=config_variables,
        labeled_from=labeled_from,
        label_model=label_model,
        input_source=input_source,
        derived_conditioning=derived_conditioning,
    )
    return {
        "run_id": run_id,
        "original_present": original_present,
        "original_inputs": original_inputs,
        "derived_conditioning": derived_conditioning,
        "earliest_visual": earliest_visual,
        "augment": augment,
        "config_variables": config_variables,
        "labeled_from": labeled_from,
        "label_model": label_model,
        "input_source": input_source,
        "summary": summary,
    }


def _origin_summary(
    *,
    run_id: str,
    original_present: bool,
    original_inputs: list[dict[str, str]],
    earliest_visual: dict[str, Any] | None,
    augment: dict[str, str] | None,
    config_variables: dict[str, Any] | None,
    labeled_from: str,
    label_model: str,
    input_source: dict[str, Any],
    derived_conditioning: list[dict[str, str]],
) -> str:
    run_tag = f" `{run_id}`" if run_id else ""
    source_kind = str(input_source.get("source_kind") or input_source.get("kind") or "")
    source_kind = {
        "npa_seeded_fixture": "synthetic_fixture",
        "operator_provided": "user_supplied",
        "stored_run_input": "user_supplied",
    }.get(source_kind, source_kind)
    if original_present:
        first = original_inputs[0]["key"]
        n = len(original_inputs)
        more = f" (+{n - 1} more)" if n > 1 else ""
        aug_bit = ""
        if augment:
            model = augment.get("model") or "Cosmos Transfer 2.5"
            aug_bit = f" and then transformed by Cosmos Transfer 2.5 ({model}) in the Augment stage"
        if source_kind == "upstream_sample":
            source_phrase = "Upstream real sample"
        elif source_kind == "user_supplied":
            source_phrase = "User-supplied input (operator-provided input data)"
        else:
            source_phrase = "stored source input"
        source_uri = str(
            input_source.get("staged_canonical_s3_uri") or input_source.get("uri") or ""
        )
        uri_bit = f" from `{source_uri}`" if source_uri else ""
        conditioning_bit = (
            f" {len(derived_conditioning)} Derived conditioning clip/frame artifact(s) "
            "were produced from that source."
            if derived_conditioning
            else ""
        )
        return (
            f"Source input for run{run_tag}: {n} {source_phrase} "
            f"{'frame' if n == 1 else 'frames'}/clip(s) — e.g. `{first}`{more}. "
            f"Source identity{uri_bit or ': recorded in the run config manifest'}. "
            f"{conditioning_bit} The derived frames were annotated by the Token Factory VLM "
            f"(Annotate input){aug_bit}."
        )
    # No original persisted → describe the earliest stored (augmented) visuals truthfully.
    if source_kind == "synthetic_fixture":
        return (
            f"Run{run_tag} uses a Synthetic seeded fixture: repository-generated geometric "
            "frames, not original real-world data. Cosmos consumes the Derived conditioning "
            "clip assembled from those frames."
        )
    if not earliest_visual:
        return (
            f"No original input image is recorded for run{run_tag}, and no stored "
            "visual artifacts were found to trace it back to."
        )
    ev = earliest_visual
    first_key = ev["keys"][0] if ev.get("keys") else ""
    count = ev.get("count", 0)
    parts = [
        f"No separate original input image was stored for run{run_tag} — the pipeline "
        "did not persist a Source-frames / Annotate-originals stage (no `input/` or "
        "`labeled_original/` artifacts exist).",
    ]
    if ev.get("component"):
        parts.append(
            f"The earliest stored visuals are the {ev['component']} outputs at the "
            f"{ev['stage']} stage: `{first_key}` (+{max(count - 1, 0)} more). Those are "
            "augment OUTPUTS, not the original."
        )
    if augment and augment.get("engine") == "cosmos_transfer_2.5_gpu":
        origin_bits = "Cosmos Transfer 2.5's control example"
        if config_variables:
            var_names = ", ".join(sorted(config_variables.keys()))
            origin_bits = f"the config sampler's appearance variables ({var_names}) and " + origin_bits
        parts.append(
            f"They were produced on GPU by Cosmos Transfer 2.5 ({augment.get('model')}) from "
            f"{origin_bits}; no user-uploaded source clip is recorded for this run."
        )
    if labeled_from:
        model_bit = f" ({label_model})" if label_model else ""
        parts.append(
            f"The Token Factory VLM{model_bit} then pseudo-labeled those augmented "
            f"frames (input_path = {labeled_from})."
        )
    return " ".join(parts)


def _augment_manifest_key(keys: list[str], run_id: str) -> str:
    for key in keys:
        if _stage_of(key, run_id) == "cosmos_augmented" and key.endswith("/manifest.json"):
            return key
    return ""


def _augment_mode(keys: list[str], run_id: str, read: Callable[[str], dict]) -> str:
    """Resolve the augment engine mode, grounded in the run's real artifacts.

    Prefers the run-level ``cosmos_augmented/manifest.json`` ``mode`` (written by
    the GPU publish path). Some runs only record the mode in a per-clip
    ``cosmos_augmented/<clip>/metadata.json`` (e.g. CPU stand-in outputs with no
    run-level manifest), so fall back to the first per-clip metadata that carries
    a mode — otherwise the augment engine would look unknown and the honesty
    banner could not distinguish a stand-in from a real Cosmos render.
    """
    man_key = _augment_manifest_key(keys, run_id)
    if man_key:
        mode = str((read(man_key) or {}).get("mode") or "")
        if mode:
            return mode
    for key in keys:
        if _stage_of(key, run_id) == "cosmos_augmented" and key.endswith("/metadata.json"):
            mode = str((read(key) or {}).get("mode") or "")
            if mode:
                return mode
    return ""


def _grade_result_key(keys: list[str], run_id: str) -> str:
    for key in keys:
        if _stage_of(key, run_id) == "grade" and key.endswith(".json") and "decision" not in key:
            return key
    return ""
