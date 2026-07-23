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
    ("input", {"stage": "Source frames", "component": "Uploaded source clips", "runtime": "input"}),
    ("labeled_original", {"stage": "Annotate originals", "component": "Token Factory VLM", "runtime": "hosted GPU (Token Factory)"}),
    ("cosmos_augmented", {"stage": "Augment", "component": "Cosmos Transfer 2.5", "runtime": "GPU (Nebius K8s)"}),
    ("grade", {"stage": "Attribute verify + quality gate", "component": "Token Factory vlm_eval + CPU gate", "runtime": "hosted GPU (Token Factory) + CPU"}),
    ("labeled_augmented", {"stage": "Pseudo-label augmented", "component": "Token Factory VLM", "runtime": "hosted GPU (Token Factory)"}),
    ("curation", {"stage": "Curation", "component": "FiftyOne-style curation report", "runtime": "CPU"}),
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
        entry["artifact_count"] = counts.get(stage, 0)
        if stage == "cosmos_augmented":
            man = _read(_augment_manifest_key(keys, run_id))
            mode = str(man.get("mode") or "")
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
        components.append(entry)

    summary = "; ".join(
        f"{c['stage']} — {c['component']}"
        + (f" ({c['model']})" if c.get("model") else "")
        + f" [{c['runtime']}]"
        for c in components
    )
    return {"run_id": run_id, "components": components, "summary": summary}


def _augment_manifest_key(keys: list[str], run_id: str) -> str:
    for key in keys:
        if _stage_of(key, run_id) == "cosmos_augmented" and key.endswith("/manifest.json"):
            return key
    return ""


def _grade_result_key(keys: list[str], run_id: str) -> str:
    for key in keys:
        if _stage_of(key, run_id) == "grade" and key.endswith(".json") and "decision" not in key:
            return key
    return ""
