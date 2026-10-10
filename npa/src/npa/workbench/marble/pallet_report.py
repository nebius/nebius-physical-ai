"""Publish the measured manufacturing experiment as headless JSON and HTML."""

import html
import tempfile
from pathlib import Path

from .api import MarbleError
from .pallet_benchmark import compare_metrics
from .runtime import _materialize, _paths, _publish


def _validate(result, run_id):
    if result["run_id"] != run_id or not result.get("gpu", {}).get("name"):
        raise MarbleError("Pallet benchmark lacks matching run/GPU evidence")
    if not result.get("generated_this_run"):
        raise MarbleError("Pallet benchmark requires an API-generated world")
    measured = compare_metrics(**result["arms"])
    if result["delta"] != measured["delta"] or result["outcome"] != measured["outcome"]:
        raise MarbleError(
            "Pallet comparison does not match its measured evaluation metrics"
        )


def _html(result):
    arms = result["arms"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{arms['baseline'][name]:.4f}</td>"
        f"<td>{arms['augmented'][name]:.4f}</td><td>{result['delta'][name]:+.4f}</td></tr>"
        for name in ("mAP", "mAP_50", "mAP_75")
    )
    outcome = (
        "Observed improvement"
        if result["outcome"] == "improved"
        else "No observed improvement"
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manufacturing · Pallet detection</title><style>
body{{margin:0;background:#0b1116;color:#e9f1ef;font:16px system-ui}}main{{max-width:1050px;margin:auto;padding:5vw}}
.tag{{color:#c6fb5f;letter-spacing:2px;font-size:12px}}h1{{font-size:clamp(36px,6vw,70px);letter-spacing:-2px}}
p{{line-height:1.7;color:#a9bac7}}table{{width:100%;border-collapse:collapse;margin:35px 0}}td,th{{padding:16px;text-align:left;border-bottom:1px solid #34444e}}
a{{color:#c6fb5f}}.result{{padding:24px;background:#19252b;border-radius:10px}}img{{width:100%;max-width:640px;border-radius:8px}}</style>
<main><div class="tag">WORLD LABS API → NEBIUS CUDA → REAL-IMAGE EVALUATION</div>
<h1>Does a generated factory<br>help detect real pallets?</h1>
<div class="result"><strong>{outcome}</strong><p>Single-seed comparison on {result["real_test_images"]} held-out real images.
Both arms used {result["optimizer_updates_per_arm"]} optimizer updates.</p></div>
<table><tr><th>Metric (0–1)</th><th>Real-only</th><th>Real + synthetic</th><th>Change</th></tr>{rows}</table>
<p>{result["real_train_images"]} real training images · {result["synthetic_images"]} synthetic images ·
{html.escape(result["gpu"]["name"])}</p>
<h2>Actual training composite</h2><img src="synthetic/0000.png" alt="Real training pallet composited on a Marble background">
<p>{html.escape(result["limitations"])}</p><a href="comparison.json" download>Download measured comparison and provenance</a>
<p>Run {html.escape(result["run_id"])}. Checkpoints and detailed training/evaluation artifacts are under the benchmark's baseline and augmented prefixes.</p></main></html>"""


def pallet_report(request):
    """Build a static experiment report without a browser or hosted inference.

    Args: RunRequest with benchmark and report S3 prefixes.
    Returns: Published artifact names and run identity.
    Raises: MarbleError for invalid provenance, metrics, or artifact hashes.
    """
    _paths(request)
    with tempfile.TemporaryDirectory(prefix="npa-pallet-report-") as directory:
        root = Path(directory)
        result = _materialize(request.input_path, "comparison.json", root)
        _validate(result, request.run_id)
        (root / "index.html").write_text(_html(result))
        _publish(root, request.output_path)
    return {
        "run_id": request.run_id,
        "report": "index.html",
        "outcome": result["outcome"],
    }
