"""Render paired GPU videos and their measured judgments as a standalone HTML report."""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import html
from pathlib import Path

from npa.workflows.physis_lang_contract import ARMS
from npa.workflows.physis_lang_generate import expected_grid

_ARM_LABELS = {
    "baseline": "Baseline",
    "physics": "Physics reasoning",
    "physics-negative": "Physics + negative guidance",
}
_STYLE = """
:root{color-scheme:dark;font:16px/1.55 system-ui,sans-serif;background:#101720;color:#e8edf4}
*{box-sizing:border-box}body{max-width:1500px;margin:auto;padding:32px}
h1{font-size:clamp(28px,4vw,48px);line-height:1.15;margin:12px 0}h2{margin:0}
h3{font-size:17px;margin:0 0 12px}p{margin:12px 0}a{color:#97caff}
.eyebrow{color:#a4b4c8;text-transform:uppercase;letter-spacing:.12em;font-size:12px}
.intro{max-width:900px;color:#becbdc}.note{border-left:3px solid #eebf73;padding:8px 16px}
table{width:100%;border-collapse:collapse;margin:24px 0}th,td{text-align:left;padding:12px}
th{color:#a4b4c8;font-weight:500}td{border-top:1px solid #354358}
.table-scroll{overflow-x:auto}nav{display:flex;flex-wrap:wrap;gap:10px;margin:28px 0}
nav a{background:#203044;padding:6px 12px;border-radius:8px;text-decoration:none}
section{margin:36px 0;scroll-margin-top:24px}.scenario{max-width:1000px;color:#becbdc}
.seed{color:#a4b4c8;margin-top:24px}.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
article{background:#182330;border:1px solid #354358;border-radius:12px;padding:16px;min-width:0}
video{width:100%;aspect-ratio:16/9;background:#000;border-radius:6px}
.score{font-weight:600;color:#b4dbff}details{margin-top:14px}summary{cursor:pointer;color:#afc9e5}
li{margin:12px 0}ol{padding-left:22px}.verdict{font-size:12px;text-transform:uppercase;font-weight:700}
.pass{color:#96ddb0}.fail{color:#ffacaa}.unknown{color:#f3ca88}
.evidence{font-size:13px;color:#b8c6d7}code{overflow-wrap:anywhere;font-size:12px}
footer{border-top:1px solid #354358;margin-top:32px;padding-top:16px;color:#a4b4c8}
@media(max-width:900px){body{padding:16px}.comparison{grid-template-columns:1fr}}
"""


def _key(row):
    return row["case_id"], row["seed"], row["arm"]


def _overview(recipe, report):
    rows = []
    for arm in ARMS:
        counts = Counter(
            assertion["verdict"]
            for judgment in report["judgments"]
            if judgment["arm"] == arm
            for assertion in judgment["assertions"]
        )
        score = report["summary"]["arm_mean_scores"][arm]
        rows.append(
            f"<tr><td>{_ARM_LABELS[arm]}</td><td>{score:.2%}</td>"
            f"<td>{counts['pass']}</td><td>{counts['fail']}</td><td>{counts['unknown']}</td></tr>"
        )
    links = "".join(
        f'<a href="#{case["id"]}">{html.escape(case["id"].replace("-", " "))}</a>'
        for case in recipe["cases"]
    )
    return (
        '<div class="table-scroll"><table><caption>Blinded physical-assertion judgments</caption>'
        "<thead><tr><th>Prompt arm</th><th>Mean score</th><th>Pass</th><th>Fail</th>"
        "<th>Unknown</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        f'<p class="note">Scores use {recipe["evaluation_frames"]} sampled frames per clip; '
        "unknown assertions do not pass. These are exploratory model judgments, not proof of "
        "continuous physical correctness. Review the complete videos below.</p>"
        f"<nav aria-label=Scenarios>{links}</nav>"
    )


def _assertions(case, judgment):
    items = []
    for assertion in judgment["assertions"]:
        verdict = assertion["verdict"]
        if verdict not in {"pass", "fail", "unknown"}:
            raise ValueError("Invalid report assertion verdict")
        text = html.escape(case["assertions"][assertion["index"]])
        evidence = html.escape(assertion["evidence"])
        items.append(
            f'<li><span class="verdict {verdict}">{verdict}</span> · {text}'
            f'<p class="evidence">{evidence}</p></li>'
        )
    return (
        "<details><summary>Assertion evidence</summary><ol>"
        + "".join(items)
        + "</ol></details>"
    )


def _generation_details(row, digest):
    settings = row["requested"]
    devices = ", ".join(device["name"] for device in row["runtime"]["devices"])
    return (
        "<details><summary>Prompts &amp; native generation receipt</summary>"
        f"<p><b>Positive prompt</b><br>{html.escape(row['prompt'])}</p>"
        f"<p><b>Negative prompt</b><br>{html.escape(row['negative_prompt']) or 'None'}</p>"
        f"<p>{html.escape(devices)}<br>{html.escape(settings['model_id'])}<br>"
        f"{settings['width']} × {settings['height']} · {settings['frames']} frames · "
        f"{settings['fps']} fps · {settings['steps']} denoising steps<br>"
        f"Weights revision: <code>{html.escape(settings['revision'])}</code></p>"
        f"<p>Original MP4 SHA-256<br><code>{digest}</code></p></details>"
    )


def _video_card(output, case, row, video, judgment):
    payload = video.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row["observed"]["sha256"] or digest != judgment["video_sha256"]:
        raise ValueError("Report video differs from its generation or judgment receipt")
    name = f"{row['case_id']}-seed-{row['seed']}-{row['arm']}"
    (output / "videos" / f"{name}.mp4").write_bytes(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    label = html.escape(f"{name}: {_ARM_LABELS[row['arm']]}")
    return (
        f"<article><h3>{_ARM_LABELS[row['arm']]}</h3>"
        f'<video controls playsinline preload="metadata" aria-label="{label}" '
        f'data-sha256="{digest}" src="data:video/mp4;base64,{encoded}"></video>'
        f'<p class="score">Assertion score: {judgment["score"]:.2%}</p>'
        + _assertions(case, judgment)
        + _generation_details(row, digest)
        + "</article>"
    )


def _scenario(output, recipe, case, indexed, judgments):
    comparisons = []
    for seed in recipe["seeds"]:
        cards = []
        for arm in ARMS:
            key = case["id"], seed, arm
            row, video = indexed[key]
            cards.append(_video_card(output, case, row, video, judgments[key]))
        comparisons.append(
            f'<p class="seed">Paired seed {seed}</p><div class="comparison">'
            + "".join(cards)
            + "</div>"
        )
    return (
        f'<section id="{case["id"]}"><h2>{html.escape(case["id"].replace("-", " ").title())}</h2>'
        f'<p class="scenario">{html.escape(case["prompt"])}</p>'
        + "".join(comparisons)
        + "</section>"
    )


def _header(recipe, report, count):
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Physis-inspired GPU comparison</title><style>{_STYLE}</style></head><body>"
        '<header><p class="eyebrow">Workbench · Physical prompting experiment</p>'
        '<h1>From physical descriptions<br>to generated motion</h1><p class="intro">'
        f"{count} original GPU-generated Wan 2.1 14B videos, embedded in this file. "
        "Open it offline and compare all three prompt arms at matched seeds. "
        "Independent NPA implementation inspired by Physis-Lang; not an upstream reproduction.</p></header>"
        + _overview(recipe, report)
        + "<main>"
    )


def write_gallery(output: Path, recipe: dict, rows: list, report: dict) -> None:
    """Embed original MP4 bytes and judgment evidence in a portable offline report.

    Args:
        output: Existing report directory, also receiving original MP4 copies.
        recipe: Validated frozen experiment recipe.
        rows: Validated native generation receipts and their video paths.
        report: Completed paired evaluation with judgments and summary.
    Returns:
        None.
    Raises:
        ValueError: Coverage or video hashes differ from their receipts.
        OSError: Reading videos or writing the report fails.
    """
    expected = expected_grid(recipe)
    indexed = {_key(row): (row, video) for row, video in rows}
    judgments = {_key(row): row for row in report["judgments"]}
    if set(indexed) != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicate report video coverage")
    if set(judgments) != expected or len(report["judgments"]) != len(expected):
        raise ValueError("Incomplete or duplicate report judgment coverage")
    (output / "videos").mkdir(exist_ok=True)
    with (output / "index.html").open("w", encoding="utf-8") as page:
        page.write(_header(recipe, report, len(rows)))
        for case in recipe["cases"]:
            page.write(_scenario(output, recipe, case, indexed, judgments))
        page.write(
            f"</main><footer>Judge: {html.escape(report['judge_model'])}. "
            "Embedded MP4s match their generation and evaluation SHA-256 receipts. "
            "No external assets, server, or network connection required.</footer></body></html>"
        )
