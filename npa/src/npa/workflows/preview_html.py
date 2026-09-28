"""Render compact, offline HTML previews from measured workflow output media."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


def image_preview(image: Image.Image, *, width: int = 480) -> str:
    """Encode a display thumbnail, preserving the separate original artifact.

    Args:
        image: Decoded source image.
        width: Maximum preview width in pixels.

    Returns:
        An embedded JPEG data URL, requiring no network requests.

    Raises:
        OSError: The source image cannot be decoded or encoded.
    """
    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((width, width), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_preview(
    path: Path, *, title: str, summary: str, metrics: dict, groups: list
) -> None:
    """Write independent image timelines with inert embedded data and escaped text.

    Args:
        path: Local HTML destination.
        title: Human-readable report title.
        summary: Scope and limitations of the preview.
        metrics: Measured scalar values safe for a portable report.
        groups: Timeline dictionaries containing title, note, and frames.

    Returns:
        None.

    Raises:
        ValueError: A group contains no frames or JSON contains nonfinite values.
        OSError: The destination cannot be written.
    """
    if not groups or any(not group["frames"] for group in groups):
        raise ValueError("preview requires actual media in every timeline")
    encoded = json.dumps(groups, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    values = "".join(
        f"<div><dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd></div>"
        for key, value in metrics.items()
    )
    body = _DOCUMENT.replace("@@TITLE@@", escape(title))
    body = body.replace("@@SUMMARY@@", escape(summary)).replace("@@METRICS@@", values)
    body = body.replace("@@DATA@@", encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


_DOCUMENT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>@@TITLE@@</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#111827;color:#e5e7eb}
body{max-width:1200px;margin:auto;padding:24px}h1{font-size:2rem}p{line-height:1.6;color:#b8c5d8}
dl{display:flex;flex-wrap:wrap;gap:12px}dl div{padding:16px;background:#1f2937;border-radius:10px}
dt{font-size:.8rem;color:#b8c5d8}dd{margin:6px 0 0;font-size:1.3rem;font-weight:bold}
section{margin:28px 0;padding:20px;background:#182235;border-radius:12px}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}
input[type=range]{flex:1;min-width:180px}button{background:#334155;color:white;padding:8px 14px;border:1px solid #64748b;border-radius:6px}
.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:12px}
figure{margin:0}img{width:100%;height:auto;border-radius:6px;object-fit:contain}figcaption{padding:6px;color:#cbd5e1}
canvas{width:100%;max-width:800px;aspect-ratio:2/1;background:#0b1020;touch-action:none;border-radius:6px}
small{color:#cbd5e1}footer{margin:32px 0;font-size:.85rem;color:#94a3b8}
</style></head><body><h1>@@TITLE@@</h1><p>@@SUMMARY@@</p><dl>@@METRICS@@</dl>
<main id="timelines"></main><footer>Compact previews derived from this run's outputs. Full resolution data, calibration, provenance and reports remain in the run artifacts. This file works offline and makes no network requests.</footer>
<script id="preview-data" type="application/json">@@DATA@@</script><script>
const groups = JSON.parse(document.getElementById('preview-data').textContent);

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

function showImages(panel, views) {
  panel.replaceChildren();
  for (const view of views) {
    const figure = element('figure');
    const image = element('img');
    image.src = view.data;
    image.alt = view.label;
    figure.append(image, element('figcaption', view.label));
    panel.append(figure);
  }
}

function addPlayback(button, slider, count, paint) {
  let timer = null;
  button.onclick = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
      button.textContent = 'Play';
      return;
    }
    button.textContent = 'Pause';
    timer = setInterval(() => {
      slider.value = (Number(slider.value) + 1) % count;
      paint();
    }, 250);
  };
}

function addPointRotation(canvas, state, paint) {
  let drag = null;
  canvas.onpointerdown = event => {
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove = event => {
    if (!drag) return;
    state.angle += (event.clientX - drag[0]) * .01;
    state.tilt += (event.clientY - drag[1]) * .01;
    drag = [event.clientX, event.clientY];
    paint();
  };
  canvas.onpointerup = () => { drag = null; };
  canvas.onpointercancel = () => { drag = null; };
}

function timeline(group) {
  const section = element('section');
  section.append(element('h2', group.title), element('p', group.note));
  const controls = element('div');
  controls.className = 'controls';
  const play = element('button', 'Play');
  const slider = element('input');
  const label = element('small');
  Object.assign(slider, {type: 'range', min: 0, max: group.frames.length - 1, value: 0});
  slider.setAttribute('aria-label', group.title + ' frame');
  controls.append(play, slider, label);
  section.append(controls);
  const images = element('div');
  images.className = 'images';
  section.append(images);
  const canvas = element('canvas');
  Object.assign(canvas, {width: 1000, height: 500});
  canvas.setAttribute('aria-label', 'Sampled colored points, drag to rotate');
  const state = {angle: .6, tilt: .7};
  const current = () => group.frames[Number(slider.value)];
  const paintCloud = () => cloud(canvas, current().points || [], state);
  const paint = () => {
    label.textContent = current().label;
    showImages(images, current().images);
    paintCloud();
  };
  if (group.frames.some(frame => frame.points)) {
    section.append(element('p', 'Colored point preview: drag to rotate. Points are evenly sampled for display; the complete fused cloud is retained in the dataset.'), canvas);
  }
  addPointRotation(canvas, state, paintCloud);
  addPlayback(play, slider, group.frames.length, paint);
  slider.oninput = paint;
  paint();
  document.getElementById('timelines').append(section);
}

function rotatedPoint(point, center, state) {
  const x = point[0] - center[0], y = point[1] - center[1], z = point[2] - center[2];
  const a = x * Math.cos(state.angle) - y * Math.sin(state.angle);
  const b = x * Math.sin(state.angle) + y * Math.cos(state.angle);
  return [a, b * Math.sin(state.tilt) - z * Math.cos(state.tilt),
          b * Math.cos(state.tilt) + z * Math.sin(state.tilt), point.slice(3)];
}

function cloud(canvas, points, state) {
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!points.length) return;
  const center = [0, 1, 2].map(axis => points.reduce((sum, p) => sum + p[axis], 0) / points.length);
  const rotated = points.map(point => rotatedPoint(point, center, state));
  const radius = Math.max(...rotated.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1]))), .01);
  const scale = canvas.height * .44 / radius;
  rotated.sort((a, b) => a[2] - b[2]);
  for (const point of rotated) {
    context.fillStyle = 'rgb(' + point[3].join(',') + ')';
    context.fillRect(canvas.width / 2 + point[0] * scale, canvas.height / 2 + point[1] * scale, 3, 3);
  }
}
groups.forEach(timeline);
</script></body></html>
"""
