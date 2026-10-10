"""Present real campus render viewpoints and measured scene scale in a portable HTML viewer."""

from __future__ import annotations

import base64
from io import BytesIO
import json

from PIL import Image


def _image(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _payload(root, record):
    native = json.loads((root / "native-render.json").read_text())
    statistics = native["scene_statistics"]
    frames = [
        {
            "index": pose["frame"],
            "route": pose["route"],
            "image": _image(root / f"frame-{pose['frame']:03d}.png"),
        }
        for pose in native["cameras"]
    ]
    return {
        "frames": frames,
        "gpu": ", ".join(record["gpu_models"]),
        "backend": record["backend"],
        "resolution": record["resolution"],
        "samples": record["samples"],
        "objects": statistics["mesh_objects"],
        "triangles": statistics["instanced_triangles"],
        "uniqueMeshes": statistics["unique_meshes"],
        "assets": _asset_counts(statistics),
        "gpuPeak": record["telemetry"]["peak_utilization_percent"],
        "duration": record["telemetry"]["elapsed_seconds"],
        "evidence": {
            key: record[key]
            for key in (
                "backend",
                "version",
                "gpu_models",
                "resolution",
                "samples",
                "files",
                "scene_sources_sha256",
                "blender_archive_sha256",
            )
        },
    }


def _asset_counts(statistics):
    fields = (
        "robot",
        "inventory_unit",
        "container",
        "solar_array",
        "carrier",
        "vessel",
    )
    return {key: statistics["assets"].get(key, 0) for key in fields}


def _write(root, record):
    if (
        record["telemetry"]["sample_count"] <= 0
        or record["telemetry"]["peak_utilization_percent"] <= 0
    ):
        raise ValueError(
            "Campus handoff requires observed GPU activity during rendering"
        )
    payload = json.dumps(_payload(root, record), allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    (root / "index.html").write_text(
        _DOCUMENT.replace("@@DATA@@", payload), encoding="utf-8"
    )


_DOCUMENT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Industrial Atlas · GPU-rendered campus</title><style>
:root{color-scheme:dark;font:14px system-ui,-apple-system,sans-serif;background:#080e13;color:#e8eef2;--line:#28373e;--muted:#8ba2ad;--mint:#98f1d0}
*{box-sizing:border-box}body{margin:0}button,input{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible{outline:2px solid var(--mint);outline-offset:4px}
header{height:82px;display:flex;align-items:center;justify-content:space-between;padding:0 36px;border-bottom:1px solid var(--line)}
.brand{font-size:17px;letter-spacing:3px;font-weight:650}.brand span{color:var(--mint)}.eyebrow{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted)}
.badge{border:1px solid #35594d;color:var(--mint);padding:8px 12px;border-radius:4px;font-size:11px;letter-spacing:1px}
main{max-width:1800px;margin:auto;padding:32px 36px}.intro{display:flex;justify-content:space-between;align-items:end;gap:28px;margin-bottom:26px}
h1{font-size:clamp(30px,3.6vw,58px);font-weight:440;letter-spacing:-2px;margin:10px 0 8px}p{line-height:1.65;color:var(--muted);margin:0}.intro p{max-width:720px}.intro aside{max-width:290px;font-size:12px;text-align:right}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px}.sidebar{border:1px solid var(--line);border-radius:9px;background:#0d171e;overflow:hidden;align-self:start}
.side-title{padding:20px 20px 12px}.routes{padding:0 10px}.route{display:block;width:100%;text-align:left;border:0;border-radius:5px;background:transparent;padding:15px 12px;color:#c2d0d7;margin:3px 0}
.route.active{background:#17352f;color:var(--mint)}.route small{display:block;color:var(--muted);font-size:11px;margin:5px 0 0 24px}.route span{color:var(--muted);margin-right:10px;font-size:11px}
.plan{margin:20px 18px;border-top:1px solid var(--line);padding-top:18px}.plan svg{width:100%;height:auto;margin-top:12px}.plan text{fill:#9bb5bf;font:8px system-ui}.plan rect{fill:#20333b;stroke:#4b6772;stroke-width:1}
.plan .roads{fill:none;stroke:#45606a;stroke-width:6}.plan .solar{fill:#193c51}.plan .energy{fill:#29443e}.plan .factory{fill:#524438}.plan .camera{fill:var(--mint);stroke:none}
.scene-scale{padding:0 20px 20px;display:flex;justify-content:space-between}.scene-scale strong{font-size:25px;font-weight:450}.scene-scale small{display:block;font-size:10px;color:var(--muted);margin-top:5px}
.viewport{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#0e1b24}.stage{position:relative;aspect-ratio:16/9;overflow:hidden;background:#1a2931}
.stage img{display:block;width:100%;height:100%;object-fit:contain}.stage:fullscreen{background:#080e13;width:100vw;height:100vh;aspect-ratio:auto}
.overlay{position:absolute;top:18px;left:20px;display:flex;gap:8px;pointer-events:none}.pill{padding:7px 10px;background:#08141ed9;border:1px solid #44616b88;border-radius:3px;font-size:10px;letter-spacing:1px}
.fullscreen{position:absolute;right:16px;bottom:16px;background:#0a1a24cf;color:white;border:1px solid #61808b;border-radius:4px;padding:9px 12px;font-size:11px}
.controls{padding:16px 20px;display:flex;align-items:center;gap:15px;border-top:1px solid var(--line)}.controls button{background:var(--mint);border:0;color:#123228;border-radius:4px;padding:9px 15px;min-width:74px}
input[type=range]{flex:1;min-width:50px;accent-color:var(--mint)}.counter{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;min-width:90px;text-align:right}
.filmstrip{display:flex;gap:8px;padding:0 20px 18px;overflow:auto}.thumb{padding:0;flex:0 0 112px;background:#172832;border:2px solid transparent;border-radius:4px;overflow:hidden}.thumb.active{border-color:var(--mint)}.thumb img{display:block;width:108px;height:61px;object-fit:cover}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:20px 0 0}.metric{background:#101b23;padding:20px}.metric strong{display:block;font-size:clamp(16px,1.8vw,27px);font-weight:450;margin-top:9px;letter-spacing:-.5px}.metric small{display:block;color:var(--muted);font-size:11px;margin-top:6px}
.below{display:grid;grid-template-columns:1.5fr 1fr;gap:36px;margin:34px 0 28px}.below h2{font-size:20px;font-weight:450;margin:0 0 12px}.asset-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:17px}.asset{border-top:1px solid var(--line);padding-top:11px;color:var(--muted);font-size:12px}.asset b{font-size:23px;color:#d8e9ec;display:block;font-weight:450;margin-bottom:5px}
details{border-top:1px solid var(--line);padding:19px 0}summary{cursor:pointer;color:var(--muted);font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;color:#a8c6cf;max-height:400px;overflow:auto}footer{border-top:1px solid var(--line);padding:20px 0;color:var(--muted);font-size:11px;display:flex;justify-content:space-between;gap:20px}
@media(max-width:1000px){main{padding:24px 18px}header{padding:0 18px}.layout{grid-template-columns:200px minmax(0,1fr)}.intro aside{display:none}.metric{padding:13px}.filmstrip{padding-left:12px}.thumb{flex-basis:84px}.thumb img{width:80px;height:45px}}
@media(max-width:720px){header{height:62px}.brand{font-size:13px;letter-spacing:2px}.badge{font-size:9px}.layout{display:block}.sidebar{margin-bottom:15px}.side-title,.plan,.scene-scale{display:none}.routes{display:flex;padding:6px;overflow:auto}.route{min-width:135px;padding:10px;font-size:12px}.route small{display:none}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metrics .metric:last-child{grid-column:span 2}.below{display:block}.below>div{margin-bottom:25px}.controls{padding:12px;gap:10px}.overlay{top:10px;left:10px}.pill{padding:5px;font-size:8px}.counter{min-width:67px;font-size:10px}}
</style></head><body>
<header><div class="brand">INDUSTRIAL <span>ATLAS</span></div><div class="badge">NATIVE GPU RENDER</div></header>
<main><div class="intro"><div><div class="eyebrow">Digital twin infrastructure / authored reference environment</div>
<h1>A campus, in perspective.</h1><p>Explore an 18-hectare industrial environment, from its production floor to its energy systems. Four inspection routes. Real GPU path tracing.</p></div>
<aside>Portable scene geometry and recorded render viewpoints. All imagery is embedded; this viewer works offline.</aside></div>
<div class="layout"><aside class="sidebar"><div class="side-title eyebrow">Inspection routes</div><nav class="routes" aria-label="Camera routes" id="routes"></nav>
<div class="plan"><div class="eyebrow">Authored site plan</div><svg viewBox="0 0 250 180" aria-label="Schematic of the authored campus layout" role="img">
<rect x="1" y="1" width="248" height="178" rx="3"/><path class="roads" d="M8 16H242M8 67H242M8 125H242M8 168H242M14 4V176M119 4V176M234 4V176"/>
<rect class="solar" x="23" y="24" width="42" height="36"/><text x="30" y="44">SOLAR</text><rect x="72" y="24" width="32" height="36"/><text x="74" y="44">OPS</text>
<rect class="energy" x="147" y="24" width="70" height="36"/><text x="163" y="44">ENERGY</text><rect class="factory" x="23" y="76" width="82" height="43"/><text x="40" y="100">ROBOTICS</text>
<rect x="130" y="77" width="78" height="40"/><text x="140" y="100">WAREHOUSE</text><rect x="28" y="134" width="180" height="25"/><text x="90" y="150">FREIGHT</text><circle class="camera" cx="223" cy="164" r="3"/>
</svg></div><div class="scene-scale"><div><strong>500 m</strong><small>SITE WIDTH</small></div><div><strong>360 m</strong><small>SITE DEPTH</small></div></div></aside>
<div><section class="viewport" aria-label="Rendered campus viewer"><div class="stage" id="stage"><img id="hero" alt="GPU-rendered industrial campus"/>
<div class="overlay"><span class="pill" id="route-label">CAMPUS AERIAL</span><span class="pill">RECORDED GPU VIEW</span></div><button class="fullscreen" id="fullscreen">Fullscreen ↗</button></div>
<div class="controls"><button id="play" aria-label="Play recorded camera views">Play</button><input id="scrub" type="range" min="0" value="0" aria-label="Camera viewpoint"/><span class="counter" id="counter"></span></div><div class="filmstrip" id="filmstrip" aria-label="Camera thumbnails"></div></section>
<div class="metrics"><div class="metric"><div class="eyebrow">Render device</div><strong id="gpu"></strong><small id="backend"></small></div>
<div class="metric"><div class="eyebrow">Geometry</div><strong id="objects"></strong><small>mesh objects in the scene</small></div>
<div class="metric"><div class="eyebrow">Resolution</div><strong id="resolution"></strong><small id="quality"></small></div>
<div class="metric"><div class="eyebrow">Camera coverage</div><strong id="coverage"></strong><small>across four inspection routes</small></div>
<div class="metric"><div class="eyebrow">Observed GPU peak</div><strong id="peak"></strong><small>device-wide utilization</small></div></div></div></div>
<div class="below"><div><h2>Built to inspect at several scales.</h2><p>The cutaway robotics and fulfillment halls expose production equipment and storage. The freight terminal, operations tower, solar field, and process yard share a single exported OpenUSD/glTF scene.</p><div class="asset-grid" id="assets"></div></div>
<div><h2>Real renders. Explicit provenance.</h2><p id="provenance"></p><p style="margin-top:12px">This is an authored reference environment. Asset counts describe modeled geometry; they are not live sensor readings, production throughput, or a physics validation.</p></div></div>
<details><summary>Inspect renderer and artifact evidence</summary><pre id="evidence"></pre></details>
<footer><span>Offline camera playback · OpenUSD + glTF exports · Native GPU path tracing</span><span>Recorded views; no live renderer or XR connection.</span></footer></main>
<script id="campus-data" type="application/json">@@DATA@@</script><script>
const data = JSON.parse(document.getElementById('campus-data').textContent);
const byId = id => document.getElementById(id);
const routeNames = [...new Set(data.frames.map(frame => frame.route))];
let route = routeNames[0], frames = [], position = 0, timer = null;
const text = (id, value) => { byId(id).textContent = value; };
function pause() { clearInterval(timer); timer = null; text('play', 'Play'); }
function paint() {
  const frame = frames[position];
  byId('hero').src = frame.image;
  byId('hero').alt = route + ', rendered camera viewpoint ' + (position + 1);
  byId('scrub').value = position;
  text('counter', String(position + 1).padStart(2, '0') + ' / ' + String(frames.length).padStart(2, '0'));
  text('route-label', route.toUpperCase());
  [...byId('filmstrip').children].forEach((button, index) => {
    button.classList.toggle('active', index === position);
    button.setAttribute('aria-current', index === position ? 'true' : 'false');
  });
}
function selectRoute(name) {
  pause(); route = name; position = 0;
  frames = data.frames.filter(frame => frame.route === name);
  byId('scrub').max = frames.length - 1;
  byId('filmstrip').replaceChildren();
  frames.forEach((frame, index) => {
    const button = document.createElement('button'), image = document.createElement('img');
    button.className = 'thumb'; button.setAttribute('aria-label', name + ' viewpoint ' + (index + 1));
    image.src = frame.image; image.alt = ''; button.append(image);
    button.onclick = () => { pause(); position = index; paint(); };
    byId('filmstrip').append(button);
  });
  [...byId('routes').children].forEach(button => {
    button.classList.toggle('active', button.dataset.route === route);
    button.setAttribute('aria-pressed', button.dataset.route === route ? 'true' : 'false');
  });
  paint();
}
routeNames.forEach((name, index) => {
  const button = document.createElement('button'), number = document.createElement('span'), note = document.createElement('small');
  button.className = 'route'; button.dataset.route = name;
  number.textContent = String(index + 1).padStart(2, '0');
  note.textContent = data.frames.filter(frame => frame.route === name).length + ' rendered viewpoints';
  button.append(number, document.createTextNode(name), note);
  button.onclick = () => selectRoute(name); byId('routes').append(button);
});
byId('scrub').oninput = event => { pause(); position = Number(event.target.value); paint(); };
byId('play').onclick = () => {
  if (timer) { pause(); return; }
  text('play', 'Pause'); timer = setInterval(() => { position = (position + 1) % frames.length; paint(); }, 600);
};
byId('fullscreen').onclick = async () => {
  if (document.fullscreenElement) { await document.exitFullscreen(); return; }
  await byId('stage').requestFullscreen();
};
document.addEventListener('keydown', event => {
  if (event.target.tagName === 'INPUT' || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
  pause(); position = (position + (event.key === 'ArrowRight' ? 1 : -1) + frames.length) % frames.length; paint();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
text('gpu', data.gpu.replace('NVIDIA ', '')); text('backend', data.backend);
text('objects', data.objects.toLocaleString()); text('resolution', data.resolution.join(' × '));
text('quality', data.samples + ' path-tracing samples'); text('coverage', data.frames.length + ' viewpoints'); text('peak', data.gpuPeak + '%');
const labels = {robot: 'industrial robots', inventory_unit: 'warehouse inventory units', container: 'freight containers', solar_array: 'solar arrays', carrier: 'autonomous carriers', vessel: 'process vessels'};
Object.entries(labels).forEach(([key, label]) => {
  const cell = document.createElement('div'), value = document.createElement('b');
  cell.className = 'asset'; value.textContent = data.assets[key].toLocaleString();
  cell.append(value, document.createTextNode(label)); byId('assets').append(cell);
});
text('provenance', data.backend + ' rendered these views on ' + data.gpu + ' with CPU rendering disabled. The scene contains ' + data.triangles.toLocaleString() + ' instanced triangles across ' + data.uniqueMeshes.toLocaleString() + ' shared meshes. Native execution took ' + data.duration.toFixed(1) + ' seconds, including scene setup and exports.');
text('evidence', JSON.stringify(data.evidence, null, 2));
selectRoute(route);
</script></body></html>
"""
