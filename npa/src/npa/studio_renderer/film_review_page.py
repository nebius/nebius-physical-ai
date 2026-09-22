"""Present synchronized narration passages, final frames and review judgments offline."""

import html

_STYLE = """body{margin:0;background:#07121b;color:#edf3ee;font:16px system-ui}
main{max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:36px}
video{width:100%;max-height:65vh;background:#000;position:sticky;top:0;z-index:1}
article{padding:24px 0;border-bottom:1px solid #345}button{background:#deff50;
border:0;border-radius:6px;padding:10px 16px;cursor:pointer;font-weight:700}
.frames{display:flex;overflow:auto;gap:12px}.frames figure{margin:12px 0;min-width:300px}
img{width:300px}figcaption,.scope{color:#b7c6cf;font-size:14px}blockquote{margin:16px 0;font-size:22px}
.verdict{font-weight:700}.mismatch,.uncertain{color:#ffb6a0}.aligned{color:#deff50}
code{overflow-wrap:anywhere}a{color:#deff50}"""

_SCRIPT = """const video=document.querySelector('video');let stop=null;
document.querySelectorAll('[data-start]').forEach(button=>button.addEventListener('click',()=>{
video.currentTime=Number(button.dataset.start);stop=Number(button.dataset.end);
video.play().catch(()=>video.focus());}));
video.addEventListener('timeupdate',()=>{if(stop!==null&&video.currentTime>=stop){video.pause();stop=null;}});
document.querySelector('#whole').addEventListener('click',()=>{stop=null;video.currentTime=0;video.play();});"""


def _cue_html(cue, judgment):
    frames = "".join(
        f'<figure><img src="{html.escape(frame["file"], quote=True)}" '
        f'alt="Final frame at {frame["seconds"]:.3f} seconds">'
        f"<figcaption>{frame['seconds']:.3f}s</figcaption></figure>"
        for frame in cue["frames"]
    )
    verdict = html.escape(judgment["verdict"])
    return (
        f'<article><button data-start="{cue["start"]}" data-end="{cue["end"]}">'
        f"Play {cue['start']:.3f}–{cue['end']:.3f}s with audio</button>"
        f"<blockquote>{html.escape(cue['text'])}</blockquote>"
        f'<p class="verdict {verdict}">{verdict}</p>'
        f"<p>{html.escape(judgment['visible_content'])}</p>"
        f'<p>{html.escape(judgment["reasoning"])}</p><div class="frames">{frames}</div></article>'
    )


def _write_review_page(packet, report, directory):
    judgments = {item["id"]: item for item in report["cues"]}
    rows = "".join(_cue_html(cue, judgments[cue["id"]]) for cue in packet["cues"])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Workbench Studio · Narration and picture review</title><style>{_STYLE}</style></head>
<body><main><h1>Narration and picture review</h1>
<p>Status: <strong>{html.escape(report["status"])}</strong></p>
<p class="scope">{html.escape(packet["scope"])} Timing overlap alone does not establish meaning.</p>
<video src="film.mp4" controls playsinline preload="metadata"></video>
<p><button id="whole">Play the complete film with audio</button> · <a href="assessment.json">Assessment JSON</a></p>
{rows}<p class="scope">Reviewed film SHA-256: <code>{packet["video_sha256"]}</code></p>
</main><script>{_SCRIPT}</script></body></html>"""
    (directory / "review.html").write_text(page, encoding="utf-8")
