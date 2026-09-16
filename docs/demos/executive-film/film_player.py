"""Build an offline chapter player from the exact rendered storyboard and captions."""

import html
import json
import re

_STYLE = """
:root{color-scheme:dark;--ink:#080d12;--paper:#f5f7f8;--muted:#9fadb7;--lime:#dcff46}
*{box-sizing:border-box}body{margin:0;padding:clamp(20px,4vw,64px);background:var(--ink);
color:var(--paper);font:16px Manrope,Inter,system-ui,sans-serif}main{max-width:1280px;margin:auto}
.brand{display:flex;align-items:center;gap:18px;font-size:11px;letter-spacing:2px;color:var(--muted)}
.brand strong{font-size:20px;letter-spacing:-1px;color:var(--lime)}
h1{max-width:1040px;font-size:clamp(34px,5vw,68px);font-weight:500;letter-spacing:-.045em;
line-height:1.05;margin:34px 0 16px}p{color:var(--muted);line-height:1.6}.intro{max-width:800px}
video{display:block;width:100%;max-height:76vh;aspect-ratio:16/9;background:#000;
border:1px solid #26353e;border-radius:12px;box-shadow:0 24px 100px #0009;margin-top:30px}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
margin:16px 0 28px}.time{font-variant-numeric:tabular-nums;font-size:13px}
.actions{display:flex;gap:18px;align-items:center;flex-wrap:wrap}a{color:var(--lime);
text-underline-offset:5px;font-size:13px}button{font:inherit;cursor:pointer;color:var(--paper);
background:#101922;border:1px solid #26353e;border-radius:8px}button:focus-visible,a:focus-visible,
video:focus-visible{outline:2px solid var(--lime);outline-offset:4px}
button:hover,button[aria-current=true]{border-color:var(--lime);background:#19262d}
button:disabled{opacity:.5;cursor:default}.captions{padding:8px 12px;font-size:13px}
.captions[aria-pressed=true]{color:var(--lime);border-color:var(--lime)}
.section-label{font-size:11px;letter-spacing:2px;color:var(--muted);margin:30px 0 12px}
.chapters{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:10px}
.chapter{display:grid;grid-template-columns:48px 1fr;gap:14px;padding:16px;text-align:left}
.chapter time{color:var(--lime);font-size:12px;font-variant-numeric:tabular-nums;padding-top:3px}
.chapter strong{display:block;font-weight:500;line-height:1.35;font-size:14px}
.chapter small{display:block;color:var(--muted);font-size:11px;line-height:1.5;margin-top:6px}
footer{margin-top:32px;border-top:1px solid #26353e;padding-top:16px;color:var(--muted);font-size:12px}
"""

_SCRIPT = """
const film = JSON.parse(document.getElementById('film-data').textContent);
const video = document.querySelector('video');
const captionButton = document.getElementById('captions');
const chapters = Array.from(document.querySelectorAll('[data-chapter]'));
const track = video.addTextTrack('captions', 'English', 'en');
film.captions.forEach(cue => track.addCue(new VTTCue(cue.start, cue.end, cue.text)));
track.mode = 'hidden';
captionButton.disabled = film.captions.length === 0;
captionButton.addEventListener('click', () => {
  track.mode = track.mode === 'showing' ? 'hidden' : 'showing';
  captionButton.setAttribute('aria-pressed', String(track.mode === 'showing'));
});
video.textTracks.addEventListener('change', () => {
  captionButton.setAttribute('aria-pressed', String(track.mode === 'showing'));
});
chapters.forEach((button, index) => button.addEventListener('click', () => {
  video.currentTime = film.chapters[index].start;
  video.play().catch(() => video.focus());
}));
function clock(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}
function updatePlayback() {
  document.getElementById('playback').textContent = `${clock(video.currentTime)} / ${clock(film.duration)}`;
  let active = 0;
  film.chapters.forEach((chapter, index) => {
    if (video.currentTime >= chapter.start) active = index;
  });
  chapters.forEach((button, index) => {
    if (index === active) button.setAttribute('aria-current', 'true');
    else button.removeAttribute('aria-current');
  });
}
video.addEventListener('timeupdate', updatePlayback);
video.addEventListener('loadedmetadata', updatePlayback);
updatePlayback();
"""


def _clock(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def _seconds(timestamp):
    if re.fullmatch(r"\d{2,}:[0-5]\d:[0-5]\d,\d{3}", timestamp) is None:
        raise ValueError("Player captions contain an invalid SRT timestamp")
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def _caption_cues(source, duration):
    cues = []
    for block in re.split(r"\n\s*\n", source.lstrip("\ufeff").strip()):
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit() or " --> " not in lines[1]:
            raise ValueError("Player captions contain an invalid SRT cue")
        start, end = map(_seconds, lines[1].split(" --> "))
        if not 0 <= start < end <= duration:
            raise ValueError("Player captions exceed the rendered timeline")
        cues.append({"start": start, "end": end,
                     "text": html.escape("\n".join(lines[2:]), quote=False)})
    return cues


def _chapters(storyboard):
    chapters, elapsed = [], 0
    for scene in storyboard["scenes"]:
        if type(scene["duration"]) is not int or scene["duration"] <= 0:
            raise ValueError("Player chapters require positive whole-second durations")
        title = scene["title"]
        chapters.append({"start": elapsed, "duration": scene["duration"],
                         "title": " ".join(title) if isinstance(title, list) else title,
                         "subtitle": scene.get("subtitle", "")})
        elapsed += scene["duration"]
    if not chapters:
        raise ValueError("Player requires at least one rendered chapter")
    return chapters, elapsed


def _chapter_buttons(chapters):
    buttons = []
    for index, chapter in enumerate(chapters):
        title = html.escape(chapter["title"])
        subtitle = html.escape(chapter["subtitle"])
        buttons.append(f'<button class="chapter" data-chapter="{index}" type="button">'
                       f'<time>{_clock(chapter["start"])}</time><span><strong>{title}</strong>'
                       f'<small>{subtitle}</small></span></button>')
    return "\n".join(buttons)


def _inline_json(value):
    encoded = json.dumps(value, ensure_ascii=True)
    for character in "<>&":
        encoded = encoded.replace(character, f"\\u{ord(character):04x}")
    return encoded


def _player_html(storyboard, captions):
    chapters, duration = _chapters(storyboard)
    data = {"chapters": chapters, "duration": duration,
            "captions": _caption_cues(captions.replace("\r\n", "\n"), duration)}
    title = html.escape(storyboard["title"])
    subtitle = html.escape(storyboard.get("subtitle", "Explore the film, one chapter at a time."))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Workbench</title><style>{_STYLE}</style></head>
<body><main><header><div class="brand"><strong>nebius</strong><span>PHYSICAL AI WORKBENCH</span></div>
<h1>{title}</h1><p class="intro">{subtitle}</p></header>
<video controls playsinline preload="metadata" tabindex="0" aria-label="{title}"
poster="poster.png" src="workbench-executive-film.mp4"></video>
<div class="toolbar"><span class="time" id="playback">0:00 / {_clock(duration)}</span>
<div class="actions"><button class="captions" id="captions" type="button" aria-pressed="false">English captions</button>
<a href="workbench-executive-film.mp4" download>Download film</a>
<a href="workbench-executive-film.srt" download>Captions</a></div></div>
<p class="section-label">EXPLORE THE FILM · {len(chapters)} CHAPTERS</p>
<nav class="chapters" aria-label="Film chapters">{_chapter_buttons(chapters)}</nav>
<footer>Nebius Physical AI Workbench</footer></main>
<script id="film-data" type="application/json">{_inline_json(data)}</script>
<script>{_SCRIPT}</script></body></html>
'''


def _write_player(directory):
    storyboard = json.loads((directory / "render-storyboard.json").read_text())
    captions = (directory / "workbench-executive-film.srt").read_text()
    (directory / "watch.html").write_text(_player_html(storyboard, captions), encoding="utf-8")
