"""Draw the film's typography and editorial graphics around verified media."""

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).parent
_WIDTH, _HEIGHT = 1920, 1080
_INK = (8, 13, 18, 255)
_WHITE = (245, 247, 243, 255)
_MUTED = (164, 178, 181, 255)
_ACCENT = (220, 255, 70, 255)


@lru_cache(maxsize=40)
def _font(size, weight=500):
    font = ImageFont.truetype(str(_ROOT / "fonts/Manrope.ttf"), size)
    font.set_variation_by_axes([weight])
    return font


def _wrap(draw, text, size, width, weight=500):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=_font(size, weight)) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def _text(draw, position, text, size, color=_WHITE, weight=500):
    draw.text(position, text, font=_font(size, weight), fill=color, anchor="lt")


def _paragraph(draw, position, text, size, width, color=_MUTED, weight=500):
    for index, line in enumerate(_wrap(draw, text, size, width, weight)):
        _text(draw, (position[0], position[1] + index * size * 1.45),
              line, size, color, weight)


def _rectangles(scene):
    layout = scene["layout"]
    if layout in {"hero", "cinematic"}:
        return [(0, 0, 1920, 1080)]
    if layout in {"close", "triptych"}:
        return [(72 + i * 600, 492, 576, 438) for i in range(3)]
    if layout in {"feature", "reason"}:
        return [(760, 206, 1088, 688)]
    if layout == "evidence":
        return [(72, 487, 1776, 486)]
    if layout == "cameras":
        return [(160, 348, 1600, 632)]
    if layout == "screen":
        return [(72, 294, 1776, 680)]
    if layout == "split":
        return [(72, 455, 1136, 520), (1232, 455, 616, 520)]
    if layout == "comparison":
        return [(72, 492, 876, 482), (972, 492, 876, 482)]
    if layout == "review":
        return [(72, 455, 652, 440), (748, 455, 652, 440)]
    if layout == "pipeline":
        return [(72 + i * 450, 587, 426, 300) for i in range(4)]
    raise ValueError(f"Unknown scene layout: {layout}")


def _branding(draw, index, total):
    draw.line((72, 96, 1848, 96), fill=(48, 59, 61, 255), width=1)
    _text(draw, (72, 36), "nebius", 39, _WHITE, 800)
    _text(draw, (1500, 51), "PHYSICAL AI WORKBENCH", 18, _MUTED, 700)
    _text(draw, (72, 1018), "INTELLIGENCE THAT MOVES", 17, _MUTED, 600)
    _text(draw, (1718, 1018), f"{index + 1:02d} / {total:02d}", 17, _MUTED)


def _base(scene, index, total):
    image = Image.new("RGBA", (_WIDTH, _HEIGHT), _INK)
    draw = ImageDraw.Draw(image)
    for x in range(72, _WIDTH, 150):
        draw.line((x, 0, x, _HEIGHT), fill=(19, 27, 31, 255))
    for x, y, width, height in _rectangles(scene):
        draw.rectangle((x - 1, y - 1, x + width, y + height),
                       fill=(46, 61, 65, 255))
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill=(0, 0, 0, 0))
    if scene["layout"] in {"hero", "cinematic"}:
        for x in range(_WIDTH):
            opacity = round(250 * (1 - x / _WIDTH) ** 1.15)
            draw.line((x, 0, x, _HEIGHT), fill=(8, 13, 18, opacity))
        draw.rectangle((0, 0, 1920, 105), fill=(8, 13, 18, 220))
        draw.rectangle((0, 990, 1920, 1080), fill=(8, 13, 18, 220))
    _branding(draw, index, total)
    return image


def _headlines(draw, scene):
    feature = scene["layout"] in {"feature", "reason"}
    title_y, size = (256, 73) if feature else (175, 98)
    if scene["layout"] == "screen":
        size = 68
    available = 630 if feature else 1776
    while any(draw.textlength(line, font=_font(size, 700)) > available for line in scene["title"]):
        size -= 1
    _text(draw, (72, 146 if feature else 129), scene["eyebrow"], 21, _ACCENT, 700)
    for index, line in enumerate(scene["title"]):
        _text(draw, (72, title_y + index * size * 1.16), line, size,
              _ACCENT if index else _WHITE, 700)
    subtitle_y = 469 if feature else 414
    if scene["layout"] == "cameras":
        subtitle_y = 292
    if scene["layout"] == "screen":
        subtitle_y = 252
    _paragraph(draw, (72, subtitle_y), scene["subtitle"], 27,
               606 if feature else 1650)


def _labels(draw, scene):
    for rectangle, label in zip(_rectangles(scene), scene["labels"], strict=True):
        x, y, width, height = rectangle
        if scene["layout"] in {"hero", "cinematic"}:
            _text(draw, (72, 929), label, 20, _WHITE, 700)
            continue
        draw.rectangle((x, y + height - 43, x + width - 1, y + height - 1),
                       fill=(8, 13, 18, 232))
        _text(draw, (x + 18, y + height - 30), label, 16, _WHITE, 700)


def _details(draw, scene):
    layout = scene["layout"]
    if layout == "reason":
        _text(draw, (72, 646), "MODEL-GENERATED RATIONALE", 18, _ACCENT, 700)
        _paragraph(draw, (72, 693), scene["quote"], 33, 585, _WHITE)
    if layout == "reason":
        _text(draw, (760, 919), "Research inference · saved outputs", 20, _MUTED)
        _text(draw, (760, 953), "Camera imagery: NVIDIA PhysicalAI-AV", 16, _MUTED)
    if layout == "evidence":
        _text(draw, (1050, 1018), "Research inference · saved outputs", 17, _MUTED)
    if layout == "review":
        for index, label in enumerate(["EVALUATE", "INSPECT", "CURATE"]):
            y = 490 + index * 142
            _text(draw, (1442, y), f"0{index + 1}", 20, _ACCENT)
            _text(draw, (1442, y + 39), label, 34, _WHITE, 700)
    if layout == "pipeline":
        _text(draw, (72, 949), "GPU COMPUTE", 21, _ACCENT, 700)
        _text(draw, (420, 949), "SKYPILOT WORKFLOWS", 21, _WHITE, 700)
        _text(draw, (1030, 949), "TRACEABLE ARTIFACTS", 21, _WHITE, 700)
    if layout == "close":
        _text(draw, (72, 953), "nebius.com/solutions/physical-ai-and-robotics", 24, _WHITE, 600)


def _draw_overlay(scene, index, total, path):
    image = _base(scene, index, total)
    draw = ImageDraw.Draw(image)
    _labels(draw, scene)
    _details(draw, scene)
    image.save(path)


def _draw_titles(scene, path):
    image = Image.new("RGBA", (_WIDTH, _HEIGHT))
    _headlines(ImageDraw.Draw(image), scene)
    image.save(path)
