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
        _text(
            draw,
            (position[0], position[1] + index * size * 1.45),
            line,
            size,
            color,
            weight,
        )


def _rectangles(scene):
    layout = scene["layout"]
    if layout == "architecture":
        return []
    if layout in {"film", "hero", "cinematic", "immersive"}:
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
    if layout == "application":
        return [(72, 132, 1776, 844)]
    if layout == "split":
        return [(72, 455, 1136, 520), (1232, 455, 616, 520)]
    if layout == "comparison":
        return [(72, 492, 876, 482), (972, 492, 876, 482)]
    if layout == "review":
        return [(72, 455, 652, 440), (748, 455, 652, 440)]
    if layout == "pipeline":
        return [(72 + i * 450, 587, 426, 300) for i in range(4)]
    raise ValueError(f"Unknown scene layout: {layout}")


@lru_cache(maxsize=1)
def _logo():
    with Image.open(_ROOT / "brand/nebius-logo.png") as source:
        return source.convert("RGBA").resize((200, 55), Image.Resampling.LANCZOS)


def _branding(image, index, total, footer):
    draw = ImageDraw.Draw(image)
    draw.line((72, 96, 1848, 96), fill=(48, 59, 61, 255), width=1)
    image.alpha_composite(_logo(), (72, 25))
    _text(draw, (1500, 51), "PHYSICAL AI WORKBENCH", 18, _MUTED, 700)
    _text(draw, (72, 1018), footer, 17, _MUTED, 600)
    _text(draw, (1718, 1018), f"{index + 1:02d} / {total:02d}", 17, _MUTED)


def _base(scene, index, total):
    if scene["layout"] == "film":
        return _film_base(scene)
    if scene["layout"] == "architecture":
        return _architecture_base(scene)
    image = Image.new("RGBA", (_WIDTH, _HEIGHT), _INK)
    draw = ImageDraw.Draw(image)
    for x in range(72, _WIDTH, 150):
        draw.line((x, 0, x, _HEIGHT), fill=(19, 27, 31, 255))
    for x, y, width, height in _rectangles(scene):
        draw.rectangle((x - 1, y - 1, x + width, y + height), fill=(46, 61, 65, 255))
        draw.rectangle((x, y, x + width - 1, y + height - 1), fill=(0, 0, 0, 0))
    if scene["layout"] in {"hero", "cinematic"}:
        for x in range(_WIDTH):
            opacity = round(250 * (1 - x / _WIDTH) ** 1.15)
            draw.line((x, 0, x, _HEIGHT), fill=(8, 13, 18, opacity))
        draw.rectangle((0, 0, 1920, 105), fill=(8, 13, 18, 220))
        draw.rectangle((0, 990, 1920, 1080), fill=(8, 13, 18, 220))
    if scene["layout"] == "immersive":
        for y in range(_HEIGHT):
            opacity = round(245 * max(0, (y - 400) / 680) ** 0.6)
            draw.line((0, y, _WIDTH, y), fill=(8, 13, 18, opacity))
        draw.rectangle((0, 0, 1920, 105), fill=(8, 13, 18, 235))
    _branding(image, index, total, scene.get("footer", "PHYSICAL AI WORKBENCH"))
    return image


def _headlines(draw, scene):
    if scene["layout"] == "architecture":
        _text(draw, (96, 155), " ".join(scene["title"]), 66, _WHITE, 700)
        _text(draw, (96, 252), scene["subtitle"], 28, _MUTED)
        return
    if scene["layout"] == "film":
        _film_headlines(draw, scene)
        return
    if scene["layout"] == "application":
        title = " ".join(scene["title"])
        size = 40
        while draw.textlength(title, font=_font(size, 700)) > 1100:
            size -= 1
        _text(draw, (330, 34), title, size, _WHITE, 700)
        _text(draw, (72, 984), scene["subtitle"], 20, _MUTED)
        return
    if scene["layout"] == "immersive":
        _immersive_headlines(draw, scene)
        return
    feature = scene["layout"] in {"feature", "reason"}
    title_y, size = (256, 73) if feature else (175, 98)
    if scene["layout"] == "screen":
        size = 68
    available = 630 if feature else 1776
    while any(
        draw.textlength(line, font=_font(size, 700)) > available
        for line in scene["title"]
    ):
        size -= 1
    _text(draw, (72, 146 if feature else 129), scene["eyebrow"], 21, _ACCENT, 700)
    for index, line in enumerate(scene["title"]):
        _text(
            draw,
            (72, title_y + index * size * 1.16),
            line,
            size,
            _ACCENT if index else _WHITE,
            700,
        )
    subtitle_y = 469 if feature else 414
    if scene["layout"] == "cameras":
        subtitle_y = 292
    if scene["layout"] == "screen":
        subtitle_y = 252
    _paragraph(draw, (72, subtitle_y), scene["subtitle"], 27, 606 if feature else 1650)


def _immersive_headlines(draw, scene):
    size = 82
    while any(
        draw.textlength(line, font=_font(size, 700)) > 1776 for line in scene["title"]
    ):
        size -= 1
    _text(draw, (72, 648), scene["eyebrow"], 21, _ACCENT, 700)
    for index, line in enumerate(scene["title"]):
        _text(
            draw,
            (72, 697 + index * 95),
            line,
            size,
            _WHITE if index == 0 else _ACCENT,
            700,
        )
    _paragraph(draw, (72, 917), scene["subtitle"], 27, 1650)


def _labels(draw, scene):
    if scene["layout"] == "film":
        return
    for rectangle, label in zip(_rectangles(scene), scene["labels"], strict=True):
        x, y, width, height = rectangle
        if scene["layout"] == "immersive":
            _text(draw, (72, 131), label, 20, _WHITE, 700)
            continue
        if scene["layout"] in {"hero", "cinematic"}:
            _text(draw, (72, 929), label, 20, _WHITE, 700)
            continue
        draw.rectangle(
            (x, y + height - 43, x + width - 1, y + height - 1), fill=(8, 13, 18, 232)
        )
        _text(draw, (x + 18, y + height - 30), label, 16, _WHITE, 700)


def _film_base(scene):
    image = Image.new("RGBA", (_WIDTH, _HEIGHT))
    draw = ImageDraw.Draw(image)
    if scene["title"] or scene["subtitle"]:
        for y in range(540, _HEIGHT):
            opacity = round(180 * ((y - 540) / 540) ** 1.8)
            draw.line((0, y, _WIDTH, y), fill=(4, 8, 12, opacity))
    if scene.get("brand", False):
        image.alpha_composite(_logo(), (96, 64))
    _film_source_label(draw, scene["labels"][0])
    return image


def _film_source_label(draw, label):
    if not label:
        return
    size = 26
    while draw.textlength(label, font=_font(size, 600)) > 1100:
        size -= 1
    width = draw.textlength(label, font=_font(size, 600))
    left = _WIDTH - 96 - width - 40
    draw.rounded_rectangle(
        (left, 64, _WIDTH - 96, 120), radius=12, fill=(4, 8, 12, 205)
    )
    _text(draw, (left + 20, 78), label, size, _WHITE, 600)


def _film_headlines(draw, scene):
    size = 76
    while any(
        draw.textlength(line, font=_font(size, 600)) > 1728 for line in scene["title"]
    ):
        size -= 1
    centered = scene.get("title_position", "bottom-left") == "center"
    spacing = round(size * 1.2)
    title_y = 470 if centered else 914 - spacing * len(scene["title"])
    for index, line in enumerate(scene["title"]):
        x = (
            (1920 - draw.textlength(line, font=_font(size, 600))) / 2
            if centered
            else 96
        )
        _text(draw, (x, title_y + index * spacing), line, size, _WHITE, 600)
    subtitle = scene["subtitle"]
    x = (1920 - draw.textlength(subtitle, font=_font(26, 500))) / 2 if centered else 100
    _text(draw, (x, title_y + len(scene["title"]) * spacing + 14), subtitle, 26, _WHITE)


def _architecture_card(draw, node, x, width):
    draw.rounded_rectangle(
        (x, 442, x + width, 700),
        radius=20,
        fill=(18, 29, 35, 255),
        outline=(58, 77, 83, 255),
        width=2,
    )
    _text(draw, (x + 30, 473), node["title"], 40, _WHITE, 700)
    _text(draw, (x + 30, 535), node["purpose"], 25, _ACCENT, 600)
    for index, model in enumerate(node["models"]):
        _text(draw, (x + 30, 582 + index * 33), model, 23, _MUTED)


def _architecture_base(scene):
    image = Image.new("RGBA", (_WIDTH, _HEIGHT), _INK)
    image.alpha_composite(_logo(), (96, 60))
    draw = ImageDraw.Draw(image)
    _text(draw, (96, 345), scene["controller"], 29, _WHITE, 600)
    nodes = scene["compute_nodes"]
    width = (1728 - 36 * (len(nodes) - 1)) // len(nodes)
    for index, node in enumerate(nodes):
        x = 96 + index * (width + 36)
        center = x + width // 2
        draw.line((center, 407, center, 442), fill=_ACCENT, width=3)
        draw.line((center, 700, center, 763), fill=_ACCENT, width=3)
        draw.polygon(
            ((center - 7, 752), (center + 7, 752), (center, 763)), fill=_ACCENT
        )
        _architecture_card(draw, node, x, width)
    draw.line((96, 407, 1824, 407), fill=(75, 98, 104, 255), width=2)
    draw.rounded_rectangle(
        (96, 775, 1824, 907),
        radius=20,
        fill=(25, 39, 41, 255),
        outline=_ACCENT,
        width=2,
    )
    _text(draw, (130, 805), scene["storage_node"]["title"], 35, _WHITE, 700)
    _text(draw, (580, 819), scene["storage_node"]["detail"], 24, _MUTED)
    _text(draw, (96, 958), scene["architecture_note"], 20, _MUTED)
    return image


def _details(draw, scene):
    layout = scene["layout"]
    if layout == "reason":
        _text(draw, (72, 646), scene.get("quote_heading", ""), 18, _ACCENT, 700)
        _paragraph(draw, (72, 693), scene.get("quote", ""), 33, 585, _WHITE)
        for index, note in enumerate(scene.get("source_notes", [])):
            _text(draw, (760, 919 + index * 34), note, 20 if index == 0 else 16, _MUTED)
    if layout == "evidence":
        _text(draw, (1050, 1018), scene.get("evidence_note", ""), 17, _MUTED)
    if layout == "review":
        for index, label in enumerate(scene.get("review_steps", [])):
            y = 490 + index * 142
            _text(draw, (1442, y), f"0{index + 1}", 20, _ACCENT)
            _text(draw, (1442, y + 39), label, 34, _WHITE, 700)
    if layout == "pipeline":
        for index, label in enumerate(scene.get("pipeline_labels", [])):
            _text(
                draw,
                ([72, 420, 1030][index], 949),
                label,
                21,
                _ACCENT if index == 0 else _WHITE,
                700,
            )
    if layout == "close":
        _text(draw, (72, 953), scene.get("cta", ""), 24, _WHITE, 600)


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
