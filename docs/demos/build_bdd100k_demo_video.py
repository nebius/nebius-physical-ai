#!/usr/bin/env python3
"""Build the BDD100K exec demo assets: an architecture diagram and a slideshow video.

Outputs (next to the committed FiftyOne screenshots):
- docs/demos/assets/bdd100k/architecture.png
- docs/demos/assets/bdd100k/bdd100k-demo.mp4   (falls back to .gif if no encoder)

Run with the repo venv:
    npa/.venv/bin/python docs/demos/build_bdd100k_demo_video.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent / "assets" / "bdd100k"
ARCH_PNG = ASSETS / "architecture.png"
VIDEO_MP4 = ASSETS / "bdd100k-demo.mp4"
VIDEO_GIF = ASSETS / "bdd100k-demo.gif"

W, H = 1920, 1080
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = "#0f172a"       # slate-900
ACCENT = "#ff6d04"    # FiftyOne / Voxel51 orange
NEBIUS = "#1f6feb"    # blue
GPU = "#16a34a"       # green for H100 stages
STORE = "#7c3aed"     # purple for data/vector store
BG = "#0b1020"        # dark slide background


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


# --------------------------------------------------------------------------- #
# Architecture diagram (matplotlib)
# --------------------------------------------------------------------------- #
def build_architecture() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.55, "BDD100K Failure-Mode Detection",
            ha="center", va="center", fontsize=26, fontweight="bold", color=INK)
    ax.text(8, 8.05, "Nebius Physical AI Workbench  ·  LanceDB + FiftyOne (Voxel51)",
            ha="center", va="center", fontsize=15, color="#475569")

    def band(y, h, color, label):
        ax.add_patch(FancyBboxPatch((0.3, y), 15.4, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     linewidth=0, facecolor=color, alpha=0.10))
        ax.text(0.55, y + h - 0.28, label, ha="left", va="center",
                fontsize=12, fontweight="bold", color=color)

    # Orchestration band (top) and substrate band (bottom).
    band(6.7, 0.95, NEBIUS, "SkyPilot — one YAML orchestrates every stage")
    band(0.45, 0.95, NEBIUS, "Nebius substrate")
    ax.text(8, 0.78, "Object Storage (artifacts)   ·   Managed Kubernetes   ·   GPU clusters (H100)",
            ha="center", va="center", fontsize=12.5, color=NEBIUS)

    def box(cx, cy, text, color, w=2.7, h=1.15):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     linewidth=2, edgecolor=color, facecolor="white"))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=11.5,
                color=INK, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=18,
                     linewidth=2, color="#64748b"))

    # Row A: ingest + enrich.
    yA = 5.35
    box(2.3, yA, "Raw BDD100K\n(Object Storage)", STORE)
    box(5.6, yA, "LanceDB ingest\nper-run table", STORE)
    box(8.9, yA, "CPU UDFs\nperson · rider · bbox · dedup", STORE)
    box(12.7, yA, "CLIP embeddings\nH100", GPU, w=3.0)
    for x0, x1 in [(3.65, 4.25), (6.95, 7.4), (10.4, 11.2)]:
        arrow(x0, yA, x1, yA)

    # Row B: train + review.
    yB = 3.1
    box(2.6, yB, "Failure-mode views\nrider · night · distant", STORE, w=3.0)
    box(6.3, yB, "Detector training ×3\nH100", GPU)
    box(9.6, yB, "Per-view eval\nmAP", GPU)
    box(13.0, yB, "FiftyOne app\npublic :5151", ACCENT, w=3.0)
    for x0, x1 in [(4.1, 4.95), (7.65, 8.45), (10.95, 11.5)]:
        arrow(x0, yB, x1, yB)

    # Wrap arrow from end of Row A down to start of Row B.
    ax.add_patch(FancyArrowPatch((12.7, yA - 0.6), (2.6, yB + 0.6),
                 connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                 mutation_scale=18, linewidth=2, color="#64748b"))

    ax.text(13.0, yB - 0.95, "Execs & engineers review here",
            ha="center", va="center", fontsize=10.5, style="italic", color=ACCENT)

    fig.savefig(ARCH_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {ARCH_PNG}")


# --------------------------------------------------------------------------- #
# Slides (PIL)
# --------------------------------------------------------------------------- #
def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text_slide(title, subtitle, lines=None, accent=ACCENT):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=accent)
    d.text((120, 150), title, font=_font(78, bold=True), fill="white")
    if subtitle:
        d.text((120, 280), subtitle, font=_font(38), fill="#9fb3c8")
    y = 430
    for ln in (lines or []):
        for seg in _wrap(d, ln, _font(40), W - 320):
            d.text((140, y), seg, font=_font(40), fill="#e2e8f0")
            y += 64
        y += 24
    d.text((120, H - 90), "Nebius Physical AI  ·  FiftyOne (Voxel51) on Nebius",
           font=_font(30), fill="#64748b")
    return img


def _image_slide(img_path, banner, takeaway, accent=ACCENT):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 100], fill=accent)
    d.text((60, 26), banner, font=_font(46, bold=True), fill="white")
    shot = Image.open(img_path).convert("RGB")
    max_w, max_h = W - 120, H - 100 - 120
    scale = min(max_w / shot.width, max_h / shot.height)
    new = shot.resize((int(shot.width * scale), int(shot.height * scale)))
    x = (W - new.width) // 2
    y = 100 + (max_h - new.height) // 2 + 10
    img.paste(new, (x, y))
    d.text((60, H - 92), takeaway, font=_font(34), fill="#e2e8f0")
    return img


def build_slides():
    title = _text_slide(
        "BDD100K Failure-Mode Detection",
        "Finding the rare, dangerous cases AV perception models miss",
        ["Riders  ·  pedestrians at night  ·  distant pedestrians",
         "Built on Nebius — reviewed visually in FiftyOne (Voxel51)"],
    )
    problem = _text_slide(
        "The problem",
        "Aggregate accuracy hides safety-critical failures",
        ["Self-driving models can score well overall yet fail on rare scenes.",
         "Those rare scenes — riders, night pedestrians, far-away people —",
         "are exactly the ones that matter most for safety.",
         "We surface, slice, and target them explicitly."],
        accent=NEBIUS,
    )
    arch = _image_slide(ARCH_PNG, "How it works",
                        "One pipeline on Nebius: ingest -> enrich -> embed -> slice -> train -> evaluate -> review in FiftyOne.",
                        accent=NEBIUS)
    # Prefer the live captures from the deployed FiftyOne app; fall back to the
    # archived screenshots if a live capture is missing.
    def shot(live: str, archived: str):
        p = ASSETS / live
        return p if p.exists() else ASSETS / archived

    s1 = _image_slide(shot("live-full.png", "01-full-dataset.png"),
                      "Every frame, searchable",
                      "Real BDD100K dashcam frames, with boxes and AI metadata, in one browsable view.")
    s2 = _image_slide(shot("live-rider.png", "02-rider-view.png"),
                      "Failure mode 1 — riders",
                      "A saved view isolating motorcyclists and cyclists, an underrepresented, high-risk class.")
    s3 = _image_slide(shot("live-nighttime.png", "03-nighttime-view.png"),
                      "Failure mode 2 — pedestrians at night",
                      "Low-light pedestrians, filtered with one SQL rule on the data.")
    s4 = _image_slide(shot("live-distant.png", "04-distant-view.png"),
                      "Failure mode 3 — distant pedestrians",
                      "Small, far-away people the model is most likely to miss.")
    s5 = _image_slide(ASSETS / "05-clip-umap-by-rider.png",
                      "AI-learned similarity map",
                      "CLIP embeddings cluster visually similar scenes — find more rare cases without writing rules.")
    results = _text_slide(
        "Results",
        "A targeted detector per failure mode, scored per view (mAP)",
        ["rider                          mAP 0.354    mAP@50 0.637",
         "nighttime pedestrian   mAP 0.274    mAP@50 0.545",
         "distant pedestrian       mAP 0.397    mAP@50 0.668",
         "",
         "Note: small 3,000-frame run — validates the end-to-end path, not final model quality."],
        accent=GPU,
    )
    close = _text_slide(
        "Reproducible, and live",
        "Anyone can re-run it; reviewers just open a URL",
        ["Whole pipeline is one YAML on Nebius (no bespoke glue).",
         "Validate with no cloud/GPU:  run_bdd100k_pipeline.py --mock-endpoints",
         "Live review: npa workbench fiftyone status  ->  public http://<ip>:5151"],
        accent=ACCENT,
    )
    return [title, problem, arch, s1, s2, s3, s4, s5, results, close]


def build_video(slides, seconds_per_slide=5, fps=30):
    import numpy as np

    frames_per_slide = seconds_per_slide * fps
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(VIDEO_MP4, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=8)
        for slide in slides:
            arr = np.asarray(slide)
            for _ in range(frames_per_slide):
                writer.append_data(arr)
        writer.close()
        print(f"wrote {VIDEO_MP4} ({len(slides)*seconds_per_slide}s)")
        return VIDEO_MP4
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"MP4 encode unavailable ({exc}); writing GIF instead")
        slides[0].save(VIDEO_GIF, save_all=True, append_images=slides[1:],
                       duration=seconds_per_slide * 1000, loop=0)
        print(f"wrote {VIDEO_GIF}")
        return VIDEO_GIF


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_architecture()
    slides = build_slides()
    build_video(slides)


if __name__ == "__main__":
    main()
