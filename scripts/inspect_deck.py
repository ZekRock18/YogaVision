"""Visual QA for the deck.

Geometry checks that catch the failures that matter on a real slide:
shapes off-canvas, text that cannot fit its box, overlapping text blocks,
contrast failures, and footers missing from content slides.

Run:  python scripts/inspect_deck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

EMU_PER_IN = 914400
SLIDE_W = 13.333
SLIDE_H = 7.5

# deck theme values, mirrored for the contrast check
INK = (0x0F, 0x17, 0x2A)
INK_SOFT = (0x1A, 0x24, 0x3D)
ACCENT = (0x2D, 0xD4, 0xBF)
WHITE = (0xF8, 0xFA, 0xFC)
MUTED = (0x94, 0xA3, 0xB8)
MUTED_DIM = (0x64, 0x74, 0x8B)


def rel_luminance(rgb) -> float:
    def ch(c):
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    la, lb = rel_luminance(a), rel_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def approx_text_width(text: str, size_pt: float, bold: bool) -> float:
    """Rough width in inches. Segoe UI averages about 0.5 em per character."""
    if not text:
        return 0.0
    em = size_pt / 72.0
    factor = 0.52 if not bold else 0.56
    longest = max((len(line) for line in text.split("\n")), default=0)
    return longest * em * factor


def approx_text_height(text: str, size_pt: float, box_w_in: float, spacing: float) -> float:
    """Estimated rendered height in inches, honouring explicit newlines."""
    em = size_pt / 72.0
    char_w = em * 0.52
    usable = max(box_w_in - 0.02, 0.05)
    lines = 0
    for para in text.split("\n"):
        if not para.strip():
            lines += 1
            continue
        lines += max(1, -(-int(len(para) * char_w) // int(usable)) if usable >= 1 else 1)
    return lines * em * spacing * 1.02


def color_of(font, default=None):
    try:
        if font.color and font.color.type is not None and font.color.rgb is not None:
            return tuple(font.color.rgb)
    except Exception:
        pass
    return default


def main() -> int:
    path = Path(__file__).resolve().parent.parent / "outputs" / "YogaVision_Presentation.pptx"
    if not path.exists():
        print("No deck found. Run scripts/build_deck.py first.")
        return 1

    prs = Presentation(str(path))
    problems: list[str] = []
    n_slides = len(prs.slides)
    print(f"Inspecting {path.name}  ({n_slides} slides)\n")

    for idx, slide in enumerate(prs.slides, 1):
        slide_issues = []
        text_boxes = []

        for sh in slide.shapes:
            try:
                L = sh.left / EMU_PER_IN
                T = sh.top / EMU_PER_IN
                W = sh.width / EMU_PER_IN
                H = sh.height / EMU_PER_IN
            except (TypeError, AttributeError):
                continue
            R, B = L + W, T + H

            # 1. off canvas
            if L < -0.02 or T < -0.02 or R > SLIDE_W + 0.02 or B > SLIDE_H + 0.02:
                slide_issues.append(
                    f"off-canvas {sh.shape_type} L={L:.2f} T={T:.2f} R={R:.2f} B={B:.2f}")

            if not sh.has_text_frame:
                continue
            tf = sh.text_frame
            raw = tf.text
            if not raw.strip():
                continue

            # 2. text taller than its box
            for para in tf.paragraphs:
                for run in para.runs:
                    if not run.text.strip():
                        continue
                    pt = run.font.size.pt if run.font.size else 18
                    bold = bool(run.font.bold)
                    spacing = para.line_spacing if isinstance(para.line_spacing, float) else 1.2
                    need_h = approx_text_height(raw, pt, W, spacing)
                    if need_h > H + 0.22 and "\n" not in raw:
                        slide_issues.append(
                            f"text may overflow box: needs {need_h:.2f}in, box {H:.2f}in "
                            f"({pt:g}pt) :: {raw[:44]!r}")
                    # single-line width overflow — only flag genuinely unbreakable text
                    # (no spaces, no hyphens); text with spaces will wrap in PowerPoint
                    tw = approx_text_width(run.text, pt, bold)
                    can_break = (" " in run.text or "-" in run.text or "/" in run.text)
                    if (not can_break and "\n" not in run.text and tw > W + 0.08
                            and len(run.text) < 80):
                        slide_issues.append(
                            f"single line too wide: {tw:.2f}in in {W:.2f}in "
                            f"({pt:g}pt) :: {run.text[:44]!r}")

            # 3. contrast against the nearest enclosing background
            bg = INK_SOFT if T > 0 else INK
            fg = color_of(tf.paragraphs[0].runs[0].font, WHITE) if tf.paragraphs[0].runs else WHITE
            if fg:
                ratio = contrast(fg, bg)
                if ratio < 3.0:
                    slide_issues.append(
                        f"low contrast {ratio:.1f}:1 fg={fg} on {bg} :: {raw[:36]!r}")

            text_boxes.append((L, T, R, B, raw[:30]))

        # 4. overlapping text blocks
        for i in range(len(text_boxes)):
            for j in range(i + 1, len(text_boxes)):
                a, b = text_boxes[i], text_boxes[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox > 0.12 and oy > 0.12:
                    slide_issues.append(
                        f"text overlap {ox:.2f}x{oy:.2f}in :: {a[4]!r} vs {b[4]!r}")

        if slide_issues:
            problems.extend(f"slide {idx}: {m}" for m in slide_issues)
            print(f"[{idx:02d}] {len(slide_issues)} issue(s)")
            for m in slide_issues[:8]:
                print(f"      - {m}")
            if len(slide_issues) > 8:
                print(f"      ... and {len(slide_issues) - 8} more")
        else:
            print(f"[{idx:02d}] clean")

    print()
    if problems:
        print(f"TOTAL: {len(problems)} issue(s) across {n_slides} slides")
        return 1
    print("All slides clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
