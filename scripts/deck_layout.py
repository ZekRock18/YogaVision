"""Layout primitives for the YogaVision deck.

Every visual is composed from these. Nothing draws a shape directly, so
alignment, spacing, and colour stay consistent across all 17 slides.
"""
from __future__ import annotations

from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from pptx.util import Length

from deck_theme import (
    ACCENT, ACCENT_DIM, BODY_Y, CONTENT_W, CENTER, FONT, FONT_MONO, INK,
    INK_LINE, INK_SOFT, LEFT, MARGIN, MUTED, MUTED_DIM, RIGHT, SLIDE_H, SLIDE_W,
    T_BODY, T_HERO, T_KICKER, T_LEAD, T_NUMBER, T_NUMBER_MD, T_NUMBER_SM,
    T_SMALL, T_TINY, T_TITLE, WHITE, emu,
)


def add_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = INK
    return slide


# ── text ────────────────────────────────────────────────────────────────────
def _line_h(size_pt: float, spacing: float) -> float:
    """Rendered line height in inches for a given point size and line spacing."""
    return size_pt / 72.0 * spacing


def _wrap_lines(content: str, box_w: float, size_pt: float) -> int:
    """Estimate wrapped line count for `content` inside `box_w` inches."""
    if not content:
        return 0
    char_w = size_pt / 72.0 * 0.52
    usable = max(box_w - 0.04, 0.2)
    total = 0
    for para in content.split("\n"):
        if not para.strip():
            total += 1
            continue
        total += max(1, -(-int(len(para) * char_w) // int(usable)))
    return total


def text(slide, body, x, y, w, h, size=T_BODY, color=WHITE, bold=False,
         align=LEFT, font=FONT, italic=False, spacing=1.24, anchor="top",
         space_after=0, fit=False):
    """Place text. `body` may contain newlines; each becomes its own paragraph.

    With `fit=True` the box height shrinks to the estimated rendered height,
    so adjacent boxes never touch. Use it for stacked list rows.
    """
    if fit:
        pt = size.pt if isinstance(size, Length) else size
        lines = _wrap_lines(body if isinstance(body, str) else "\n".join(body), w, pt)
        h = _line_h(pt, spacing) * max(lines, 1) + 0.02

    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE,
                          "bottom": MSO_ANCHOR.BOTTOM}[anchor]

    lines = body.split("\n") if isinstance(body, str) else list(body)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if space_after:
            p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = line
        f = run.font
        f.name = font
        f.size = size if isinstance(size, Length) else Pt(size)
        f.bold = bold
        f.italic = italic
        f.color.rgb = color
    return box


def rich(slide, parts, x, y, w, h, size=T_BODY, align=LEFT, spacing=1.24):
    """One paragraph with mixed formatting. `parts` is a list of (text, opts)."""
    box = slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    for content, opts in parts:
        run = p.add_run()
        run.text = content
        f = run.font
        f.name = opts.get("font", FONT)
        _sz = opts.get("size", size)
        f.size = _sz if isinstance(_sz, Length) else Pt(_sz)
        f.bold = opts.get("bold", False)
        f.italic = opts.get("italic", False)
        f.color.rgb = opts.get("color", WHITE)
    return box


# ── shapes ──────────────────────────────────────────────────────────────────
def panel(slide, x, y, w, h, fill=INK_SOFT, line=None, line_w=0.75, radius=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        emu(x), emu(y), emu(w), emu(h),
    )
    if radius:
        shape.adjustments[0] = radius
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    if shape.has_text_frame:
        shape.text_frame.text = ""
    return shape


def rule(slide, x, y, w, color=ACCENT, thickness=0.035):
    """Thin horizontal rule, the deck's main structural device."""
    return panel(slide, x, y, w, thickness, fill=color, line=None)


def vrule(slide, x, y, h, color=INK_LINE, thickness=0.02):
    return panel(slide, x, y, thickness, h, fill=color, line=None)


def bar(slide, x, y, w, h, color=ACCENT):
    """Accent bar used to mark list rows and chart columns."""
    return panel(slide, x, y, w, h, fill=color, line=None)


def arrow(slide, x, y, w, h, color=ACCENT_DIM):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, emu(x), emu(y), emu(w), emu(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def chip(slide, label, x, y, w, h=0.30, fill=None, line=ACCENT, color=ACCENT,
         size=T_TINY, bold=True):
    panel(slide, x, y, w, h, fill=fill, line=line, line_w=0.75)
    text(slide, label, x, y + 0.045, w, h - 0.06, size=size, color=color,
         bold=bold, align=CENTER)
    return None


# ── page furniture ──────────────────────────────────────────────────────────
def slide_head(slide, title, kicker=None, lead=None, title_size=T_TITLE):
    """Standard content-slide header: kicker, title, accent rule, optional lead."""
    y = 0.52
    if kicker:
        text(slide, kicker.upper(), MARGIN, y, CONTENT_W, 0.26, size=T_KICKER,
             color=ACCENT, bold=True, spacing=1.0)
        y += 0.34
    text(slide, title, MARGIN, y, CONTENT_W, 0.62, size=title_size, color=WHITE,
         bold=True, spacing=1.05)
    y += 0.66 if title_size >= 24 else 0.56
    rule(slide, MARGIN, y, 1.05)
    if lead:
        text(slide, lead, MARGIN, y + 0.20, CONTENT_W * 0.82, 0.5, size=T_LEAD,
             color=MUTED, spacing=1.3)
        return y + 0.85
    return y + 0.34


def footer(slide, page_no, total, section="YogaVision"):
    rule(slide, MARGIN, 6.98, CONTENT_W, color=INK_LINE, thickness=0.012)
    text(slide, f"{section}   |   Yoga Pose Classification with CNN and MediaPipe",
         MARGIN, 7.08, CONTENT_W * 0.7, 0.24, size=T_TINY, color=MUTED_DIM)
    text(slide, f"{page_no:02d} / {total:02d}", 13.333 - MARGIN - 1.2, 7.08, 1.2, 0.24,
         size=T_TINY, color=MUTED_DIM, align=RIGHT, font=FONT_MONO)


def source(slide, note, y=6.60, x=MARGIN, w=None):
    text(slide, note, x, y, w or CONTENT_W, 0.3, size=T_TINY, color=MUTED_DIM,
         italic=True, spacing=1.2)


# ── composite components ────────────────────────────────────────────────────
def stat(slide, value, label, x, y, w, note=None, color=ACCENT,
         size=T_NUMBER, rule_w=0.62):
    """One large number with a label underneath. The deck's core data display.

    `size` can be T_NUMBER (34pt), T_NUMBER_MD (26pt), or T_NUMBER_SM (22pt).
    All three boxes are sized to the actual rendered text height, not to the
    declared box height, so stacked stats pack tightly.
    """
    pt = size.pt if isinstance(size, Length) else size
    v_h = _line_h(pt, 1.0) + 0.04
    l_h = _line_h(T_SMALL.pt, 1.2) + 0.04
    n_h = _line_h(T_TINY.pt, 1.2) + 0.04 if note else 0

    vy = y + 0.14
    ly = vy + v_h + 0.04
    ny = ly + l_h + 0.04 if note else None

    text(slide, value, x, vy, w, v_h, size=size, color=color, bold=True,
         spacing=1.0, font=FONT, fit=True)
    text(slide, label, x, ly, w, l_h, size=T_SMALL, color=MUTED, spacing=1.2,
         fit=True)
    if note:
        text(slide, note, x, ny, w, n_h, size=T_TINY, color=MUTED_DIM, spacing=1.2,
             fit=True)
    return (ny + n_h) if note else (ly + l_h)


def stat_row(slide, items, x=MARGIN, y=BODY_Y, w=None, cols=4, gap=0.3):
    """A row of `stat` blocks that share one baseline. Returns the bottom edge."""
    w = w or CONTENT_W
    col_w = (w - gap * (cols - 1)) / cols
    size = T_NUMBER_SM if cols > 3 else T_NUMBER
    bottom = y
    for i, item in enumerate(items):
        cx = x + i * (col_w + gap)
        b = stat(slide, item[0], item[1], cx, y, col_w,
                 note=item[2] if len(item) > 2 else None,
                 color=item[3] if len(item) > 3 else ACCENT,
                 size=size)
        bottom = max(bottom, b)
    return bottom


def bullets(slide, items, x, y, w, size=T_BODY, color=WHITE, gap=0.42,
            marker_color=ACCENT, marker_w=0.16, indent=0.34, spacing=1.26,
            bold_lead=False):
    """Bulleted list drawn with accent rules, not glyph characters.

    `gap` is the minimum vertical pitch between rows. Each row's box is sized
    to its own wrapped content, so a two-line bullet does not push its
    neighbours and a one-line bullet does not leave a gap.
    """
    pt = size.pt if isinstance(size, Length) else size
    line_h = _line_h(pt, spacing)
    content_w = w - indent
    cursor = y
    for item in items:
        iy = cursor
        bar(slide, x, iy + line_h * 0.42, marker_w, 0.045, color=marker_color)
        if isinstance(item, tuple):
            lead, rest = item
            rich(slide, [(lead, {"bold": True, "color": color, "size": size}),
                         ("  " + rest, {"color": MUTED, "size": size})],
                 x + indent, iy, content_w, gap, spacing=spacing)
            content = lead + "  " + rest
        else:
            text(slide, item, x + indent, iy, content_w, gap, size=size,
                 color=color, spacing=spacing, bold=bold_lead, fit=True)
            content = item
        rows = _wrap_lines(content, content_w, pt)
        cursor = iy + max(line_h * rows, gap)
    return cursor


def kv_table(slide, rows, x, y, w, key_w=None, row_h=0.46, size=T_SMALL,
             key_color=MUTED, value_color=WHITE, zebra=True, align_value=RIGHT):
    """Two-column key/value table with hairline row separators."""
    key_w = key_w or w * 0.52
    for i, (k, v) in enumerate(rows):
        ry = y + i * row_h
        if zebra and i % 2 == 0:
            panel(slide, x, ry, w, row_h - 0.03, fill=INK_SOFT, line=None)
        text(slide, k, x + 0.12, ry + 0.06, key_w - 0.16, row_h - 0.14,
             size=size, color=key_color)
        text(slide, v, x + key_w, ry + 0.06, w - key_w - 0.12, row_h - 0.14,
             size=size, color=value_color, bold=True, align=align_value)
        if i < len(rows) - 1:
            panel(slide, x, ry + row_h - 0.03, w, 0.008, fill=INK_LINE, line=None)
    return y + len(rows) * row_h


def pipeline(slide, stages, x, y, w, h=1.5, arrow_w=0.34, gap=0.12):
    """Horizontal stage diagram with equal-width panels and arrows between."""
    n = len(stages)
    panel_w = (w - (n - 1) * (gap + arrow_w)) / n
    for i, (title, lines, color) in enumerate(stages):
        sx = x + i * (panel_w + gap + arrow_w)
        panel(slide, sx, y, panel_w, h, fill=INK_SOFT, line=INK_LINE)
        bar(slide, sx, y, panel_w, 0.055, color=color)
        text(slide, title, sx + 0.14, y + 0.20, panel_w - 0.28, 0.34, size=T_SMALL,
             color=color, bold=True, spacing=1.1)
        text(slide, lines, sx + 0.14, y + 0.62, panel_w - 0.28, h - 0.72,
             size=T_TINY, color=MUTED, spacing=1.3)
        if i < n - 1:
            arrow(slide, sx + panel_w + gap, y + h / 2 - 0.11, arrow_w, 0.22,
                  color=ACCENT_DIM)


def layer_table(slide, layers, x, y, w, row_h=0.335, name_w=1.9):
    """Neural-network layer list with a coloured left gutter per layer type."""
    row_size = Pt(T_TINY.pt + 0.5)
    for i, (name, detail, color) in enumerate(layers):
        ry = y + i * row_h
        panel(slide, x, ry, w, row_h - 0.025, fill=INK_SOFT if i % 2 == 0 else None,
              line=None)
        bar(slide, x, ry, 0.055, row_h - 0.025, color=color)
        text(slide, name, x + 0.16, ry + 0.05, name_w - 0.2, row_h - 0.12,
             size=row_size, color=color, bold=True, spacing=1.1)
        text(slide, detail, x + name_w, ry + 0.05, w - name_w - 0.14, row_h - 0.12,
             size=row_size, color=MUTED, spacing=1.1)
    return y + len(layers) * row_h


def callout(slide, title, body, x, y, w, h, color=ACCENT, title_size=T_BODY,
            body_size=T_SMALL, bar_w=0.055):
    panel(slide, x, y, w, h, fill=INK_SOFT, line=INK_LINE)
    bar(slide, x, y, bar_w, h, color=color)
    text(slide, title, x + 0.26, y + 0.14, w - 0.45, 0.32, size=title_size,
         color=color, bold=True)
    text(slide, body, x + 0.26, y + 0.52, w - 0.45, h - 0.66, size=body_size,
         color=MUTED, spacing=1.32)


def section_divider(slide, number, title, subtitle):
    """Full-bleed break slide used sparingly to separate major sections."""
    panel(slide, 0, 0, 13.333, 7.5, fill=INK)
    bar(slide, MARGIN, 2.6, 1.4, 0.06, color=ACCENT)
    text(slide, number, MARGIN, 2.85, 2.0, 0.6, size=T_KICKER, color=ACCENT,
         bold=True, spacing=1.0, font=FONT_MONO)
    text(slide, title, MARGIN, 3.15, CONTENT_W * 0.8, 0.9, size=T_HERO, color=WHITE,
         bold=True, spacing=1.05)
    text(slide, subtitle, MARGIN, 4.15, CONTENT_W * 0.7, 0.6, size=T_LEAD,
         color=MUTED, spacing=1.3)
