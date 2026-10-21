"""YogaVision presentation design system.

Three colours only, one type scale, one spacing grid. Every slide is built
from these primitives so the deck reads as one designed object.

Palette (dark, print-safe):
  INK     near-black navy, primary surface and text
  ACCENT  teal, single highlight colour for data, emphasis, and rules
  MUTED   cool grey, secondary text and de-emphasised labels

Plus one semantic red for the paper's numbers on the comparison slide only,
so the reader can tell "published claim" from "our measured result".
"""
from __future__ import annotations

from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ── canvas ──────────────────────────────────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = 0.62                 # single left/right margin, inches
CONTENT_W = 13.333 - 2 * MARGIN

# ── palette: 3 colours + 1 semantic ────────────────────────────────────────
INK = RGBColor(0x0F, 0x17, 0x2A)        # deep navy, surfaces and primary text
INK_SOFT = RGBColor(0x1A, 0x24, 0x3D)   # raised card surface
INK_LINE = RGBColor(0x2A, 0x36, 0x52)   # hairline divider
ACCENT = RGBColor(0x2D, 0xD4, 0xBF)     # teal, the one highlight colour
ACCENT_DIM = RGBColor(0x1B, 0x7F, 0x74)  # teal at low emphasis
MUTED = RGBColor(0x94, 0xA3, 0xB8)      # secondary text
MUTED_DIM = RGBColor(0x64, 0x74, 0x8B)  # tertiary text
WHITE = RGBColor(0xF8, 0xFA, 0xFC)      # primary text on dark
WARN = RGBColor(0xF8, 0x71, 0x71)       # semantic red, comparison slide only

# ── type scale ──────────────────────────────────────────────────────────────
FONT = "Segoe UI"             # ships with Windows, renders cleanly in Office
FONT_MONO = "Consolas"        # code, file names, numbers

T_HERO = Pt(44)               # title slide
T_TITLE = Pt(28)              # slide titles
T_KICKER = Pt(12)             # eyebrow label above a title
T_LEAD = Pt(16)               # deck-level lead paragraph
T_BODY = Pt(13)               # default body copy
T_SMALL = Pt(11)              # supporting copy
T_TINY = Pt(9.5)              # captions, sources
T_NUMBER = Pt(34)             # stat numerals
T_NUMBER_MD = Pt(26)          # stat numerals, compact columns
T_NUMBER_SM = Pt(22)

# ── vertical rhythm ─────────────────────────────────────────────────────────
TITLE_Y = 0.52                # every content slide's title baseline
BODY_Y = 1.55                 # first row of content
FOOTER_Y = 6.95
ROW_GAP = 0.22

LEFT = PP_ALIGN.LEFT
CENTER = PP_ALIGN.CENTER
RIGHT = PP_ALIGN.RIGHT


def emu(inches: float) -> Emu:
    return Emu(int(round(inches * 914400)))
