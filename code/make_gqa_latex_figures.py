#!/usr/bin/env python3
"""Create LaTeX-ready GQA result figures with Georgia typography."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BASE = Path(__file__).resolve().parents[1]
SUMMARY_CSV = BASE / "runs/gqa_multimodel_summary/condition_summary.csv"
OUT_DIR = Path("output/pdf/gqa_latex_figures")

GEORGIA = Path("/System/Library/Fonts/Supplemental/Georgia.ttf")
GEORGIA_BOLD = Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf")

MODELS = [
    "GPT-5.4-mini",
    "Gemini-3.1-Pro",
    "GLM-4.6V",
    "Mistral-Small-3.2-Vision",
    "Qwen2.5-VL-72B",
]
MODEL_SHORT = {
    "GPT-5.4-mini": "GPT",
    "Gemini-3.1-Pro": "Gemini",
    "GLM-4.6V": "GLM",
    "Mistral-Small-3.2-Vision": "Mistral",
    "Qwen2.5-VL-72B": "Qwen",
}
MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]
MODE_LABELS = {
    "default": "Default",
    "evidence": "Evidence",
    "marked_no_prior": "Marked",
    "visual_sketchpad": "Sketchpad",
}

MODEL_COLORS = {
    "GPT-5.4-mini": colors.HexColor("#3366A6"),
    "Gemini-3.1-Pro": colors.HexColor("#B95F89"),
    "GLM-4.6V": colors.HexColor("#4C8A55"),
    "Mistral-Small-3.2-Vision": colors.HexColor("#7A5EB8"),
    "Qwen2.5-VL-72B": colors.HexColor("#D0732F"),
}
MODE_COLORS = {
    "default": colors.HexColor("#C94C4C"),
    "evidence": colors.HexColor("#7D8491"),
    "marked_no_prior": colors.HexColor("#6B77B8"),
    "visual_sketchpad": colors.HexColor("#3F8B65"),
}
POLICY_COLORS = {
    "answer": colors.HexColor("#C94C4C"),
    "clarify": colors.HexColor("#3F8B65"),
    "enumerate": colors.HexColor("#5B6EC0"),
    "uncertain": colors.HexColor("#9A9A9A"),
}
TEXT = colors.HexColor("#202020")
MUTED = colors.HexColor("#666666")
GRID = colors.HexColor("#E3E3E3")
AXIS = colors.HexColor("#303030")


def register_fonts() -> tuple[str, str]:
    if GEORGIA.exists() and GEORGIA_BOLD.exists():
        pdfmetrics.registerFont(TTFont("GeorgiaCustom", str(GEORGIA)))
        pdfmetrics.registerFont(TTFont("GeorgiaCustom-Bold", str(GEORGIA_BOLD)))
        return "GeorgiaCustom", "GeorgiaCustom-Bold"
    return "Times-Roman", "Times-Bold"


FONT, FONT_BOLD = register_fonts()


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_for(rows: list[dict[str, str]], task: str, model: str, mode: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["model"] == model and row["mode"] == mode:
            return row
    raise KeyError((task, model, mode))


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def set_font(c: canvas.Canvas, bold: bool = False, size: float = 9) -> None:
    c.setFont(FONT_BOLD if bold else FONT, size)


def draw_panel_label(c: canvas.Canvas, label: str, x: float, y: float) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 11)
    c.drawString(x, y, label)


def draw_title(c: canvas.Canvas, title: str, subtitle: str | None = None, *, width: float) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 16)
    c.drawString(0.45 * inch, 0.45 * inch + width * 0, title)
    if subtitle:
        c.setFillColor(MUTED)
        set_font(c, False, 8.5)
        c.drawString(0.45 * inch, 0.21 * inch + width * 0, subtitle)


def axes(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    y_ticks: list[float] | None = None,
    y_label: str | None = None,
    value_max: float = 1.0,
) -> None:
    y_ticks = y_ticks or [0, 0.25, 0.5, 0.75, 1.0]
    c.setStrokeColor(GRID)
    c.setLineWidth(0.35)
    set_font(c, False, 7)
    for tick in y_ticks:
        yy = y + tick * h
        c.line(x, yy, x + w, yy)
        c.setFillColor(MUTED)
        c.drawRightString(x - 0.06 * inch, yy - 0.025 * inch, f"{int(tick * value_max * 100)}")
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.7)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    if y_label:
        c.saveState()
        c.translate(x - 0.37 * inch, y + h / 2)
        c.rotate(90)
        c.setFillColor(MUTED)
        set_font(c, False, 7.5)
        c.drawCentredString(0, 0, y_label)
        c.restoreState()


def legend(c: canvas.Canvas, items: list[tuple[str, colors.Color]], x: float, y: float, *, cols: int = 3) -> None:
    set_font(c, False, 7.4)
    for i, (label, color) in enumerate(items):
        col = i % cols
        row = i // cols
        xx = x + col * 1.0 * inch
        yy = y - row * 0.18 * inch
        c.setFillColor(color)
        c.rect(xx, yy - 0.06 * inch, 0.09 * inch, 0.09 * inch, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.drawString(xx + 0.12 * inch, yy - 0.055 * inch, label)


def grouped_error_bars(
    c: canvas.Canvas,
    rows: list[dict[str, str]],
    *,
    task: str,
    metric: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    y_label: str,
    annotate_threshold: float = 0.08,
    y_max: float = 1.0,
) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 10)
    c.drawString(x, y + h + 0.2 * inch, title)
    tick_values = [0, 0.25, 0.5, 0.75, 1.0] if y_max == 1.0 else [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes(c, x, y, w, h, y_label=y_label, y_ticks=tick_values, value_max=y_max)
    group_w = w / len(MODES)
    bar_w = group_w / (len(MODELS) + 1.5)
    for i, mode in enumerate(MODES):
        gx = x + i * group_w
        for j, model in enumerate(MODELS):
            value = float(row_for(rows, task, model, mode)[metric])
            bx = gx + 0.35 * bar_w + j * bar_w
            bh = min(value / y_max, 1.0) * h
            c.setFillColor(MODEL_COLORS[model])
            c.rect(bx, y, bar_w * 0.78, bh, stroke=0, fill=1)
            if value >= annotate_threshold:
                c.setFillColor(TEXT)
                set_font(c, False, 5.9)
                c.drawCentredString(bx + bar_w * 0.39, y + bh + 0.035 * inch, pct(value))
        c.setFillColor(TEXT)
        set_font(c, False, 7)
        c.drawCentredString(gx + group_w / 2, y - 0.16 * inch, MODE_LABELS[mode])


def dumbbell_reduction(
    c: canvas.Canvas,
    rows: list[dict[str, str]],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 10)
    c.drawString(x, y + h + 0.2 * inch, "Task1: process prior reduces over-answering")
    set_font(c, False, 7.3)
    c.setFillColor(MUTED)
    c.drawString(x, y + h + 0.06 * inch, "Default vs. visual-sketchpad; lower is better")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = x + tick * w
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(xx, y, xx, y + h)
        c.setFillColor(MUTED)
        set_font(c, False, 6.7)
        c.drawCentredString(xx, y - 0.15 * inch, f"{int(tick * 100)}")
    set_font(c, False, 7)
    c.setFillColor(MUTED)
    c.drawCentredString(x + w / 2, y - 0.31 * inch, "Over-answer rate")

    row_gap = h / (len(MODELS) + 0.4)
    for i, model in enumerate(MODELS):
        yy = y + h - (i + 0.75) * row_gap
        d = float(row_for(rows, "gqa_task1", model, "default")["main_error"])
        s = float(row_for(rows, "gqa_task1", model, "visual_sketchpad")["main_error"])
        dx, sx = x + d * w, x + s * w
        c.setStrokeColor(MODEL_COLORS[model])
        c.setLineWidth(3.2)
        c.line(sx, yy, dx, yy)
        c.setFillColor(MODE_COLORS["default"])
        c.circle(dx, yy, 3.8, stroke=0, fill=1)
        c.setFillColor(MODE_COLORS["visual_sketchpad"])
        c.circle(sx, yy, 3.8, stroke=0, fill=1)
        c.setFillColor(TEXT)
        set_font(c, True, 7)
        c.drawRightString(x - 0.09 * inch, yy - 0.025 * inch, MODEL_SHORT[model])
        set_font(c, False, 6.5)
        c.drawString(x + w + 0.08 * inch, yy - 0.025 * inch, f"-{(d - s) * 100:.0f} pts")


def tradeoff_scatter(
    c: canvas.Canvas,
    rows: list[dict[str, str]],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 10)
    c.drawString(x, y + h + 0.2 * inch, "Benefit vs. false-clarification cost")
    set_font(c, False, 7.3)
    c.setFillColor(MUTED)
    c.drawString(x, y + h + 0.06 * inch, "x: Task1 reduction; y: Task2 sketchpad unnecessary clarification")
    max_x, max_y = 1.0, 0.05
    c.setStrokeColor(GRID)
    c.setLineWidth(0.35)
    set_font(c, False, 6.7)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = x + tick / max_x * w
        c.line(xx, y, xx, y + h)
        c.setFillColor(MUTED)
        c.drawCentredString(xx, y - 0.15 * inch, f"{int(tick * 100)}")
    for tick in [0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        yy = y + tick / max_y * h
        c.line(x, yy, x + w, yy)
        c.setFillColor(MUTED)
        c.drawRightString(x - 0.05 * inch, yy - 0.025 * inch, f"{int(tick * 100)}")
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.7)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    label_offsets = {
        "GPT-5.4-mini": (-0.20 * inch, -0.14 * inch),
        "Gemini-3.1-Pro": (0.04 * inch, 0.03 * inch),
        "GLM-4.6V": (0.05 * inch, 0.15 * inch),
        "Mistral-Small-3.2-Vision": (0.04 * inch, 0.03 * inch),
        "Qwen2.5-VL-72B": (-0.28 * inch, 0.08 * inch),
    }
    for model in MODELS:
        default_err = float(row_for(rows, "gqa_task1", model, "default")["main_error"])
        sketch_err = float(row_for(rows, "gqa_task1", model, "visual_sketchpad")["main_error"])
        task2_cost = float(row_for(rows, "gqa_task2", model, "visual_sketchpad")["main_error"])
        xx = x + (default_err - sketch_err) / max_x * w
        yy = y + min(task2_cost, max_y) / max_y * h
        c.setFillColor(MODEL_COLORS[model])
        c.circle(xx, yy, 4.6, stroke=0, fill=1)
        c.setFillColor(TEXT)
        set_font(c, False, 6.7)
        ox, oy = label_offsets[model]
        c.drawString(xx + ox, yy + oy, MODEL_SHORT[model])
    c.setFillColor(MUTED)
    set_font(c, False, 7)
    c.drawCentredString(x + w / 2, y - 0.31 * inch, "Reduction in Task1 over-answer rate")
    c.saveState()
    c.translate(x - 0.35 * inch, y + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Task2 unnecessary clarification")
    c.restoreState()


def policy_stack(
    c: canvas.Canvas,
    rows: list[dict[str, str]],
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 10)
    c.drawString(x, y + h + 0.2 * inch, "Task1 visual-sketchpad policy distribution")
    axes(c, x, y, w, h, y_label="Policy share")
    policies = ["answer", "clarify", "enumerate", "uncertain"]
    bar_w = w / (len(MODELS) * 1.45)
    for i, model in enumerate(MODELS):
        row = row_for(rows, "gqa_task1", model, "visual_sketchpad")
        n = int(row["n"])
        bx = x + (i + 0.3) * (w / len(MODELS))
        cursor = y
        for pol in policies:
            frac = int(row[pol]) / n if n else 0
            c.setFillColor(POLICY_COLORS[pol])
            c.rect(bx, cursor, bar_w, frac * h, stroke=0, fill=1)
            cursor += frac * h
        c.setFillColor(TEXT)
        set_font(c, False, 7)
        c.drawCentredString(bx + bar_w / 2, y - 0.16 * inch, MODEL_SHORT[model])
    legend(c, [(p.title(), POLICY_COLORS[p]) for p in policies], x + w - 2.45 * inch, y + h + 0.60 * inch, cols=2)


def create_pdf(path: Path, size: tuple[float, float], draw_func: Callable[[canvas.Canvas, list[dict[str, str]]], None]) -> None:
    rows = load_rows()
    c = canvas.Canvas(str(path), pagesize=size)
    c.setTitle(path.stem)
    c.setFillColor(colors.white)
    c.rect(0, 0, size[0], size[1], stroke=0, fill=1)
    draw_func(c, rows)
    c.showPage()
    c.save()


def main_panel(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_w, page_h = landscape(letter)
    c.setFillColor(TEXT)
    set_font(c, True, 16)
    c.drawString(0.45 * inch, page_h - 0.42 * inch, "GQA underspecified VQA: process prior effects")
    c.setFillColor(MUTED)
    set_font(c, False, 8.2)
    c.drawString(0.45 * inch, page_h - 0.62 * inch, "Five VLMs, two tasks, four prompting/grounding conditions. Georgia font; vector PDF.")
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 7.7 * inch, page_h - 0.42 * inch, cols=3)

    panel_w = 4.35 * inch
    panel_h = 2.35 * inch
    x1, x2 = 0.72 * inch, 5.82 * inch
    y_top, y_bot = 4.45 * inch, 0.82 * inch
    draw_panel_label(c, "A", x1 - 0.32 * inch, y_top + panel_h + 0.17 * inch)
    grouped_error_bars(
        c,
        rows,
        task="gqa_task1",
        metric="main_error",
        x=x1,
        y=y_top,
        w=panel_w,
        h=panel_h,
        title="Ambiguous referents",
        y_label="Over-answer rate (%)",
    )
    draw_panel_label(c, "B", x2 - 0.32 * inch, y_top + panel_h + 0.17 * inch)
    grouped_error_bars(
        c,
        rows,
        task="gqa_task2",
        metric="main_error",
        x=x2,
        y=y_top,
        w=panel_w,
        h=panel_h,
        title="Unique referents",
        y_label="Unnecessary clarification (%)",
        annotate_threshold=0.025,
        y_max=0.25,
    )
    draw_panel_label(c, "C", x1 - 0.32 * inch, y_bot + panel_h + 0.17 * inch)
    dumbbell_reduction(c, rows, x=x1 + 0.4 * inch, y=y_bot, w=panel_w - 0.78 * inch, h=panel_h)
    draw_panel_label(c, "D", x2 - 0.32 * inch, y_bot + panel_h + 0.17 * inch)
    tradeoff_scatter(c, rows, x=x2 + 0.4 * inch, y=y_bot, w=panel_w - 0.82 * inch, h=panel_h)


def single_task1(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    grouped_error_bars(c, rows, task="gqa_task1", metric="main_error", x=0.72 * inch, y=0.7 * inch, w=6.5 * inch, h=3.0 * inch, title="GQA Task1: ambiguous referents", y_label="Over-answer rate (%)")
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 0.85 * inch, 4.15 * inch, cols=5)


def single_task2(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    grouped_error_bars(c, rows, task="gqa_task2", metric="main_error", x=0.72 * inch, y=0.7 * inch, w=6.5 * inch, h=3.0 * inch, title="GQA Task2: unique referents", y_label="Unnecessary clarification (%)", annotate_threshold=0.025, y_max=0.25)
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 0.85 * inch, 4.15 * inch, cols=5)


def single_reduction(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    dumbbell_reduction(c, rows, x=1.45 * inch, y=0.85 * inch, w=5.75 * inch, h=3.0 * inch)


def single_tradeoff(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    tradeoff_scatter(c, rows, x=1.1 * inch, y=0.85 * inch, w=5.8 * inch, h=3.0 * inch)


def single_policy(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    policy_stack(c, rows, x=0.75 * inch, y=0.75 * inch, w=6.5 * inch, h=3.0 * inch)


def pack_pdf(path: Path, rows: list[dict[str, str]]) -> None:
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    c.setTitle("GQA LaTeX figure pack")
    page_w, page_h = landscape(letter)
    for draw in [main_panel, single_task1, single_task2, single_reduction, single_tradeoff, single_policy]:
        c.setFillColor(colors.white)
        c.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        draw(c, rows)
        c.showPage()
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    figure_size = (7.6 * inch, 4.7 * inch)
    create_pdf(OUT_DIR / "fig_gqa_main_results.pdf", landscape(letter), main_panel)
    create_pdf(OUT_DIR / "fig_gqa_task1_overanswer.pdf", figure_size, single_task1)
    create_pdf(OUT_DIR / "fig_gqa_task2_unnecessary_clarification.pdf", figure_size, single_task2)
    create_pdf(OUT_DIR / "fig_gqa_overanswer_reduction.pdf", figure_size, single_reduction)
    create_pdf(OUT_DIR / "fig_gqa_tradeoff.pdf", figure_size, single_tradeoff)
    create_pdf(OUT_DIR / "fig_gqa_task1_sketchpad_policy_distribution.pdf", figure_size, single_policy)
    pack_pdf(OUT_DIR / "gqa_latex_figures_pack.pdf", rows)
    (OUT_DIR / "latex_include_snippet.tex").write_text(
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\linewidth]{fig_gqa_main_results.pdf}\n"
        "  \\caption{GQA underspecified VQA results across five VLMs.}\n"
        "  \\label{fig:gqa-main-results}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
