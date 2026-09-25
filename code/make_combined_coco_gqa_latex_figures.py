#!/usr/bin/env python3
"""Create LaTeX-ready combined COCO+GQA figures with Georgia typography."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BASE = Path(__file__).resolve().parents[1]
COCO_CSV = BASE / "runs/model_expansion_limit100/condition_summary.csv"
GQA_CSV = BASE / "runs/gqa_multimodel_summary/condition_summary.csv"
OUT_DIR = Path("output/pdf/combined_coco_gqa_latex_figures")

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
TASK1 = {"COCO": "task1", "GQA": "gqa_task1"}
TASK2 = {"COCO": "task2", "GQA": "gqa_task2"}

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


def set_font(c: canvas.Canvas, bold: bool = False, size: float = 9) -> None:
    c.setFont(FONT_BOLD if bold else FONT, size)


def load_dataset(path: Path, dataset: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["dataset"] = dataset
    return rows


def load_rows() -> list[dict[str, str]]:
    return load_dataset(COCO_CSV, "COCO") + load_dataset(GQA_CSV, "GQA")


def row_for(rows: list[dict[str, str]], dataset: str, task: str, model: str, mode: str) -> dict[str, str]:
    for row in rows:
        if row["dataset"] == dataset and row["task"] == task and row["model"] == model and row["mode"] == mode:
            return row
    raise KeyError((dataset, task, model, mode))


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def axes(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    y_label: str | None = None,
    y_max: float = 1.0,
    ticks: list[float] | None = None,
) -> None:
    ticks = ticks or [0, 0.25, 0.5, 0.75, 1.0]
    c.setStrokeColor(GRID)
    c.setLineWidth(0.35)
    set_font(c, False, 6.6)
    for tick in ticks:
        yy = y + tick * h
        c.line(x, yy, x + w, yy)
        c.setFillColor(MUTED)
        c.drawRightString(x - 0.045 * inch, yy - 0.023 * inch, f"{int(tick * y_max * 100)}")
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.65)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    if y_label:
        c.saveState()
        c.translate(x - 0.31 * inch, y + h / 2)
        c.rotate(90)
        c.setFillColor(MUTED)
        set_font(c, False, 6.8)
        c.drawCentredString(0, 0, y_label)
        c.restoreState()


def legend(c: canvas.Canvas, items: list[tuple[str, colors.Color]], x: float, y: float, *, cols: int) -> None:
    set_font(c, False, 7.1)
    for i, (label, color) in enumerate(items):
        col = i % cols
        row = i // cols
        xx = x + col * 0.86 * inch
        yy = y - row * 0.16 * inch
        c.setFillColor(color)
        c.rect(xx, yy - 0.055 * inch, 0.08 * inch, 0.08 * inch, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.drawString(xx + 0.105 * inch, yy - 0.052 * inch, label)


def grouped_panel(
    c: canvas.Canvas,
    rows: list[dict[str, str]],
    *,
    dataset: str,
    task: str,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    y_label: str,
    y_max: float,
    annotate_threshold: float,
) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 9.2)
    c.drawString(x, y + h + 0.16 * inch, title)
    tick_values = [0, 0.25, 0.5, 0.75, 1.0] if y_max == 1 else [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axes(c, x, y, w, h, y_label=y_label, y_max=y_max, ticks=tick_values)
    group_w = w / len(MODES)
    bar_w = group_w / (len(MODELS) + 1.45)
    for i, mode in enumerate(MODES):
        gx = x + i * group_w
        for j, model in enumerate(MODELS):
            value = float(row_for(rows, dataset, task, model, mode)["main_error"])
            bx = gx + 0.30 * bar_w + j * bar_w
            bh = min(value / y_max, 1.0) * h
            c.setFillColor(MODEL_COLORS[model])
            c.rect(bx, y, bar_w * 0.76, bh, stroke=0, fill=1)
            if value >= annotate_threshold:
                c.setFillColor(TEXT)
                set_font(c, False, 5.2)
                c.drawCentredString(bx + bar_w * 0.38, y + bh + 0.027 * inch, pct(value))
        c.setFillColor(TEXT)
        set_font(c, False, 6.2)
        c.drawCentredString(gx + group_w / 2, y - 0.13 * inch, MODE_LABELS[mode])


def average_task1_error(rows: list[dict[str, str]], model: str, mode: str) -> float:
    values = [
        float(row_for(rows, "COCO", TASK1["COCO"], model, mode)["main_error"]),
        float(row_for(rows, "GQA", TASK1["GQA"], model, mode)["main_error"]),
    ]
    return sum(values) / len(values)


def average_task2_cost(rows: list[dict[str, str]], model: str, mode: str) -> float:
    values = [
        float(row_for(rows, "COCO", TASK2["COCO"], model, mode)["main_error"]),
        float(row_for(rows, "GQA", TASK2["GQA"], model, mode)["main_error"]),
    ]
    return sum(values) / len(values)


def average_reduction_panel(c: canvas.Canvas, rows: list[dict[str, str]], *, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 9.2)
    c.drawString(x, y + h + 0.16 * inch, "Average Task1 reduction across COCO and GQA")
    set_font(c, False, 6.6)
    c.setFillColor(MUTED)
    c.drawString(x, y + h + 0.035 * inch, "Default vs. visual-sketchpad; lower over-answer is better")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = x + tick * w
        c.setStrokeColor(GRID)
        c.line(xx, y, xx, y + h)
        c.setFillColor(MUTED)
        set_font(c, False, 6.2)
        c.drawCentredString(xx, y - 0.13 * inch, f"{int(tick * 100)}")
    row_gap = h / (len(MODELS) + 0.35)
    for i, model in enumerate(MODELS):
        yy = y + h - (i + 0.72) * row_gap
        d = average_task1_error(rows, model, "default")
        s = average_task1_error(rows, model, "visual_sketchpad")
        dx, sx = x + d * w, x + s * w
        c.setStrokeColor(MODEL_COLORS[model])
        c.setLineWidth(3.0)
        c.line(sx, yy, dx, yy)
        c.setFillColor(MODE_COLORS["default"])
        c.circle(dx, yy, 3.5, stroke=0, fill=1)
        c.setFillColor(MODE_COLORS["visual_sketchpad"])
        c.circle(sx, yy, 3.5, stroke=0, fill=1)
        c.setFillColor(TEXT)
        set_font(c, True, 6.6)
        c.drawRightString(x - 0.07 * inch, yy - 0.023 * inch, MODEL_SHORT[model])
        set_font(c, False, 6.1)
        c.drawString(x + w + 0.05 * inch, yy - 0.023 * inch, f"-{(d - s) * 100:.0f} pts")
    c.setFillColor(MUTED)
    set_font(c, False, 6.4)
    c.drawCentredString(x + w / 2, y - 0.29 * inch, "Mean over-answer rate")


def tradeoff_panel(c: canvas.Canvas, rows: list[dict[str, str]], *, x: float, y: float, w: float, h: float) -> None:
    c.setFillColor(TEXT)
    set_font(c, True, 9.2)
    c.drawString(x, y + h + 0.16 * inch, "Average benefit vs. false-clarification cost")
    set_font(c, False, 6.6)
    c.setFillColor(MUTED)
    c.drawString(x, y + h + 0.035 * inch, "x: Task1 reduction; y: Task2 sketchpad cost")
    max_x, max_y = 1.0, 0.03
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        xx = x + tick * w
        c.setStrokeColor(GRID)
        c.line(xx, y, xx, y + h)
        c.setFillColor(MUTED)
        set_font(c, False, 6.2)
        c.drawCentredString(xx, y - 0.13 * inch, f"{int(tick * 100)}")
    for tick in [0, 0.01, 0.02, 0.03]:
        yy = y + tick / max_y * h
        c.setStrokeColor(GRID)
        c.line(x, yy, x + w, yy)
        c.setFillColor(MUTED)
        set_font(c, False, 6.2)
        c.drawRightString(x - 0.045 * inch, yy - 0.023 * inch, f"{int(tick * 100)}")
    c.setStrokeColor(AXIS)
    c.line(x, y, x + w, y)
    c.line(x, y, x, y + h)
    offsets = {
        "GPT-5.4-mini": (-0.19 * inch, -0.13 * inch),
        "Gemini-3.1-Pro": (0.04 * inch, 0.04 * inch),
        "GLM-4.6V": (0.04 * inch, 0.12 * inch),
        "Mistral-Small-3.2-Vision": (0.04 * inch, 0.04 * inch),
        "Qwen2.5-VL-72B": (-0.24 * inch, 0.08 * inch),
    }
    for model in MODELS:
        reduction = average_task1_error(rows, model, "default") - average_task1_error(rows, model, "visual_sketchpad")
        cost = average_task2_cost(rows, model, "visual_sketchpad")
        xx = x + reduction / max_x * w
        yy = y + min(cost, max_y) / max_y * h
        c.setFillColor(MODEL_COLORS[model])
        c.circle(xx, yy, 4.2, stroke=0, fill=1)
        c.setFillColor(TEXT)
        set_font(c, False, 6.4)
        ox, oy = offsets[model]
        c.drawString(xx + ox, yy + oy, MODEL_SHORT[model])
    c.setFillColor(MUTED)
    set_font(c, False, 6.4)
    c.drawCentredString(x + w / 2, y - 0.29 * inch, "Mean Task1 over-answer reduction")
    c.saveState()
    c.translate(x - 0.28 * inch, y + h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Mean Task2 cost")
    c.restoreState()


def main_figure(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_w, page_h = landscape(letter)
    c.setFillColor(TEXT)
    set_font(c, True, 15.4)
    c.drawString(0.42 * inch, page_h - 0.38 * inch, "Process prior generalizes across COCO and GQA")
    c.setFillColor(MUTED)
    set_font(c, False, 7.8)
    c.drawString(0.42 * inch, page_h - 0.57 * inch, "Five VLMs, two datasets, two referent tasks, four conditions. Georgia font; vector PDF.")
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 7.52 * inch, page_h - 0.38 * inch, cols=3)

    x1, x2 = 0.70 * inch, 5.78 * inch
    w = 4.32 * inch
    h = 1.38 * inch
    y1, y2, y3 = 4.53 * inch, 2.60 * inch, 0.67 * inch
    panels = [
        ("A", x1, y1, "COCO ambiguous referents", "COCO", TASK1["COCO"], "Over-answer (%)", 1.0, 0.08),
        ("B", x2, y1, "GQA ambiguous referents", "GQA", TASK1["GQA"], "Over-answer (%)", 1.0, 0.08),
        ("C", x1, y2, "COCO unique referents", "COCO", TASK2["COCO"], "False clarify (%)", 0.08, 0.006),
        ("D", x2, y2, "GQA unique referents", "GQA", TASK2["GQA"], "False clarify (%)", 0.25, 0.025),
    ]
    for label, x, y, title, dataset, task, ylabel, ymax, thresh in panels:
        c.setFillColor(TEXT)
        set_font(c, True, 10.2)
        c.drawString(x - 0.30 * inch, y + h + 0.16 * inch, label)
        grouped_panel(c, rows, dataset=dataset, task=task, x=x, y=y, w=w, h=h, title=title, y_label=ylabel, y_max=ymax, annotate_threshold=thresh)
    c.setFillColor(TEXT)
    set_font(c, True, 10.2)
    c.drawString(x1 - 0.30 * inch, y3 + h + 0.16 * inch, "E")
    average_reduction_panel(c, rows, x=x1 + 0.36 * inch, y=y3, w=w - 0.76 * inch, h=h)
    set_font(c, True, 10.2)
    c.drawString(x2 - 0.30 * inch, y3 + h + 0.16 * inch, "F")
    tradeoff_panel(c, rows, x=x2 + 0.38 * inch, y=y3, w=w - 0.78 * inch, h=h)


def task1_side_by_side(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_w, page_h = landscape(letter)
    c.setFillColor(TEXT)
    set_font(c, True, 14)
    c.drawString(0.45 * inch, page_h - 0.45 * inch, "Ambiguous referents: COCO and GQA")
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 7.15 * inch, page_h - 0.45 * inch, cols=3)
    grouped_panel(
        c,
        rows,
        dataset="COCO",
        task=TASK1["COCO"],
        x=0.8 * inch,
        y=1.12 * inch,
        w=4.35 * inch,
        h=3.65 * inch,
        title="COCO Task1",
        y_label="Over-answer (%)",
        y_max=1.0,
        annotate_threshold=0.08,
    )
    grouped_panel(
        c,
        rows,
        dataset="GQA",
        task=TASK1["GQA"],
        x=5.9 * inch,
        y=1.12 * inch,
        w=4.35 * inch,
        h=3.65 * inch,
        title="GQA Task1",
        y_label="Over-answer (%)",
        y_max=1.0,
        annotate_threshold=0.08,
    )


def task2_side_by_side(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_w, page_h = landscape(letter)
    c.setFillColor(TEXT)
    set_font(c, True, 14)
    c.drawString(0.45 * inch, page_h - 0.45 * inch, "Unique referents: COCO and GQA")
    legend(c, [(MODEL_SHORT[m], MODEL_COLORS[m]) for m in MODELS], 7.15 * inch, page_h - 0.45 * inch, cols=3)
    grouped_panel(
        c,
        rows,
        dataset="COCO",
        task=TASK2["COCO"],
        x=0.8 * inch,
        y=1.12 * inch,
        w=4.35 * inch,
        h=3.65 * inch,
        title="COCO Task2",
        y_label="False clarify (%)",
        y_max=0.08,
        annotate_threshold=0.006,
    )
    grouped_panel(
        c,
        rows,
        dataset="GQA",
        task=TASK2["GQA"],
        x=5.9 * inch,
        y=1.12 * inch,
        w=4.35 * inch,
        h=3.65 * inch,
        title="GQA Task2",
        y_label="False clarify (%)",
        y_max=0.25,
        annotate_threshold=0.025,
    )


def average_effects_page(c: canvas.Canvas, rows: list[dict[str, str]]) -> None:
    page_w, page_h = landscape(letter)
    c.setFillColor(TEXT)
    set_font(c, True, 14)
    c.drawString(0.45 * inch, page_h - 0.45 * inch, "Average effects across datasets")
    average_reduction_panel(c, rows, x=1.05 * inch, y=1.05 * inch, w=4.0 * inch, h=3.0 * inch)
    tradeoff_panel(c, rows, x=6.0 * inch, y=1.05 * inch, w=3.9 * inch, h=3.0 * inch)


def create_pdf(path: Path, size: tuple[float, float], draw_func: Callable[[canvas.Canvas, list[dict[str, str]]], None]) -> None:
    rows = load_rows()
    c = canvas.Canvas(str(path), pagesize=size)
    c.setTitle(path.stem)
    c.setFillColor(colors.white)
    c.rect(0, 0, size[0], size[1], stroke=0, fill=1)
    draw_func(c, rows)
    c.showPage()
    c.save()


def pack_pdf(path: Path, rows: list[dict[str, str]]) -> None:
    size = landscape(letter)
    c = canvas.Canvas(str(path), pagesize=size)
    c.setTitle("Combined COCO GQA figure pack")
    for draw in [main_figure, task1_side_by_side, task2_side_by_side, average_effects_page]:
        c.setFillColor(colors.white)
        c.rect(0, 0, size[0], size[1], stroke=0, fill=1)
        draw(c, rows)
        c.showPage()
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    create_pdf(OUT_DIR / "fig_combined_coco_gqa_main.pdf", landscape(letter), main_figure)
    create_pdf(OUT_DIR / "fig_coco_gqa_task1_side_by_side.pdf", landscape(letter), task1_side_by_side)
    create_pdf(OUT_DIR / "fig_coco_gqa_task2_side_by_side.pdf", landscape(letter), task2_side_by_side)
    create_pdf(OUT_DIR / "fig_coco_gqa_average_effects.pdf", landscape(letter), average_effects_page)
    pack_pdf(OUT_DIR / "combined_coco_gqa_figures_pack.pdf", rows)
    (OUT_DIR / "latex_include_snippet.tex").write_text(
        "\\begin{figure*}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{fig_combined_coco_gqa_main.pdf}\n"
        "  \\caption{Process-prior prompting reduces over-answering on underspecified visual questions across COCO and GQA while preserving performance on unique-referent controls.}\n"
        "  \\label{fig:combined-coco-gqa}\n"
        "\\end{figure*}\n",
        encoding="utf-8",
    )
    print(f"Wrote combined figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
