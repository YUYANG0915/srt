#!/usr/bin/env python3
"""Create a PDF pack of all expanded VLM condition figures."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


BASE = Path(__file__).resolve().parents[1]
SUMMARY_CSV = BASE / "runs/model_expansion_limit100/condition_summary.csv"
OUT_DIR = Path("output/pdf")
OUT_PDF = OUT_DIR / "vlm_condition_figures.pdf"

MODELS = [
    "GPT-5.4-mini",
    "Qwen2.5-VL-72B",
    "GLM-4.6V",
    "Mistral-Small-3.2-Vision",
    "Gemini-3.1-Pro",
]
MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]
MODE_LABELS = {
    "default": "Default",
    "evidence": "Evidence",
    "marked_no_prior": "Marked no prior",
    "visual_sketchpad": "Visual sketchpad",
}
MODEL_COLORS = {
    "GPT-5.4-mini": colors.HexColor("#2F6CA3"),
    "Qwen2.5-VL-72B": colors.HexColor("#D16A2E"),
    "GLM-4.6V": colors.HexColor("#5F8F3E"),
    "Mistral-Small-3.2-Vision": colors.HexColor("#8A5FBF"),
    "Gemini-3.1-Pro": colors.HexColor("#C34E73"),
}
POLICY_COLORS = {
    "answer": colors.HexColor("#C94C4C"),
    "clarify": colors.HexColor("#377D5B"),
    "enumerate": colors.HexColor("#5C6BC0"),
    "uncertain": colors.HexColor("#9E9E9E"),
}
GRID = colors.HexColor("#E5E5E5")
AXIS = colors.HexColor("#2C2C2C")
TEXT = colors.HexColor("#222222")


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_for(rows: list[dict[str, str]], task: str, model: str, mode: str) -> dict[str, str]:
    for row in rows:
        if row["task"] == task and row["model"] == model and row["mode"] == mode:
            return row
    raise KeyError((task, model, mode))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def title(c: canvas.Canvas, text: str, subtitle: str | None = None) -> None:
    w, h = landscape(letter)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.55 * inch, h - 0.55 * inch, text)
    if subtitle:
        c.setFont("Helvetica", 10.5)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(0.55 * inch, h - 0.78 * inch, subtitle)


def footer(c: canvas.Canvas, page: int) -> None:
    w, _ = landscape(letter)
    c.setFillColor(colors.HexColor("#777777"))
    c.setFont("Helvetica", 8.5)
    c.drawRightString(w - 0.45 * inch, 0.32 * inch, f"VLM underspecification pilot - page {page}")


def draw_legend(c: canvas.Canvas, x: float, y: float, labels: list[str], color_map: dict[str, colors.Color], cols: int = 3) -> None:
    c.setFont("Helvetica", 8.5)
    for i, label in enumerate(labels):
        col = i % cols
        row = i // cols
        lx = x + col * 1.95 * inch
        ly = y - row * 0.22 * inch
        c.setFillColor(color_map[label])
        c.roundRect(lx, ly - 0.08 * inch, 0.12 * inch, 0.12 * inch, 2, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.drawString(lx + 0.17 * inch, ly - 0.055 * inch, label)


def grouped_bar_page(c: canvas.Canvas, rows: list[dict[str, str]], *, task: str, metric: str, heading: str, y_label: str, page: int) -> None:
    w, h = landscape(letter)
    title(c, heading, y_label)
    left, right = 0.72 * inch, w - 0.35 * inch
    top, bottom = h - 1.55 * inch, 1.15 * inch
    plot_w, plot_h = right - left, top - bottom
    c.setStrokeColor(GRID)
    c.setLineWidth(0.5)
    c.setFont("Helvetica", 8.5)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom + tick * plot_h
        c.line(left, y, right, y)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawRightString(left - 0.08 * inch, y - 0.035 * inch, f"{int(tick * 100)}%")
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.8)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)

    group_w = plot_w / len(MODES)
    bar_w = group_w / (len(MODELS) + 2.4)
    for i, mode in enumerate(MODES):
        group_left = left + i * group_w
        for j, model in enumerate(MODELS):
            value = float(row_for(rows, task, model, mode)[metric])
            x = group_left + 0.45 * bar_w + j * bar_w
            bar_h = value * plot_h
            c.setFillColor(MODEL_COLORS[model])
            c.roundRect(x, bottom, bar_w * 0.82, bar_h, 2, stroke=0, fill=1)
            if value >= 0.07:
                c.setFillColor(TEXT)
                c.setFont("Helvetica", 6.8)
                c.drawCentredString(x + bar_w * 0.41, bottom + bar_h + 0.045 * inch, pct(value))
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(group_left + group_w / 2, bottom - 0.24 * inch, MODE_LABELS[mode])

    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(left, bottom - 0.52 * inch, "Each bar is one model. Lower is better for both plotted error metrics.")
    draw_legend(c, left + 3.75 * inch, h - 0.78 * inch, MODELS, MODEL_COLORS, cols=3)
    footer(c, page)
    c.showPage()


def policy_distribution_page(c: canvas.Canvas, rows: list[dict[str, str]], page: int) -> None:
    w, h = landscape(letter)
    title(c, "Task1 policy distribution under ambiguous referents", "Stacked policy proportions for each model and condition.")
    left, right = 0.62 * inch, w - 0.35 * inch
    top, bottom = h - 1.05 * inch, 1.08 * inch
    plot_w, plot_h = right - left, top - bottom
    groups = [(model, mode) for model in MODELS for mode in MODES]
    group_w = plot_w / len(groups)
    bar_w = group_w * 0.58
    policies = ["answer", "clarify", "enumerate", "uncertain"]

    c.setStrokeColor(GRID)
    c.setLineWidth(0.5)
    c.setFont("Helvetica", 8)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom + tick * plot_h
        c.line(left, y, right, y)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawRightString(left - 0.07 * inch, y - 0.03 * inch, f"{int(tick * 100)}%")
    c.setStrokeColor(AXIS)
    c.setLineWidth(0.8)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)

    for i, (model, mode) in enumerate(groups):
        row = row_for(rows, "task1", model, mode)
        n = int(row["n"])
        x = left + i * group_w + group_w / 2 - bar_w / 2
        y_cursor = bottom
        for pol in policies:
            frac = int(row[pol]) / n if n else 0
            seg_h = frac * plot_h
            c.setFillColor(POLICY_COLORS[pol])
            c.rect(x, y_cursor, bar_w, seg_h, stroke=0, fill=1)
            y_cursor += seg_h
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 6.6)
        c.saveState()
        c.translate(x + bar_w / 2, bottom - 0.13 * inch)
        c.rotate(65)
        c.drawString(0, 0, MODE_LABELS[mode])
        c.restoreState()
        if mode == "evidence":
            c.setFont("Helvetica-Bold", 7)
            c.drawCentredString(x + bar_w / 2, bottom - 0.55 * inch, model.split("-")[0])

    draw_legend(c, left + 5.9 * inch, h - 0.82 * inch, policies, POLICY_COLORS, cols=2)
    footer(c, page)
    c.showPage()


def effect_page(c: canvas.Canvas, rows: list[dict[str, str]], page: int) -> None:
    w, h = landscape(letter)
    title(
        c,
        "Effect of structured process prior",
        "Dumbbell plot: Task1 over-answer rate from marked-no-prior to visual-sketchpad.",
    )
    c.setFont("Helvetica", 8.5)
    legend_y = h - 0.78 * inch
    legend_x = w - 3.05 * inch
    c.setFillColor(colors.HexColor("#C94C4C"))
    c.circle(legend_x, legend_y, 5, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.drawString(legend_x + 0.12 * inch, legend_y - 0.04 * inch, "Marked no prior")
    c.setFillColor(colors.HexColor("#377D5B"))
    c.circle(legend_x + 1.38 * inch, legend_y, 5, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.drawString(legend_x + 1.50 * inch, legend_y - 0.04 * inch, "Visual sketchpad")
    x0, x1 = 2.2 * inch, w - 1.0 * inch
    y0 = h - 1.45 * inch
    row_gap = 0.62 * inch
    c.setFont("Helvetica", 8)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = x0 + tick * (x1 - x0)
        c.setStrokeColor(GRID)
        c.line(x, 1.0 * inch, x, h - 1.15 * inch)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawCentredString(x, 0.78 * inch, f"{int(tick * 100)}%")
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 9)
    c.drawCentredString((x0 + x1) / 2, 0.52 * inch, "Task1 over-answer rate")
    for i, model in enumerate(MODELS):
        y = y0 - i * row_gap
        marked = float(row_for(rows, "task1", model, "marked_no_prior")["main_error"])
        sketch = float(row_for(rows, "task1", model, "visual_sketchpad")["main_error"])
        mx = x0 + marked * (x1 - x0)
        sx = x0 + sketch * (x1 - x0)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", 9)
        c.drawRightString(x0 - 0.18 * inch, y - 0.03 * inch, model)
        c.setStrokeColor(MODEL_COLORS[model])
        c.setLineWidth(5)
        c.line(sx, y, mx, y)
        c.setFillColor(colors.HexColor("#C94C4C"))
        c.circle(mx, y, 5.5, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#377D5B"))
        c.circle(sx, y, 5.5, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica", 8)
        c.drawCentredString(mx, y + 0.13 * inch, pct(marked))
        c.drawCentredString(sx, y - 0.22 * inch, pct(sketch))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(x1 + 0.15 * inch, y - 0.035 * inch, f"-{(marked - sketch) * 100:.1f} pts")
    footer(c, page)
    c.showPage()


def overview_page(c: canvas.Canvas, rows: list[dict[str, str]], page: int) -> None:
    w, h = landscape(letter)
    title(c, "VLM underspecification: expanded results", "Five models, two tasks, four conditions. Expansion models use n=100 per condition.")
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.65 * inch, h - 1.25 * inch, "Main reading")
    c.setFont("Helvetica", 10)
    bullets = [
        "Task1: ambiguous referent questions. Lower over-answer rate is better.",
        "Task2: unique referent control. Lower unnecessary clarification is better.",
        "Default prompting strongly over-answers Task1 across all models.",
        "Visual sketchpad keeps Task2 answer behavior while sharply reducing Task1 over-answering.",
        "Gemini is atypically strong with evidence/marked conditions, but still over-answers under default prompting.",
    ]
    y = h - 1.55 * inch
    for bullet in bullets:
        c.circle(0.78 * inch, y + 0.035 * inch, 2.2, stroke=0, fill=1)
        c.drawString(0.9 * inch, y, bullet)
        y -= 0.25 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.65 * inch, y - 0.12 * inch, "Task1 over-answer rates")
    y -= 0.42 * inch
    c.setFont("Helvetica-Bold", 8)
    x_cols = [2.7 * inch, 3.7 * inch, 4.85 * inch, 6.25 * inch, 7.65 * inch]
    headers = ["Default", "Evidence", "Marked", "Sketchpad", "Drop"]
    c.drawString(0.75 * inch, y, "Model")
    for x, header in zip(x_cols, headers, strict=True):
        c.drawRightString(x, y, header)
    c.setFont("Helvetica", 8)
    y -= 0.18 * inch
    for model in MODELS:
        default = float(row_for(rows, "task1", model, "default")["main_error"])
        evidence = float(row_for(rows, "task1", model, "evidence")["main_error"])
        marked = float(row_for(rows, "task1", model, "marked_no_prior")["main_error"])
        sketch = float(row_for(rows, "task1", model, "visual_sketchpad")["main_error"])
        vals = [pct(default), pct(evidence), pct(marked), pct(sketch), f"{(marked - sketch) * 100:.1f} pts"]
        c.drawString(0.75 * inch, y, model)
        for x, val in zip(x_cols, vals, strict=True):
            c.drawRightString(x, y, val)
        y -= 0.2 * inch
    footer(c, page)
    c.showPage()


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    c = canvas.Canvas(str(OUT_PDF), pagesize=landscape(letter))
    overview_page(c, rows, 1)
    grouped_bar_page(
        c,
        rows,
        task="task1",
        metric="main_error",
        heading="Task1: over-answer rate under ambiguity",
        y_label="Lower is better. The expected policy is clarify or enumerate.",
        page=2,
    )
    grouped_bar_page(
        c,
        rows,
        task="task2",
        metric="main_error",
        heading="Task2: unnecessary clarification under unique referents",
        y_label="Lower is better. The expected policy is answer.",
        page=3,
    )
    policy_distribution_page(c, rows, 4)
    effect_page(c, rows, 5)
    c.save()


if __name__ == "__main__":
    build_pdf()
