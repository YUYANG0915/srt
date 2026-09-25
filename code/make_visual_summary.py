#!/usr/bin/env python3
"""Create dependency-free SVG visualizations for the GPT/Qwen pilot summary."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "runs/visual_summary"
FIG_DIR = OUT_DIR / "figures"

MODELS = ["GPT-5.4-mini", "Qwen2.5-VL-72B"]
MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]
MODE_LABELS = {
    "default": "Default",
    "evidence": "Evidence",
    "marked_no_prior": "Marked\nNo Prior",
    "visual_sketchpad": "Visual\nSketchpad",
}

DATA = [
    {
        "model": "GPT-5.4-mini",
        "task": "Task1 ambiguous",
        "mode": "default",
        "n": 457,
        "policy_accuracy": 0.000,
        "main_error": 0.985,
        "answer": 450,
        "clarify": 0,
        "enumerate": 0,
        "uncertain": 7,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task1 ambiguous",
        "mode": "evidence",
        "n": 457,
        "policy_accuracy": 0.451,
        "main_error": 0.547,
        "answer": 250,
        "clarify": 206,
        "enumerate": 0,
        "uncertain": 1,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task1 ambiguous",
        "mode": "marked_no_prior",
        "n": 457,
        "policy_accuracy": 0.007,
        "main_error": 0.989,
        "answer": 452,
        "clarify": 0,
        "enumerate": 3,
        "uncertain": 2,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task1 ambiguous",
        "mode": "visual_sketchpad",
        "n": 457,
        "policy_accuracy": 0.888,
        "main_error": 0.092,
        "answer": 42,
        "clarify": 126,
        "enumerate": 280,
        "uncertain": 9,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task1 ambiguous",
        "mode": "default",
        "n": 457,
        "policy_accuracy": 0.035,
        "main_error": 0.958,
        "answer": 438,
        "clarify": 9,
        "enumerate": 7,
        "uncertain": 3,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task1 ambiguous",
        "mode": "evidence",
        "n": 457,
        "policy_accuracy": 0.427,
        "main_error": 0.573,
        "answer": 262,
        "clarify": 195,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task1 ambiguous",
        "mode": "marked_no_prior",
        "n": 457,
        "policy_accuracy": 0.098,
        "main_error": 0.902,
        "answer": 412,
        "clarify": 3,
        "enumerate": 42,
        "uncertain": 0,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task1 ambiguous",
        "mode": "visual_sketchpad",
        "n": 457,
        "policy_accuracy": 0.910,
        "main_error": 0.085,
        "answer": 39,
        "clarify": 398,
        "enumerate": 18,
        "uncertain": 2,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task2 unique",
        "mode": "default",
        "n": 330,
        "policy_accuracy": 1.000,
        "main_error": 0.000,
        "answer": 330,
        "clarify": 0,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task2 unique",
        "mode": "evidence",
        "n": 330,
        "policy_accuracy": 0.973,
        "main_error": 0.024,
        "answer": 321,
        "clarify": 8,
        "enumerate": 0,
        "uncertain": 1,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task2 unique",
        "mode": "marked_no_prior",
        "n": 330,
        "policy_accuracy": 1.000,
        "main_error": 0.000,
        "answer": 330,
        "clarify": 0,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "GPT-5.4-mini",
        "task": "Task2 unique",
        "mode": "visual_sketchpad",
        "n": 330,
        "policy_accuracy": 1.000,
        "main_error": 0.000,
        "answer": 330,
        "clarify": 0,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task2 unique",
        "mode": "default",
        "n": 330,
        "policy_accuracy": 0.979,
        "main_error": 0.003,
        "answer": 323,
        "clarify": 1,
        "enumerate": 0,
        "uncertain": 6,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task2 unique",
        "mode": "evidence",
        "n": 330,
        "policy_accuracy": 0.948,
        "main_error": 0.052,
        "answer": 313,
        "clarify": 17,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task2 unique",
        "mode": "marked_no_prior",
        "n": 330,
        "policy_accuracy": 0.985,
        "main_error": 0.015,
        "answer": 325,
        "clarify": 5,
        "enumerate": 0,
        "uncertain": 0,
    },
    {
        "model": "Qwen2.5-VL-72B",
        "task": "Task2 unique",
        "mode": "visual_sketchpad",
        "n": 330,
        "policy_accuracy": 0.994,
        "main_error": 0.000,
        "answer": 328,
        "clarify": 0,
        "enumerate": 0,
        "uncertain": 2,
    },
]

PALETTE = {
    "GPT-5.4-mini": "#3267A8",
    "Qwen2.5-VL-72B": "#D66B2A",
    "answer": "#C94C4C",
    "clarify": "#377D5B",
    "enumerate": "#5C6BC0",
    "uncertain": "#9E9E9E",
    "axis": "#333333",
    "grid": "#E6E6E6",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, value: str, size: int = 13, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#222">{esc(value)}</text>'
    )


def multiline_label(x: float, y: float, value: str, size: int = 12) -> list[str]:
    lines = value.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(text(x, y + i * (size + 3), line, size=size))
    return out


def row_for(model: str, task: str, mode: str) -> dict[str, object]:
    for row in DATA:
        if row["model"] == model and row["task"] == task and row["mode"] == mode:
            return row
    raise KeyError((model, task, mode))


def grouped_bar_svg(path: Path, task: str, metric: str, title: str, y_label: str) -> None:
    width, height = 920, 520
    margin = {"left": 80, "right": 30, "top": 70, "bottom": 95}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    baseline = margin["top"] + plot_h
    scale = plot_h / 1.0
    group_w = plot_w / len(MODES)
    bar_w = 42
    gap = 10
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(width / 2, 34, title, size=22, weight="700"),
        text(22, height / 2, y_label, size=13, anchor="middle"),
    ]
    parts.append(f'<g transform="rotate(-90 22 {height / 2})"></g>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = baseline - tick * scale
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(margin["left"] - 12, y + 4, f"{int(tick * 100)}%", size=12, anchor="end"))
    parts.append(f'<line x1="{margin["left"]}" y1="{baseline}" x2="{width - margin["right"]}" y2="{baseline}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{baseline}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')

    for i, mode in enumerate(MODES):
        center = margin["left"] + group_w * i + group_w / 2
        for j, model in enumerate(MODELS):
            value = float(row_for(model, task, mode)[metric])
            x = center - bar_w - gap / 2 if j == 0 else center + gap / 2
            y = baseline - value * scale
            h = value * scale
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{PALETTE[model]}" rx="3"/>')
            parts.append(text(x + bar_w / 2, max(y - 7, margin["top"] - 8), pct(value), size=11))
        parts.extend(multiline_label(center, baseline + 25, MODE_LABELS[mode], size=12))

    legend_x = width - 310
    legend_y = 54
    for i, model in enumerate(MODELS):
        x = legend_x + i * 150
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{PALETTE[model]}" rx="2"/>')
        parts.append(text(x + 20, legend_y + 12, model, size=12, anchor="start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def stacked_policy_svg(path: Path) -> None:
    width, height = 1020, 560
    margin = {"left": 105, "right": 30, "top": 75, "bottom": 115}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    baseline = margin["top"] + plot_h
    scale = plot_h
    bar_w = 38
    groups = [(model, mode) for model in MODELS for mode in MODES]
    group_w = plot_w / len(groups)
    policies = ["answer", "clarify", "enumerate", "uncertain"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(width / 2, 34, "Task 1 Policy Distribution Under Ambiguity", size=22, weight="700"),
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = baseline - tick * scale
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(margin["left"] - 12, y + 4, f"{int(tick * 100)}%", size=12, anchor="end"))
    parts.append(f'<line x1="{margin["left"]}" y1="{baseline}" x2="{width - margin["right"]}" y2="{baseline}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{baseline}" stroke="{PALETTE["axis"]}" stroke-width="1.2"/>')

    for i, (model, mode) in enumerate(groups):
        row = row_for(model, "Task1 ambiguous", mode)
        x = margin["left"] + group_w * i + group_w / 2 - bar_w / 2
        y_cursor = baseline
        for pol in policies:
            frac = int(row[pol]) / int(row["n"])
            h = frac * scale
            y_cursor -= h
            parts.append(f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w}" height="{h:.1f}" fill="{PALETTE[pol]}" rx="1"/>')
        parts.extend(multiline_label(x + bar_w / 2, baseline + 24, MODE_LABELS[mode], size=11))
        if mode == "evidence":
            parts.append(text(x + bar_w / 2, baseline + 72, model.replace("-72B", ""), size=11, weight="700"))

    legend_x = 610
    legend_y = 52
    for i, pol in enumerate(policies):
        x = legend_x + i * 95
        parts.append(f'<rect x="{x}" y="{legend_y}" width="13" height="13" fill="{PALETTE[pol]}" rx="2"/>')
        parts.append(text(x + 18, legend_y + 11, pol, size=12, anchor="start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def effect_svg(path: Path) -> None:
    width, height = 760, 380
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(width / 2, 34, "Marked Visual Evidence Is Not Enough", size=22, weight="700"),
        text(width / 2, 58, "Over-answer rate drops only when marked evidence is paired with a structured process prior.", size=13),
    ]
    x0, x1 = 170, 590
    y_gpt, y_qwen = 145, 245
    for y, model in [(y_gpt, "GPT-5.4-mini"), (y_qwen, "Qwen2.5-VL-72B")]:
        marked = float(row_for(model, "Task1 ambiguous", "marked_no_prior")["main_error"])
        sketch = float(row_for(model, "Task1 ambiguous", "visual_sketchpad")["main_error"])
        marked_x = x0 + marked * (x1 - x0)
        sketch_x = x0 + sketch * (x1 - x0)
        parts.append(text(32, y + 5, model, size=13, anchor="start", weight="700"))
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#D8D8D8" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<line x1="{sketch_x:.1f}" y1="{y}" x2="{marked_x:.1f}" y2="{y}" stroke="{PALETTE[model]}" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{marked_x:.1f}" cy="{y}" r="10" fill="{PALETTE["answer"]}"/>')
        parts.append(f'<circle cx="{sketch_x:.1f}" cy="{y}" r="10" fill="{PALETTE["clarify"]}"/>')
        parts.append(text(marked_x, y - 18, pct(marked), size=12))
        parts.append(text(sketch_x, y - 18, pct(sketch), size=12))
        parts.append(text((marked_x + sketch_x) / 2, y + 35, f"-{(marked - sketch) * 100:.1f} pts", size=13, weight="700"))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = x0 + tick * (x1 - x0)
        parts.append(f'<line x1="{x:.1f}" y1="300" x2="{x:.1f}" y2="306" stroke="{PALETTE["axis"]}"/>')
        parts.append(text(x, 326, f"{int(tick * 100)}%", size=11))
    parts.append(text(x0, 356, "Visual sketchpad", size=12, anchor="start"))
    parts.append(text(x1, 356, "Marked no prior", size=12, anchor="end"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv() -> None:
    csv_path = OUT_DIR / "summary_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DATA[0].keys()))
        writer.writeheader()
        writer.writerows(DATA)


def write_markdown() -> None:
    md = [
        "# Visual Summary: GPT vs Qwen",
        "",
        "This summary visualizes the current 2 models x 2 tasks x 4 conditions pilot.",
        "",
        "## Main Takeaway",
        "",
        "Marked visual evidence alone does not fix over-answering under ambiguity. The large gain appears when marked evidence is paired with an explicit referential process prior.",
        "",
        "![Marked evidence is not enough](figures/sketchpad_effect.svg)",
        "",
        "## Task 1: Ambiguous Referent",
        "",
        "Lower over-answer is better. Both models remain highly over-confident with `marked_no_prior`, but improve sharply under `visual_sketchpad`.",
        "",
        "![Task1 over-answer](figures/task1_over_answer.svg)",
        "",
        "Policy distribution shows that visual sketchpad shifts models from forced answers toward clarification/enumeration.",
        "",
        "![Task1 policy distribution](figures/task1_policy_distribution.svg)",
        "",
        "## Task 2: Unique Referent Control",
        "",
        "Lower unnecessary clarification is better. Visual sketchpad does not induce a trivial always-clarify strategy.",
        "",
        "![Task2 unnecessary clarification](figures/task2_unnecessary_clarification.svg)",
        "",
        "## Data",
        "",
        "The source table for these figures is `summary_metrics.csv` in this directory.",
    ]
    (OUT_DIR / "visual_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grouped_bar_svg(
        FIG_DIR / "task1_over_answer.svg",
        "Task1 ambiguous",
        "main_error",
        "Task 1: Over-Answer Rate Under Ambiguity",
        "Over-answer rate",
    )
    grouped_bar_svg(
        FIG_DIR / "task2_unnecessary_clarification.svg",
        "Task2 unique",
        "main_error",
        "Task 2: Unnecessary Clarification Under Unique Referents",
        "Unnecessary clarification",
    )
    stacked_policy_svg(FIG_DIR / "task1_policy_distribution.svg")
    effect_svg(FIG_DIR / "sketchpad_effect.svg")
    write_csv()
    write_markdown()


if __name__ == "__main__":
    main()
