#!/usr/bin/env python3
"""Summarize SRT effects across completed process families."""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "runs/srt_cross_family_summary_2026-07-02"

FAMILIES = [
    {
        "id": "P4",
        "name": "Tool-use",
        "n": 52,
        "path": ROOT / "runs/task3_tooluse_v2focused52_5model_4cond_summary_2026-06-30/task3_tooluse_summary.md",
    },
    {
        "id": "P5",
        "name": "Cleaning",
        "n": 56,
        "path": ROOT / "runs/task3_cleaning_v1reviewed56_5model_router_summary_2026-07-01/task3_cleaning_summary.md",
    },
    {
        "id": "P6",
        "name": "Craft",
        "n": 77,
        "path": ROOT / "runs/task3_craft_v1corrected77_5model_5cond_summary_2026-07-02/task3_craft_summary.md",
    },
    {
        "id": "P7",
        "name": "Mobility",
        "n": 106,
        "path": ROOT / "runs/task3_mobility_v1reviewed106_5model_5cond_summary_2026-07-02/task3_mobility_summary.md",
    },
]

MODELS = ["GLM-4.6V", "GPT-5.4-mini", "Gemini-3.1-Pro", "Mistral-Small-3.2", "Qwen2.5-VL-72B"]
CONDITION_ORDER = ["Default", "Generic SRT", "Scenario SRT", "Scenario SRT v2", "Boundary Router"]
COLORS = {
    "GLM-4.6V": "#E15759",
    "GPT-5.4-mini": "#767676",
    "Gemini-3.1-Pro": "#59A14F",
    "Mistral-Small-3.2": "#4C78A8",
    "Qwen2.5-VL-72B": "#B07AA1",
}


def parse_accuracy(cell: str) -> float | None:
    cell = cell.strip()
    if cell == "-":
        return None
    match = re.search(r"\(([-+]?\d+(?:\.\d+)?)%\)", cell)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r"([-+]?\d+(?:\.\d+)?)%", cell)
    if match:
        return float(match.group(1)) / 100.0
    return None


def parse_overall_table(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## Overall")
    table_lines: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## ") and table_lines:
            break
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif table_lines and not line.strip():
            continue
        elif table_lines:
            break
    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        record = dict(zip(header, cells, strict=True))
        model = record["Model"]
        accs = {k: parse_accuracy(v) for k, v in record.items() if k != "Model" and "Delta" not in k}
        rows.append({"model": model, "accs": accs})
    return rows


def build_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family in FAMILIES:
        for parsed in parse_overall_table(family["path"]):
            model = parsed["model"]
            accs = parsed["accs"]
            default = accs.get("Default")
            candidates = {k: v for k, v in accs.items() if k != "Default" and v is not None}
            if default is None or not candidates:
                continue
            best_condition, best_acc = max(candidates.items(), key=lambda kv: kv[1])
            router = accs.get("Boundary Router")
            out.append(
                {
                    "family_id": family["id"],
                    "family": family["name"],
                    "n": family["n"],
                    "model": model,
                    "default": default,
                    "best_srt_condition": best_condition,
                    "best_srt": best_acc,
                    "best_gain_pp": (best_acc - default) * 100,
                    "router": router,
                    "router_gain_pp": (router - default) * 100 if router is not None else None,
                    **{f"acc_{k.lower().replace(' ', '_')}": v for k, v in accs.items()},
                }
            )
    return out


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def fmt_pp(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:+.1f} pp"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "family_id",
        "family",
        "n",
        "model",
        "default",
        "best_srt_condition",
        "best_srt",
        "best_gain_pp",
        "router",
        "router_gain_pp",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def svg_text(x: float, y: float, text: str, size: int = 16, weight: str = "400", anchor: str = "start") -> str:
    from html import escape

    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Georgia, Times New Roman, serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#222">{escape(str(text))}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")
    try:
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(path.with_suffix(".pdf")), str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def plot_gain_heatmap(rows: list[dict[str, Any]]) -> None:
    width, height = 1120, 620
    margin_l, margin_t = 170, 130
    cell_w, cell_h = 205, 62
    by_key = {(r["model"], r["family_id"]): r["best_gain_pp"] for r in rows}
    body = [
        svg_text(40, 45, "SRT Cross-Family Summary: Best SRT Gain over Default", 28, "700"),
        svg_text(40, 72, "Positive gains indicate that at least one process prior improves the model on that process family.", 15),
    ]
    for j, family in enumerate(FAMILIES):
        label = f"{family['id']} {family['name']}"
        body.append(svg_text(margin_l + j * cell_w + cell_w / 2, margin_t - 28, label, 15, "700", "middle"))
    for i, model in enumerate(MODELS):
        y = margin_t + i * cell_h
        body.append(svg_text(margin_l - 14, y + 37, model, 15, "700", "end"))
        for j, family in enumerate(FAMILIES):
            gain = by_key.get((model, family["id"]), 0.0)
            x = margin_l + j * cell_w
            if gain >= 0:
                strength = min(gain / 40.0, 1.0)
                r = int(238 * (1 - strength) + 76 * strength)
                g = int(247 * (1 - strength) + 175 * strength)
                b = int(238 * (1 - strength) + 80 * strength)
            else:
                strength = min(abs(gain) / 50.0, 1.0)
                r = int(247 * (1 - strength) + 225 * strength)
                g = int(238 * (1 - strength) + 87 * strength)
                b = int(238 * (1 - strength) + 89 * strength)
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" fill="#{r:02x}{g:02x}{b:02x}" stroke="#fff"/>')
            body.append(svg_text(x + cell_w / 2, y + 36, fmt_pp(gain), 16, "700", "middle"))
    path = OUT_DIR / "srt_cross_family_best_gain_heatmap.svg"
    write_svg(path, width, height, body)


def plot_model_profiles(rows: list[dict[str, Any]]) -> None:
    width, height = 1260, 560
    margin_l, margin_r, margin_t, margin_b = 140, 170, 90, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_min, y_max = -50, 45

    def y_for(v: float) -> float:
        return margin_t + (y_max - v) / (y_max - y_min) * plot_h

    x_step = plot_w / (len(FAMILIES) - 1)
    body = [
        svg_text(40, 45, "Model-Sensitive SRT Profiles across Process Families", 28, "700"),
        svg_text(40, 72, "Each line shows best SRT gain over default. GLM repeatedly exhibits process-overload.", 15),
    ]
    for tick in [-40, -20, 0, 20, 40]:
        y = y_for(tick)
        body.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#E6E6E6"/>')
        body.append(svg_text(margin_l - 12, y + 5, f"{tick:+d}", 13, anchor="end"))
    zero_y = y_for(0)
    body.append(f'<line x1="{margin_l}" y1="{zero_y:.1f}" x2="{width-margin_r}" y2="{zero_y:.1f}" stroke="#222" stroke-width="1.2"/>')

    by_model: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append((row["family_id"], row["best_gain_pp"]))
    fam_index = {f["id"]: i for i, f in enumerate(FAMILIES)}
    for model in MODELS:
        points = sorted(by_model[model], key=lambda p: fam_index[p[0]])
        coords = [(margin_l + fam_index[fid] * x_step, y_for(gain), gain) for fid, gain in points]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coords)
        body.append(f'<polyline points="{poly}" fill="none" stroke="{COLORS[model]}" stroke-width="3"/>')
        for x, y, gain in coords:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[model]}"/>')
    legend_x, legend_y = width - 330, 105
    for i, model in enumerate(MODELS):
        y = legend_y + i * 22
        body.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{COLORS[model]}" stroke-width="3"/>')
        body.append(f'<circle cx="{legend_x + 12}" cy="{y}" r="4" fill="{COLORS[model]}"/>')
        body.append(svg_text(legend_x + 34, y + 5, model, 13, "700"))

    for j, family in enumerate(FAMILIES):
        x = margin_l + j * x_step
        body.append(svg_text(x, height - margin_b + 34, f"{family['id']} {family['name']}", 14, "700", "middle"))
        body.append(svg_text(x, height - margin_b + 54, family["name"], 13, anchor="middle"))
    path = OUT_DIR / "srt_cross_family_model_profiles.svg"
    write_svg(path, width, height, body)


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SRT Cross-Family Summary",
        "",
        "Families included: P4 tool-use, P5 cleaning, P6 craft, P7 mobility.",
        "",
        "## Best SRT Gain",
        "",
        "| Family | Model | Default | Best SRT | Best condition | Gain | Router | Router gain |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for model in MODELS:
            row = next(r for r in rows if r["family_id"] == family["id"] and r["model"] == model)
            lines.append(
                f"| {family['id']} {family['name']} | {model} | {fmt_pct(row['default'])} | "
                f"{fmt_pct(row['best_srt'])} | {row['best_srt_condition']} | {fmt_pp(row['best_gain_pp'])} | "
                f"{fmt_pct(row['router'])} | {fmt_pp(row['router_gain_pp'])} |"
            )
    lines += [
        "",
        "## Model-Level Pattern",
        "",
    ]
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        positive = sum(1 for r in model_rows if r["best_gain_pp"] > 0)
        avg_gain = sum(r["best_gain_pp"] for r in model_rows) / len(model_rows)
        lines.append(f"- **{model}**: positive on {positive}/{len(model_rows)} families, average best gain {avg_gain:+.1f} pp.")
    lines += [
        "",
        "## Files",
        "",
        "- `srt_cross_family_best_gain_heatmap.svg/pdf`",
        "- `srt_cross_family_model_profiles.svg/pdf`",
        "- `srt_cross_family_summary.csv`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT_DIR / "srt_cross_family_summary.csv", rows)
    write_markdown(rows)
    plot_gain_heatmap(rows)
    plot_model_profiles(rows)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
