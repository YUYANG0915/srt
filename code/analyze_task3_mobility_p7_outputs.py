#!/usr/bin/env python3
"""Create P7 mobility figures and representative case reports."""

from __future__ import annotations

import csv
import html
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "runs/task3_mobility_v1reviewed106_5model_5cond_summary_2026-07-02"
CASE_TABLE = SUMMARY_DIR / "task3_mobility_case_table.csv"
ITEMS = ROOT / "data/task3_mobility_eval_v1_reviewed.jsonl"
OUT_DIR = ROOT / "runs/task3_mobility_v1reviewed106_5model_figures_cases_2026-07-02"

MODELS = [
    "GLM-4.6V",
    "GPT-5.4-mini",
    "Gemini-3.1-Pro",
    "Mistral-Small-3.2",
    "Qwen2.5-VL-72B",
]
CONDITIONS = ["default", "generic_srt", "scenario_srt", "scenario_srt_v2", "boundary_router"]
CONDITION_LABELS = {
    "default": "Default",
    "generic_srt": "Generic",
    "scenario_srt": "Scenario",
    "scenario_srt_v2": "Scenario v2",
    "boundary_router": "Router",
}
COLORS = {
    "default": "#767676",
    "generic_srt": "#4C78A8",
    "scenario_srt": "#59A14F",
    "scenario_srt_v2": "#F28E2B",
    "boundary_router": "#B07AA1",
}


def load_items() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with ITEMS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def svg_text(x: float, y: float, text: str, size: int = 16, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Georgia, Times New Roman, serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#222">{esc(text)}</text>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def convert_svg_to_pdf(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return


def pct_text(value: float) -> str:
    return f"{value * 100:.1f}%"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["model", "condition"], as_index=False)["ok"].agg(["sum", "count"])
    grouped["accuracy"] = grouped["sum"] / grouped["count"]
    return grouped.rename(columns={"sum": "correct", "count": "total"})


def plot_overall(summary: pd.DataFrame) -> None:
    width, height = 1250, 640
    margin_l, margin_r, margin_t, margin_b = 130, 40, 90, 95
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_max = 0.82
    group_w = plot_w / len(MODELS)
    bar_w = group_w / 7.0
    body: list[str] = [
        svg_text(40, 45, "P7 Mobility / Transport Process: Accuracy by Model and SRT Condition", 28, "700"),
        svg_text(40, 72, "Boundary-aware SRT improves Qwen, Mistral, and Gemini; GLM exposes process-overload.", 15),
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = margin_t + plot_h - tick / y_max * plot_h
        body.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#E6E6E6"/>')
        body.append(svg_text(margin_l - 12, y + 5, f"{int(tick*100)}", 13, anchor="end"))
    body.append(svg_text(42, margin_t + plot_h / 2, "Accuracy (%)", 15, "700"))

    lookup = {(r.model, r.condition): r.accuracy for r in summary.itertuples()}
    for mi, model in enumerate(MODELS):
        gx = margin_l + mi * group_w
        for ci, condition in enumerate(CONDITIONS):
            acc = lookup.get((model, condition), 0.0)
            x = gx + group_w * 0.18 + ci * bar_w
            h = acc / y_max * plot_h
            y = margin_t + plot_h - h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" '
                f'fill="{COLORS[condition]}" rx="2"/>'
            )
            if condition in ("default", "boundary_router"):
                body.append(svg_text(x + bar_w * 0.41, y - 6, pct_text(acc), 11, "700", "middle"))
        body.append(svg_text(gx + group_w / 2, margin_t + plot_h + 34, model, 15, "700", "middle"))

    legend_x, legend_y = margin_l, height - 38
    for ci, condition in enumerate(CONDITIONS):
        x = legend_x + ci * 180
        body.append(f'<rect x="{x}" y="{legend_y-13}" width="15" height="15" fill="{COLORS[condition]}" rx="2"/>')
        body.append(svg_text(x + 23, legend_y, CONDITION_LABELS[condition], 14))

    path = OUT_DIR / "p7_mobility_overall_accuracy.svg"
    write_svg(path, width, height, body)
    convert_svg_to_pdf(path)


def plot_gain(summary: pd.DataFrame) -> None:
    width, height = 980, 480
    margin_l, margin_r, margin_t, margin_b = 160, 70, 80, 70
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    lookup = {(r.model, r.condition): r.accuracy for r in summary.itertuples()}
    gains = [(m, lookup[(m, "boundary_router")] - lookup[(m, "default")]) for m in MODELS]
    min_gain, max_gain = -0.50, 0.15

    def x_for(v: float) -> float:
        return margin_l + (v - min_gain) / (max_gain - min_gain) * plot_w

    body = [
        svg_text(40, 45, "P7 Mobility: Boundary-Router Gain over Default", 27, "700"),
        svg_text(40, 72, "Negative values are diagnostic: the process prior can overload a model instead of helping it.", 15),
    ]
    zero_x = x_for(0)
    body.append(f'<line x1="{zero_x:.1f}" y1="{margin_t}" x2="{zero_x:.1f}" y2="{height-margin_b}" stroke="#222" stroke-width="1.2"/>')
    for tick in [-0.4, -0.2, 0.0, 0.1]:
        x = x_for(tick)
        body.append(f'<line x1="{x:.1f}" y1="{height-margin_b}" x2="{x:.1f}" y2="{height-margin_b+7}" stroke="#222"/>')
        body.append(svg_text(x, height - margin_b + 26, f"{tick*100:+.0f} pp", 13, anchor="middle"))
    row_h = plot_h / len(gains)
    for i, (model, gain) in enumerate(gains):
        y = margin_t + i * row_h + row_h * 0.55
        body.append(svg_text(margin_l - 14, y + 5, model, 15, "700", "end"))
        x0, x1 = zero_x, x_for(gain)
        color = "#4C78A8" if gain >= 0 else "#E15759"
        body.append(
            f'<rect x="{min(x0,x1):.1f}" y="{y-12:.1f}" width="{abs(x1-x0):.1f}" height="24" '
            f'fill="{color}" rx="2"/>'
        )
        if gain >= 0:
            body.append(svg_text(x1 + 8, y + 5, f"{gain*100:+.1f} pp", 14, "700", "start"))
        else:
            body.append(svg_text(x1 + 10, y + 5, f"{gain*100:+.1f} pp", 14, "700", "start"))
    path = OUT_DIR / "p7_mobility_router_gain.svg"
    write_svg(path, width, height, body)
    convert_svg_to_pdf(path)


def plot_boundary_heatmap(df: pd.DataFrame) -> None:
    width, height = 1180, 650
    margin_l, margin_t = 190, 110
    cell_w, cell_h = 245, 54
    boundaries = ["active_vs_parked", "mobility_vs_salience", "setup_vs_active"]
    boundary_labels = {
        "active_vs_parked": "Active vs. parked",
        "mobility_vs_salience": "Mobility vs. salience",
        "setup_vs_active": "Setup vs. active",
    }
    rows = [(m, c) for m in MODELS for c in ["default", "boundary_router"]]
    grouped = df.groupby(["model", "condition", "boundary_family"])["ok"].mean().to_dict()

    def color(v: float) -> str:
        # Red -> white -> green
        v = max(0.0, min(1.0, v))
        if v < 0.5:
            t = v / 0.5
            r, g, b = 224, int(224 * t + 247 * (1 - t)), int(224 * t + 247 * (1 - t))
        else:
            t = (v - 0.5) / 0.5
            r, g, b = int(247 * (1 - t) + 76 * t), int(247 * (1 - t) + 175 * t), int(247 * (1 - t) + 80 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    body = [
        svg_text(40, 45, "P7 Mobility: Boundary-Level Accuracy", 27, "700"),
        svg_text(40, 72, "Default vs. boundary-router, split by process decision boundary.", 15),
    ]
    for j, boundary in enumerate(boundaries):
        body.append(svg_text(margin_l + j * cell_w + cell_w / 2, margin_t - 25, boundary_labels[boundary], 14, "700", "middle"))
    for i, (model, condition) in enumerate(rows):
        y = margin_t + i * cell_h
        label = f"{model} / {CONDITION_LABELS[condition]}"
        body.append(svg_text(margin_l - 14, y + 33, label, 13, "700" if condition == "default" else "400", "end"))
        for j, boundary in enumerate(boundaries):
            v = float(grouped.get((model, condition, boundary), 0.0))
            x = margin_l + j * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" fill="{color(v)}" stroke="#ffffff"/>')
            body.append(svg_text(x + cell_w / 2, y + 33, pct_text(v), 14, "700", "middle"))
    path = OUT_DIR / "p7_mobility_boundary_heatmap.svg"
    write_svg(path, width, height, body)
    convert_svg_to_pdf(path)


def raw_snippet(text: str, limit: int = 420) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def select_cases(df: pd.DataFrame, items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_model_id: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in df.to_dict("records"):
        by_model_id[(row["model"], row["id"])][row["condition"]] = row

    specs = [
        ("router_fix_qwen", "Qwen2.5-VL-72B", lambda d: not d["default"]["ok"] and d["boundary_router"]["ok"], 8),
        ("router_fix_mistral", "Mistral-Small-3.2", lambda d: not d["default"]["ok"] and d["boundary_router"]["ok"], 8),
        ("router_fix_gemini", "Gemini-3.1-Pro", lambda d: not d["default"]["ok"] and d["boundary_router"]["ok"], 6),
        ("gpt_boundary_tradeoff", "GPT-5.4-mini", lambda d: d["default"]["ok"] and not d["boundary_router"]["ok"], 8),
        ("glm_process_overload", "GLM-4.6V", lambda d: d["default"]["ok"] and not d["boundary_router"]["ok"], 10),
    ]
    seen: set[tuple[str, str, str]] = set()
    for category, model, pred, limit in specs:
        rows = []
        for (m, item_id), conds in by_model_id.items():
            if m != model or not all(c in conds for c in CONDITIONS):
                continue
            if pred(conds):
                item = items[item_id]
                rows.append((item.get("boundary_family", ""), item.get("subfamily", ""), item_id, conds))
        for rank, (_, _, item_id, conds) in enumerate(sorted(rows)[:limit], start=1):
            item = items[item_id]
            key = (category, model, item_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "case_category": category,
                    "rank": rank,
                    "model": model,
                    "id": item_id,
                    "image_id": item.get("image_id"),
                    "image_path": item.get("image_path"),
                    "subfamily": item.get("subfamily"),
                    "boundary_family": item.get("boundary_family"),
                    "gold": item.get("gold_label"),
                    "visible_actor": item.get("visible_actor"),
                    "visible_vehicle": item.get("visible_vehicle"),
                    "visible_relation": item.get("visible_relation"),
                    "visible_state_cues": item.get("visible_state_cues"),
                    "default_pred": conds["default"]["pred"],
                    "router_pred": conds["boundary_router"]["pred"],
                    "generic_pred": conds["generic_srt"]["pred"],
                    "scenario_pred": conds["scenario_srt"]["pred"],
                    "scenario_v2_pred": conds["scenario_srt_v2"]["pred"],
                    "default_raw": raw_snippet(conds["default"]["raw_text"]),
                    "router_raw": raw_snippet(conds["boundary_router"]["raw_text"]),
                }
            )
    return selected


def write_cases(cases: list[dict[str, Any]]) -> None:
    csv_path = OUT_DIR / "p7_mobility_representative_cases.csv"
    jsonl_path = OUT_DIR / "p7_mobility_representative_cases.jsonl"
    if cases:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
            writer.writeheader()
            writer.writerows(cases)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in cases:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["case_category"]].append(case)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>P7 Mobility Representative Cases</title>",
        "<style>",
        "body{font-family:Georgia,'Times New Roman',serif;margin:32px;color:#222;line-height:1.42}",
        "h1{font-size:34px;margin-bottom:4px} h2{margin-top:34px;border-top:2px solid #222;padding-top:14px}",
        ".case{display:grid;grid-template-columns:260px 1fr;gap:18px;margin:22px 0 32px;page-break-inside:avoid}",
        "img{max-width:260px;max-height:210px;border:1px solid #ddd}",
        ".meta{font-size:14px;color:#444}.pred{display:grid;grid-template-columns:150px 1fr;gap:4px 12px;margin-top:8px}",
        "code{font-family:Menlo,monospace;font-size:12px;background:#f6f6f6;padding:2px 4px}",
        ".raw{font-size:13px;background:#fafafa;border-left:3px solid #aaa;padding:8px;margin-top:8px}",
        "</style></head><body>",
        "<h1>P7 Mobility / Transport Representative Cases</h1>",
        "<p>Selected automatically from the five-model, five-condition P7 case table.</p>",
    ]
    for category, rows in grouped.items():
        parts.append(f"<h2>{esc(category)}</h2>")
        for case in rows:
            parts.append("<div class='case'>")
            parts.append(f"<div><img src='{esc(case['image_path'])}'><div class='meta'>{esc(case['id'])}</div></div>")
            parts.append("<div>")
            parts.append(f"<h3>{esc(case['model'])}: {esc(case['gold'])}</h3>")
            parts.append(
                f"<div class='meta'>Boundary: <code>{esc(case['boundary_family'])}</code> | "
                f"Subfamily: <code>{esc(case['subfamily'])}</code></div>"
            )
            parts.append(
                f"<div class='meta'>Actor: {esc(case['visible_actor'])}; Vehicle: {esc(case['visible_vehicle'])}; "
                f"Relation: {esc(case['visible_relation'])}</div>"
            )
            parts.append(f"<div class='meta'>Cues: {esc(case['visible_state_cues'])}</div>")
            parts.append("<div class='pred'>")
            for label in ["default_pred", "generic_pred", "scenario_pred", "scenario_v2_pred", "router_pred"]:
                parts.append(f"<b>{esc(label.replace('_pred',''))}</b><code>{esc(case[label])}</code>")
            parts.append("</div>")
            parts.append(f"<div class='raw'><b>Default:</b> {esc(case['default_raw'])}</div>")
            parts.append(f"<div class='raw'><b>Router:</b> {esc(case['router_raw'])}</div>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    (OUT_DIR / "p7_mobility_case_report.html").write_text("\n".join(parts), encoding="utf-8")


def write_index(summary: pd.DataFrame, cases: list[dict[str, Any]]) -> None:
    lookup = {(r.model, r.condition): r.accuracy for r in summary.itertuples()}
    lines = [
        "# P7 Mobility Figures and Cases",
        "",
        "## Files",
        "",
        "- `p7_mobility_overall_accuracy.svg/pdf`",
        "- `p7_mobility_router_gain.svg/pdf`",
        "- `p7_mobility_boundary_heatmap.svg/pdf`",
        "- `p7_mobility_all_figures.svg/pdf`",
        "- `p7_mobility_representative_cases.csv/jsonl`",
        "- `p7_mobility_case_report.html`",
        "",
        "## Overall",
        "",
        "| Model | Default | Router | Delta |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        default = lookup[(model, "default")]
        router = lookup[(model, "boundary_router")]
        lines.append(f"| {model} | {pct_text(default)} | {pct_text(router)} | {(router-default)*100:+.1f} pp |")
    lines += [
        "",
        "## Case Counts",
        "",
    ]
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case["case_category"]] += 1
    for key, count in sorted(counts.items()):
        lines.append(f"- `{key}`: {count}")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_combined_figure() -> None:
    width, height = 1250, 1760
    panels = [
        ("p7_mobility_overall_accuracy.svg", 0, 0),
        ("p7_mobility_router_gain.svg", 110, 650),
        ("p7_mobility_boundary_heatmap.svg", 35, 1120),
    ]
    body = []
    for filename, x, y in panels:
        src = (OUT_DIR / filename).read_text(encoding="utf-8").splitlines()
        inner = "\n".join(src[1:-1])
        body.append(f'<g transform="translate({x},{y})">{inner}</g>')
    path = OUT_DIR / "p7_mobility_all_figures.svg"
    write_svg(path, width, height, body)
    convert_svg_to_pdf(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CASE_TABLE)
    items = load_items()
    summary = summarize(df)
    plot_overall(summary)
    plot_gain(summary)
    plot_boundary_heatmap(df)
    cases = select_cases(df, items)
    write_cases(cases)
    write_index(summary, cases)
    write_combined_figure()
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
