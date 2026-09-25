#!/usr/bin/env python3
"""Build a focused correction review pack for P4 tool-use after first Qwen run."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def suggest_revision(row: dict[str, Any]) -> tuple[str, str]:
    captions = str(row.get("captions", "")).lower()
    default_pred = row.get("default_pred", "")
    srt_pred = row.get("srt_pred", "")
    current = row.get("current_gold", "")

    if any(term in captions for term in ["play a game", "playing a video game", "wii", "video game controller"]):
        return "active_tool_use", "game/controller use is an active operation if people are playing or gesturing with controllers"
    if any(term in captions for term in ["food truck", "plate", "dining table", "restaurant", "hamburger", "birds"]) and row.get("visible_tool") == "knife":
        return "result_or_finished", "food/dining scene with knife present is result/display, not preparing_tool"
    if any(term in captions for term in ["cat resting on a laptop", "cat laying", "cat is sit"]):
        return "result_or_finished", "laptop is occupied/resting surface; no human tool operation"
    if any(term in captions for term in ["holding a toothbrush", "showing the camera a toothbrush"]) and "brushing" not in captions:
        return "preparing_tool", "tool is held/shown but active brushing is not visible"
    if default_pred and default_pred == srt_pred:
        return str(default_pred), "both model conditions agree; verify visually"
    return str(current), "needs human visual correction"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "keep",
        "corrected_gold_label",
        "corrected_boundary_family",
        "why_not_neighbor",
        "issue_type",
        "current_gold",
        "default_pred",
        "srt_pred",
        "default_ok",
        "srt_ok",
        "suggested_revision",
        "revision_reason",
        "visible_tool",
        "visible_object",
        "captions",
        "categories",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for idx, row in enumerate(rows, start=1):
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(row['image_path'])}" alt="{html.escape(row['id'])}">
              <div class="body">
                <h3>{idx}. {html.escape(row['id'])}</h3>
                <p><b>current gold:</b> {html.escape(row['current_gold'])}</p>
                <p><b>default pred:</b> {html.escape(row['default_pred'])} ({html.escape(str(row['default_ok']))})</p>
                <p><b>SRT pred:</b> {html.escape(row['srt_pred'])} ({html.escape(str(row['srt_ok']))})</p>
                <p><b>suggested revision:</b> {html.escape(row['suggested_revision'])}</p>
                <p><b>reason:</b> {html.escape(row['revision_reason'])}</p>
                <p><b>tool-object:</b> {html.escape(row['visible_tool'])} -> {html.escape(row['visible_object'])}</p>
                <p><b>captions:</b> {html.escape(row['captions'])}</p>
                <p><b>categories:</b> {html.escape(row['categories'])}</p>
                <p class="path">{html.escape(row['image_path'])}</p>
              </div>
            </div>
            """
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>P4 Tool-use Focused Correction Review</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ border: 1.5px solid #222; border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ width: 100%; height: 280px; object-fit: contain; display: block; background: #f4f4f4; }}
    .body {{ padding: 14px; }}
    h3 {{ margin: 0 0 8px; font-size: 21px; }}
    p {{ font-size: 15px; line-height: 1.35; }}
    .path {{ color: #555; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>P4 Tool-use Focused Correction Review</h1>
  <p>Total cases: {len(rows)}. Correct only these rows before rerunning P4.</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/task3_tooluse_eval_v1_bootstrap52.jsonl")
    parser.add_argument("--case-table", default="runs/task3_tooluse_v1bootstrap52_qwen_summary_2026-06-29/task3_tooluse_case_table.csv")
    parser.add_argument("--output-csv", default="review/task3_tooluse_focused_correction_v1.csv")
    parser.add_argument("--output-html", default="review/task3_tooluse_focused_correction_v1.html")
    args = parser.parse_args()

    items = {row["id"]: row for row in load_jsonl(Path(args.items))}
    cases = read_csv(Path(args.case_table))
    grouped: dict[str, dict[str, str]] = {}
    for row in cases:
        grouped.setdefault(row["id"], {})[row["condition"]] = row

    rows: list[dict[str, Any]] = []
    for item_id, preds in sorted(grouped.items()):
        default = preds.get("default")
        srt = preds.get("generic_srt")
        if not default or not srt:
            continue
        if default["ok"] == "True" and srt["ok"] == "True":
            continue
        item = items[item_id]
        row = {
            "id": item_id,
            "image_path": item["image_path"],
            "keep": "yes",
            "corrected_gold_label": "",
            "corrected_boundary_family": item.get("boundary_family", ""),
            "why_not_neighbor": item.get("why_not_neighbor", ""),
            "issue_type": "",
            "current_gold": item.get("gold_label", ""),
            "default_pred": default.get("pred", ""),
            "srt_pred": srt.get("pred", ""),
            "default_ok": default.get("ok", ""),
            "srt_ok": srt.get("ok", ""),
            "visible_tool": item.get("visible_tool", ""),
            "visible_object": item.get("visible_object", ""),
            "captions": item.get("captions", ""),
            "categories": item.get("categories", ""),
        }
        revision, reason = suggest_revision(row)
        row["suggested_revision"] = revision
        row["revision_reason"] = reason
        rows.append(row)

    write_csv(Path(args.output_csv), rows)
    write_html(Path(args.output_html), rows)
    print(json.dumps({"rows": len(rows), "output_csv": args.output_csv, "output_html": args.output_html}, indent=2))


if __name__ == "__main__":
    main()
