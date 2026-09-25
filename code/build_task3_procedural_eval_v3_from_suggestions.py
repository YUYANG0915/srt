#!/usr/bin/env python3
"""Build Task3 procedural eval v3 from human eval v2 plus OpenRouter suggestions."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def confidence_value(value: str) -> float:
    try:
        conf = float(value or 0)
    except ValueError:
        return 0.0
    if conf > 1.0:
        conf = conf / 10.0 if conf <= 10.0 else 1.0
    return conf


def build_item(row: dict[str, str]) -> dict:
    label = row["suggested_gold_step_label"].strip()
    family = row["suggested_boundary_family"].strip()
    question = "What step of the process is happening in this scene?"
    return {
        "id": row["id"],
        "image_id": row.get("source_image_id", "").strip() or row["id"].split("_")[-1],
        "source": "procedural_openrouter_suggestions_v2",
        "source_dataset": "coco_val2017_local_subset",
        "task": "task3_procedural_eval_v3_auto",
        "scenario_domain": "cooking",
        "activity_name": "cooking_or_food_preparation",
        "underspecified_question": question,
        "original_question": question,
        "question": question,
        "gold_step_label": label,
        "expected_answer": label,
        "answer": label,
        "expected_policy": "answer",
        "boundary_family": family,
        "boundary_family_hint": family,
        "neighbor_label": row.get("suggested_neighbor_label", "").strip(),
        "why_not_neighbor_label": row.get("suggested_why_not_neighbor_label", "").strip(),
        "visible_actor": row.get("suggested_visible_actor", "").strip(),
        "visible_tool": row.get("suggested_visible_tool", "").strip(),
        "visible_object": row.get("suggested_visible_object", "").strip(),
        "visible_state_cues": row.get("suggested_visible_state_cues", "").strip(),
        "suggested_confidence": f"{confidence_value(row.get('suggested_confidence', '')):.3f}",
        "issue_type": "openrouter_assisted_high_confidence",
        "notes": row.get("suggested_notes", "").strip(),
        "image_path": row.get("image_path", "").strip(),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_review_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "image_id",
        "image_path",
        "auto_label_ok",
        "gold_step_label",
        "boundary_family",
        "suggested_confidence",
        "neighbor_label",
        "why_not_neighbor_label",
        "visible_actor",
        "visible_tool",
        "visible_object",
        "visible_state_cues",
        "issue_type",
        "human_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "image_id": row.get("image_id", ""),
                    "image_path": row.get("image_path", ""),
                    "auto_label_ok": "",
                    "gold_step_label": row.get("gold_step_label", ""),
                    "boundary_family": row.get("boundary_family", ""),
                    "suggested_confidence": row.get("suggested_confidence", ""),
                    "neighbor_label": row.get("neighbor_label", ""),
                    "why_not_neighbor_label": row.get("why_not_neighbor_label", ""),
                    "visible_actor": row.get("visible_actor", ""),
                    "visible_tool": row.get("visible_tool", ""),
                    "visible_object": row.get("visible_object", ""),
                    "visible_state_cues": row.get("visible_state_cues", ""),
                    "issue_type": row.get("issue_type", ""),
                    "human_notes": "",
                }
            )


def write_review_html(path: Path, rows: list[dict]) -> None:
    cards = []
    for idx, row in enumerate(rows, start=1):
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(row.get('image_path', ''))}" alt="{html.escape(row.get('id', ''))}">
              <div class="body">
                <h3>{idx}. {html.escape(row.get('id', ''))}</h3>
                <p><b>label:</b> {html.escape(row.get('gold_step_label', ''))}</p>
                <p><b>family:</b> {html.escape(row.get('boundary_family', ''))}</p>
                <p><b>confidence:</b> {html.escape(row.get('suggested_confidence', ''))}</p>
                <p><b>why not neighbor:</b> {html.escape(row.get('why_not_neighbor_label', ''))}</p>
                <p><b>state cues:</b> {html.escape(row.get('visible_state_cues', ''))}</p>
                <p><b>notes:</b> {html.escape(row.get('notes', ''))}</p>
                <p class="path">{html.escape(row.get('image_path', ''))}</p>
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
  <title>Task3 Procedural Eval V3 Auto Review</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 18px; }}
    .card {{ border: 1.5px solid #222; border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ width: 100%; height: 270px; object-fit: contain; display: block; background: #f4f4f4; }}
    .body {{ padding: 14px; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ font-size: 15px; line-height: 1.35; }}
    .path {{ color: #555; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Task3 Procedural Eval V3 Auto Review</h1>
  <p>Total auto-suggested additions: {len(rows)}</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", default="data/task3_procedural_eval_v2.jsonl")
    parser.add_argument("--suggestions-csv", default="review/task3_procedural_cooking_openrouter_suggestions_v2.csv")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--output-jsonl", default="data/task3_procedural_eval_v3_auto58.jsonl")
    parser.add_argument("--output-review-csv", default="review/task3_procedural_eval_v3_auto58_spotcheck.csv")
    parser.add_argument("--output-review-html", default="review/task3_procedural_eval_v3_auto58_spotcheck.html")
    parser.add_argument("--output-summary", default="runs/task3_procedural_eval_v3_auto58_build_summary_2026-06-28.json")
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT / args.base_jsonl)
    seen = {str(row.get("image_id")) for row in base_rows}

    added_rows: list[dict] = []
    skipped_rows: list[dict[str, str]] = []
    for row in read_csv(ROOT / args.suggestions_csv):
        conf = confidence_value(row.get("suggested_confidence", ""))
        keep = row.get("suggested_keep", "").strip().lower()
        label = row.get("suggested_gold_step_label", "").strip()
        family = row.get("suggested_boundary_family", "").strip()
        image_id = row.get("source_image_id", "").strip() or row.get("id", "").split("_")[-1]
        if keep == "yes" and label and family and conf >= args.min_confidence and image_id not in seen:
            item = build_item(row)
            added_rows.append(item)
            seen.add(image_id)
        else:
            skipped_rows.append(row)

    merged_rows = base_rows + added_rows
    write_jsonl(ROOT / args.output_jsonl, merged_rows)
    write_review_csv(ROOT / args.output_review_csv, added_rows)
    write_review_html(ROOT / args.output_review_html, added_rows)

    summary = {
        "base_size": len(base_rows),
        "added_size": len(added_rows),
        "merged_size": len(merged_rows),
        "skipped_size": len(skipped_rows),
        "min_confidence": args.min_confidence,
        "output_jsonl": str(ROOT / args.output_jsonl),
        "review_csv": str(ROOT / args.output_review_csv),
        "review_html": str(ROOT / args.output_review_html),
        "label_distribution": {},
        "family_distribution": {},
    }
    for row in merged_rows:
        summary["label_distribution"][row.get("gold_step_label", "")] = summary["label_distribution"].get(row.get("gold_step_label", ""), 0) + 1
        summary["family_distribution"][row.get("boundary_family", "")] = summary["family_distribution"].get(row.get("boundary_family", ""), 0) + 1
    out_summary = ROOT / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
