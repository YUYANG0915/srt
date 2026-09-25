#!/usr/bin/env python3
"""Prepare a focused review pack for sports-action smoke-test disagreements."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "review" / "task3_sports_action_candidates_v1_review.csv"
OUT_CSV = ROOT / "review" / "task3_sports_action_focused_review_v1.csv"
OUT_HTML = ROOT / "review" / "task3_sports_action_focused_review_v1.html"

REVIEW_NOTES = {
    "sports_v1_0001_214539": {
        "model_pattern": "all three conditions predicted active_execution",
        "review_question": "Is the player visibly mid-kick / interacting with the ball, or only preparing?",
        "suggested_recheck": "active_execution if foot/body is already in the kicking action; otherwise pre_action_setup",
    },
    "sports_v1_0002_261097": {
        "model_pattern": "all three conditions predicted resting_or_non_action",
        "review_question": "Is there visible follow-through evidence, or only a standing tennis player?",
        "suggested_recheck": "keep follow_through_or_result only if swing/result is visible; otherwise resting_or_non_action or reject",
    },
    "sports_v1_0004_448263": {
        "model_pattern": "all three conditions predicted pre_action_setup",
        "review_question": "Is this coaching/preparation before play, or truly non-action/rest?",
        "suggested_recheck": "pre_action_setup if instruction/pre-game preparation is visible; resting_or_non_action if no action phase is supported",
    },
    "sports_v1_0009_521259": {
        "model_pattern": "models predicted resting_or_non_action or pre_action_setup",
        "review_question": "Is an active throw/catch visibly happening, or are we inferring from the frisbee being in the scene?",
        "suggested_recheck": "active_execution only if throw/catch pose is visible; follow_through_or_result if release/result is visible; otherwise reject or pre_action_setup",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "sport_hint",
        "current_keep",
        "current_gold_action_phase",
        "current_phase_boundary_family",
        "current_neighbor_phase",
        "current_why_not_neighbor_phase",
        "model_pattern",
        "review_question",
        "suggested_recheck",
        "revised_keep",
        "revised_gold_action_phase",
        "revised_phase_boundary_family",
        "revised_neighbor_phase",
        "revised_why_not_neighbor_phase",
        "revised_notes",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(row['image_path'])}" alt="{html.escape(row['id'])}">
              <div class="meta">
                <h2>{html.escape(row['id'])}</h2>
                <p><b>sport:</b> {html.escape(row['sport_hint'])}</p>
                <p><b>current gold:</b> {html.escape(row['current_gold_action_phase'])}</p>
                <p><b>current family:</b> {html.escape(row['current_phase_boundary_family'])}</p>
                <p><b>model pattern:</b> {html.escape(row['model_pattern'])}</p>
                <p><b>review question:</b> {html.escape(row['review_question'])}</p>
                <p><b>suggested recheck:</b> {html.escape(row['suggested_recheck'])}</p>
              </div>
            </div>
            """
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Task3 Sports Focused Review V1</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #bbb; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: 320px; object-fit: contain; background: #f7f7f7; }}
    .meta {{ font-size: 14px; line-height: 1.35; }}
    h2 {{ font-size: 18px; margin-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>Task3 Sports Focused Review V1</h1>
  <p>Only the 4 smoke-test disagreement cases are included.</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    OUT_HTML.write_text(doc, encoding="utf-8")


def main() -> None:
    source_rows = {row["id"]: row for row in read_csv(SOURCE_CSV)}
    output_rows: list[dict[str, str]] = []
    for item_id, note in REVIEW_NOTES.items():
        row = source_rows[item_id]
        output_rows.append(
            {
                "id": item_id,
                "image_path": row.get("image_path", ""),
                "sport_hint": row.get("sport_hint", ""),
                "current_keep": row.get("keep", ""),
                "current_gold_action_phase": row.get("gold_action_phase", ""),
                "current_phase_boundary_family": row.get("phase_boundary_family", ""),
                "current_neighbor_phase": row.get("neighbor_phase", ""),
                "current_why_not_neighbor_phase": row.get("why_not_neighbor_phase", ""),
                "model_pattern": note["model_pattern"],
                "review_question": note["review_question"],
                "suggested_recheck": note["suggested_recheck"],
                "revised_keep": "",
                "revised_gold_action_phase": "",
                "revised_phase_boundary_family": "",
                "revised_neighbor_phase": "",
                "revised_why_not_neighbor_phase": "",
                "revised_notes": "",
            }
        )
    write_csv(output_rows)
    write_html(output_rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
