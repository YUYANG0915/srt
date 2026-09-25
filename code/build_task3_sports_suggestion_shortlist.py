#!/usr/bin/env python3
"""Build a shortlist review file from OpenAI sports annotation suggestions."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUGGESTIONS_CSV = ROOT / "review" / "task3_sports_action_openai_suggestions_v1.csv"
OUT_CSV = ROOT / "review" / "task3_sports_action_shortlist_v1_review.csv"
OUT_HTML = ROOT / "review" / "task3_sports_action_shortlist_v1_review.html"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def confidence(row: dict[str, str]) -> float:
    try:
        return float(row.get("suggested_confidence") or 0)
    except ValueError:
        return 0.0


def build_rows() -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv(SUGGESTIONS_CSV)
        if row.get("suggested_keep") == "yes" and confidence(row) >= 0.75
    ]
    rows.sort(key=lambda row: (row.get("sport_hint", ""), -confidence(row)))
    output = []
    for row in rows:
        output.append(
            {
                "id": row["id"],
                "image_path": row["image_path"],
                "sport_hint": row.get("sport_hint", ""),
                "review_keep": row.get("suggested_keep", ""),
                "review_gold_action_phase": row.get("suggested_gold_action_phase", ""),
                "review_phase_boundary_family": row.get("suggested_phase_boundary_family", ""),
                "review_neighbor_phase": row.get("suggested_neighbor_phase", ""),
                "review_why_not_neighbor_phase": row.get("suggested_why_not_neighbor_phase", ""),
                "review_visible_actor_pose": row.get("suggested_visible_actor_pose", ""),
                "review_visible_ball_or_target": row.get("suggested_visible_ball_or_target", ""),
                "review_visible_motion_cues": row.get("suggested_visible_motion_cues", ""),
                "suggested_confidence": row.get("suggested_confidence", ""),
                "suggested_notes": row.get("suggested_notes", ""),
                "human_decision": "",
                "human_correction_notes": "",
            }
        )
    return output


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "sport_hint",
        "review_keep",
        "review_gold_action_phase",
        "review_phase_boundary_family",
        "review_neighbor_phase",
        "review_why_not_neighbor_phase",
        "review_visible_actor_pose",
        "review_visible_ball_or_target",
        "review_visible_motion_cues",
        "suggested_confidence",
        "suggested_notes",
        "human_decision",
        "human_correction_notes",
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
                <p><b>suggested phase:</b> {html.escape(row['review_gold_action_phase'])}</p>
                <p><b>suggested family:</b> {html.escape(row['review_phase_boundary_family'])}</p>
                <p><b>confidence:</b> {html.escape(row['suggested_confidence'])}</p>
                <p><b>notes:</b> {html.escape(row['suggested_notes'])}</p>
              </div>
            </div>
            """
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Task3 Sports Action Shortlist V1</title>
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
  <h1>Task3 Sports Action Shortlist V1</h1>
  <p>High-confidence OpenAI suggestions only. Review these first.</p>
  <p>Total: {len(rows)}</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    OUT_HTML.write_text(doc, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_html(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")
    print(f"shortlist_size={len(rows)}")


if __name__ == "__main__":
    main()
