#!/usr/bin/env python3
"""Prepare a compact human spot-check pack for sports held-out items."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "task3_sports_action_eval_v3_heldout31.jsonl"
RUNS = {
    "default": ROOT / "runs" / "task3_sports_v3heldout31_gpt54mini_default.jsonl",
    "generic_srt": ROOT / "runs" / "task3_sports_v3heldout31_gpt54mini_generic_srt.jsonl",
    "phase_prior": ROOT / "runs" / "task3_sports_v3heldout31_gpt54mini_phase_prior.jsonl",
    "visible_boundary_srt": ROOT / "runs" / "task3_sports_v3heldout31_gpt54mini_visible_boundary_srt_rerun.jsonl",
    "boundary_verify": ROOT / "runs" / "task3_sports_v3heldout31_gpt54mini_boundary_verify.jsonl",
}
OUT_CSV = ROOT / "review" / "task3_sports_heldout31_spotcheck_2026-06-26.csv"
OUT_HTML = ROOT / "review" / "task3_sports_heldout31_spotcheck_2026-06-26.html"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def parse_phase(row: dict[str, Any]) -> tuple[str, str]:
    raw_text = str(row.get("raw_text", ""))
    parsed = parse_jsonish(raw_text)
    phase = parsed.get("action_phase") or parsed.get("phase") or parsed.get("answer") or ""
    rationale = parsed.get("rationale") or parsed.get("boundary_check") or raw_text[:400]
    return str(phase), str(rationale)


def main() -> None:
    items = load_jsonl(ITEMS)
    runs = {mode: {row["id"]: row for row in load_jsonl(path)} for mode, path in RUNS.items()}
    rows: list[dict[str, str]] = []
    for item in items:
        item_id = item["id"]
        preds: dict[str, str] = {}
        rationales: dict[str, str] = {}
        for mode, run_rows in runs.items():
            pred, rationale = parse_phase(run_rows[item_id])
            preds[mode] = pred
            rationales[mode] = rationale
        default_ok = preds["default"] == item["expected_answer"]
        generic_ok = preds["generic_srt"] == item["expected_answer"]
        if generic_ok and not default_ok:
            contrast = "generic_fixes_default"
        elif default_ok and not generic_ok:
            contrast = "generic_hurts_default"
        elif generic_ok and default_ok:
            contrast = "both_correct"
        else:
            contrast = "both_wrong"
        rows.append(
            {
                "id": item_id,
                "image_path": item.get("image_path", ""),
                "sport_hint": item.get("sport_hint", ""),
                "auto_phase": item.get("expected_answer", ""),
                "auto_family": item.get("phase_boundary_family", ""),
                "auto_confidence": item.get("suggested_confidence", ""),
                "contrast": contrast,
                "default_pred": preds["default"],
                "generic_srt_pred": preds["generic_srt"],
                "phase_prior_pred": preds["phase_prior"],
                "boundary_verify_pred": preds["boundary_verify"],
                "visible_boundary_pred": preds["visible_boundary_srt"],
                "default_rationale": rationales["default"],
                "generic_srt_rationale": rationales["generic_srt"],
                "visible_cues": item.get("visible_motion_cues", ""),
                "why_not_neighbor_phase": item.get("why_not_neighbor_phase", ""),
                "auto_label_ok": "",
                "revised_phase": "",
                "revised_family": "",
                "issue_type": "",
                "human_notes": "",
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cards: list[str] = []
    for row in rows:
        cards.append(
            f"""
            <section class="card {html.escape(row['contrast'])}">
              <img src="file://{html.escape(row['image_path'])}" alt="{html.escape(row['id'])}">
              <div>
                <h2>{html.escape(row['id'])} <span>{html.escape(row['contrast'])}</span></h2>
                <p><b>sport:</b> {html.escape(row['sport_hint'])}</p>
                <p><b>auto label:</b> {html.escape(row['auto_phase'])} / {html.escape(row['auto_family'])} / conf {html.escape(row['auto_confidence'])}</p>
                <p><b>default:</b> {html.escape(row['default_pred'])}</p>
                <p><b>generic SRT:</b> {html.escape(row['generic_srt_pred'])}</p>
                <p><b>phase prior:</b> {html.escape(row['phase_prior_pred'])}; <b>boundary verify:</b> {html.escape(row['boundary_verify_pred'])}; <b>visible boundary:</b> {html.escape(row['visible_boundary_pred'])}</p>
                <p><b>generic rationale:</b> {html.escape(row['generic_srt_rationale'][:450])}</p>
                <p><b>visible cues:</b> {html.escape(row['visible_cues'][:300])}</p>
                <p class="path">{html.escape(row['image_path'])}</p>
              </div>
            </section>
            """
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sports Held-out31 Spot-check</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; color: #1e1e1e; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ color: #555; margin-bottom: 20px; }}
    .card {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; border: 1px solid #bbb; border-left: 7px solid #999; border-radius: 6px; padding: 12px; margin-bottom: 16px; }}
    .generic_fixes_default {{ border-left-color: #178a4f; }}
    .generic_hurts_default {{ border-left-color: #b43b3b; }}
    .both_correct {{ border-left-color: #4c6fa8; }}
    .both_wrong {{ border-left-color: #8a6f17; }}
    img {{ width: 320px; height: 240px; object-fit: contain; background: #f6f6f6; }}
    h2 {{ font-size: 18px; margin: 0 0 8px; }}
    h2 span {{ font-size: 13px; font-weight: normal; color: #555; }}
    p {{ margin: 6px 0; line-height: 1.35; }}
    .path {{ color: #777; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Sports Held-out31 Spot-check</h1>
  <p class="summary">Green means generic SRT fixes default; red means generic SRT hurts default.</p>
  {''.join(cards)}
</body>
</html>
"""
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
