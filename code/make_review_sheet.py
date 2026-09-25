#!/usr/bin/env python3
"""Create HTML/CSV review sheets for manual pilot cleaning."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_runs(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            out[(str(row.get("id")), str(row.get("mode")))] = row
    return out


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = text.strip()
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


def summarize_run(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "policy": "",
            "answer": "",
            "confidence": "",
            "candidates": "",
            "candidate_count": "",
            "rationale": "",
            "raw_text": "",
        }
    parsed = parse_jsonish(str(row.get("raw_text", "")))
    candidates = parsed.get("relevant_candidates")
    candidate_text = ""
    if isinstance(candidates, list):
        parts = []
        for candidate in candidates[:5]:
            if isinstance(candidate, dict):
                name = str(candidate.get("name", ""))
                status = str(candidate.get("target_status", ""))
                pos = str(candidate.get("position", ""))
                parts.append(f"{name} [{status}] {pos}".strip())
        if len(candidates) > 5:
            parts.append(f"... +{len(candidates) - 5} more")
        candidate_text = " | ".join(parts)
    return {
        "policy": str(parsed.get("policy", "")),
        "answer": str(parsed.get("answer", "")),
        "confidence": str(parsed.get("confidence", "")),
        "candidates": candidate_text,
        "candidate_count": len(candidates) if isinstance(candidates, list) else "",
        "rationale": str(parsed.get("rationale", "")),
        "raw_text": str(row.get("raw_text", "")),
    }


def h(text: Any) -> str:
    return html.escape(str(text), quote=True)


def write_csv(path: Path, rows: list[dict[str, Any]], runs: dict[tuple[str, str], dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "target_object",
        "question",
        "expected_policy",
        "image_path",
        "auto_candidate_count",
        "default_policy",
        "default_answer",
        "evidence_policy",
        "evidence_candidate_count",
        "evidence_answer",
        "keep",
        "human_candidate_count",
        "human_expected_policy",
        "issue_type",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            default = summarize_run(runs.get((str(row["id"]), "default")))
            evidence = summarize_run(runs.get((str(row["id"]), "evidence")))
            writer.writerow(
                {
                    "id": row["id"],
                    "target_object": row.get("target_object", ""),
                    "question": row.get("underspecified_question", ""),
                    "expected_policy": row.get("expected_policy", ""),
                    "image_path": row.get("image_path", ""),
                    "auto_candidate_count": row.get("candidate_count_scene_graph", ""),
                    "default_policy": default["policy"],
                    "default_answer": default["answer"],
                    "evidence_policy": evidence["policy"],
                    "evidence_candidate_count": evidence["candidate_count"],
                    "evidence_answer": evidence["answer"],
                    "keep": "",
                    "human_candidate_count": "",
                    "human_expected_policy": "",
                    "issue_type": "",
                    "notes": "",
                }
            )


def write_html(path: Path, title: str, rows: list[dict[str, Any]], runs: dict[tuple[str, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for idx, row in enumerate(rows, start=1):
        default = summarize_run(runs.get((str(row["id"]), "default")))
        evidence = summarize_run(runs.get((str(row["id"]), "evidence")))
        image_path = Path(str(row.get("image_path", ""))).resolve()
        cards.append(
            f"""
<section class="card">
  <div class="meta">
    <h2>{idx}. {h(row.get("id"))}</h2>
    <p><b>Question:</b> {h(row.get("underspecified_question"))}</p>
    <p><b>Expected:</b> {h(row.get("expected_policy"))} · <b>Target:</b> {h(row.get("target_object"))} · <b>Auto candidates:</b> {h(row.get("candidate_count_scene_graph"))}</p>
    <p><b>Manual:</b> keep? ___ · human candidate count ___ · expected policy ___ · issue ___</p>
  </div>
  <img src="{h(image_path)}" alt="{h(row.get("id"))}" />
  <div class="runs">
    <div>
      <h3>Default</h3>
      <p><b>Policy:</b> {h(default["policy"])} · <b>Confidence:</b> {h(default["confidence"])}</p>
      <p><b>Answer:</b> {h(default["answer"])}</p>
      <p><b>Rationale:</b> {h(default["rationale"])}</p>
    </div>
    <div>
      <h3>Evidence</h3>
      <p><b>Policy:</b> {h(evidence["policy"])} · <b>Confidence:</b> {h(evidence["confidence"])} · <b>Listed candidates:</b> {h(evidence["candidate_count"])}</p>
      <p><b>Candidates:</b> {h(evidence["candidates"])}</p>
      <p><b>Answer:</b> {h(evidence["answer"])}</p>
      <p><b>Rationale:</b> {h(evidence["rationale"])}</p>
    </div>
  </div>
</section>
"""
        )
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{h(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f6f3; color: #1f2933; }}
    h1 {{ font-size: 24px; margin: 0 0 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 0 0 18px; }}
    .card img {{ max-width: 520px; max-height: 360px; display: block; margin: 12px 0; border: 1px solid #ccc; }}
    .meta h2 {{ font-size: 18px; margin: 0 0 8px; }}
    .meta p, .runs p {{ margin: 6px 0; line-height: 1.35; }}
    .runs {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }}
    .runs div {{ border-top: 1px solid #eee; padding-top: 8px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
  </style>
</head>
<body>
  <h1>{h(title)}</h1>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--title", required=True)
    parser.add_argument("--html-output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.items))
    runs = load_runs([Path(p) for p in args.runs])
    write_html(Path(args.html_output), args.title, rows, runs)
    write_csv(Path(args.csv_output), rows, runs)
    print(f"Wrote {args.html_output}")
    print(f"Wrote {args.csv_output}")


if __name__ == "__main__":
    main()
