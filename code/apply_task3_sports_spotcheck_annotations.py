#!/usr/bin/env python3
"""Apply human spot-check annotations from the sports held-out XLSX."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def col_to_idx(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def read_xlsx_sheet(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as z:
        shared_strings: list[str] = []
        sst_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in sst_root.findall("a:si", NS):
            texts = [t.text or "" for t in si.findall(".//a:t", NS)]
            shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[dict[int, str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                value_node = cell.find("a:v", NS)
                value = ""
                if value_node is not None:
                    value = value_node.text or ""
                    if cell.attrib.get("t") == "s":
                        value = shared_strings[int(value)]
                values[col_to_idx(ref)] = value
            if values:
                raw_rows.append(values)

    headers = [raw_rows[0].get(i, "") for i in range(max(raw_rows[0]) + 1)]
    rows: list[dict[str, str]] = []
    for values in raw_rows[1:]:
        rows.append({headers[i]: values.get(i, "") for i in range(len(headers))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/task3_sports_action_eval_v3_heldout31.jsonl")
    parser.add_argument("--annotations-xlsx", default="review/task3_sports_heldout31_spotcheck_2026-06-26.xlsx")
    parser.add_argument("--output-jsonl", default="data/task3_sports_action_eval_v3_heldout31_humanchecked.jsonl")
    parser.add_argument("--output-summary", default="runs/task3_sports_action_eval_v3_heldout31_humanchecked_summary_2026-06-28.json")
    args = parser.parse_args()

    items = {row["id"]: row for row in load_jsonl(Path(args.items))}
    annotations = {row["id"]: row for row in read_xlsx_sheet(Path(args.annotations_xlsx))}

    output_rows: list[dict] = []
    decisions = Counter()
    changed_phase = 0
    changed_family = 0
    for item_id, item in items.items():
        ann = annotations[item_id]
        decision = ann.get("auto_label_ok", "").strip().lower()
        decisions[decision] += 1
        revised_phase = ann.get("revised_phase", "").strip()
        revised_family = ann.get("revised_family", "").strip()

        updated = dict(item)
        updated["human_spotcheck_decision"] = decision
        updated["human_issue_type"] = ann.get("issue_type", "").strip()
        updated["human_notes"] = ann.get("human_notes", "").strip()
        updated["source"] = "sports_action_heldout31_humanchecked"

        if revised_phase:
            changed_phase += 1
            updated["gold_action_phase"] = revised_phase
            updated["expected_answer"] = revised_phase
            updated["answer"] = revised_phase
        if revised_family:
            changed_family += 1
            updated["phase_boundary_family"] = revised_family
            updated["boundary_family"] = revised_family
        output_rows.append(updated)

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "items": args.items,
        "annotations_xlsx": args.annotations_xlsx,
        "output_jsonl": args.output_jsonl,
        "n": len(output_rows),
        "human_decisions": dict(decisions),
        "changed_phase": changed_phase,
        "changed_family": changed_family,
        "phase_distribution": dict(Counter(row["expected_answer"] for row in output_rows)),
        "family_distribution": dict(Counter(row["phase_boundary_family"] for row in output_rows)),
        "sport_distribution": dict(Counter(row.get("sport_hint", "") for row in output_rows)),
    }
    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_jsonl}")
    print(f"Wrote {out_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
