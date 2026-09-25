#!/usr/bin/env python3
"""Build process-consistency summaries and representative case banks."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "runs/process_analysis_coco_gqa"
COCO_SUMMARY = BASE / "runs/model_expansion_limit100/condition_summary.csv"
GQA_SUMMARY = BASE / "runs/gqa_multimodel_summary/condition_summary.csv"

MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]
MODELS = [
    "GPT-5.4-mini",
    "Gemini-3.1-Pro",
    "GLM-4.6V",
    "Mistral-Small-3.2-Vision",
    "Qwen2.5-VL-72B",
]
TASK_CFG = {
    ("COCO", "task1"): {
        "task_family": "ambiguous",
        "expected_policy": {"clarify", "enumerate"},
        "expected_uniqueness": "ambiguous",
    },
    ("COCO", "task2"): {
        "task_family": "unique",
        "expected_policy": {"answer"},
        "expected_uniqueness": "unique",
    },
    ("GQA", "gqa_task1"): {
        "task_family": "ambiguous",
        "expected_policy": {"clarify", "enumerate"},
        "expected_uniqueness": "ambiguous",
    },
    ("GQA", "gqa_task2"): {
        "task_family": "unique",
        "expected_policy": {"answer"},
        "expected_uniqueness": "unique",
    },
}
ITEM_FILES = [
    BASE / "data/task1_ambiguous_auto_clean_marked_items.jsonl",
    BASE / "data/task2_unique_auto_clean_marked_items.jsonl",
    BASE / "data/gqa_task1_ambiguous_marked_items.jsonl",
    BASE / "data/gqa_task2_unique_marked_items.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
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


def load_summary_rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for dataset, path in [("COCO", COCO_SUMMARY), ("GQA", GQA_SUMMARY)]:
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["dataset"] = dataset
                out.append(row)
    return out


def load_items() -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in ITEM_FILES:
        for row in read_jsonl(path):
            items[str(row["id"])] = row
    return items


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, tuple[bool, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("id"))
        parsed = parse_jsonish(str(row.get("raw_text", "")))
        valid = not row.get("error") and bool(parsed.get("policy"))
        previous = best.get(item_id)
        if previous is None or valid or not previous[0]:
            best[item_id] = (valid, row)
    return {item_id: row for item_id, (_, row) in best.items()}


def policy(parsed: dict[str, Any]) -> str:
    return str(parsed.get("policy", "")).strip().lower() or "<blank>"


def correctness(dataset: str, task: str, pol: str) -> bool:
    return pol in TASK_CFG[(dataset, task)]["expected_policy"]


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def observation_plausible_count(parsed: dict[str, Any]) -> int | None:
    obs = parsed.get("observation_0")
    if not isinstance(obs, list):
        return None
    return sum(1 for item in obs if isinstance(item, dict) and is_true(item.get("is_plausible_referent")))


def observation_implied_uniqueness(count: int | None) -> str:
    if count is None:
        return "missing_observation"
    if count == 0:
        return "missing"
    if count == 1:
        return "unique"
    return "ambiguous"


def candidate_count_from_evidence(parsed: dict[str, Any]) -> int | None:
    candidates = parsed.get("relevant_candidates")
    if not isinstance(candidates, list):
        return None
    count = 0
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        status = str(cand.get("target_status", "")).lower()
        if status in {"possible", "target", ""}:
            count += 1
    return count


def expected_uniqueness(dataset: str, task: str) -> str:
    return str(TASK_CFG[(dataset, task)]["expected_uniqueness"])


def process_policy_consistent(uniqueness: str, pol: str) -> bool:
    if uniqueness == "ambiguous":
        return pol in {"clarify", "enumerate"}
    if uniqueness == "unique":
        return pol == "answer"
    if uniqueness == "missing":
        return pol == "uncertain"
    return False


def failure_type(dataset: str, task: str, parsed: dict[str, Any]) -> str:
    pol = policy(parsed)
    unique = str(parsed.get("referent_uniqueness", "")).lower()
    plausible_count = observation_plausible_count(parsed)
    obs_unique = observation_implied_uniqueness(plausible_count)
    expected = expected_uniqueness(dataset, task)
    final_ok = correctness(dataset, task, pol)
    proc_final_ok = process_policy_consistent(unique, pol)

    if final_ok and unique == expected and obs_unique == expected and proc_final_ok:
        return "correct_consistent"
    if unique == expected and not proc_final_ok:
        return "process_answer_inconsistent"
    if obs_unique != expected and unique == obs_unique:
        return "visual_observation_failure"
    if obs_unique == expected and unique != expected:
        return "uniqueness_reasoning_failure"
    if not final_ok:
        if expected == "ambiguous" and pol == "answer":
            return "over_answer_after_process"
        if expected == "unique" and pol in {"clarify", "enumerate"}:
            return "over_conservative_after_process"
    return "other"


def truncate(text: str, n: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def build_runs(summary_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, dict[str, Any]]]:
    runs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = (row["dataset"], row["task"], row["model"], row["mode"])
        runs[key] = load_run(Path(row["path"]))
    return runs


def process_rows(summary_rows: list[dict[str, str]], runs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["mode"] != "visual_sketchpad":
            continue
        dataset, task, model = row["dataset"], row["task"], row["model"]
        for item_id, record in runs[(dataset, task, model, "visual_sketchpad")].items():
            parsed = parse_jsonish(str(record.get("raw_text", "")))
            pol = policy(parsed)
            plausible_count = observation_plausible_count(parsed)
            obs_unique = observation_implied_uniqueness(plausible_count)
            ref_unique = str(parsed.get("referent_uniqueness", "")).lower() or "<blank>"
            out.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "task_family": TASK_CFG[(dataset, task)]["task_family"],
                    "model": model,
                    "id": item_id,
                    "question": record.get("question", ""),
                    "expected_uniqueness": expected_uniqueness(dataset, task),
                    "expected_policy": record.get("expected_policy", ""),
                    "policy": pol,
                    "final_policy_correct": correctness(dataset, task, pol),
                    "referent_uniqueness": ref_unique,
                    "plausible_count": plausible_count if plausible_count is not None else "",
                    "observation_implied_uniqueness": obs_unique,
                    "obs_matches_expected": obs_unique == expected_uniqueness(dataset, task),
                    "uniqueness_matches_expected": ref_unique == expected_uniqueness(dataset, task),
                    "process_policy_consistent": process_policy_consistent(ref_unique, pol),
                    "failure_type": failure_type(dataset, task, parsed),
                    "rationale": truncate(parsed.get("rationale", ""), 500),
                }
            )
    return out


def summarize_process(process: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in process:
        groups[(row["dataset"], row["task_family"], row["model"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, task_family, model), rows in sorted(groups.items()):
        n = len(rows)
        final_correct = sum(1 for r in rows if r["final_policy_correct"])
        obs_match = sum(1 for r in rows if r["obs_matches_expected"])
        uniq_match = sum(1 for r in rows if r["uniqueness_matches_expected"])
        proc_consistent = sum(1 for r in rows if r["process_policy_consistent"])
        correct_consistent = sum(1 for r in rows if r["failure_type"] == "correct_consistent")
        failures = Counter(str(r["failure_type"]) for r in rows)
        summaries.append(
            {
                "dataset": dataset,
                "task_family": task_family,
                "model": model,
                "n": n,
                "final_policy_accuracy": final_correct / n if n else 0,
                "observation_matches_expected": obs_match / n if n else 0,
                "uniqueness_matches_expected": uniq_match / n if n else 0,
                "process_policy_consistency": proc_consistent / n if n else 0,
                "correct_consistent_rate": correct_consistent / n if n else 0,
                "failure_counts": json.dumps(dict(failures), ensure_ascii=False, sort_keys=True),
            }
        )
    return summaries


def summarize_failures(process: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in process:
        groups[(row["dataset"], row["task_family"], row["model"])][str(row["failure_type"])] += 1
    out: list[dict[str, Any]] = []
    for (dataset, task_family, model), counts in sorted(groups.items()):
        total = sum(counts.values())
        for failure, count in sorted(counts.items()):
            out.append(
                {
                    "dataset": dataset,
                    "task_family": task_family,
                    "model": model,
                    "failure_type": failure,
                    "count": count,
                    "rate": count / total if total else 0.0,
                    "n": total,
                }
            )
    return out


def mode_parsed(
    runs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]],
    dataset: str,
    task: str,
    model: str,
    item_id: str,
    mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    record = runs.get((dataset, task, model, mode), {}).get(item_id)
    if not record:
        return None, {}
    return record, parse_jsonish(str(record.get("raw_text", "")))


def collect_case_candidates(
    summary_rows: list[dict[str, str]],
    runs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]],
    items: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_keys = sorted({(r["dataset"], r["task"], r["model"]) for r in summary_rows})
    for dataset, task, model in task_keys:
        sketch_run = runs.get((dataset, task, model, "visual_sketchpad"), {})
        for item_id in sketch_run:
            records: dict[str, dict[str, Any] | None] = {}
            parsed: dict[str, dict[str, Any]] = {}
            for mode in MODES:
                rec, par = mode_parsed(runs, dataset, task, model, item_id, mode)
                records[mode] = rec
                parsed[mode] = par
            if not all(records[m] for m in MODES):
                continue
            policies = {mode: policy(parsed[mode]) for mode in MODES}
            family = TASK_CFG[(dataset, task)]["task_family"]
            item = items.get(item_id, {})
            base = {
                "dataset": dataset,
                "task": task,
                "task_family": family,
                "model": model,
                "id": item_id,
                "image_id": records["visual_sketchpad"].get("image_id", ""),
                "question": records["visual_sketchpad"].get("question", ""),
                "candidate_count": records["visual_sketchpad"].get("candidate_count_scene_graph", ""),
                "original_image_path": item.get("original_image_path", ""),
                "marked_image_path": item.get("image_path", ""),
                "default_policy": policies["default"],
                "evidence_policy": policies["evidence"],
                "marked_policy": policies["marked_no_prior"],
                "sketchpad_policy": policies["visual_sketchpad"],
                "sketchpad_uniqueness": str(parsed["visual_sketchpad"].get("referent_uniqueness", "")).lower(),
                "sketchpad_plausible_count": observation_plausible_count(parsed["visual_sketchpad"]),
                "default_answer": truncate(parsed["default"].get("answer", ""), 180),
                "evidence_answer": truncate(parsed["evidence"].get("answer", ""), 180),
                "marked_answer": truncate(parsed["marked_no_prior"].get("answer", ""), 180),
                "sketchpad_answer": truncate(parsed["visual_sketchpad"].get("answer", ""), 180),
                "default_rationale": truncate(parsed["default"].get("rationale", ""), 300),
                "evidence_rationale": truncate(parsed["evidence"].get("rationale", ""), 300),
                "marked_rationale": truncate(parsed["marked_no_prior"].get("rationale", ""), 300),
                "sketchpad_rationale": truncate(parsed["visual_sketchpad"].get("rationale", ""), 420),
                "sketchpad_observation": truncate(parsed["visual_sketchpad"].get("observation_0", ""), 700),
            }
            if family == "ambiguous" and policies["default"] == "answer" and policies["visual_sketchpad"] in {"clarify", "enumerate"}:
                by_case["default_overanswer_fixed_by_sketchpad"].append(base)
            if family == "ambiguous" and policies["marked_no_prior"] == "answer" and policies["visual_sketchpad"] in {"clarify", "enumerate"}:
                by_case["marked_evidence_not_enough_process_prior_fixes"].append(base)
            ev_count = candidate_count_from_evidence(parsed["evidence"])
            if family == "ambiguous" and ev_count is not None and ev_count > 1 and policies["evidence"] == "answer":
                case = dict(base)
                case["evidence_candidate_count"] = ev_count
                by_case["evidence_lists_multiple_but_answers"].append(case)
            if family == "unique" and policies["visual_sketchpad"] == "answer":
                by_case["unique_control_sketchpad_answers"].append(base)
            if family == "ambiguous" and policies["visual_sketchpad"] == "answer":
                by_case["sketchpad_failure_overanswer"].append(base)
            if family == "unique" and policies["visual_sketchpad"] in {"clarify", "enumerate"}:
                by_case["sketchpad_failure_overclarify"].append(base)
    return by_case


def choose_cases(cases: dict[str, list[dict[str, Any]]], per_category: int = 12) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for category, rows in sorted(cases.items()):
        seen_models: set[str] = set()
        seen_datasets: set[str] = set()
        ranked = sorted(rows, key=lambda r: (r["dataset"], r["model"], str(r["id"])))
        picks: list[dict[str, Any]] = []
        for row in ranked:
            if len(picks) >= per_category:
                break
            if row["model"] not in seen_models or row["dataset"] not in seen_datasets:
                picks.append(row)
                seen_models.add(row["model"])
                seen_datasets.add(row["dataset"])
        for row in ranked:
            if len(picks) >= per_category:
                break
            if row not in picks:
                picks.append(row)
        for i, row in enumerate(picks, start=1):
            out = dict(row)
            out["case_category"] = category
            out["case_rank"] = i
            selected.append(out)
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(process_summary: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# COCO+GQA Process Analysis", ""]
    lines += [
        "## Process Consistency Summary",
        "",
        "| dataset | task | model | n | final acc | obs match | uniqueness match | process-policy consistency | correct-consistent |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in process_summary:
        lines.append(
            "| {dataset} | {task_family} | {model} | {n} | {fa} | {om} | {um} | {pc} | {cc} |".format(
                dataset=row["dataset"],
                task_family=row["task_family"],
                model=row["model"],
                n=row["n"],
                fa=pct(float(row["final_policy_accuracy"])),
                om=pct(float(row["observation_matches_expected"])),
                um=pct(float(row["uniqueness_matches_expected"])),
                pc=pct(float(row["process_policy_consistency"])),
                cc=pct(float(row["correct_consistent_rate"])),
            )
        )
    lines += ["", "## Representative Case Bank", ""]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["case_category"])].append(case)
    for category, rows in grouped.items():
        lines += [f"### {category}", ""]
        for row in rows[:8]:
            lines += [
                f"- **{row['dataset']} / {row['task_family']} / {row['model']} / {row['id']}**",
                f"  - Question: {row['question']}",
                f"  - Policies: default={row['default_policy']}, evidence={row['evidence_policy']}, marked={row['marked_policy']}, sketchpad={row['sketchpad_policy']} ({row['sketchpad_uniqueness']}, plausible={row['sketchpad_plausible_count']})",
                f"  - Default: {row['default_answer']} | {row['default_rationale']}",
                f"  - Sketchpad: {row['sketchpad_answer']} | {row['sketchpad_rationale']}",
                f"  - Images: original=`{row['original_image_path']}`, marked=`{row['marked_image_path']}`",
            ]
        lines.append("")
    (OUT_DIR / "process_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex_tables(process_summary: list[dict[str, Any]]) -> None:
    avg: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in process_summary:
        avg[(row["dataset"], row["task_family"])].append(row)
    lines = [
        "% Auto-generated by analyze_process_and_cases.py",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Task & Final acc. & Obs. match & Uniq. match & Process consistency \\\\",
        "\\midrule",
    ]
    for (dataset, task), rows in sorted(avg.items()):
        n = len(rows)
        final = sum(float(r["final_policy_accuracy"]) for r in rows) / n
        obs = sum(float(r["observation_matches_expected"]) for r in rows) / n
        uniq = sum(float(r["uniqueness_matches_expected"]) for r in rows) / n
        proc = sum(float(r["process_policy_consistency"]) for r in rows) / n
        lines.append(f"{dataset} & {task} & {final*100:.1f} & {obs*100:.1f} & {uniq*100:.1f} & {proc*100:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT_DIR / "process_consistency_table.tex").write_text("\n".join(lines), encoding="utf-8")


def html_escape(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def file_uri(path: Any) -> str:
    if not path:
        return ""
    return Path(str(path)).expanduser().resolve().as_uri()


def write_case_report_html(cases: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["case_category"])].append(case)
    css = """
body { font-family: Georgia, 'Times New Roman', serif; margin: 32px; color: #202020; }
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
.case { display: grid; grid-template-columns: 240px 1fr; gap: 18px; margin: 20px 0 30px; page-break-inside: avoid; }
.imgs { display: grid; grid-template-columns: 1fr; gap: 10px; }
img { max-width: 240px; border: 1px solid #ccc; }
.meta { font-size: 14px; color: #555; margin-bottom: 8px; }
.question { font-weight: bold; font-size: 17px; margin-bottom: 8px; }
.policies { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; background: #f6f6f6; padding: 8px; border-radius: 4px; }
.rationale { margin-top: 8px; line-height: 1.35; }
.label { font-weight: bold; }
"""
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>COCO+GQA Case Report</title>",
        f"<style>{css}</style></head><body>",
        "<h1>COCO+GQA Representative Case Report</h1>",
        "<p>Selected automatically from the case bank. Each case shows the original and marked image plus policy changes across conditions.</p>",
    ]
    for category, rows in grouped.items():
        parts.append(f"<h2>{html_escape(category)}</h2>")
        for row in rows[:8]:
            original_uri = file_uri(row.get("original_image_path"))
            marked_uri = file_uri(row.get("marked_image_path"))
            parts.append("<div class='case'>")
            parts.append("<div class='imgs'>")
            if original_uri:
                parts.append(f"<div><div class='meta'>Original</div><img src='{original_uri}'></div>")
            if marked_uri:
                parts.append(f"<div><div class='meta'>Marked</div><img src='{marked_uri}'></div>")
            parts.append("</div>")
            parts.append("<div>")
            parts.append(
                f"<div class='meta'>{html_escape(row['dataset'])} / {html_escape(row['task_family'])} / {html_escape(row['model'])} / {html_escape(row['id'])}</div>"
            )
            parts.append(f"<div class='question'>{html_escape(row['question'])}</div>")
            parts.append(
                "<div class='policies'>"
                f"default={html_escape(row['default_policy'])}; "
                f"evidence={html_escape(row['evidence_policy'])}; "
                f"marked={html_escape(row['marked_policy'])}; "
                f"sketchpad={html_escape(row['sketchpad_policy'])}; "
                f"uniqueness={html_escape(row['sketchpad_uniqueness'])}; "
                f"plausible={html_escape(row['sketchpad_plausible_count'])}"
                "</div>"
            )
            parts.append(f"<div class='rationale'><span class='label'>Default:</span> {html_escape(row['default_answer'])} - {html_escape(row['default_rationale'])}</div>")
            parts.append(f"<div class='rationale'><span class='label'>Evidence:</span> {html_escape(row['evidence_answer'])} - {html_escape(row['evidence_rationale'])}</div>")
            parts.append(f"<div class='rationale'><span class='label'>Marked:</span> {html_escape(row['marked_answer'])} - {html_escape(row['marked_rationale'])}</div>")
            parts.append(f"<div class='rationale'><span class='label'>Sketchpad:</span> {html_escape(row['sketchpad_answer'])} - {html_escape(row['sketchpad_rationale'])}</div>")
            parts.append("</div></div>")
    parts.append("</body></html>")
    (OUT_DIR / "case_report.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary_rows()
    runs = build_runs(summary_rows)
    items = load_items()
    process = process_rows(summary_rows, runs)
    process_summary = summarize_process(process)
    failure_summary = summarize_failures(process)
    case_candidates = collect_case_candidates(summary_rows, runs, items)
    selected_cases = choose_cases(case_candidates)

    write_csv(OUT_DIR / "process_level_rows.csv", process)
    write_jsonl(OUT_DIR / "process_level_rows.jsonl", process)
    write_csv(OUT_DIR / "process_consistency_summary.csv", process_summary)
    write_csv(OUT_DIR / "process_failure_type_summary.csv", failure_summary)
    write_jsonl(OUT_DIR / "case_bank_selected.jsonl", selected_cases)
    write_csv(OUT_DIR / "case_bank_selected.csv", selected_cases)
    write_markdown(process_summary, selected_cases)
    write_latex_tables(process_summary)
    write_case_report_html(selected_cases)

    counts = {category: len(rows) for category, rows in sorted(case_candidates.items())}
    (OUT_DIR / "case_candidate_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote process analysis to {OUT_DIR}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
