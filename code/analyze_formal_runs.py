#!/usr/bin/env python3
"""Analyze formal VLM runs for the underspecified-question pilot."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_run(path: Path, *retry_paths: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    for retry_path in retry_paths:
        if retry_path.exists():
            rows.update({str(row["id"]): row for row in read_jsonl(retry_path)})
    return rows


def policy(parsed: dict[str, Any]) -> str:
    return str(parsed.get("policy", "")).lower() or "<blank>"


def candidate_counts(parsed: dict[str, Any]) -> tuple[int, int, int]:
    candidates = parsed.get("relevant_candidates")
    if isinstance(candidates, list):
        possible = 0
        target = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            status = str(candidate.get("target_status", "")).lower()
            if status in {"possible", "target"}:
                possible += 1
            if status == "target":
                target += 1
        return len(candidates), possible, target

    observations = parsed.get("observation_0")
    if isinstance(observations, list):
        plausible = 0
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            if bool(observation.get("is_plausible_referent")):
                plausible += 1
        uniqueness = str(parsed.get("referent_uniqueness", "")).lower()
        target = 1 if uniqueness == "unique" and plausible == 1 else 0
        return len(observations), plausible, target

    return 0, 0, 0


def summarize_candidates(parsed: dict[str, Any], limit: int = 4) -> str:
    candidates = parsed.get("relevant_candidates")
    if not isinstance(candidates, list):
        return ""
    parts = []
    for candidate in candidates[:limit]:
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name", "object")
        status = candidate.get("target_status", "")
        position = candidate.get("position", "")
        attrs = candidate.get("attributes", "")
        parts.append(f"{name} [{status}] {position}: {attrs}")
    if len(candidates) > limit:
        parts.append(f"... +{len(candidates) - limit} more")
    return " | ".join(parts)


def short(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def pct(num: int, den: int) -> str:
    if den == 0:
        return "0.0%"
    return f"{num / den:.1%}"


def format_image(path: str) -> str:
    return f"![image]({path})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(BASE / "runs/formal_analysis"))
    args = parser.parse_args()

    base = BASE
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task1_items = read_jsonl(base / "data/task1_ambiguous_auto_clean_items.jsonl")
    task2_items = read_jsonl(base / "data/task2_unique_auto_clean_items.jsonl")
    runs = {
        "task1_default": load_run(base / "runs/task1_auto_clean457_openai_gpt54mini_default.jsonl"),
        "task1_evidence": load_run(base / "runs/task1_auto_clean457_openai_gpt54mini_evidence.jsonl"),
        "task1_structured_prior": load_run(
            base / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior.jsonl",
            base / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry.jsonl",
            base / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry2.jsonl",
            base / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry3.jsonl",
            base / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry4.jsonl",
        ),
        "task1_visual_sketchpad": load_run(
            base / "runs/task1_auto_clean457_openai_gpt54mini_visual_sketchpad.jsonl",
        ),
        "task1_marked_no_prior": load_run(
            base / "runs/task1_auto_clean457_openai_gpt54mini_marked_no_prior.jsonl",
        ),
        "task2_default": load_run(base / "runs/task2_auto_clean330_openai_gpt54mini_default.jsonl"),
        "task2_evidence": load_run(
            base / "runs/task2_auto_clean330_openai_gpt54mini_evidence.jsonl",
            base / "runs/task2_auto_clean330_openai_gpt54mini_evidence_retry4.jsonl",
        ),
        "task2_structured_prior": load_run(
            base / "runs/task2_auto_clean330_openai_gpt54mini_structured_prior.jsonl",
            base / "runs/task2_auto_clean330_openai_gpt54mini_structured_prior_retry.jsonl",
            base / "runs/task2_auto_clean330_openai_gpt54mini_structured_prior_retry2.jsonl",
            base / "runs/task2_auto_clean330_openai_gpt54mini_structured_prior_retry3.jsonl",
            base / "runs/task2_auto_clean330_openai_gpt54mini_structured_prior_retry4.jsonl",
        ),
        "task2_visual_sketchpad": load_run(
            base / "runs/task2_auto_clean330_openai_gpt54mini_visual_sketchpad.jsonl",
        ),
        "task2_marked_no_prior": load_run(
            base / "runs/task2_auto_clean330_openai_gpt54mini_marked_no_prior.jsonl",
        ),
    }

    parsed_runs: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in runs.items():
        parsed_runs[name] = {
            row_id: {
                "record": row,
                "parsed": parse_jsonish(row.get("raw_text", "")),
            }
            for row_id, row in rows.items()
        }

    breakdown_rows: list[dict[str, Any]] = []

    def add_breakdown(task_name: str, mode: str, items: list[dict[str, Any]], run_key: str) -> None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[str(item.get("target_object", ""))].append(item)
        for category, category_items in sorted(grouped.items()):
            total = len(category_items)
            policies = Counter()
            parse_fail = 0
            multi_possible = 0
            pai = 0
            overcorr = 0
            for item in category_items:
                parsed = parsed_runs[run_key][str(item["id"])]["parsed"]
                if not parsed:
                    parse_fail += 1
                pol = policy(parsed)
                policies[pol] += 1
                _, possible, target = candidate_counts(parsed)
                if possible >= 2:
                    multi_possible += 1
                if possible >= 2 and pol == "answer":
                    pai += 1
                if possible <= 1 and target <= 1 and pol in {"clarify", "enumerate"}:
                    overcorr += 1
            if task_name == "task1":
                correct = policies["clarify"] + policies["enumerate"]
                main_error = policies["answer"]
            else:
                correct = policies["answer"]
                main_error = policies["clarify"] + policies["enumerate"]
            breakdown_rows.append(
                {
                    "task": task_name,
                    "mode": mode,
                    "category": category,
                    "n": total,
                    "policy_accuracy": correct / total if total else 0,
                    "answer": policies["answer"],
                    "clarify": policies["clarify"],
                    "enumerate": policies["enumerate"],
                    "uncertain": policies["uncertain"],
                    "blank": policies["<blank>"],
                    "main_error": main_error,
                    "parse_fail": parse_fail,
                    "multi_possible": multi_possible,
                    "process_answer_inconsistency": pai,
                    "over_correction": overcorr,
                }
            )

    modes = ("default", "evidence", "structured_prior", "marked_no_prior", "visual_sketchpad")
    for mode in modes:
        add_breakdown("task1", mode, task1_items, f"task1_{mode}")
    for mode in modes:
        add_breakdown("task2", mode, task2_items, f"task2_{mode}")

    breakdown_csv = output_dir / "category_breakdown.csv"
    with breakdown_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
        writer.writeheader()
        writer.writerows(breakdown_rows)

    def total_summary(task_name: str, mode: str) -> dict[str, Any]:
        rows = [row for row in breakdown_rows if row["task"] == task_name and row["mode"] == mode]
        total = sum(row["n"] for row in rows)
        answer = sum(row["answer"] for row in rows)
        clarify = sum(row["clarify"] for row in rows)
        enumerate_ = sum(row["enumerate"] for row in rows)
        uncertain = sum(row["uncertain"] for row in rows)
        parse_fail = sum(row["parse_fail"] for row in rows)
        pai = sum(row["process_answer_inconsistency"] for row in rows)
        overcorr = sum(row["over_correction"] for row in rows)
        multi_possible = sum(row["multi_possible"] for row in rows)
        if task_name == "task1":
            correct = clarify + enumerate_
            main_error = answer
        else:
            correct = answer
            main_error = clarify + enumerate_
        return {
            "task": task_name,
            "mode": mode,
            "n": total,
            "policy_accuracy": correct,
            "main_error": main_error,
            "answer": answer,
            "clarify": clarify,
            "enumerate": enumerate_,
            "uncertain": uncertain,
            "parse_fail": parse_fail,
            "multi_possible": multi_possible,
            "process_answer_inconsistency": pai,
            "over_correction": overcorr,
        }

    total_rows = [total_summary(task_name, mode) for task_name in ("task1", "task2") for mode in modes]
    summary_md = ["# Condition Summary", ""]
    summary_md += [
        "## Task 1: Ambiguous Referent",
        "",
        "| mode | n | policy accuracy | over-answer | answer | clarify | enumerate | uncertain | P-A inconsistency | parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in total_rows if r["task"] == "task1"]:
        summary_md.append(
            "| {mode} | {n} | {acc} | {err} | {answer} | {clarify} | {enumerate} | {uncertain} | {pai} | {parse_fail} |".format(
                mode=row["mode"],
                n=row["n"],
                acc=pct(row["policy_accuracy"], row["n"]),
                err=pct(row["main_error"], row["n"]),
                answer=row["answer"],
                clarify=row["clarify"],
                enumerate=row["enumerate"],
                uncertain=row["uncertain"],
                pai=pct(row["process_answer_inconsistency"], row["n"]),
                parse_fail=row["parse_fail"],
            )
        )
    summary_md += [
        "",
        "## Task 2: Unique Referent",
        "",
        "| mode | n | policy accuracy | unnecessary clarification | answer | clarify | enumerate | uncertain | over-correction | parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in total_rows if r["task"] == "task2"]:
        summary_md.append(
            "| {mode} | {n} | {acc} | {err} | {answer} | {clarify} | {enumerate} | {uncertain} | {over} | {parse_fail} |".format(
                mode=row["mode"],
                n=row["n"],
                acc=pct(row["policy_accuracy"], row["n"]),
                err=pct(row["main_error"], row["n"]),
                answer=row["answer"],
                clarify=row["clarify"],
                enumerate=row["enumerate"],
                uncertain=row["uncertain"],
                over=pct(row["over_correction"], row["n"]),
                parse_fail=row["parse_fail"],
            )
        )
    summary_md += [
        "",
        "Interpretation:",
        "- `evidence` asks the model to list visual candidates before answering.",
        "- `structured_prior` additionally states the referent-disambiguation prior: if multiple plausible referents remain, do not choose by salience or prominence.",
        "- `marked_no_prior` uses the same marked images as visual aids but does not state the referent-uniqueness prior.",
        "- `visual_sketchpad` uses a marked-image visual artifact: candidate referents are boxed and labeled before the model reasons over them.",
        "- Task 1 should improve as over-answer decreases; Task 2 should remain high-accuracy to avoid a trivial always-clarify strategy.",
    ]
    condition_summary = output_dir / "condition_summary.md"
    condition_summary.write_text("\n".join(summary_md), encoding="utf-8")

    md = ["# Category Breakdown", ""]
    for task_name in ("task1", "task2"):
        for mode in modes:
            rows = [row for row in breakdown_rows if row["task"] == task_name and row["mode"] == mode]
            md += [f"## {task_name} / {mode}", ""]
            if task_name == "task1":
                md.append("| category | n | policy accuracy | answer | clarify | uncertain | P-A inconsistency | over-correction |")
                md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            else:
                md.append("| category | n | policy accuracy | answer | clarify | uncertain | multi-possible listed | unnecessary clarification |")
                md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for row in sorted(rows, key=lambda r: (-r["n"], r["category"])):
                if task_name == "task1":
                    md.append(
                        "| {category} | {n} | {acc} | {answer} | {clarify} | {uncertain} | {pai} | {over} |".format(
                            category=row["category"],
                            n=row["n"],
                            acc=pct(round(row["policy_accuracy"] * row["n"]), row["n"]),
                            answer=row["answer"],
                            clarify=row["clarify"],
                            uncertain=row["uncertain"],
                            pai=row["process_answer_inconsistency"],
                            over=row["over_correction"],
                        )
                    )
                else:
                    unnecessary = row["clarify"] + row["enumerate"]
                    md.append(
                        "| {category} | {n} | {acc} | {answer} | {clarify} | {uncertain} | {multi} | {unnec} |".format(
                            category=row["category"],
                            n=row["n"],
                            acc=pct(round(row["policy_accuracy"] * row["n"]), row["n"]),
                            answer=row["answer"],
                            clarify=row["clarify"],
                            uncertain=row["uncertain"],
                            multi=row["multi_possible"],
                            unnec=unnecessary,
                        )
                    )
            md.append("")
    (output_dir / "category_breakdown.md").write_text("\n".join(md), encoding="utf-8")

    def enrich(item: dict[str, Any], run_key: str) -> dict[str, Any]:
        row = parsed_runs[run_key][str(item["id"])]["record"]
        parsed = parsed_runs[run_key][str(item["id"])]["parsed"]
        count, possible, target = candidate_counts(parsed)
        return {
            "item": item,
            "row": row,
            "parsed": parsed,
            "policy": policy(parsed),
            "candidate_count": count,
            "possible_count": possible,
            "target_count": target,
        }

    task1_pairs = [
        (
            item,
            enrich(item, "task1_default"),
            enrich(item, "task1_evidence"),
            enrich(item, "task1_structured_prior"),
            enrich(item, "task1_visual_sketchpad"),
        )
        for item in task1_items
    ]
    task2_pairs = [
        (
            item,
            enrich(item, "task2_default"),
            enrich(item, "task2_evidence"),
            enrich(item, "task2_structured_prior"),
            enrich(item, "task2_visual_sketchpad"),
        )
        for item in task2_items
    ]

    case_groups = {
        "Task 1: default over-answers, evidence repairs to clarification": [
            pair for pair in task1_pairs
            if pair[1]["policy"] == "answer" and pair[2]["policy"] in {"clarify", "enumerate"} and pair[2]["possible_count"] >= 2
        ],
        "Task 1: evidence process-answer inconsistency": [
            pair for pair in task1_pairs
            if pair[2]["policy"] == "answer" and pair[2]["possible_count"] >= 2
        ],
        "Task 1: structured prior repairs evidence inconsistency": [
            pair for pair in task1_pairs
            if pair[2]["policy"] == "answer" and pair[2]["possible_count"] >= 2 and pair[3]["policy"] in {"clarify", "enumerate"}
        ],
        "Task 1: structured prior still violates its own process": [
            pair for pair in task1_pairs
            if pair[3]["policy"] == "answer" and pair[3]["possible_count"] >= 2
        ],
        "Task 1: visual sketchpad repairs structured-prior over-answer": [
            pair for pair in task1_pairs
            if pair[3]["policy"] == "answer" and pair[4]["policy"] in {"clarify", "enumerate"}
        ],
        "Task 1: visual sketchpad still over-answers": [
            pair for pair in task1_pairs
            if pair[4]["policy"] == "answer"
        ],
        "Task 2: evidence still answers correctly": [
            pair for pair in task2_pairs
            if pair[2]["policy"] == "answer" and pair[2]["possible_count"] <= 1
        ],
        "Task 2: structured prior still answers correctly": [
            pair for pair in task2_pairs
            if pair[3]["policy"] == "answer" and pair[3]["possible_count"] <= 1
        ],
        "Task 2: visual sketchpad still answers correctly": [
            pair for pair in task2_pairs
            if pair[4]["policy"] == "answer"
        ],
        "Task 2: evidence over-clarifies": [
            pair for pair in task2_pairs
            if pair[2]["policy"] in {"clarify", "enumerate"}
        ],
        "Task 2: structured prior over-clarifies": [
            pair for pair in task2_pairs
            if pair[3]["policy"] in {"clarify", "enumerate"}
        ],
        "Task 2: visual sketchpad over-clarifies": [
            pair for pair in task2_pairs
            if pair[4]["policy"] in {"clarify", "enumerate"}
        ],
    }

    cases = ["# Qualitative Cases", ""]
    case_json_rows: list[dict[str, Any]] = []
    for title, pairs in case_groups.items():
        cases += [f"## {title}", ""]
        for item, default, evidence, structured, sketchpad in pairs[:6]:
            parsed_default = default["parsed"]
            parsed_evidence = evidence["parsed"]
            parsed_structured = structured["parsed"]
            parsed_sketchpad = sketchpad["parsed"]
            case_json_rows.append(
                {
                    "case_type": title,
                    "id": item["id"],
                    "target_object": item.get("target_object"),
                    "question": item.get("underspecified_question"),
                    "image_path": item.get("image_path"),
                    "default_policy": default["policy"],
                    "default_answer": parsed_default.get("answer", ""),
                    "evidence_policy": evidence["policy"],
                    "evidence_answer": parsed_evidence.get("answer", ""),
                    "possible_count": evidence["possible_count"],
                    "candidate_summary": summarize_candidates(parsed_evidence),
                    "structured_policy": structured["policy"],
                    "structured_answer": parsed_structured.get("answer", ""),
                    "structured_possible_count": structured["possible_count"],
                    "structured_candidate_summary": summarize_candidates(parsed_structured),
                    "visual_sketchpad_policy": sketchpad["policy"],
                    "visual_sketchpad_answer": parsed_sketchpad.get("answer", ""),
                    "visual_sketchpad_observation_count": len(parsed_sketchpad.get("observation_0", []) or []),
                }
            )
            cases += [
                f"### {item['id']} ({item.get('target_object')})",
                "",
                format_image(str(item.get("image_path", ""))),
                "",
                f"- Question: `{item.get('underspecified_question')}`",
                f"- Default: `{default['policy']}` / {short(parsed_default.get('answer', ''))}",
                f"- Evidence: `{evidence['policy']}` / {short(parsed_evidence.get('answer', ''))}",
                f"- Structured prior: `{structured['policy']}` / {short(parsed_structured.get('answer', ''))}",
                f"- Visual sketchpad: `{sketchpad['policy']}` / {short(parsed_sketchpad.get('answer', ''))}",
                f"- Evidence candidates: {short(summarize_candidates(parsed_evidence), 420)}",
                f"- Structured candidates: {short(summarize_candidates(parsed_structured), 420)}",
                f"- Visual sketchpad observations: {len(parsed_sketchpad.get('observation_0', []) or [])}",
                f"- Evidence rationale: {short(parsed_evidence.get('rationale', ''), 360)}",
                f"- Structured rationale: {short(parsed_structured.get('rationale', ''), 360)}",
                f"- Visual sketchpad rationale: {short(parsed_sketchpad.get('rationale', ''), 360)}",
                "",
            ]
    (output_dir / "qualitative_cases.md").write_text("\n".join(cases), encoding="utf-8")
    with (output_dir / "qualitative_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in case_json_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    overview = [
        "# Formal Analysis Outputs",
        "",
        f"- Condition summary: `{condition_summary}`",
        f"- Category breakdown: `{output_dir / 'category_breakdown.md'}`",
        f"- Category breakdown CSV: `{breakdown_csv}`",
        f"- Qualitative cases: `{output_dir / 'qualitative_cases.md'}`",
        f"- Qualitative cases JSONL: `{output_dir / 'qualitative_cases.jsonl'}`",
        "",
        "Suggested reading order:",
        "1. Start with `condition_summary.md` for the three-condition result.",
        "2. Use `category_breakdown.md` to explain where repair/inconsistency concentrates.",
        "3. Use `qualitative_cases.md` as slide-ready examples.",
    ]
    (output_dir / "README.md").write_text("\n".join(overview), encoding="utf-8")

    print(output_dir / "README.md")
    print(condition_summary)
    print(output_dir / "category_breakdown.md")
    print(output_dir / "qualitative_cases.md")


if __name__ == "__main__":
    main()
