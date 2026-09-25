#!/usr/bin/env python3
"""Build representative case report and annotation templates."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "runs/process_failure_annotation"
MODELS = {
    "gpt": "GPT-5.4-mini",
    "qwen": "Qwen2.5-VL-72B",
}
MODES = ("default", "evidence", "marked_no_prior", "visual_sketchpad")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(path: Path, *retry_paths: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    for retry_path in retry_paths:
        if retry_path.exists():
            rows.update({str(row["id"]): row for row in read_jsonl(retry_path)})
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


def policy(parsed: dict[str, Any]) -> str:
    return str(parsed.get("policy", "")).lower() or "<blank>"


def short(value: Any, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def evidence_candidate_counts(parsed: dict[str, Any]) -> tuple[int, int, int]:
    candidates = parsed.get("relevant_candidates")
    if not isinstance(candidates, list):
        return 0, 0, 0
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


def visual_observation_counts(parsed: dict[str, Any]) -> tuple[int, int]:
    observations = parsed.get("observation_0")
    if not isinstance(observations, list):
        return 0, 0
    plausible = sum(
        1
        for observation in observations
        if isinstance(observation, dict) and bool(observation.get("is_plausible_referent"))
    )
    return len(observations), plausible


def candidate_summary(parsed: dict[str, Any], limit: int = 5) -> str:
    candidates = parsed.get("relevant_candidates")
    if not isinstance(candidates, list):
        return ""
    parts = []
    for candidate in candidates[:limit]:
        if not isinstance(candidate, dict):
            continue
        parts.append(
            f"{candidate.get('name', '')} [{candidate.get('target_status', '')}] "
            f"{candidate.get('position', '')}: {candidate.get('attributes', '')}"
        )
    if len(candidates) > limit:
        parts.append(f"... +{len(candidates) - limit} more")
    return " | ".join(parts)


def observation_summary(parsed: dict[str, Any], limit: int = 6) -> str:
    observations = parsed.get("observation_0")
    if not isinstance(observations, list):
        return ""
    parts = []
    for obs in observations[:limit]:
        if not isinstance(obs, dict):
            continue
        parts.append(
            f"{obs.get('mark_id', '')} plausible={obs.get('is_plausible_referent', '')}: "
            f"{obs.get('position', '')}; {obs.get('visible_attributes', '')}"
        )
    if len(observations) > limit:
        parts.append(f"... +{len(observations) - limit} more")
    return " | ".join(parts)


def load_all_runs() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {"task1": {}, "task2": {}}
    runs["task1"]["gpt"] = {
        "default": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_default.jsonl"),
        "evidence": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_evidence.jsonl"),
        "marked_no_prior": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_marked_no_prior.jsonl"),
        "visual_sketchpad": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_visual_sketchpad.jsonl"),
    }
    runs["task1"]["qwen"] = {
        "default": load_run(BASE / "runs/task1_auto_clean457_qwen25vl72b_default.jsonl"),
        "evidence": load_run(
            BASE / "runs/task1_auto_clean457_qwen25vl72b_evidence.jsonl",
            BASE / "runs/task1_auto_clean457_qwen25vl72b_evidence_retry.jsonl",
        ),
        "marked_no_prior": load_run(BASE / "runs/task1_auto_clean457_qwen25vl72b_marked_no_prior.jsonl"),
        "visual_sketchpad": load_run(BASE / "runs/task1_auto_clean457_qwen25vl72b_visual_sketchpad.jsonl"),
    }
    runs["task2"]["gpt"] = {
        "default": load_run(BASE / "runs/task2_auto_clean330_openai_gpt54mini_default.jsonl"),
        "evidence": load_run(
            BASE / "runs/task2_auto_clean330_openai_gpt54mini_evidence.jsonl",
            BASE / "runs/task2_auto_clean330_openai_gpt54mini_evidence_retry4.jsonl",
        ),
        "marked_no_prior": load_run(BASE / "runs/task2_auto_clean330_openai_gpt54mini_marked_no_prior.jsonl"),
        "visual_sketchpad": load_run(BASE / "runs/task2_auto_clean330_openai_gpt54mini_visual_sketchpad.jsonl"),
    }
    runs["task2"]["qwen"] = {
        "default": load_run(
            BASE / "runs/task2_auto_clean330_qwen25vl72b_default.jsonl",
            BASE / "runs/task2_auto_clean330_qwen25vl72b_default_retry.jsonl",
        ),
        "evidence": load_run(BASE / "runs/task2_auto_clean330_qwen25vl72b_evidence.jsonl"),
        "marked_no_prior": load_run(BASE / "runs/task2_auto_clean330_qwen25vl72b_marked_no_prior.jsonl"),
        "visual_sketchpad": load_run(BASE / "runs/task2_auto_clean330_qwen25vl72b_visual_sketchpad.jsonl"),
    }
    return runs


def parsed_for(runs: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    return {
        task: {
            model: {
                mode: {
                    item_id: parse_jsonish(row.get("raw_text", ""))
                    for item_id, row in mode_rows.items()
                }
                for mode, mode_rows in model_rows.items()
            }
            for model, model_rows in task_rows.items()
        }
        for task, task_rows in runs.items()
    }


def make_row(
    case_type: str,
    model_key: str,
    task: str,
    item: dict[str, Any],
    marked_item: dict[str, Any],
    parsed_modes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = parsed_modes["evidence"]
    visual = parsed_modes["visual_sketchpad"]
    evidence_total, evidence_possible, evidence_target = evidence_candidate_counts(evidence)
    visual_total, visual_plausible = visual_observation_counts(visual)
    expected = item.get("expected_policy", "")
    visual_pol = policy(visual)
    suggested_failure = ""
    if task == "task1":
        if visual_pol == "answer" and visual_plausible >= 2:
            suggested_failure = "process_answer_inconsistency"
        elif visual_pol == "answer" and visual_plausible <= 1:
            suggested_failure = "perception_or_relevance_filtering_failure"
        elif policy(evidence) == "answer" and evidence_possible >= 2:
            suggested_failure = "evidence_process_answer_inconsistency"
        else:
            suggested_failure = "none_or_repaired"
    else:
        if visual_pol in {"clarify", "enumerate"}:
            suggested_failure = "over_conservative_clarification"
        else:
            suggested_failure = "none_control_success"

    row = {
        "annotation_id": f"{model_key}_{task}_{item['id']}",
        "case_type": case_type,
        "model": MODELS[model_key],
        "task": task,
        "sample_id": item["id"],
        "target_object": item.get("target_object", ""),
        "question": item.get("underspecified_question", ""),
        "expected_policy": expected,
        "candidate_count_scene_graph": item.get("candidate_count_scene_graph", ""),
        "marked_candidate_count": marked_item.get("marked_candidate_count", ""),
        "original_image_path": marked_item.get("original_image_path", item.get("image_path", "")),
        "marked_image_path": marked_item.get("image_path", ""),
        "default_policy": policy(parsed_modes["default"]),
        "default_answer": parsed_modes["default"].get("answer", ""),
        "evidence_policy": policy(evidence),
        "evidence_answer": evidence.get("answer", ""),
        "evidence_possible_count": evidence_possible,
        "evidence_candidates": candidate_summary(evidence),
        "marked_no_prior_policy": policy(parsed_modes["marked_no_prior"]),
        "marked_no_prior_answer": parsed_modes["marked_no_prior"].get("answer", ""),
        "visual_policy": visual_pol,
        "visual_answer": visual.get("answer", ""),
        "visual_referent_uniqueness": visual.get("referent_uniqueness", ""),
        "visual_observation_count": visual_total,
        "visual_plausible_count": visual_plausible,
        "visual_observations": observation_summary(visual),
        "visual_rationale": visual.get("rationale", ""),
        "suggested_failure_label": suggested_failure,
        "human_keep": "",
        "human_failure_label_primary": "",
        "human_failure_label_secondary": "",
        "human_failure_source": "",
        "human_process_answer_consistent": "",
        "human_expected_policy": "",
        "human_notes": "",
    }
    return row


def select_cases(items: dict[str, dict[str, Any]], parsed: dict[str, Any], model_key: str, task: str) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(case_type: str, predicate, limit: int) -> None:
        count = 0
        for item_id in items:
            if count >= limit:
                break
            if item_id in seen:
                continue
            modes = parsed[task][model_key]
            try:
                if predicate(item_id, modes):
                    selected.append((case_type, item_id))
                    seen.add(item_id)
                    count += 1
            except KeyError:
                continue

    if task == "task1":
        add(
            "T1_marked_no_prior_overanswers_visual_repairs",
            lambda item_id, m: policy(m["marked_no_prior"][item_id]) == "answer"
            and policy(m["visual_sketchpad"][item_id]) in {"clarify", "enumerate"},
            12,
        )
        add(
            "T1_evidence_lists_multiple_but_answers",
            lambda item_id, m: policy(m["evidence"][item_id]) == "answer"
            and evidence_candidate_counts(m["evidence"][item_id])[1] >= 2,
            10,
        )
        add(
            "T1_visual_sketchpad_still_overanswers",
            lambda item_id, m: policy(m["visual_sketchpad"][item_id]) == "answer",
            8,
        )
        add(
            "T1_all_nonvisual_conditions_fail_visual_repairs",
            lambda item_id, m: policy(m["default"][item_id]) == "answer"
            and policy(m["evidence"][item_id]) == "answer"
            and policy(m["marked_no_prior"][item_id]) == "answer"
            and policy(m["visual_sketchpad"][item_id]) in {"clarify", "enumerate"},
            5,
        )
    else:
        add(
            "T2_unique_control_visual_answers",
            lambda item_id, m: policy(m["visual_sketchpad"][item_id]) == "answer"
            and policy(m["marked_no_prior"][item_id]) == "answer",
            8,
        )
        add(
            "T2_over_conservative_any_condition",
            lambda item_id, m: any(policy(m[mode][item_id]) in {"clarify", "enumerate"} for mode in MODES),
            2,
        )
    return selected


def write_report(rows: list[dict[str, Any]]) -> None:
    md = [
        "# Representative Process-Failure Cases",
        "",
        "This report samples cases for manual process-failure annotation. The same rows are available in JSONL/CSV/XLSX templates.",
        "",
        "Failure label options:",
        "- `candidate_detection_failure`: process misses relevant visual candidates.",
        "- `candidate_relevance_failure`: process sees candidates but filters plausible referents incorrectly.",
        "- `uniqueness_judgment_failure`: process lists multiple plausible candidates but judges the referent as unique.",
        "- `process_answer_inconsistency`: process indicates ambiguity but final policy still answers.",
        "- `salience_guessing`: model picks the most salient/central/large object instead of clarifying.",
        "- `over_conservative_clarification`: unique referent case but model unnecessarily clarifies/enumerates.",
        "- `none_control_success`: no process failure for this sampled control row.",
        "",
    ]
    for case_type in sorted({row["case_type"] for row in rows}):
        subset = [row for row in rows if row["case_type"] == case_type]
        md += [f"## {case_type}", ""]
        for row in subset[:8]:
            md += [
                f"### {row['annotation_id']}",
                "",
                f"- Model: `{row['model']}`",
                f"- Task: `{row['task']}`",
                f"- Question: `{row['question']}`",
                f"- Expected policy: `{row['expected_policy']}`",
                f"- Suggested label: `{row['suggested_failure_label']}`",
                f"- Original image: `{row['original_image_path']}`",
                f"- Marked image: `{row['marked_image_path']}`",
                "",
                f"![marked]({row['marked_image_path']})",
                "",
                "| condition | policy | answer / evidence |",
                "|---|---|---|",
                f"| default | `{row['default_policy']}` | {short(row['default_answer'], 260)} |",
                f"| evidence | `{row['evidence_policy']}` | {short(row['evidence_candidates'] or row['evidence_answer'], 360)} |",
                f"| marked_no_prior | `{row['marked_no_prior_policy']}` | {short(row['marked_no_prior_answer'], 260)} |",
                f"| visual_sketchpad | `{row['visual_policy']}` | {short(row['visual_observations'] or row['visual_answer'], 420)} |",
                "",
            ]
    (OUT_DIR / "representative_case_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task_items = {
        "task1": {row["id"]: row for row in read_jsonl(BASE / "data/task1_ambiguous_auto_clean_items.jsonl")},
        "task2": {row["id"]: row for row in read_jsonl(BASE / "data/task2_unique_auto_clean_items.jsonl")},
    }
    marked_items = {
        "task1": {row["id"]: row for row in read_jsonl(BASE / "data/task1_ambiguous_auto_clean_marked_items.jsonl")},
        "task2": {row["id"]: row for row in read_jsonl(BASE / "data/task2_unique_auto_clean_marked_items.jsonl")},
    }
    runs = load_all_runs()
    parsed = parsed_for(runs)

    rows: list[dict[str, Any]] = []
    for model_key in MODELS:
        for task in ("task1", "task2"):
            for case_type, item_id in select_cases(task_items[task], parsed, model_key, task):
                rows.append(
                    make_row(
                        case_type,
                        model_key,
                        task,
                        task_items[task][item_id],
                        marked_items[task][item_id],
                        {mode: parsed[task][model_key][mode][item_id] for mode in MODES},
                    )
                )

    jsonl_path = OUT_DIR / "process_failure_annotation_template.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUT_DIR / "process_failure_annotation_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_report(rows)

    summary = {
        "n_rows": len(rows),
        "by_case_type": {},
        "by_model": {},
        "by_task": {},
    }
    for row in rows:
        summary["by_case_type"][row["case_type"]] = summary["by_case_type"].get(row["case_type"], 0) + 1
        summary["by_model"][row["model"]] = summary["by_model"].get(row["model"], 0) + 1
        summary["by_task"][row["task"]] = summary["by_task"].get(row["task"], 0) + 1
    (OUT_DIR / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
