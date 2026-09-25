#!/usr/bin/env python3
"""Select slide-ready qualitative cases for the visual-sketchpad condition."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def short(text: Any, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def observation_summary(parsed: dict[str, Any], limit: int = 5) -> str:
    observations = parsed.get("observation_0")
    if not isinstance(observations, list):
        return ""
    parts = []
    for obs in observations[:limit]:
        if not isinstance(obs, dict):
            continue
        mark_id = obs.get("mark_id", "")
        attrs = obs.get("visible_attributes", "")
        pos = obs.get("position", "")
        plausible = obs.get("is_plausible_referent", "")
        parts.append(f"{mark_id} plausible={plausible}: {pos}; {attrs}")
    if len(observations) > limit:
        parts.append(f"... +{len(observations) - limit} more")
    return " | ".join(parts)


def candidate_summary(parsed: dict[str, Any], limit: int = 4) -> str:
    candidates = parsed.get("relevant_candidates")
    if not isinstance(candidates, list):
        return ""
    parts = []
    for cand in candidates[:limit]:
        if not isinstance(cand, dict):
            continue
        parts.append(
            f"{cand.get('name', '')} [{cand.get('target_status', '')}] "
            f"{cand.get('position', '')}: {cand.get('attributes', '')}"
        )
    if len(candidates) > limit:
        parts.append(f"... +{len(candidates) - limit} more")
    return " | ".join(parts)


def main() -> None:
    items = {row["id"]: row for row in read_jsonl(BASE / "data/task1_ambiguous_auto_clean_items.jsonl")}
    marked_items = {row["id"]: row for row in read_jsonl(BASE / "data/task1_ambiguous_auto_clean_marked_items.jsonl")}
    task2_items = {row["id"]: row for row in read_jsonl(BASE / "data/task2_unique_auto_clean_items.jsonl")}
    task2_marked_items = {row["id"]: row for row in read_jsonl(BASE / "data/task2_unique_auto_clean_marked_items.jsonl")}

    runs = {
        "default": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_default.jsonl"),
        "evidence": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_evidence.jsonl"),
        "structured_prior": load_run(
            BASE / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior.jsonl",
            BASE / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry.jsonl",
            BASE / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry2.jsonl",
            BASE / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry3.jsonl",
            BASE / "runs/task1_auto_clean457_openai_gpt54mini_structured_prior_retry4.jsonl",
        ),
        "visual_sketchpad": load_run(BASE / "runs/task1_auto_clean457_openai_gpt54mini_visual_sketchpad.jsonl"),
    }
    task2_runs = {
        "default": load_run(BASE / "runs/task2_auto_clean330_openai_gpt54mini_default.jsonl"),
        "visual_sketchpad": load_run(BASE / "runs/task2_auto_clean330_openai_gpt54mini_visual_sketchpad.jsonl"),
    }

    parsed: dict[str, dict[str, dict[str, Any]]] = {
        mode: {item_id: parse_jsonish(row.get("raw_text", "")) for item_id, row in rows.items()}
        for mode, rows in runs.items()
    }
    task2_parsed: dict[str, dict[str, dict[str, Any]]] = {
        mode: {item_id: parse_jsonish(row.get("raw_text", "")) for item_id, row in rows.items()}
        for mode, rows in task2_runs.items()
    }

    def has(item_id: str, mode: str, pol: str) -> bool:
        return policy(parsed[mode][item_id]) == pol

    selected: list[tuple[str, str, str]] = []

    def add_group(title: str, predicate, limit: int) -> None:
        count = 0
        for item_id, item in items.items():
            if count >= limit:
                break
            if predicate(item_id, item):
                selected.append((title, item_id, "task1"))
                count += 1

    add_group(
        "All Text Conditions Over-Answer, Visual Sketchpad Repairs",
        lambda item_id, _: has(item_id, "default", "answer")
        and has(item_id, "evidence", "answer")
        and has(item_id, "structured_prior", "answer")
        and policy(parsed["visual_sketchpad"][item_id]) in {"clarify", "enumerate"},
        3,
    )
    add_group(
        "Evidence Over-Answers, Structured Prior and Visual Sketchpad Repair",
        lambda item_id, _: has(item_id, "default", "answer")
        and has(item_id, "evidence", "answer")
        and policy(parsed["structured_prior"][item_id]) in {"clarify", "enumerate"}
        and policy(parsed["visual_sketchpad"][item_id]) in {"clarify", "enumerate"},
        2,
    )
    add_group(
        "Visual Sketchpad Still Over-Answers",
        lambda item_id, _: has(item_id, "visual_sketchpad", "answer"),
        2,
    )

    # Add one Task 2 control case showing that marked images do not force clarification.
    for item_id, item in task2_items.items():
        if (
            policy(task2_parsed["default"][item_id]) == "answer"
            and policy(task2_parsed["visual_sketchpad"][item_id]) == "answer"
        ):
            selected.append(("Task 2 Control: Unique Referent Still Answers", item_id, "task2"))
            break

    out_dir = BASE / "runs/formal_analysis_with_visual_sketchpad"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Visual Sketchpad Case Selection",
        "",
        "These cases are selected for slides/proposal writing. Each case compares the original image with the marked sketchpad artifact and the four answer conditions.",
        "",
    ]
    json_rows = []
    for group, item_id, task in selected:
        if task == "task1":
            item = items[item_id]
            marked = marked_items[item_id]
            parsed_modes = {mode: parsed[mode][item_id] for mode in runs}
        else:
            item = task2_items[item_id]
            marked = task2_marked_items[item_id]
            parsed_modes = {
                "default": task2_parsed["default"][item_id],
                "visual_sketchpad": task2_parsed["visual_sketchpad"][item_id],
            }
        json_rows.append(
            {
                "group": group,
                "task": task,
                "id": item_id,
                "target_object": item.get("target_object"),
                "question": item.get("underspecified_question"),
                "original_image_path": marked.get("original_image_path", item.get("image_path")),
                "marked_image_path": marked.get("image_path"),
                "policies": {mode: policy(p) for mode, p in parsed_modes.items()},
                "answers": {mode: p.get("answer", "") for mode, p in parsed_modes.items()},
            }
        )
        md += [
            f"## {group}",
            "",
            f"### {item_id} ({item.get('target_object')})",
            "",
            f"- Question: `{item.get('underspecified_question')}`",
            f"- Expected policy: `{item.get('expected_policy')}`",
            f"- Scene-graph candidate count: `{item.get('candidate_count_scene_graph')}`",
            f"- Marked candidate count: `{marked.get('marked_candidate_count')}`",
            "",
            "**Original Image**",
            "",
            f"![original]({marked.get('original_image_path', item.get('image_path'))})",
            "",
            "**Marked Sketchpad Image**",
            "",
            f"![marked]({marked.get('image_path')})",
            "",
            "| condition | policy | answer | rationale/process note |",
            "|---|---|---|---|",
        ]
        for mode in ("default", "evidence", "structured_prior", "visual_sketchpad"):
            if mode not in parsed_modes:
                continue
            p = parsed_modes[mode]
            note = p.get("rationale", "")
            if mode == "evidence":
                note = candidate_summary(p) or note
            if mode == "visual_sketchpad":
                note = observation_summary(p) or note
            md.append(
                f"| `{mode}` | `{policy(p)}` | {short(p.get('answer', ''), 220)} | {short(note, 360)} |"
            )
        md.append("")

    (out_dir / "visual_sketchpad_case_selection.md").write_text("\n".join(md), encoding="utf-8")
    with (out_dir / "visual_sketchpad_case_selection.jsonl").open("w", encoding="utf-8") as f:
        for row in json_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(out_dir / "visual_sketchpad_case_selection.md")
    print(out_dir / "visual_sketchpad_case_selection.jsonl")


if __name__ == "__main__":
    main()
