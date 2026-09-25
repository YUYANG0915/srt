#!/usr/bin/env python3
"""Generate OpenAI VLM suggestions for Task3 sports-action annotation."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


SYSTEM_INSTRUCTIONS = """You are a careful dataset annotation reviewer for a sports-action VLM benchmark.

Your job is to inspect the image and suggest strict visible-evidence labels.

Return only valid JSON:
{
  "keep": "yes" | "no",
  "gold_action_phase": "pre_action_setup" | "active_execution" | "follow_through_or_result" | "resting_or_non_action" | "",
  "phase_boundary_family": "setup_vs_execution" | "execution_vs_followthrough" | "action_vs_nonaction" | "equipment_context" | "",
  "neighbor_phase": "pre_action_setup" | "active_execution" | "follow_through_or_result" | "resting_or_non_action" | "",
  "why_not_neighbor_phase": string,
  "visible_actor_pose": string,
  "visible_ball_or_target": string,
  "visible_motion_cues": string,
  "confidence": number,
  "notes": string
}

Strict visible-evidence rule:
- Label the visible phase, not an inferred event.
- Do not label active_execution only because sports equipment is nearby.
- Do not label active_execution only because the action probably happened before or after the frame.
- Use active_execution only when body pose, equipment relation, and ball/target relation visibly show the main action happening now.
- Use follow_through_or_result only when the completed action/result is visibly supported.
- If the image needs a story like "the person probably just threw/hit/kicked it", choose pre_action_setup, resting_or_non_action, or reject if unclear.
- Use resting_or_non_action when equipment is present but the actor-equipment relation does not support a sports action phase.

Reject with keep=no when the image is too ambiguous, too small, too occluded, or not a clean sports-action phase item."""


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def extract_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    texts: list[str] = []
    for item in response.get("output", []) or []:
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                texts.append(str(part.get("text", "")))
    if texts:
        return "\n".join(texts)
    return json.dumps(response, ensure_ascii=False)


def build_prompt(row: dict[str, str]) -> str:
    return f"""Inspect the image and suggest strict sports-action labels.

Question: {row.get("question", "What phase of the sports action is happening in this scene?")}
Sport hint: {row.get("sport_hint", "")}
COCO sports equipment: {row.get("coco_sports_equipment", "")}
COCO categories: {row.get("coco_categories", "")}

Remember: label the visible phase only. Do not infer hidden before/after events.
Return only JSON."""


def call_openai(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_output_tokens: int,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTIONS}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            },
        ],
        "max_output_tokens": max_output_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(OPENAI_RESPONSES_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI HTTP {exc.code}: {detail}")
            if exc.code in {400, 401, 403, 404}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_error}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def write_suggestions_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "sport_hint",
        "image_path",
        "existing_keep",
        "existing_gold_action_phase",
        "suggested_keep",
        "suggested_gold_action_phase",
        "suggested_phase_boundary_family",
        "suggested_neighbor_phase",
        "suggested_why_not_neighbor_phase",
        "suggested_visible_actor_pose",
        "suggested_visible_ball_or_target",
        "suggested_visible_motion_cues",
        "suggested_confidence",
        "suggested_notes",
        "raw_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def should_annotate(row: dict[str, str], include_existing: bool) -> bool:
    if include_existing:
        return True
    return not row.get("keep", "").strip() and not row.get("gold_action_phase", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="review/task3_sports_action_candidates_v1_review.csv")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-jsonl", default="runs/task3_sports_action_openai_suggestions_v1.jsonl")
    parser.add_argument("--output-csv", default="review/task3_sports_action_openai_suggestions_v1.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=450)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    rows = [row for row in read_csv(Path(args.review_csv)) if should_annotate(row, args.include_existing)]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_jsonl = Path(args.output_jsonl)
    existing = load_jsonl(output_jsonl)
    done_ids = {str(row.get("id")) for row in existing}
    records = list(existing)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            item_id = row["id"]
            if item_id in done_ids:
                continue
            image_path = Path(row["image_path"]).expanduser()
            response = call_openai(
                api_key=api_key,
                model=args.model,
                prompt=build_prompt(row),
                image_path=image_path,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
                timeout=args.timeout,
            )
            raw_text = extract_text(response)
            parsed = parse_jsonish(raw_text)
            record = {
                "id": item_id,
                "sport_hint": row.get("sport_hint", ""),
                "image_path": row.get("image_path", ""),
                "existing_keep": row.get("keep", ""),
                "existing_gold_action_phase": row.get("gold_action_phase", ""),
                "model": args.model,
                "suggested_keep": parsed.get("keep", ""),
                "suggested_gold_action_phase": parsed.get("gold_action_phase", ""),
                "suggested_phase_boundary_family": parsed.get("phase_boundary_family", ""),
                "suggested_neighbor_phase": parsed.get("neighbor_phase", ""),
                "suggested_why_not_neighbor_phase": parsed.get("why_not_neighbor_phase", ""),
                "suggested_visible_actor_pose": parsed.get("visible_actor_pose", ""),
                "suggested_visible_ball_or_target": parsed.get("visible_ball_or_target", ""),
                "suggested_visible_motion_cues": parsed.get("visible_motion_cues", ""),
                "suggested_confidence": parsed.get("confidence", ""),
                "suggested_notes": parsed.get("notes", ""),
                "raw_text": raw_text,
                "raw_response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            records.append(record)
            print(f"[{idx}/{len(rows)}] wrote {item_id}")

    write_suggestions_csv(Path(args.output_csv), records)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
