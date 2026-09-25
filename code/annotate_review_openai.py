#!/usr/bin/env python3
"""Use an OpenAI VLM as a reviewer for pilot item filtering."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


SYSTEM_INSTRUCTIONS = """You are a careful dataset annotation reviewer for a VLM benchmark.

Your job is to inspect the image and the underspecified visual question, then decide whether the item is clean enough to keep.

Return only valid JSON with:
{
  "keep": "yes" | "no",
  "human_candidate_count": integer,
  "human_expected_policy": "answer" | "clarify" | "enumerate" | "uncertain",
  "issue_type": "clean" | "not_unique" | "not_ambiguous" | "occluded" | "tiny" | "category_boundary" | "unnatural_question" | "unclear_attribute" | "no_target" | "too_cluttered" | "other",
  "notes": string
}

Definitions:
- answer: exactly one clear plausible target exists and the requested attribute/action is visible.
- clarify: multiple plausible targets exist and selecting one would be unsafe.
- enumerate: two or a few plausible targets exist and their answers can be separately listed.
- uncertain: target/evidence is missing, too occluded, too small, or the attribute/action cannot be determined.

For Task 1 ambiguous-referent items, keep only if multiple plausible target-category candidates are visible and the ideal policy is clarify or enumerate.
For Task 2 unique-referent items, keep only if exactly one plausible target-category candidate is visible and the ideal policy is answer.

Be strict. Reject noisy images, tiny/occluded targets, category-boundary cases, or unnatural questions.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_prompt(item: dict[str, Any], task: str) -> str:
    if task == "task1":
        task_desc = (
            "Task 1: Ambiguous Referent. The intended clean item should have multiple "
            "plausible candidates for the target object, so the ideal policy should be "
            "clarify or enumerate."
        )
    elif task == "task2":
        task_desc = (
            "Task 2: Unique Referent. The intended clean item should have exactly one "
            "plausible candidate for the target object, so the ideal policy should be answer."
        )
    else:
        raise ValueError(f"Unknown task: {task}")
    return f"""{task_desc}

Inspect the image and question.

Question: {item.get("underspecified_question")}
Target object/category: {item.get("target_object")}
Automatically estimated candidate count: {item.get("candidate_count_scene_graph")}
Original source question: {item.get("original_question")}

Decide the manual review labels. Return only JSON."""


def call_openai(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_output_tokens: int,
    retries: int,
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(OPENAI_RESPONSES_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "target_object",
        "question",
        "image_path",
        "keep",
        "human_candidate_count",
        "human_expected_policy",
        "issue_type",
        "notes",
        "raw_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--task", choices=["task1", "task2"], required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    items = load_jsonl(Path(args.items))
    if args.limit is not None:
        items = items[: args.limit]

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    existing_rows = []
    if output_jsonl.exists():
        existing_rows = load_jsonl(output_jsonl)
        done_ids = {str(row.get("id")) for row in existing_rows}

    csv_rows = list(existing_rows)
    with output_jsonl.open("a", encoding="utf-8") as out:
        for idx, item in enumerate(items, start=1):
            item_id = str(item["id"])
            if item_id in done_ids:
                continue
            image_path = Path(str(item["image_path"])).expanduser()
            response = call_openai(
                api_key=api_key,
                model=args.model,
                prompt=build_prompt(item, args.task),
                image_path=image_path,
                max_output_tokens=args.max_output_tokens,
                retries=args.retries,
            )
            raw_text = extract_text(response)
            parsed = parse_jsonish(raw_text)
            record = {
                "id": item_id,
                "target_object": item.get("target_object", ""),
                "question": item.get("underspecified_question", ""),
                "image_path": item.get("image_path", ""),
                "model": args.model,
                "task": args.task,
                "keep": parsed.get("keep", ""),
                "human_candidate_count": parsed.get("human_candidate_count", ""),
                "human_expected_policy": parsed.get("human_expected_policy", ""),
                "issue_type": parsed.get("issue_type", ""),
                "notes": parsed.get("notes", ""),
                "raw_text": raw_text,
                "raw_response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            csv_rows.append(record)
            print(f"[{idx}/{len(items)}] wrote {item_id}")

    write_csv(Path(args.output_csv), csv_rows)
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

