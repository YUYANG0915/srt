#!/usr/bin/env python3
"""Generate OpenRouter VLM suggestions for Task3 procedural/cooking annotation."""

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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_INSTRUCTIONS = """You are a careful dataset annotation reviewer for a procedural cooking/serving VLM benchmark.

Inspect the image and suggest strict visible-evidence labels.

Return only valid JSON:
{
  "keep": "yes" | "no",
  "gold_step_label": "heating_cooking" | "cutting" | "mixing" | "plating_serving" | "",
  "boundary_family": "active_heat_vs_ready_result" | "assembly_vs_serving" | "explicit_action" | "",
  "neighbor_label": "heating_cooking" | "cutting" | "mixing" | "plating_serving" | "",
  "why_not_neighbor_label": string,
  "visible_actor": string,
  "visible_tool": string,
  "visible_object": string,
  "visible_state_cues": string,
  "confidence": number,
  "notes": string
}

Definitions:
- heating_cooking: the visible scene supports active heat/cooking, e.g. food in oven, on grill/stove, or a person cooking with heat.
- cutting: a visible cutting/slicing/chopping action or its immediate setup with knife-object contact.
- mixing: a visible mixing/stirring/combining action or clear mixing container/tool state.
- plating_serving: food is being assembled for presentation, plated, displayed, served, held for eating, or already ready-to-eat.

Boundary-family rules:
- active_heat_vs_ready_result: the hard distinction is cooking-with-heat versus finished/ready food.
- assembly_vs_serving: the hard distinction is preparation/assembly/presentation versus already served or ready-to-eat.
- explicit_action: the label depends on an explicit visible action such as cutting or mixing.

Strict visible-evidence rule:
- Label what is visible, not a hidden before/after story.
- Reject with keep=no when the image is not food-process related, too ambiguous, too cropped, or the suggested label would require guessing.
- Do not infer cooking just because food usually was cooked earlier.
- Do not infer cutting or mixing unless the action or immediate action setup is visible.
- If food is simply displayed, plated, held, or being eaten, prefer plating_serving."""


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
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(response, ensure_ascii=False)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def build_prompt(row: dict[str, str]) -> str:
    return f"""Inspect the image and suggest strict procedural cooking labels.

Question: {row.get("question", "What step of the process is happening in this scene?")}
Heuristic suggested label: {row.get("suggested_step_label", "")}
Heuristic suggested boundary family: {row.get("suggested_boundary_family", "")}
Caption cues: {row.get("captions", "")}
COCO categories: {row.get("categories", "")}

Use the image as the primary evidence. Captions/categories are only weak hints.
Return only JSON."""


def call_openrouter(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
    }
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort, "exclude": True}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/vlm-underspec-pilot",
        "X-Title": "VLM Underspecification Pilot",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}")
            if exc.code in {400, 401, 403, 404}:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenRouter call failed after {retries} retries: {last_error}")


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
        "image_path",
        "source_image_id",
        "heuristic_step_label",
        "heuristic_boundary_family",
        "suggested_keep",
        "suggested_gold_step_label",
        "suggested_boundary_family",
        "suggested_neighbor_label",
        "suggested_why_not_neighbor_label",
        "suggested_visible_actor",
        "suggested_visible_tool",
        "suggested_visible_object",
        "suggested_visible_state_cues",
        "suggested_confidence",
        "suggested_notes",
        "captions",
        "categories",
        "raw_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="review/task3_procedural_cooking_candidates_v2_review.csv")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--output-jsonl", default="runs/task3_procedural_cooking_openrouter_suggestions_v2.jsonl")
    parser.add_argument("--output-csv", default="review/task3_procedural_cooking_openrouter_suggestions_v2.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="low")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    rows = read_csv(Path(args.review_csv))
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
            response = call_openrouter(
                api_key=api_key,
                model=args.model,
                prompt=build_prompt(row),
                image_path=Path(row["image_path"]).expanduser(),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                reasoning_effort=args.reasoning_effort,
                retries=args.retries,
                timeout=args.timeout,
            )
            raw_text = extract_text(response)
            parsed = parse_jsonish(raw_text)
            record = {
                "id": item_id,
                "image_path": row.get("image_path", ""),
                "source_image_id": row.get("source_image_id", ""),
                "heuristic_step_label": row.get("suggested_step_label", ""),
                "heuristic_boundary_family": row.get("suggested_boundary_family", ""),
                "model": args.model,
                "suggested_keep": parsed.get("keep", ""),
                "suggested_gold_step_label": parsed.get("gold_step_label", ""),
                "suggested_boundary_family": parsed.get("boundary_family", ""),
                "suggested_neighbor_label": parsed.get("neighbor_label", ""),
                "suggested_why_not_neighbor_label": parsed.get("why_not_neighbor_label", ""),
                "suggested_visible_actor": parsed.get("visible_actor", ""),
                "suggested_visible_tool": parsed.get("visible_tool", ""),
                "suggested_visible_object": parsed.get("visible_object", ""),
                "suggested_visible_state_cues": parsed.get("visible_state_cues", ""),
                "suggested_confidence": parsed.get("confidence", ""),
                "suggested_notes": parsed.get("notes", ""),
                "captions": row.get("captions", ""),
                "categories": row.get("categories", ""),
                "raw_text": raw_text,
                "raw_response": response,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            records.append(record)
            print(f"[{idx}/{len(rows)}] wrote {item_id}")

    write_suggestions_csv(Path(args.output_csv), records)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
