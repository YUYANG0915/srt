#!/usr/bin/env python3
"""Generate OpenRouter VLM suggestions for Task3 real egocentric review rows."""

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

SYSTEM_INSTRUCTIONS = """You are a strict dataset annotation reviewer for a camera-holder/viewpoint participation benchmark.

The task asks whether an image can support a participant-count question such as:
"How many people are cycling/skiing/surfing/skateboarding/etc. in this scene?"

Return only valid JSON:
{
  "keep": "yes" | "no",
  "split": "first_person_prior" | "third_person_control" | "",
  "activity": string,
  "question": string,
  "visible_participant_count": number,
  "implied_camera_holder_count": number,
  "expected_total_participants": number,
  "expected_reasoning": "include_hidden_camera_holder" | "count_visible_only" | "",
  "first_person_cues": string,
  "issue_type": "clean" | "unclear_activity" | "not_activity_count" | "too_ambiguous" | "bad_image" | "",
  "confidence": number,
  "notes": string
}

Rules:
- Use keep=yes only if the image supports a clear activity participant-count item.
- first_person_prior means the camera/viewpoint itself provides strong participation evidence: visible hands, arms, legs, skis, bicycle handlebars, surfboard, kayak bow/paddle, motorcycle/bike cockpit, or other first-person body/equipment cues compatible with the activity.
- third_person_control means the image shows activity participants from an outside view and has no first-person participation cue.
- visible_participant_count counts directly visible people doing the activity.
- implied_camera_holder_count is 1 only for strong first-person participation cues; otherwise 0.
- expected_total_participants = visible_participant_count + implied_camera_holder_count.
- Reject static object scenes, unclear activities, images where people are not doing a countable activity, or cases where the camera holder is merely a photographer not participating.
- Do not infer a hidden participant without strong first-person activity cues."""


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


def call_openrouter(
    *,
    api_key: str,
    model: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    prompt = "Inspect this image and decide whether it is useful for the camera-holder/viewpoint participation benchmark. Return only JSON."
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
        "suggested_keep",
        "suggested_split",
        "suggested_activity",
        "suggested_question",
        "suggested_visible_participant_count",
        "suggested_implied_camera_holder_count",
        "suggested_expected_total_participants",
        "suggested_expected_reasoning",
        "suggested_first_person_cues",
        "suggested_issue_type",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="review/task3_real_egocentric_real_images_review.csv")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--output-jsonl", default="runs/task3_real_egocentric_real_images_openrouter_suggestions.jsonl")
    parser.add_argument("--output-csv", default="review/task3_real_egocentric_real_images_openrouter_suggestions.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=550)
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
                "model": args.model,
                "suggested_keep": parsed.get("keep", ""),
                "suggested_split": parsed.get("split", ""),
                "suggested_activity": parsed.get("activity", ""),
                "suggested_question": parsed.get("question", ""),
                "suggested_visible_participant_count": parsed.get("visible_participant_count", ""),
                "suggested_implied_camera_holder_count": parsed.get("implied_camera_holder_count", ""),
                "suggested_expected_total_participants": parsed.get("expected_total_participants", ""),
                "suggested_expected_reasoning": parsed.get("expected_reasoning", ""),
                "suggested_first_person_cues": parsed.get("first_person_cues", ""),
                "suggested_issue_type": parsed.get("issue_type", ""),
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
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
