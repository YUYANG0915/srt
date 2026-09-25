#!/usr/bin/env python3
"""Run a small VLM pilot through OpenRouter chat completions."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from prompts import make_prompt_with_item


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_image(image_dir: Path, image_id: str, image_path: str | None = None) -> Path:
    if image_path:
        path = Path(image_path).expanduser()
        if path.exists():
            return path
        if not path.is_absolute():
            relative_path = image_dir / path
            if relative_path.exists():
                return relative_path
        raise FileNotFoundError(f"image_path does not exist: {image_path}")
    candidates = [
        image_dir / f"{image_id}.jpg",
        image_dir / f"{image_id}.jpeg",
        image_dir / f"{image_id}.png",
        image_dir / f"{int(image_id):012d}.jpg" if image_id.isdigit() else image_dir / "__missing__",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No image file found for image_id={image_id} under {image_dir}")


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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


def call_openrouter(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenRouter call failed after {retries} retries: {last_error}")


def extract_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(response, ensure_ascii=False)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument(
        "--mode",
        choices=[
            "default",
            "direct",
            "evidence",
            "structured_prior",
            "visual_sketchpad",
            "marked_no_prior",
            "task3_evidence",
            "task3_marked_no_prior",
            "task3_visual_sketchpad",
            "procedural_default",
            "procedural_evidence",
            "procedural_marked_no_prior",
            "procedural_visual_sketchpad",
            "procedural_srt_generic",
            "procedural_srt_targeted_process",
            "procedural_srt_targeted_process_v2",
            "procedural_srt_targeted_process_v3",
            "procedural_srt_action_priority",
            "procedural_srt_auto_router_action",
            "procedural_boundary_classifier",
            "procedural_srt_targeted_router",
            "procedural_srt_targeted_router_v2",
            "procedural_srt_targeted_router_oracle",
            "procedural_srt_targeted_router_oracle_v2",
            "procedural_srt_gpt_verify",
            "procedural_srt_semantic_boundary",
            "procedural_srt_overload",
            "srt_spatial_default",
            "srt_spatial_evidence",
            "srt_spatial",
            "srt_spatial_v2",
            "srt_spatial_lite",
            "srt_spatial_minimal",
            "srt_spatial_verify",
            "srt_spatial_adaptive",
            "srt_spatial_contact_micro",
            "srt_spatial_vsr_rules",
            "srt_spatial_orientation_pose",
            "srt_spatial_proximity_strict",
            "srt_spatial_enclosure_open",
            "srt_spatial_above_support",
            "srt_spatial_opposition_shared_space",
            "srt_spatial_facing_away_bearing",
            "srt_spatial_behind_occlusion",
            "srt_spatial_on_top_surface",
            "srt_spatial_edge_rim",
            "srt_spatial_vsr_rules_glm",
            "srt_spatial_enclosure_open_glm",
            "srt_spatial_opposition_shared_space_glm",
            "srt_spatial_vsr_rules_glm_ultra",
            "srt_spatial_vsr_rules_glm_refined",
            "srt_spatial_vsr_rules_gpt_refined",
            "srt_spatial_containment_gpt_refined",
            "srt_spatial_vsr_rules_mistral_short",
            "srt_spatial_containment_mistral_short",
            "srt_spatial_enclosure_open_glm_ultra",
            "srt_spatial_opposition_shared_space_glm_ultra",
            "sports_action_default",
            "sports_action_generic_srt",
            "sports_action_phase_prior",
            "sports_action_visible_boundary_srt",
            "sports_action_boundary_verify",
            "sports_action_setup_guard",
            "sports_action_boundary_router",
            "tooluse_default",
            "tooluse_srt_generic",
            "tooluse_srt_scenario",
            "tooluse_srt_scenario_v2",
            "cleaning_default",
            "cleaning_srt_generic",
            "cleaning_srt_scenario",
            "cleaning_srt_scenario_v2",
            "cleaning_srt_salience_guard",
            "cleaning_srt_boundary_router_oracle",
            "craft_default",
            "craft_srt_generic",
            "craft_srt_scenario",
            "craft_srt_scenario_v2",
            "craft_srt_salience_guard",
            "craft_srt_boundary_router_oracle",
            "mobility_default",
            "mobility_srt_generic",
            "mobility_srt_scenario",
            "mobility_srt_scenario_v2",
            "mobility_srt_salience_guard",
            "mobility_srt_boundary_router_oracle",
            "process_default",
            "process_visual_evidence_only",
            "process_plain_cot",
            "process_multimodal_cot",
            "process_self_verification",
            "process_llava_cot_style",
            "process_grounded_cot",
            "process_srt_policy_only",
            "process_srt_s_t",
            "process_srt_r_t",
            "process_srt_s_r_no_t",
            "process_srt_generic",
            "process_srt_scenario",
            "process_srt_wrong_boundary_router",
            "process_srt_self_router",
            "process_srt_boundary_router_oracle",
            "process_boundary_label_only",
        ],
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="none")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--error-log", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    rows = load_jsonl(Path(args.items))
    if args.limit is not None:
        rows = rows[: args.limit]
    image_dir = Path(args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if output.exists():
        for row in load_jsonl(output):
            raw_text = str(row.get("raw_text", ""))
            parsed = parse_jsonish(raw_text)
            if not row.get("error") and (raw_text.strip() or parsed.get("policy") or parsed.get("answer")):
                done_ids.add(str(row.get("id")))

    with output.open("a", encoding="utf-8") as out:
        for idx, item in enumerate(rows, start=1):
            item_id = str(item["id"])
            if item_id in done_ids:
                continue
            question = item["underspecified_question"]
            image_path = find_image(image_dir, str(item.get("image_id", "")), item.get("image_path"))
            prompt = make_prompt_with_item(args.mode, item)
            try:
                response = call_openrouter(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    image_path=image_path,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    retries=args.retries,
                    timeout=args.timeout,
                )
                record = {
                    "id": item_id,
                    "model": args.model,
                    "mode": args.mode,
                    "image_id": item["image_id"],
                    "question": question,
                    "expected_policy": item.get("expected_policy"),
                    "expected_answer": item.get("expected_answer"),
                    "label": item.get("label"),
                    "relation": item.get("relation"),
                    "subj": item.get("subj"),
                    "obj": item.get("obj"),
                    "candidate_count_scene_graph": item.get("candidate_count_scene_graph"),
                    "raw_text": extract_text(response),
                    "raw_response": response,
                }
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_error:
                    raise
                record = {
                    "id": item_id,
                    "model": args.model,
                    "mode": args.mode,
                    "image_id": item["image_id"],
                    "question": question,
                    "expected_policy": item.get("expected_policy"),
                    "expected_answer": item.get("expected_answer"),
                    "label": item.get("label"),
                    "relation": item.get("relation"),
                    "subj": item.get("subj"),
                    "obj": item.get("obj"),
                    "candidate_count_scene_graph": item.get("candidate_count_scene_graph"),
                    "raw_text": "",
                    "raw_response": {},
                    "error": str(exc),
                }
                if args.error_log:
                    error_log = Path(args.error_log)
                    error_log.parent.mkdir(parents=True, exist_ok=True)
                    with error_log.open("a", encoding="utf-8") as err:
                        err.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{idx}/{len(rows)}] wrote {item_id}")
            if args.request_delay > 0:
                time.sleep(args.request_delay)


if __name__ == "__main__":
    main()
