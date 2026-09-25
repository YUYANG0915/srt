#!/usr/bin/env python3
"""Build manually reviewable underspecified-question candidates from GQA.

The script is deliberately conservative: it only rewrites simple "What color is
the <object> <spatial relation> ..." questions into "What color is the <object>?"
and uses the GQA scene graph to count same-class candidate objects.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SPATIAL_RE = re.compile(
    r"^(?P<prefix>what color (?:is|are) the (?P<object>[a-z0-9 -]+?)) "
    r"(?P<relation>to the (?:left|right) of|in front of|behind|next to|near|beside) "
    r"(?P<rest>.+?)\?$",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_questions(raw: Any):
    if isinstance(raw, dict):
        for qid, row in raw.items():
            if isinstance(row, dict):
                yield str(qid), row
    elif isinstance(raw, list):
        for idx, row in enumerate(raw):
            if isinstance(row, dict):
                yield str(row.get("question_id") or row.get("id") or idx), row


def normalize_object_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .?")


def get_image_id(row: dict[str, Any]) -> str | None:
    for key in ("imageId", "image_id", "image"):
        if key in row:
            return str(row[key])
    return None


def candidate_objects(scene_graphs: dict[str, Any], image_id: str, object_name: str):
    graph = scene_graphs.get(str(image_id)) or scene_graphs.get(image_id)
    objects = (graph or {}).get("objects", {})
    target = normalize_object_name(object_name)
    matches = []
    for object_id, obj in objects.items():
        name = normalize_object_name(str(obj.get("name", "")))
        if name == target or name.endswith(" " + target) or target.endswith(" " + name):
            matches.append(
                {
                    "object_id": str(object_id),
                    "name": obj.get("name"),
                    "attributes": obj.get("attributes", []),
                    "x": obj.get("x"),
                    "y": obj.get("y"),
                    "w": obj.get("w"),
                    "h": obj.get("h"),
                }
            )
    return matches


def resolve_image_path(image_dir: str | None, image_id: str) -> str:
    if not image_dir:
        return ""
    root = Path(image_dir)
    for suffix in (".jpg", ".jpeg", ".png"):
        path = root / f"{image_id}{suffix}"
        if path.exists():
            return str(path.resolve())
    return str((root / f"{image_id}.jpg").resolve())


def build_candidates(args: argparse.Namespace) -> int:
    questions = load_json(Path(args.questions))
    scene_graphs = load_json(Path(args.scene_graphs))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as out:
        for qid, row in iter_questions(questions):
            question = str(row.get("question", "")).strip()
            match = SPATIAL_RE.match(question)
            if not match:
                continue
            image_id = get_image_id(row)
            if image_id is None:
                continue

            object_name = normalize_object_name(match.group("object"))
            candidates = candidate_objects(scene_graphs, image_id, object_name)
            if args.task == "ambiguous" and len(candidates) < args.min_candidates:
                continue
            if args.task == "unique" and len(candidates) != 1:
                continue

            underspecified = f"{match.group('prefix')}?"
            item = {
                "id": f"gqa_{args.task}_{qid}",
                "image_id": image_id,
                "image_path": resolve_image_path(args.image_dir, image_id),
                "source": "gqa",
                "original_question": question,
                "underspecified_question": underspecified[0].upper() + underspecified[1:],
                "answer": row.get("answer"),
                "target_object": object_name,
                "relation_phrase": f"{match.group('relation')} {match.group('rest')}",
                "candidate_count_scene_graph": len(candidates),
                "candidate_objects_scene_graph": candidates,
                "expected_policy": "answer" if args.task == "unique" else "clarify",
                "manual_status": "needs_review",
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--scene-graphs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--task", choices=["ambiguous", "unique"], default="ambiguous")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-candidates", type=int, default=2)
    args = parser.parse_args()
    count = build_candidates(args)
    print(f"Wrote {count} candidates to {args.output}")


if __name__ == "__main__":
    main()
