#!/usr/bin/env python3
"""Build a small procedural-cooking bootstrap review set from local COCO images."""

from __future__ import annotations

import csv
import html
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "coco_annotations" / "annotations_trainval2017.zip"
IMAGE_DIRS = [
    DATA_DIR / "coco_val2017_task2_raw500_images",
    DATA_DIR / "coco_val2017_pilot_images",
    DATA_DIR / "coco_val2017_unique_strict_images",
]
OUT_CSV = ROOT / "review" / "task3_procedural_cooking_bootstrap_candidates.csv"
OUT_HTML = ROOT / "review" / "task3_procedural_cooking_bootstrap_candidates.html"
SEED_CSV = ROOT / "review" / "task3_procedural_cooking_seed_review.csv"
SEED_HTML = ROOT / "review" / "task3_procedural_cooking_seed_review.html"

SEED_OVERRIDES = {
    258883: ("heating_cooking", "person actively grilling/topping flatbread on grill"),
    309678: ("plating_serving", "person placing tomato/topping onto sauce-covered tray"),
    330818: ("heating_cooking", "person working at commercial stove/range in kitchen"),
    346703: ("plating_serving", "serving/flambe-style cake action sequence in single collage frame"),
    402433: ("heating_cooking", "pizza visibly in oven/tray during cooking stage"),
    470773: ("plating_serving", "chef-side pastry station with toppings and ready-to-serve items"),
    279541: ("plating_serving", "pizzas arranged in buffet/display serving stage"),
    226903: ("plating_serving", "food counter with serving utensils and ready items"),
    63965: ("plating_serving", "holding plated cake slice ready for serving"),
    353027: ("plating_serving", "pizza slice being lifted from pie, serving stage"),
}


@dataclass
class Candidate:
    image_id: int
    image_path: str
    captions: list[str]
    categories: list[str]
    trigger_words: list[str]
    suggested_step_label: str
    suggestion_reason: str
    priority_score: int


def load_coco_metadata() -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        caps = json.load(io.TextIOWrapper(zf.open("annotations/captions_val2017.json"), encoding="utf-8"))
        inst = json.load(io.TextIOWrapper(zf.open("annotations/instances_val2017.json"), encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    caps_by_img: dict[int, list[str]] = defaultdict(list)
    cats_by_img: dict[int, list[str]] = defaultdict(list)

    for ann in caps["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])
    for ann in inst["annotations"]:
        cats_by_img[ann["image_id"]].append(cat_name[ann["category_id"]])
    return caps_by_img, cats_by_img


def find_local_images() -> dict[int, str]:
    image_map: dict[int, str] = {}
    for image_dir in IMAGE_DIRS:
        for path in image_dir.glob("*.jpg"):
            image_map[int(path.stem)] = str(path.resolve())
    return image_map


def infer_step(captions: list[str], categories: list[str]) -> tuple[str, str, list[str], int] | None:
    text = " ".join(captions).lower()
    cats = set(categories)
    food_cats = {"banana", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
    kitchen_cats = {"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "dining table", "oven"}
    if not (cats & food_cats):
        return None

    keywords: list[str] = []
    score = 0

    def has(*terms: str) -> bool:
        return any(term in text for term in terms)

    if has("grill", "grilling", "cooking", "working at a range", "oven") or (
        "pizza" in cats and has("oven")
    ):
        if has("grill"):
            keywords.append("grill")
            score += 3
        if has("cooking", "grilling"):
            keywords.append("cooking")
            score += 3
        if has("oven"):
            keywords.append("oven")
            score += 2
        return "heating_cooking", "caption suggests active heat-based cooking", keywords, score + 3

    if has("holding a tomato", "tray that has sauce", "topped with", "get topped"):
        if has("holding a tomato"):
            keywords.append("holding a tomato")
        if has("topped with", "get topped"):
            keywords.append("topping")
        return "plating_serving", "caption suggests ingredient placement / final assembly", keywords, 6

    if has("pastry station", "buffet", "serving utensils", "holding a plate of cake"):
        if has("pastry station"):
            keywords.append("pastry station")
        if has("buffet"):
            keywords.append("buffet")
        if has("holding a plate of cake"):
            keywords.append("plate of cake")
        return "plating_serving", "caption suggests serving or food presentation", keywords, 5

    if has("cut", "cutting", "chopping", "slice of pizza", "taking a slice"):
        if has("cut", "cutting", "chopping"):
            keywords.append("cutting")
            return "cutting", "caption explicitly suggests cutting/chopping", keywords, 6
        keywords.append("slice")
        return "plating_serving", "caption suggests serving a sliced item rather than active cooking", keywords, 4

    if "person" in cats and (cats & {"pizza", "cake", "sandwich", "hot dog"}) and (
        cats & {"spoon", "knife", "bowl", "fork", "dining table"}
    ):
        keywords.append("food+tool")
        return "plating_serving", "food scene with person and serving/prep tools", keywords, 3

    return None


def build_candidates() -> list[Candidate]:
    image_map = find_local_images()
    caps_by_img, cats_by_img = load_coco_metadata()

    candidates: list[Candidate] = []
    for image_id, image_path in image_map.items():
        captions = caps_by_img.get(image_id, [])
        categories = sorted(set(cats_by_img.get(image_id, [])))
        inferred = infer_step(captions, categories)
        if not inferred:
            continue
        step, reason, trigger_words, score = inferred
        candidates.append(
            Candidate(
                image_id=image_id,
                image_path=image_path,
                captions=captions[:3],
                categories=categories,
                trigger_words=trigger_words,
                suggested_step_label=step,
                suggestion_reason=reason,
                priority_score=score,
            )
        )

    # Keep the strongest items first and remove a few obvious non-process edge cases.
    candidates.sort(key=lambda c: (-c.priority_score, c.image_id))
    filtered: list[Candidate] = []
    for cand in candidates:
        text = " ".join(cand.captions).lower()
        if "eating" in text and cand.suggested_step_label != "heating_cooking":
            continue
        filtered.append(cand)
    return filtered


def write_csv(candidates: list[Candidate]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "image_path",
                "keep",
                "scenario_domain",
                "activity_name",
                "question",
                "gold_step_label",
                "visible_actor",
                "visible_tool",
                "visible_object",
                "visible_state_cues",
                "actor_box",
                "tool_box",
                "object_box",
                "state_box",
                "issue_type",
                "notes",
                "source_dataset",
                "source_image_id",
                "suggested_step_label",
                "suggestion_reason",
                "captions",
                "categories",
                "trigger_words",
                "priority_score",
            ]
        )
        for idx, cand in enumerate(candidates, start=1):
            writer.writerow(
                [
                    f"task3cook_{idx:04d}_{cand.image_id}",
                    cand.image_path,
                    "",
                    "cooking",
                    "cooking_or_food_preparation",
                    "What step of the process is happening in this scene?",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "coco_val2017_local_subset",
                    cand.image_id,
                    cand.suggested_step_label,
                    cand.suggestion_reason,
                    " || ".join(cand.captions),
                    " | ".join(cand.categories),
                    " | ".join(cand.trigger_words),
                    cand.priority_score,
                ]
            )


def write_review_csv(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "image_path",
                "keep",
                "scenario_domain",
                "activity_name",
                "question",
                "gold_step_label",
                "visible_actor",
                "visible_tool",
                "visible_object",
                "visible_state_cues",
                "actor_box",
                "tool_box",
                "object_box",
                "state_box",
                "issue_type",
                "notes",
                "source_dataset",
                "source_image_id",
                "suggested_step_label",
                "suggestion_reason",
                "captions",
                "categories",
                "trigger_words",
                "priority_score",
            ]
        )
        for idx, cand in enumerate(candidates, start=1):
            writer.writerow(
                [
                    f"task3cook_seed_{idx:04d}_{cand.image_id}",
                    cand.image_path,
                    "",
                    "cooking",
                    "cooking_or_food_preparation",
                    "What step of the process is happening in this scene?",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "coco_val2017_local_subset",
                    cand.image_id,
                    cand.suggested_step_label,
                    cand.suggestion_reason,
                    " || ".join(cand.captions),
                    " | ".join(cand.categories),
                    " | ".join(cand.trigger_words),
                    cand.priority_score,
                ]
            )


def write_html(path: Path, title: str, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for idx, cand in enumerate(candidates, start=1):
        captions_html = "".join(f"<li>{html.escape(cap)}</li>" for cap in cand.captions)
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(cand.image_path)}" alt="{cand.image_id}">
              <div class="body">
                <h3>{idx}. image {cand.image_id}</h3>
                <p><b>suggested label:</b> {html.escape(cand.suggested_step_label)}</p>
                <p><b>reason:</b> {html.escape(cand.suggestion_reason)}</p>
                <p><b>categories:</b> {html.escape(", ".join(cand.categories))}</p>
                <p><b>triggers:</b> {html.escape(", ".join(cand.trigger_words))}</p>
                <p><b>score:</b> {cand.priority_score}</p>
                <ul>{captions_html}</ul>
                <p class="path">{html.escape(cand.image_path)}</p>
              </div>
            </div>
            """
        )

    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }}
    .card {{ border: 1.5px solid #222; border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ width: 100%; display: block; background: #eee; }}
    .body {{ padding: 14px; }}
    h3 {{ margin: 0 0 8px; font-size: 22px; }}
    p, li {{ font-size: 16px; line-height: 1.35; }}
    ul {{ padding-left: 20px; }}
    .path {{ color: #555; font-size: 13px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Total candidates: {len(candidates)}</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_seed_candidates(image_map: dict[int, str], caps_by_img: dict[int, list[str]], cats_by_img: dict[int, list[str]]) -> list[Candidate]:
    seeds: list[Candidate] = []
    for image_id, (label, reason) in SEED_OVERRIDES.items():
        image_path = image_map.get(image_id)
        if not image_path:
            continue
        captions = caps_by_img.get(image_id, [])[:3]
        categories = sorted(set(cats_by_img.get(image_id, [])))
        seeds.append(
            Candidate(
                image_id=image_id,
                image_path=image_path,
                captions=captions,
                categories=categories,
                trigger_words=["seed_override"],
                suggested_step_label=label,
                suggestion_reason=reason,
                priority_score=100,
            )
        )
    return seeds


def main() -> None:
    image_map = find_local_images()
    caps_by_img, cats_by_img = load_coco_metadata()
    candidates = build_candidates()
    write_csv(candidates)
    write_html(OUT_HTML, "Task3 Procedural Cooking Bootstrap Candidates", candidates)
    seed_candidates = build_seed_candidates(image_map, caps_by_img, cats_by_img)
    write_review_csv(SEED_CSV, seed_candidates)
    write_html(SEED_HTML, "Task3 Procedural Cooking Seed Review", seed_candidates)
    print(f"Wrote {len(candidates)} candidates to {OUT_CSV}")
    print(f"Wrote gallery to {OUT_HTML}")
    print(f"Wrote {len(seed_candidates)} seed review rows to {SEED_CSV}")
    print(f"Wrote seed gallery to {SEED_HTML}")


if __name__ == "__main__":
    main()
