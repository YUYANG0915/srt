#!/usr/bin/env python3
"""Build a controlled Task3 pilot for camera-holder / viewpoint priors.

The generated images are intentionally simple, controlled scenes. They test
whether VLMs answer participant-count questions from only directly visible
people or whether they use first-person viewpoint cues as a structured prior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CANVAS = (900, 620)
ORIGINAL_DIR_NAME = "task3_camera_prior_images"
MARKED_DIR_NAME = "task3_camera_prior_marked_images"


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    activity: str
    surface: str
    first_person: bool
    visible_count: int
    expected_total: int
    visible_actor: str
    cue: str
    palette: tuple[str, str, str]


SCENES = [
    SceneSpec("ski_fp_01", "skiing", "snowy slope", True, 1, 2, "skier ahead", "first-person skis and poles", ("#d8f3ff", "#f7fbff", "#e63946")),
    SceneSpec("ski_fp_02", "skiing", "mountain trail", True, 1, 2, "skier ahead", "first-person skis", ("#c9e8ff", "#ffffff", "#457b9d")),
    SceneSpec("snowboard_fp_01", "snowboarding", "snow park", True, 1, 2, "snowboarder ahead", "first-person snowboard nose", ("#dff7ff", "#f8fbff", "#2a9d8f")),
    SceneSpec("cycle_fp_01", "cycling", "forest bike trail", True, 1, 2, "cyclist ahead", "first-person handlebars", ("#d9f2d0", "#8cc084", "#f4a261")),
    SceneSpec("cycle_fp_02", "cycling", "road", True, 1, 2, "cyclist ahead", "first-person handlebars and gloves", ("#dbeafe", "#94a3b8", "#2563eb")),
    SceneSpec("surf_fp_01", "surfing", "ocean wave", True, 1, 2, "surfer ahead", "first-person surfboard nose", ("#bde0fe", "#48cae4", "#ffb703")),
    SceneSpec("kayak_fp_01", "kayaking", "river", True, 1, 2, "kayaker ahead", "first-person paddle and kayak bow", ("#caf0f8", "#00b4d8", "#fb8500")),
    SceneSpec("skate_fp_01", "skateboarding", "skate park", True, 1, 2, "skateboarder ahead", "first-person skateboard nose", ("#e5e7eb", "#9ca3af", "#7c3aed")),
    SceneSpec("ski_third_01", "skiing", "snowy slope", False, 1, 1, "skier", "none", ("#d8f3ff", "#f7fbff", "#e63946")),
    SceneSpec("snowboard_third_01", "snowboarding", "snow park", False, 1, 1, "snowboarder", "none", ("#dff7ff", "#f8fbff", "#2a9d8f")),
    SceneSpec("cycle_third_01", "cycling", "forest bike trail", False, 1, 1, "cyclist", "none", ("#d9f2d0", "#8cc084", "#f4a261")),
    SceneSpec("surf_third_01", "surfing", "ocean wave", False, 1, 1, "surfer", "none", ("#bde0fe", "#48cae4", "#ffb703")),
]


def safe_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in paths:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_person(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float, color: str, activity: str) -> dict[str, int]:
    head_r = round(20 * scale)
    torso_h = round(70 * scale)
    torso_w = round(36 * scale)
    leg_len = round(58 * scale)
    arm_len = round(46 * scale)

    draw.ellipse([x - head_r, y - head_r, x + head_r, y + head_r], fill="#f3c7a4", outline="#1f2937", width=3)
    draw.line([x, y + head_r, x, y + head_r + torso_h], fill=color, width=max(5, round(12 * scale)))
    draw.line([x - torso_w, y + head_r + 20, x + torso_w, y + head_r + 20], fill=color, width=max(4, round(9 * scale)))
    draw.line([x - torso_w, y + head_r + 20, x - torso_w - arm_len, y + head_r + 55], fill=color, width=max(4, round(8 * scale)))
    draw.line([x + torso_w, y + head_r + 20, x + torso_w + arm_len, y + head_r + 55], fill=color, width=max(4, round(8 * scale)))
    hip_y = y + head_r + torso_h
    draw.line([x, hip_y, x - 34 * scale, hip_y + leg_len], fill="#1f2937", width=max(4, round(9 * scale)))
    draw.line([x, hip_y, x + 34 * scale, hip_y + leg_len], fill="#1f2937", width=max(4, round(9 * scale)))

    foot_y = hip_y + leg_len
    if activity in {"skiing", "snowboarding"}:
        if activity == "skiing":
            draw.line([x - 80 * scale, foot_y + 8, x + 75 * scale, foot_y + 8], fill="#111827", width=max(3, round(5 * scale)))
            draw.line([x - 65 * scale, foot_y + 24, x + 90 * scale, foot_y + 24], fill="#111827", width=max(3, round(5 * scale)))
        else:
            draw.rounded_rectangle([x - 75 * scale, foot_y + 8, x + 80 * scale, foot_y + 28], radius=8, fill="#111827")
    elif activity == "cycling":
        draw.ellipse([x - 90 * scale, foot_y - 10, x - 35 * scale, foot_y + 45], outline="#111827", width=max(3, round(5 * scale)))
        draw.ellipse([x + 35 * scale, foot_y - 10, x + 90 * scale, foot_y + 45], outline="#111827", width=max(3, round(5 * scale)))
        draw.line([x - 62 * scale, foot_y + 18, x, foot_y - 18, x + 62 * scale, foot_y + 18], fill="#111827", width=max(3, round(5 * scale)))
    elif activity == "surfing":
        draw.ellipse([x - 85 * scale, foot_y + 4, x + 95 * scale, foot_y + 38], fill="#fef3c7", outline="#111827", width=3)
    elif activity == "kayaking":
        draw.ellipse([x - 100 * scale, foot_y - 10, x + 100 * scale, foot_y + 48], fill="#f97316", outline="#111827", width=3)
        draw.line([x - 110 * scale, y + 40 * scale, x + 110 * scale, y + 90 * scale], fill="#111827", width=5)
    elif activity == "skateboarding":
        draw.rounded_rectangle([x - 85 * scale, foot_y + 10, x + 90 * scale, foot_y + 28], radius=8, fill="#111827")

    return {
        "x0": round(x - 120 * scale),
        "y0": round(y - 30 * scale),
        "x1": round(x + 120 * scale),
        "y1": round(foot_y + 55 * scale),
    }


def draw_first_person_cue(draw: ImageDraw.ImageDraw, spec: SceneSpec) -> dict[str, int] | None:
    if not spec.first_person:
        return None
    w, h = CANVAS
    if spec.activity == "skiing":
        draw.line([w * 0.38, h * 0.73, w * 0.25, h * 0.98], fill="#111827", width=12)
        draw.line([w * 0.62, h * 0.73, w * 0.75, h * 0.98], fill="#111827", width=12)
        draw.line([w * 0.32, h * 0.70, w * 0.18, h * 0.92], fill="#6b7280", width=5)
        draw.line([w * 0.68, h * 0.70, w * 0.82, h * 0.92], fill="#6b7280", width=5)
    elif spec.activity == "snowboarding":
        draw.rounded_rectangle([w * 0.30, h * 0.78, w * 0.70, h * 0.94], radius=34, fill="#111827")
        draw.ellipse([w * 0.42, h * 0.80, w * 0.48, h * 0.88], fill="#f3c7a4")
        draw.ellipse([w * 0.53, h * 0.80, w * 0.59, h * 0.88], fill="#f3c7a4")
    elif spec.activity == "cycling":
        draw.arc([w * 0.22, h * 0.58, w * 0.78, h * 0.92], 195, 345, fill="#111827", width=16)
        draw.ellipse([w * 0.20, h * 0.66, w * 0.30, h * 0.78], fill="#f3c7a4", outline="#111827", width=3)
        draw.ellipse([w * 0.70, h * 0.66, w * 0.80, h * 0.78], fill="#f3c7a4", outline="#111827", width=3)
    elif spec.activity == "surfing":
        draw.ellipse([w * 0.28, h * 0.72, w * 0.72, h * 1.05], fill="#fef3c7", outline="#111827", width=4)
        draw.line([w * 0.50, h * 0.75, w * 0.50, h * 0.98], fill="#111827", width=4)
    elif spec.activity == "kayaking":
        draw.polygon([(w * 0.50, h * 0.68), (w * 0.28, h * 0.98), (w * 0.72, h * 0.98)], fill="#fb8500", outline="#111827")
        draw.line([w * 0.18, h * 0.76, w * 0.82, h * 0.88], fill="#111827", width=8)
    elif spec.activity == "skateboarding":
        draw.rounded_rectangle([w * 0.30, h * 0.80, w * 0.70, h * 0.91], radius=26, fill="#111827")
        draw.ellipse([w * 0.34, h * 0.88, w * 0.40, h * 0.95], fill="#374151")
        draw.ellipse([w * 0.60, h * 0.88, w * 0.66, h * 0.95], fill="#374151")
    return {"x0": 150, "y0": 355, "x1": 750, "y1": 610}


def draw_scene(spec: SceneSpec, path: Path, marked_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sky, ground, accent = spec.palette
    image = Image.new("RGB", CANVAS, sky)
    draw = ImageDraw.Draw(image)
    w, h = CANVAS

    draw.rectangle([0, h * 0.45, w, h], fill=ground)
    if spec.activity in {"surfing", "kayaking"}:
        for i in range(8):
            y = round(h * 0.52 + i * 32)
            draw.arc([20, y - 24, w - 20, y + 24], 0, 180, fill="#ffffff", width=4)
    elif spec.activity == "cycling":
        draw.polygon([(0, h), (w * 0.42, h * 0.45), (w * 0.58, h * 0.45), (w, h)], fill="#6b7280")
        draw.line([w * 0.50, h * 0.45, w * 0.50, h], fill="#fef3c7", width=4)
    elif spec.activity == "skateboarding":
        draw.rectangle([0, h * 0.45, w, h], fill="#cbd5e1")
        draw.polygon([(80, h * 0.75), (280, h * 0.55), (360, h * 0.75)], fill="#94a3b8", outline="#334155")
    else:
        draw.polygon([(0, h), (w * 0.35, h * 0.42), (w, h)], fill=ground)
        draw.polygon([(0, h * 0.55), (w * 0.28, h * 0.20), (w * 0.55, h * 0.55)], fill="#edf2f7")
        draw.polygon([(w * 0.42, h * 0.55), (w * 0.72, h * 0.18), (w, h * 0.55)], fill="#e2e8f0")

    actor_box = draw_person(draw, 465, 225, 1.15, accent, spec.activity)
    cue_box = draw_first_person_cue(draw, spec)

    image.save(path)

    marked = image.copy()
    marked_draw = ImageDraw.Draw(marked, "RGBA")
    font = safe_font(24, bold=True)
    small = safe_font(18, bold=True)
    boxes = [
        {
            "mark_id": "visible_participant_1",
            "bbox_xyxy": [actor_box["x0"], actor_box["y0"], actor_box["x1"], actor_box["y1"]],
            "kind": "visible_participant",
            "description": spec.visible_actor,
        }
    ]
    if cue_box:
        boxes.append(
            {
                "mark_id": "viewpoint_cue_1",
                "bbox_xyxy": [cue_box["x0"], cue_box["y0"], cue_box["x1"], cue_box["y1"]],
                "kind": "viewpoint_cue",
                "description": spec.cue,
            }
        )
    colors = {"visible_participant": (230, 57, 70), "viewpoint_cue": (29, 112, 184)}
    for box in boxes:
        x0, y0, x1, y1 = box["bbox_xyxy"]
        color = colors[box["kind"]]
        marked_draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=5)
        label = box["mark_id"]
        tb = marked_draw.textbbox((0, 0), label, font=font)
        marked_draw.rectangle([x0, max(0, y0 - 34), x0 + tb[2] + 12, max(34, y0)], fill=color + (230,))
        marked_draw.text((x0 + 6, max(0, y0 - 31)), label, fill=(255, 255, 255, 255), font=small)

    footer = "Sketchpad visual action: marked visible participants and first-person viewpoint cues."
    footer_font = safe_font(18, bold=True)
    fb = marked_draw.textbbox((0, 0), footer, font=footer_font)
    marked_draw.rectangle([0, h - 34, w, h], fill=(0, 0, 0, 150))
    marked_draw.text((10, h - 28), footer, fill=(255, 255, 255, 255), font=footer_font)
    marked.save(marked_path)
    return boxes, boxes


def row_for(spec: SceneSpec, image_path: Path, marked_path: Path, marks: list[dict[str, Any]], marked: bool) -> dict[str, Any]:
    question = f"How many people are {spec.activity} in this scene?"
    base = {
        "id": f"task3_{spec.scene_id}",
        "image_id": spec.scene_id,
        "source": "controlled_synthetic_camera_prior",
        "task": "task3_camera_holder_prior",
        "underspecified_question": question,
        "original_question": question,
        "activity": spec.activity,
        "surface": spec.surface,
        "visible_participant_count": spec.visible_count,
        "expected_total_participants": spec.expected_total,
        "expected_policy": "answer",
        "expected_reasoning": "include_hidden_camera_holder" if spec.first_person else "count_visible_only",
        "hidden_camera_holder_expected": spec.first_person,
        "main_error": "visible_only_count" if spec.first_person else "over_infer_hidden_camera_holder",
        "answer": str(spec.expected_total),
        "manual_status": "synthetic_controlled",
        "notes": (
            "First-person body/equipment cue implies the camera wearer is also participating."
            if spec.first_person
            else "Third-person control scene has no first-person participation cue."
        ),
    }
    if marked:
        base["image_path"] = str(marked_path.resolve())
        base["original_image_path"] = str(image_path.resolve())
        base["sketchpad_action"] = "mark_visible_participants_and_viewpoint_cues"
        base["marked_candidates"] = marks
        base["marked_candidate_count"] = len(marks)
    else:
        base["image_path"] = str(image_path.resolve())
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-clean", default="data/task3_camera_prior_clean_items.jsonl")
    parser.add_argument("--output-marked", default="data/task3_camera_prior_marked_items.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    original_dir = data_dir / ORIGINAL_DIR_NAME
    marked_dir = data_dir / MARKED_DIR_NAME
    original_dir.mkdir(parents=True, exist_ok=True)
    marked_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = []
    marked_rows = []
    for spec in SCENES:
        image_path = original_dir / f"{spec.scene_id}.jpg"
        marked_path = marked_dir / f"{spec.scene_id}_marked.jpg"
        marks, _ = draw_scene(spec, image_path, marked_path)
        clean_rows.append(row_for(spec, image_path, marked_path, marks, marked=False))
        marked_rows.append(row_for(spec, image_path, marked_path, marks, marked=True))

    Path(args.output_clean).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_rows),
        encoding="utf-8",
    )
    Path(args.output_marked).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in marked_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(clean_rows)} clean Task3 items to {args.output_clean}")
    print(f"Wrote {len(marked_rows)} marked Task3 items to {args.output_marked}")
    print(f"Wrote images to {original_dir} and {marked_dir}")


if __name__ == "__main__":
    main()
