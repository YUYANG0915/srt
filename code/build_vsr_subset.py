#!/usr/bin/env python3
"""Build filtered VSR subsets by relation family or exact relation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_vsr_srt_router import relation_family


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family")
    parser.add_argument("--relation")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.family and not args.relation:
        raise SystemExit("Provide at least one of --family or --relation.")

    rows = load_jsonl(Path(args.items))
    kept = []
    for row in rows:
        row_relation = str(row.get("relation", ""))
        row_family = relation_family(row_relation)
        if args.family and row_family != args.family:
            continue
        if args.relation and row_relation != args.relation:
            continue
        kept.append(row)

    if args.limit is not None:
        kept = kept[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    print(f"Wrote {len(kept)} rows to {out}")


if __name__ == "__main__":
    main()
