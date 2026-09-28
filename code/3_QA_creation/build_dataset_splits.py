#!/usr/bin/env python3
"""Validate and split curated clause-grounded QA records reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


CATEGORIES = {"single", "cross", "correlation"}
REQUIRED_FIELDS = {
    "id", "question", "answer", "type", "standard_ids", "clause_refs",
    "supporting_facts",
}


def canonical_key(item: dict) -> str:
    explicit = str(item.get("dedup_key") or "").strip()
    if explicit:
        return explicit
    payload = " ".join(str(item.get(key) or "").split() for key in ("question", "answer"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate(item: dict) -> None:
    missing = sorted(REQUIRED_FIELDS - set(item))
    if missing:
        raise ValueError(f"{item.get('id', '<unknown>')}: missing {missing}")
    category = str(item["type"]).strip().lower()
    if category not in CATEGORIES:
        raise ValueError(f"{item['id']}: unsupported category {category!r}")
    refs = item.get("clause_refs") or []
    if not refs or any(not ref.get("evidence_id") for ref in refs):
        raise ValueError(f"{item['id']}: every item requires canonical clause evidence")


def stratified_split(records: list[dict], sizes: tuple[int, int, int], seed: int) -> dict:
    for item in records:
        validate(item)
    keys = [canonical_key(item) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate QA records detected before splitting")

    total = sum(sizes)
    if len(records) != total:
        raise ValueError(f"expected {total} records, found {len(records)}")
    labels = [item["type"] for item in records]
    train, held_out = train_test_split(
        records, train_size=sizes[0], random_state=seed, stratify=labels
    )
    held_labels = [item["type"] for item in held_out]
    dev, test = train_test_split(
        held_out, train_size=sizes[1], test_size=sizes[2],
        random_state=seed, stratify=held_labels,
    )
    return {"train": train, "dev": dev, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="curated JSON list containing exactly 800 QA records")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    splits = stratified_split(records, (500, 100, 200), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "categories": sorted(CATEGORIES), "splits": {}}
    for name, rows in splits.items():
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["splits"][name] = {
            "size": len(rows),
            "categories": dict(sorted(Counter(row["type"] for row in rows).items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
