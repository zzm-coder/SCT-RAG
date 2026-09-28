#!/usr/bin/env python3
"""Bind extracted triples to canonical source clauses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
from clause_ground_truth import alignment_score, compact_text


DEFAULT_TRIPLES = CODE_ROOT / "1_extract_data" / "kg_data" / "triplet_cache.json"
DEFAULT_CLAUSES = HERE / "clause_records" / "clauses.jsonl"
DEFAULT_OUTPUT = HERE / "kg_clause_cache"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_clauses(paragraph: str, clauses: list[dict], limit: int = 12) -> list[dict]:
    """Use character 4-gram overlap to bound expensive fuzzy comparisons."""
    p = compact_text(paragraph)
    if not p:
        return []
    exact = []
    scored = []
    pgrams = {p[i : i + 4] for i in range(max(1, len(p) - 3))}
    for clause in clauses:
        c = compact_text(clause.get("raw_text", ""))
        if not c:
            continue
        if p in c or c in p:
            exact.append(clause)
            continue
        cgrams = {c[i : i + 4] for i in range(max(1, len(c) - 3))}
        overlap = len(pgrams & cgrams) / max(1, len(pgrams))
        scored.append((overlap, clause))
    scored.sort(key=lambda row: row[0], reverse=True)
    return exact + [row[1] for row in scored[:limit]]


def align_paragraph(paragraph: str, clauses: list[dict]) -> dict:
    candidates = candidate_clauses(paragraph, clauses)
    ranked = []
    for clause in candidates:
        score, method = alignment_score(paragraph, clause.get("raw_text", ""))
        ranked.append((score, method, clause))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked:
        return {"status": "unresolved", "score": 0.0, "method": "no_clause"}
    score, method, clause = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = score - second
    if score >= 0.90 and margin >= 0.02:
        status = "auto_high"
    elif score >= 0.65 and margin >= 0.01:
        status = "auto_medium"
    else:
        status = "needs_review"
    return {
        "status": status,
        "score": round(score, 6),
        "runner_up_score": round(second, 6),
        "margin": round(margin, 6),
        "method": method,
        "standard_id": clause.get("standard_id"),
        "standard_version": clause.get("standard_version"),
        "clause_id": clause.get("clause_id"),
        "clause_title": clause.get("clause_title"),
        "evidence_id": clause.get("evidence_id"),
    }


def build(triples_path: Path, clauses_path: Path, output_dir: Path) -> dict:
    clauses = [json.loads(line) for line in clauses_path.read_text(encoding="utf-8").splitlines() if line]
    by_source = defaultdict(list)
    for clause in clauses:
        by_source[clause["source_file"]].append(clause)
    triples_by_source = json.loads(triples_path.read_text(encoding="utf-8"))

    paragraph_alignment = {}
    relevant_rows = []
    for values in triples_by_source.values():
        for triple in values:
            source = triple.get("source")
            paragraph = triple.get("paragraph")
            if source in by_source and paragraph:
                relevant_rows.append(triple)
                paragraph_alignment.setdefault((source, paragraph), None)
    for source, paragraph in paragraph_alignment:
        paragraph_alignment[(source, paragraph)] = align_paragraph(paragraph, by_source[source])

    linked = defaultdict(list)
    status_counts = Counter()
    for triple in relevant_rows:
        row = dict(triple)
        alignment = paragraph_alignment[(triple["source"], triple["paragraph"])]
        accepted = alignment["status"] in {"auto_high", "auto_medium"}
        row.update(
            {
                # Only accepted alignments receive canonical clause identities.
                "standard_id": alignment.get("standard_id") if accepted else None,
                "standard_version": alignment.get("standard_version") if accepted else None,
                "clause_id": alignment.get("clause_id") if accepted else None,
                "clause_title": alignment.get("clause_title") if accepted else None,
                "evidence_id": alignment.get("evidence_id") if accepted else None,
                "clause_alignment": {
                    key: alignment.get(key)
                    for key in ("status", "score", "runner_up_score", "margin", "method")
                },
            }
        )
        linked[triple["source"]].append(row)
        status_counts[alignment["status"]] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "triplet_clause_cache.json"
    cache_path.write_text(json.dumps(linked, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path = output_dir / "paragraph_review.jsonl"
    review_rows = [
        {"source": source, "paragraph_preview": paragraph[:500], **alignment}
        for (source, paragraph), alignment in paragraph_alignment.items()
        if alignment["status"] in {"needs_review", "unresolved"}
    ]
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows), encoding="utf-8"
    )
    manifest = {
        "schema_version": "sct-clause-kg-v1",
        "scope": "the 200 standards represented in clauses.jsonl",
        "source_triples_sha256": sha256(triples_path),
        "clauses_sha256": sha256(clauses_path),
        "source_total_rows": sum(len(values) for values in triples_by_source.values()),
        "scoped_sources": len(linked),
        "scoped_triples": len(relevant_rows),
        "unique_paragraphs": len(paragraph_alignment),
        "triple_alignment_status": dict(sorted(status_counts.items())),
        "triples_with_accepted_clause_identity": sum(
            count for status, count in status_counts.items() if status in {"auto_high", "auto_medium"}
        ),
        "paragraphs_needing_review": len(review_rows),
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--clauses", type=Path, default=DEFAULT_CLAUSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.triples, args.clauses, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
