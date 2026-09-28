#!/usr/bin/env python3
"""Build canonical clause records and align QA evidence to clause IDs.

Every alignment retains its method, score, and review status so that uncertain
matches cannot silently become gold labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_CORPUS = CODE_ROOT / "0_mineru_pdf" / "data_md_filtered"
DEFAULT_QA_DIR = CODE_ROOT / "3_QA_creation" / "3_QA_data" / "indstd_design_qa_clause"
DEFAULT_OUTPUT = CODE_ROOT / "2_kg_construction" / "clause_records"

CLAUSE_SCHEMA_VERSION = "sct-clause-v1"
NUMERIC_CLAUSE_RE = re.compile(
    r"^(\d+(?:\.\d+){0,7})(?:\s+|$|(?=[\u4e00-\u9fffA-Za-z（(]))(.*)$"
)
ALPHA_CLAUSE_RE = re.compile(
    r"^([A-Z]\.(?:\d+)(?:\.\d+){0,6})(?:\s+|$|(?=[\u4e00-\u9fffA-Za-z（(]))(.*)$",
    re.IGNORECASE,
)
ANNEX_RE = re.compile(
    r"^(?:附\s*录|附\s*件|appendix|annex)\s*([A-ZＡ-Ｚ0-9一二三四五六七八九十]+)\b[\s:：.-]*(.*)$",
    re.IGNORECASE,
)
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")


@dataclass(frozen=True)
class ClauseRecord:
    schema_version: str
    standard_id: str
    standard_version: str
    clause_id: str
    clause_title: str
    clause_path: list[str]
    evidence_id: str
    source_file: str
    source_path: str
    start_offset: int
    end_offset: int
    raw_text: str
    content_hash: str
    parse_method: str


def normalize_unicode(text: str) -> str:
    """Normalize OCR-compatible punctuation without changing source files."""
    out = unicodedata.normalize("NFKC", str(text or ""))
    return (
        out.replace("．", ".")
        .replace("。", ".")
        .replace("—", "-")
        .replace("–", "-")
        .replace("－", "-")
        .replace("／", "/")
    )


def compact_text(text: str) -> str:
    text = normalize_unicode(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def canonical_standard_id(source_file: str, text: str = "") -> tuple[str, str]:
    """Return a stable display ID and year from source name, with text fallback."""
    stem = Path(source_file).stem
    s = normalize_unicode(stem).replace("+", " ").replace("_", ".")
    s = re.sub(r"-(?:gbt|gb|hb|gjb)-(?:cd|e)-\d+$", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    year_match = re.search(r"(?:19|20)\d{2}", s)
    year = year_match.group(0) if year_match else ""

    upper = s.upper()
    if upper.startswith("GBT "):
        display = "GB/T " + s[4:]
    elif upper.startswith("GB "):
        display = "GB " + s[3:]
    elif upper.startswith("HB "):
        display = "HB " + s[3:]
    elif upper.startswith("GJB "):
        display = "GJB " + s[4:]
    elif re.match(r"^\d", s):
        display = "GB/T " + s.replace("-", "-", 1)
    else:
        display = s.replace(" ", " ")

    display = re.sub(r"\s*-\s*", "-", display)
    display = re.sub(r"\s+", " ", display).strip()
    if not year:
        head = normalize_unicode(text[:1200])
        m = re.search(
            r"\b((?:GB\s*/?\s*T|GB|HB|GJB)\s*[0-9.]+)\s*[-]\s*((?:19|20)\d{2})",
            head,
            re.I,
        )
        if m:
            display = re.sub(r"\s+", " ", m.group(1)).upper() + "-" + m.group(2)
            display = display.replace("GB / T", "GB/T").replace("GB/ T", "GB/T")
            year = m.group(2)
    return display, year


def _clause_token(title: str) -> tuple[str, str] | None:
    clean = normalize_unicode(title).strip().strip("#").strip()
    clean = re.sub(r"\s*\.\s*", ".", clean)
    m = NUMERIC_CLAUSE_RE.match(clean)
    if m:
        return m.group(1), m.group(2).strip(" :：.-")
    m = ALPHA_CLAUSE_RE.match(clean)
    if m:
        return m.group(1).upper(), m.group(2).strip(" :：.-")
    m = ANNEX_RE.match(clean)
    if m:
        annex = normalize_unicode(m.group(1)).upper()
        return f"Annex {annex}", m.group(2).strip()
    return None


def _iter_markers(text: str) -> Iterable[tuple[int, int, str, str, int, str]]:
    """Yield (start, end, clause_id, title, level, method) markers."""
    pending_id: tuple[str, int, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        start = offset
        end = offset + len(line)
        offset = end
        if not stripped:
            continue

        heading = MARKDOWN_HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            token = _clause_token(title)
            if token:
                pending_id = None
                yield start, end, token[0], token[1], level, "markdown_heading"
            elif pending_id:
                cid, pending_start, pending_level = pending_id
                pending_id = None
                yield pending_start, end, cid, title, min(level, pending_level), "split_number_heading"
            continue

        token = _clause_token(stripped)
        if not token:
            continue
        cid, title = token
        # Standalone IDs are frequently followed by a Markdown title on the next line.
        if not title and re.fullmatch(r"\d+(?:\.\d+){0,7}", cid):
            pending_id = (cid, start, min(6, cid.count(".") + 2))
            continue
        # Avoid treating long OCR prose beginning with a number as a heading.
        if len(stripped) <= 180 and not stripped.startswith("<"):
            pending_id = None
            yield start, end, cid, title, min(6, cid.count(".") + 2), "plain_number_heading"


def parse_clauses(path: Path, corpus_root: Path) -> list[ClauseRecord]:
    text = path.read_text(encoding="utf-8", errors="replace")
    standard_id, version = canonical_standard_id(path.name, text)
    markers = list(_iter_markers(text))
    # Remove exact duplicate markers while preserving document order.
    unique: list[tuple[int, int, str, str, int, str]] = []
    seen = set()
    for marker in markers:
        key = (marker[0], marker[2])
        if key not in seen:
            seen.add(key)
            unique.append(marker)

    records: list[ClauseRecord] = []
    title_stack: dict[int, str] = {}
    for idx, marker in enumerate(unique):
        start, heading_end, cid, title, level, method = marker
        if "." not in cid and cid.isdigit() and level >= 3 and len(cid) == level - 1:
            cid = ".".join(cid)
            method += ":ocr_compact_id_repaired"
        end = unique[idx + 1][0] if idx + 1 < len(unique) else len(text)
        raw_text = text[start:end].strip()
        if not raw_text:
            continue
        for old_level in [n for n in title_stack if n >= level]:
            del title_stack[old_level]
        title_stack[level] = f"{cid} {title}".strip()
        path_titles = [title_stack[n] for n in sorted(title_stack)]

        # A clause ID denotes the same logical clause even if OCR repeats it in a TOC.
        # Duplicate physical occurrences are collapsed below.
        evidence_key = f"{standard_id}|{cid}|{CLAUSE_SCHEMA_VERSION}"
        evidence_id = "ev_" + hashlib.sha256(evidence_key.encode("utf-8")).hexdigest()[:20]
        content_hash = hashlib.sha256(compact_text(raw_text).encode("utf-8")).hexdigest()
        records.append(
            ClauseRecord(
                schema_version=CLAUSE_SCHEMA_VERSION,
                standard_id=standard_id,
                standard_version=version,
                clause_id=cid,
                clause_title=title,
                clause_path=path_titles,
                evidence_id=evidence_id,
                source_file=path.name,
                source_path=str(path.relative_to(corpus_root)),
                start_offset=start,
                end_offset=end,
                raw_text=raw_text,
                content_hash=content_hash,
                parse_method=method,
            )
        )
    # Prefer the information-rich body occurrence over a short TOC occurrence.
    # This also prevents duplicate candidates from creating a false zero margin.
    best_by_evidence: dict[str, ClauseRecord] = {}
    for record in records:
        compact = compact_text(record.raw_text)
        toc_penalty = 1200 if "目次" in record.raw_text[:300] else 0
        ellipsis_penalty = min(800, record.raw_text[:500].count("…") * 80)
        quality = min(len(compact), 6000) - toc_penalty - ellipsis_penalty
        previous = best_by_evidence.get(record.evidence_id)
        if previous is None:
            best_by_evidence[record.evidence_id] = record
            continue
        prev_compact = compact_text(previous.raw_text)
        prev_quality = min(len(prev_compact), 6000)
        if "目次" in previous.raw_text[:300]:
            prev_quality -= 1200
        prev_quality -= min(800, previous.raw_text[:500].count("…") * 80)
        if quality > prev_quality:
            best_by_evidence[record.evidence_id] = record
    return sorted(best_by_evidence.values(), key=lambda record: record.start_offset)


def _ngrams(text: str, n: int = 4) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def alignment_score(gold_text: str, clause_text: str) -> tuple[float, str]:
    gold = compact_text(gold_text)
    clause = compact_text(clause_text)
    if not gold or not clause:
        return 0.0, "empty"
    if gold in clause:
        coverage = min(1.0, len(gold) / max(1, len(clause)))
        return min(1.0, 0.92 + 0.08 * coverage), "exact_substring"
    if clause in gold:
        coverage = len(clause) / max(1, len(gold))
        return min(0.91, 0.72 + 0.19 * coverage), "clause_substring"
    ga, ca = _ngrams(gold), _ngrams(clause)
    overlap = len(ga & ca)
    recall = overlap / max(1, len(ga))
    precision = overlap / max(1, len(ca))
    ngram_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    seq = SequenceMatcher(None, gold[:4000], clause[:8000], autojunk=False).ratio()
    return 0.72 * recall + 0.18 * ngram_f1 + 0.10 * seq, "fuzzy_ngram"


def align_fact(fact: dict, clauses: list[ClauseRecord]) -> tuple[dict, dict]:
    gold = str(fact.get("chunk") or fact.get("text") or "")
    ranked = []
    for clause in clauses:
        score, method = alignment_score(gold, clause.raw_text)
        ranked.append((score, method, clause))
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0] if ranked else (0.0, "no_clause", None)
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    score, method, clause = best
    margin = score - second_score
    if clause is None:
        status = "unresolved"
    elif score >= 0.90 and margin >= 0.03:
        status = "auto_high"
    elif score >= 0.72 and margin >= 0.015:
        status = "auto_medium"
    else:
        status = "needs_review"

    enriched = dict(fact)
    alignment = {
        "schema_version": CLAUSE_SCHEMA_VERSION,
        "status": status,
        "score": round(score, 6),
        "runner_up_score": round(second_score, 6),
        "margin": round(margin, 6),
        "method": method,
        "candidate_clauses": [
            {
                "rank": rank,
                "score": round(candidate_score, 6),
                "method": candidate_method,
                "standard_id": candidate.standard_id,
                "clause_id": candidate.clause_id,
                "clause_title": candidate.clause_title,
                "evidence_id": candidate.evidence_id,
                "preview": candidate.raw_text[:360],
            }
            for rank, (candidate_score, candidate_method, candidate) in enumerate(ranked[:5], start=1)
        ],
    }
    if clause:
        enriched.update(
            {
                "standard_id": clause.standard_id,
                "standard_version": clause.standard_version,
                "clause_id": clause.clause_id,
                "clause_title": clause.clause_title,
                "clause_path": clause.clause_path,
                "evidence_id": clause.evidence_id,
                "clause_text": clause.raw_text,
                "clause_alignment": alignment,
            }
        )
    else:
        enriched["clause_alignment"] = alignment

    audit = {
        "source": fact.get("source", ""),
        "status": status,
        "score": alignment["score"],
        "runner_up_score": alignment["runner_up_score"],
        "margin": alignment["margin"],
        "method": method,
        "standard_id": clause.standard_id if clause else "",
        "clause_id": clause.clause_id if clause else "",
        "evidence_id": clause.evidence_id if clause else "",
        "gold_preview": gold[:240],
        "clause_preview": clause.raw_text[:320] if clause else "",
        "candidate_clauses": alignment["candidate_clauses"],
    }
    return enriched, audit


def corpus_index(corpus_root: Path) -> tuple[dict[str, list[ClauseRecord]], list[dict]]:
    by_source: dict[str, list[ClauseRecord]] = {}
    clause_rows: list[dict] = []
    for path in sorted(corpus_root.rglob("*.md")):
        records = parse_clauses(path, corpus_root)
        by_source[path.name] = records
        clause_rows.extend(asdict(record) for record in records)
    return by_source, clause_rows


def build_dataset(qa_dir: Path, corpus_root: Path, output_dir: Path) -> dict:
    by_source, clause_rows = corpus_index(corpus_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clauses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clause_rows),
        encoding="utf-8",
    )

    audit_rows: list[dict] = []
    split_summary: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        source_path = qa_dir / f"{split}.json"
        if not source_path.exists():
            continue
        items = json.loads(source_path.read_text(encoding="utf-8"))
        enriched_items = []
        counts: dict[str, int] = {}
        for item in items:
            new_item = dict(item)
            new_facts = []
            for index, fact in enumerate(item.get("supporting_facts") or []):
                source = Path(str(fact.get("source") or "")).name
                clauses = by_source.get(source, [])
                enriched, audit = align_fact(fact, clauses)
                audit.update(
                    {
                        "split": split,
                        "qa_id": item.get("id", ""),
                        "fact_index": index,
                        "question": item.get("question", ""),
                        "candidate_clause_count": len(clauses),
                    }
                )
                counts[audit["status"]] = counts.get(audit["status"], 0) + 1
                audit_rows.append(audit)
                new_facts.append(enriched)
            new_item["supporting_facts"] = new_facts
            new_item["evidence_schema_version"] = CLAUSE_SCHEMA_VERSION
            enriched_items.append(new_item)
        (output_dir / f"{split}.json").write_text(
            json.dumps(enriched_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        split_summary[split] = {
            "items": len(enriched_items),
            "facts": sum(counts.values()),
            "alignment_status": counts,
        }

    review_rows = [row for row in audit_rows if row["status"] == "needs_review"]
    (output_dir / "alignment_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    (output_dir / "manual_review.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": CLAUSE_SCHEMA_VERSION,
        "corpus_root": str(corpus_root),
        "qa_dir": str(qa_dir),
        "output_dir": str(output_dir),
        "source_files": len(by_source),
        "parsed_clauses": len(clause_rows),
        "sources_without_clauses": sorted(k for k, value in by_source.items() if not value),
        "splits": split_summary,
        "manual_review_facts": len(review_rows),
    }
    (output_dir / "alignment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--qa-dir", type=Path, default=DEFAULT_QA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_dataset(args.qa_dir, args.corpus_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
