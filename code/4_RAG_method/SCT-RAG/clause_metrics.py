"""Transparent clause-level retrieval and citation metrics for SCT-RAG."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


@dataclass(frozen=True)
class CitationMetrics:
    precision: float
    recall: float
    f1: float
    exact_match: float
    true_positive: int
    predicted_count: int
    gold_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def clause_key(value: Any) -> str:
    """Return a canonical logical-clause key from a dict or string reference."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    evidence_id = str(value.get("evidence_id") or "").strip()
    if evidence_id:
        return evidence_id
    standard_id = str(value.get("standard_id") or "").strip()
    clause_id = str(value.get("clause_id") or "").strip()
    if standard_id and clause_id:
        return f"{standard_id}::{clause_id}"
    metadata = value.get("metadata") or {}
    if isinstance(metadata, dict):
        evidence_id = str(metadata.get("evidence_id") or "").strip()
        if evidence_id:
            return evidence_id
        standard_id = str(metadata.get("standard_id") or "").strip()
        clause_id = str(metadata.get("clause_id") or "").strip()
        if standard_id and clause_id:
            return f"{standard_id}::{clause_id}"
    return ""


def clause_set(values: Iterable[Any] | None) -> set[str]:
    return {key for key in (clause_key(value) for value in (values or [])) if key}


def citation_metrics(predicted: Iterable[Any] | None, gold: Iterable[Any] | None) -> CitationMetrics:
    predicted_set = clause_set(predicted)
    gold_set = clause_set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else (1.0 if not gold_set else 0.0)
    recall = true_positive / len(gold_set) if gold_set else (1.0 if not predicted_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CitationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=float(predicted_set == gold_set),
        true_positive=true_positive,
        predicted_count=len(predicted_set),
        gold_count=len(gold_set),
    )


def _item_std(value: Any) -> str:
    """取出标准号，供同标准内排序评测。"""
    if value is None:
        return ""
    if isinstance(value, dict):
        sid = str(value.get("standard_id") or "").strip()
        if sid:
            return sid
        meta = value.get("metadata") or {}
        if isinstance(meta, dict):
            sid = str(meta.get("standard_id") or "").strip()
            if sid:
                return sid
        src = str(value.get("source") or "")
        return src
    return str(getattr(value, "standard_id", "") or getattr(value, "source", "") or "")


def same_std_ranking_metrics(
    retrieved: Iterable[Any] | None,
    gold: Iterable[Any] | None,
) -> dict:
    """同标准候选内：金标条是否压过邻条，以及是否全部进入送 LLM 的上下文。

    same_std_recall_at_1：金标证据中，在同标准检索列表里排第 1 的比例。
    same_std_mrr：同标准内 1/rank 均值（未出现记 0）。
    gold_all_in_context：全部金标 ID 都出现在当前检索列表。
    """
    gold_items = [g for g in (gold or []) if clause_key(g)]
    ranked = []
    seen = set()
    for value in retrieved or []:
        key = clause_key(value)
        if key and key not in seen:
            seen.add(key)
            ranked.append(value)
    gold_keys = {clause_key(g) for g in gold_items}
    if not gold_keys:
        return {
            "same_std_recall_at_1": 0.0,
            "same_std_mrr": 0.0,
            "gold_all_in_context": 0.0,
        }

    def _std_of(item: Any) -> str:
        raw = _item_std(item)
        return raw.replace("+", " ").replace(".md", "").strip().lower()

    hits_at_1 = 0
    rr_sum = 0.0
    for g in gold_items:
        gk = clause_key(g)
        gstd = _std_of(g)
        same = [c for c in ranked if (not gstd) or (_std_of(c) and (gstd in _std_of(c) or _std_of(c) in gstd))]
        if not same:
            same = list(ranked)
        rank = None
        for i, c in enumerate(same, start=1):
            if clause_key(c) == gk:
                rank = i
                break
        if rank is None:
            continue
        if rank == 1:
            hits_at_1 += 1
        rr_sum += 1.0 / float(rank)
    n = max(1, len(gold_items))
    ctx_keys = {clause_key(c) for c in ranked}
    return {
        "same_std_recall_at_1": float(hits_at_1) / n,
        "same_std_mrr": float(rr_sum) / n,
        "gold_all_in_context": float(gold_keys <= ctx_keys),
    }


def retrieval_at_k(retrieved: Iterable[Any] | None, gold: Iterable[Any] | None, k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = clause_set(gold)
    ranked = []
    seen = set()
    for value in retrieved or []:
        key = clause_key(value)
        if key and key not in seen:
            seen.add(key)
            ranked.append(key)
    top = set(ranked[:k])
    matched = len(top & gold_set)
    return {
        f"clause_hit_at_{k}": float(matched > 0) if gold_set else 0.0,
        f"clause_recall_at_{k}": matched / len(gold_set) if gold_set else 0.0,
        f"clause_precision_at_{k}": matched / len(top) if top else 0.0,
    }


def penalized_accuracy(answer_score: float, exact_clause_match: float) -> float:
    """Credit an answer only when its predicted citation set exactly matches gold."""
    answer = max(0.0, min(1.0, float(answer_score)))
    exact = 1.0 if float(exact_clause_match) >= 1.0 else 0.0
    return answer * exact


def classify_error(
    clause_recall_at_k: float,
    exact_clause_match: float,
    answer_score: float,
    answer_threshold: float = 0.8,
) -> str:
    """Mutually exclusive error taxonomy with preregistered precedence."""
    if float(clause_recall_at_k) < 1.0:
        return "retrieval_error"
    if float(exact_clause_match) < 1.0:
        return "citation_error"
    if float(answer_score) < float(answer_threshold):
        return "generation_error"
    return "success"
