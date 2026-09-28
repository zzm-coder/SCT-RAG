# -*- coding: utf-8 -*-
"""HotPot 局部子图 KG 检索：英文实体精确/模糊 + 本题 text_unit 约束。"""
from __future__ import annotations

import ast
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from data_types import KGResult, KGTriple, QuestionType

logger = logging.getLogger(__name__)

# 标题/问句通用词：禁止单独用来匹配实体（避免 “company” 命中 World Publishing Company）
_STOP_TOKENS = {
    "what", "which", "where", "when", "who", "whom", "whose", "why", "how",
    "that", "this", "these", "those", "with", "from", "have", "has", "had",
    "were", "been", "being", "into", "about", "after", "before", "during",
    "company", "companies", "organization", "organisations", "organization",
    "people", "person", "group", "film", "movie", "album", "song", "book",
    "city", "country", "state", "year", "years", "name", "named", "also",
    "known", "called", "based", "located", "founded", "born", "died",
    "american", "british", "english", "first", "second", "third", "same",
}


def parse_id_list(text_unit_ids) -> List[str]:
    if text_unit_ids is None:
        return []
    if isinstance(text_unit_ids, (list, tuple, set)):
        return [str(x).strip() for x in text_unit_ids if str(x).strip()]
    s = str(text_unit_ids).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [s]


def extract_question_names(question: str, extra: Optional[List[str]] = None) -> List[str]:
    names: List[str] = []
    for n in re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b", question or ""):
        names.append(n)
    for n in re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+\d+)\b", question or ""):
        names.append(n)
    # 单专名：长度够且非停用词（如 Roseanne）
    for n in re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", question or ""):
        if n.lower() not in _STOP_TOKENS:
            names.append(n)
    for n in extra or []:
        if n and str(n).strip():
            names.append(str(n).strip())
    out, seen = [], set()
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out


def _sig_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3 and t not in _STOP_TOKENS]


def _name_variant_match(a: str, b: str) -> bool:
    """判断两个英文专名是否为别名（如 Rex Maughan ↔ Rex G. Maughan）。"""
    al, bl = (a or "").lower().strip(), (b or "").lower().strip()
    if not al or not bl:
        return False
    if al == bl:
        return True
    if al in bl or bl in al:
        shorter, longer = (al, bl) if len(al) <= len(bl) else (bl, al)
        if len(shorter) >= 6:
            return True
    ta, tb = _sig_tokens(a), _sig_tokens(b)
    if not ta or not tb:
        return False
    # 较短一方的全部显著词都出现在较长一方
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) >= 2 and all(t in long_ for t in short):
        return True
    if len(short) == 1 and short[0] in long_ and len(short[0]) >= 6:
        return True
    return False


def _name_in_text(name: str, text_l: str) -> bool:
    nl = (name or "").lower().strip()
    if len(nl) < 4:
        return False
    return nl in text_l


def query_hotpot_local_kg(
    kg_retriever,
    question: str,
    entities: List[str],
    question_type: QuestionType,
    local_chunks: List[dict],
) -> KGResult:
    """在本题 distractor 段落内做局部图谱检索，并缓存命中原文供 GraphRAG 注入。"""
    start = time.time()
    hotpot_path = Path(kg_retriever.config.hotpot_kg_json_path)
    if not kg_retriever.hotpot_data.get("entities") or not kg_retriever.hotpot_data.get("relations"):
        kg_retriever.load_from_json(str(hotpot_path))

    local_sources: Set[str] = set()
    title_by_src: Dict[str, str] = {}
    local_parts: List[str] = []
    for c in local_chunks or []:
        src = str(c.get("source") or "").strip()
        title = str(c.get("title") or "").strip()
        body = str(c.get("chunk") or c.get("chunk_text") or c.get("text") or "")
        if src:
            local_sources.add(src)
            if title:
                title_by_src[src] = title
        local_parts.append(f"{title}\n{body}")
    blob_l = "\n".join(local_parts).lower()
    q_l = (question or "").lower()

    ent_index: Dict[str, Any] = kg_retriever.hotpot_data.get("entities") or {}
    all_names = list(ent_index.keys())
    cand = extract_question_names(question, entities)

    matched: List[str] = []
    seen: Set[str] = set()

    def _add(name: str):
        k = (name or "").lower()
        if not name or k in seen:
            return
        seen.add(k)
        matched.append(name)

    # 1) 问句专名 → 精确 / 别名匹配（只在全库名上找，但优先局部出现过的）
    for cn in cand:
        if cn in ent_index:
            _add(cn)
            continue
        cl = cn.lower()
        exact = next((en for en in all_names if en.lower() == cl), None)
        if exact:
            _add(exact)
            continue
        # 别名：优先 blob 内出现过的实体
        local_hits, other_hits = [], []
        for en in all_names:
            if not _name_variant_match(cn, en):
                continue
            (local_hits if en.lower() in blob_l else other_hits).append(en)
        for en in local_hits[:3] + other_hits[:1]:
            _add(en)

    # 2) 局部 chunk 标题：标题在问句中，或与已匹配种子别名一致
    for c in local_chunks or []:
        title = str(c.get("title") or "").strip()
        if not title:
            continue
        tl = title.lower()
        if tl in q_l or any(_name_variant_match(title, m) for m in matched) or any(
            _name_variant_match(title, cn) for cn in cand
        ):
            if title in ent_index:
                _add(title)
            else:
                # 标题可能是实体别名
                for en in all_names:
                    if _name_variant_match(title, en) and en.lower() in blob_l:
                        _add(en)
                        break

    seed_names = list(matched)  # 扩展前的问句种子
    relations = kg_retriever.hotpot_data.get("relations") or {}
    tus = kg_retriever.hotpot_data.get("text_units") or {}
    # 两跳扩展（限制在本题 text_unit）：种子 → 桥接 → 答案实体
    frontier = list(seed_names)
    for _hop in range(2):
        nxt = []
        for _key, rel in relations.items():
            tu_ids = parse_id_list(rel.get("text_unit_ids"))
            if local_sources and not any(t in local_sources for t in tu_ids):
                continue
            src = str(rel.get("source") or "")
            tgt = str(rel.get("target") or "")
            for seed in frontier:
                if _name_variant_match(seed, src):
                    if tgt.lower() not in seen:
                        _add(tgt)
                        nxt.append(tgt)
                    _add(src)
                elif _name_variant_match(seed, tgt):
                    if src.lower() not in seen:
                        _add(src)
                        nxt.append(src)
                    _add(tgt)
        frontier = nxt
        if not frontier:
            break
    matched = matched[:24]
    matched_l = {m.lower() for m in matched}
    seed_l = {m.lower() for m in seed_names}

    def _ent_related(name: str) -> bool:
        if not name:
            return False
        nl = name.lower()
        if nl in matched_l:
            return True
        return any(_name_variant_match(name, m) for m in matched)

    def _seed_related(name: str) -> bool:
        if not name:
            return False
        if name.lower() in seed_l:
            return True
        return any(_name_variant_match(name, m) for m in seed_names)

    # 4) 收集三元组：必须落在本题 text_unit，且至少一端与种子相关（优先），否则允许扩展实体
    triples: List[KGTriple] = []
    used_units: Set[str] = set()

    for _key, rel in relations.items():
        try:
            src = str(rel.get("source") or "")
            tgt = str(rel.get("target") or "")
            rtype = str(rel.get("relation_type") or rel.get("relation") or "")
            tu_ids = parse_id_list(rel.get("text_unit_ids"))
            in_local = bool(local_sources) and any(t in local_sources for t in tu_ids)
            if local_sources and not in_local:
                continue
            seed_hit = _seed_related(src) or _seed_related(tgt)
            expand_hit = _ent_related(src) or _ent_related(tgt)
            if matched and not (seed_hit or expand_hit):
                continue

            evidences = rel.get("evidences", [])
            if isinstance(evidences, str):
                try:
                    evidences = ast.literal_eval(evidences) if evidences.startswith("[") else [evidences]
                except Exception:
                    evidences = [evidences]
            paragraph = "; ".join(str(x) for x in (evidences or []) if x)
            if not paragraph and tu_ids:
                tu = tus.get(tu_ids[0]) or {}
                paragraph = str(tu.get("content") or "")[:800]
            para_l = (paragraph or "").lower()
            q_tok_in_para = sum(
                1
                for cn in cand
                if _name_in_text(cn, para_l) or _name_in_text(cn, f"{src} {tgt}".lower())
            )

            source_str = next((t for t in tu_ids if t in local_sources), None) or (
                tu_ids[0] if tu_ids else "hotpot_local"
            )
            for t in tu_ids:
                if (not local_sources) or (t in local_sources):
                    used_units.add(t)

            # 种子边优先；扩展边保留以覆盖多跳答案实体（如 acquired → Aloe Vera）
            both_seed = _seed_related(src) and _seed_related(tgt)
            conf = 0.75
            if seed_hit:
                conf = 1.5 if q_tok_in_para >= 2 else (1.35 if q_tok_in_para >= 1 else 1.15)
                if both_seed:
                    conf += 0.15
            elif expand_hit:
                conf = 1.05 if q_tok_in_para >= 1 else 0.9
            triples.append(
                KGTriple(
                    head=src,
                    relation=rtype,
                    tail=tgt,
                    source=str(source_str),
                    paragraph=paragraph,
                    confidence=conf,
                )
            )
        except Exception:
            continue

    # 5) 兜底：种子实体对应的局部原文（即使边很少）
    if len(triples) < 4:
        for src in list(local_sources):
            title = title_by_src.get(src, "")
            keep = False
            if title and (title.lower() in q_l or any(_name_variant_match(title, m) for m in matched)):
                keep = True
            if not keep:
                for m in matched:
                    if _name_in_text(m, blob_l) and src in {
                        u for u in parse_id_list((ent_index.get(m) or {}).get("text_unit_ids"))
                    }:
                        keep = True
                        break
            if not keep and matched:
                # 段落正文含种子专名
                tu = tus.get(src) or {}
                content = str(tu.get("content") or "")
                if not content:
                    for c in local_chunks or []:
                        if str(c.get("source")) == src:
                            content = str(c.get("chunk") or c.get("chunk_text") or "")
                            break
                if any(_name_in_text(m, content.lower()) for m in matched):
                    keep = True
            if not keep:
                continue
            tu = tus.get(src) or {}
            content = str(tu.get("content") or "")
            title = str(tu.get("title") or title_by_src.get(src) or src)
            if not content:
                for c in local_chunks or []:
                    if str(c.get("source")) == src:
                        content = str(c.get("chunk") or c.get("chunk_text") or "")
                        title = str(c.get("title") or title)
                        break
            if not content:
                continue
            head = matched[0] if matched else title
            triples.append(
                KGTriple(
                    head=head,
                    relation="context",
                    tail=title,
                    source=src,
                    paragraph=content[:1200],
                    confidence=0.75,
                )
            )
            used_units.add(src)

    triples.sort(key=lambda x: x.confidence, reverse=True)
    max_n = {
        QuestionType.SINGLE_STANDARD: 24,
        QuestionType.CROSS_STANDARD: 20,
        QuestionType.CORRELATION: 16,
    }.get(question_type, 20)
    triples = kg_retriever._deduplicate_triples(triples)[:max_n]

    # 6) 缓存原文：命中 unit + 种子实体 text_unit + 问句标题段落
    text_units: List[dict] = []
    seen_uid: Set[str] = set()

    def _push_uid(uid: str):
        if not uid or uid in seen_uid:
            return
        if local_sources and uid not in local_sources:
            return
        seen_uid.add(uid)
        tu = tus.get(uid)
        if tu:
            text_units.append(tu)
            return
        for c in local_chunks or []:
            if str(c.get("source")) == uid:
                text_units.append(
                    {
                        "id": uid,
                        "title": c.get("title") or "",
                        "content": c.get("chunk") or c.get("chunk_text") or "",
                    }
                )
                break

    # 优先：标题/正文含问句种子的局部段
    ranked_uids: List[tuple] = []
    for c in local_chunks or []:
        src = str(c.get("source") or "")
        title = str(c.get("title") or "")
        body = str(c.get("chunk") or c.get("chunk_text") or c.get("text") or "")
        bl = f"{title}\n{body}".lower()
        score = 0
        if title.lower() in q_l:
            score += 5
        for m in seed_names:
            if _name_variant_match(title, m) or _name_in_text(m, bl):
                score += 3
        for cn in cand:
            if _name_in_text(cn, bl):
                score += 2
        if src in used_units:
            score += 1
        if score > 0:
            ranked_uids.append((score, src))
    ranked_uids.sort(key=lambda x: x[0], reverse=True)
    for _sc, uid in ranked_uids:
        _push_uid(uid)
    for uid in list(used_units):
        _push_uid(uid)
    for name in seed_names + matched:
        ent = ent_index.get(name) or {}
        for uid in parse_id_list(ent.get("text_unit_ids")):
            _push_uid(uid)

    kg_retriever._hotpot_last_text_units = text_units[:12]

    ents = list(dict.fromkeys(matched + [t.head for t in triples] + [t.tail for t in triples]))
    result = KGResult(triples=triples, entities=ents, query_time=time.time() - start)
    logger.info(
        "HotPot局部KG: 实体=%d 三元组=%d 原文=%d | %s...",
        len(matched),
        len(triples),
        len(kg_retriever._hotpot_last_text_units),
        (question or "")[:40],
    )
    return result
