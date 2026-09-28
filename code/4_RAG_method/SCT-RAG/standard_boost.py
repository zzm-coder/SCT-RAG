# -*- coding: utf-8 -*-
"""Paper-aligned standard-anchored retrieval and identity-safe reranking.

The released SCT-RAG path contains StdDirect, WithinStd, and clause-linked KG
candidates.  Candidates retain their canonical clause identifiers, are merged
by identity, scored by the cross-encoder, and receive the fixed standard-number
boost ``beta=1.5``.  The final evidence budgets are 10/10/20 for single,
cross, and correlation queries.  Corpus-wide dense and sparse retrieval are
baseline methods, not SCT-RAG channels.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from compare_rag.utils import canonical_standard_id, docs_match, normalize_doc_key
from data_types import KGResult, QuestionType, RAGContext, RetrievedChunk

logger = logging.getLogger(__name__)

# Locked paper setting: one standard-number boost for every routed category.
STANDARD_RERANK_BETA = 1.5
FINAL_EVIDENCE_BUDGETS = {"single": 10, "cross": 10, "correlation": 20}

_STD_PATTERN = re.compile(
    r"(?:GB/T|GBT|GB|HB(?:/Z)?|HB_Z|GJB|QJ|ISO)[\s\+]*[\d\-—－/\.A-Za-z（）()]+",
    re.IGNORECASE,
)


def paper_query_type(router_response) -> str:
    """将路由器输出映射为 single / cross / correlation。"""
    mapping = {
        QuestionType.SINGLE_STANDARD: "single",
        QuestionType.CROSS_STANDARD: "cross",
        QuestionType.CORRELATION: "correlation",
    }
    return mapping.get(getattr(router_response, "question_type", None), "single")



def extract_explicit_std_codes(text: str) -> List[str]:
    """仅用正则从文本抽取显式标准号，不做主题词推断。"""
    keys: List[str] = []
    seen = set()
    for item in _STD_PATTERN.findall(text or ""):
        canon = canonical_standard_id(item)
        k = canon or normalize_doc_key(item).replace(".md", "")
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def extract_std_codes(
    question: str,
    entities: List = None,
) -> List[str]:
    """Extract and normalize only explicit standard identifiers."""
    keys = extract_explicit_std_codes(question or "")
    if keys:
        return keys
    seen = set()
    extra: List[str] = []
    for e in entities or []:
        for k in extract_explicit_std_codes(str(e)):
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return extra


def resolve_paper_type(question: str, entities: List, router_response) -> str:
    """Use the LLM router prediction without rule-based relabeling."""
    return paper_query_type(router_response)


def source_matches_std(source: str, std_keys: List[str]) -> bool:
    """判断来源文件是否命中标准编号。"""
    if not source or not std_keys:
        return False
    fname_canon = canonical_standard_id(source)
    for k in std_keys:
        k_str = str(k or "")
        k_canon = k_str if ":" in k_str else (canonical_standard_id(k_str) or k_str)
        if fname_canon and k_canon and fname_canon == k_canon:
            return True
        if fname_canon and k_canon and "::" in fname_canon and "::" in k_canon:
            if fname_canon.split("::", 1)[0] == k_canon.split("::", 1)[0]:
                return True
        if docs_match(source, k) or docs_match(source, f"{k}.md"):
            return True
        src_core = normalize_doc_key(source).replace(".md", "")
        k_core = normalize_doc_key(k).replace(".md", "")
        if src_core and k_core and (src_core == k_core or k_core in src_core or src_core in k_core):
            return True
    return False


def sentence_is_unusable_evidence(sentence: str) -> bool:
    """过滤不宜作为证据句的 LaTeX/参考文献/规章噪声。"""
    s = str(sentence or "")
    if len(s.strip()) < 8:
        return True
    if re.search(r"\\mathrm|\\frac|\$\s*\\", s):
        # 保留可抄公式本体；仅滤掉无 $$/式中/等号的碎片 LaTeX
        if not re.search(r"\$\$|式中|\\eta\s*\{?\s*=|[ηη]\s*=|=", s):
            return True
    if re.search(
        r"参\s*考\s*文\s*献|CCAR－|民航总局令|中华人民共和国(?:航空)?(?:行业)?标准|"
        r"Requirements\s+for",
        s,
        flags=re.I,
    ):
        return True
    if re.search(r"ARINC\s*\d+|Air\s+transport\s+avionics|民用飞机飞行控制系统通用要求", s, flags=re.I):
        return True
    if re.search(r"本标准代替\s*HB|代替\s*HB\s*\d", s):
        return True
    if re.search(r"分为以下\s*\d*\s*个部分|规范性引用文件|下列文件中的条款", s):
        return True
    if re.search(r"^/\s*《|分为以下\s*个部分", s):
        return True
    if re.search(r"零件\s*材料\s*材料技术要求", re.sub(r"\s+", "", s)):
        return True
    return False


def chunk_std_reference_list_ratio(text: str) -> float:
    """估算段落中标准号罗列密度（前言/引用文件列表特征）。"""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return 0.0
    hits = len(re.findall(r"(?:HB|GB/T|GBT|GB|GJB|QJ)\s*\d", text or "", flags=re.I))
    return hits / max(1.0, len(compact) / 40.0)


def looks_like_preamble(text: str) -> bool:
    """前言/范围/封面类段落（结构特征过滤，不做约束词加分）。"""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if any(k in compact for k in ("目次", "目录", "前言", "出版", "印刷", "发行", "书号", "定价")):
        return True
    # 短「适用范围」声明：无具体设计约束内容时视为前言
    if any(k in compact for k in ("适用于", "本标准规定", "本部分规定")) and len(compact) < 80:
        return True
    if re.search(r"^(?:范围|目次|前言)", compact) and len(compact) < 60:
        return True
    # 型号代号/总规范封面，无设计约束
    if re.search(r"HB[xX]{2,}|总规范", compact) and not re.search(
        r"应|宜|不得|不小于|不大于", compact
    ):
        return True
    return False


def chunk_clause_quality_ok(text: str) -> bool:
    """RerankBoost 结构门控：跳过前言/引用列表/不可用噪声；不要求约束词命中。"""
    if not text or sentence_is_unusable_evidence(str(text)[:480]):
        return False
    compact = re.sub(r"\s+", "", str(text))
    if len(compact) < 12:
        return False
    if looks_like_preamble(text):
        return False
    if chunk_std_reference_list_ratio(text) >= 0.42:
        return False
    if re.search(
        r"中华人民共和国(?:航空)?(?:行业)?标准|Requirements\s+for|通用规范|"
        r"规范性引用文件|分为以下\s*\d+\s*个部分",
        str(text),
        flags=re.I,
    ):
        return False
    return True


_EMPTY_FORMULA_RE = re.compile(r"公式\s*[\(（]\s*[\)）]")
_LATEX_EQ_RE = re.compile(r"\$\$[\s\S]{6,}\$\$|\$[^$]{8,}\$")
_STD_NUM_STRIP_RE = re.compile(
    r"(?:GB/T|GBT|GB|HB(?:/Z)?|GJB|QJ|ISO)[\s\+]*[\d\-—－/\.A-Za-z（）()]+",
    re.I,
)


def chunk_has_real_formula(text: str) -> bool:
    """块内含可抄公式本体：$$、η=、式中。"""
    t = text or ""
    if _LATEX_EQ_RE.search(t):
        return True
    if "式中" in t:
        return True
    if re.search(r"[ηη]\s*=", t) or re.search(r"\\eta\s*\{?\s*=", t):
        return True
    return False


def rerank_beta_for_type(paper_type: str) -> float:
    """Return beta=1.5; the override is reserved for the sensitivity runner."""
    override = os.environ.get("SCT_RERANK_BETA_OVERRIDE", "").strip()
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning("Ignoring invalid SCT_RERANK_BETA_OVERRIDE=%r", override)
    return STANDARD_RERANK_BETA


def apply_standard_number_boost(
    question: str,
    entities: List,
    chunks: List[RetrievedChunk],
    paper_type: str = "cross",
    top_k: int = 15,
) -> List[RetrievedChunk]:
    """论文 RerankBoost：仅对命中问题标准号的非噪声块做 β 加权。"""
    if not chunks:
        return chunks
    beta = rerank_beta_for_type(paper_type)
    # The controlled sensitivity run sets beta=1.0 for the "off" condition.
    if beta <= 1.0:
        return chunks[:top_k]

    std_keys = extract_std_codes(question, entities)

    for c in chunks:
        base = float(getattr(c, "rerank_score", 0) or getattr(c, "similarity_score", 0) or 0)
        src = getattr(c, "source", "") or ""
        text = getattr(c, "chunk_text", "") or ""
        matched = bool(std_keys and source_matches_std(src, std_keys))
        eligible = bool(
            matched
            and chunk_clause_quality_ok(text)
            and not looks_like_preamble(text)
            and not _clause_is_forward_ref(text)
        )
        meta = dict(getattr(c, "metadata", None) or {})
        meta["rerank_score_before_std_boost"] = base
        meta["standard_number_match"] = matched
        meta["standard_number_boost"] = beta if eligible else 1.0
        c.metadata = meta
        # Paper scoring rule: multiply standard-matched candidates by beta.
        c.rerank_score = base * beta if eligible else base

    return sorted(
        chunks,
        key=lambda c: float(getattr(c, "rerank_score", 0.0) or 0.0),
        reverse=True,
    )[:top_k]


def _clause_is_forward_ref(text: str) -> bool:
    """仅写「应符合 x.x.x 的规定」、没有具体阈值的转发句。"""
    t = text or ""
    if not re.search(r"应符合.{0,20}的规定", t):
        return False
    return not bool(
        re.search(
            r"\d+(?:\.\d+)?\s*(?:MΩ|kPa|mm|μm|%|次|℃|°C|N\b|Hz|h\b)",
            t,
            flags=re.I,
        )
    )


def inject_standard_chunks(
    question: str,
    entities: List,
    rag_context: RAGContext,
    vector_retriever: Any,
    top_k_vector: int = 20,
    limit: int = 20,
    paper_type: str = "cross",
) -> RAGContext:
    """StdDirect：题干标准号定库；无条款号时在该标准内按问句稠密排序，禁止按文件头截断。"""
    if not vector_retriever:
        return rag_context
    std_keys = extract_std_codes(question, entities)
    if not std_keys:
        return rag_context
    # 仅注入题干（或实体）显式标准号对应文件，不按关键词猜库。
    ref_pattern = re.compile(
        r"((?:GB/T|GBT|GB|HB(?:/Z)?|GJB|QJ)\s*[\d.]+-\d{4}(?:\(\d{4}\))?)"
        r"\s*第?\s*([A-Za-z]?\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))?)\s*条",
        flags=re.I,
    )
    references = [(m.group(1).strip(), m.group(2).strip()) for m in ref_pattern.finditer(question or "")]
    try:
        if references:
            injected = vector_retriever.find_chunks_by_standard_clause(
                references, limit=max(limit, len(references) * 3)
            )
        else:
            # 论文 R_std：该标准条款集合，再按问句取 Top。
            injected = []
            dense_fn = getattr(vector_retriever, "dense_retrieve_within_standards", None)
            per = max(8, int(limit // max(1, len(std_keys))))
            if callable(dense_fn):
                for std in std_keys:
                    injected.extend(
                        dense_fn(
                            question,
                            [std],
                            limit=per,
                            sentence_level=False,
                        )
                        or []
                    )
            if not injected:
                injected = vector_retriever.find_chunks_by_standard(
                    std_keys, limit=max(limit, len(std_keys) * 6)
                )
    except Exception as e:
        logger.warning(f"标准号定向注入失败: {e}")
        return rag_context
    if not injected:
        return rag_context

    # 过滤明显前言与引用列表噪声、以及无阈值的转发句
    prefer = [
        c
        for c in injected
        if chunk_clause_quality_ok(getattr(c, "chunk_text", "") or "")
        and not looks_like_preamble(getattr(c, "chunk_text", "") or "")
        and not _clause_is_forward_ref(getattr(c, "chunk_text", "") or "")
    ]
    if not prefer:
        prefer = [
            c
            for c in injected
            if not looks_like_preamble(getattr(c, "chunk_text", "") or "")
        ] or injected

    # MergeByID: preserve one candidate per canonical clause identity.
    merged = list(rag_context.vector_chunks or [])
    seen = {_chunk_key(c) for c in merged}
    for c in prefer:
        key = _chunk_key(c)
        if key not in seen:
            seen.add(key)
            merged.insert(0, c)
    # StdDirect contributes at most B=20 candidates; the merged pool may also
    # contain WithinStd and clause-linked KG candidates.
    rag_context.vector_chunks = merged[: max(3 * int(top_k_vector), int(top_k_vector))]
    return rag_context


def enrich_with_within_std_dense(
    question: str,
    entities: List,
    rag_context: RAGContext,
    vector_retriever: Any,
    limit: int = 48,
    pool_limit: int = 96,
    paper_type: str = "",
) -> RAGContext:
    """同标准内二次稠密检索（句级），把更贴问题的条款并入候选池。

    cross：额外按两侧维度词分别检索，减轻同标准错维度导致的 ParaMiss。
    """
    if not vector_retriever or not rag_context:
        return rag_context
    std_keys = extract_std_codes(question, entities)
    if not std_keys:
        return rag_context
    queries = [question]
    # cross：每侧维度单独再检一刀
    if paper_type == "cross":
        dims = parse_cross_dimensions(question)
        for si, std in enumerate(std_keys[:2]):
            side = side_topic_for_standard(question, si)
            if side and side != question:
                queries.append(f"{std} {side}")
            elif si < len(dims) and dims[si]:
                queries.append(f"{std} {dims[si]}")
    elif paper_type == "correlation":
        # 论文 SplitByTopic：每个标准单独一条子查询，再 WithinStd 并集
        topic = _strip_std_ids_from_question(question)
        terms = [t for t in question_topic_terms(topic) if len(t) >= 3][:4]
        topic_q = (topic or "").strip()
        if topic_q:
            queries.append(topic_q)
        for std in std_keys[:4]:
            if topic_q:
                queries.append(f"{std} {topic_q[:48]}")
            for t in terms:
                queries.append(f"{std} {t}")
    elif paper_type == "single":
        # single：标准号+题干核心词，减轻同文件错条款
        topic = re.sub(
            r"(?:根据|按照|依据)?\s*(?:GB/T|GBT|GB|HB(?:/Z)?|GJB|QJ)[\s\+]*[\d\-—－/\.A-Za-z（）()]+[，,]?",
            " ",
            question or "",
            flags=re.I,
        )
        topic = re.sub(r"\s+", " ", topic).strip(" ，,？?。")
        if len(topic) >= 6 and std_keys:
            queries.append(f"{std_keys[0]} {topic[:48]}")
            # 再抽 2–4 字主题片段做窄检索
            terms = question_topic_terms(topic)[:4]
            for t in terms:
                if len(t) >= 3:
                    queries.append(f"{std_keys[0]} {t}")
                    break

    dense_hits: List[RetrievedChunk] = []
    seen_q = set()
    for q in queries:
        if paper_type == "correlation":
            # 论文 WithinStd(q_s, D, {s})：下面按标准分别检索，不在多库上混排
            continue
        qk = re.sub(r"\s+", "", q or "")
        if not qk or qk in seen_q:
            continue
        seen_q.add(qk)
        try:
            hits = vector_retriever.dense_retrieve_within_standards(
                q,
                std_keys,
                limit=max(24, limit // max(1, len(queries))),
                sentence_level=False,
                sentence_neighbor_window=0,
            )
        except Exception as e:
            logger.warning(f"同标准内二次检索调用失败: {e}")
            continue
        dense_hits.extend(hits or [])
    # cross：每侧维度词只在该标准文件内检索，再按维度分把 Top1 提前
    if paper_type == "cross":
        for si, std in enumerate(std_keys):
            side = side_topic_for_standard(question, si)
            q_side = (side or "").strip() or question
            try:
                hits = vector_retriever.dense_retrieve_within_standards(
                    q_side,
                    [std],
                    limit=max(12, limit // max(2, len(std_keys))),
                    sentence_level=False,
                )
            except Exception as e:
                logger.warning(f"cross 单标准维度检索失败: {e}")
                hits = []
            dense_hits.extend(hits or [])
    if paper_type == "correlation":
        topic = _strip_std_ids_from_question(question)
        terms = [t for t in question_topic_terms(topic) if len(t) >= 3][:4]
        q_topic = " ".join([topic] + terms).strip() or question
        for std in std_keys[:4]:
            sub_qs = [q_topic]
            for t in terms[:2]:
                sub_qs.append(t)
            seen_sub = set()
            for q_s in sub_qs:
                qk = re.sub(r"\s+", "", q_s or "")
                if not qk or qk in seen_sub:
                    continue
                seen_sub.add(qk)
                try:
                    hits = vector_retriever.dense_retrieve_within_standards(
                        q_s,
                        [std],
                        limit=max(12, limit // max(2, len(std_keys))),
                        sentence_level=False,
                    )
                except Exception as e:
                    logger.warning(f"correlation 单标准主题检索失败: {e}")
                    hits = []
                dense_hits.extend(hits or [])
    if not dense_hits:
        return rag_context
    # 去重保序
    hit_seen = set()
    uniq_hits: List[RetrievedChunk] = []
    for c in dense_hits:
        k = _chunk_key(c)
        if k in hit_seen:
            continue
        hit_seen.add(k)
        uniq_hits.append(c)
    # The paper fixes the WithinStd candidate budget at B=20 (scaled only by the controlled sensitivity runner).
    uniq_hits.sort(key=_chunk_rank_score, reverse=True)
    uniq_hits = uniq_hits[: max(1, int(limit))]
    merged = list(uniq_hits) + list(rag_context.vector_chunks or [])
    seen = set()
    uniq: List[RetrievedChunk] = []
    for c in merged:
        k = _chunk_key(c)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    rag_context.vector_chunks = uniq[: max(pool_limit, limit)]
    # 句级命中还原父条款，便于抽出「分离压合」等同条后半句
    if vector_retriever:
        extra = []
        seen_p = {_chunk_key(c) for c in (rag_context.vector_chunks or [])}
        for c in list(rag_context.vector_chunks or []):
            p = _parent_clause_from_sentence(c, vector_retriever)
            if p is None:
                continue
            k = _chunk_key(p)
            if k in seen_p:
                continue
            seen_p.add(k)
            extra.append(p)
        if extra:
            rag_context.vector_chunks = extra + list(rag_context.vector_chunks or [])
    logger.info(
        "同标准二次检索并入: +%d pool=%d queries=%d type=%s",
        len(uniq_hits),
        len(rag_context.vector_chunks),
        len(seen_q),
        paper_type or "-",
    )
    return rag_context


def prioritize_kg_by_standard(kg_results: KGResult, question: str, entities: List) -> KGResult:
    """KG 结果按标准号文件名优先排序。"""
    std_keys = extract_std_codes(question, entities)
    if not std_keys or not kg_results or not kg_results.triples:
        return kg_results
    matched = [t for t in kg_results.triples if source_matches_std(t.source or "", std_keys)]
    if not matched:
        return kg_results
    others = [t for t in kg_results.triples if t not in matched]
    kg_results.triples = matched + others[: max(0, 25 - len(matched))]
    return kg_results


def _chunk_key(c: RetrievedChunk) -> str:
    meta = getattr(c, "metadata", None) or {}
    if meta.get("evidence_id"):
        return f"evidence::{meta['evidence_id']}"
    if meta.get("standard_id") and meta.get("clause_id"):
        return f"clause::{meta['standard_id']}::{meta['clause_id']}"
    return str(getattr(c, "chunk_id", None) or id(c))


def merge_chunks_by_id(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Merge retrieval channels while preserving one canonical clause ID."""
    merged: List[RetrievedChunk] = []
    seen = set()
    for chunk in chunks or []:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
    return merged






def parse_cross_dimensions(question: str) -> List[str]:
    """解析题干对比维度（如「在A与B方面」），供 cross 分标准对齐。不扫库、不读答案。"""
    q = question or ""
    m = re.search(r"在(.{4,80}?)方面", q)
    if not m:
        m = re.search(r"围绕(.{4,80}?)(?:应如何|如何统筹|有何)", q)
    # 共享维度：「A和B对传感器安装位置的要求有何不同」。
    if not m:
        m = re.search(r"对(.{2,60}?)(?:的)?(?:要求|规定|应用|设置)?(?:有何|有什么)", q)
    # 前置对象：「电连接器的安装和标识要求在 A 和 B 中有何异同」。
    if not m:
        m = re.search(
            r"^(.{2,60}?)(?:要求|规定)?在\s*(?:(?:GB/T|GBT|GB|HB(?:/Z)?|GJB|QJ)\s*[0-9])",
            q,
            flags=re.I,
        )
    if not m:
        return []
    body = m.group(1)
    parts = re.split(r"(?:\s*[与和、／/]\s*|\s+与\s+)", body)
    out = []
    for p in parts:
        p = re.sub(r"\s+", "", (p or "").strip(" ，,、"))
        if len(p) >= 2:
            out.append(p)
    # 去重保序
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq[:4]


def side_topic_for_standard(question: str, std_index: int) -> str:
    """第 i 个标准对应的题干侧主题；无则回退整题。"""
    dims = parse_cross_dimensions(question)
    # 一个维度是多标准共享的比较对象，不是只属于第一个标准。
    if len(dims) == 1:
        return dims[0]
    if dims and 0 <= std_index < len(dims):
        return dims[std_index]
    return question or ""


def extract_question_std_surfaces(question: str) -> List[str]:
    """题干里的标准号原样（含年号括号），供答案与金标写法对齐。"""
    out: List[str] = []
    seen = set()
    for m in _STD_PATTERN.finditer(question or ""):
        raw = re.sub(r"\s+", " ", (m.group(0) or "").strip(" ，,;；"))
        raw = raw.rstrip("的和与及")
        if len(raw) < 6:
            continue
        key = re.sub(r"\s+", "", raw).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def _canon_token_to_id(token: str) -> str:
    """把 gbt:28878:1:2012 或 GB/T 28878.1-2012 归一成 canonical。"""
    t = (token or "").strip().strip("{}[]")
    t = re.sub(r"\s+", " ", t)
    if not t:
        return ""
    low = t.lower()
    m = re.match(
        r"^(gbt|gb|hbz|hb|gjb|qj):(\d{2,6}):(\d*):(\d{4})$",
        low,
    )
    if m:
        k, num, part, year = m.groups()
        if k == "gb":
            k = "gbt"
        return f"{k}:{num}:{part}:{year}"
    return canonical_standard_id(t) or ""


def _surface_for_canon(canon: str, surfaces: List[str]) -> str:
    """优先用题干写法，否则从 canonical 还原 GB/T / HB 形式。"""
    for s in surfaces or []:
        if canonical_standard_id(s) == canon:
            return s
    parts = (canon or "").split(":")
    if len(parts) != 4:
        return canon or ""
    k, num, part, year = parts
    prefix = {
        "gbt": "GB/T",
        "gb": "GB",
        "hb": "HB",
        "hbz": "HB/Z",
        "gjb": "GJB",
        "qj": "QJ",
    }.get(k, k.upper())
    mid = f"{num}.{part}" if part else num
    return f"{prefix} {mid}-{year}"


def rewrite_pred_standard_ids(question: str, text: str) -> str:
    """把答案里的内部标准键、模板占位符还原成题干写法，不改正文事实。"""
    if not text:
        return text
    surfaces = extract_question_std_surfaces(question or "")
    out = text
    # 模型常把提示词「{标准A}/{标准1}」原样抄出，按题干顺序换成真实标准号
    if surfaces:
        named = [
            (r"\{标准\s*A\}", 0),
            (r"\{标准\s*B\}", 1),
            (r"\{标准\s*1\}", 0),
            (r"\{标准\s*2\}", 1),
            (r"\{标准\s*3\}", 2),
            (r"\{标准号\}", 0),
        ]
        for pat, idx in named:
            if idx < len(surfaces):
                out = re.sub(pat, surfaces[idx], out)
        out = re.sub(r"\{标准…\}", "、".join(surfaces), out)
    if not surfaces:
        return out

    def _replace_brace(match: re.Match) -> str:
        inner = match.group(1) or ""
        toks = [p.strip() for p in re.split(r"[,，;；、]+", inner) if p.strip()]
        if len(toks) < 1:
            return match.group(0)
        mapped = []
        ok = 0
        for tok in toks:
            cid = _canon_token_to_id(tok)
            if cid:
                mapped.append(_surface_for_canon(cid, surfaces))
                ok += 1
            else:
                mapped.append(tok)
        if ok == 0:
            return match.group(0)
        return "、".join(mapped)

    out = re.sub(r"\{([^{}]{6,160})\}", _replace_brace, out)

    def _replace_bare(match: re.Match) -> str:
        cid = _canon_token_to_id(match.group(0))
        if not cid:
            return match.group(0)
        return _surface_for_canon(cid, surfaces)

    out = re.sub(
        r"\b(?:gbt|gb|hbz|hb|gjb|qj):\d{2,6}:\d*:\d{4}\b",
        _replace_bare,
        out,
        flags=re.I,
    )
    return out


def _strip_std_ids_from_question(question: str) -> str:
    """去掉题干中的标准号，留下主题表述。"""
    topic = re.sub(
        r"(?:GB/T|GBT|GB|HB(?:/Z)?|GJB|QJ)[\s\+]*[\d\-—－/\.A-Za-z（）()]+",
        " ",
        question or "",
        flags=re.I,
    )
    topic = re.sub(r"综合参考|综合", " ", topic)
    return re.sub(r"\s+", " ", topic).strip(" ，,？?。")


def prefer_fuller_clause_text(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """同一 evidence_id 保留更长的条款全文，避免句级切分丢掉金标命题。"""
    best: Dict[str, RetrievedChunk] = {}
    order: List[str] = []
    for c in chunks or []:
        k = _chunk_key(c)
        if k not in best:
            order.append(k)
            best[k] = c
            continue
        old = best[k]
        if len(getattr(c, "chunk_text", "") or "") > len(getattr(old, "chunk_text", "") or ""):
            best[k] = c
    return [best[k] for k in order]


def _clone_chunk_text(
    chunk: RetrievedChunk, text: str, source_tag: str = ""
) -> RetrievedChunk:
    """只换正文，保留条款身份与分数。"""
    return RetrievedChunk(
        chunk_id=getattr(chunk, "chunk_id", "") or "",
        source=getattr(chunk, "source", "") or "",
        chunk_text=str(text or ""),
        metadata=dict(getattr(chunk, "metadata", None) or {}),
        similarity_score=float(getattr(chunk, "similarity_score", 0) or 0),
        rerank_score=float(getattr(chunk, "rerank_score", 0) or 0),
        retrieval_source=source_tag or (getattr(chunk, "retrieval_source", "") or ""),
    )


def _corpus_clause_by_evidence_id(vector_retriever: Any, evidence_id: str) -> str:
    """按 evidence_id 取语料库中更长的条款原文（句窗切分前）。"""
    eid = str(evidence_id or "").strip()
    if not eid or vector_retriever is None:
        return ""
    cache = getattr(vector_retriever, "_eid_full_text", None)
    if cache is None:
        cache = {}
        mapping = getattr(vector_retriever, "chunk_mapping", None) or {}
        getter = getattr(vector_retriever, "_get_best_chunk_text", None)
        for data in mapping.values():
            if not isinstance(data, dict):
                continue
            meta = data.get("metadata") or {}
            key = str(meta.get("evidence_id") or data.get("evidence_id") or "").strip()
            if not key:
                continue
            if callable(getter):
                text = str(getter(data) or "")
            else:
                text = str(
                    data.get("original_text")
                    or data.get("context_text")
                    or data.get("text")
                    or ""
                )
            if len(text) > len(cache.get(key, "")):
                cache[key] = text
        try:
            setattr(vector_retriever, "_eid_full_text", cache)
        except Exception:
            pass
    return str(cache.get(eid) or "")


def expand_hits_to_parent_clauses(
    chunks: List[RetrievedChunk], vector_retriever: Any, max_compact: int = 2000
) -> List[RetrievedChunk]:
    """句级/KG 命中还原为同一 evidence 的条款全文，便于字面抄写金标 chunk。"""
    if not chunks:
        return chunks
    out: List[RetrievedChunk] = []
    for c in chunks:
        text = getattr(c, "chunk_text", "") or ""
        best = text
        tag = getattr(c, "retrieval_source", "") or ""
        parent = _parent_clause_from_sentence(c, vector_retriever) if vector_retriever else None
        ptext = getattr(parent, "chunk_text", "") if parent is not None else ""
        eid = str((getattr(c, "metadata", None) or {}).get("evidence_id") or "").strip()
        etext = _corpus_clause_by_evidence_id(vector_retriever, eid) if eid else ""
        for cand, src_tag in ((ptext, "within_std_parent"), (etext, "eid_parent")):
            if not cand:
                continue
            if len(re.sub(r"\s+", "", cand)) > len(re.sub(r"\s+", "", best)) + 4:
                if len(re.sub(r"\s+", "", cand)) <= max_compact:
                    best = cand
                    tag = src_tag
        if best != text:
            out.append(_clone_chunk_text(c, best, tag))
        else:
            out.append(c)
    return prefer_fuller_clause_text(out)


def _parent_clause_from_sentence(
    chunk: RetrievedChunk, vector_retriever: Any
) -> Optional[RetrievedChunk]:
    """句级命中还原父条款全文。"""
    cid = str(getattr(chunk, "chunk_id", "") or "")
    if "#w" not in cid:
        return None
    parent_idx = cid.split("#w", 1)[0]
    mapping = getattr(vector_retriever, "chunk_mapping", None) or {}
    data = mapping.get(str(parent_idx))
    if data is None:
        try:
            data = mapping.get(int(parent_idx))
        except Exception:
            data = None
    if data is None:
        try:
            data = mapping.get(str(int(parent_idx)))
        except Exception:
            data = None
    if not isinstance(data, dict):
        return None
    getter = getattr(vector_retriever, "_get_best_chunk_text", None)
    text = getter(data) if callable(getter) else (
        data.get("original_text") or data.get("context_text") or data.get("text") or ""
    )
    if not str(text).strip():
        return None
    meta = dict(data.get("metadata") or {})
    fname = meta.get("file_name") or meta.get("source") or data.get("source") or getattr(chunk, "source", "") or ""
    return RetrievedChunk(
        chunk_id=str(parent_idx),
        source=str(fname),
        chunk_text=str(text),
        metadata=meta,
        similarity_score=float(getattr(chunk, "similarity_score", 0) or 0),
        retrieval_source="within_std_parent",
    )


def split_evidence_sentences(text: str) -> List[str]:
    """将段落切成可作答证据句（结构噪声过滤，无题干词加减分）。

    优先按 Markdown 小节标题 / 条款号切开，避免「651贮存+652板级」粘成一句导致错段。
    公式块（$$ / 式中）不与「按公式()」拆开，避免空括号句单独进池。
    """
    if not text:
        return []
    holders: List[str] = []

    def _hold_latex(m):
        holders.append(m.group(0))
        return f"\n⟦FORMULA{len(holders) - 1}⟧\n"

    protected = re.sub(r"\$\$[\s\S]*?\$\$", _hold_latex, text)
    rough = re.sub(r"(?=#{2,4}\s*)", "\n", protected)
    rough = re.sub(r"(?<!\d)(?=\d{3,4}\s*[\u4e00-\u9fff]{2,24})", "\n", rough)
    parts = re.split(r"(?<=[。；;！？\n])|(?=#{2,4})", rough)
    out: List[str] = []
    seen = set()
    for p in parts:
        s = (p or "").strip()
        # 去掉纯标题壳，保留标题后正文
        s = re.sub(r"^#{1,4}\s*", "", s).strip()
        for i, raw in enumerate(holders):
            s = s.replace(f"⟦FORMULA{i}⟧", raw)
        compact = re.sub(r"\s+", "", s)
        max_len = 900 if chunk_has_real_formula(s) else 480
        if not (12 <= len(compact) <= max_len):
            continue
        if sentence_is_unusable_evidence(s) or looks_like_preamble(s):
            continue
        if compact in seen:
            continue
        seen.add(compact)
        out.append(s)
    out = _glue_formula_sentence_parts(out)
    # 切不出句时，整段过短噪声检查后回退
    if not out:
        compact = re.sub(r"\s+", "", text or "")
        if 12 <= len(compact) <= 480 and not sentence_is_unusable_evidence(text):
            return [text.strip()]
        if 12 <= len(compact) <= 1200 and chunk_has_real_formula(text or ""):
            return [text.strip()]
    return out


def _glue_formula_sentence_parts(parts: List[str]) -> List[str]:
    """把「按公式()」与随后的 $$ / 式中 粘回一句。"""
    if not parts:
        return parts
    glued: List[str] = []
    for s in parts:
        if not glued:
            glued.append(s)
            continue
        prev = glued[-1]
        prev_needs = bool(_EMPTY_FORMULA_RE.search(prev) or re.search(r"按公式|公式\s*[\(（]", prev)) and (
            not chunk_has_real_formula(prev)
        )
        if prev_needs and chunk_has_real_formula(s):
            merged = prev.rstrip() + "\n" + s
            if len(re.sub(r"\s+", "", merged)) <= 1200:
                glued[-1] = merged
                continue
        glued.append(s)
    return glued


def question_topic_terms(question: str) -> List[str]:
    """从题干抽取主题词（去掉标准号），仅用于同标准内句级对齐，不扫全库。"""
    q = _STD_PATTERN.sub(" ", question or "")
    stop = {
        "根据", "按照", "规定", "要求", "标准", "分别", "哪些", "什么", "如何", "有何",
        "不同", "异同", "方面", "内容", "开展", "结合", "还应", "提出", "进行", "以及",
        "或者", "综合", "参考", "统筹", "除", "外", "时", "中", "的", "与", "和", "对",
        "在", "为", "是", "了", "等", "及其", "具体", "相关",
    }
    terms: List[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", q):
        if run in stop:
            continue
        if len(run) <= 4:
            if run not in stop:
                terms.append(run)
            # 短维度补充 2 字片段，避免「密封要求」整词匹配过严
            if len(run) >= 3:
                for i in range(len(run) - 1):
                    t = run[i : i + 2]
                    if t not in stop:
                        terms.append(t)
            continue
        # 长串切 2~4 字主题片段
        for n in (4, 3, 2):
            for i in range(0, len(run) - n + 1):
                t = run[i : i + n]
                if t not in stop and t not in terms:
                    terms.append(t)
    # 去重保序，优先较长词
    terms = sorted(dict.fromkeys(terms), key=lambda x: (-len(x), x))
    return terms[:24]



def ce_query_for_chunk(question: str, source: str = "", std_index: int | None = None) -> str:
    """重排查询：交叉/相关题带上该标准侧维度，与训练输入对齐。"""
    q = (question or "").strip()
    if not q:
        return ""
    side = ""
    if std_index is not None:
        side = side_topic_for_standard(q, std_index) or ""
    else:
        stds = extract_std_codes(q)
        if source and stds:
            for i, std in enumerate(stds):
                if source_matches_std(source or "", [std]):
                    side = side_topic_for_standard(q, i) or ""
                    break
    if side and side != q and len(re.sub(r"\s+", "", side)) >= 2:
        return f"关注：{side}\n{q}"
    return q




def inject_kg_source_chunks(
    rag_context: RAGContext,
    vector_retriever: Any,
    per_source: int = 3,
    pool_limit: int = 40,
    question: str = "",
    entities: List = None,
) -> RAGContext:
    """把 KG 三元组的 source 文件块并入候选；有题干标准号时不得跨库。"""
    if not vector_retriever or not rag_context:
        return rag_context
    kg = getattr(rag_context, "kg_results", None)
    triples = getattr(kg, "triples", None) if kg else None
    if not triples:
        return rag_context
    std_keys = extract_std_codes(question, entities)
    sources: List[str] = []
    seen = set()
    for t in triples:
        src = str(getattr(t, "source", None) or "").strip()
        if not src or src in seen:
            continue
        if std_keys and not source_matches_std(src, std_keys):
            continue
        seen.add(src)
        sources.append(src)
    if not sources:
        return rag_context
    try:
        injected = vector_retriever.find_chunks_by_sources(sources, per_source=per_source)
    except Exception as e:
        logger.warning(f"KG source 注入失败: {e}")
        return rag_context
    if not injected:
        return rag_context
    merged = list(rag_context.vector_chunks or [])
    seen_ids = {_chunk_key(c) for c in merged}
    for c in injected:
        k = _chunk_key(c)
        if k not in seen_ids:
            seen_ids.add(k)
            merged.insert(0, c)
    rag_context.vector_chunks = merged[:pool_limit]
    logger.info("correlation KG-source 注入: sources=%d chunks+=%d", len(sources), len(injected))
    return rag_context


def assemble_top_k_override() -> int | None:
    """敏感性：强制组篇条数；未设置时保持原题型上限。"""
    raw = os.environ.get("SCT_ASSEMBLE_TOP_K", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Ignoring invalid SCT_ASSEMBLE_TOP_K=%r", raw)
        return None


def assemble_context_by_paper_type(
    question: str,
    entities: List,
    ranked_chunks: List[RetrievedChunk],
    paper_type: str,
    top_k: int = 15,
) -> List[RetrievedChunk]:
    """Select the top identity-unique evidence using paper budgets K_tau."""
    chunks = []
    seen = set()
    for chunk in ranked_chunks or []:
        key = _chunk_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(chunk)
    if not chunks:
        return []
    forced_k = assemble_top_k_override()
    final_k = forced_k or FINAL_EVIDENCE_BUDGETS.get(paper_type, int(top_k))
    return chunks[:final_k]


class StandardAnchoredRetrievalMixin:
    """StdDirect, WithinStd, identity fusion, and standard-aware reranking."""

    STANDARD_RERANK_BETA = STANDARD_RERANK_BETA

    def _paper_query_type(self, router_response) -> str:
        return paper_query_type(router_response)

    def _resolve_paper_type(self, question: str, entities: List, router_response) -> str:
        return resolve_paper_type(question, entities, router_response)

    def _extract_std_codes(self, question: str, entities: List = None) -> List[str]:
        return extract_std_codes(question, entities)

    def _source_matches_std(self, source: str, std_keys: List[str]) -> bool:
        return source_matches_std(source, std_keys)

    def _sentence_is_unusable_evidence(self, sentence: str) -> bool:
        return sentence_is_unusable_evidence(sentence)

    def _chunk_clause_quality_ok(self, text: str, *args, **kwargs) -> bool:
        return chunk_clause_quality_ok(text)

    def _apply_standard_number_boost(
        self,
        question: str,
        entities: List,
        chunks: List[RetrievedChunk],
        paper_type: str = "cross",
        top_k: int = None,
    ) -> List[RetrievedChunk]:
        if top_k is None:
            top_k = int(getattr(getattr(self, "config", None), "top_k_vector", 20) or 20)
        return apply_standard_number_boost(
            question, entities, chunks, paper_type=paper_type, top_k=top_k
        )

    def _merge_chunks_by_id(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        return merge_chunks_by_id(chunks)

    def _inject_standard_chunks(
        self, question: str, router_response, rag_context: RAGContext, paper_type: str = "cross"
    ) -> RAGContext:
        return inject_standard_chunks(
            question,
            getattr(router_response, "entities", []) or [],
            rag_context,
            getattr(self, "vector_retriever", None),
            top_k_vector=int(getattr(getattr(self, "config", None), "top_k_vector", 20) or 20),
            paper_type=paper_type,
        )

    def _enrich_with_within_std_dense(
        self, question: str, router_response, rag_context: RAGContext, paper_type: str = ""
    ) -> RAGContext:
        # B=20 is the default WithinStd candidate budget for every category.
        limit = 20
        pool = 60
        scale_raw = os.environ.get("SCT_WITHIN_BUDGET_SCALE", "").strip()
        if scale_raw:
            try:
                scale = float(scale_raw)
                limit = max(8, int(round(limit * scale)))
                pool = max(16, int(round(pool * scale)))
            except ValueError:
                logger.warning("Ignoring invalid SCT_WITHIN_BUDGET_SCALE=%r", scale_raw)
        rag_context = enrich_with_within_std_dense(
            question,
            getattr(router_response, "entities", []) or [],
            rag_context,
            getattr(self, "vector_retriever", None),
            limit=limit,
            pool_limit=pool,
            paper_type=paper_type or "",
        )
        return rag_context

    def _prioritize_kg_by_standard(self, kg_results: KGResult, question: str, entities: List) -> KGResult:
        return prioritize_kg_by_standard(kg_results, question, entities)

    def _inject_kg_source_chunks(
        self,
        rag_context: RAGContext,
        question: str = "",
        entities: List = None,
    ) -> RAGContext:
        return inject_kg_source_chunks(
            rag_context,
            getattr(self, "vector_retriever", None),
            per_source=4,
            pool_limit=60,
            question=question,
            entities=entities,
        )

    def _assemble_context_by_paper_type(
        self,
        question: str,
        entities: List,
        ranked_chunks: List[RetrievedChunk],
        paper_type: str,
        top_k: int = 15,
    ) -> List[RetrievedChunk]:
        return assemble_context_by_paper_type(
            question, entities, ranked_chunks, paper_type, top_k=top_k
        )

