# -*- coding: utf-8 -*-
"""SCT-RAG: standard-anchored retrieval with clause-identity preservation.

The final architecture uses StdDirect, within-standard dense retrieval, and
clause-linked KG retrieval. Corpus-wide dense retrieval is instantiated only
by the separately reported DPR baseline.
"""
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from data_types import (
    SystemConfig,
    RAGContext,
    QuestionType,
    KGResult,
    LLMResponse,
    RetrievedChunk,
    evidence_top_k,
)
from query_router import QueryRouter, RouterResponse
from kg_retriever import KnowledgeGraphRetriever
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator
from standard_boost import StandardAnchoredRetrievalMixin

logger = logging.getLogger(__name__)


class SCTRAGSystem(StandardAnchoredRetrievalMixin):
    """Final SCT-RAG pipeline: StdDirect + WithinStd + KG + ID-safe ranking."""

    def __init__(
        self,
        config: SystemConfig,
        use_kg: bool = True,
        use_std_direct: bool = True,
        use_within_std_dense: bool = True,
        use_rerank_boost: bool = True,
        use_global_dense: bool = False,
        use_graphrag_paper: bool = False,
        oracle_routing: bool = False,
        use_gold_type: bool = False,
        force_paper_type: str = None,
    ):
        self.config = config
        self.use_kg = use_kg
        # SCT-RAG uses StdDirect + WithinStd + ClauseKG. Global dense is DPR-only.
        self.use_global_dense = bool(use_global_dense)
        self.use_dense = self.use_global_dense
        self.use_std_direct = bool(use_std_direct)
        self.use_rerank_boost = bool(use_rerank_boost)
        self.oracle_routing = bool(oracle_routing)
        # Main experiments use predicted routes; oracle routing is explicit.
        self.use_gold_type = bool(use_gold_type)
        # Used only by the explicitly labelled category-policy ablation.
        forced = str(force_paper_type or "").strip().lower()
        self.force_paper_type = forced if forced in ("single", "cross", "correlation") else None
        self.use_within_std_dense = bool(use_within_std_dense)
        # The GraphRAG baseline uses local entity neighborhoods and summaries.
        self.use_graphrag_paper = bool(use_graphrag_paper)
        self._run_log_jsonl = None
        self._local_chunks = None

        self.query_router = QueryRouter(config)
        self.kg_retriever = None
        self.vector_retriever = None
        self.llm_generator = LLMGenerator(config)

        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("SCT-RAG 系统初始化完成")

    def _normalize_source(self, raw_src) -> list:
        """将 raw_src 规范为字符串列表。返回至少包含一个字符串的列表。

        - 如果 raw_src 是列表或元组，返回其中每个元素的字符串形式（过滤空值）；
        - 否则返回 [str(raw_src)]（空或 None 会转换为 "unknown"）。
        """
        out = []
        try:
            if isinstance(raw_src, (list, tuple)):
                for s in raw_src:
                    if s is None:
                        continue
                    s_str = str(s).strip()
                    if s_str:
                        out.append(s_str)
            else:
                if raw_src is None:
                    out = ["unknown"]
                else:
                    s_str = str(raw_src).strip()
                    out = [s_str if s_str else "unknown"]
        except Exception:
            out = ["unknown"]
        if not out:
            out = ["unknown"]
        return out
    
    def process_query(
        self,
        question: str,
        dataset_class: str,
        image_base64: str = None,
        qa_item: dict = None,
    ) -> Dict[str, Any]:
        """处理查询的完整流程（StdDirect + WithinStd + clause-linked KG）。"""
        start_time = time.time()
        try:
            # HotPot 局部段落
            if dataset_class == "hotpot":
                try:
                    from hotpot_local_retrieval import get_local_chunks

                    self._local_chunks = get_local_chunks(qa_item)
                except Exception:
                    self._local_chunks = None
            else:
                self._local_chunks = None

            # 1. 路由：ht 走 LLM 路由；其它数据集强制 complex
            if dataset_class == "ht":
                router_response = self.query_router.analyze_query(
                    question, dataset_class=dataset_class
                )
                logger.info(
                    f"问题分析: 类型={router_response.question_type.value}"
                    f"(ID={router_response.type_id}), 实体={router_response.entities}, "
                    f"意图={router_response.intent}"
                )

            else:
                router_response = self.query_router.analyze_query(
                    question, dataset_class=dataset_class
                )
                router_response.type_id = 2
                router_response.question_type = QuestionType.CROSS_STANDARD
                logger.info(
                    f"强制路由为跨文档: 类型={router_response.question_type.value}"
                    f"(ID={router_response.type_id}), 实体={router_response.entities}, "
                    f"意图={router_response.intent}"
                )

            # Oracle experiment changes only the route label. Entity extraction
            # and every downstream retrieval/generation setting remain fixed.
            if self.oracle_routing and dataset_class == "ht" and qa_item:
                oracle_map = {
                    "single": (1, QuestionType.SINGLE_STANDARD),
                    "cross": (2, QuestionType.CROSS_STANDARD),
                    "correlation": (3, QuestionType.CORRELATION),
                }
                gold_type = str(
                    qa_item.get("type") or qa_item.get("question_type") or ""
                ).strip().lower()
                if gold_type in oracle_map:
                    router_response.type_id, router_response.question_type = oracle_map[gold_type]
                    router_response.metadata = dict(router_response.metadata or {})
                    router_response.metadata.update(
                        {"oracle_routing": True, "oracle_gold_type": gold_type}
                    )

            if self.use_kg and self.kg_retriever is None:
                self.kg_retriever = KnowledgeGraphRetriever(
                    self.config, dataset_class=dataset_class
                )
            if (
                self.use_global_dense
                or self.use_std_direct
                or self.use_within_std_dense
            ) and self.vector_retriever is None:
                self.vector_retriever = VectorRetriever(
                    self.config, dataset_class=dataset_class
                )

            # 2. 混合检索；GraphRAG 走论文 local/global，不走 SCT 混合检索
            is_ht = dataset_class == "ht"
            if getattr(self, "use_graphrag_paper", False) and is_ht:
                rag_context = self._graphrag_paper_retrieve(
                    question, router_response, dataset_class
                )
            else:
                rag_context = self._hybrid_retrieval(question, router_response, dataset_class)

            # 工业集：按标准号优先 KG（论文 GraphRAG 已在社区内排序，不再二次打乱）
            if is_ht and rag_context.kg_results and not getattr(self, "use_graphrag_paper", False):
                rag_context.kg_results = self._prioritize_kg_by_standard(
                    rag_context.kg_results, question, router_response.entities
                )
                # KG 不只用于抽象实体/关系：命中的三元组必须回到带
                # canonical evidence_id 的条款原文，才能与 StdDirect/WithinStd
                # 一起进入同一去重、重排、生成和 Clause Hit@k 协议。
                rag_context = self._materialize_kg_clause_chunks(rag_context)

            # 标准约束检索：先定 paper_type，再执行 StdDirect/WithinStd。
            paper_type = self._resolve_paper_type(
                question, router_response.entities, router_response
            )
            gold_item_type = str(
                (qa_item or {}).get("type") or (qa_item or {}).get("question_type") or ""
            ).strip().lower()
            if self.use_gold_type:
                if gold_item_type == "correlation":
                    paper_type = "correlation"
                elif gold_item_type in ("single", "cross"):
                    paper_type = gold_item_type
            if self.force_paper_type:
                paper_type = self.force_paper_type

            retrieve_type = paper_type

            if is_ht and self.use_std_direct:
                rag_context = self._inject_standard_chunks(
                    question, router_response, rag_context, paper_type=retrieve_type
                )

            # 同标准内二次稠密检索（句级）。
            if (
                is_ht
                and retrieve_type in ("single", "cross", "correlation")
                and self.use_within_std_dense
            ):
                rag_context = self._enrich_with_within_std_dense(
                    question,
                    router_response,
                    rag_context,
                    paper_type=retrieve_type,
                )

            # 跨标准：KG source 仅补题干标准号对应文件，禁止关键词扩到其它库
            if (
                is_ht
                and retrieve_type in ("cross", "correlation")
                and not getattr(self, "use_graphrag_paper", False)
            ):
                rag_context = self._inject_kg_source_chunks(
                    rag_context,
                    question=question,
                    entities=router_response.entities,
                )

            # 3. CrossEncoder 重排 + RerankBoost + 题型化上下文组装
            if rag_context.vector_chunks and self.vector_retriever:
                top_k = evidence_top_k(self.config)
                # MergeByID is applied before cross-encoder reranking so that
                # all channels share one canonical clause-level candidate set.
                pool = self._merge_chunks_by_id(list(rag_context.vector_chunks))
                rag_context.vector_chunks = list(pool)
                if self.use_rerank_boost and dataset_class != "hotpot":
                    scored = self.vector_retriever.rerank_chunks(
                        question, pool, top_k=max(40, len(pool))
                    )
                    # 保留条款身份融合后的稳定顺序，CE 仅作来源感知重排。
                    # alpha 仅在 dev 选择，test 评估时固定为 0.5。
                    rank_by_key = {}
                    for rank, chunk in enumerate(pool):
                        meta = getattr(chunk, "metadata", None) or {}
                        key = meta.get("evidence_id") or getattr(chunk, "chunk_id", str(rank))
                        rank_by_key[str(key)] = rank
                    ce_values = [float(getattr(chunk, "rerank_score", 0.0) or 0.0) for chunk in scored]
                    if scored and ce_values:
                        # Industrial SCT-RAG keeps the raw cross-encoder score;
                        # the standard-number multiplier is applied next.
                        if is_ht:
                            for chunk, ce_score in zip(scored, ce_values):
                                meta = dict(getattr(chunk, "metadata", None) or {})
                                meta["ce_raw"] = float(ce_score)
                                chunk.metadata = meta
                                chunk.rerank_score = float(ce_score)
                        else:
                            ce_lo, ce_hi = min(ce_values), max(ce_values)
                            denom = max(1, len(pool) - 1)
                            alpha = float(
                                getattr(self.config, "rerank_ce_alpha", 0.5) or 0.5
                            )
                            for chunk, ce_score in zip(scored, ce_values):
                                meta = dict(getattr(chunk, "metadata", None) or {})
                                meta["ce_raw"] = float(ce_score)
                                chunk.metadata = meta
                                key = str(meta.get("evidence_id") or getattr(chunk, "chunk_id", ""))
                                original_rank = rank_by_key.get(key, len(pool) - 1)
                                prior = 1.0 - float(original_rank) / float(denom)
                                ce_norm = (ce_score - ce_lo) / max(ce_hi - ce_lo, 1e-8)
                                chunk.rerank_score = alpha * ce_norm + (1.0 - alpha) * prior
                        scored.sort(
                            key=lambda chunk: float(getattr(chunk, "rerank_score", 0.0) or 0.0),
                            reverse=True,
                        )
                    if is_ht:
                        scored = self._apply_standard_number_boost(
                            question,
                            router_response.entities,
                            scored,
                            paper_type=retrieve_type,
                            top_k=len(scored),
                        )
                        rag_context.reranked_chunks = self._assemble_context_by_paper_type(
                            question,
                            router_response.entities,
                            scored,
                            retrieve_type,
                            top_k=top_k,
                        )
                    else:
                        rag_context.reranked_chunks = scored[:top_k]
                else:
                    ranked = pool[: max(40, top_k)]
                    if (
                        is_ht
                        and not self.use_rerank_boost
                        and not self.use_std_direct
                    ):
                        # Pure DPR baseline: preserve the dense rank exactly.
                        # Applying paper-type assembly here made the declared DPR
                        # baseline depend on an SCT-specific heuristic.
                        rag_context.reranked_chunks = ranked[:top_k]
                    elif is_ht:
                        rag_context.reranked_chunks = self._assemble_context_by_paper_type(
                            question,
                            router_response.entities,
                            ranked,
                            retrieve_type,
                            top_k=top_k,
                        )
                        # 最终方法不再调用句子级后选。
                    else:
                        # HotPot：保留局部 dense+BM25+实体覆盖排序，勿用英文假 CE 打乱
                        rag_context.reranked_chunks = ranked[:top_k]
                    if dataset_class == "hotpot":
                        logger.info("HotPot 已跳过 CrossEncoder/RerankBoost，保留局部多跳排序")
                    elif not self.use_rerank_boost:
                        logger.info("已跳过 CrossEncoder/RerankBoost（use_rerank_boost=False）")

            # GraphRAG 等无向量器路径：仍要用物化后的条款列表参与 Hit@5 与 [n] 映射
            if is_ht and rag_context.vector_chunks and not rag_context.reranked_chunks:
                rag_context.reranked_chunks = list(rag_context.vector_chunks)[
                    : evidence_top_k(self.config)
                ]

            # 4. Build the selected evidence context and generate the answer.
            if dataset_class == "hotpot":
                try:
                    from hotpot_citation_utils import build_hotpot_llm_context

                    # 确保局部段落尽量齐全（distractor 仅 10 段）
                    if self._local_chunks:
                        existing = {
                            getattr(c, "source", None)
                            for c in (rag_context.reranked_chunks or rag_context.vector_chunks or [])
                        }
                        from data_types import RetrievedChunk

                        extra = []
                        for lc in self._local_chunks:
                            src = str(lc.get("source") or "")
                            if src in existing:
                                continue
                            txt = (
                                lc.get("chunk")
                                or lc.get("chunk_text")
                                or lc.get("text")
                                or ""
                            ).strip()
                            if not txt:
                                continue
                            extra.append(
                                RetrievedChunk(
                                    chunk_id=str(lc.get("id", "")),
                                    source=src,
                                    chunk_text=txt,
                                    metadata={"title": lc.get("title", ""), "file_name": src},
                                    similarity_score=0.01,
                                    retrieval_source="local_fill",
                                )
                            )
                        if extra:
                            base = list(rag_context.reranked_chunks or rag_context.vector_chunks or [])
                            rag_context.reranked_chunks = base + extra
                    enhanced_context = build_hotpot_llm_context(rag_context, top_k=12)
                except Exception:
                    enhanced_context = rag_context.get_enhanced_context()
            else:
                llm_chunks = list(
                    rag_context.reranked_chunks or rag_context.vector_chunks or []
                )
                llm_chunks = llm_chunks[: evidence_top_k(self.config)]
                enhanced_context = rag_context.get_enhanced_context(doc_chunks=llm_chunks)
                setattr(
                    rag_context,
                    "_llm_context_chunks",
                    llm_chunks,
                )

            llm_response = self.llm_generator.generate_answer(
                question,
                enhanced_context,
                dataset_class=dataset_class,
                query_type=paper_type if is_ht else None,
            )
            # HotPot 短答精炼（可选）
            if dataset_class == "hotpot":
                try:
                    refined = self._refine_hotpot_response(
                        question, rag_context, llm_response
                    )
                    if refined:
                        llm_response = refined
                except Exception:
                    pass

            # 仅保留检索侧侧主题 CE + 维度对齐上下文

            self._save_rag_context(rag_context, router_response)
            total_time = time.time() - start_time
            result = self._package_results(
                question=question,
                router_response=router_response,
                rag_context=rag_context,
                llm_response=llm_response,
                total_time=total_time,
            )
            self._save_result(result, dataset_class)
            return result
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat(),
            }

    def _graphrag_paper_retrieve(
        self,
        question: str,
        router_response: RouterResponse,
        dataset_class: str,
    ) -> RAGContext:
        """论文 GraphRAG：local 原文条款 + global 标准社区摘要，再对齐语料 identity。"""
        start_time = time.time()
        rag_context = RAGContext(
            question=question,
            question_type=router_response.question_type,
            retrieval_time=0.0,
        )
        if self.kg_retriever is None:
            self.kg_retriever = KnowledgeGraphRetriever(
                self.config, dataset_class=dataset_class
            )
        kg_result, reports = self.kg_retriever.query_graphrag(
            question, router_response.entities, router_response.question_type
        )
        rag_context.kg_results = kg_result
        rag_context = self._materialize_kg_clause_chunks(rag_context)
        rag_context = self._rank_graphrag_text_units(question, rag_context)
        setattr(rag_context, "_community_reports", reports or [])
        rag_context.retrieval_time = time.time() - start_time
        logger.info(
            "GraphRAG 完成: KG=%d clauses=%d reports=%d",
            len(kg_result.triples or []),
            len(rag_context.vector_chunks or []),
            len(reports or []),
        )
        return rag_context

    def _rank_graphrag_text_units(
        self, question: str, rag_context: RAGContext
    ) -> RAGContext:
        """Local search：对实体邻域原文按题干词面排序，截断为证据 top-k。"""
        chunks = list(rag_context.vector_chunks or [])
        if not chunks:
            rag_context.reranked_chunks = []
            return rag_context
        from compare_rag.utils import tokenize_zh

        q_toks = set(tokenize_zh(question))
        wants_def = bool(re.search(r"何谓|是什么|定义|指什么", question or ""))
        wants_formula = bool(re.search(r"公式|式中|附录|计算|η", question or ""))

        def _score(chunk) -> float:
            text = getattr(chunk, "chunk_text", "") or ""
            meta = getattr(chunk, "metadata", None) or {}
            t_toks = set(tokenize_zh(text))
            lex = len(q_toks & t_toks) / max(1, len(q_toks))
            cid = str(meta.get("clause_id") or "")
            bonus = 0.0
            if wants_def and re.match(r"^3(\.|$)", cid) and re.search(r"指|定义为|是由", text):
                bonus += 0.4
            if wants_formula and re.search(r"C\.|\\\\frac|η\s*=|公式", text, re.I):
                bonus += 0.5
            return lex + bonus

        ranked = sorted(chunks, key=_score, reverse=True)
        top_k = evidence_top_k(self.config)
        rag_context.vector_chunks = ranked
        rag_context.reranked_chunks = ranked[:top_k]
        return rag_context
    
    def _hybrid_retrieval(self, question: str, router_response: RouterResponse, dataset_class: str) -> RAGContext:
        """
        执行混合检索
        """
        start_time = time.time()
        
        # 初始化上下文
        rag_context = RAGContext(
            question=question,
            question_type=router_response.question_type,
            retrieval_time=0.0
        )
        
        # 根据问题类型执行不同检索策略
        if router_response.question_type == QuestionType.SINGLE_STANDARD:
            # 单文档：KG为主，向量为辅
            logger.info("使用单文档检索策略")
            rag_context = self._retrieve_for_single_standard(question, router_response, rag_context, dataset_class)
            
        elif router_response.question_type == QuestionType.CROSS_STANDARD:
            # 跨文档：KG多跳推理 + 向量语义检索
            logger.info("使用跨文档检索策略")
            rag_context = self._retrieve_for_cross_standard(question, router_response, rag_context, dataset_class)
            
        else:  # QuestionType.CORRELATION
            # 关联：向量为主，KG提供锚点
            logger.info("使用关联检索策略")
            rag_context = self._retrieve_for_correlation(question, router_response, rag_context, dataset_class)
        
        rag_context.retrieval_time = time.time() - start_time
        
        logger.info(f"检索完成: KG三元组={len(rag_context.kg_results.triples) if rag_context.kg_results else 0}, 向量块={len(rag_context.vector_chunks)}")

        # HotPot GraphRAG：把局部原文注入段落列表，便于生成与 Hit@5
        if dataset_class == "hotpot" and self.use_kg and not self.use_global_dense:
            rag_context = self._inject_hotpot_kg_passages(rag_context)
        elif dataset_class == "hotpot" and self.use_kg and self.use_global_dense:
            # SCT：用局部子图命中源提升向量段落排序
            rag_context = self._boost_hotpot_chunks_by_kg(rag_context)

        return rag_context

    def _kg_query(self, question: str, router_response: RouterResponse, dataset_class: str):
        """统一 KG 查询；HotPot 传入本题 local_chunks。"""
        return self.kg_retriever.query_kg(
            question=question,
            entities=router_response.entities,
            question_type=router_response.question_type,
            dataset_class=dataset_class,
            local_chunks=self._local_chunks if dataset_class == "hotpot" else None,
        )

    def _materialize_kg_clause_chunks(self, rag_context: RAGContext) -> RAGContext:
        """将 KG 命中物化为可验证条款候选，并对齐语料库条款全文/evidence_id。"""
        triples = list(getattr(rag_context.kg_results, "triples", None) or [])
        if not triples:
            return rag_context
        merged = list(rag_context.vector_chunks or [])
        seen = set()
        for chunk in merged:
            meta = getattr(chunk, "metadata", None) or {}
            key = meta.get("evidence_id") or (
                f"{meta.get('standard_id')}::{meta.get('clause_id')}"
                if meta.get("standard_id") and meta.get("clause_id") else getattr(chunk, "chunk_id", "")
            )
            if key:
                seen.add(str(key))
        for rank, triple in enumerate(triples):
            text = str(getattr(triple, "paragraph", "") or "").strip()
            evidence_id = str(getattr(triple, "evidence_id", "") or "").strip()
            standard_id = str(getattr(triple, "standard_id", "") or "").strip()
            clause_id = str(getattr(triple, "clause_id", "") or "").strip()
            source = str(getattr(triple, "source", "") or "").strip()
            if not text and not evidence_id:
                continue
            if not evidence_id and not (standard_id and clause_id):
                continue
            key = evidence_id or f"{standard_id}::{clause_id}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                RetrievedChunk(
                    chunk_id=evidence_id or key,
                    source=source,
                    chunk_text=text,
                    metadata={
                        "file_name": source,
                        "standard_id": standard_id,
                        "clause_id": clause_id,
                        "evidence_id": evidence_id,
                    },
                    similarity_score=max(0.05, float(getattr(triple, "confidence", 0.0) or 0.0)),
                    retrieval_source="clause_linked_kg",
                )
            )
        corpus_path = str(getattr(self.config, "vector_db_path1", "") or "")
        if corpus_path:
            try:
                from compare_rag.utils import align_retrieved_to_corpus_clauses

                merged = align_retrieved_to_corpus_clauses(merged, corpus_path)
            except Exception as e:
                logger.warning("KG 条款对齐语料库失败，沿用图谱短句: %s", e)
        rag_context.vector_chunks = merged
        return rag_context

    def _inject_hotpot_kg_passages(self, rag_context: RAGContext) -> RAGContext:
        """将局部 KG 命中的 text_unit 转为 RetrievedChunk，供 LLM/评测使用。"""
        from data_types import RetrievedChunk

        units = list(getattr(self.kg_retriever, "_hotpot_last_text_units", None) or [])
        chunks: List[RetrievedChunk] = []
        seen = set()
        for i, tu in enumerate(units):
            uid = str(tu.get("id") or "")
            text = str(tu.get("content") or "").strip()
            if not uid or not text or uid in seen:
                continue
            seen.add(uid)
            chunks.append(
                RetrievedChunk(
                    chunk_id=uid,
                    source=uid,
                    chunk_text=text[:2000],
                    metadata={"title": tu.get("title") or "", "file_name": uid},
                    similarity_score=1.0 - 0.02 * i,
                    retrieval_source="kg_text_unit",
                )
            )
        if rag_context.kg_results and rag_context.kg_results.triples:
            for t in rag_context.kg_results.triples:
                src = str(t.source or "")
                para = str(t.paragraph or "").strip()
                if not src or not para or src in seen:
                    continue
                seen.add(src)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=src,
                        source=src,
                        chunk_text=para[:2000],
                        metadata={"title": t.head or "", "file_name": src},
                        similarity_score=float(t.confidence or 0.5),
                        retrieval_source="kg_triple_para",
                    )
                )
        # 兜底：本题局部段中标题/正文与 KG 实体相关的段落
        if self._local_chunks and len(chunks) < 4:
            ents = []
            if rag_context.kg_results:
                ents = list(rag_context.kg_results.entities or [])
            for lc in self._local_chunks:
                src = str(lc.get("source") or "")
                if not src or src in seen:
                    continue
                title = str(lc.get("title") or "")
                txt = (lc.get("chunk") or lc.get("chunk_text") or lc.get("text") or "").strip()
                if not txt:
                    continue
                tl = f"{title}\n{txt}".lower()
                if ents and not any(str(e).lower() in tl for e in ents if len(str(e)) >= 4):
                    continue
                seen.add(src)
                chunks.append(
                    RetrievedChunk(
                        chunk_id=src,
                        source=src,
                        chunk_text=txt[:2000],
                        metadata={"title": title, "file_name": src},
                        similarity_score=0.55,
                        retrieval_source="kg_local_fallback",
                    )
                )
                if len(chunks) >= 8:
                    break
        if chunks:
            rag_context.vector_chunks = chunks
            rag_context.reranked_chunks = chunks
        return rag_context

    def _boost_hotpot_chunks_by_kg(self, rag_context: RAGContext) -> RAGContext:
        """SCT：把局部 KG 命中的原文源抬到检索列表前面。"""
        units = list(getattr(self.kg_retriever, "_hotpot_last_text_units", None) or [])
        prefer = {str(u.get("id") or "") for u in units if u.get("id")}
        if rag_context.kg_results and rag_context.kg_results.triples:
            for t in rag_context.kg_results.triples:
                if t.source:
                    prefer.add(str(t.source))
        if not prefer:
            return rag_context
        pool = list(rag_context.reranked_chunks or rag_context.vector_chunks or [])
        if not pool:
            return rag_context
        head, tail = [], []
        for c in pool:
            src = str(getattr(c, "source", "") or "")
            (head if src in prefer else tail).append(c)
        if head:
            for i, c in enumerate(head):
                try:
                    c.similarity_score = max(float(c.similarity_score or 0), 0.95 - 0.01 * i)
                except Exception:
                    pass
            merged = head + tail
            rag_context.reranked_chunks = merged
            if rag_context.vector_chunks:
                rag_context.vector_chunks = merged
        return rag_context    
    def _retrieve_for_single_standard(self, question: str, router_response: RouterResponse, 
                                 rag_context: RAGContext, dataset_class: str) -> RAGContext:
        """单文档检索策略"""
        # 1. KG检索（主要）
        if self.use_kg and self.kg_retriever:
            rag_context.kg_results = self._kg_query(question, router_response, dataset_class)
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
        
        # 2. 向量检索（补充）
        if self.vector_retriever and self.use_global_dense:
            pure_dpr = bool(self.use_global_dense and not self.use_kg)
            kg_entities = [] if pure_dpr else (
                rag_context.kg_results.entities if rag_context.kg_results else []
            )
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=[] if pure_dpr else router_response.entities,
                kg_entities=kg_entities,
                question_type=router_response.question_type,
                local_chunks=self._local_chunks if dataset_class == "hotpot" else None,
                use_dense=True,
                use_bm25=False,
            )
            rag_context.vector_chunks = vector_chunks
        else:
            rag_context.vector_chunks = []
        
        return rag_context
    
    def _retrieve_for_cross_standard(self, question: str, router_response: RouterResponse,
                                   rag_context: RAGContext, dataset_class: str) -> RAGContext:
        """跨文档检索策略"""
        # 1. KG检索（多跳推理）
        if self.use_kg and self.kg_retriever:
            rag_context.kg_results = self._kg_query(question, router_response, dataset_class)
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
        
        # 2. 向量检索（语义补充）
        # 规范化并去重路由器和 KG 返回的实体
        all_entities = []
        seen_entities = set()
        for e in router_response.entities:
            key = ', '.join(map(str, e)) if isinstance(e, (list, tuple)) else str(e)
            if key and key not in seen_entities:
                seen_entities.add(key)
                all_entities.append(key)

        if rag_context.kg_results and rag_context.kg_results.entities:
            for e in rag_context.kg_results.entities:
                key = ', '.join(map(str, e)) if isinstance(e, (list, tuple)) else str(e)
                if key and key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(key)

        if self.vector_retriever and self.use_global_dense:
            pure_dpr = bool(self.use_global_dense and not self.use_kg)
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=[] if pure_dpr else router_response.entities,
                kg_entities=[] if pure_dpr else all_entities,
                question_type=router_response.question_type,
                local_chunks=self._local_chunks if dataset_class == "hotpot" else None,
                use_dense=True,
                use_bm25=False,
            )
            rag_context.vector_chunks = vector_chunks
        else:
            rag_context.vector_chunks = []
        
        return rag_context
    
    def _retrieve_for_correlation(self, question: str, router_response: RouterResponse,
                                   rag_context: RAGContext, dataset_class: str) -> RAGContext:
        """关联检索策略"""
        # 1. 先 KG，再向量（原先顺序导致向量阶段 kg_entities 恒为空）
        if self.use_kg and self.kg_retriever:
            rag_context.kg_results = self._kg_query(question, router_response, dataset_class)
        else:
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)

        all_entities = []
        seen = set()
        for e in list(router_response.entities or []) + list(
            getattr(rag_context.kg_results, "entities", None) or []
        ):
            key = ", ".join(map(str, e)) if isinstance(e, (list, tuple)) else str(e)
            if key and key not in seen:
                seen.add(key)
                all_entities.append(key)

        # 从 KG 三元组抽取短实体（供稠密辅查询，不做题干关键词扫库）
        triple_terms = []
        for t in list(getattr(rag_context.kg_results, "triples", None) or [])[:40]:
            for part in (
                getattr(t, "head", None),
                getattr(t, "tail", None),
                getattr(t, "relation", None),
            ):
                s = re.sub(r"\s+", "", str(part or ""))
                if 2 <= len(s) <= 12 and s not in seen:
                    seen.add(s)
                    triple_terms.append(s)
        all_entities = all_entities + triple_terms[:8]

        if self.vector_retriever and self.use_global_dense:
            pure_dpr = bool(self.use_global_dense and not self.use_kg)
            # 主查询（稠密检索）
            main = self.vector_retriever.retrieve(
                question=question,
                entities=[] if pure_dpr else router_response.entities,
                kg_entities=[] if pure_dpr else all_entities,
                question_type=router_response.question_type,
                local_chunks=self._local_chunks if dataset_class == "hotpot" else None,
                use_dense=True,
                use_bm25=False,
            )
            # 辅查询：路由/KG 实体（语义检索，禁止题干短语/参数词字面漏题）
            aux_chunks = []
            aux_terms = [] if pure_dpr else [
                t for t in all_entities if 1 < len(re.sub(r"\s+", "", t)) <= 16
            ][:8]
            if aux_terms:
                try:
                    aux_chunks = self.vector_retriever.retrieve(
                        question=" ".join(aux_terms),
                        entities=router_response.entities,
                        kg_entities=all_entities,
                        question_type=router_response.question_type,
                        local_chunks=self._local_chunks if dataset_class == "hotpot" else None,
                        use_dense=True,
                        use_bm25=False,
                    )
                except Exception as e:
                    logger.warning(f"correlation 实体辅查询失败: {e}")
            merged = []
            seen_ids = set()
            for c in list(main or []) + list(aux_chunks or []):
                k = self.vector_retriever._evidence_key(c)
                if k not in seen_ids:
                    seen_ids.add(k)
                    merged.append(c)
            # 注入 KG 段落原文
            if not pure_dpr and self.use_kg:
                for t in list(getattr(rag_context.kg_results, "triples", None) or [])[:20]:
                    para = (getattr(t, "paragraph", None) or "").strip()
                    src = str(getattr(t, "source", "") or "").strip()
                    if len(para) < 24 or not src:
                        continue
                    k = f"kgpara::{src}::{hash(para) & 0xfffffff}"
                    if k in seen_ids:
                        continue
                    seen_ids.add(k)
                    merged.append(
                        RetrievedChunk(
                            chunk_id=k,
                            source=src,
                            chunk_text=para,
                            metadata={"file_name": src},
                            similarity_score=0.72,
                            retrieval_source="kg_paragraph",
                        )
                    )
            rag_context.vector_chunks = merged[:32]
        else:
            rag_context.vector_chunks = []

        return rag_context
    
    def _save_rag_context(self, rag_context: RAGContext, router_response: RouterResponse):
        """保存RAG上下文（用于评估）"""
        try:
            # 提取详细信息
            context_details = {
                "question": rag_context.question,
                "question_type": rag_context.question_type.value,
                "router_entities": router_response.entities,
                "router_intent": router_response.intent,
                "retrieval_time": rag_context.retrieval_time,
                "kg_results": None,
                "vector_results": [],
                "reranked_results": [],
                "combined_context": rag_context.get_enhanced_context()
            }
            
            # KG结果详情：按 source 分组，每个 source 包含段落列表，每段落包含相关三元组
            if rag_context.kg_results:
                from collections import defaultdict

                # 按 source -> paragraph -> 列表（三元组）分组
                grouped = defaultdict(lambda: defaultdict(list))
                for triple in rag_context.kg_results.triples:
                    para = triple.paragraph or ""
                    norm_srcs = self._normalize_source(triple.source)
                    for src in norm_srcs:
                        grouped[src][para].append({
                            "head": triple.head,
                            "relation": triple.relation,
                            "tail": triple.tail,
                            "confidence": triple.confidence
                        })

                ke_results = []
                for src, para_map in grouped.items():
                    para_list = []
                    for para_text, triples in para_map.items():
                        para_list.append({
                            "text": para_text,
                            "triples": triples
                        })

                    ke_results.append({
                        "source": str(src),
                        "paragrepa": para_list
                    })

                context_details["kg_results"] = {
                    "ke_results": ke_results,
                    "entities": rag_context.kg_results.entities,
                    "query_time": rag_context.kg_results.query_time
                }
            
            # 向量结果详情
            for chunk in rag_context.vector_chunks:
                # 将 chunk.source 规范为字符串输出
                csrc = chunk.source
                csrc_str = ', '.join(map(str, csrc)) if isinstance(csrc, (list, tuple)) else str(csrc)
                context_details["vector_results"].append({
                    "chunk_id": chunk.chunk_id,
                    "source": csrc_str,
                    "text_preview": chunk.chunk_text,
                    "similarity_score": chunk.similarity_score
                })
            
            # 重排序结果详情
            for chunk in rag_context.reranked_chunks:
                rsrc = chunk.source
                rsrc_str = ', '.join(map(str, rsrc)) if isinstance(rsrc, (list, tuple)) else str(rsrc)
                context_details["reranked_results"].append({
                    "chunk_id": chunk.chunk_id,
                    "source": rsrc_str,
                    "text_preview": chunk.chunk_text,
                    "rerank_score": chunk.rerank_score
                })
            
            # # 保存到文件
            
        except Exception as e:
            logger.error(f"保存RAG上下文失败: {e}")
    
    @staticmethod
    def _explicit_clause_citations(response: str, context_refs: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
        """Accept only evidence IDs explicitly emitted by the model and present in context."""
        normalized_refs = {str(key).lower(): value for key, value in context_refs.items()}
        from compare_rag.utils import citation_scan_text

        scan = citation_scan_text(response or "")
        explicit_ids = re.findall(r"\bev_[0-9a-f]{20}\b", scan, flags=re.I)
        citations = []
        seen = set()
        for evidence_id in explicit_ids:
            key = evidence_id.lower()
            ref = normalized_refs.get(key)
            if ref and key not in seen:
                seen.add(key)
                citations.append(ref)
        return citations

    def _package_results(self, question: str, router_response: RouterResponse,
                        rag_context: RAGContext, llm_response: LLMResponse,
                        total_time: float) -> Dict[str, Any]:
        """打包所有结果"""
        # 将 KG 结果转换为按 source 分组的结构（与保存上下文的格式一致）
        kg_dict = {}
        if rag_context.kg_results:
            from collections import defaultdict
            grouped = defaultdict(lambda: defaultdict(list))
            for triple in rag_context.kg_results.triples:
                para = triple.paragraph or ""
                norm_srcs = self._normalize_source(triple.source)
                for src in norm_srcs:
                    grouped[src][para].append({
                        "head": triple.head,
                        "relation": triple.relation,
                        "tail": triple.tail,
                        "confidence": triple.confidence,
                        "standard_id": triple.standard_id,
                        "clause_id": triple.clause_id,
                        "evidence_id": triple.evidence_id,
                    })

            ke_results = []
            for src, para_map in grouped.items():
                para_list = []
                for para_text, triples in para_map.items():
                    para_list.append({
                        "text": para_text,
                        "triples": triples
                    })

                ke_results.append({
                    "source": str(src),
                    "paragrepa": para_list
                })

            kg_dict = {
                "ke_results": ke_results,
                "entities": rag_context.kg_results.entities,
                "query_time": rag_context.kg_results.query_time
            }
        else:
            kg_dict = {
                "ke_results": [],
                "entities": [],
                "query_time": 0.0
            }
        
        # 构建候选 md 列表：来自 KG ke_results 的 source 与 重排序结果的 source
        candidate_md = []
        # 从 ke_results 提取
        try:
            for item in kg_dict.get("ke_results", []):
                src = None
                if isinstance(item, dict):
                    src = item.get("source")
                if src is not None:
                    src_str = ', '.join(map(str, src)) if isinstance(src, (list, tuple)) else str(src)
                    # 提取以 .md 结尾的文件名或路径最后一部分
                    m = re.search(r'([^\\/\s]+\.md)', src_str, re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        # 有时 source 本身就是文件名但不包含 .md，尝试直接取字符串
                        candidate_md.append(src_str)
        except Exception:
            pass

        # 从重排序结果中提取 source
        try:
            for c in rag_context.reranked_chunks:
                if isinstance(c, dict):
                    src = c.get('source')
                else:
                    src = getattr(c, 'source', None)
                if src is not None:
                    src_str = ', '.join(map(str, src)) if isinstance(src, (list, tuple)) else str(src)
                    m = re.search(r'([^\\/\s]+\.md)', src_str, re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        candidate_md.append(src_str)
        except Exception:
            pass

        # 去重并保持顺序
        seen = set(); candidate_md_list = []
        for x in candidate_md:
            if x and x not in seen:
                seen.add(x); candidate_md_list.append(x)

        # 从 LLM 原始响应中提取【证据】区文本，作为 evidence_block 保存
        evidence_block = ""
        try:
            em = re.search(r'【(?:证据|Evidence)】\s*(.*)', llm_response.raw_response or "", re.DOTALL)
            if em:
                evidence_block = em.group(1).strip()
        except Exception:
            evidence_block = ""

        # 从答案 [n]、【证据】区、真实 .md 名提取引用；禁止模板「文件名.md」
        matched_citations = []
        try:
            raw = llm_response.raw_response or ""
            ans = llm_response.answer or ""
            blob = f"{ans}\n{evidence_block}\n{raw}"
            low_blob = blob.lower()
            # [n] 对候选文件顺序（与检索证据列表一致）
            for n_s in re.findall(r"\[(\d+)\]", blob):
                idx = int(n_s) - 1
                if 0 <= idx < len(candidate_md_list):
                    md = candidate_md_list[idx]
                    if md and "文件名" not in md and md not in matched_citations:
                        matched_citations.append(md)
            for md in candidate_md_list:
                if md and md.lower() in low_blob and "文件名" not in md:
                    if md not in matched_citations:
                        matched_citations.append(md)
            found = re.findall(r"([^\\/\s]+\.md)", blob, re.IGNORECASE)
            for f in found:
                fn = f.split("/")[-1]
                if (not fn) or ("文件名" in fn):
                    continue
                if fn not in matched_citations:
                    if fn in candidate_md_list or any(
                        fn.lower() == x.lower() for x in candidate_md_list
                    ):
                        matched_citations.append(fn)
        except Exception:
            matched_citations = []
        matched_citations = [
            m for m in matched_citations if m and "文件名" not in str(m)
        ]

        # 为返回结构准备向量与重排序结果，确保 source 字段为字符串
        vector_results_out = []
        for c in rag_context.vector_chunks:
            try:
                d = c.to_dict()
            except Exception:
                d = {k: getattr(c, k, None) for k in ('chunk_id', 'chunk_text', 'similarity_score')}
            csrc = getattr(c, 'source', None)
            csrc_str = ', '.join(map(str, csrc)) if isinstance(csrc, (list, tuple)) else str(csrc)
            d['source'] = csrc_str
            vector_results_out.append(d)

        def _chunks_to_out(chunks):
            out = []
            for c in chunks or []:
                try:
                    d = c.to_dict()
                except Exception:
                    d = {k: getattr(c, k, None) for k in ('chunk_id', 'chunk_text', 'rerank_score')}
                rsrc = getattr(c, 'source', None)
                rsrc_str = ', '.join(map(str, rsrc)) if isinstance(rsrc, (list, tuple)) else str(rsrc)
                d['source'] = rsrc_str
                out.append(d)
            return out

        # 句级重排原序保留；Hit@5/Recall@5 用生成侧父条款还原后的列表，与 Hybrid 整段证据同粒度
        sentence_reranked_out = _chunks_to_out(rag_context.reranked_chunks)
        llm_chunks = getattr(rag_context, "_llm_context_chunks", None)
        reranked_results_out = _chunks_to_out(llm_chunks) if llm_chunks else sentence_reranked_out

        # Clause-v1 protocol: only explicit, context-backed evidence IDs count.
        # Do not infer a clause from a document-only citation.
        context_refs = {}
        for c in (
            list(getattr(rag_context, "_llm_context_chunks", None) or [])
            + list(rag_context.reranked_chunks or [])
            + list(rag_context.vector_chunks or [])
        ):
            meta = getattr(c, "metadata", None) or {}
            evidence_id = meta.get("evidence_id")
            if evidence_id:
                context_refs[evidence_id] = {
                    "evidence_id": evidence_id,
                    "standard_id": meta.get("standard_id", ""),
                    "clause_id": meta.get("clause_id", ""),
                }
        for triple in list(getattr(rag_context.kg_results, "triples", None) or []):
            if triple.evidence_id:
                context_refs[triple.evidence_id] = {
                    "evidence_id": triple.evidence_id,
                    "standard_id": triple.standard_id,
                    "clause_id": triple.clause_id,
                }
        from compare_rag.utils import explicit_clause_citation_refs
        cite_chunks = reranked_results_out or vector_results_out
        citation_refs = explicit_clause_citation_refs(
            llm_response.raw_response or "", cite_chunks
        )
        # 仍保留原显式 ev_ 抽取，合并进 citation_refs（不覆盖已有 id）
        explicit_only = self._explicit_clause_citations(
            llm_response.raw_response, context_refs
        )
        seen_eids = {
            str(r.get("evidence_id") or "").lower() for r in citation_refs if r
        }
        for ref in explicit_only:
            key = str(ref.get("evidence_id") or "").lower()
            if key and key not in seen_eids:
                seen_eids.add(key)
                citation_refs.append(ref)

        return {
            "question": question,
            "router_analysis": {
                "type_id": router_response.type_id,
                "question_type": router_response.question_type.value,
                "entities": router_response.entities,
                "intent": router_response.intent,
                "metadata": router_response.metadata
            },
            "retrieval": {
                "kg_results": kg_dict,
                "vector_results": vector_results_out,
                "reranked_results": reranked_results_out,
                "sentence_reranked_results": sentence_reranked_out,
                "retrieval_time": rag_context.retrieval_time
            },
            "generation": {
                "answer": llm_response.answer,
                # matched_citations 基于上下文 candidate_md_list 与 LLM 依据块的匹配结果
                "citations_all": candidate_md_list,
                "citation_llm_use": matched_citations,
                "citation_refs": citation_refs,
                "evidence_block": evidence_block,
                "generation_time": llm_response.generation_time,
                "raw_response": llm_response.raw_response
            },
            "performance": {
                "total_time": total_time,
                "retrieval_time": rag_context.retrieval_time,
                "generation_time": llm_response.generation_time
            },
            "timestamp": datetime.now().isoformat(),
            # "system_info": {
            #     "kg_uri": self.config.neo4j_uri,
            #     "vector_db": str(self.config.vector_db_path1),
            #     "llm_model": self.config.llm_model
            # }
        }

    def _save_result(self, result: Dict[str, Any], dataset_class: str = None):
        """保存结果到文件"""
        output_path = Path(self.config.output_path)
        
        
        # A benchmark may bind a method-specific JSONL destination.
        if getattr(self, "_run_log_jsonl", None):
            log_file = Path(self._run_log_jsonl)
        else:
            log_file = output_path / f"sct_rag_log_{dataset_class}.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"结果已保存: {log_file}")
    
    def close(self):
        """关闭连接"""
        if self.kg_retriever:
            try:
                close_fn = getattr(self.kg_retriever, 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        if self.vector_retriever:
            try:
                close_fn = getattr(self.vector_retriever, 'close', None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass


