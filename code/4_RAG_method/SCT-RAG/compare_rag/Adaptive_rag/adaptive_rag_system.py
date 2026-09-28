# adaptive_rag_system.py
"""Adaptive-RAG 系统 - 根据问题复杂度动态选择检索策略"""
import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from data_types import (
    SystemConfig, RAGContext, QuestionType, KGResult, RetrievedChunk, LLMResponse
)
from query_router import QueryRouter, RouterResponse
from kg_retriever import KnowledgeGraphRetriever
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator
from compare_rag.Adaptive_rag.adaptive_router import AdaptiveRouter

logger = logging.getLogger(__name__)


class AdaptiveRAGSystem:
    """Adaptive-RAG 系统"""
    
    def __init__(self, config: SystemConfig, use_kg: bool = True, use_vector: bool = True, adaptive_model_dir: str = None):
        self.config = config
        self.use_kg = use_kg
        self.use_vector = use_vector
        
        # 初始化所有底层组件（始终加载，按需调用）
        self.adaptive_router = AdaptiveRouter(config, model_dir=adaptive_model_dir)
        self.query_router = QueryRouter(config)
        # 延迟按 dataset_class 实例化 KnowledgeGraphRetriever 与 VectorRetriever
        self.kg_retriever = None
        self.vector_retriever = None
        self.config = config
        self.llm_generator = LLMGenerator(config)
        
        # 创建输出目录
        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.context_save_path = output_path / "adaptive_contexts"
        self.context_save_path.mkdir(exist_ok=True)
        
        logger.info("Adaptive-RAG 系统初始化完成")

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> Dict[str, Any]:
        """Process a query using single, cross, or correlation routing."""
        from hotpot_local_retrieval import get_local_chunks
        self._local_chunks = get_local_chunks(qa_item) if dataset_class == "hotpot" else None
        
        if self.use_kg and self.kg_retriever is None:
            self.kg_retriever = KnowledgeGraphRetriever(self.config, dataset_class=dataset_class)

        if self.use_vector and self.vector_retriever is None:
            self.vector_retriever = VectorRetriever(self.config, dataset_class=dataset_class)

        start_time = time.time()

        
        try:
            if dataset_class == "hotpot" or dataset_class == "other":
                adaptive_label = "cross"
            else:
                adaptive_label, _ = self.adaptive_router.classify(question)

            type_mapping = {
                'single': QuestionType.SINGLE_STANDARD,
                'cross': QuestionType.CROSS_STANDARD,
                'correlation': QuestionType.CORRELATION,
            }
            question_type = type_mapping.get(adaptive_label, QuestionType.SINGLE_STANDARD)
            
            router_response = RouterResponse(
                question_type=question_type,
                type_id=self._get_type_id(question_type),
                entities=[],
                intent="adaptive_router",
                metadata={"adaptive_label": adaptive_label}
            )
            
            # 实体提取仍可复用 QueryRouter 的分析（可选）
            try:
                base_analysis = self.query_router.analyze_query(question)
                router_response.entities = base_analysis.entities
                router_response.intent = base_analysis.intent
            except Exception as e:
                logger.warning(f"QueryRouter 分析失败，使用默认实体: {e}")
            
            logger.info(f"Adaptive-RAG 路由: label={adaptive_label} → type={question_type.value}")

            rag_context = self._adaptive_retrieval(
                question, router_response, adaptive_label, dataset_class
            )
                
            if rag_context.vector_chunks:
                rag_context.reranked_chunks = self.vector_retriever.rerank_chunks(
                    question, rag_context.vector_chunks
                )
            else:
                rag_context.reranked_chunks = []

            if dataset_class == "hotpot":
                from hotpot_citation_utils import build_hotpot_llm_context
                enhanced_context = build_hotpot_llm_context(
                    rag_context, top_k=self.config.top_k_vector
                )
            else:
                enhanced_context = rag_context.get_enhanced_context()
            llm_response = self.llm_generator.generate_answer(
                question, enhanced_context, dataset_class
            )

            # Step 5: 打包结果
            total_time = time.time() - start_time
            result = self._package_results(
                question=question,
                router_response=router_response,
                rag_context=rag_context,
                llm_response=llm_response,
                total_time=total_time,
                dataset_class=dataset_class,
            )
            
            # Step 6: 保存结果
            self._save_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Adaptive-RAG 处理查询失败: {e}")
            return {
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }

    # adaptive_rag_system.py

    def _local_retrieve_kwargs(self, dataset_class: str) -> dict:
        if dataset_class == "hotpot" and getattr(self, "_local_chunks", None):
            return {"local_chunks": self._local_chunks}
        return {}

    def _adaptive_multistep_retrieve(
        self,
        question: str,
        router_response: RouterResponse,
        first_chunks: list,
        dataset_class: str,
    ) -> list:
        """论文 Adaptive-RAG 的 multi-step：根据首轮证据生成下一跳查询再检索。"""
        from data_types import RetrievedChunk

        merged = list(first_chunks or [])
        seen = set()
        for c in merged:
            key = ""
            if hasattr(c, "metadata"):
                key = str((c.metadata or {}).get("evidence_id") or getattr(c, "chunk_id", ""))
            elif isinstance(c, dict):
                key = str((c.get("metadata") or {}).get("evidence_id") or c.get("chunk_id") or "")
            if key:
                seen.add(key)
        snippets = []
        for i, c in enumerate(merged[:6], start=1):
            text = getattr(c, "chunk_text", None) if not isinstance(c, dict) else (
                c.get("chunk_text") or c.get("text_preview") or ""
            )
            snippets.append(f"[{i}] {(text or '')[:220]}")
        prompt = (
            "这是多跳问答的中间步。根据问题和已检索证据，写出下一条检索查询，"
            "必须保留标准号与尚未覆盖的问点。只输出查询。\n"
            f"问题：{question}\n证据：\n{chr(10).join(snippets) or '（空）'}\n/no_think"
        )
        follow = ""
        try:
            raw = self.llm_generator._call_llm(prompt) or ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip()
            follow = raw.splitlines()[0].strip().strip('"')
        except Exception:
            follow = ""
        if not (8 <= len(follow) <= 200) or follow == question:
            return merged
        try:
            extra = self.vector_retriever.retrieve(
                question=follow,
                entities=router_response.entities,
                kg_entities=[],
                **self._local_retrieve_kwargs(dataset_class),
            )
        except Exception:
            extra = []
        for c in extra or []:
            key = str((getattr(c, "metadata", None) or {}).get("evidence_id") or getattr(c, "chunk_id", ""))
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            merged.append(c)
        top_k = int(getattr(self.config, "top_k_vector", 20) or 20)
        return merged[:top_k]

    def _adaptive_retrieval(self, question: str, router_response: RouterResponse, label: str, dataset_class: str) -> RAGContext:
        """Apply one-step retrieval to single and iterative retrieval otherwise."""
        start_time = time.time()
        rag_context = RAGContext(
            question=question,
            question_type=router_response.question_type,
            retrieval_time=0.0
        )
        
        if label == 'single':
            logger.info("Adaptive-RAG: single-step retrieval")
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=router_response.entities,
                kg_entities=[],  # 不传入 KG 实体
                **self._local_retrieve_kwargs(dataset_class),
            )
            rag_context.vector_chunks = vector_chunks
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
            
        elif label in {'cross', 'correlation'}:
            logger.info("Adaptive-RAG: iterative retrieval for %s", label)
            if self.use_kg and self.kg_retriever is not None:
                kg_results = self.kg_retriever.query_kg(
                    question=question,
                    entities=router_response.entities,
                    question_type=router_response.question_type,
                    dataset_class=dataset_class
                )
                rag_context.kg_results = kg_results
            else:
                rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
            kg_entities = rag_context.kg_results.entities if rag_context.kg_results else []
            vector_chunks = self.vector_retriever.retrieve(
                question=question,
                entities=router_response.entities,
                kg_entities=kg_entities if self.use_kg else [],
                **self._local_retrieve_kwargs(dataset_class),
            )
            vector_chunks = self._adaptive_multistep_retrieve(
                question, router_response, vector_chunks, dataset_class
            )
            rag_context.vector_chunks = vector_chunks
        else:
            # Unknown predictions use the single-step policy.
            vector_chunks = self.vector_retriever.retrieve(
                question, router_response.entities, [],
                **self._local_retrieve_kwargs(dataset_class),
            )
            rag_context.vector_chunks = vector_chunks
            rag_context.kg_results = KGResult(triples=[], entities=[], query_time=0.0)
        
        rag_context.retrieval_time = time.time() - start_time
        return rag_context


    def _get_type_id(self, qtype: QuestionType) -> int:
        mapping = {
            QuestionType.SINGLE_STANDARD: 1,
            QuestionType.CROSS_STANDARD: 2,
            QuestionType.CORRELATION: 3
        }
        return mapping.get(qtype, 1)

    def _package_results(self, question: str, router_response: RouterResponse,
                        rag_context: RAGContext, llm_response: LLMResponse,
                        total_time: float, dataset_class: str = "ht") -> Dict[str, Any]:
        """打包所有结果"""
        # 将 KG 结果转换为按 source 分组的结构（与保存上下文的格式一致）
        kg_dict = {}
        if rag_context.kg_results:
            from collections import defaultdict
            grouped = defaultdict(lambda: defaultdict(list))
            for triple in rag_context.kg_results.triples:
                src = triple.source or "unknown"
                para = triple.paragraph or ""
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
                    "source": src,
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
                src = item.get("source") if isinstance(item, dict) else None
                if src:
                    # 提取以 .md 结尾的文件名或路径最后一部分
                    m = re.search(r'([^\\/\s]+\.md)', str(src), re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        # 有时 source 本身就是文件名但不包含 .md，尝试直接取字符串
                        candidate_md.append(str(src))
        except Exception:
            pass

        # 从重排序结果中提取 source
        try:
            for c in rag_context.reranked_chunks:
                src = getattr(c, 'source', None) or (c.get('source') if isinstance(c, dict) else None)
                if src:
                    m = re.search(r'([^\\/\s]+\.md)', str(src), re.IGNORECASE)
                    if m:
                        fname = m.group(1).split('/')[-1]
                        candidate_md.append(fname)
                    else:
                        candidate_md.append(str(src))
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

        # 引用提取
        matched_citations = []
        try:
            if dataset_class == "hotpot":
                from hotpot_citation_utils import resolve_hotpot_citations
                matched_citations = resolve_hotpot_citations(
                    answer=llm_response.answer or "",
                    raw_response=llm_response.raw_response or "",
                    evidence_citations=llm_response.evidence_citations,
                    rag_context=rag_context,
                    evidence_block=evidence_block,
                )
            else:
                low_block = (evidence_block or "").lower()
                for md in candidate_md_list:
                    if md.lower() in low_block:
                        matched_citations.append(md)
                if not matched_citations and evidence_block:
                    found = re.findall(r'([^\\/\s]+\.md)', evidence_block, re.IGNORECASE)
                    seen2 = set()
                    for f in found:
                        fn = f.split('/')[-1]
                        if fn not in seen2:
                            seen2.add(fn); matched_citations.append(fn)
        except Exception:
            matched_citations = llm_response.evidence_citations or []

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
                "vector_results": [c.to_dict() for c in rag_context.vector_chunks],
                "reranked_results": [c.to_dict() for c in rag_context.reranked_chunks],
                "retrieval_time": rag_context.retrieval_time
            },
            "generation": {
                "answer": llm_response.answer,
                # matched_citations 基于上下文 candidate_md_list 与 LLM 依据块的匹配结果
                "citations_all": candidate_md_list,
                "citation_llm_use": matched_citations,
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
            "system_info": {
                "kg_uri": self.config.neo4j_uri,
                "vector_db": str(self.config.vector_db_path1),
                "llm_model": self.config.llm_model
            }
        }

    def _save_result(self, result: Dict[str, Any]):
        from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
        log_file = resolve_run_jsonl(
            self,
            getattr(self, "_run_method_name", "Adaptive-RAG"),
            getattr(self, "_output_subdir", ""),
        )
        append_run_jsonl(log_file, result)
        logger.info("Adaptive-RAG 结果已保存: %s", log_file)

    def close(self):
        """关闭资源"""
        if self.kg_retriever is not None:
            try:
                self.kg_retriever.close()
            except Exception:
                pass
