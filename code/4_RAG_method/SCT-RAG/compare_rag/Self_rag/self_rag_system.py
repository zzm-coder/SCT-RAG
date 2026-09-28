"""Self-RAG implementation (skeleton).
Provides SelfRAGSystem class with `process_query` similar to HybridRAGSystem interface.
Uses SystemConfig from data_types and interacts with HybridRAGSystem components where possible.
"""
from typing import Dict, Any
from data_types import SystemConfig
from vector_retriever import VectorRetriever
from llm_generator import LLMGenerator
import time
import re
from pathlib import Path
import json
import logging
from compare_rag.utils import build_ctx_from_chunks, baseline_router_stub

logger = logging.getLogger(__name__)


class SelfRAGSystem:
    """Self-RAG 实现：向量检索 + LLM 自反思循环

    行为：
    - Self-RAG 主循环：initial -> (retrieve -> refine) x N -> critique
    - 不依赖 KG
    """

    def __init__(self, config: SystemConfig, use_vector: bool = True):
        self.config = config
        self.use_vector = use_vector
        self.vector_retriever = None
        self.llm = LLMGenerator(config)
        self.llm_generator = self.llm

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> Dict[str, Any]:

        if self.use_vector and self.vector_retriever is None:
                self.vector_retriever = VectorRetriever(self.config, dataset_class=dataset_class)
        
        from hotpot_local_retrieval import get_local_chunks
        self._local_chunks = get_local_chunks(qa_item) if dataset_class == "hotpot" else None
        
        start = time.time()

        # Self-RAG 主循环：initial -> (retrieve -> refine) x N -> critique
        max_iterations = 2
        all_retrieved = []
        retrieval_queries = []

        # 初次调用：要求模型输出初步答案与是否需要检索及检索查询（JSON 格式首选）
        init_prompt = self._build_prompt_initial(question) + "\n\n请以 JSON 格式返回: {\"initial_answer\":..., \"should_retrieve\": true/false, \"retrieval_queries\": [..]}"
        try:
            raw_init = self.llm._call_llm(init_prompt)
            raw_init = re.sub(r'<think>.*?</think>', '', raw_init, flags=re.DOTALL | re.IGNORECASE)
        except Exception as e:
            logger.warning(f"SelfRAG 初次调用 LLM 失败: {e}")
            raw_init = ''

        initial_answer = ''
        should_retrieve = False
        parse_ok = False
        try:
            m = re.search(r'\{[\s\S]*?\}', raw_init)
            if m:
                parsed = json.loads(m.group())
                parse_ok = True
                initial_answer = parsed.get('initial_answer','')
                should_retrieve = bool(parsed.get('should_retrieve', False))
                retrieval_queries = parsed.get('retrieval_queries') or []
                if isinstance(retrieval_queries, str):
                    retrieval_queries = [retrieval_queries]
        except Exception:
            parse_ok = False
            # 回退到解析文本答案
            try:
                initial_answer, _ = self.llm._parse_response(raw_init)
            except Exception:
                initial_answer = raw_init.strip().split('\n\n')[0] if raw_init else ''
        if (not parse_ok) or (
            not should_retrieve and "<retrieve>" in (raw_init or "").lower()
        ):
            should_retrieve = True
            if not retrieval_queries:
                retrieval_queries = [question]

        result = {
            'question': question,
            'router_analysis': baseline_router_stub('Self-RAG'),
            'generation': {'initial_answer': initial_answer, 'answer': None, 'citation_llm_use': []},
            'retrieval': {'vector_results': []},
            'analysis': {'retrieval_queries': retrieval_queries, 'iterations': 0, 'critique': None},
            'performance': {'retrieval_time': 0.0, 'generation_time': 0.0, 'total_time': 0.0}
        }

        if 'not_supported' in (initial_answer or '').lower() or '<not_supported>' in (raw_init or '').lower():
            result['generation']['answer'] = 'NOT_SUPPORTED'
            result['analysis']['critique'] = 'model_refused'
            result['performance']['total_time'] = time.time() - start
            if result['performance']['retrieval_time'] == 0.0:
                result['performance']['retrieval_time'] = max(
                    0.0, result['performance']['total_time'] - result['performance']['generation_time']
                )
            self._append_result_log(result)
            return result

        # 论文：模型未发出 Retrieve 则无检索直接作答
        if (
            not should_retrieve
            and not retrieval_queries
            and "<retrieve>" not in (raw_init or "").lower()
        ):
            gen_start = time.time()
            try:
                gen_obj = self.llm.generate_answer(question, "", dataset_class)
                result['generation']['answer'] = gen_obj.answer or initial_answer
                result['generation']['citation_llm_use'] = gen_obj.evidence_citations
                result['generation']['raw_response'] = gen_obj.raw_response or raw_init
            except Exception:
                result['generation']['answer'] = initial_answer or ''
            result['performance']['generation_time'] += time.time() - gen_start
            result['analysis']['critique'] = {'needs_more_retrieval': False, 'notes': 'no_retrieve_token'}
            result['performance']['total_time'] = time.time() - start
            self._append_result_log(result)
            return result

        # 迭代检索与生成
        for it in range(max_iterations):
            result['analysis']['iterations'] = it + 1
            queries = retrieval_queries if retrieval_queries else []
            if not queries and it == 0 and should_retrieve:
                # Reuse the question if no separate retrieval query is emitted.
                queries = [' '.join([t for t in question.split() if len(t) > 1][:6])]

            # 执行检索（可能为空）
            retrieved_passages = []
            retr_start = time.time()
            for q in queries[:5]:
                try:
                    vecs = self.vector_retriever.retrieve(
                        question=q, entities=[], kg_entities=[],
                        local_chunks=self._local_chunks,
                    )
                    for v in vecs:
                        retrieved_passages.append(v.to_dict() if hasattr(v, 'to_dict') else v)
                except Exception:
                    continue

            # 如果没有显式 queries 且迭代为0，可进行 cheap retrieval from question
            if not queries and it == 0 and not retrieved_passages:
                simple_q = ' '.join([t for t in question.split() if len(t) > 1][:6])
                vecs = self.vector_retriever.retrieve(
                    question=simple_q, entities=[], kg_entities=[],
                    local_chunks=self._local_chunks,
                )
                retrieved_passages.extend([v.to_dict() if hasattr(v, 'to_dict') else v for v in vecs])
            result['performance']['retrieval_time'] += time.time() - retr_start

            # 合并、去重并截取 top-K
            seen = set(); unique = []
            for p in retrieved_passages:
                txt = p.get('chunk_text') or p.get('text_preview','') or str(p)
                key = (p.get('source',''), txt[:200])
                if key not in seen:
                    seen.add(key); unique.append(p)
            unique = unique[: self.config.top_k_vector]
            # 论文 Self-RAG：对每条证据做 ISREL，丢掉 irrelevant
            unique = self._filter_relevant_passages(question, unique) or unique
            # Rerank the merged candidates after each retrieval round.
            merged = all_retrieved + unique
            best = {}
            for p in merged:
                meta = dict(p.get('metadata') or {})
                key = (
                    meta.get('evidence_id') or p.get('evidence_id') or
                    (p.get('source', ''), (p.get('chunk_text') or p.get('text_preview', ''))[:200])
                )
                score = float(p.get('similarity_score') or p.get('score') or 0.0)
                if key not in best or score > best[key][0]:
                    best[key] = (score, p)
            all_retrieved = [item[1] for item in sorted(best.values(), key=lambda x: x[0], reverse=True)]
            result['retrieval']['vector_results'] = all_retrieved[: self.config.top_k_vector]
            result['retrieval']['all_iteration_candidates'] = all_retrieved

            ctx = build_ctx_from_chunks(all_retrieved, limit=self.config.top_k_vector) if dataset_class != "hotpot" else ""
            gen_start = time.time()
            final_raw_response = ""
            try:
                if dataset_class == "hotpot":
                    from hotpot_citation_utils import build_hotpot_ctx_from_passages
                    ctx = build_hotpot_ctx_from_passages(unique, top_k=min(10, len(unique)))
                else:
                    ctx = build_ctx_from_chunks(all_retrieved, limit=self.config.top_k_vector)
                gen_obj = self.llm.generate_answer(question, ctx, dataset_class)
                final_answer = gen_obj.answer
                final_raw_response = gen_obj.raw_response or ""
                if dataset_class == "hotpot":
                    from hotpot_citation_utils import resolve_hotpot_citations_from_passages
                    final_cites = resolve_hotpot_citations_from_passages(
                        answer=final_answer,
                        raw_response=gen_obj.raw_response or "",
                        evidence_citations=gen_obj.evidence_citations,
                        passages=unique,
                    )
                else:
                    final_cites = gen_obj.evidence_citations
            except Exception as e:
                logger.warning(f"Self-RAG 生成失败: {e}")
                final_answer = ''
                final_cites = []
            result['performance']['generation_time'] += time.time() - gen_start

            result['generation']['answer'] = final_answer
            result['generation']['citation_llm_use'] = final_cites
            result['generation']['raw_response'] = final_raw_response

            # Self-reflection / critique：让模型基于证据评估当前答案是否充分
            critique_prompt = (
                f"请基于以下证据评估答案是否充分且仅基于证据回答。\n证据:\n{ctx}\n\n问题:{question}\n\n答案:{final_answer}\n\n"
                "请以 JSON 返回 {\"needs_more_retrieval\": true/false, "
                "\"issup\": \"fully_supported|partially_supported|no_support\", "
                "\"isuse\": \"5|4|3|2|1\", \"notes\": \"...\", \"retrieval_queries\": []}"
            )
            try:
                raw_crit = self.llm._call_llm(critique_prompt)
                raw_crit = re.sub(r'<think>.*?</think>', '', raw_crit, flags=re.DOTALL | re.IGNORECASE)
                m = re.search(r'\{[\s\S]*?\}', raw_crit)
                critique = json.loads(m.group()) if m else {'needs_more_retrieval': False, 'notes': raw_crit}
            except Exception:
                critique = {'needs_more_retrieval': False, 'notes': ''}

            result['analysis']['critique'] = critique
            if str(critique.get("issup") or "").lower() in ("no_support", "not_supported"):
                critique["needs_more_retrieval"] = True

            # 若不需要更多检索，则结束
            if not critique.get('needs_more_retrieval', False):
                break
            else:
                # 若 critique 建议新的检索 queries，则使用；否则扩展现有检索
                new_qs = critique.get('suggested_queries') or critique.get('retrieval_queries') or []
                if new_qs:
                    retrieval_queries = new_qs
                else:
                    # 扩展简单关键词
                    retrieval_queries = retrieval_queries + [' '.join([t for t in question.split() if len(t) > 1][:8])] if retrieval_queries else [' '.join([t for t in question.split() if len(t) > 1][:8])]

        result['performance']['total_time'] = time.time() - start
        if result['performance']['retrieval_time'] == 0.0:
            result['performance']['retrieval_time'] = max(
                0.0, result['performance']['total_time'] - result['performance']['generation_time']
            )
        if result['generation']['answer'] is None:
            result['generation']['answer'] = initial_answer or ''

        self._append_result_log(result)

        logger.info('Self-RAG 处理完成')
        return result

    def _append_result_log(self, result: Dict[str, Any]) -> None:
        """Log every return path, including early NOT_SUPPORTED decisions."""
        try:
            from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
            log_file = resolve_run_jsonl(
                self,
                getattr(self, "_run_method_name", "Self-RAG"),
                getattr(self, "_output_subdir", ""),
            )
            append_run_jsonl(log_file, result)
        except Exception:
            logger.debug('无法写入 Self-RAG 日志', exc_info=True)
        

    def close(self):
        """Release resources after a benchmark run."""
        self.vector_retriever = None

    def _filter_relevant_passages(self, question: str, passages: list) -> list:
        """Self-RAG ISREL：只保留与问题相关的条款，避免无关证据进入生成。"""
        if not passages:
            return []
        kept = []
        for p in passages[: min(8, self.config.top_k_vector)]:
            if not isinstance(p, dict):
                continue
            text = (p.get("chunk_text") or p.get("text_preview") or "")[:320]
            ident = str((p.get("metadata") or {}).get("evidence_id") or "")
            prompt = (
                "判断证据是否与问题相关。只输出 relevant 或 irrelevant。\n"
                f"问题：{question}\n证据 {ident}：{text}\n/no_think"
            )
            try:
                raw = self.llm._call_llm(prompt) or ""
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.I | re.S).strip().lower()
            except Exception:
                kept.append(p)
                continue
            if "irrelevant" in raw and "relevant" not in raw.replace("irrelevant", ""):
                continue
            if "irrelevant" in raw:
                continue
            kept.append(p)
        return kept or list(passages[: self.config.top_k_vector])

    def _build_prompt_initial(self, question: str) -> str:
        return (
            "若回答需要外部证据，请在回答中插入特殊标记 <RETRIEVE>；"
            "若需要引用请插入 <CITE> 并在回答末尾列出引用来源；"
            "若问题超出资料范围，请输出 <NOT_SUPPORTED>。\n"
            f"问题：{question}\n\n输出格式：\n\n【答案】...\n\n【证据】..."
        )
