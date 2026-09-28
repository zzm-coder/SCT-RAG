from data_types import SystemConfig
from rank_bm25 import BM25Okapi
from llm_generator import LLMGenerator
from compare_rag.utils import (tokenize_zh, build_ctx_from_chunks,
                               baseline_router_stub, load_clause_corpus,
                               explicit_clause_citation_refs)
import logging
import time
import os
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BM25system:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.corpus_meta = []
        self.corpus_texts = []
        self._build_corpus()
        if self.corpus_texts:
            tokenized = [tokenize_zh(t) for t in self.corpus_texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
        self.llm_generator = LLMGenerator(config)
        self._hotpot_vr = None

    def _build_corpus(self):
        """遍历 doc_dir，读取每个 md 文件并拆分为段落，保存段落与源文件映射"""
        clause_rows = load_clause_corpus(getattr(self.config, "vector_db_path1", ""))
        if clause_rows:
            self.corpus_meta = clause_rows
            self.corpus_texts = [row["text"] for row in clause_rows]
            return
        doc_dir = getattr(self.config, 'doc_dir', None)
        if not doc_dir or not os.path.isdir(doc_dir):
            logger.warning(f"doc_dir 未配置或不存在: {doc_dir}")
            return

        id_counter = 0
        for root, dirs, files in os.walk(doc_dir):
            for filename in files:
                if not filename.endswith('.md'):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw = f.read()
                except Exception:
                    continue

                parts = [p.strip() for p in re.split(r'\n\s*\n+', raw) if p.strip()]
                for p in parts:
                    cid = f"p{id_counter}"
                    self.corpus_meta.append({'id': cid, 'filename': filename, 'text': p})
                    self.corpus_texts.append(p)
                    id_counter += 1

        # KG paragraph 补充语料（由 enrich_vector_index_kg_paragraphs.py 生成）
        sup_path = getattr(self.config, 'vector_db_path1', None)
        if sup_path:
            sup_file = Path(sup_path) / "kg_paragraph_bm25_supplement.json"
            if sup_file.exists():
                try:
                    sup_items = json.loads(sup_file.read_text(encoding='utf-8'))
                    for item in sup_items:
                        text = str(item.get('text') or '').strip()
                        if not text:
                            continue
                        cid = str(item.get('id') or f"kgp{id_counter}")
                        fname = str(item.get('filename') or 'kg_paragraph.md')
                        self.corpus_meta.append({'id': cid, 'filename': fname, 'text': text})
                        self.corpus_texts.append(text)
                        id_counter += 1
                except Exception as e:
                    logger.warning(f"加载 KG BM25 补充语料失败: {e}")

    def retrieve(self, question: str, top_k: int = 5):
        """使用 BM25 对段落级别进行检索，返回带 source 和 chunk_text 的列表"""
        if not self.bm25:
            return []
        q_tokens = tokenize_zh(question)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            meta = self.corpus_meta[idx]
            results.append({
                'chunk_id': meta['id'],
                'source': meta['filename'],
                'chunk_text': meta['text'],
                'metadata': dict(meta.get('metadata') or {}),
                'similarity_score': float(score)
            })
        return results

    def process_query(self, question: str, dataset_class: str, qa_item: dict = None) -> dict:
        start_time = time.time()
        try:
            top_k = getattr(self.config, 'top_k_vector', 20)
            if dataset_class == 'hotpot':
                from vector_retriever import VectorRetriever
                from hotpot_local_retrieval import get_local_chunks, bm25_rank_local, dense_rank_local, merge_local_hits
                from hotpot_citation_utils import build_hotpot_ctx_from_passages, resolve_hotpot_citations_from_passages
                if self._hotpot_vr is None:
                    self._hotpot_vr = VectorRetriever(self.config, dataset_class='hotpot')
                retrieval_start = time.time()
                local = get_local_chunks(qa_item)
                if local:
                    # 论文 BM25：只走稀疏通道。
                    _ = dense_rank_local(question, local, self._hotpot_vr.embedding_model, top_k=top_k)
                    hits = bm25_rank_local(question, local, top_k=top_k)
                else:
                    hits = self._hotpot_vr.retrieve(question=question, entities=[], kg_entities=[])
                passages = []
                for h in hits:
                    d = h.to_dict() if hasattr(h, 'to_dict') else h
                    passages.append({
                        'chunk_id': d.get('chunk_id'),
                        'source': d.get('source', ''),
                        'chunk_text': d.get('chunk_text') or d.get('text', ''),
                        'similarity_score': float(d.get('similarity_score', 0)),
                    })
                retrieval_time = time.time() - retrieval_start
                context = build_hotpot_ctx_from_passages(passages, top_k=min(10, len(passages)))
            else:
                passages = self.retrieve(question, top_k=top_k)
                retrieval_time = time.time() - start_time
                context = build_ctx_from_chunks(passages, limit=len(passages))

            generation_start = time.time()
            llm_response = self.llm_generator.generate_answer(question, context, dataset_class)
            generation_time = time.time() - generation_start

            if dataset_class == 'hotpot':
                citations = resolve_hotpot_citations_from_passages(
                    answer=llm_response.answer,
                    raw_response=llm_response.raw_response,
                    evidence_citations=llm_response.evidence_citations,
                    passages=passages,
                )
            else:
                citations = llm_response.evidence_citations

            result = {
                'question': question,
                'router_analysis': baseline_router_stub('BM25'),
                'retrieval': {
                    'bm25_results': passages,
                    'vector_results': passages,
                    'retrieval_time': retrieval_time,
                },
                'generation': {
                    'answer': llm_response.answer,
                    'citation_llm_use': citations,
                    'citation_refs': explicit_clause_citation_refs(llm_response.raw_response, passages),
                    'generation_time': generation_time,
                    'raw_response': llm_response.raw_response
                },
                'performance': {
                    'total_time': retrieval_time + generation_time,
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_class': dataset_class
            }
            from benchmark_log_utils import append_run_jsonl, resolve_run_jsonl
            append_run_jsonl(resolve_run_jsonl(
                self,
                getattr(self, "_run_method_name", "BM25"),
                getattr(self, "_output_subdir", ""),
            ), result)
            return result
        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return {'error': str(e), 'question': question, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'dataset_class': dataset_class}

    def close(self):
        """Release resources after a benchmark run."""
        pass
