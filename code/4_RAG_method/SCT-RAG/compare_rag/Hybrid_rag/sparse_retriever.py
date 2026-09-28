from typing import List, Dict
import os
import re
from rank_bm25 import BM25Okapi
from data_types import SystemConfig
from compare_rag.utils import tokenize_zh, load_clause_corpus

class SparseRetriever:
    """基于 BM25 的稀疏检索器（段落级）。

    使用 `config.doc_dir` 读取 md 文档，按空行拆为段落并建立 BM25 索引。
    返回格式为列表，元素为 dict 包含 `chunk_id`,`filename`/`source`,`chunk_text`,`similarity_score`。
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.corpus_meta = []  # list of {'id', 'filename', 'text'}
        self.corpus_texts = []
        self._build_corpus()
        if self.corpus_texts:
            tokenized = [tokenize_zh(t) for t in self.corpus_texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def _build_corpus(self):
        clause_rows = load_clause_corpus(getattr(self.config, "vector_db_path1", ""))
        if clause_rows:
            self.corpus_meta = clause_rows
            self.corpus_texts = [row["text"] for row in clause_rows]
            return
        doc_dir = getattr(self.config, 'doc_dir', None)
        if not doc_dir or not os.path.isdir(doc_dir):
            return
        idc = 0
        for root, dirs, files in os.walk(doc_dir):
            for fn in files:
                if not fn.endswith('.md'):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = f.read()
                except Exception:
                    continue
                parts = [p.strip() for p in re.split(r'\n\s*\n+', raw) if p.strip()]
                for p in parts:
                    cid = f'p{idc}'
                    self.corpus_meta.append({'id': cid, 'filename': fn, 'text': p})
                    self.corpus_texts.append(p)
                    idc += 1

    def retrieve(self, question: str, top_k: int = None, dataset_class: str = None) -> List[Dict]:
        if not self.bm25:
            return []

        if top_k is None:
            top_k = getattr(self.config, 'top_k_vector', 20)

        q_tokens = tokenize_zh(question)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, score in ranked:
            if score <= 0:
                continue
            meta = self.corpus_meta[idx]
            out.append({
                'chunk_id': meta['id'],
                'source': meta['filename'],
                'chunk_text': meta['text'],
                'metadata': dict(meta.get('metadata') or {}),
                'similarity_score': float(score)
            })
        return out
