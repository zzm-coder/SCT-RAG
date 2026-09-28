from typing import List, Dict
import os
import re
import numpy as np
from data_types import SystemConfig
from api_embedding import APIEmbeddingModel

class DenseRetriever:
    """简单的稠密检索器：使用 sentence-transformers 生成段落向量并做余弦相似度检索。

    如果未安装 sentence-transformers，则会返回空列表（Hybrid 层会降级处理）。
    返回格式与 SparseRetriever 一致。
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.corpus_meta = []
        self.corpus_texts = []
        # 支持按 dataset_class 缓存多个模型与嵌入
        self.model_cache = {}
        self.embeddings_cache = {}
        self._build_corpus()
        # 不在初始化中强制加载模型/嵌入，按需按 dataset_class 加载并缓存

    def _build_corpus(self):
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


    def _load_model_and_embeddings_for(self, dataset_class: str):
        """按 dataset_class 加载（或返回缓存的）模型与嵌入。

        支持三种配置方式：
        - config.semantic_model_map: dict, key 为 dataset_class, value 为模型路径
        - config.semantic_model_path / semantic_model_path2: 作为默认/备用
        - embeddings are obtained from the configured OpenAI-compatible API
        """
        model_map = getattr(self.config, 'semantic_model_map', None)
        if model_map and isinstance(model_map, dict):
            model_path = model_map.get(dataset_class) or model_map.get('default')
        else:
            # 与 vector_retriever 保持一致的选择逻辑：dataset_class == 'ht' 使用 semantic_model_path，否则使用 semantic_model_path2
            if dataset_class and dataset_class == 'ht' and getattr(self.config, 'semantic_model_path', None):
                model_path = getattr(self.config, 'semantic_model_path')
            else:
                model_path = getattr(self.config, 'semantic_model_path2', None)

        # 根据 dataset_class 选择向量缓存路径（vector_db_path1 / vector_db_path2）
        if dataset_class and dataset_class == 'ht':
            vector_db_path = getattr(self.config, 'vector_db_path1', None)
        else:
            vector_db_path = getattr(self.config, 'vector_db_path2', None)

        # 如果没有模型路径或 sentence-transformers 未安装，则无法进行稠密检索
        if not model_path:
            return None, None

        # 使用组合 key 保证模型路径 + 向量库路径 对应唯一缓存项
        cache_key = f"{model_path}||{vector_db_path or ''}"
        if cache_key in self.model_cache and cache_key in self.embeddings_cache:
            return self.model_cache[cache_key], self.embeddings_cache[cache_key]

        # 1) 优先尝试从 vector_db_path 中加载预先计算好的 embeddings.npy 与 chunks.json
        if vector_db_path and os.path.isdir(vector_db_path):
            try:
                emb_path = os.path.join(vector_db_path, 'embeddings.npy')
                chunks_path = os.path.join(vector_db_path, 'chunks.json')
                if os.path.exists(emb_path):
                    emb = np.load(emb_path)
                    # 尝试从 chunks.json 恢复 corpus_meta / corpus_texts
                    meta_list = []
                    texts = []
                    if os.path.exists(chunks_path):
                        import json
                        try:
                            with open(chunks_path, 'r', encoding='utf-8') as f:
                                chunks_data = json.load(f)
                            if isinstance(chunks_data, list):
                                for i, c in enumerate(chunks_data):
                                    cid = c.get('id', f'p{i}')
                                    filename = c.get('metadata', {}).get('file_name') or c.get('metadata', {}).get('filename') or c.get('source') or c.get('file_name') or f'chunk_{i}'
                                    text = c.get('original_text') or c.get('text') or c.get('content') or ''
                                    meta_list.append({'id': cid, 'filename': filename, 'text': text,
                                                      'metadata': dict(c.get('metadata') or {})})
                                    texts.append(text)
                            elif isinstance(chunks_data, dict):
                                for k, c in chunks_data.items():
                                    cid = c.get('id', k)
                                    filename = c.get('metadata', {}).get('file_name') or c.get('metadata', {}).get('filename') or c.get('source') or c.get('file_name') or str(k)
                                    text = c.get('original_text') or c.get('text') or c.get('content') or ''
                                    meta_list.append({'id': cid, 'filename': filename, 'text': text,
                                                      'metadata': dict(c.get('metadata') or {})})
                                    texts.append(text)
                        except Exception:
                            meta_list = []
                            texts = []

                    # 如果加载到了 embeddings，但没有 texts，我们仍可继续，但需要保证长度匹配
                    if emb is not None:
                        # 尝试加载模型（必须能编码 query）
                        try:
                            model = APIEmbeddingModel(model_path) if model_path else None
                        except Exception:
                            model = None

                        # 若模型可用且 embeddings 与 texts 长度匹配（或 texts 已空），则采用缓存
                        if model is not None:
                            # 如果 meta_list 为空但 self.corpus_texts 有数据，则尝试使用已有文本作为 meta
                            if not texts and self.corpus_texts:
                                meta_list = [{'id': f'p{i}', 'filename': (self.corpus_meta[i]['filename'] if i < len(self.corpus_meta) and isinstance(self.corpus_meta[i], dict) and 'filename' in self.corpus_meta[i] else 'unknown'), 'text': t} for i, t in enumerate(self.corpus_texts)]
                                texts = list(self.corpus_texts)

                            # 若长度不匹配且 texts 可用，尝试截断或放弃
                            if texts and emb.shape[0] != len(texts):
                                if emb.shape[0] >= len(texts):
                                    emb = emb[:len(texts)]
                                else:
                                    emb = None

                        if emb is not None and model is not None:
                            # 保存到缓存并更新 corpus meta/text
                            self.model_cache[cache_key] = model
                            self.embeddings_cache[cache_key] = emb
                            if meta_list:
                                self.corpus_meta = meta_list
                                self.corpus_texts = texts
                            return model, emb
            except Exception:
                # 忽略加载缓存的错误，回退到按模型生成嵌入
                pass

        # 2) 回退：按模型路径加载模型并对 corpus_texts 计算嵌入
        try:
            model = APIEmbeddingModel(model_path)
            embeddings = model.encode(self.corpus_texts, convert_to_numpy=True, show_progress_bar=False)
            self.model_cache[cache_key] = model
            self.embeddings_cache[cache_key] = embeddings
            return model, embeddings
        except Exception:
            return None, None

    def retrieve(self, question: str, top_k: int = None, dataset_class: str = None) -> List[Dict]:
        """按 dataset_class 检索。若 dataset_class 为 None 则使用默认模型。"""
        if top_k is None:
            top_k = getattr(self.config, 'top_k_vector', 20)

        model, embeddings = self._load_model_and_embeddings_for(dataset_class or 'default')
        if model is None or embeddings is None:
            return []

        q_emb = model.encode([question], convert_to_numpy=True)[0]
        # cosine similarity
        sims = (embeddings @ q_emb) / (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-12))
        ranked = np.argsort(-sims)[:top_k]
        out = []
        for idx in ranked:
            meta = self.corpus_meta[int(idx)]
            out.append({
                'chunk_id': meta['id'],
                'source': meta['filename'],
                'chunk_text': meta['text'],
                'metadata': dict(meta.get('metadata') or {}),
                'similarity_score': float(sims[int(idx)])
            })
        return out
