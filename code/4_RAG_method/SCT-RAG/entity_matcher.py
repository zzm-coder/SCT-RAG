"""实体匹配器 - 使用实体ID到索引的映射"""
import sys
from pathlib import Path

_code_root = Path(__file__).resolve().parent.parent.parent
if str(_code_root) not in sys.path:
    sys.path.insert(0, str(_code_root))

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from api_embedding import APIEmbeddingModel

logger = logging.getLogger(__name__)

class EntityMatcher:
    """实体匹配器，使用实体ID到索引的映射"""
    
    def __init__(self, config, dataset_class: str = None):
        """初始化实体匹配器"""
        self.config = config
        # 预初始化属性，避免在非 ht 路径中缺少属性
        self.cache_path = None
        self.entity_cache = {}
        self.id_to_index = {}
        self.name_to_ids = {}
        self.entity_embeddings = None
        self.embedding_model = None

        if dataset_class == "ht":
            self.cache_path = Path(config.kg_cache_path)
            # 加载语义模型
            logger.info(f"加载语义模型: {config.semantic_model_path}")
            self.embedding_model = APIEmbeddingModel(config.semantic_model_path)
            # 加载实体缓存和ID映射
            self.entity_cache = self._load_entity_cache()
            self.id_to_index = self._load_id_to_index()
            
            # 建立实体名称到ID的映射
            self.name_to_ids = self._build_name_to_ids_mapping()
            
            # 加载实体嵌入
            self.entity_embeddings = self._load_entity_embeddings()
            
            logger.info(f"实体匹配器初始化完成，实体数量: {len(self.entity_cache)}")
        else:
            self.cache_path = Path(config.hotpot_kg_cache_path)
            # 加载语义模型
            logger.info(f"加载语义模型: {config.semantic_model_path2}")
            self.embedding_model = APIEmbeddingModel(config.semantic_model_path2)
            # 加载实体缓存和ID映射
            self.entity_cache = self._load_entity_cache()
            self.id_to_index = self._load_id_to_index()
            
            # 建立实体名称到ID的映射
            self.name_to_ids = self._build_name_to_ids_mapping()
            
            # 加载实体嵌入
            self.entity_embeddings = self._load_entity_embeddings()
            
            logger.info(f"实体匹配器初始化完成，实体数量: {len(self.entity_cache)}")


        # 保证所有属性存在（即使为空），避免上层代码访问时报错
        if self.entity_cache is None:
            self.entity_cache = {}
        if self.id_to_index is None:
            self.id_to_index = {}
        if self.name_to_ids is None:
            self.name_to_ids = {}
        if self.entity_embeddings is None:
            self.entity_embeddings = None
        
        
    def _load_entity_cache(self) -> Dict[str, Dict]:
        """加载实体缓存"""
        cache_file = self.cache_path / "entity_cache.json"
        if not cache_file.exists():
            logger.error(f"实体缓存文件不存在: {cache_file}")
            return {}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载实体缓存失败: {e}")
            return {}
    
    def _load_id_to_index(self) -> Dict[str, int]:
        """加载实体ID到索引的映射"""
        mapping_file = self.cache_path / "entity_id_to_index.json"
        if not mapping_file.exists():
            logger.warning(f"实体ID到索引映射文件不存在: {mapping_file}")
            return {}
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载实体ID到索引映射失败: {e}")
            return {}
    
    def _build_name_to_ids_mapping(self) -> Dict[str, List[str]]:
        """构建实体名称到ID列表的映射"""
        name_to_ids = {}
        
        for entity_id, entity_data in self.entity_cache.items():
            entity_name = entity_data.get("name", "")
            if not entity_name:
                continue
            
            # 清理实体名称
            entity_name = entity_name.strip()
            
            # 添加到映射
            if entity_name not in name_to_ids:
                name_to_ids[entity_name] = []
            name_to_ids[entity_name].append(entity_id)
            
            # 也考虑将标准化的名称添加进来
            normalized_name = self._normalize_entity_name(entity_name)
            if normalized_name != entity_name and normalized_name:
                if normalized_name not in name_to_ids:
                    name_to_ids[normalized_name] = []
                name_to_ids[normalized_name].append(entity_id)
        
        return name_to_ids
    
    def _normalize_entity_name(self, name: str) -> str:
        """标准化实体名称"""
        # 去除多余空格
        name = re.sub(r'\s+', ' ', name.strip())
        
        # 标准化标准编号格式
        patterns = [
            (r'HB\s*(\d+)\s*[-—]\s*(\d{4})', r'HB \1-\2'),
            (r'GB/T\s*(\d+)\s*[-—]\s*(\d{4})', r'GB/T \1-\2'),
            (r'GJB\s*(\d+)\s*[-—]\s*(\d{4})', r'GJB \1-\2')
        ]
        
        for pattern, replacement in patterns:
            name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
        
        return name
    
    def _load_entity_embeddings(self) -> np.ndarray:
        """加载实体嵌入矩阵"""
        embeddings_path = self.cache_path / "entity_embeddings.npy"
        if not embeddings_path.exists():
            logger.warning(f"实体嵌入文件不存在: {embeddings_path}")
            return None
        
        try:
            embeddings = np.load(str(embeddings_path))
            logger.info(f"加载实体嵌入矩阵成功，形状: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"加载实体嵌入矩阵失败: {e}")
            return None
    
    def get_entity_embedding_by_id(self, entity_id: str) -> np.ndarray:
        """根据实体ID获取嵌入向量"""
        if self.entity_embeddings is None or not self.id_to_index:
            return None
        
        if entity_id in self.id_to_index:
            index = self.id_to_index[entity_id]
            if index < len(self.entity_embeddings):
                return self.entity_embeddings[index]
        
        return None
    
    def match_entities(self, question: str, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        匹配问题中的实体
        
        Args:
            question: 问题文本
            keywords: 提取的关键词列表
            
        Returns:
            匹配的实体列表，包含实体信息、ID和相似度分数
        """
        if self.entity_embeddings is None:
            logger.warning("实体嵌入未加载，无法进行匹配")
            return []
        
        try:
            # 计算问题嵌入
            question_embedding = self.embedding_model.encode([question])[0]
            
            matched_entities = []
            
            # 1. 首先匹配关键词对应的实体
            keyword_matches = self._match_by_keywords(keywords, question_embedding)
            matched_entities.extend(keyword_matches)
            
            # 2. 匹配所有实体（如果关键词匹配不够）
            if len(matched_entities) < 5:
                all_matches = self._match_all_entities(question_embedding)
                matched_entities.extend(all_matches)
            
            # 3. 去重并排序
            unique_entities = self._deduplicate_and_sort(matched_entities)
            
            logger.info(f"实体匹配完成: 问题='{question[:50]}...'，匹配到 {len(unique_entities)} 个实体")
            
            return unique_entities[:20]  # 返回前20个
            
        except Exception as e:
            logger.error(f"实体匹配失败: {e}")
            return []
    
    def _match_by_keywords(self, keywords: List[str], question_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """通过关键词匹配实体"""
        matches = []
        
        if not keywords:
            return matches
        
        for keyword in keywords:
            # 查找包含关键词的实体
            for entity_name, entity_ids in self.name_to_ids.items():
                if keyword.lower() in entity_name.lower():
                    for entity_id in entity_ids:
                        entity_embedding = self.get_entity_embedding_by_id(entity_id)
                        if entity_embedding is not None:
                            similarity = self._calculate_similarity(question_embedding, entity_embedding)
                            if similarity > 0.5:
                                entity_data = self.entity_cache.get(entity_id, {})
                                matches.append({
                                    "id": entity_id,
                                    "name": entity_name,
                                    "similarity": similarity,
                                    "data": entity_data,
                                    "match_type": "keyword"
                                })
        
        return matches
    
    def _match_all_entities(self, question_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """匹配所有实体"""
        matches = []
        
        if self.entity_embeddings is None or not self.id_to_index:
            return matches
        
        # 批量计算相似度
        similarities = np.dot(self.entity_embeddings, question_embedding)
        
        # 找到相似度最高的实体
        top_indices = np.argsort(similarities)[-50:][::-1]  # 取前50个
        
        # 构建反向索引映射
        index_to_id = {v: k for k, v in self.id_to_index.items()}
        
        for idx in top_indices:
            if idx in index_to_id:
                entity_id = index_to_id[idx]
                entity_data = self.entity_cache.get(entity_id, {})
                entity_name = entity_data.get("name", "")
                similarity = float(similarities[idx])
                
                if similarity > 0.5 and entity_name:
                    matches.append({
                        "id": entity_id,
                        "name": entity_name,
                        "similarity": similarity,
                        "data": entity_data,
                        "match_type": "semantic"
                    })
        
        return matches
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _deduplicate_and_sort(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重并排序实体"""
        # 按ID去重
        seen_ids = set()
        unique_entities = []
        
        for entity in entities:
            if entity["id"] not in seen_ids:
                seen_ids.add(entity["id"])
                unique_entities.append(entity)
        
        # 按相似度排序
        unique_entities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return unique_entities
    
    def get_entity_by_id(self, entity_id: str) -> Dict[str, Any]:
        """根据实体ID获取实体信息"""
        if entity_id in self.entity_cache:
            entity_data = self.entity_cache[entity_id].copy()
            entity_data["entity_id"] = entity_id
            return entity_data
        return {}
    
    def get_entities_by_name(self, entity_name: str) -> List[Dict[str, Any]]:
        """根据实体名称获取实体信息列表"""
        entities = []
        
        if entity_name in self.name_to_ids:
            for entity_id in self.name_to_ids[entity_name]:
                entity_data = self.get_entity_by_id(entity_id)
                if entity_data:
                    entities.append(entity_data)
        
        return entities
