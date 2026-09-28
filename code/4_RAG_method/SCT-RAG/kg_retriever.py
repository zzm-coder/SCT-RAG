"""知识图谱检索器 - 使用增强的实体匹配"""
import json
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
import os
try:
    import networkx as nx
except Exception:
    nx = None

from data_types import SystemConfig, KGResult, KGTriple, QuestionType
from entity_matcher import EntityMatcher
from compare_rag.utils import canonical_standard_id, tokenize_zh

logger = logging.getLogger(__name__)

class KnowledgeGraphRetriever:
    """增强版知识图谱检索器"""
    
    def __init__(self, config: SystemConfig, dataset_class: str = None):
        self.config = config
        self.dataset_class = dataset_class
        self.cache_path = None
        # hotpot 内存图谱存储
        self.hotpot_data: Dict[str, Any] = {
            "text_units": {},
            "entities": {},
            "relations": {},
            "communities": [],
            "graph": None
        }
        
        # 初始化Neo4j连接
        try:
            if dataset_class == "ht" and not str(config.neo4j_password or "").strip():
                self.driver = None
                logger.info("Neo4j 未配置凭据，工业集使用本地条款 KG 缓存")
            elif dataset_class == "ht" and GraphDatabase is not None:
                self.driver = GraphDatabase.driver(
                    config.neo4j_uri,
                    auth=(config.neo4j_user, config.neo4j_password)
                )
            elif GraphDatabase is not None:
                self.driver = GraphDatabase.driver(
                    config.hotpot_neo4j_uri,
                    auth=(config.hotpot_neo4j_user, config.hotpot_neo4j_password)
                )
            else:
                self.driver = None
        except Exception as e:
            logger.warning(f'无法初始化 Neo4j 驱动，稍后将回退到本地 KG JSON 或缓存: {e}')
            self.driver = None
        
        # 初始化实体匹配器
        self.entity_matcher = EntityMatcher(config, dataset_class=dataset_class)
        
        # 加载三联缓存
        if dataset_class == "ht":
            configured_triplets = getattr(config, "kg_triplet_cache_path", "") or config.kg_cache_path
            self.cache_path = Path(configured_triplets)
            self.triplet_cache = self._load_triplet_cache()
        else:
            self.cache_path = Path(config.hotpot_kg_cache_path)
            self.triplet_cache = self._load_triplet_cache()
        
        logger.info("知识图谱检索器初始化完成")

    def _first_text_unit(self, text_unit_ids):
        """从可能是列表的 text_unit_ids 中提取第一个元素并返回字符串（无值返回空字符串）"""
        if isinstance(text_unit_ids, (list, tuple)):
            if text_unit_ids:
                return str(text_unit_ids[0])
            return ""
        if text_unit_ids:
            return str(text_unit_ids)
        return ""
    
    def _load_triplet_cache(self) -> Dict[str, List[Dict]]:
        """加载三联缓存"""
        cache_file = self.cache_path / "triplet_clause_cache.json"
        if not cache_file.exists():
            cache_file = self.cache_path / "triplet_cache.json"
        if not cache_file.exists():
            logger.error(f"三联缓存文件不存在: {cache_file}")
            return {}
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载三联缓存失败: {e}")
            return {}
    
    def query_kg(
        self,
        question: str,
        entities: List[str],
        question_type: QuestionType,
        dataset_class: str = None,
        local_chunks: List[dict] = None,
    ) -> KGResult:
        """
        查询知识图谱。
        HotPot：若提供 local_chunks，走本题局部子图（避免全库语义乱匹配）。
        """
        start_time = time.time()
        try:
            dc = dataset_class or self.dataset_class
            local = local_chunks if local_chunks is not None else getattr(self, "_hotpot_local_chunks", None)
            if dc != "ht" and local:
                from hotpot_kg_local import query_hotpot_local_kg

                return query_hotpot_local_kg(self, question, entities, question_type, local)

            # Map the entities predicted by the router to graph nodes.
            matched_entities = self.entity_matcher.match_entities(
                question, list(entities or [])
            )

            # 3. 从缓存/JSON中获取三联
            if dc == "ht":
                cache_triples = self._get_triples_from_cache(
                    matched_entities, question_type, question=question
                )
            else:
                cache_triples = self._get_triples_from_hotpot_json(matched_entities, question_type)

            # 4. Neo4j：仅工业集补充；HotPot 无局部段时也不走全局 Neo4j（噪声大）
            if dc == "ht":
                neo4j_triples = self._query_triples_from_neo4j(matched_entities, question_type)
            else:
                neo4j_triples = []

            all_triples = cache_triples + neo4j_triples
            unique_triples = self._deduplicate_triples(all_triples)

            all_kg_entities = set()
            for triple in unique_triples:
                all_kg_entities.add(triple.head)
                all_kg_entities.add(triple.tail)
            for entity_info in matched_entities:
                all_kg_entities.add(entity_info["name"])

            result = KGResult(
                triples=unique_triples,
                entities=list(all_kg_entities),
                query_time=time.time() - start_time,
            )
            logger.info(
                f"KG查询完成: 问题='{question[:50]}...'，找到 {len(unique_triples)} 个三元组，{len(all_kg_entities)} 个实体"
            )
            return result
        except Exception as e:
            logger.error(f"KG查询失败: {e}")
            return KGResult([], [], time.time() - start_time)

    def query_graphrag(
        self,
        question: str,
        entities: List[str],
        question_type: QuestionType,
    ) -> Tuple[KGResult, List[str]]:
        """GraphRAG 论文查询：Local Search（实体邻域+原文）与 Global（标准社区摘要）。

        工业集没有 Leiden 社区报告，用「同一标准号」作为社区；多标准题按社区 map 再合并。
        """
        start_time = time.time()
        from standard_boost import extract_explicit_std_codes, source_matches_std

        matched_entities = self.entity_matcher.match_entities(
            question, list(entities or [])
        )
        entity_names = [e.get("name", "") for e in matched_entities[:12] if e.get("name")]
        std_keys = extract_explicit_std_codes(question) or []
        query_tokens = set(tokenize_zh(question))
        wants_def = bool(re.search(r"何谓|是什么|定义|指什么", question or ""))
        wants_formula = bool(re.search(r"公式|式中|附录|计算|η", question or ""))

        def _std_ok(td, source_file: str) -> bool:
            if not std_keys:
                return True
            triple_standard = td.get("standard_id") or source_file
            return source_matches_std(str(triple_standard), std_keys) or source_matches_std(
                str(source_file), std_keys
            )

        def _score_triple(td, source_file: str) -> float:
            head = (td.get("head") or {}).get("name", "")
            tail = (td.get("tail") or {}).get("name", "")
            para = str(td.get("paragraph") or "")
            blob = f"{head} {tail} {para}"
            blob_toks = set(tokenize_zh(blob))
            lexical = (
                len(query_tokens & blob_toks) / max(1, len(query_tokens))
            )
            ent = 0.0
            for i, name in enumerate(entity_names):
                if name and (name in head or name in tail or name in para):
                    ent = max(ent, 1.0 - 0.05 * i)
            cid = str(td.get("clause_id") or "")
            bonus = 0.0
            if wants_def and re.match(r"^3(\.|$)", cid):
                bonus += 0.35
            if wants_formula and re.search(r"C\.|附录|公式|\\\\frac|η", para + cid, re.I):
                bonus += 0.45
            return 2.2 * lexical + 0.8 * ent + bonus

        # Local：先收标准社区内（或实体命中）的三元组
        scored: List[Tuple[float, Any, str, Dict]] = []
        for source_file, triple_list in (self.triplet_cache or {}).items():
            for td in triple_list:
                if not isinstance(td, dict):
                    continue
                if not _std_ok(td, source_file):
                    continue
                head = (td.get("head") or {}).get("name", "")
                tail = (td.get("tail") or {}).get("name", "")
                para = str(td.get("paragraph") or "")
                ent_hit = any(
                    n and (n in head or n in tail or n in para) for n in entity_names
                )
                lex_hit = bool(query_tokens & set(tokenize_zh(para + head + tail)))
                cid = str(td.get("clause_id") or "")
                topical = bool(
                    (wants_def and re.match(r"^3(\.|$)", cid))
                    or (
                        wants_formula
                        and re.search(r"C\.|附录|公式|\\\\frac|η", para + cid, re.I)
                    )
                )
                if std_keys:
                    # 论文 local：社区内与实体/问点/定义公式相关的 text unit
                    if not (ent_hit or lex_hit or topical):
                        continue
                elif not (ent_hit or lex_hit):
                    continue
                sc = _score_triple(td, source_file)
                scored.append((sc, source_file, td))

        scored.sort(key=lambda x: x[0], reverse=True)
        hop1_nodes = set()
        for sc, source_file, td in scored[:40]:
            hop1_nodes.add((td.get("head") or {}).get("name", ""))
            hop1_nodes.add((td.get("tail") or {}).get("name", ""))
        hop1_nodes.discard("")

        # 2-hop：沿已命中实体扩一跳（论文 local neighborhood）
        seen = {
            (
                (td.get("head") or {}).get("name", ""),
                td.get("relation", ""),
                (td.get("tail") or {}).get("name", ""),
            )
            for _, _, td in scored[:40]
        }
        extra = []
        if hop1_nodes:
            for source_file, triple_list in (self.triplet_cache or {}).items():
                for td in triple_list:
                    if not isinstance(td, dict) or not _std_ok(td, source_file):
                        continue
                    head = (td.get("head") or {}).get("name", "")
                    tail = (td.get("tail") or {}).get("name", "")
                    key = (head, td.get("relation", ""), tail)
                    if key in seen:
                        continue
                    if head not in hop1_nodes and tail not in hop1_nodes:
                        continue
                    para = str(td.get("paragraph") or "")
                    if query_tokens and not (query_tokens & set(tokenize_zh(para + head + tail))):
                        continue
                    seen.add(key)
                    extra.append((_score_triple(td, source_file) + 0.15, source_file, td))
                    if len(extra) >= 30:
                        break
                if len(extra) >= 30:
                    break
        scored = scored[:50] + extra
        scored.sort(key=lambda x: x[0], reverse=True)

        triples: List[KGTriple] = []
        used_eid = set()
        for sc, source_file, td in scored:
            eid = str(td.get("evidence_id") or "").strip()
            key = eid or (
                f"{td.get('standard_id')}::{td.get('clause_id')}::{(td.get('head') or {}).get('name')}"
            )
            if key in used_eid:
                continue
            used_eid.add(key)
            triples.append(
                KGTriple(
                    head=(td.get("head") or {}).get("name", ""),
                    relation=td.get("relation", ""),
                    tail=(td.get("tail") or {}).get("name", ""),
                    source=source_file,
                    paragraph=td.get("paragraph", "") or "",
                    confidence=float(td.get("confidence") or sc),
                    standard_id=td.get("standard_id") or "",
                    clause_id=td.get("clause_id") or "",
                    evidence_id=eid,
                )
            )
            if len(triples) >= 40:
                break

        # Global：按标准社区汇总关系，供 map-reduce 式阅读
        reports: List[str] = []
        grouped: Dict[str, List[KGTriple]] = {}
        for t in triples:
            grouped.setdefault(t.standard_id or t.source or "unknown", []).append(t)
        for std, items in grouped.items():
            rels = []
            clauses = []
            for t in items[:8]:
                rels.append(f"{t.head} -{t.relation}-> {t.tail}")
                if t.clause_id:
                    clauses.append(str(t.clause_id))
            reports.append(
                f"社区 {std}：条款 {', '.join(dict.fromkeys(clauses) or ['-'])}；"
                f"关系：" + "；".join(rels[:6])
            )

        ents = list({t.head for t in triples} | {t.tail for t in triples} | set(entity_names))
        result = KGResult(triples=triples, entities=ents, query_time=time.time() - start_time)
        logger.info(
            "GraphRAG 论文检索: triples=%d communities=%d stds=%s",
            len(triples),
            len(reports),
            ",".join(std_keys) or "-",
        )
        return result, reports
    
    def _get_triples_from_cache(
        self,
        matched_entities: List[Dict],
        question_type: QuestionType,
        question: str = "",
    ) -> List[KGTriple]:
        """Rank cached triples before mapping them to canonical clauses."""
        scored_triples = []
        
        if not self.triplet_cache:
            return []
        
        entity_names = [entity["name"] for entity in matched_entities[:10]]
        entity_ids = [entity["id"] for entity in matched_entities[:10]]
        mentioned_standards = self._mentioned_standard_ids(question)
        from standard_boost import extract_explicit_std_codes, source_matches_std
        mentioned_std_keys = extract_explicit_std_codes(question)
        query_tokens = set(tokenize_zh(question))
        
        # 遍历所有三联缓存
        for source_file, triple_list in self.triplet_cache.items():
            for triple_data in triple_list:
                head_data = triple_data.get("head", {})
                tail_data = triple_data.get("tail", {})
                
                head_name = head_data.get("name", "")
                tail_name = tail_data.get("name", "")
                head_id = head_data.get("node_id", "")
                tail_id = tail_data.get("node_id", "")
                
                relation = triple_data.get("relation", "")
                paragraph = triple_data.get("paragraph", "")
                confidence = triple_data.get("confidence", 1.0)
                
                triple_standard = triple_data.get("standard_id") or source_file
                source_anchored = canonical_standard_id(str(triple_standard)) in mentioned_standards
                if mentioned_std_keys and not (
                    source_matches_std(str(triple_standard), mentioned_std_keys)
                    or source_matches_std(str(source_file), mentioned_std_keys)
                ):
                    # 题干已写明标准号：丢掉关键词命中的其它标准三元组
                    continue

                # 检查是否与任何实体匹配
                entity_match = False
                matched_rank = len(entity_names) + 1
                
                # 检查名称匹配
                for rank, entity_name in enumerate(entity_names):
                    if entity_name in head_name or entity_name in tail_name:
                        entity_match = True
                        matched_rank = rank
                        break
                
                # 检查ID匹配
                if not entity_match:
                    for entity_id in entity_ids:
                        if entity_id == head_id or entity_id == tail_id:
                            entity_match = True
                            break
                
                # 检查关系是否与问题相关
                relation_relevant = self._is_relation_relevant(relation, question_type)
                
                # 显式标准号是工业问题的强约束：即使实体嵌入没有
                # 命中专业术语，也应允许该标准内的条款参与排序。
                # Correlation traversal remains within the parsed standard scope.
                if question_type == QuestionType.CORRELATION and mentioned_std_keys:
                    relation_relevant = True
                if (
                    question_type == QuestionType.CORRELATION
                    and source_anchored
                    and not entity_match
                ):
                    preview = f"{head_name} {tail_name} {paragraph}"
                    if not (query_tokens & set(tokenize_zh(preview))):
                        continue
                if (entity_match or source_anchored) and relation_relevant:
                    triple = KGTriple(
                        head=head_name,
                        relation=relation,
                        tail=tail_name,
                        source=source_file,
                        paragraph=paragraph,
                        confidence=confidence,
                        standard_id=triple_data.get("standard_id") or "",
                        clause_id=triple_data.get("clause_id") or "",
                        evidence_id=triple_data.get("evidence_id") or "",
                    )
                    content_tokens = set(tokenize_zh(
                        f"{head_name} {relation} {tail_name} {paragraph}"
                    ))
                    lexical = (
                        len(query_tokens & content_tokens) / len(query_tokens)
                        if query_tokens else 0.0
                    )
                    entity_rank_score = (
                        1.0 - matched_rank / max(1, len(entity_names))
                        if entity_match else 0.0
                    )
                    score = (
                        3.0 * float(source_anchored)
                        + 1.5 * lexical
                        + 0.5 * entity_rank_score
                        + 0.05 * float(confidence or 0.0)
                    )
                    scored_triples.append((score, triple))

        # 论文 correlation：JSON 缓存为 1-hop 列表，再沿已命中节点扩一跳，仍约束 E_std
        if question_type == QuestionType.CORRELATION and scored_triples and mentioned_std_keys:
            hop1 = {t.head for _, t in scored_triples} | {t.tail for _, t in scored_triples}
            seen_keys = {(t.head, t.relation, t.tail) for _, t in scored_triples}
            extra: List[Tuple[float, KGTriple]] = []
            for source_file, triple_list in self.triplet_cache.items():
                for triple_data in triple_list:
                    head_name = (triple_data.get("head") or {}).get("name", "")
                    tail_name = (triple_data.get("tail") or {}).get("name", "")
                    relation = triple_data.get("relation", "")
                    key = (head_name, relation, tail_name)
                    if key in seen_keys:
                        continue
                    if not (head_name in hop1 or tail_name in hop1):
                        continue
                    triple_standard = triple_data.get("standard_id") or source_file
                    if not (
                        source_matches_std(str(triple_standard), mentioned_std_keys)
                        or source_matches_std(str(source_file), mentioned_std_keys)
                    ):
                        continue
                    paragraph = triple_data.get("paragraph", "")
                    preview = f"{head_name} {tail_name} {paragraph}"
                    if query_tokens and not (query_tokens & set(tokenize_zh(preview))):
                        continue
                    extra.append(
                        (
                            2.0,
                            KGTriple(
                                head=head_name,
                                relation=relation,
                                tail=tail_name,
                                source=source_file,
                                paragraph=paragraph,
                                confidence=triple_data.get("confidence", 1.0),
                                standard_id=triple_data.get("standard_id") or "",
                                clause_id=triple_data.get("clause_id") or "",
                                evidence_id=triple_data.get("evidence_id") or "",
                            ),
                        )
                    )
                    seen_keys.add(key)
                    if len(extra) >= 20:
                        break
                if len(extra) >= 20:
                    break
            scored_triples.extend(extra)
        
        # 根据问题类型调整返回数量
        max_triples = {
            QuestionType.SINGLE_STANDARD: 10, # 单独10
            QuestionType.CROSS_STANDARD: 20, # 单独20
            QuestionType.CORRELATION: 15 # 单独15
        }.get(question_type, 15)
        
        scored_triples.sort(key=lambda item: item[0], reverse=True)
        return [triple for _, triple in scored_triples[:max_triples]]

    @staticmethod
    def _mentioned_standard_ids(question: str) -> Set[str]:
        """只从题干抽取显式标准号，与 StdDirect 共用同一套正则。"""
        from standard_boost import extract_explicit_std_codes

        ids = set()
        for value in extract_explicit_std_codes(question):
            key = canonical_standard_id(value) or value
            if key:
                ids.add(key)
        return ids
    

    def _query_triples_from_neo4j(self, matched_entities: List[Dict], question_type: QuestionType) -> List[KGTriple]:
        """从Neo4j查询三元组"""
        triples: List[KGTriple] = []

        if not matched_entities or not self.driver:
            return triples

        # 只取前几个匹配的实体来控制查询量
        entity_names = [entity.get("name", "") for entity in matched_entities[:10]]

        try:
            with self.driver.session() as session:
                for entity_name in entity_names:
                    # 根据问题类型选择查询深度/策略
                    if question_type == QuestionType.SINGLE_STANDARD:
                        query = """
                        MATCH (n)-[r]->(m)
                        WHERE n.name CONTAINS $entity
                        RETURN n.name as head, type(r) as relation, m.name as tail, properties(r) as rel_props
                        """
                    elif question_type == QuestionType.CROSS_STANDARD:
                        query = """
                        MATCH path = (n)-[r*2..4]->(m)
                        WHERE n.name CONTAINS $entity
                        UNWIND relationships(path) as rel
                        RETURN startNode(rel).name as head, type(rel) as relation, endNode(rel).name as tail, properties(rel) as rel_props
                        """
                    else:  # correlation: constrained one-to-two-hop traversal
                        query = """
                        MATCH path = (n)-[r*1..2]->(m)
                        WHERE n.name CONTAINS $entity
                        UNWIND relationships(path) as rel
                        RETURN startNode(rel).name as head, type(rel) as relation,
                               endNode(rel).name as tail, properties(rel) as rel_props
                        """

                    result = session.run(query, {"entity": entity_name})
                    temp_triples= []
                    for record in result:
                        try:
                            head = record.get("head") or ""
                            relation = record.get("relation") or ""
                            tail = record.get("tail") or ""
                            rel_props = record.get("rel_props", {}) or {}

                            paragraph = rel_props.get("paragraph", "") if rel_props else ""
                            source = rel_props.get("source", "") if rel_props else ""
                            confidence = float(rel_props.get("confidence", 0.9)) if rel_props.get("confidence") is not None else 0.9

                            # 检查是否与任何匹配实体相关（名称子串或完全匹配）
                            matched = False
                            for en in entity_names:
                                if en and (en in str(head) or en in str(tail) or en == str(head) or en == str(tail)):
                                    matched = True
                                    break

                            # 关系相关性过滤
                            relation_relevant = self._is_relation_relevant(relation, question_type)

                            if matched and relation_relevant:
                                temp_triples.append(KGTriple(
                                    head=str(head),
                                    relation=str(relation),
                                    tail=str(tail),
                                    source=str(source),
                                    paragraph=str(paragraph),
                                    confidence=confidence
                                ))
                        except Exception:
                            continue
                    triples.extend(temp_triples[:10])

        except Exception as e:
            logger.error(f"Neo4j查询失败: {e}")

        return triples

    def _get_triples_from_hotpot_json(self, matched_entities: List[Dict], question_type: QuestionType) -> List[KGTriple]:
        """从 HotPotQA 专用 JSON 加载并筛选三元组（优先使用已加载的 hotpot_data）"""
        triples: List[KGTriple] = []
        hotpot_path = Path(self.config.hotpot_kg_json_path)

        # 如果尚未加载 hotpot 数据，则尝试加载
        if not self.hotpot_data.get("entities") or not self.hotpot_data.get("relations"):
            self.load_from_json(str(hotpot_path))

        if not matched_entities:
            return triples

        entity_names = [e["name"] for e in matched_entities[:10]]

        # 优先使用 relations 列表来生成三元组
        relations = self.hotpot_data.get("relations", {})
        for key, rel in relations.items():
            try:
                src = rel.get("source")
                tgt = rel.get("target")
                rtype = rel.get("relation_type") or rel.get("relation") or rel.get("type", "")
                evidences = rel.get("evidences", [])
                paragraph = "; ".join(evidences) if evidences else rel.get("description", "")
                text_unit_ids = rel.get("text_unit_ids", [])
                # 仅取列表中的第一个元素，避免上游返回列表导致的不可哈希或格式不一致问题
                text_unit_id = self._first_text_unit(text_unit_ids)
                confidence = float(rel.get("confidence", 1.0)) if rel.get("confidence") is not None else 1.0

                matched = False
                for en in entity_names:
                    if en and (en in str(src) or en in str(tgt)):
                        matched = True
                        break

                if matched:
                    # 使用第一个 text_unit id 作为 source 标识（若不存在则回退到 source_file）
                    source_str = text_unit_id if text_unit_id else str(rel.get('source_file') or 'hotpot_json')

                    triples.append(KGTriple(
                        head=str(src),
                        relation=str(rtype),
                        tail=str(tgt),
                        source=source_str,
                        paragraph=str(paragraph),
                        confidence=confidence
                    ))
            except Exception:
                continue

        # 如果没有关系数据，回退到基于 text_units 的简单抽取
        if not triples:
            tus = self.hotpot_data.get("text_units", {})
            for tu_id, tu in tus.items():
                content = tu.get("content", "")
                for en in entity_names:
                    if en and en in content:
                        # 创建伪三元组：(en, appears_in, title)
                        triples.append(KGTriple(
                            head=en,
                            relation="appears_in",
                            tail=tu.get("title", tu_id),
                            source="hotpot_text_unit",
                            paragraph=content[:512],
                            confidence=0.8
                        ))

        # 返回数量限制（与缓存策略一致）
        max_triples = {
            QuestionType.SINGLE_STANDARD: 20,
            QuestionType.CROSS_STANDARD: 15,
            QuestionType.CORRELATION: 10
        }.get(question_type, 10)

        triples.sort(key=lambda x: x.confidence, reverse=True)
        return triples[:max_triples]

    def load_from_json(self, file_path: str):
        """按照用户提供的格式从 JSON 加载 hotpot 图谱到内存结构"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Knowledge graph file not found: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # reset
            self.hotpot_data["text_units"].clear()
            self.hotpot_data["entities"].clear()
            self.hotpot_data["relations"].clear()
            self.hotpot_data["communities"].clear()
            if nx:
                self.hotpot_data["graph"] = nx.Graph()
            else:
                self.hotpot_data["graph"] = None

            for tu_data in data.get("text_units", []):
                tu = {
                    "id": tu_data["id"],
                    "title": tu_data.get("title", ""),
                    "content": tu_data.get("content", ""),
                    "entities": tu_data.get("entities", []),
                    "relations": [tuple(rel) for rel in tu_data.get("relations", [])]
                }
                self.hotpot_data["text_units"][tu["id"]] = tu

            for ent_data in data.get("entities", []):
                raw_ids = ent_data.get("text_unit_ids", [])
                # Preserve the complete identity list required by the HotPot subgraph.
                from hotpot_kg_local import parse_id_list

                id_list = parse_id_list(raw_ids)
                ent = {
                    "name": ent_data["name"],
                    "entity_type": ent_data.get("entity_type", ""),
                    "descriptions": ent_data.get("descriptions", []),
                    "text_unit_ids": id_list if id_list else self._first_text_unit(raw_ids),
                    "degree": ent_data.get("degree", 0),
                }
                self.hotpot_data["entities"][ent["name"]] = ent

            for rel_data in data.get("relations", []):
                from hotpot_kg_local import parse_id_list

                id_list = parse_id_list(rel_data.get("text_unit_ids", []))
                rel = {
                    "source": rel_data.get("source"),
                    "target": rel_data.get("target"),
                    "relation_type": rel_data.get("relation_type"),
                    "evidences": rel_data.get("evidences", []),
                    "text_unit_ids": id_list if id_list else self._first_text_unit(rel_data.get("text_unit_ids", [])),
                    "description": rel_data.get("description", ""),
                    "confidence": rel_data.get("confidence", 1.0),
                    "source_file": rel_data.get("source_file", "hotpot_json"),
                }
                key = (rel["source"], rel["relation_type"], rel["target"]) if rel["relation_type"] is not None else (rel["source"], rel["target"]) 
                self.hotpot_data["relations"][key] = rel

            for com_data in data.get("communities", []):
                com = {
                    "id": com_data.get("id"),
                    "level": com_data.get("level"),
                    "nodes": com_data.get("nodes", []),
                    "report": com_data.get("report", {}),
                    "importance_score": com_data.get("importance_score", 0.0)
                }
                self.hotpot_data["communities"].append(com)

            if "graph_adjacency" in data and nx:
                try:
                    self.hotpot_data["graph"] = nx.from_dict_of_dicts(data["graph_adjacency"])
                except Exception:
                    self.hotpot_data["graph"] = None

            logger.info(f"Knowledge graph loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge graph: {e}")
            return False

    def clear_all(self):
        if self.driver:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")

    def save_entity(self, entity: Dict):
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.degree = $degree,
                        e.text_unit_ids = $text_unit_ids
                    """,
                    id=f"entity_{entity.get('name')}",
                    name=entity.get('name'),
                    type=entity.get('entity_type'),
                    description='; '.join(entity.get('descriptions', [])) if entity.get('descriptions') else '',
                    degree=entity.get('degree', 0),
                    text_unit_ids=entity.get('text_unit_ids', []))
        else:
            # 存入内存
            self.hotpot_data['entities'][entity.get('name')] = entity

    def save_relation(self, relation: Dict):
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MATCH (s:Entity {id: $source_id})
                    MATCH (t:Entity {id: $target_id})
                    MERGE (s)-[r:RELATES_TO {type: $type}]->(t)
                    SET r.evidences = $evidences,
                        r.text_unit_ids = $text_unit_ids,
                        r.description = $description
                    """,
                    source_id=f"entity_{relation.get('source')}",
                    target_id=f"entity_{relation.get('target')}",
                    type=relation.get('relation_type'),
                    evidences=relation.get('evidences', []),
                    text_unit_ids=relation.get('text_unit_ids', []),
                    description=relation.get('description', ''))
        else:
            key = (relation.get('source'), relation.get('relation_type'), relation.get('target'))
            self.hotpot_data['relations'][key] = relation

    def save_text_unit(self, text_unit: Dict):
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MERGE (t:TextUnit {id: $id})
                    SET t.title = $title,
                        t.content = $content,
                        t.entities = $entities
                    """,
                    id=text_unit.get('id'),
                    title=text_unit.get('title'),
                    content=text_unit.get('content'),
                    entities=text_unit.get('entities', []))
        else:
            self.hotpot_data['text_units'][text_unit.get('id')] = text_unit

    def save_community(self, community: Dict):
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MERGE (c:Community {id: $id})
                    SET c.level = $level,
                        c.nodes = $nodes,
                        c.report_title = $report_title,
                        c.report_summary = $report_summary,
                        c.importance_score = $importance_score
                    """,
                    id=community.get('id'),
                    level=community.get('level'),
                    nodes=community.get('nodes', []),
                    report_title=community.get('report', {}).get('title', ''),
                    report_summary=community.get('report', {}).get('summary', ''),
                    importance_score=community.get('importance_score', 0.0))
        else:
            self.hotpot_data['communities'].append(community)

    def link_entity_to_text_unit(self, entity_name: str, text_unit_id: str):
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (t:TextUnit {id: $text_unit_id})
                    MERGE (e)-[:APPEARS_IN]->(t)
                    """,
                    entity_id=f"entity_{entity_name}",
                    text_unit_id=text_unit_id)
        else:
            tu = self.hotpot_data['text_units'].get(text_unit_id)
            if tu:
                tu.setdefault('entities', []).append(entity_name)

    def link_community_to_entities(self, community_id: str, entity_names: List[str]):
        if self.driver:
            with self.driver.session() as session:
                for entity_name in entity_names:
                    session.run("""
                        MATCH (c:Community {id: $community_id})
                        MATCH (e:Entity {id: $entity_id})
                        MERGE (c)-[:CONTAINS]->(e)
                        """,
                        community_id=community_id,
                        entity_id=f"entity_{entity_name}")
        else:
            for com in self.hotpot_data['communities']:
                if com.get('id') == community_id:
                    com.setdefault('nodes', []).extend(entity_names)

    def get_entities_by_names(self, entity_names: List[str]) -> List[Dict]:
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name IN $names
                    RETURN e.id as id, e.name as name, e.type as type,
                           e.description as description, e.degree as degree,
                           e.text_unit_ids as text_unit_ids
                    """, names=entity_names)
                return [dict(record) for record in result]
        else:
            out = []
            for name in entity_names:
                ent = self.hotpot_data['entities'].get(name)
                if ent:
                    out.append({
                        'id': f"entity_{name}",
                        'name': ent.get('name'),
                        'type': ent.get('entity_type'),
                        'description': '; '.join(ent.get('descriptions', [])) if ent.get('descriptions') else '',
                        'degree': ent.get('degree', 0),
                        'text_unit_ids': self._first_text_unit(ent.get('text_unit_ids', []))
                    })
            return out

    def get_relations_for_entities(self, entity_names: List[str]) -> List[Dict]:
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
                    WHERE e1.name IN $names OR e2.name IN $names
                    RETURN e1.name as source, e2.name as target,
                           r.type as relation_type, r.description as description,
                           r.text_unit_ids as text_unit_ids
                    """, names=entity_names)
                return [dict(record) for record in result]
        else:
            out = []
            for (s, rtype, t), rel in self.hotpot_data['relations'].items():
                if s in entity_names or t in entity_names:
                    out.append({
                        'source': s,
                        'target': t,
                        'relation_type': rtype,
                        'description': rel.get('description', ''),
                        'text_unit_ids': self._first_text_unit(rel.get('text_unit_ids', []))
                    })
            return out

    def get_text_units_for_entities(self, entity_names: List[str]) -> List[Dict]:
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)-[:APPEARS_IN]->(t:TextUnit)
                    WHERE e.name IN $names
                    RETURN DISTINCT t.id as id, t.title as title,
                           t.content as content, t.entities as entities
                    """, names=entity_names)
                return [dict(record) for record in result]
        else:
            out = []
            for tu_id, tu in self.hotpot_data['text_units'].items():
                if any(n in tu.get('content', '') for n in entity_names) or any(n in tu.get('entities', []) for n in entity_names):
                    out.append({
                        'id': tu_id,
                        'title': tu.get('title', ''),
                        'content': tu.get('content', ''),
                        'entities': tu.get('entities', [])
                    })
            return out

    def get_communities_for_entities(self, entity_names: List[str]) -> List[Dict]:
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Community)-[:CONTAINS]->(e:Entity)
                    WHERE e.name IN $names
                    RETURN c.id as id, c.level as level, c.nodes as nodes,
                           c.report_title as report_title,
                           c.report_summary as report_summary,
                           c.importance_score as importance_score
                    """, names=entity_names)
                return [dict(record) for record in result]
        else:
            out = []
            for com in self.hotpot_data['communities']:
                if any(n in com.get('nodes', []) for n in entity_names):
                    out.append({
                        'id': com.get('id'),
                        'level': com.get('level'),
                        'nodes': com.get('nodes', []),
                        'report_title': com.get('report', {}).get('title', ''),
                        'report_summary': com.get('report', {}).get('summary', ''),
                        'importance_score': com.get('importance_score', 0.0)
                    })
            return out

    def _query_triples_from_neo4j_hotpot(self, matched_entities: List[Dict], question_type: QuestionType) -> List[KGTriple]:
        """针对 HotPotQA 构建的 Neo4j 查询（使用专用 URI/auth）"""
        triples: List[KGTriple] = []

        if not matched_entities:
            return triples

        # hotpot 专用 neo4j 连接信息（从配置读取）
        HOTPOT_URI = getattr(self.config, "hotpot_neo4j_uri", "bolt://localhost:7688/")
        HOTPOT_USER = getattr(self.config, "hotpot_neo4j_user", "neo4j")
        HOTPOT_PASSWORD = getattr(self.config, "hotpot_neo4j_password", "zzm021012")

        try:
            driver = GraphDatabase.driver(HOTPOT_URI, auth=(HOTPOT_USER, HOTPOT_PASSWORD))
        except Exception as e:
            logger.warning(f'无法连接 Hotpot Neo4j ({HOTPOT_URI}): {e}')
            return triples

        entity_names = [entity["name"] for entity in matched_entities[:5]]

        try:
            with driver.session() as session:
                for entity_name in entity_names:
                    # Hotpot 图谱字段可能使用 title/ wiki_title/ name 等字段，做宽松匹配
                    query = """
                    MATCH (n)-[r]->(m)
                    WHERE toLower(coalesce(n.name,'')) CONTAINS toLower($entity)
                        OR toLower(coalesce(n.id,'')) CONTAINS toLower($entity)
                    RETURN n.name as head, type(r) as relation, m.name as tail, properties(r) as rel_props
                    LIMIT 12
                    """

                    result = session.run(query, {"entity": entity_name})
                    for record in result:
                        rel_props = record.get("rel_props", {})
                        paragraph = rel_props.get("paragraph", "") if rel_props else ""

                        triples.append(KGTriple(
                            head=record.get("head") or "",
                            relation=record.get("relation") or "",
                            tail=record.get("tail") or "",
                            source="hotpot_neo4j",
                            paragraph=paragraph,
                            confidence=0.9
                        ))
        except Exception as e:
            logger.error(f'Hotpot Neo4j 查询失败: {e}')
        finally:
            try:
                driver.close()
            except Exception:
                pass

        return triples
    
    def _is_relation_relevant(self, relation: str, question_type: QuestionType) -> bool:
        """检查关系是否与问题类型相关"""
        if not relation:
            return False
        
        # 关系类型分类（扩展版，包含常见事实、逻辑与语义类关系）
        fact_relations = [
            "is_a", "has", "has_parameter", "has_value", "has_standard", "has_property",
            "has_method", "has_test", "has_test_method", "has_requirement", "has_component",
            "has_part", "has_material", "has_condition", "has_version", "has_result",
            "has_output", "has_input", "has_function", "has_subsystem", "has_subcomponent",
            "has_definition", "has_structure", "has_type", "has_formula", "has_format",
            "has_step", "has_process", "has_service", "has_equipment", "has_behavior",
            "provided_by", "produced_by", "published_by", "issued_by", "organized_by",
            "developed_by", "drafted_by", "administered_by", "responsible_for", "provided_by"
        ]
        logic_relations = [
            "requires", "require", "depends_on", "depends", "affects", "affect", "causes",
            "leads_to", "influences", "interfere_with", "prevent", "protect", "replace",
            "combine", "combine_with", "combined_with", "follow", "should_follow", "must_follow",
            "not_require", "not_required", "not_recommended", "prefer", "allow", "access",
            "connect_to", "connects_to", "connected_to", "connects_with", "call_service",
            "transmit_by", "transmit", "output_to", "output", "return_value", "return_type",
            "calculate_by", "calculate", "calculated_by", "calculate"
        ]
        semantic_relations = [
            "related_to", "similar_to", "part_of", "belongs_to", "associated_with", "compatible_with",
            "used_with", "used_by", "used_in", "used_for", "representation", "representation_of",
            "representation_in", "representation_to", "equivalent", "matches_with", "equal_to",
            "based_on", "constitute", "contribute_to", "related", "belongs", "associated"
        ]
        
        relation_lower = relation.lower()
        
        if question_type == QuestionType.SINGLE_STANDARD:
            return any(r in relation_lower for r in fact_relations + logic_relations + semantic_relations)
        elif question_type == QuestionType.CROSS_STANDARD:
            return any(r in relation_lower for r in fact_relations + logic_relations + semantic_relations)
        else:  # correlation
            return any(r in relation_lower for r in fact_relations + logic_relations + semantic_relations)
    
    def _deduplicate_triples(self, triples: List[KGTriple]) -> List[KGTriple]:
        """去重三元组"""
        seen = set()
        unique_triples = []
        
        for triple in triples:
            key = (triple.head, triple.relation, triple.tail)
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        # 按置信度排序
        unique_triples.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_triples
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
