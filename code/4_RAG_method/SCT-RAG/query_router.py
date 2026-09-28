"""查询路由器：LLM 意图分类 + 可审计的标准号符号解析。"""
import json
import requests
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re

from config import SystemConfig
from data_types import QuestionType

logger = logging.getLogger(__name__)

@dataclass
class RouterResponse:
    """路由器响应"""
    question_type: QuestionType
    type_id: int
    entities: List[str]
    intent: str
    metadata: Dict[str, Any]

class QueryRouter:
    """LLM router for category prediction and entity extraction."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.service_url = config.llm_service_url
        self.api_key = config.llm_api_key
        self.model = config.llm_model
    
    def analyze_query(self, question: str, dataset_class: str = None) -> RouterResponse:
        """
        使用大模型分析查询
        
        Returns:
            RouterResponse: 包含问题类型、实体列表、用户意图
        """
        try:
            # Category prediction and entity extraction share one zero-shot call.
            result = self._analyze_with_llm(question, dataset_class=dataset_class)

            if result:
                type_id, entities, intent = result
                # 类别严格保留 LLM 的三分类结果，不再根据题干模板词
                # 二次改类。标准号属于格式化实体，由符号解析器保底，
                # 再与 LLM 抽取的其他专业实体合并。
                standard_ids = self.extract_standard_ids(question)
                entities = self._merge_grounded_entities(question, standard_ids, entities)
                question_type = QuestionType.from_int(type_id)

                logger.info(f"查询分析成功: 类型={question_type.value}(ID={type_id}), 实体={len(entities)}个, 意图={intent}")

                return RouterResponse(
                    question_type=question_type,
                    type_id=type_id,
                    entities=entities,
                    intent=intent,
                    metadata={
                        "source": "llm_prompt_classifier",
                        "classification_source": "llm_prompt_only",
                        "standard_extraction_source": "symbolic_standard_parser",
                        "standard_ids": standard_ids,
                    }
                )
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
        
        # LLM 失败时不用模板规则猜测类别；选择保守的多证据路径，
        # 并显式标记失败，便于评估时单独统计。
        type_id = 3
        question_type = QuestionType.from_int(type_id)
        standard_ids = self.extract_standard_ids(question)
        return RouterResponse(
            question_type=question_type,
            type_id=type_id,
            entities=standard_ids,
            intent="查询事实信息",
            metadata={
                "source": "conservative_failure_fallback",
                "error": "LLM 分析失败",
                "classification_source": "fixed_safe_fallback",
                "standard_extraction_source": "symbolic_standard_parser",
                "standard_ids": standard_ids,
            }
        )

    @staticmethod
    def extract_standard_ids(question: str) -> List[str]:
        """抽取题干中显式标准号，不参与题型判定。

        这是神经-符号路由中的符号层：LLM 负责语义意图，解析器
        负责高精度格式化标识符，不访问标签或黄金证据。
        """
        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?:GB\s*/?\s*T|GBT|HB\s*[/._]\s*Z|HB[_\s]*Z|HB|GJB|QJ)"
            r"[+\s]*\d{2,6}(?:[._-]\d+)?\s*[-—－]\s*\d{4}(?:\s*\(\d{4}\))?",
            flags=re.I,
        )
        out, seen = [], set()
        for match in pattern.finditer(question or ""):
            value = re.sub(r"\s+", " ", match.group(0).replace("+", " ")).strip()
            value = re.sub(r"\s*[-—－]\s*", "-", value)
            value = re.sub(r"GB\s*/?\s*T", "GB/T", value, flags=re.I)
            value = re.sub(r"HB\s*(?:/|\.|_)\s*Z", "HB/Z", value, flags=re.I)
            value = re.sub(
                r"^(HB|GJB|QJ)\s*(?=\d)",
                lambda m: m.group(1).upper() + " ",
                value,
                flags=re.I,
            )
            key = value.upper().replace(" ", "")
            if key not in seen:
                seen.add(key)
                out.append(value)
        return out

    @staticmethod
    def _merge_grounded_entities(
        question: str, standard_ids: List[str], llm_entities: List[str]
    ) -> List[str]:
        """保留题干中可验证的 LLM 实体，标准号以符号解析为准。"""
        compact_question = re.sub(r"\s+", "", question or "").lower()
        merged, seen = [], set()
        for value in list(standard_ids) + list(llm_entities or []):
            entity = str(value or "").strip()
            if not entity:
                continue
            compact = re.sub(r"\s+", "", entity).lower()
            is_standard_like = bool(re.match(r"^(?:gb/?t|gbt|hb(?:/?z)?|gjb|qj)", compact))
            if is_standard_like and entity not in standard_ids:
                # 丢弃 LLM 幻觉或格式错误的标准号。
                continue
            if entity not in standard_ids and compact not in compact_question:
                continue
            key = compact
            if key not in seen:
                seen.add(key)
                merged.append(entity)
        return merged
    
    def _analyze_with_llm(self, question: str, dataset_class: str = None) -> Optional[Tuple[int, List[str], str]]:
        """使用大模型分析查询"""
        try:
            if dataset_class == "ht":
                prompt = self._build_analysis_prompt(question)
            else:
                prompt = self._build_analysis_prompt2(question)
            response = self._call_llm(prompt)
            
            if response:
                return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"大模型分析失败: {e}")
        
        return None

    def _build_analysis_prompt(self, question: str) -> str:
        """构建分析提示词：严格区分 single / cross / correlation。"""
        return f"""请分析以下航空航天工业标准问答问题，并判定检索题型。

问题：{question}

## 题型定义（必须三选一：1/2/3）
只允许输出下面三类之一，不得创建额外类别。

1 = single（单标准）
- 针对一个产品/部位/参数，从单一标准摘录、解释或核实取值
- 问法如“根据某标准…是多少/有哪些/对不对”

2 = cross（跨标准对照或串联推理）
- 两部及以上标准的对照、取交、取更严、或 A 的输出代入 B
- 问法如是否相同、哪一档更严、两标准同时满足时应取哪一区间
- 两标准对照「哪一档更严这样设计对不对」也是 2，不是判断类

3 = correlation（跨标准综合方案）
- 综合参考多部标准，给出同一对象的可执行方案
- 综合方案「对不对」也是 3

## 判别优先级
1) 综合参考/统筹/兼顾/协同/给出方案 → 3
2) 对照/更严/取交/代入计算，或题干两个及以上标准做比较 → 2
3) 其余单标准事实问或单标准对错核实 → 1

## 另需输出
- entities：只提取题干中出现的标准号与关键术语，禁止编造
- intent：一句话概括信息需求

严格输出 JSON：
{{
    "type_id": 1或2或3,
    "entities": ["实体1", "实体2"],
    "intent": "用户核心意图"
}}
/no_think"""

    def _build_analysis_prompt2(self, question: str) -> str:
        """构建分析提示词"""
        return f"""请分析以下问题：

问题：{question}

请提供以下分析结果：
关键实体列表：从问题中精确提取的关键实体,禁止添加问题中未出现的实体。

请严格按照以下JSON格式输出：
{{
    "type_id": 2,
    "entities": ["实体1", "实体2", ...],
    "intent": ""
}}

请确保分析准确、简洁，并严格按JSON格式输出。
/no_think"""
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用大模型"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "你是一个严谨的航空航天制造领域专家，请严格按照要求分析问题并提供准确、专业的回答。"
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,  # 低温度确保稳定性
                "max_tokens": 1024,
                "top_p": 0.9
            }
            if getattr(self.config, "llm_seed", None) is not None:
                payload["seed"] = int(self.config.llm_seed)
            
            start_time = time.time()
            response = requests.post(
                f"{self.service_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                logger.info(f"大模型调用成功，耗时: {elapsed_time:.2f}s")
                return content
            else:
                logger.error(f"大模型API错误: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("大模型调用超时")
            return None
        except Exception as e:
            logger.error(f"调用大模型失败: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Tuple[int, List[str], str]:
        """解析大模型响应"""
        try:
            # 提取JSON部分
            json_match = response.strip()
            
            # 尝试直接解析整个响应
            try:
                result = json.loads(json_match)
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.error(f"无法从响应中提取JSON: {response[:200]}...")
                    raise ValueError("无效的响应格式")
            
            # 提取和分析数据
            type_id = int(result.get("type_id", 1))
            entities = result.get("entities", [])
            intent = result.get("intent", "查询信息")
            
            # 验证和清理数据
            if type_id not in [1, 2, 3]:
                logger.warning(f"type_id超出范围: {type_id}，调整为1")
                type_id = 1
            
            if not isinstance(entities, list):
                entities = []
            
            # 清理实体列表
            entities = [str(e).strip() for e in entities if e and str(e).strip()]
            
            return type_id, entities, intent
            
        except Exception as e:
            logger.error(f"解析大模型响应失败: {e}")
            raise ValueError("无法解析 LLM 路由响应") from e
