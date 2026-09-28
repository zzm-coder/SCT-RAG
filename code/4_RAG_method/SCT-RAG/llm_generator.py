"""LLM生成器"""
import re
import requests
import logging
import time
import sys
from pathlib import Path
from typing import Tuple, List

from config import SystemConfig
from data_types import LLMResponse

# 保证可导入 code/indstd_answer_style.py
_CODE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from indstd_answer_style import (
    STYLE_SINGLE,
    STYLE_CROSS,
    STYLE_CORRELATION,
    RAG_CITATION_RULES,
    ANSWER_SHAPE_SINGLE_RAG,
    ANSWER_SHAPE_CROSS_RAG,
    ANSWER_SHAPE_CORR_RAG,
)

logger = logging.getLogger(__name__)

class LLMGenerator:
    """LLM生成器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.service_url = config.llm_service_url
    
    def generate_answer(self, question: str, context: str, dataset_class: str = None, query_type: str = None) -> LLMResponse:
        """生成答案：工业集默认与 Hybrid-RAG 同一套终答提示；Closed-book 走 generate_closed_book。"""
        start_time = time.time()
        context = self._trim_context(context)
        self._active_query_type = query_type

        try:
            if dataset_class == "ht":
                prompt = self._build_ht_prompt(question, context, query_type=query_type)
            else:
                prompt = self._build_prompt2(question, context)
            response = self._strip_thinking(self._call_llm(prompt))
            if dataset_class == "ht":
                try:
                    from standard_boost import rewrite_pred_standard_ids

                    response = rewrite_pred_standard_ids(question, response)
                except Exception:
                    pass
            answer, citations = self._parse_response(response)
            return LLMResponse(
                answer=answer,
                evidence_citations=citations,
                raw_response=response,
                generation_time=time.time() - start_time,
            )
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            return LLMResponse(
                answer="生成答案时发生错误，请稍后重试。",
                evidence_citations=[],
                raw_response="",
                generation_time=time.time() - start_time
            )

    def generate_closed_book(self, question: str, dataset_class: str = None) -> LLMResponse:
        """无检索作答：不提供任何标准条文，仅用模型参数知识。"""
        start_time = time.time()
        if dataset_class == "hotpot":
            prompt = (
                "Answer the question using only your parametric knowledge. "
                "If unsure, say you do not know. Do not invent citations.\n\n"
                f"Question: {question}\n/no_think"
            )
        else:
            prompt = f"""你是工业标准问答助手。当前**没有**检索到任何标准原文，只能依据模型已有知识作答。

要求：
1. 不要编造不存在的条款号或数值；不确定时明确说不知道。
2. 若记得相关标准，写出标准号（如 GB/T 43927-2024、HB 8519-2015）。
3. 输出格式：
【答案】
……
【证据】
1. 标准号（若无可写「无检索证据，来自模型记忆」）

问题：{question}
/no_think"""
        try:
            response = self._strip_thinking(self._call_llm(prompt))
            answer, citations = self._parse_response(response)
            # 无 .md 证据时，从答案中抽取标准号，供 CiteAcc 做别名匹配
            if not citations:
                citations = self._extract_standard_ids(answer + "\n" + response)
            return LLMResponse(
                answer=answer or response,
                evidence_citations=citations,
                raw_response=response,
                generation_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error("closed-book 生成失败: %s", e)
            return LLMResponse(
                answer="生成答案时发生错误，请稍后重试。",
                evidence_citations=[],
                raw_response="",
                generation_time=time.time() - start_time,
            )

    def _extract_standard_ids(self, text: str) -> List[str]:
        """从文本抽取 GB/T、HB 等标准号，作为 closed-book 引用候选。"""
        if not text:
            return []
        pats = [
            r"GB/T\s*\d+(?:\.\d+)?[—\-]\d{4}",
            r"GBT\s*\d+(?:\.\d+)?[—\-]\d{4}",
            r"HB(?:/Z)?\s*\d+(?:\.\d+)?[—\-]\d{4}",
            r"GJB\s*\d+(?:\.\d+)?[—\-]\d{4}",
        ]
        found, seen = [], set()
        for p in pats:
            for m in re.findall(p, text, flags=re.I):
                key = re.sub(r"\s+", "", m).upper()
                if key not in seen:
                    seen.add(key)
                    found.append(m.strip())
        return found

    def _strip_thinking(self, response: str) -> str:
        """Remove Qwen-style thinking traces, including unterminated <think> blocks."""
        if not response:
            return ""
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'^\s*<think>.*?(?=【(?:答案|Answer)】)', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'^\s*<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE) if "【" not in cleaned else cleaned
        return cleaned.strip()

    def _build_ht_prompt(self, question: str, context: str, query_type: str = None) -> str:
        """Build the category-conditioned, citation-constrained prompt."""
        hop = str(query_type or "single").strip().lower()
        if hop not in ("single", "cross", "correlation"):
            hop = "single"
        if hop == "single":
            type_hint = STYLE_SINGLE
            answer_shape = ANSWER_SHAPE_SINGLE_RAG
        elif hop == "cross":
            type_hint = STYLE_CROSS
            answer_shape = ANSWER_SHAPE_CROSS_RAG
        else:
            type_hint = STYLE_CORRELATION
            answer_shape = ANSWER_SHAPE_CORR_RAG

        std_surfaces = []
        try:
            from standard_boost import extract_question_std_surfaces, extract_std_codes

            std_surfaces = extract_question_std_surfaces(question) or extract_std_codes(question, None) or []
        except Exception:
            std_surfaces = []
        if std_surfaces:
            a = std_surfaces[0]
            b = std_surfaces[1] if len(std_surfaces) > 1 else a
            joined = "、".join(std_surfaces)

            def _fill_std_slots(s: str) -> str:
                return (
                    s.replace("{标准A}", a)
                    .replace("{标准B}", b)
                    .replace("{标准1}", a)
                    .replace("{标准2}", b)
                    .replace("{标准号}", a)
                    .replace("{标准…}", joined)
                )

            answer_shape = _fill_std_slots(answer_shape)
            type_hint = _fill_std_slots(type_hint)

        return f"""你是航空航天工业标准问答助手。下面「检索证据」是系统已经检索到的全部 Top 段落，请自行判断哪些与问题相关并据此作答。

## 题型（{hop}）
{type_hint}

## 硬约束
1. 只根据检索证据推理作答：先判断哪几条与问点直接相关，再据此写答案；禁止凭记忆或外部知识补全。
2. 证据不足时明确写无法根据当前证据作答，不要用外部知识补全。
3. 不漏题：题干每个问点（定义/限值/清单/步骤/是否满足/如何掌握/给出方案）都必须回答；题干点名的每个标准，只要证据中有对应内容就必须写到，禁止只答一半。
4. {RAG_CITATION_RULES}
5. 按题型组织答案：single-standard 直接回答条款事实；cross-standard 分别列出依据后对照或推导；correlation 按主题综合多条证据。
6. 禁止把格式模板中的省略号「……」或「{{标准1}}」「{{标准A}}」「{{标准号}}」原样抄进答案；标准号必须写成题干里的真实编号。
7. 同标准会检索到相邻条款。必须选用直接回答问点的那一条（对象+参数/措施），禁止抄范围/总则/引用文件，也禁止只写「应符合某条款号的规定」而不写出该条中的具体约束。
8. 如问题要求计算或对照，应展示证据中的必要参数、推导过程与明确结论。
9. 【证据】必须抄检索证据中的真实 .md 文件名，并抄写该条的 `Evidence ev_...`（连同 Standard_ID、Clause_ID），禁止写「文件名.md」，禁止编造证据号。
10. 禁止用封面/总规范/型号代号冒充技术要求。禁止只写「应符合某条」而不写出具体约束。

## 输出格式
{answer_shape}
【证据】
1. （真实.md文件名）；Standard_ID; Clause Clause_ID; Evidence ev_xxx：支撑该句的原文摘录

## 检索证据
{context}

## 用户问题
{question}
/no_think
"""

    def _build_prompt2(self, question: str, context: str) -> str:
        """HotPot / 通用英文 QA 提示词（multi-hop，短答案，减少拒答）"""
        prompt_template = f"""
You are a multi-hop question answering system. Answer strictly from the provided context passages.

## Task Requirements
1. Answer based on evidence only: use facts explicitly stated in the context; do not use outside knowledge.
2. Multi-hop reasoning: combine information from two or more passages when the question requires bridging entities.
   Prefer passages that jointly mention multiple entities from the question (bridge evidence).
3. Short final answer: output ONLY the entity name, date, number, or yes/no — avoid full sentences unless necessary.
   Do NOT answer with a distractor that only matches part of the question (e.g. a co-star who is not the asked entity).
4. Prefer answering: if the context contains facts that reasonably support an answer, give your best short answer instead of refusing.
5. Refuse only when the context truly has no relevant information. If refusing, write exactly: Unable to answer based on the provided information
6. **Mandatory citations**: You MUST cite every passage used. Copy the exact `source` id from each `[N] source: ...` line into 【Evidence】.
7. In 【Answer】, add reference markers [1], [2] matching the evidence numbers.

## Context
{context}

## User Question
{question}

## Output Format
【Answer】<short answer only> [1] [2]
【Evidence】1. <exact source id from context>, <short snippet>; 2. <exact source id>, <snippet>; ...

## Example Evidence line
1. 5ac3d9135542995c82c4ac4c_0, the population was 729 at the 2010 census

## Notes
- Each evidence line MUST start with the exact source id (format: 24-char hex + _ + digit), copied from the context.
- For yes/no questions, answer exactly "yes" or "no".
- Do not omit 【Evidence】; list at least one supporting source id you used.
/no_think
"""
        return prompt_template

    def _trim_context(self, context: str) -> str:
        """截断上下文，避免 prompt + max_tokens 超出模型窗口"""
        max_chars = getattr(self.config, "llm_max_context_chars", 24000)
        if not context or len(context) <= max_chars:
            return context or ""
        logger.warning("上下文过长(%d chars)，截断至 %d chars", len(context), max_chars)
        return context[:max_chars] + "\n…[上下文已截断]"
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM服务"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.llm_api_key}"
            }
            max_tokens = getattr(self.config, "llm_max_tokens", 1200)
            service_url = self.service_url
            model_name = self.config.llm_model
            system_msg = "你是一个严谨的问答系统。"
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                "temperature": float(getattr(self.config, "llm_temperature", 0.0)),
                "max_tokens": max_tokens,
                "top_p": 0.9
            }
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            if getattr(self.config, "llm_seed", None) is not None:
                payload["seed"] = int(self.config.llm_seed)
            
            response = requests.post(
                f"{service_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
                proxies={"http": None, "https": None},
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                detail = response.text[:300] if response.text else ""
                raise Exception(f"LLM API错误: {response.status_code}, {detail}")
                
        except requests.exceptions.Timeout:
            raise Exception("LLM服务响应超时")
        except Exception as e:
            raise Exception(f"调用LLM失败: {e}")
    
    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        """解析LLM响应"""
        answer_match = re.search(r'【(?:答案|Answer)】\s*(.*?)(?=\n【(?:证据|Evidence)】|\Z)', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else response
        # 解析证据区，尝试提取编号->文档名的映射
        evidence_match = re.search(r'【(?:证据|Evidence)】\s*(.*)', response, re.DOTALL)
        index_to_md = {}
        index_to_src = {}
        evidence_md_list = []
        evidence_src_list = []
        evidence_id_list = []
        # HotPot chunk source 形如 5ac3d9135542995c82c4ac4c_0
        hotpot_src_pat = re.compile(r'\b([a-f0-9]{24}_\d+)\b', re.IGNORECASE)
        if evidence_match:
            evidence_text = evidence_match.group(1)
            # 工业条款：证据可能以分号分隔写在同一行，因此先全局抽取稳定 ID。
            evidence_id_list = re.findall(
                r'\bev_[0-9a-f]{20}\b', evidence_text, flags=re.IGNORECASE
            )
            # 每行可能为: 1. 文档名称 (段落...)：描述
            for line in evidence_text.splitlines():
                # 先尝试解析行首的编号
                idx_m = re.match(r'\s*(\d+)[\.)）]?\s*(.*)', line)
                content_part = line
                idx = None
                if idx_m:
                    idx = idx_m.group(1)
                    content_part = idx_m.group(2)

                # HotPot：优先提取 chunk source id
                src_m = hotpot_src_pat.search(content_part)
                if src_m:
                    src_id = src_m.group(1)
                    evidence_src_list.append(src_id)
                    if idx:
                        index_to_src[idx] = src_id
                    continue

                # 工业语料：在行中寻找第一个以 .md 结尾的文件名
                md_m = re.search(r'([^\s\(\)\[\]]+?\.md)', content_part, re.IGNORECASE)
                if md_m:
                    md = md_m.group(1).split('/')[-1]
                    evidence_md_list.append(md)
                    if idx:
                        index_to_md[idx] = md

            # 去重并保持出现顺序
            seen_md = set()
            evidence_md_list = [m for m in evidence_md_list if not (m in seen_md or seen_md.add(m))]
            seen_src = set()
            evidence_src_list = [s for s in evidence_src_list if not (s in seen_src or seen_src.add(s))]
            seen_eid = set()
            evidence_id_list = [
                e for e in evidence_id_list
                if not (e.lower() in seen_eid or seen_eid.add(e.lower()))
            ]

        else:
            evidence_text = ""

        # 从答案中提取编号引用并映射为 source / md
        citations = []
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        for m in matches:
            if m in index_to_src:
                citations.append(index_to_src[m])
            elif m in index_to_md:
                md_name = index_to_md[m]
                if md_name and md_name.lower().endswith('.md'):
                    citations.append(md_name)

        # 回退：证据区列出的 source / md
        if not citations:
            citations = (
                list(evidence_id_list)
                or list(evidence_src_list)
                or list(evidence_md_list)
            )

        # 去重并保持顺序
        seen = set()
        final = []
        for c in citations:
            if c not in seen:
                seen.add(c)
                final.append(c)

        return answer, final

    @staticmethod
    def build_answer_judge_prompt(question: str, answer: str, ground_truth: str) -> str:
        """工业标准 Acc 判分提示：必须命中金标决定性结论，禁止空泛原则分。"""
        return f"""评估任务：判断生成答案是否答出了参考答案的决定性结论。给出 0.0~1.0 分。

问题：{question}

参考答案：{ground_truth}

生成答案：{answer}

决定性内容（以参考答案和问题要求为准）：
- 应采用的值、范围、等级及其单位；
- 支持该结论的标准号；
- 限值、合格判据或明确的通过/不通过判断；
- 问题要求的公式、关键符号和必要的代入结果；
- 程序中必须给出的时长、温度、次数或关键步骤。

评分规则：
1. 0.85~1.00：直接回答问题，所有决定性内容均正确，且结论与参考答案一致或语义等价。
2. 0.70~0.84：决定性结论完整且正确，仅缺少不影响答案成立的次要修饰、解释或背景。
3. 0.40~0.69：至少一项决定性内容正确，但遗漏了其他必要标准、数值、条件、步骤或最终判断；答案对问题有实质性回应，但不足以判为完整正确。
4. 0.00~0.39：未给出可验证的决定性内容，仅给出模糊方向或通用原则，或答非所问、拒答、抄写无关条款。
5. 如数值、标准号、采用档、公式结果或通过/不通过判断与参考答案冲突，不得超过 0.39。
用词不同但意思相同仍应高分，不要因为没有逐字相同就压分。
以下情况一律低分，即使句子在回答问题：
- 只说“按更严格/更高/较低/较严的一档掌握”，却未写出具体采用值及对应标准号；
- 只复述题干或给出通用设计原则，未落到参考答案中的具体对象与限值。
示例：参考答案为“同时满足时按 1000 次掌握（HB 8312-2012(2017)）”，生成答案仅写“按两标准中更严格的一档掌握”→ 必须给 0.000。

只输出一个 [0.000, 1.000] 数值，保留三位小数，不要其他文字。/no_think"""

    def evaluate_accuracy(self, question: str, answer: str, ground_truth: str, timeout: int = 10) -> float:
        """使用大模型评估答案与参考答案的准确率（0.0-1.0）。
        返回浮点数，失败时抛出异常或返回 -1.0 标识不可用。
        """
        try:
            prompt = self.build_answer_judge_prompt(question, answer, ground_truth)
            # 直接调用 _call_llm 以复用配置
            resp = self._call_llm(prompt)
            resp = self._strip_thinking(resp)
            # 提取第一个 0-1 浮点数
            m = re.search(r'(?:(?:1(?:\.0+)?)|(?:0(?:\.\d+)?))', resp)
            if m:
                val = float(m.group(0))
                return max(0.0, min(1.0, val))
            # 退回到寻找小数点表示
            m2 = re.search(r'(0?\.\d+|1(?:\.0+)?)', resp)
            if m2:
                val = float(m2.group(0))
                return max(0.0, min(1.0, val))
            raise Exception(f"无法从模型响应解析出数值: {resp}")
        except Exception as e:
            logger.warning(f"通过LLM评估准确率失败: {e}")
            return -1.0
