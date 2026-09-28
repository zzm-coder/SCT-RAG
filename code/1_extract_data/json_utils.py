"""LLM 输出 JSON 解析工具：清理控制字符、修复常见格式问题"""
import json
import re
from typing import List, Any


def _strip_think_blocks(text: str) -> str:
    return re.sub(r'(\<think\>.*?\<\/think\>)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_json_array(text: str) -> str:
    start, end = text.find('['), text.rfind(']')
    if start == -1 or end <= start:
        return '[]'
    return text[start:end + 1]


def _sanitize_json_string(json_str: str) -> str:
    """移除 JSON 中非法控制字符，修复 trailing comma"""
    # 移除除 \t 外的 ASCII 控制字符（保留已转义的 \n \r）
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', json_str)
    # 移除 trailing comma: ,] 或 ,}
    cleaned = re.sub(r',\s*(\]|\})', r'\1', cleaned)
    return cleaned


def _try_load(json_str: str) -> Any:
    return json.loads(json_str, strict=False)


def parse_llm_json_array(raw: str) -> List[dict]:
    """
    从 LLM 原始输出解析 JSON 数组，失败时逐级降级修复。
    """
    cleaned = _strip_think_blocks(raw)
    json_str = _extract_json_array(cleaned)
    if json_str == '[]' and '[' not in cleaned:
        return []

    attempts = [
        json_str,
        _sanitize_json_string(json_str),
    ]

    for attempt in attempts:
        try:
            data = _try_load(attempt)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            return []
        except json.JSONDecodeError:
            continue

    # 最后尝试：逐行提取完整 JSON 对象
    objects = []
    for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL):
        try:
            obj = _try_load(_sanitize_json_string(m.group()))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects
