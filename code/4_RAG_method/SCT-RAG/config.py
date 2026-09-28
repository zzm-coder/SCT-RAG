"""配置辅助函数"""
import json
from pathlib import Path

from data_types import SystemConfig

def load_config_from_json(json_path: str) -> SystemConfig:
    """从JSON文件加载配置"""
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return SystemConfig(**config_dict)

def save_config_to_json(config: SystemConfig, json_path: str):
    """保存配置到JSON文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, ensure_ascii=False, indent=2)