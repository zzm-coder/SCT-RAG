# adaptive_router.py
import os
from pathlib import Path
from typing import Tuple, Dict
import joblib
from api_embedding import APIEmbeddingModel
from data_types import SystemConfig
import logging
import json

class AdaptiveRouter:
    """
    Adaptive-RAG 路由器（严格遵循论文：o = Classifier(q)）
    - 仅使用训练好的 ML 分类器
    - 不提供规则回退
    - 若未训练，初始化即报错
    """

    def __init__(self, config: SystemConfig, model_dir: str = None):
        self.config = config
        if model_dir is None:
            model_dir = os.environ.get("SCT_ADAPTIVE_ROUTER_MODEL", "models/adaptive_router")
        self.model_dir = Path(model_dir)
        self.classifier_path = self.model_dir / "classifier.joblib"

        # 检查分类器是否存在
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Adaptive Router 分类器未找到: {self.classifier_path}\n"
                "请先运行训练脚本生成分类器。"
            )

        # 加载 text2vec 编码器（冻结），优先使用 config 中的路径
        sbert_path = getattr(
            self.config, "semantic_model_path", ""
        )
        if not sbert_path:
            raise ValueError("SCT_EMBEDDING_MODEL is required")
        self.encoder = APIEmbeddingModel(sbert_path)

        # 加载训练好的分类器
        self.classifier = joblib.load(self.classifier_path)
        labels_path = self.model_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Adaptive Router labels not found: {labels_path}")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"]
        expected = ["single", "cross", "correlation"]
        if labels != expected:
            raise ValueError(f"Adaptive Router labels must be {expected}, got {labels}")
        self.label_map = {label: index for index, label in enumerate(labels)}
        self.reverse_label_map = {index: label for label, index in self.label_map.items()}
        logging.info(f"Adaptive Router 分类器加载成功: {self.classifier_path}")

    def classify(self, question: str) -> Tuple[str, Dict]:
        """严格使用训练好的分类器进行预测"""
        return self._predict_with_model(question)

    def _predict_with_model(self, question: str) -> Tuple[str, Dict]:
        embedding = self.encoder.encode([question], normalize_embeddings=True)
        pred_id = self.classifier.predict(embedding)[0]
        label = self.reverse_label_map[int(pred_id)]
        type_ids = {"single": 1, "cross": 2, "correlation": 3}
        if label not in type_ids:
            raise ValueError(f"Unsupported router label: {label}")
        return label, {
            "question_type": label,
            "type_id": type_ids[label],
            "entities": [],
            "intent": "adaptive_router_ml",
        }

