# src/config_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal

@dataclass
class VisualizationConfig:
    emotion_colors: Dict[str, str]
    emotion_order: List[str] # ← この行を追加

@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    text_column: str
    label_column: str
    label_map: Dict[int, str]
    visualization: VisualizationConfig

@dataclass
class EmbeddingConfig:
    name: str
    type: str
    model_name: str
    # Word2Vec用のパラメータなど、特定のモデルのみで使う設定も定義可能
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    sg: int = 0

@dataclass
class CEBRAConfig:
    output_dim: int
    max_iterations: int
    conditional: str 
    model_architecture: Literal[
        "offset0-model",
        "offset5-model",
        "offset10-model",
    ] = "offset0-model"
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationConfig:
    test_size: float
    random_state: int
    knn_neighbors: int

@dataclass
class MLflowConfig:
    experiment_name: str
    run_name: str
    tracking_uri: Optional[str] = None

@dataclass
class PathsConfig:
    embedding_cache_dir: str

@dataclass
class ConsistencyCheckConfig:
    enabled: bool = True  # チェックを有効にするか
    num_runs: int = 5     # 何回モデルを訓練するか


# 全ての設定をまとめるトップレベルのデータクラス
@dataclass
class AppConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    cebra: CEBRAConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    consistency_check: ConsistencyCheckConfig
