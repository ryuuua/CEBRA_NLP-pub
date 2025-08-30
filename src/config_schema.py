# src/config_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal

@dataclass
class VisualizationConfig:
    emotion_colors: Dict[str, str]
    emotion_order: List[str]

@dataclass
class DatasetConfig:
    name: str
    text_column: str
    label_map: Dict[int, str]
    visualization: VisualizationConfig
    label_column: Optional[str] = None
    hf_path: Optional[str] = None
    source: str = "hf"
    data_files: Optional[str] = None

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
    # Allow arbitrary model names so new architectures can be specified
    model_architecture: str = "offset1-model"
    params: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

@dataclass
class EvaluationConfig:
    test_size: float
    random_state: int
    knn_neighbors: int
    enable_plots: bool = True

@dataclass
class WandBConfig:
    project: str
    run_name: str
    entity: Optional[str] = None

@dataclass
class DDPConfig:
    world_size: int
    rank: int
    local_rank: int

@dataclass
class PathsConfig:
    embedding_cache_dir: str
    kaggle_data_dir: str = "data/kaggle/hierarchical-text-classification"

@dataclass
class ConsistencyCheckConfig:
    enabled: bool = True  # チェックを有効にするか
    num_runs: int = 5     # 何回モデルを訓練するか


@dataclass
class HyperParamTuningConfig:
    """Hyperparameter ranges for grid search."""
    output_dims: List[int] = field(default_factory=lambda: list(range(2, 21)))
    batch_sizes: List[int] = field(default_factory=lambda: [512])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3])


# 全ての設定をまとめるトップレベルのデータクラス
@dataclass
class AppConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    cebra: CEBRAConfig
    evaluation: EvaluationConfig
    wandb: WandBConfig
    consistency_check: ConsistencyCheckConfig
    hpt: HyperParamTuningConfig
    ddp: DDPConfig
    device: str
