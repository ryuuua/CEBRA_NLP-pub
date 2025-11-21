# src/config_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

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
    label_remap: Dict[int, int] = field(default_factory=dict)
    label_column: Optional[str] = None
    # When working with multi-label data, either specify multiple
    # columns containing binary indicators or provide a single column
    # with delimited label strings. The `label_map` defines the order of
    # labels in the resulting multi-hot vectors.
    multi_label: bool = False
    label_columns: Optional[List[str]] = None
    label_delimiter: Optional[str] = None
    drop_multi_label_samples: bool = False

    hf_path: Optional[str] = None
    trust_remote_code: bool = False
    # Kaggle datasets require a handle to download the data.
    kaggle_handle: Optional[str] = None
    sklearn_dataset: Optional[str] = None
    source: str = "hf"
    data_files: Optional[str] = None
    splits: List[str] = field(default_factory=list)
    shuffle: bool = False
    shuffle_seed: Optional[int] = None


@dataclass
class EmbeddingConfig:
    name: str
    type: str
    model_name: str
    output_dim: int = 0
    hidden_state_layer: Optional[int] = None
    pooling: str = "mean"
    trust_remote_code: bool = False
    # Word2Vec用のパラメータなど、特定のモデルのみで使う設定も定義可能
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    sg: int = 0
    workers: int = 4

@dataclass
class CEBRAConfig:
    name: str = ""
    output_dim: int = 0
    max_iterations: int = 0
    conditional: str = "none"
    criterion: str = "infonce"
    # Allow arbitrary model names so new architectures can be specified
    model_architecture: str = "offset1-model"
    params: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    save_embeddings: bool = False

@dataclass
class EvaluationConfig:
    test_size: float
    random_state: int
    knn_neighbors: int
    enable_plots: bool = True
    knn_backend: str = "auto"
    faiss_gpu_id: int = 0


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
class ReproducibilityConfig:
    seed: int
    deterministic: bool = False
    cudnn_benchmark: bool = False


@dataclass
class PCAAnalysisConfig:
    residual_variance_threshold: float = 0.01
    component_variance_floor: float = 0.001
    max_components: Optional[int] = None
    plot_sample_limit: Optional[int] = None
    min_components_for_plots: int = 3
    export_dir: Optional[str] = None

@dataclass
class PathsConfig:
    embedding_cache_dir: str
    kaggle_data_dir: str = "data/kaggle/hierarchical-text-classification"

@dataclass
class ConsistencyCheckConfig:
    enabled: bool = True  # チェックを有効にするか
    mode: str = "runs"     # "runs" or "datasets"
    num_runs: int = 5     # 何回モデルを訓練するか
    dataset_ids: List[str] = field(default_factory=list)


@dataclass
class HyperParamTuningConfig:
    """Hyperparameter ranges for grid search."""
    hydra: Dict[str, Any] = field(default_factory=dict)
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
    reproducibility: ReproducibilityConfig 
    pca_analysis: PCAAnalysisConfig
    device: str = "cpu"
