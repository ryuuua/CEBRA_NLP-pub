from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeAlias


MaxMemoryKey: TypeAlias = str
MaxMemoryValue: TypeAlias = int | str


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
    # Multi-label datasets can specify multiple binary columns or a delimited label column.
    multi_label: bool = False
    label_columns: Optional[List[str]] = None
    label_delimiter: Optional[str] = None
    drop_multi_label_samples: bool = False

    # Kaggle datasets require a handle to download the data.
    kaggle_handle: Optional[str] = None

    hf_path: Optional[str] = None
    trust_remote_code: bool = False
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
    model_name: Optional[str] = None
    output_dim: Optional[int] = None
    cache_tag: Optional[str] = None
    embedding_seed: Optional[int] = None
    parallel_strategy: str = "auto"
    pooling: str = "mean"
    hidden_state_layer: Optional[int] = None
    trust_remote_code: bool = False
    normalize_embeddings: bool = False
    batch_size: int = 32
    cache_all_layers: bool = False
    device_map: Optional[str] = None
    max_memory: Optional[Dict[MaxMemoryKey, MaxMemoryValue]] = None
    torch_dtype: Optional[str] = "float32"
    data_parallel: bool = False
    multi_process: bool = False
    multi_process_devices: Optional[List[int]] = None


@dataclass
class EmbeddingCollectionConfig:
    name: str
    combine_mode: str = "concat"
    embeddings: List[EmbeddingConfig] = field(default_factory=list)


@dataclass
class CEBRAParamsConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    temperature: float = 1.0
    distance: str = "cosine"
    prior: str = "uniform"
    verbose: bool = True
    # Backward-compatibility shim for older artifacts where num_units lived under params.
    num_units: Optional[int] = None


@dataclass
class CEBRAConfig:
    name: str = ""
    output_dim: int = 0
    max_iterations: int = 0
    conditional: str = "none"
    criterion: str = "infonce"
    model_architecture: str = "offset1-model"
    num_units: int = 512
    params: CEBRAParamsConfig = field(default_factory=CEBRAParamsConfig)
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    save_embeddings: bool = False
    save_epoch_trajectory: bool = False
    trajectory_every_n_epochs: int = 1
    trajectory_sample_size: Optional[int] = 1000
    trajectory_fps: int = 8
    trajectory_max_frames: int = 180
    trajectory_one_epoch_one_frame: bool = False
    trajectory_connect_segments: bool = True
    trajectory_rotate_camera: bool = False
    trajectory_camera_elev: float = 18.0
    trajectory_camera_azim: float = 42.0
    trajectory_trail_length: int = 10
    trajectory_axis_padding: float = 0.08
    trajectory_frame_width: int = 1920
    trajectory_frame_height: int = 1080
    trajectory_dpi: int = 180
    trajectory_mp4_crf: int = 18
    trajectory_mp4_preset: str = "slow"


@dataclass
class LocalLinearityProbeConfig:
    enabled: bool = False
    neighbors: int = 15
    sample_size: Optional[int] = None
    random_state: Optional[int] = None
    ridge_alpha: float = 1e-3
    store_scores: bool = False


@dataclass
class EvaluationConfig:
    test_size: float
    random_state: int
    knn_neighbors: int
    enable_plots: bool = True
    knn_backend: str = "auto"
    faiss_gpu_id: int = 0
    local_linearity_probe: LocalLinearityProbeConfig = field(
        default_factory=LocalLinearityProbeConfig
    )


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
class LabelRandomizationConfig:
    """
    Controls optional randomization of labels used for CEBRA training.
    mode: "none" (default), "permutation" (shuffle labels), "random_int" (draw new ints).
    num_classes: cap for random_int generation; defaults to number of unique labels.
    """

    mode: str = "none"
    num_classes: Optional[int] = None


@dataclass
class LabelOverlayConfig:
    enabled: bool = False
    cache_in_cache_stage: bool = True
    text_mode: str = "label_name"
    show_in_cebra_space: bool = True
    show_in_pca: bool = True
    show_centroids_in_pca: bool = True
    include_split_views: bool = True


@dataclass
class TrajectoryAnalysisConfig:
    enabled: bool = False
    checkpoint_every_n_steps: int = 100
    save_initial_checkpoint: bool = True
    save_final_checkpoint: bool = True
    render_after_train: bool = True
    centroid_scope: str = "train"
    render_dims: int = 3
    include_sample_cloud: bool = True
    export_animation: bool = True
    export_static_panels: bool = True
    show_centroids: bool = True
    show_trajectory_lines: bool = True
    export_clean_variant: bool = False
    render_checkpoint_stride: int = 1
    fps: int = 8
    max_frames: int = 180


@dataclass
class CinematicRenderConfig:
    enabled: bool = False
    render_backend: str = "auto"
    export_poster: bool = True
    export_animation: bool = True
    export_beauty_master: bool = True
    export_analysis_master: bool = True
    export_cinematic_master: bool = True
    axis_style: str = "corner_guides"
    look_preset: str = "glass_wireframe"
    fps: int = 18
    max_frames: int = 240
    trail_length: int = 36
    poster_width: int = 3840
    poster_height: int = 2160
    video_width: int = 3840
    video_height: int = 2160
    gif_width: int = 1920
    gif_height: int = 1080
    supersample_scale: float = 1.25
    beauty_video_supersample_scale: float = 4.0
    beauty_poster_supersample_scale: float = 4.0
    analysis_supersample_scale: float = 1.25
    beauty_hold_final_seconds: float = 1.0
    beauty_depth_fog_strength: float = 0.20
    beauty_depth_fog_cool_mix: float = 0.18
    beauty_particle_core_scale: float = 1.0
    beauty_particle_halo_scale: float = 1.0
    beauty_trail_mode: str = "label_only"
    glow_blur_small_px: float = 6.0
    glow_blur_large_px: float = 18.0
    glow_gain: float = 0.9
    prefer_gpu_encode: bool = True
    figure_width: float = 16.0
    figure_height: float = 9.0
    static_dpi: int = 240
    animation_dpi: int = 180
    camera_distance_scale: float = 0.74
    camera_fov_degrees: float = 23.0
    camera_auto_zoom_out: bool = True
    camera_zoom_margin: float = 0.88
    camera_zoom_curve_power: float = 1.8
    camera_elev: float = 22.0
    camera_elev_wobble: float = 0.35
    camera_azim_start: float = 16.0
    camera_azim_end: float = 24.0
    background_color: str = "#030712"


@dataclass
class PCAAnalysisConfig:
    residual_variance_threshold: float = 0.01
    component_variance_floor: float = 0.001
    max_components: Optional[int] = None
    plot_sample_limit: Optional[int] = None
    min_components_for_plots: int = 3
    share_full_basis_across_views: bool = True
    export_dir: Optional[str] = None


@dataclass
class PathsConfig:
    embedding_cache_dir: str
    model_dir: str
    kaggle_data_dir: str = "data/kaggle/hierarchical-text-classification"
    embedding_cache_tag: Optional[str] = None


@dataclass
class ConsistencyCheckConfig:
    enabled: bool = True
    mode: str = "runs"
    num_runs: int = 5
    dataset_ids: List[str] = field(default_factory=list)


@dataclass
class HyperParamTuningConfig:
    """Hyperparameter ranges for grid search."""

    # NOTE: Hydra nested runtime options are structurally open-ended and OmegaConf
    # cannot represent recursive/union container value types in structured configs.
    hydra: Dict[str, Any] = field(default_factory=dict)
    output_dims: List[int] = field(default_factory=lambda: list(range(2, 21)))
    batch_sizes: List[int] = field(default_factory=lambda: [512])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3])


@dataclass
class StageConfig:
    name: str
    require_cache: bool = False
    require_model: bool = False
    run_tag: Optional[str] = None
    save_splits: bool = True
    use_saved_splits: bool = True
    use_saved_embeddings: bool = True
    model_path: Optional[str] = None
    evaluate_after_train: bool = False


@dataclass
class AppConfig:
    paths: PathsConfig
    stage: StageConfig
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    cebra: CEBRAConfig
    evaluation: EvaluationConfig
    wandb: WandBConfig
    consistency_check: ConsistencyCheckConfig
    hpt: HyperParamTuningConfig
    ddp: DDPConfig
    reproducibility: ReproducibilityConfig
    label_overlay: LabelOverlayConfig = field(default_factory=LabelOverlayConfig)
    trajectory_analysis: TrajectoryAnalysisConfig = field(
        default_factory=TrajectoryAnalysisConfig
    )
    cinematic_render: CinematicRenderConfig = field(
        default_factory=CinematicRenderConfig
    )
    label_randomization: LabelRandomizationConfig = field(
        default_factory=LabelRandomizationConfig
    )
    pca_analysis: PCAAnalysisConfig = field(default_factory=PCAAnalysisConfig)
    device: str = "cpu"
    embedding_collection: Optional[EmbeddingCollectionConfig] = None
