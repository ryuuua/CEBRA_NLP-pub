import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from src.config_schema import AppConfig
from src.utils import normalize_binary_labels
from src.results import save_interactive_plot, save_static_2d_plots


def prepare_plot_labels(
    cfg: AppConfig, conditional_data: np.ndarray
) -> Tuple[List, Optional[dict], List]:
    """Return text labels plus palette/order for plotting functions."""
    if cfg.cebra.conditional == "discrete":
        data = normalize_binary_labels(np.asarray(conditional_data).reshape(-1))
        label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
        labels = [label_map[int(v)] for v in data]

        vis_cfg = getattr(cfg.dataset, "visualization", None)
        palette = None
        if vis_cfg is not None and getattr(vis_cfg, "emotion_colors", None):
            palette = OmegaConf.to_container(vis_cfg.emotion_colors, resolve=True)

        if vis_cfg is not None and getattr(vis_cfg, "emotion_order", None):
            order = list(OmegaConf.to_container(vis_cfg.emotion_order, resolve=True))
        else:
            order = sorted({label_map[int(v)] for v in data})
        return labels, palette, order

    values = np.asarray(conditional_data)
    if values.ndim == 2 and values.shape[1] >= 1:
        labels = values[:, 0].tolist()
    else:
        labels = values.reshape(-1).tolist()
    return labels, None, []

def plot_embedding_distributions(
    embeddings: np.ndarray, labels: np.ndarray, cfg: DictConfig, output_dir: Path
):
    
    print("\n--- Step 5: Visualizing embedding distributions per dimension ---")

    dim = embeddings.shape[1]

    # 設定ファイルからラベルと色の情報を取得
    label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
    emotion_colors = {k: v for k, v in cfg.dataset.visualization.emotion_colors.items()}
    emotion_order = list(cfg.dataset.visualization.emotion_order)

    # 数値ラベルをテキストラベルに変換
    text_labels = [label_map[label] for label in labels]

    n_cols = 3
    n_rows = (dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    fig.suptitle(f"CEBRA {dim}-Dimensional Embedding Distributions", fontsize=18)
    axes = axes.flatten()

    for i in range(dim):
        ax = axes[i]
        plot_df = pd.DataFrame({"value": embeddings[:, i], "label": text_labels})
        sns.kdeplot(
            data=plot_df,
            x="value",
            hue="label",
            palette=emotion_colors,
            hue_order=emotion_order,
            ax=ax,
            fill=True,
            common_norm=False,
            alpha=0.5,
        )
        ax.set_title(f"Dimension {i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        legend = ax.get_legend()
        if i != 0 and legend is not None:
            legend.remove()

    # 不要な軸を非表示に
    for i in range(dim, len(axes)):
        axes[i].set_visible(False)

    # 凡例を一つにまとめる
    first_legend = axes[0].get_legend()
    if first_legend is not None:
        handles, legend_labels = first_legend.get_legend_handles_labels()
        first_legend.remove()
        fig.legend(
            handles,
            legend_labels,
            title="Emotion",
            bbox_to_anchor=(1.0, 0.9),
            loc="upper left",
        )

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    filename = output_dir / f"cebra_{dim}d_distributions_from_artifact.png"
    plt.savefig(filename)
    print(f"Distribution plot saved to: {filename}")
    plt.close()


def render_discrete_visualizations(
    cfg: AppConfig,
    embeddings_full: np.ndarray,
    text_labels_full: Sequence[str],
    output_dir: Path,
    *,
    interactive_path: Optional[Path] = None,
    wandb_hook: Optional[Callable[[Iterable[Path]], None]] = None,
) -> None:
    palette = OmegaConf.to_container(cfg.dataset.visualization.emotion_colors, resolve=True)
    order = OmegaConf.to_container(cfg.dataset.visualization.emotion_order, resolve=True)

    if not cfg.evaluation.enable_plots:
        return

    interactive = interactive_path or (output_dir / "cebra_interactive_discrete.html")
    save_interactive_plot(
        embeddings_full,
        text_labels_full,
        cfg.cebra.output_dim,
        palette,
        "Interactive CEBRA (Discrete)",
        interactive,
    )
    static_paths: List[Path] = []
    if wandb_hook is not None and interactive.exists():
        wandb_hook([interactive])
    save_static_2d_plots(
        embeddings_full,
        text_labels_full,
        palette,
        "CEBRA Embeddings (Discrete)",
        output_dir,
        order,
    )
    if wandb_hook is not None:
        static_paths = [
            output_dir / "static_PCA_plot.png",
            output_dir / "static_UMAP_plot.png",
        ]
        existing = [path for path in static_paths if path.exists()]
        if existing:
            wandb_hook(existing)


def render_continuous_visualizations(
    cfg: AppConfig,
    embeddings_full: np.ndarray,
    values: Sequence[float],
    output_dir: Path,
    *,
    interactive_path: Optional[Path] = None,
    wandb_hook: Optional[Callable[[Iterable[Path]], None]] = None,
) -> None:
    if not cfg.evaluation.enable_plots:
        return

    interactive = interactive_path or (output_dir / "None.html")
    save_interactive_plot(
        embeddings=embeddings_full,
        text_labels=values,
        output_dim=cfg.cebra.output_dim,
        palette=None,
        title="Interactive CEBRA (None - Colored by Valence)",
        output_path=interactive,
    )
    if wandb_hook is not None and interactive.exists():
        wandb_hook([interactive])
