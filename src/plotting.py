import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf

from src.config_schema import AppConfig


def prepare_plot_labels(
    cfg: AppConfig, conditional_data: np.ndarray
) -> Tuple[List, Optional[dict], List]:
    """Return text labels plus palette/order for plotting functions."""
    if cfg.cebra.conditional == "discrete":
        data = np.asarray(conditional_data).reshape(-1)
        unique = set(np.unique(data).tolist())
        if unique == {-1, 1}:
            data = np.where(data == -1, 0, 1)
        label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
        labels = [label_map[int(v)] for v in data]

        vis_cfg = getattr(cfg.dataset, "visualization", None)
        palette = (
            OmegaConf.to_container(vis_cfg.emotion_colors, resolve=True)
            if vis_cfg is not None and getattr(vis_cfg, "emotion_colors", None)
            else None
        )
        order = (
            list(OmegaConf.to_container(vis_cfg.emotion_order, resolve=True))
            if vis_cfg is not None and getattr(vis_cfg, "emotion_order", None)
            else sorted({label_map[int(v)] for v in data})
        )
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
    """
    埋め込みの各次元の分布を、感情ラベルで色分けしてプロットする。
    """
    print("\n--- Step 5: Visualizing embedding distributions per dimension ---")

    dim = embeddings.shape[1]

    # 設定ファイルからラベルと色の情報を取得
    label_map = {int(k): v for k, v in cfg.dataset.label_map.items()}
    emotion_colors = {k: v for k, v in cfg.dataset.visualization.emotion_colors.items()}
    emotion_order = list(cfg.dataset.visualization.emotion_order)

    # 数値ラベルをテキストラベルに変換
    text_labels = [label_map[l] for l in labels]

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
        if i != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

    # 不要な軸を非表示に
    for i in range(dim, len(axes)):
        axes[i].set_visible(False)

    # 凡例を一つにまとめる
    if axes[0].get_legend() is not None:
        handles, plot_labels = axes[0].get_legend_handles_labels()
        axes[0].get_legend().remove()
        fig.legend(
            handles,
            plot_labels,
            title="Emotion",
            bbox_to_anchor=(1.0, 0.9),
            loc="upper left",
        )

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    filename = output_dir / f"cebra_{dim}d_distributions_from_artifact.png"
    plt.savefig(filename)
    print(f"Distribution plot saved to: {filename}")
    plt.close()
