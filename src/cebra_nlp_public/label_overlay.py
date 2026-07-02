from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_schema import AppConfig


@dataclass(frozen=True)
class LabelOverlaySpec:
    ids: np.ndarray
    texts: list[str]
    label_ids: np.ndarray
    label_names: list[str]


def _load_single_embedding_cache(
    cfg: AppConfig,
    texts: list[str],
    ids: np.ndarray,
    *,
    require_cache: bool,
) -> np.ndarray:
    from .embeddings import load_or_generate_embeddings

    return load_or_generate_embeddings(
        cfg,
        texts,
        ids,
        labels=None,
        require_cache=require_cache,
    )


def _load_embedding_collection_cache(
    cfg: AppConfig,
    texts: list[str],
    ids: np.ndarray,
    *,
    require_cache: bool,
) -> np.ndarray:
    from .embeddings import load_or_generate_embedding_collection

    return load_or_generate_embedding_collection(
        cfg,
        texts,
        ids,
        labels=None,
        require_cache=require_cache,
    )


def build_label_overlay_spec(cfg: AppConfig) -> LabelOverlaySpec | None:
    return _build_label_overlay_spec(cfg, force_enabled=False)


def _build_label_overlay_spec(
    cfg: AppConfig,
    *,
    force_enabled: bool,
) -> LabelOverlaySpec | None:
    overlay_cfg = getattr(cfg, "label_overlay", None)
    if overlay_cfg is None or (
        not force_enabled and not bool(getattr(overlay_cfg, "enabled", False))
    ):
        return None
    if str(getattr(cfg.cebra, "conditional", "none")).lower() != "discrete":
        return None

    text_mode = str(getattr(overlay_cfg, "text_mode", "label_name") or "").strip().lower()
    if text_mode != "label_name":
        raise ValueError(
            "Unsupported label_overlay.text_mode="
            f"{getattr(overlay_cfg, 'text_mode', None)!r}. Only 'label_name' is supported."
        )

    raw_label_map = getattr(cfg.dataset, "label_map", None) or {}
    if not raw_label_map:
        return None

    label_pairs = sorted((int(label_id), str(label_name)) for label_id, label_name in raw_label_map.items())
    dataset_name = str(getattr(cfg.dataset, "name", "") or "dataset")
    label_ids = np.asarray([label_id for label_id, _ in label_pairs], dtype=np.int64)
    label_names = [label_name for _, label_name in label_pairs]
    ids = np.asarray(
        [f"label::{dataset_name}::{label_id}" for label_id in label_ids.tolist()],
        dtype=str,
    )
    texts = list(label_names)
    return LabelOverlaySpec(
        ids=ids,
        texts=texts,
        label_ids=label_ids,
        label_names=label_names,
    )


def load_or_generate_label_overlay_embeddings(
    cfg: AppConfig,
    *,
    require_cache: bool = False,
    force_enabled: bool = False,
) -> tuple[LabelOverlaySpec | None, np.ndarray | None]:
    spec = _build_label_overlay_spec(cfg, force_enabled=force_enabled)
    if spec is None:
        return None, None

    overlay_require_cache = bool(
        require_cache and getattr(cfg.label_overlay, "cache_in_cache_stage", True)
    )
    if getattr(cfg, "embedding_collection", None) and cfg.embedding_collection.embeddings:
        vectors = _load_embedding_collection_cache(
            cfg,
            spec.texts,
            spec.ids,
            require_cache=overlay_require_cache,
        )
    else:
        vectors = _load_single_embedding_cache(
            cfg,
            spec.texts,
            spec.ids,
            require_cache=overlay_require_cache,
        )
    return spec, vectors


def write_label_overlay_manifest(
    spec: LabelOverlaySpec,
    output_path: Path,
    *,
    cache_path: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "overlay_id",
                "label_id",
                "label_name",
                "overlay_text",
                "cache_path",
            ],
        )
        writer.writeheader()
        for overlay_id, label_id, label_name, overlay_text in zip(
            spec.ids.tolist(),
            spec.label_ids.tolist(),
            spec.label_names,
            spec.texts,
        ):
            writer.writerow(
                {
                    "overlay_id": overlay_id,
                    "label_id": int(label_id),
                    "label_name": label_name,
                    "overlay_text": overlay_text,
                    "cache_path": cache_path or "",
                }
            )
    return output_path


def write_label_overlay_points_csv(
    spec: LabelOverlaySpec,
    cebra_embeddings: np.ndarray,
    pca_embeddings: np.ndarray,
    output_path: Path,
    *,
    scope_name: str,
) -> Path:
    cebra_embeddings = np.asarray(cebra_embeddings, dtype=np.float32)
    pca_embeddings = np.asarray(pca_embeddings, dtype=np.float32)
    if cebra_embeddings.shape[0] != len(spec.label_names):
        raise ValueError(
            "Label overlay CEBRA embeddings row count does not match label overlay spec."
        )
    if pca_embeddings.ndim != 2 or pca_embeddings.shape[0] != len(spec.label_names):
        raise ValueError(
            "Label overlay PCA embeddings must have shape (n_labels, n_dims), got "
            f"{pca_embeddings.shape}."
        )

    fieldnames = [
        "scope",
        "overlay_id",
        "label_id",
        "label_name",
        "overlay_text",
    ]
    fieldnames.extend(
        f"pca_dim_{dim_index + 1}" for dim_index in range(pca_embeddings.shape[1])
    )
    fieldnames.extend(
        f"cebra_dim_{dim_index + 1}" for dim_index in range(cebra_embeddings.shape[1])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (overlay_id, label_id, label_name, overlay_text) in enumerate(
            zip(
                spec.ids.tolist(),
                spec.label_ids.tolist(),
                spec.label_names,
                spec.texts,
            )
        ):
            row = {
                "scope": scope_name,
                "overlay_id": overlay_id,
                "label_id": int(label_id),
                "label_name": label_name,
                "overlay_text": overlay_text,
            }
            for dim_index, value in enumerate(pca_embeddings[row_index], start=1):
                row[f"pca_dim_{dim_index}"] = float(value)
            for dim_index, value in enumerate(cebra_embeddings[row_index], start=1):
                row[f"cebra_dim_{dim_index}"] = float(value)
            writer.writerow(row)
    return output_path


def write_label_centroid_points_csv(
    spec: LabelOverlaySpec,
    centroid_cebra_embeddings: np.ndarray,
    centroid_pca_embeddings: np.ndarray,
    output_path: Path,
    *,
    scope_name: str,
    fit_scope_name: str,
    sample_counts: np.ndarray | None = None,
    overlay_cebra_embeddings: np.ndarray | None = None,
    overlay_pca_embeddings: np.ndarray | None = None,
) -> Path:
    centroid_cebra_embeddings = np.asarray(centroid_cebra_embeddings, dtype=np.float32)
    centroid_pca_embeddings = np.asarray(centroid_pca_embeddings, dtype=np.float32)
    if centroid_cebra_embeddings.shape[0] != len(spec.label_names):
        raise ValueError(
            "Label centroid CEBRA embeddings row count does not match label overlay spec."
        )
    if (
        centroid_pca_embeddings.ndim != 2
        or centroid_pca_embeddings.shape[0] != len(spec.label_names)
    ):
        raise ValueError(
            "Label centroid PCA embeddings must have shape (n_labels, n_dims), got "
            f"{centroid_pca_embeddings.shape}."
        )

    counts = (
        np.zeros(len(spec.label_names), dtype=np.int64)
        if sample_counts is None
        else np.asarray(sample_counts, dtype=np.int64)
    )
    if counts.shape != (len(spec.label_names),):
        raise ValueError(
            "sample_counts must have shape (n_labels,), got "
            f"{counts.shape}."
        )

    overlay_cebra = None
    if overlay_cebra_embeddings is not None:
        overlay_cebra = np.asarray(overlay_cebra_embeddings, dtype=np.float32)
        if overlay_cebra.shape != centroid_cebra_embeddings.shape:
            raise ValueError(
                "overlay_cebra_embeddings must match centroid_cebra_embeddings shape, got "
                f"{overlay_cebra.shape} vs {centroid_cebra_embeddings.shape}."
            )

    overlay_pca = None
    if overlay_pca_embeddings is not None:
        overlay_pca = np.asarray(overlay_pca_embeddings, dtype=np.float32)
        if overlay_pca.shape != centroid_pca_embeddings.shape:
            raise ValueError(
                "overlay_pca_embeddings must match centroid_pca_embeddings shape, got "
                f"{overlay_pca.shape} vs {centroid_pca_embeddings.shape}."
            )

    fieldnames = [
        "scope",
        "fit_scope",
        "overlay_id",
        "label_id",
        "label_name",
        "overlay_text",
        "sample_count",
    ]
    fieldnames.extend(
        f"centroid_pca_dim_{dim_index + 1}"
        for dim_index in range(centroid_pca_embeddings.shape[1])
    )
    if overlay_pca is not None:
        fieldnames.extend(
            f"overlay_pca_dim_{dim_index + 1}"
            for dim_index in range(overlay_pca.shape[1])
        )
        fieldnames.extend(
            f"delta_pca_dim_{dim_index + 1}"
            for dim_index in range(overlay_pca.shape[1])
        )
        fieldnames.append("delta_pca_l2")
    fieldnames.extend(
        f"centroid_cebra_dim_{dim_index + 1}"
        for dim_index in range(centroid_cebra_embeddings.shape[1])
    )
    if overlay_cebra is not None:
        fieldnames.extend(
            f"overlay_cebra_dim_{dim_index + 1}"
            for dim_index in range(overlay_cebra.shape[1])
        )
        fieldnames.extend(["delta_cebra_l2", "delta_cebra_cosine"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, (overlay_id, label_id, label_name, overlay_text) in enumerate(
            zip(
                spec.ids.tolist(),
                spec.label_ids.tolist(),
                spec.label_names,
                spec.texts,
            )
        ):
            row = {
                "scope": scope_name,
                "fit_scope": fit_scope_name,
                "overlay_id": overlay_id,
                "label_id": int(label_id),
                "label_name": label_name,
                "overlay_text": overlay_text,
                "sample_count": int(counts[row_index]),
            }
            for dim_index, value in enumerate(
                centroid_pca_embeddings[row_index], start=1
            ):
                row[f"centroid_pca_dim_{dim_index}"] = float(value)
            if overlay_pca is not None:
                delta_pca = overlay_pca[row_index] - centroid_pca_embeddings[row_index]
                for dim_index, value in enumerate(overlay_pca[row_index], start=1):
                    row[f"overlay_pca_dim_{dim_index}"] = float(value)
                for dim_index, value in enumerate(delta_pca, start=1):
                    row[f"delta_pca_dim_{dim_index}"] = float(value)
                row["delta_pca_l2"] = float(np.linalg.norm(delta_pca))
            for dim_index, value in enumerate(
                centroid_cebra_embeddings[row_index], start=1
            ):
                row[f"centroid_cebra_dim_{dim_index}"] = float(value)
            if overlay_cebra is not None:
                delta_cebra = overlay_cebra[row_index] - centroid_cebra_embeddings[row_index]
                centroid_norm = float(np.linalg.norm(centroid_cebra_embeddings[row_index]))
                overlay_norm = float(np.linalg.norm(overlay_cebra[row_index]))
                cosine = np.nan
                if centroid_norm > 0.0 and overlay_norm > 0.0:
                    cosine = float(
                        np.dot(overlay_cebra[row_index], centroid_cebra_embeddings[row_index])
                        / (overlay_norm * centroid_norm)
                    )
                for dim_index, value in enumerate(overlay_cebra[row_index], start=1):
                    row[f"overlay_cebra_dim_{dim_index}"] = float(value)
                row["delta_cebra_l2"] = float(np.linalg.norm(delta_cebra))
                row["delta_cebra_cosine"] = cosine
            writer.writerow(row)
    return output_path


__all__ = [
    "LabelOverlaySpec",
    "build_label_overlay_spec",
    "load_or_generate_label_overlay_embeddings",
    "write_label_centroid_points_csv",
    "write_label_overlay_manifest",
    "write_label_overlay_points_csv",
]
