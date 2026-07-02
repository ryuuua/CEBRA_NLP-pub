"""Compatibility wrapper for trajectory visualization helpers.

This package is the target home for future trajectory and cinematic
visualization extraction. For now it re-exports the existing stage functions so
root scripts and stage modules keep their current behavior.
"""

from __future__ import annotations

from typing import Any


def _trajectory_viz_attr(name: str) -> Any:
    from ...stages import trajectory_viz

    return getattr(trajectory_viz, name)


def _cinematic_viz_attr(name: str) -> Any:
    from ...stages import trajectory_cinematic_viz

    return getattr(trajectory_cinematic_viz, name)


def _epoch_viz_attr(name: str) -> Any:
    from . import epoch

    return getattr(epoch, name)


def validate_trajectory_requirements(*args: Any, **kwargs: Any) -> Any:
    return _trajectory_viz_attr("validate_trajectory_requirements")(*args, **kwargs)


def load_label_drift_checkpoint_records(*args: Any, **kwargs: Any) -> Any:
    return _trajectory_viz_attr("load_label_drift_checkpoint_records")(*args, **kwargs)


def build_label_drift_metrics_frame(*args: Any, **kwargs: Any) -> Any:
    return _trajectory_viz_attr("build_label_drift_metrics_frame")(*args, **kwargs)


def render_label_drift_trajectory(*args: Any, **kwargs: Any) -> Any:
    return _trajectory_viz_attr("render_label_drift_trajectory")(*args, **kwargs)


def run_trajectory_viz(*args: Any, **kwargs: Any) -> Any:
    return _trajectory_viz_attr("run")(*args, **kwargs)


def render_cinematic_trajectory(*args: Any, **kwargs: Any) -> Any:
    return _cinematic_viz_attr("render_cinematic_trajectory")(*args, **kwargs)


def run_cinematic_trajectory_viz(*args: Any, **kwargs: Any) -> Any:
    return _cinematic_viz_attr("run")(*args, **kwargs)


def render_saved_epoch_trajectory(*args: Any, **kwargs: Any) -> Any:
    return _epoch_viz_attr("render_saved_epoch_trajectory")(*args, **kwargs)


run_trajectory_viz.__name__ = "run"
run_cinematic_trajectory_viz.__name__ = "run"

__all__ = [
    "build_label_drift_metrics_frame",
    "load_label_drift_checkpoint_records",
    "render_cinematic_trajectory",
    "render_label_drift_trajectory",
    "render_saved_epoch_trajectory",
    "run_cinematic_trajectory_viz",
    "run_trajectory_viz",
    "validate_trajectory_requirements",
]
