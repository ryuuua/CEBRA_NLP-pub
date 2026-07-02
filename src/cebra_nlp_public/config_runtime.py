from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from omegaconf import OmegaConf

from .config_schema import AppConfig


def _to_omegaconf(cfg: Any):
    if OmegaConf.is_config(cfg):
        return cfg
    return OmegaConf.create(cfg)


def to_typed_app_config(cfg: Any) -> AppConfig:
    """Convert Hydra/OmegaConf config objects into an AppConfig instance."""
    if isinstance(cfg, AppConfig):
        return cfg

    if not isinstance(cfg, Mapping) and not OmegaConf.is_config(cfg):
        raise TypeError(f"Expected OmegaConf config or AppConfig, got {type(cfg)!r}")

    merged = OmegaConf.merge(OmegaConf.structured(AppConfig), _to_omegaconf(cfg))
    typed = OmegaConf.to_object(merged)
    if not isinstance(typed, AppConfig):
        raise TypeError(f"Expected AppConfig, got {type(typed)!r}")
    return typed


def to_config_container(cfg: Any, *, resolve: bool = True) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=resolve)
    if is_dataclass(cfg):
        field_names = [f.name for f in fields(cfg)]
        converted = {
            name: to_config_container(getattr(cfg, name), resolve=resolve)
            for name in field_names
        }
        field_name_set = set(field_names)
        extra_attrs = getattr(cfg, "__dict__", {})
        for name, value in extra_attrs.items():
            if name not in field_name_set:
                converted[name] = to_config_container(value, resolve=resolve)
        return converted
    if isinstance(cfg, Mapping):
        return {
            key: to_config_container(value, resolve=resolve)
            for key, value in cfg.items()
        }
    if isinstance(cfg, (list, tuple, set)):
        return [to_config_container(value, resolve=resolve) for value in cfg]
    return cfg


def app_config_to_dict(cfg: AppConfig) -> dict[str, object]:
    """Return a plain dict representation suitable for logging/serialization."""
    container = to_config_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            f"Expected config container to be dict, got {type(container)!r}"
        )
    return dict(container)
