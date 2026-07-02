from __future__ import annotations


class _MissingWandb:
    run = None
    summary: dict[str, object] = {}

    class _Config:
        @staticmethod
        def update(*args, **kwargs) -> None:
            return None

    config = _Config()

    def init(self, *args, **kwargs):
        raise RuntimeError("W&B logging requires `pip install 'cebra-nlp-public[tracking]'`.")

    def finish(self) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        return None

    def save(self, *args, **kwargs) -> None:
        return None

    def Artifact(self, *args, **kwargs):
        raise RuntimeError("W&B artifacts require `pip install 'cebra-nlp-public[tracking]'`.")

    def Histogram(self, values):
        return values


try:
    import wandb as wandb  # type: ignore[no-redef]
except Exception:
    wandb = _MissingWandb()


__all__ = ["wandb"]
