from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from cebra_nlp_public import embedding_cache_adapter as adapter


def _raise_import_error(_name: str) -> ModuleType:
    raise ImportError("not installed")


def test_adapter_reports_public_fallback_when_labenv_missing(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "import_module", _raise_import_error)

    backend = adapter.active_cache_backend()

    assert backend.name == "cebra_nlp_public"
    assert backend.enabled is False
    assert backend.reason is not None


def test_fallback_save_and_load_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(adapter, "import_module", _raise_import_error)
    path = tmp_path / "cache" / "fallback.npz"

    backend = adapter.save_embedding_cache(
        ["id-1", "id-2"],
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        7,
        path,
        embedding_type="sentence_transformer",
        registry_key="sentence_bert",
        rulebook_id="cebra_nlp_public_local_cache_v1",
        metadata_payload={"dataset_key": "tiny", "variant_tag": "unit"},
    )
    loaded = adapter.load_embedding_cache(path)

    assert backend.name == "cebra_nlp_public"
    assert loaded is not None
    ids, embeddings, shuffle_seed, *_rest = loaded.payload
    assert ids.tolist() == ["id-1", "id-2"]
    assert embeddings.shape == (2, 2)
    assert shuffle_seed == 7


class _FakeLabenvCache(ModuleType):
    __version__ = "0.3.2"

    def save_v2_embedding_cache(
        self,
        *,
        ids: Any,
        embeddings: Any,
        path: str | Path,
        metadata: dict[str, Any],
        layer_embeddings: Any = None,
        provenance: dict[str, Any] | None = None,
        require_locator: bool = False,
    ) -> dict[str, Any]:
        del require_locator
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = dict(metadata)
        metadata_json["schema_version"] = 2
        metadata_json["cache_format"] = "labenv_embedding_cache.self_describing_npz"
        metadata_json["provenance"] = dict(provenance or {})
        payload = {
            "ids": np.asarray(ids, dtype=str),
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "metadata_json": np.asarray(json.dumps(metadata_json), dtype=object),
        }
        if layer_embeddings is not None:
            payload["layer_embeddings"] = np.asarray(layer_embeddings, dtype=np.float32)
        np.savez(output, **payload)
        return {"path": str(output), "verification_status": "passed"}

    def validate_cache_npz(self, path: str | Path, *, verification_mode: str = "fast") -> dict[str, Any]:
        del verification_mode
        return {"path": str(path), "verification_status": "passed", "verification_errors": []}

    def load_cache_arrays(
        self,
        path: str | Path,
        *,
        load_layer_embeddings: bool = True,
    ) -> dict[str, Any]:
        with np.load(path, allow_pickle=True) as payload:
            metadata_json = json.loads(str(payload["metadata_json"].item()))
            result: dict[str, Any] = {
                "ids": payload["ids"],
                "embeddings": payload["embeddings"],
                "metadata": metadata_json,
                "layer_embeddings": None,
            }
            if load_layer_embeddings and "layer_embeddings" in payload.files:
                result["layer_embeddings"] = payload["layer_embeddings"]
            return result


def test_labenv_v2_adapter_normalizes_public_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_labenv = _FakeLabenvCache("labenv_embedding_cache")

    def _missing_dist_version(name: str) -> str:
        raise adapter.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "import_module", lambda _name: fake_labenv)
    monkeypatch.setattr(adapter.metadata, "version", _missing_dist_version)
    path = tmp_path / "cache" / "labenv-v2.npz"

    backend = adapter.save_embedding_cache(
        ["id-1"],
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        11,
        path,
        hidden_state_layer=-1,
        embedding_type="hf_transformer",
        pooling="mean",
        registry_key="bert",
        rulebook_id="cebra_nlp_public_local_cache_v1",
        metadata_payload={
            "dataset_name": "ag_news",
            "dataset_key": "ag_news__unit",
            "embedding_name": "bert",
            "embedding_model_name": "bert-base-uncased",
            "variant_tag": "bert__unit",
        },
    )
    loaded = adapter.load_embedding_cache(path)

    assert backend.name == "labenv_embedding_cache"
    assert loaded is not None
    ids, embeddings, shuffle_seed, _layers, layer, embedding_type, pooling, rulebook_id, registry_key = loaded.payload
    assert ids.tolist() == ["id-1"]
    assert embeddings.shape == (1, 2)
    assert shuffle_seed == 11
    assert layer == -1
    assert embedding_type == "hf_transformer"
    assert pooling == "mean"
    assert rulebook_id == "cebra_nlp_public_local_cache_v1"
    assert registry_key == "bert"
