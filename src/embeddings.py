import numpy as np
import torch
from tqdm import tqdm

from src.config_schema import AppConfig
from src.utils import (
    get_embedding_cache_path,
    load_text_embedding,
    save_text_embedding,
    resolve_shuffle_seed,
    build_id_index_map,
)

from typing import Optional, Sequence, List

_LAST_LAYER_CACHE: Optional[np.ndarray] = None


def get_last_hidden_state_cache() -> Optional[np.ndarray]:
    """Return the cached mean-pooled hidden states for all layers from the most recent transformer run."""
    return _LAST_LAYER_CACHE


def clear_last_hidden_state_cache() -> None:
    """Reset the cached transformer hidden states."""
    global _LAST_LAYER_CACHE
    _LAST_LAYER_CACHE = None

def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply attention-mask-aware mean pooling to a hidden state tensor."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _select_layer(
    hidden_states: Sequence[torch.Tensor], layer_index: int, model_name: str
) -> torch.Tensor:
    """Return the desired hidden state layer, supporting negative indices."""
    total_layers = len(hidden_states)
    if layer_index < 0:
        layer_index += total_layers
    if layer_index < 0 or layer_index >= total_layers:
        raise ValueError(
            f"Layer index {layer_index} is out of bounds for model '{model_name}' "
            f"which exposes {total_layers} hidden states."
        )
    return hidden_states[layer_index]


def resolve_layer_index(layer_count: int, requested: Optional[int]) -> int:
    """
    Resolve the requested hidden state layer index into a non-negative integer.

    Parameters
    ----------
    layer_count : int
        Total number of layers available in the cached tensor.
    requested : Optional[int]
        The layer index specified in the configuration (can be negative or None).
        When None, the final hidden state is selected.
    """
    if layer_count <= 0:
        raise ValueError("Layer cache is empty; cannot select a hidden state layer.")
    index = layer_count - 1 if requested is None else requested
    if index < 0:
        index += layer_count
    if index < 0 or index >= layer_count:
        raise ValueError(
            f"Layer index {requested} is out of bounds for cached tensor with "
            f"{layer_count} layers."
        )
    return index


def get_hf_transformer_embeddings(
    texts,
    model_name,
    device,
    *,
    layer_index: Optional[int] = None,
    trust_remote_code: bool = False,
):
    """Generates embeddings using a Hugging Face Transformer."""
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub.errors import GatedRepoError

    global _LAST_LAYER_CACHE

    def _raise_access_error(exc: Exception) -> None:
        guidance = (
            f"Unable to download Hugging Face model '{model_name}'. "
            "If you are selecting the gated `google/embeddinggemma-300M` embeddings you must request access at "
            "https://huggingface.co/google/embeddinggemma-300M, accept the terms of use, and authenticate with a Hugging Face "
            "token (for example by running `huggingface-cli login`). If access is not available, update your configuration "
            "to use a public embedding model such as `sentence-transformers/all-MiniLM-L6-v2`."
        )
        raise RuntimeError(f"{guidance} Original error: {exc}") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except (GatedRepoError, OSError) as exc:
        _raise_access_error(exc)

    # Set pad_token if missing
    if getattr(tokenizer, "pad_token", None) is None:
        for attr in ("eos_token", "sep_token", "cls_token", "bos_token"):
            fallback_token = getattr(tokenizer, attr, None)
            if fallback_token is not None:
                tokenizer.pad_token = fallback_token
                if getattr(tokenizer, "pad_token_id", None) is None and hasattr(
                    tokenizer, "convert_tokens_to_ids"
                ):
                    pad_token_id = tokenizer.convert_tokens_to_ids(fallback_token)
                    if pad_token_id is not None:
                        tokenizer.pad_token_id = pad_token_id
                break

    try:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        ).to(device)
    except (GatedRepoError, OSError) as exc:
        _raise_access_error(exc)

    embeddings = []
    layer_accumulator: Optional[List[List[np.ndarray]]] = None
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc=f"Vectorizing with {model_name}"):
            batch = texts[i : i + 32]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.hidden_states

            # Select the appropriate hidden state layer
            if layer_index is None:
                hidden_state = outputs.last_hidden_state
            else:
                if hidden_states is None:
                    raise RuntimeError(
                        f"Model '{model_name}' did not return hidden states even though "
                        "one was requested."
                    )
                hidden_state = _select_layer(hidden_states, layer_index, model_name)

            pooled_selected = _mean_pool(hidden_state, attention_mask).to(
                dtype=torch.float32
            )
            batch_embeddings = pooled_selected.cpu().numpy()
            embeddings.append(batch_embeddings)

            # Accumulate all layer hidden states for caching
            if hidden_states is not None:
                if layer_accumulator is None:
                    layer_accumulator = [[] for _ in range(len(hidden_states))]
                for idx, state in enumerate(hidden_states):
                    pooled_tensor = _mean_pool(state, attention_mask).to(
                        dtype=torch.float32
                    )
                    pooled = pooled_tensor.cpu().numpy()
                    layer_accumulator[idx].append(pooled)

    if layer_accumulator is not None:
        per_layer = [
            np.vstack(chunks).astype(np.float32, copy=False)
            for chunks in layer_accumulator
        ]
        layer_cache = np.stack(per_layer, axis=1)
    else:
        layer_cache = None
    _LAST_LAYER_CACHE = layer_cache
    return np.vstack(embeddings)


def get_sentence_transformer_embeddings(texts, model_name, device):
    """Generates embeddings using the SentenceTransformers library."""
    from sentence_transformers import SentenceTransformer

    clear_last_hidden_state_cache()
    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, show_progress_bar=True)


def get_word2vec_embeddings(texts, w2v_params, cfg: Optional[AppConfig] = None):
    """Trains a Word2Vec model and generates sentence embeddings by averaging word vectors."""
    from gensim.models import Word2Vec

    clear_last_hidden_state_cache()
    print("Training Word2Vec model from scratch...")
    # Simple tokenization
    tokenized_sentences = [text.lower().split() for text in texts]

    # Resolve seed for reproducibility
    seed = None
    if cfg is not None:
        reproducibility = getattr(cfg, "reproducibility", None)
        if reproducibility is not None:
            deterministic = bool(getattr(reproducibility, "deterministic", False))
            if deterministic:
                seed = getattr(reproducibility, "seed", None)
                if seed is None:
                    eval_cfg = getattr(cfg, "evaluation", None)
                    if eval_cfg is not None:
                        seed = getattr(eval_cfg, "random_state", None)

    workers = getattr(w2v_params, "workers", 4)
    deterministic = seed is not None
    word2vec_kwargs = dict(
        sentences=tokenized_sentences,
        vector_size=w2v_params.vector_size,
        window=w2v_params.window,
        min_count=w2v_params.min_count,
        sg=w2v_params.sg,
        workers=1 if deterministic else workers,
    )
    if deterministic:
        word2vec_kwargs["seed"] = seed

    model = Word2Vec(**word2vec_kwargs)

    wv = model.wv
    embedding_dim = model.vector_size

    sentence_embeddings = []
    for tokens in tqdm(tokenized_sentences, desc="Averaging Word2Vec vectors"):
        word_vectors = [wv[word] for word in tokens if word in wv]
        if word_vectors:
            sentence_embeddings.append(np.mean(word_vectors, axis=0))
        else:
            # For sentences with no words in vocab, use a zero vector
            sentence_embeddings.append(np.zeros(embedding_dim))

    return np.vstack(sentence_embeddings)


def get_embeddings(texts: list, cfg: AppConfig) -> np.ndarray:

    """Factory function to select and run the appropriate embedding model."""
    clear_last_hidden_state_cache()
    emb_cfg = cfg.embedding
    print(f"\n--- Generating embeddings using model: {emb_cfg.name} ---")

    # Dictionary-based dispatch for embedding type selection
    embedding_dispatchers = {
        "hf_transformer": lambda: get_hf_transformer_embeddings(
            texts,
            emb_cfg.model_name,
            cfg.device,
            layer_index=emb_cfg.hidden_state_layer,
            trust_remote_code=emb_cfg.trust_remote_code,
        ),
        "sentence_transformer": lambda: get_sentence_transformer_embeddings(
            texts, emb_cfg.model_name, cfg.device
        ),
        "word2vec": lambda: get_word2vec_embeddings(texts, emb_cfg, cfg),
    }

    dispatcher = embedding_dispatchers.get(emb_cfg.type)
    if dispatcher is not None:
        return dispatcher()
    else:
        raise ValueError(f"Unknown embedding type: {emb_cfg.type}")


def load_or_generate_embeddings(
    cfg: AppConfig, texts: Sequence[str], ids: Sequence
) -> np.ndarray:
    """
    Load cached embeddings when available; otherwise compute and cache them.
    """
    embedding_cache_path = get_embedding_cache_path(cfg)
    cache = load_text_embedding(embedding_cache_path)
    resolved_seed = resolve_shuffle_seed(cfg)

    # Try to load from cache
    if cache is not None:
        cached_ids, cached_embeddings, cached_seed, cached_layer_embeddings = cache
        if cached_seed == resolved_seed:
            # Convert ids to strings once for efficient lookup
            str_ids, id_to_index = build_id_index_map(ids, cached_ids)
            try:
                selection_indices = np.asarray(
                    [id_to_index[sid] for sid in str_ids], dtype=int
                )
                if (
                    cfg.embedding.type == "hf_transformer"
                    and cached_layer_embeddings is not None
                ):
                    target_layer = resolve_layer_index(
                        cached_layer_embeddings.shape[1],
                        getattr(cfg.embedding, "hidden_state_layer", None),
                    )
                    return cached_layer_embeddings[
                        selection_indices, target_layer, :
                    ]
                else:
                    cached = np.asarray(cached_embeddings)
                    return cached[selection_indices]
            except (KeyError, ValueError) as exc:
                if isinstance(exc, KeyError):
                    print("Cached embeddings are missing required ids. Recomputing...")
                else:
                    print(f"{exc} Recomputing embeddings...")
        else:
            print("Cached embeddings shuffle seed mismatch. Recomputing...")

    # Generate new embeddings
    X_vectors = get_embeddings(texts, cfg)
    layer_cache = get_last_hidden_state_cache()
    save_text_embedding(
        ids,
        X_vectors,
        resolved_seed,
        embedding_cache_path,
        layer_embeddings=layer_cache,
    )
    clear_last_hidden_state_cache()

    return X_vectors
