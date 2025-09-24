import numpy as np
import torch
from tqdm import tqdm

from src.config_schema import AppConfig

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.config_schema import AppConfig

def get_hf_transformer_embeddings(texts, model_name, device):
    """Generates embeddings using a standard Hugging Face Transformer (BERT, RoBERTa)."""
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub.errors import GatedRepoError

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
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (GatedRepoError, OSError) as exc:
        _raise_access_error(exc)

    if getattr(tokenizer, "pad_token", None) is None:
        fallback_token = None
        for attr in ("eos_token", "sep_token", "cls_token", "bos_token"):
            token = getattr(tokenizer, attr, None)
            if token is not None:
                fallback_token = token
                break
        if fallback_token is not None:
            tokenizer.pad_token = fallback_token
            if getattr(tokenizer, "pad_token_id", None) is None and hasattr(
                tokenizer, "convert_tokens_to_ids"
            ):
                pad_token_id = tokenizer.convert_tokens_to_ids(fallback_token)
                if pad_token_id is not None:
                    tokenizer.pad_token_id = pad_token_id
    try:
        model = AutoModel.from_pretrained(model_name).to(device)
    except (GatedRepoError, OSError) as exc:
        _raise_access_error(exc)
    embeddings = []
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
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            last_hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def get_sentence_transformer_embeddings(texts, model_name, device):
    """Generates embeddings using the SentenceTransformers library."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, show_progress_bar=True)


def get_word2vec_embeddings(texts, w2v_params, cfg: Optional[AppConfig] = None):
    """Trains a Word2Vec model and generates sentence embeddings by averaging word vectors."""
    from gensim.models import Word2Vec

    print("Training Word2Vec model from scratch...")
    # Simple tokenization
    tokenized_sentences = [text.lower().split() for text in texts]

    reproducibility = getattr(cfg, "reproducibility", None) if cfg is not None else None
    deterministic = bool(getattr(reproducibility, "deterministic", False))
    seed = None
    if deterministic:
        seed = getattr(reproducibility, "seed", None)
        if seed is None and cfg is not None:
            eval_cfg = getattr(cfg, "evaluation", None)
            if eval_cfg is not None:
                seed = getattr(eval_cfg, "random_state", None)

    workers = getattr(w2v_params, "workers", 4)
    word2vec_kwargs = dict(
        sentences=tokenized_sentences,
        vector_size=w2v_params.vector_size,
        window=w2v_params.window,
        min_count=w2v_params.min_count,
        sg=w2v_params.sg,
        workers=1 if deterministic else workers,
    )
    if deterministic and seed is not None:
        word2vec_kwargs["seed"] = seed

    model = Word2Vec(**word2vec_kwargs)

    wv = model.wv
    embedding_dim = model.vector_size

    sentence_embeddings = []
    for tokens in tqdm(tokenized_sentences, desc="Averaging Word2Vec vectors"):
        word_vectors = [wv[word] for word in tokens if word in wv]
        if len(word_vectors) > 0:
            sentence_embeddings.append(np.mean(word_vectors, axis=0))
        else:
            # For sentences with no words in vocab, use a zero vector
            sentence_embeddings.append(np.zeros(embedding_dim))

    return np.vstack(sentence_embeddings)


def get_embeddings(texts: list, cfg: AppConfig) -> np.ndarray:

    """Factory function to select and run the appropriate embedding model."""
    emb_cfg = cfg.embedding
    print(f"\n--- Generating embeddings using model: {emb_cfg.name} ---")

    if emb_cfg.type == "hf_transformer":
        # AppConfigとして扱われることで、cfg.deviceに正しくアクセスできる
        return get_hf_transformer_embeddings(texts, emb_cfg.model_name, cfg.device)
    elif emb_cfg.type == "sentence_transformer":
        return get_sentence_transformer_embeddings(
            texts, emb_cfg.model_name, cfg.device
        )
    elif emb_cfg.type == "word2vec":
        return get_word2vec_embeddings(texts, emb_cfg, cfg)
    else:
        raise ValueError(f"Unknown embedding type: {emb_cfg.type}")
