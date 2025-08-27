import numpy as np
import torch
from tqdm import tqdm

from src.config_schema import AppConfig  # ← この行を追加

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_schema import AppConfi

def get_hf_transformer_embeddings(texts, model_name, device):
    """Generates embeddings using a standard Hugging Face Transformer (BERT, RoBERTa)."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
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
            # Use the [CLS] token's last hidden state
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def get_sentence_transformer_embeddings(texts, model_name, device):
    """Generates embeddings using the SentenceTransformers library."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(texts, show_progress_bar=True)


def get_word2vec_embeddings(texts, w2v_params):
    """Trains a Word2Vec model and generates sentence embeddings by averaging word vectors."""
    from gensim.models import Word2Vec

    print("Training Word2Vec model from scratch...")
    # Simple tokenization
    tokenized_sentences = [text.lower().split() for text in texts]

    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=w2v_params.vector_size,
        window=w2v_params.window,
        min_count=w2v_params.min_count,
        sg=w2v_params.sg,
        workers=4,
    )

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
        return get_word2vec_embeddings(texts, emb_cfg)
    else:
        raise ValueError(f"Unknown embedding type: {emb_cfg.type}")
