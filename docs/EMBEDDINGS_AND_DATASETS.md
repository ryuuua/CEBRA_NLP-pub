# Embeddings And Datasets

The public config keeps the legacy Hydra names for the first supported Hugging
Face embeddings and datasets.

## Embeddings

| Use | Hydra override | Model |
| --- | --- | --- |
| BERT | `embedding=bert` | `bert-base-uncased` |
| embedding_gemma | `embedding=embeddinggemma` | `google/embeddinggemma-300m` |
| all mini | `embedding=sentence_bert` | `sentence-transformers/all-MiniLM-L6-v2` |

Install the Hugging Face dependencies before using these embeddings:

```bash
python -m pip install -e ".[hf,video]"
```

`embedding=embeddinggemma` uses a Hugging Face model that requires access on
the Hugging Face Hub. Request access for the model page, accept the model terms,
and authenticate the local environment with Hugging Face before running cache
generation.

## Datasets

| Use | Hydra override | Hugging Face dataset |
| --- | --- | --- |
| dair-ai/emotion | `dataset=dair-ai` | `dair-ai/emotion` |
| AG News | `dataset=ag_news` | `ag_news` |

## Examples

```bash
python cache_embeddings.py dataset=dair-ai embedding=bert
python cache_embeddings.py dataset=ag_news embedding=sentence_bert
python cache_embeddings.py dataset=dair-ai embedding=embeddinggemma
```

The tiny CSV tutorial path remains the default quickstart and now uses
`embedding=sentence_bert`. The dataset configs above are for users who want to
run the public pipeline with Hugging Face datasets.
