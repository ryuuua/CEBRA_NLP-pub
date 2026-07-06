# cebra-nlp-public

Minimal public tutorial pipeline for text embeddings, CEBRA training, trajectory
rendering, and MP4/GIF/PNG video artifacts.

## Acknowledgement

This repository is a downstream NLP tutorial and visualization pipeline built
on the original CEBRA method and software. The core CEBRA model and training
concepts come from:

- Paper: Schneider, S., Lee, J. H. & Mathis, M. W. "Learnable latent embeddings
  for joint behavioural and neural analysis." Nature 617, 360-368 (2023).
  https://doi.org/10.1038/s41586-023-06031-6
- Official CEBRA repository: https://github.com/AdaptiveMotorControlLab/CEBRA

This project is not an official CEBRA release. It depends on the public
`cebra` Python package and adds NLP dataset configs, embedding-cache utilities,
and trajectory/video tutorial wrappers around that foundation. We gratefully
acknowledge the CEBRA authors and maintainers. See
[docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) and [CITATION.cff](CITATION.cff).

The default tutorial uses a tiny CSV dataset and `embedding=sentence_bert`
(`sentence-transformers/all-MiniLM-L6-v2`) on CPU.

## Quick Start With Docker

Build the image:

```bash
docker build -t cebra-nlp-public .
```

Run the tutorial pipeline:

```bash
docker run --rm \
  -v "$PWD/runs:/app/runs" \
  cebra-nlp-public \
  python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

The tutorial runs:

1. embedding cache generation
2. CEBRA training
3. label drift trajectory rendering
4. cinematic poster, GIF preview, and MP4 export

Primary output:

```text
runs/tutorial_video/models/**/tutorial_video/label_drift_trajectory/
  label_drift_beauty_master_3d.mp4
  label_drift_beauty_master_preview_3d.gif
  label_drift_beauty_master_3d.png
  manifest.json
  trajectory_render_report.json
```

Docker Compose is also available:

```bash
docker compose run --rm app
```

## Local Python

Docker is the canonical path for the public tutorial. For local Python with
`uv`:

```bash
uv sync --extra hf --extra video
uv run python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

Plain pip also works:

```bash
python -m pip install -e ".[hf,video]"
python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

Local Python requires `ffmpeg` to be available on `PATH` for MP4 export.
See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the Docker and uv contract.

## Embeddings And Datasets

The public config includes the legacy Hydra names for the first Hugging Face
embedding and dataset options:

- `embedding=bert`
- `embedding=embeddinggemma`
- `embedding=sentence_bert`
- `dataset=dair-ai`
- `dataset=ag_news`

Use `python -m pip install -e ".[hf,video]"` before running Hugging Face
datasets or embedding models. `embedding=embeddinggemma` requires Hugging Face
Hub access and local Hugging Face authentication before cache generation.

See [docs/EMBEDDINGS_AND_DATASETS.md](docs/EMBEDDINGS_AND_DATASETS.md).

## Stage Entry Points

Advanced users can run stages directly:

```bash
python cache_embeddings.py dataset=tiny_sentiment
python train_cebra.py dataset=tiny_sentiment trajectory_analysis.enabled=true
python visualize_trajectory.py dataset=tiny_sentiment trajectory_analysis.enabled=true
python visualize_trajectory_cinematic.py dataset=tiny_sentiment cinematic_render.enabled=true
```

Hugging Face examples:

```bash
python cache_embeddings.py dataset=dair-ai embedding=bert
python cache_embeddings.py dataset=ag_news embedding=sentence_bert
python cache_embeddings.py dataset=dair-ai embedding=embeddinggemma
```

The one-command tutorial script is preferred because it passes the complete
set of small CPU-friendly overrides and locates the generated artifact path.

## Cache And Artifacts

The public cache implementation is local and neutral:

- embedding cache default: `artifacts/cache/embeddings`
- CEBRA model default: `artifacts/models`
- override cache root: `CEBRA_NLP_CACHE_DIR`
- override model root: `CEBRA_NLP_MODEL_DIR`

CEBRA-nlp-public uses `labenv-embedding-cache>=0.3.2` as the standard
self-describing NPZ writer and validator for newly generated embedding caches.
The adapter does not change the public cache directory, does not add new
environment variables, and does not expose external execution profiles. The
built-in neutral cache writer remains as a fallback for source checkouts where
the dependency is intentionally removed.

Generated artifacts are ignored by default.

## Validation

This repository was prepared so validation can run in a separate clean
environment. See [docs/EXTERNAL_TESTING.md](docs/EXTERNAL_TESTING.md).

## License

MIT

This repository is a downstream project built on CEBRA. CEBRA itself is
developed by the upstream CEBRA authors and released separately; see the
official repository and license:
https://github.com/AdaptiveMotorControlLab/CEBRA
