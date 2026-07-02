# Environment Reproducibility

Use `pyproject.toml` as the single source of truth for Python dependencies.
Docker and uv should both consume that same project metadata.

## Recommended Contract

- `pyproject.toml`: declared runtime dependencies and optional extras.
- `uv`: local Python environment creation, dependency resolution, and command
  execution.
- Dockerfile: system packages, CPU runtime baseline, and the same Python extras
  installed inside a container.
- `.python-version`: Python version hint for local tools.

The standard public extras are:

```bash
uv sync --extra hf --extra video
```

On Linux, uv is configured to resolve `torch` from the PyTorch CPU wheel index.
The Dockerfile uses the same CPU runtime baseline.

For validation:

```bash
uv sync --extra dev --extra hf --extra video
```

Run the tutorial locally with uv:

```bash
uv run python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

Run the same path through Docker:

```bash
docker build -t cebra-nlp-public .
docker run --rm \
  -v "$PWD/runs:/app/runs" \
  cebra-nlp-public \
  python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

## Locking Policy

For releases, create and commit `uv.lock` from a clean environment:

```bash
uv lock
```

After `uv.lock` is committed, local validation should use:

```bash
uv sync --locked --extra dev --extra hf --extra video
```

Docker can also be tightened to `uv sync --locked` once the lockfile is part of
the repository. Until then, Docker and uv are aligned by `pyproject.toml`, but
not fully pinned.

## Credentials

Do not store Hugging Face credentials in this repository. For
`embedding=embeddinggemma`, authenticate the execution environment with
Hugging Face before running cache generation.
