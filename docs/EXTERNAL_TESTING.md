# External Testing

Run these checks in a clean environment after the scaffold is copied or
published. The local creation pass intentionally does not run pytest, Docker
builds, or tutorial E2E execution.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,hf,video]"
```

Equivalent uv path:

```bash
uv sync --extra dev --extra hf --extra video
uv run pytest -q tests/test_cache_utils.py tests/test_epoch_trajectory.py
```

## Targeted Tests

```bash
pytest -q tests/test_cache_utils.py tests/test_embedding_cache_adapter.py tests/test_epoch_trajectory.py
pytest -q tests/test_epoch_trajectory_e2e.py tests/test_trajectory_analysis_e2e.py tests/test_trajectory_cinematic_viz.py
```

## labenv_embedding_cache Adapter

The public repo depends on `labenv-embedding-cache>=0.3.2`, so a normal pip or
uv install should pull the published cache library before running the adapter
test:

```bash
pytest -q tests/test_embedding_cache_adapter.py
```

The fallback path is covered by the same test module through monkeypatched
import behavior; do not remove the fallback because it keeps source checkouts
usable when the dependency is intentionally absent.

## Docker Tutorial

```bash
docker build -t cebra-nlp-public .
docker run --rm \
  -v "$PWD/runs:/app/runs" \
  cebra-nlp-public \
  python scripts/run_tutorial_video.py --workdir runs/tutorial_video --force
```

## Artifact Acceptance

Confirm the tutorial produced:

```text
runs/tutorial_video/models/**/tutorial_video/label_drift_trajectory/
  label_drift_beauty_master_3d.mp4
  label_drift_beauty_master_preview_3d.gif
  label_drift_beauty_master_3d.png
```

Confirm the MP4 resolution:

```bash
MP4_PATH="$(find runs/tutorial_video/models -path '*/tutorial_video/label_drift_trajectory/label_drift_beauty_master_3d.mp4' -print -quit)"
test -n "$MP4_PATH"
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height \
  -of csv=p=0:s=x \
  "$MP4_PATH"
```

Expected output:

```text
640x360
```

## Public-Safety Scan

Run this after any future file copy from a source workspace. Keep project-specific
denylist terms outside the repository:

```bash
test -s /tmp/cebra-public-denylist.txt
rg -n -f /tmp/cebra-public-denylist.txt .
rg -n "AGENTS[.]md|[.]codex|[.]mcp|[A-Za-z0-9_]*([T]OKEN|[P]ASSWORD|[S]ECRET|API[_-]?[Kk]EY)[A-Za-z0-9_]*" .
```

Both `rg` commands should return no matches.
