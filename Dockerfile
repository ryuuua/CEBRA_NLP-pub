FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        libegl1 \
        libgl1 \
        libgomp1 \
        libgles2 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY conf ./conf
COPY examples ./examples
COPY scripts ./scripts
COPY src ./src
COPY tools ./tools
COPY cache_embeddings.py train_cebra.py visualize.py visualize_trajectory.py visualize_trajectory_cinematic.py ./

RUN python -m pip install --upgrade pip uv \
    && uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
    && uv pip install --system --no-cache -e ".[hf,video]"

CMD ["python", "scripts/run_tutorial_video.py", "--workdir", "runs/tutorial_video", "--force"]
