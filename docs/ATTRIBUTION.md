# Attribution

This repository is intentionally and explicitly based on CEBRA.

CEBRA stands for Consistent EmBeddings of high-dimensional Recordings using
Auxiliary variables. The original method, software, and scientific framing are
from the CEBRA authors and maintainers. This repository is a downstream public
NLP tutorial pipeline that uses the public `cebra` Python package and builds
additional NLP-oriented configuration, local cache handling, and trajectory
video rendering wrappers around that foundation.

This repository is not an official CEBRA release.

## Upstream CEBRA

- Official repository: https://github.com/AdaptiveMotorControlLab/CEBRA
- Project website and documentation: https://cebra.ai/
- Python package dependency used here: `cebra==0.6.1`

## Paper To Cite

Please cite the original CEBRA paper when using this repository:

```bibtex
@article{schneider2023cebra,
  title = {Learnable latent embeddings for joint behavioural and neural analysis},
  author = {Schneider, Steffen and Lee, Jin Hwa and Mathis, Mackenzie Weygandt},
  journal = {Nature},
  volume = {617},
  pages = {360--368},
  year = {2023},
  doi = {10.1038/s41586-023-06031-6},
  url = {https://doi.org/10.1038/s41586-023-06031-6}
}
```

## What This Repository Adds

- NLP dataset and embedding configs for a small public tutorial path.
- A neutral local embedding cache implementation for this public cutout.
- CPU-oriented Docker and uv environment instructions.
- Trajectory and cinematic MP4/GIF/PNG rendering entry points.

All credit for the CEBRA method and upstream software belongs to the CEBRA
authors and maintainers.
