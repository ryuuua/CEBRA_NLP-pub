# CEBRA_NLP

このプロジェクトでは CEBRA を用いた NLP 実験を行います。

## 実行方法

単一 GPU での実行:

```bash
python main.py
```

Mac (CUDA/MPS/CPU) での実行 (自動デバイス検出):

```bash
python macmain.py
# or
python macmainoptimize.py
```

分散学習 (2 GPU) での実行例:

```bash
torchrun --nproc_per_node=2 main.py
```

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.
Configure your project, run name, and optional entity in `conf/config.yaml`
under the `wandb` section. Runs are initialized automatically by the
training scripts and metrics, parameters, and artifacts are logged to W&B.

## Reproducibility

`evaluation.random_state` で指定した値を用いて Python の `random`、NumPy、PyTorch (CUDA が利用可能な場合は `torch.cuda` も含む) の乱数シードを設定し、結果の再現性を高めています。
