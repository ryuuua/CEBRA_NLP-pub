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

ハイパーパラメータスイープの実行例:

```bash
python main.py -m hpt=my_sweep
```

## Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking.
Configure your project, run name, and optional entity in `conf/config.yaml`
under the `wandb` section. Runs are initialized automatically by the
training scripts and metrics, parameters, and artifacts are logged to W&B.

複数ランを同じグループにまとめるには `group` 引数を設定します:

```python
from hydra.core.hydra_config import HydraConfig
run = wandb.init(
    project=cfg.wandb.project,
    entity=cfg.wandb.entity,
    name=HydraConfig.get().job.name,
    group=HydraConfig.get().job.name,
    config=OmegaConf.to_container(cfg, resolve=True),
)
```

実験結果のファイルを W&B の Artifact として保存する例:

```python
artifact = wandb.Artifact("results", type="analysis")
artifact.add_dir(output_dir)
run.log_artifact(artifact)
```

W&B 上で結果を確認する手順:

1. 上記のように Artifact をログする。
2. [W&B](https://wandb.ai/) にアクセスし、対象の run または group を開く。
3. `Artifacts` タブでアップロードされたファイルを確認できる。
4. `Group` でまとめた複数 run の結果を比較できる。

## Reproducibility

`evaluation.random_state` で指定した値を用いて Python の `random`、NumPy、PyTorch (CUDA が利用可能な場合は `torch.cuda` も含む) の乱数シードを設定し、結果の再現性を高めています。
