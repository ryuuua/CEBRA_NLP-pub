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


### TREC データセットでの実行

質問分類タスク [TREC](https://huggingface.co/datasets/trec) を利用する場合は、以下のようにデータセットを切り替えます:

```bash
python main.py dataset=trec
```

### Kaggle データセットを使用する

Kaggle のデータセットを利用する場合は、`dataset.kaggle_handle` に
対象データセットのハンドルを指定します。`kagglehub` によりデータセットは
自動的にダウンロードされます:

```bash
python main.py dataset=hierarchical_text_classification dataset.kaggle_handle=kashnitsky/hierarchical-text-classification
```

デフォルトでは `kagglehub` のキャッシュディレクトリに保存されます。保存場所を
変更したい場合は `paths.kaggle_data_dir` を設定してください:

```bash
python main.py dataset=hierarchical_text_classification dataset.kaggle_handle=kashnitsky/hierarchical-text-classification paths.kaggle_data_dir=/path/to/data
```

オフライン環境や独自のディレクトリを使用する場合のみ、データセットを
`paths.kaggle_data_dir` に手動で配置する必要があります。

任意の Kaggle データセットを自動ダウンロードして使用する場合は、`dataset.kaggle_handle` に Kaggle のハンドル (`<user>/<dataset>`) を指定します。`dataset.source=kaggle` とあわせて、必要に応じてテキスト列やラベル列を設定してください。

```bash
python main.py dataset=my_kaggle_run \
    dataset.source=kaggle \
    dataset.kaggle_handle=user/dataset_name \
    dataset.text_column=text \
    dataset.label_column=label
```

### マルチラベル分類

複数のラベル列を持つデータセットでは、`multi_label` フラグを有効にし、`label_columns` に対象の列を指定することでマルチラベル分類に対応できます。

```bash
python main.py dataset=my_dataset \
    dataset.multi_label=true \
    dataset.label_columns='["label_a", "label_b"]'
```

### MSE Loss Targets

MSE 損失を使用する場合、ラベルは `cebra.output_dim` と同じ次元を持つ
ベクトルである必要があります。整数ラベルを与えた場合は自動的に
ワンホットベクトルに変換されます。ラベルの最大値からクラス数を自動
推定し (`num_classes = int(labels.max()) + 1`)、`cebra.output_dim` が一致
しない場合は警告とともにこの値に更新されます。

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



## Conditional Modes

`cebra.conditional` はラベルの条件付け方法を指定します。利用可能なモードは以下のとおりです。

- `none`: 条件付けなし
- `discrete`: 離散ラベルを使用
- `custom`: 任意の条件データを使用


