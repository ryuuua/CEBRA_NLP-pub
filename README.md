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

### Kaggle データセットを使用する

Kaggle の階層型テキスト分類データセットを利用する場合は、データを
`data/kaggle/hierarchical-text-classification` に配置し、
以下のようにデータセット設定を切り替えます:

```bash
python main.py dataset=hierarchical_text_classification
```

`conf/paths/default.yaml` の `kaggle_data_dir` を変更することで、データの
保存場所をカスタマイズできます。


## Conditional Modes

`cebra.conditional` はラベルの条件付け方法を指定します。利用可能なモードは以下のとおりです。

- `none`: 条件付けなし
- `discrete`: 離散ラベルを使用
- `custom`: 任意の条件データを使用


