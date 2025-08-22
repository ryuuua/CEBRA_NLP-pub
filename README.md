# CEBRA_NLP

このプロジェクトでは CEBRA を用いた NLP 実験を行います。

## 実行方法

単一 GPU での実行:

```bash
python main.py
```

Mac (CPU/MPS) での実行:

```bash
python macmain.py
```

分散学習 (2 GPU) での実行例:

```bash
torchrun --nproc_per_node=2 main.py
```
