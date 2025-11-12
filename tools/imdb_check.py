# imdb_check.py （動く版）
from datasets import load_dataset
import pandas as pd

# 1) 元データざっと確認
ds_train, ds_test = load_dataset("imdb", split=["train", "test"])
raw_df = pd.concat([ds_train.to_pandas(), ds_test.to_pandas()], ignore_index=True)
print(raw_df.head())

# 2) リポジトリの前処理を実行（Hydra で compose）
# 2) リポジトリの前処理を実行（Hydra で compose → AppConfig に落とす）
from hydra import initialize, compose
from omegaconf import OmegaConf
from src.config_schema import AppConfig
from src.data import load_and_prepare_dataset

with initialize(version_base=None, config_path="conf"):
    # defaults を展開しつつ dataset=imdb で上書き
    cfg = compose(config_name="config", overrides=["dataset=imdb"])

# Hydra メタキーを落としてから schema にマージ
for k in ("hydra", "defaults"):
    if k in cfg:
        cfg.pop(k)

# AppConfig の構造（入れ子の dataclass 定義）を土台にしてマージ
schema = OmegaConf.structured(AppConfig)         # ← AppConfig の“型”を持つ DictConfig
cfg_on_schema = OmegaConf.merge(schema, cfg)     # ← 型に沿ってはめ込む
app_cfg: AppConfig = OmegaConf.to_object(cfg_on_schema)  # ← 本物の dataclass ツリーへ変換

texts, labels, time_idx = load_and_prepare_dataset(app_cfg)

# 3) 変換後サンプルの確認
for t, l in zip(texts[:30], labels[:30]):
    print(l, t[:80])

    # ==== 4) CSV に保存 ====
import os
from datetime import datetime
from omegaconf import OmegaConf

os.makedirs("outputs/csv", exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# (A) 元データ（train+testを結合した raw_df は既に上で作成済み）
raw_path = f"outputs/csv/imdb_raw_{stamp}.csv"
raw_df.to_csv(raw_path, index=False, encoding="utf-8")
print(f"[saved] {raw_path}")

# (B) 前処理後（texts, labels, time_idx）を1本のCSVに
# time_idx が無い/長さが合わないケースにも耐える
proc = {"text": texts, "label": labels}
try:
    if time_idx is not None and len(time_idx) == len(texts):
        proc["time_idx"] = time_idx
except Exception:
    pass

import pandas as pd
proc_df = pd.DataFrame(proc)
proc_path = f"outputs/csv/imdb_processed_{stamp}.csv"
proc_df.to_csv(proc_path, index=False, encoding="utf-8")
print(f"[saved] {proc_path}")

# (C) 参考：使用した設定も保存（再現性のため）
try:
    cfg_yaml_path = f"outputs/csv/config_snapshot_{stamp}.yaml"
    # cfg_on_schema または cfg（あなたのスクリプト内の最終的な設定オブジェクト）を使ってください
    # 例: OmegaConf.save(cfg_on_schema, cfg_yaml_path) でもOK
    from pathlib import Path
    # ここでは compose した cfg を保存する例（適宜変えてください）
    OmegaConf.save(cfg, cfg_yaml_path)
    print(f"[saved] {cfg_yaml_path}")
except Exception as e:
    print(f"[warn] config snapshot not saved: {e}")