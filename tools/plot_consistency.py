# -*- coding: utf-8 -*-
"""
CSV から「2つのモデル × (train/valid) を、dim=2..7 で可視化」し、
PNG を出力します。valid は視覚的な「波線」で描画します
（ピクセル空間の微小な正弦オフセットを付与）。

前提となる CSV 列名（本コードの既定値）:
- cebra.output_dim            : 出力次元（2..7）
- cebra.params.distance       : モデル名（例: euclidean / cosine）
- consistency_score_train     : train の値
- consistency_score_valid     : valid の値

必要に応じて下の「設定」ブロックを書き換えてください。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 設定（必要なら編集） ==========================================
SRC = "/Users/ryua/Code/CEBRAEMO/CEBRA_NLP-main/Consisitency check - bert.csv"   # 入力 CSV のパス（UTF-8 / UTF-8-SIG 推奨）
COL_DIM = "cebra.output_dim"
COL_MODEL = "cebra.params.distance"
COL_TRAIN = "consistency_score_train"
COL_VALID = "consistency_score_valid"

OUT_WAVY = "consistency_wavy_python.png"     # train=実線, valid=波線（視覚効果）
OUT_DASH = "consistency_dashed_python.png"   # 参考：valid=破線

FIG_SIZE = (4.8, 3.4)     # 図サイズ（インチ）
DPI_MAIN = 600            # PNG 解像度
XTICKS_INTEGER = True     # x 目盛りを整数の dim に固定する
# ====================================================================


def build_wavy_path(ax, x, y,
                    amplitude_px=1.6,
                    cycles_per_segment=2,
                    min_samples_per_segment=50,
                    px_per_sample=3):
    """
    折れ線の各セグメントに対し、表示座標（ピクセル）空間で微小な正弦オフセットを
    直交方向に加えて「波線風」にした座標列 (xw, yw) を返す。
    値自体は変えず、線の見た目だけをわずかに揺らすアプローチ。
    """
    # まずはデータを表示座標へ変換
    xy_disp = ax.transData.transform(np.column_stack([x, y]))
    xs, ys = xy_disp[:, 0], xy_disp[:, 1]

    Xw_parts, Yw_parts = [], []
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L == 0:
            continue

        # セグメントに直交する単位法線
        nx, ny = (-dy / L, dx / L)

        # サンプリング点数（約 px_per_sample ピクセルごと）。最低サンプル数も保証
        n = max(min_samples_per_segment, int(L / px_per_sample))
        t = np.linspace(0.0, 1.0, n, endpoint=True)

        # セグメント上の基準線（まっすぐ）
        xb = x0 + dx * t
        yb = y0 + dy * t

        # 正弦オフセット
        k = cycles_per_segment  # 1セグメント当たりの波の数
        offset = amplitude_px * np.sin(2 * np.pi * k * t)

        xw = xb + nx * offset
        yw = yb + ny * offset

        # セグメント継ぎ目での重複を避ける
        if i > 0:
            xw = xw[1:]
            yw = yw[1:]

        Xw_parts.append(xw)
        Yw_parts.append(yw)

    if not Xw_parts:
        return x, y

    Xw = np.concatenate(Xw_parts)
    Yw = np.concatenate(Yw_parts)

    # データ座標に戻す
    xw_data, yw_data = ax.transData.inverted().transform(
        np.column_stack([Xw, Yw])
    ).T
    return xw_data, yw_data


def load_and_prepare(src_path):
    # BOM つき CSV にも対応
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(src_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"CSV の読み込みに失敗しました: {src_path}")

    # 必要列のみ
    need = [COL_DIM, COL_MODEL, COL_TRAIN, COL_VALID]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"列が見つかりません: {c}（CSVに存在する列: {list(df.columns)}）")
    df = df[need].copy()

    # 型を揃える
    df[COL_DIM] = pd.to_numeric(df[COL_DIM], errors="coerce").astype("Int64")
    df[COL_TRAIN] = pd.to_numeric(df[COL_TRAIN], errors="coerce")
    df[COL_VALID] = pd.to_numeric(df[COL_VALID], errors="coerce")

    df = df.dropna(subset=[COL_DIM, COL_MODEL, COL_TRAIN, COL_VALID])
    return df


def plot_pngs(df):
    # モデル（距離）の順序を CSV 出現順で確定し、2つまで使う
    models = list(dict.fromkeys(df[COL_MODEL].astype(str).tolist()))
    if len(models) > 2:
        models = models[:2]

    # モデルごとの系列を用意
    series_by_model = {}
    for m in models:
        sub = df[df[COL_MODEL] == m].sort_values(COL_DIM)
        x = sub[COL_DIM].astype(int).to_numpy()
        y_tr = sub[COL_TRAIN].to_numpy()
        y_va = sub[COL_VALID].to_numpy()
        series_by_model[m] = (x, y_tr, y_va)

    # 軸レンジ（あとで波線を計算する前に固定しておく）
    all_x = np.concatenate([series_by_model[m][0] for m in models])
    all_y = np.concatenate([np.concatenate([series_by_model[m][1], series_by_model[m][2]]) for m in models])
    x_min, x_max = int(all_x.min()), int(all_x.max())
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.04 * (y_max - y_min if y_max > y_min else 1.0)
    y_min, y_max = y_min - pad, y_max + pad

    # ========= 1) valid を波線（視覚効果）で描画 =========
    fig = plt.figure(figsize=FIG_SIZE, dpi=300)
    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 事前にレンダリング情報を確定
    fig.canvas.draw()

    # train（実線）を先に描画
    for idx, m in enumerate(models):
        x, y_tr, _ = series_by_model[m]
        ax.plot(
            x, y_tr,
            linestyle="-",
            marker=("o" if idx == 0 else "s"),
            label=f"{m} (train)",
        )

    # valid（波線風）
    for idx, m in enumerate(models):
        x, _, y_va = series_by_model[m]
        xw, yw = build_wavy_path(ax, x, y_va,
                                 amplitude_px=1.6,      # 波の振幅（ピクセル）
                                 cycles_per_segment=2)  # セグメントあたりの波数
        ax.plot(xw, yw, linestyle="-", label=f"{m} (valid)")
        # データ点位置を明示
        ax.plot(x, y_va, linestyle="None",
                marker=("o" if idx == 0 else "s"),
                markerfacecolor="none")

    ax.set_xlabel("Output dimension")
    ax.set_ylabel("consistency score")
    if XTICKS_INTEGER:
        ax.set_xticks(sorted(set(all_x.tolist())))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_WAVY, dpi=DPI_MAIN)
    plt.close(fig)

    # ========= 2) 代替：valid を破線で描画 =========
    fig2 = plt.figure(figsize=FIG_SIZE, dpi=300)
    ax2 = plt.gca()
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    for idx, m in enumerate(models):
        x, y_tr, y_va = series_by_model[m]
        ax2.plot(x, y_tr, linestyle="-", marker=("o" if idx == 0 else "s"), label=f"{m} (train)")
        ax2.plot(x, y_va, linestyle="--", marker=("o" if idx == 0 else "s"), label=f"{m} (valid)")

    ax2.set_xlabel("Output dimension")
    ax2.set_ylabel("consistency score")
    if XTICKS_INTEGER:
        ax2.set_xticks(sorted(set(all_x.tolist())))
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DASH, dpi=DPI_MAIN)
    plt.close(fig2)

    print(f"[OK] 保存: {OUT_WAVY}")
    print(f"[OK] 保存: {OUT_DASH}")


def main():
    df = load_and_prepare(SRC)
    plot_pngs(df)


if __name__ == "__main__":
    main()
