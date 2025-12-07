# smaとボラティリティの関係性を分析
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../..')
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from sqlalchemy import create_engine

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# %%
# ① データベースより読み込み
# バックテスト結果
df_bt = pd.read_sql(
    "SELECT symbol, short, long, sharpe, max_dd, cagr, profit_factor, trades "
    "FROM backtest_results.sma_cross "
    "WHERE cagr IS NOT NULL", engine)

# ボラティリティ
df_atr = pd.read_sql(
    "SELECT date, symbol, atr_14 "
    "FROM calculated_metrics.technical_indicators "
    "WHERE atr_14 IS NOT NULL", engine)

# 日足データ
df_close = pd.read_sql(
    "SELECT date, symbol, close "
    "FROM fmp_data.daily_prices "
    "WHERE close > 0", engine)       # 0 を除外

# ② ボラ+日足JOIN → 比率計算
df_vol = (
    df_atr.merge(df_close, on=["date", "symbol"])
          .assign(atr_ratio=lambda d: d.atr_14 / d.close)
          .groupby("symbol")
          .agg(atr_ratio=("atr_ratio", "mean"),
               atr_median=("atr_ratio", "median"))
          .reset_index()
)

# ③ バックテスト結果とマージ
df     = df_bt.merge(df_vol, on="symbol", how="inner")

# %%



# ボラティリティを帯に分ける
q1, q2, q3 = df['atr_ratio'].quantile([0.25, 0.5, 0.75])
def band(x):
    if x < q1: return 'Low'
    if x < q3: return 'Mid'
    return 'High'

df['vol_band'] = df['atr_ratio'].apply(band)
# %%




# Step 3: ピボットで”帯 * SMA” パフォーマンス比較
pivot = (
    df.groupby(['vol_band', 'short', 'long'])['sharpe']
    .mean()
    .unstack('vol_band') #列：Low/Mid/High
    .reset_index()
) 

# Heatmap 例(Low ボラを可視化)
import plotly.express as px
fig = px.density_heatmap(
    pivot, x='short', y='long',
    z='Low', color_continuous_scale='RdBu',
    title='Sharpe (Low Volatility)'
)
fig.show()


# %%
# Heatmap High可視化
fig = px.density_heatmap(
    pivot, x='short', y='long',
    z='High', color_continuous_scale='RdBu',
    title='Sharpe (High Volatility)'
)
fig.show()


# %%
# Heatmap Mid可視化
fig = px.density_heatmap(
    pivot, x='short', y='long',
    z='Mid', color_continuous_scale='RdBu',
    title='Sharpe (Mid Volatility)'
)
fig.show()


# %%
# df['trades']が10以下の銘柄除外
df_filtered = df[df['trades'] > 10]

# sharpの代わりにprofit_factorを使用して帯を作る
pivot_profit = (
    df_filtered.groupby(['vol_band', 'short', 'long'])['profit_factor']
    .mean()
    .unstack('vol_band')
    .reset_index()
)

# Heatmap Profit Factor可視化
fig = px.density_heatmap(
    pivot_profit, x='short', y='long',
    
    
    z='Low', color_continuous_scale='RdBu',
    title='Profit Factor (Low Volatility)'
)
fig.show()

# %%
# Mid 可視化
fig = px.density_heatmap(
    pivot_profit, x='short', y='long',
    z='Mid', color_continuous_scale='RdBu',
    title='Profit Factor (Mid Volatility)'
)
fig.show()

# %%
# High 可視化
fig = px.density_heatmap(
    pivot_profit, x='short', y='long',
    z='High', color_continuous_scale='RdBu',
    title='Profit Factor (High Volatility)'
)
fig.show()
# %%

# ヒートマップから候補SMAペアを抜き出す
# ボラ帯×SMA の集計を整形 
agg = (
    df.groupby(['vol_band', 'short', 'long'])
    .agg(sharpe=('sharpe', 'mean'),
         pf =('profit_factor', 'mean'),
         trades =('symbol', 'size'))
    .reset_index()
    .query('trades >= 10')
)

# 　ボラ帯ごとに　TOP5を抜き出す
topN = (agg.sort_values(['vol_band', 'sharpe'], ascending=[True, False])
        .groupby('vol_band')
        .head(5)
)
topN
# %%
# 統計的篩にかける
# 銘柄ごとに「ベース期間 vs 候補期間でSharpe差を取る
# ベース＝固定 20/50と仮定
BASE_S = 10; BASE_L = 30

def subset(df_, s, l):
    return df_.query('short == @s and long == @l')[['symbol', 'sharpe']]

df_base = subset(df, BASE_S, BASE_L)
df_cand = subset(df, 14, 50) #例：Mid帯で濃かったペアなど

df_compare = (df_base.merge(df_cand, on='symbol', suffixes=('_base', '_cand')).dropna())

from scipy.stats import wilcoxon #ペア比較のノンバラ検定
stat, p = wilcoxon(df_compare['sharpe_cand'], df_compare['sharpe_base'])
print(f'Sharpe 差の検定結果： stat={stat:.2f}, p={p:.4f}')
# %%

# 前準備
BASE_S, BASE_L = 5, 25
df_base = (df.query('short == @BASE_S and long ==@BASE_L')
           .loc[:, ['symbol', 'sharpe']]
           .rename(columns={'sharpe': 'sharpe_base'}))

# 候補ペアを作成
cand_pairs = [(s, l) for s in range(10, 21)
              for l in range(20, 41)
              if s < l]
results = []

for cs, cl in cand_pairs:
    df_cand = (df.query("short == @cs and long == @cl")
                 .loc[:, ["symbol", "sharpe"]]
                 .rename(columns={"sharpe": "sharpe_cand"}))

    merged = df_base.merge(df_cand, on="symbol").dropna()
    if merged.empty or len(merged) < 10:                 # 銘柄少なすぎ除外
        continue

    diff = merged["sharpe_cand"].mean() - merged["sharpe_base"].mean()
    stat, p = wilcoxon(merged["sharpe_cand"], merged["sharpe_base"])

    results.append({"short": cs, "long": cl, "diff": diff,
                    "p": p, "n": len(merged)})

df_cmp_sharpe = (pd.DataFrame(results)
            .query("p < 0.05")              # 有意なものだけ残す
            .sort_values("diff", ascending=False)
            .reset_index(drop=True))

print(df_cmp_sharpe.head(10))  # ベスト 10 を表示
# %%
from scipy.stats import wilcoxon
import pandas as pd

# ---------------------------
# 0) 基準ペア
# ---------------------------
BASE_S, BASE_L = 5, 25
df_base = (df.query("short == @BASE_S and long == @BASE_L")
             .loc[:, ["symbol", "profit_factor"]]
             .rename(columns={"profit_factor": "pf_base"}))

# ---------------------------
# 1) 候補ペアリスト
# ---------------------------
cand_pairs = [(s, l) for s in range(10, 21)      # short 10-20
                        for l in range(20, 41)    # long  20-40
                        if s < l]

records = []

# ---------------------------
# 2) ループで比較
# ---------------------------
for cs, cl in cand_pairs:
    df_cand = (df.query("short == @cs and long == @cl")
                 .loc[:, ["symbol", "profit_factor"]]
                 .rename(columns={"profit_factor": "pf_cand"}))

    merged = df_base.merge(df_cand, on="symbol").dropna()
    if len(merged) < 10:           # 共通銘柄が 10 未満ならスキップ
        continue

    diff = merged["pf_cand"].mean() - merged["pf_base"].mean()
    stat, p = wilcoxon(merged["pf_cand"], merged["pf_base"])

    records.append({"short": cs, "long": cl,
                    "diff_pf": diff, "p_pf": p, "n": len(merged)})

# ---------------------------
# 3) 結果を並べ替え
# ---------------------------
df_cmp_pf = (pd.DataFrame(records)
            .query("p_pf < 0.05")          # PF 改善が有意
            .sort_values("diff_pf", ascending=False)
            .reset_index(drop=True))

print(df_cmp_pf.head(10))



# %%
# (1) すでに計算した Sharpe・PF の比較結果を df_sh, df_pf とする
df_sh = df_cmp_sharpe.rename(columns={"diff": "diff_sharpe", "p": "p_sharpe"})
df_pf = df_cmp_pf.rename(columns   ={"diff_pf": "diff_pf",    "p_pf": "p_pf"})

# (2) MaxDD の diff を追加で計算（Sharpe のループ中でやっていれば不要）
maxdd_records = []
for cs, cl in cand_pairs:
    df_base_dd = subset(df, 5, 25, "max_dd")
    df_cand_dd = subset(df, cs, cl, "max_dd")
    merged_dd  = df_base_dd.merge(df_cand_dd, on="symbol", suffixes=("_base","_cand"))
    if len(merged_dd) < 10: continue
    diff_dd = merged_dd["max_dd_val_cand"].mean() - merged_dd["max_dd_val_base"].mean()
    maxdd_records.append({"short": cs, "long": cl, "diff_dd": diff_dd})

df_dd = pd.DataFrame(maxdd_records)

# (3) 3 つを JOIN
df_all = (df_sh.merge(df_pf, on=["short","long","n"])
                 .merge(df_dd, on=["short","long"]))
# %%


# フィルタ：Sharpe/PF がともに有意に改善
mask = (
    (df_all["diff_sharpe"] > 0) & (df_all["p_sharpe"] < 0.05) &
    (df_all["diff_pf"]     > 0) & (df_all["p_pf"]     < 0.05)
)
df_ok = df_all[mask].copy()
print(f"候補数 after filter: {len(df_ok)}")
# %%

# スコアリング＆ランキング
# Sharpe差とPF差：大きいほどいい
# MaxDD差：負の方がいい
from scipy.stats import zscore

cols_pos = ["diff_sharpe", "diff_pf"]
cols_neg = ["diff_dd"]

# 正方向 = z、負方向 = −z で統一
for c in cols_pos:
    df_ok[f"z_{c}"] = zscore(df_ok[c])
for c in cols_neg:
    df_ok[f"z_{c}"] = -zscore(df_ok[c])

df_ok["score"] = df_ok[[f"z_{c}" for c in cols_pos + cols_neg]].sum(axis=1)
df_rank = df_ok.sort_values("score", ascending=False).reset_index(drop=True)

print(df_rank.head(10))      # ← 最上位ペアが “最終案”
best_pair = df_rank.loc[0, ["short","long"]].tolist()
print("\n>>>> 最終ベスト SMA =", best_pair)
# %%
# ヒートマップで総合スコアを可視化
import plotly.express as px

heat = (df_ok.pivot(index="long", columns="short", values="score")
              .sort_index(ascending=False))

fig = px.imshow(
    heat, color_continuous_scale="Viridis", text_auto=".2f",
    title="Total Score ( Sharpe↑  PF↑  DD↓ )"
)
fig.update_xaxes(side="top")
fig.show()
# 2025-05-02>>>> 最終ベスト SMA = [20.0, 40.0]
# %%