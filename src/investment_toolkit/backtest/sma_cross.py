#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMA crossing strategy backtester

Usage:
    python -m src.backtest.sma_cross --symbols AAPL NVDA VTI TSM 3826.T 4847.T 7163.T MSFT DFH TSLA 1514.T 4776.T --start 2010-01-01 --end 2025-04-30 --grid "5,25;5,50;7,21;7,40;10,20;10,35;10,40;14,28;14,50;20,50;20,25;20,35;20,40;20,60;20,70;25,50;25,70;30,80" --save --heatmap
    
    AAPL NVDA VTI TSM 3826.T 4847.T 7163.T MSFT DFH TSLA 1514.T 4776.T
    5,25;5,30;5,35;5,40;5,50;5,60;7,21;7,28;7,35;7,40;7,50;7,60;10,20;10,25;10,30;10,35;10,40;10,50;10,60;10,70;14,28;14,35;14,40;14,50;14,60;14,70;20,25;20,30;20,35;20,40;20,50;20,60;20,70;20,80;25,50;25,60;25,70;25,80;30,60;30,70;30,80
"""

import argparse
import os
import sys
import webbrowser
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.io
import plotly.graph_objects as go
from joblib import Parallel, delayed

# プロジェクトのルートディレクトリをPythonのパスに追加（直接実行時のため）
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME


def get_price_data(
    symbols: List[str], start: str, end: str, engine: Any
) -> Dict[str, pd.DataFrame]:
    """
    指定された銘柄の価格データをデータベースから取得

    Args:
        symbols: 銘柄リスト (例: ['AAPL', '7203.T'])
        start: 開始日 (YYYY-MM-DD)
        end: 終了日 (YYYY-MM-DD)
        engine: SQLAlchemyエンジン

    Returns:
        銘柄をキーとし、価格データフレームを値とする辞書
    """
    result = {}

    for symbol in symbols:
        query = text(
            """
            SELECT date, symbol, open, high, low, close, volume
            FROM fmp_data.daily_prices
            WHERE symbol = :symbol
              AND date BETWEEN :start_date AND :end_date
            ORDER BY date ASC
            """
        )

        with engine.connect() as conn:
            df = pd.read_sql_query(
                query, conn, params={"symbol": symbol, "start_date": start, "end_date": end}
            )

        if not df.empty:
            # 日付をインデックスに設定
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            result[symbol] = df

    return result


def backtest_sma(
    df: pd.DataFrame, short: int, long: int, cost: float = 0.002
) -> Dict[str, Any]:
    """
    SMAクロス戦略のバックテストを実行

    Args:
        df: 価格データフレーム (date indexed)
        short: 短期SMAの期間
        long: 長期SMAの期間
        cost: 取引コスト (片道・割合)

    Returns:
        バックテスト結果を含む辞書
    """
    # 必要なカラムが存在するか確認
    if "close" not in df.columns:
        raise ValueError("DataFrameには'close'カラムが必要です")

    # データフレームのコピーを作成
    data = df.copy()

    # SMAを計算
    data[f"sma_{short}"] = data["close"].rolling(window=short).mean()
    data[f"sma_{long}"] = data["close"].rolling(window=long).mean()

    # シグナル生成: 短期SMA > 長期SMA → 1、それ以外 → 0
    data["signal"] = (data[f"sma_{short}"] > data[f"sma_{long}"]).astype(int)

    # ポジション: 翌日の寄付でエントリー
    data["position"] = data["signal"].shift(1).fillna(0).astype(int)

    # 取引フラグ: ポジション変化時に1、それ以外0
    data["trade_flag"] = data["position"].diff().abs()

    # リターン計算
    data["ret"] = data["close"].pct_change()

    # 取引コスト反映後のネットリターン
    data["net_ret"] = data["position"] * data["ret"] - data["trade_flag"] * cost

    # 累積リターン
    data["cum_ret"] = (1 + data["net_ret"]).cumprod()

    # 累積最大値（ドローダウン計算用）
    data["cum_max"] = data["cum_ret"].cummax()
    data["drawdown"] = (data["cum_max"] - data["cum_ret"]) / data["cum_max"]

    # 取引回数
    trades = data["trade_flag"].sum()

    # 勝率とプロフィットファクターの計算
    # 取引フラグが立っている日の翌日のリターンを利用
    # 取引日のポジションに翌日のリターンを掛け合わせる
    trade_days = data[data["trade_flag"] == 1].index
    winning_trades = 0
    
    if len(trade_days) > 0:
        for day in trade_days:
            try:
                next_day = data.index[data.index.get_loc(day) + 1]
                # ポジションが1なら上昇で勝ち、0なら下落で勝ち
                position = data.loc[day, "position"]
                next_return = data.loc[next_day, "ret"]
                
                if (position == 1 and next_return > 0) or (position == 0 and next_return < 0):
                    winning_trades += 1
            except (IndexError, KeyError):
                # 最終日の場合はスキップ
                pass
    
    win_rate = winning_trades / trades if trades > 0 else 0.0

    # ネットリターンの合計（プロフィットファクター用）
    positive_returns = data.loc[data["net_ret"] > 0, "net_ret"].sum()
    negative_returns = abs(data.loc[data["net_ret"] < 0, "net_ret"].sum())
    profit_factor = positive_returns / negative_returns if negative_returns > 0 else float("inf")

    # 有効データ（NaN除去後）
    valid_data = data.dropna(subset=["net_ret", f"sma_{long}"])

    # パフォーマンス統計値
    days = len(valid_data)
    if days > 0:
        cagr = (1 + valid_data["net_ret"]).prod() ** (252 / days) - 1
        sharpe = (
            valid_data["net_ret"].mean() / valid_data["net_ret"].std(ddof=0) * np.sqrt(252)
            if valid_data["net_ret"].std() > 0
            else 0.0
        )
        max_dd = data["drawdown"].max()
        start_date = valid_data.index[0].strftime("%Y-%m-%d")
        end_date = valid_data.index[-1].strftime("%Y-%m-%d")
    else:
        cagr, sharpe, max_dd = 0.0, 0.0, 0.0
        start_date = end_date = "n/a"

    # 結果を辞書で返す
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": trades,
        "start_date": start_date,
        "end_date": end_date,
        "data": data,  # 分析用にデータフレーム全体を含める
    }


def _run_backtest(
    symbol: str, df: pd.DataFrame, short: int, long: int, cost: float
) -> Dict[str, Any]:
    """
    並列処理用のバックテスト関数

    Args:
        symbol: 銘柄シンボル
        df: 価格データフレーム
        short: 短期SMAの期間
        long: 長期SMAの期間
        cost: 取引コスト

    Returns:
        バックテスト結果
    """
    result = backtest_sma(df, short, long, cost)
    return {
        "symbol": symbol,
        "short": short,
        "long": long,
        "cagr": result["cagr"],
        "sharpe": result["sharpe"],
        "max_dd": result["max_dd"],
        "win_rate": result["win_rate"],
        "profit_factor": result["profit_factor"],
        "trades": result["trades"],
        "start_date": result["start_date"],
        "end_date": result["end_date"],
    }


def run_grid_search(
    dfs: Dict[str, pd.DataFrame],
    grid: List[Tuple[int, int]],
    cost: float = 0.002,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    複数の銘柄と期間パラメータでグリッドサーチを実行

    Args:
        dfs: {symbol: dataframe} 形式の価格データ辞書
        grid: (short, long) のパラメータセットのリスト
        cost: 取引コスト
        n_jobs: 並列ジョブ数 (-1 は全コア)

    Returns:
        バックテスト結果のDataFrame（sharpe降順）
    """
    # 銘柄×パラメータの全組み合わせを作成
    tasks = []
    for symbol, df in dfs.items():
        for short, long in grid:
            if short >= long:
                continue  # 短期が長期以上の場合はスキップ
            tasks.append((symbol, df, short, long, cost))

    # 並列実行
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_backtest)(symbol, df, short, long, cost)
        for symbol, df, short, long, cost in tasks
    )

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # sharpeの降順でソート
    if not results_df.empty:
        results_df = results_df.sort_values("sharpe", ascending=False)

    return results_df


def save_results_to_sql(
    df: pd.DataFrame,
    engine: Any,
    table_name: str = "backtest_results.sma_cross",
    if_exists: str = "replace",
) -> None:
    """
    バックテスト結果をSQLデータベースに保存

    Args:
        df: バックテスト結果のDataFrame
        engine: SQLAlchemyエンジン
        table_name: 保存先テーブル名
        if_exists: テーブル存在時の動作 ('replace', 'append', 'fail')
    """
    # タイムスタンプを追加
    df_to_save = df.copy()
    df_to_save["timestamp"] = datetime.now()

    # スキーマとテーブル名を分割
    schema, table = table_name.split(".")

    # スキーマが存在しない場合は作成
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        conn.commit()

    # データをSQLに保存
    df_to_save.to_sql(table, engine, schema=schema, if_exists=if_exists, index=False)

    print(f"{len(df_to_save)}行のデータを{table_name}に保存しました")


def optional_heatmap(
    df: pd.DataFrame, metric: str = "sharpe", out_file: str = "sma_heatmap.html"
) -> None:
    """
    SMAパラメータのヒートマップを生成

    Args:
        df: run_grid_searchの結果DataFrame
        metric: ヒートマップで表示する指標
        out_file: 出力ファイル名
    """
    if df.empty:
        print("データが空のため、ヒートマップを生成できません")
        return

    # 各銘柄ごとにヒートマップを生成
    symbols = df["symbol"].unique()
    fig_dict = {}

    for symbol in symbols:
        symbol_df = df[df["symbol"] == symbol].copy()
        
        # ピボットテーブルを作成
        pivot = symbol_df.pivot_table(
            index="long", columns="short", values=metric, aggfunc="mean"
        )
        
        # ヒートマップを生成
        fig = px.imshow(
            pivot,
            labels=dict(x="Short SMA", y="Long SMA", color=metric.capitalize()),
            x=pivot.columns,
            y=pivot.index,
            color_continuous_scale="RdBu_r",
            title=f"{symbol} SMA Cross {metric.capitalize()} Heatmap",
            aspect="auto",
        )
        
        # 軸とレイアウトの設定
        fig.update_layout(
            xaxis_title="Short SMA",
            yaxis_title="Long SMA",
            coloraxis_colorbar=dict(title=metric.capitalize()),
        )
        
        fig_dict[symbol] = fig
    
    # 結果テーブルをHTML形式に変換
    df_html = df.to_html(index=False, float_format=lambda f: '{:.4f}'.format(f))
    
    # 指標の説明文
    metrics_explanation = """
    <div class="explanation">
        <h2>バックテスト結果の各指標の説明</h2>
        
        <h3>基本情報</h3>
        <ul>
            <li><strong>symbol</strong>: 分析対象の銘柄コード</li>
            <li><strong>short</strong>: 短期移動平均線の期間（日数）。短すぎると過敏に反応し、長すぎると遅れが生じる</li>
            <li><strong>long</strong>: 長期移動平均線の期間（日数）。短すぎるとシグナル過多、長すぎるとシグナル不足になる</li>
        </ul>
        
        <h3>パフォーマンス指標</h3>
        <ul>
            <li><strong>cagr</strong> (Compound Annual Growth Rate): 年率換算複利成長率
                <ul>
                    <li>0.10（10%）以上: 良好</li>
                    <li>0.20（20%）以上: 非常に良好</li>
                    <li>マイナス: 戦略に問題あり</li>
                </ul>
            </li>
            <li><strong>sharpe</strong> (シャープレシオ): リスク調整後リターン（リターン÷リスク）× √252
                <ul>
                    <li>1.0未満: 効率が悪い</li>
                    <li>1.0～2.0: 良好</li>
                    <li>2.0以上: 非常に良好</li>
                    <li>マイナス: リスクに見合わないリターン</li>
                </ul>
            </li>
            <li><strong>max_dd</strong> (最大ドローダウン): 最大の資産価値下落率（高値から安値への下落幅）
                <ul>
                    <li>0.10（10%）未満: 優れた安定性</li>
                    <li>0.20（20%）未満: 一般的な安定性</li>
                    <li>0.30（30%）以上: リスクが高い</li>
                </ul>
            </li>
        </ul>
        
        <h3>トレード効率指標</h3>
        <ul>
            <li><strong>win_rate</strong> (勝率): 勝ったトレードの比率
                <ul>
                    <li>0.50（50%）以上: 一般的に良好</li>
                    <li>0.60（60%）以上: 優れた勝率</li>
                    <li>0.40（40%）未満: 戦略に要調整</li>
                </ul>
            </li>
            <li><strong>profit_factor</strong> (プロフィットファクター): 総利益÷総損失の比率
                <ul>
                    <li>1.0: 収支トントン</li>
                    <li>1.5以上: 良好</li>
                    <li>2.0以上: 非常に良好</li>
                    <li>1.0未満: 損失が利益を上回る</li>
                </ul>
            </li>
            <li><strong>trades</strong> (トレード回数): 期間中の売買回数
                <ul>
                    <li>30回以上: 統計的に有意</li>
                    <li>10回未満: サンプル不足で信頼性に欠ける</li>
                    <li>極端に多い: 取引コストが大きくなりすぎる可能性</li>
                </ul>
            </li>
        </ul>
        
        <h3>結果の解釈方法</h3>
        <ul>
            <li><strong>理想的な組み合わせ</strong>
                <ul>
                    <li>高いCAGR + 高いシャープレシオ: リターンとリスク効率が良い</li>
                    <li>高い勝率 + 高いプロフィットファクター: トレード効率が良い</li>
                    <li>適度なトレード回数: 統計的に信頼性がある</li>
                </ul>
            </li>
            <li><strong>要注意のケース</strong>
                <ul>
                    <li>高いCAGR + 低いシャープレシオ: リスクが高すぎる</li>
                    <li>低いCAGR + 高いシャープレシオ: リターンが少なすぎる</li>
                    <li>高い勝率 + 低いプロフィットファクター: 少額の勝ちが多く大きな負けがある</li>
                    <li>極端に少ないトレード回数: 統計的信頼性に欠ける</li>
                </ul>
            </li>
        </ul>
    </div>
    """
    
    # 単一のHTML出力ファイルにすべてのヒートマップを保存
    if len(fig_dict) == 1:
        # 単一銘柄のHTMLを生成
        single_html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>SMA Cross Strategy Backtest Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; flex-direction: column; }}
        .results {{ margin-bottom: 20px; }}
        .plot {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        h1, h2, h3 {{ color: #333; }}
        .explanation {{ margin-top: 30px; line-height: 1.6; }}
        .explanation ul {{ margin-bottom: 15px; }}
        .explanation li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>SMA Cross Strategy Backtest Results</h1>
    <div class="container">
        <div class="results">
            <h2>結果サマリー</h2>
            {df_html}
        </div>
        <div class="plot">
            {list(fig_dict.values())[0].to_html(full_html=False, include_plotlyjs="cdn")}
        </div>
        {metrics_explanation}
    </div>
</body>
</html>
"""
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(single_html)
    else:
        # 複数銘柄のHTMLを生成
        multi_html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>SMA Cross Strategy Backtest Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ display: flex; flex-direction: column; }}
        .results {{ margin-bottom: 20px; }}
        .plot {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f5f5f5; }}
        h1, h2, h3 {{ color: #333; }}
        .explanation {{ margin-top: 30px; line-height: 1.6; }}
        .explanation ul {{ margin-bottom: 15px; }}
        .explanation li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>SMA Cross Strategy Backtest Results - {metric.capitalize()}</h1>
    <div class="container">
        <div class="results">
            <h2>結果サマリー</h2>
            {df_html}
        </div>
"""
        
        for symbol, fig in fig_dict.items():
            multi_html += f'<div class="plot">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
        
        multi_html += f"""
        {metrics_explanation}
    </div>
</body>
</html>
"""
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(multi_html)
    
    print(f"ヒートマップを{out_file}に保存しました")


def parse_grid(grid_str: str) -> List[Tuple[int, int]]:
    """
    グリッドパラメータをパース

    Args:
        grid_str: "short,long;short,long" 形式の文字列

    Returns:
        (short, long) のタプルのリスト
    """
    grid = []
    pairs = grid_str.split(";")
    for pair in pairs:
        short, long = map(int, pair.split(","))
        grid.append((short, long))
    return grid


def get_random_symbols_from_file(file_path: str, n_symbols: int = 1000) -> List[str]:
    """
    シンボルファイルからランダムに指定数の銘柄を抽出

    Args:
        file_path: シンボルファイルのパス
        n_symbols: 抽出する銘柄数

    Returns:
        抽出された銘柄のリスト
    """
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        if len(symbols) < n_symbols:
            print(f"警告: ファイルには{n_symbols}銘柄未満しかありません。全ての銘柄を使用します。")
            return symbols
        
        import random
        return random.sample(symbols, n_symbols)
    except FileNotFoundError:
        print(f"エラー: シンボルファイル {file_path} が見つかりません。")
        return []


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SMA Cross Backtester")
    parser.add_argument(
        "--symbols_file",
        help="Symbols file path (e.g., symbols.txt). If specified, random symbols will be selected from this file."
    )
    parser.add_argument(
        "--n_symbols",
        type=int,
        default=1000,
        help="Number of random symbols to select from the file (default: 1000)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to backtest (e.g., AAPL 7203.T). Ignored if --symbols_file is specified."
    )
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--grid",
        default="5,25;5,30;5,35;5,40;5,50;5,60;7,21;7,28;7,35;7,40;7,50;7,60;10,20;10,25;10,30;10,35;10,40;10,50;10,60;10,70;14,28;14,35;14,40;14,50;14,60;14,70;20,25;20,30;20,35;20,40;20,50;20,60;20,70;20,80;25,50;25,60;25,70;25,80;30,60;30,70;30,80",
        help="SMA parameter pairs as 'short,long;short,long'"
    )
    parser.add_argument("--cost", type=float, default=0.002, help="Trading cost (default: 0.002)")
    parser.add_argument("--save", action="store_true", help="Save results to database")
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap")
    parser.add_argument(
        "--output", default="reports/sma_heatmap.html", help="Output file for heatmap"
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the heatmap in browser"
    )
    
    args = parser.parse_args()
    
    # シンボルの取得
    if args.symbols_file:
        symbols = get_random_symbols_from_file(args.symbols_file, args.n_symbols)
        if not symbols:
            return
        print(f"ファイルから{len(symbols)}銘柄をランダムに抽出しました")
    elif args.symbols:
        symbols = args.symbols
    else:
        print("エラー: --symbols_file または --symbols のいずれかを指定してください")
        return
    
    # データベース接続
    db_uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_uri)
    
    # データ取得
    print(f"以下の銘柄のデータを取得中: {', '.join(symbols[:5])}... (合計{len(symbols)}銘柄)")
    price_data = get_price_data(symbols, args.start, args.end, engine)
    
    # 空のデータフレームを除外
    price_data = {k: v for k, v in price_data.items() if not v.empty}
    
    if not price_data:
        print("データが取得できませんでした。終了します。")
        return
    
    print(f"有効なデータを取得できた銘柄数: {len(price_data)}")
    
    # グリッドパラメータをパース
    grid = parse_grid(args.grid)
    
    # バックテスト実行
    print(f"グリッドサーチ実行中 ({len(grid)}パラメータセット x {len(price_data)}銘柄)...")
    results = run_grid_search(price_data, grid, args.cost)
    
    # 結果表示
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\nバックテスト結果 (sharpe降順):")
    
    if results.empty:
        print("有効な結果がありません。")
    else:
        display_cols = [
            "symbol", "short", "long", "cagr", "sharpe", "max_dd", 
            "win_rate", "profit_factor", "trades"
        ]
        print(results[display_cols].to_string(index=False))
    
    # 結果をデータベースに保存
    if args.save and not results.empty:
        save_results_to_sql(results, engine)
    
    # ヒートマップ生成
    heatmap_file = None
    if args.heatmap and not results.empty:
        # 出力ディレクトリ確認
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        optional_heatmap(results, "sharpe", args.output)
        heatmap_file = args.output
    
    # ヒートマップをブラウザで開く
    if heatmap_file and not args.no_open:
        heatmap_path = os.path.abspath(heatmap_file)
        print(f"ヒートマップをブラウザで開いています: {heatmap_path}")
        
        # プラットフォームに応じてブラウザを開く方法を選択
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', heatmap_path], check=True)
            else:
                webbrowser.open(f"file://{heatmap_path}")
        except Exception as e:
            print(f"ブラウザでの表示に失敗しました: {e}")
            print("手動でファイルを開いてください")


if __name__ == "__main__":
    main() 