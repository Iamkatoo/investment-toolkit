#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
経済指標ダッシュボード生成スクリプト
データベースから各種経済指標を取得し、分析・スコアリング・可視化を実行する

実行方法:
    python src/analysis/daily_report.py

出力:
    /Users/HOME/Codes/Investment/reports/ ディレクトリに dashboard.html が保存される
    /Users/HOME/Codes/Investment/reports/graphs/ ディレクトリに各種グラフが保存される
"""

import os
import sys
import pandas as pd
import numpy as npk
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
import webbrowser
import subprocess
import plotly.graph_objects as go
from sqlalchemy.engine import Engine
from typing import Tuple, List, Dict, Optional
import requests
import time
import socket
# requests, time, threading - APIサーバー自動起動で使用していたが削除

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内のモジュールをインポート（エラーハンドリング付き）
try:
    from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    print("✅ データベース設定読み込み成功")
except ImportError as e:
    print(f"❌ データベース設定の読み込み失敗: {e}")
    print("🔧 src/utilities/config.py ファイルが存在し、適切な設定があることを確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.scoring_functions import (
        evaluate_economic_indicators,          # 短期 (=3M) 既存
        evaluate_economic_indicators_long      # ★ 追加
    )
    print("✅ スコア計算モジュール読み込み成功")
except ImportError as e:
    print(f"❌ スコア計算モジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/scoring_functions.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.visualize_dashboard import (
        plot_normalized_indices, plot_normalized_indices_6w, plot_normalized_indices_3m, plot_vix_vs_sp500, 
        plot_gold_dollar_yen, plot_gold_dollar_yen_6w, plot_gold_dollar_yen_3m,
        plot_currency_pairs, plot_currency_pairs_6w, plot_currency_pairs_3m, 
        plot_interest_rates, plot_inflation,
        plot_economic_score, plot_economic_subplots
    )
    print("✅ ダッシュボード可視化モジュール読み込み成功")
except ImportError as e:
    print(f"❌ ダッシュボード可視化モジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/visualize_dashboard.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.portfolio_utils import build_portfolio_section, build_alltime_portfolio_section
    print("✅ ポートフォリオモジュール読み込み成功")
except ImportError as e:
    print(f"❌ ポートフォリオモジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/portfolio_utils.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.market_scoring import calculate_combined_score, get_portfolio_symbols
    print("✅ マーケットスコアモジュール読み込み成功")
except ImportError as e:
    print(f"❌ マーケットスコアモジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/market_scoring.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.score_visualization import (
        plot_combined_score, create_score_sparklines, 
        plot_market_score_report, generate_market_score_html
    )
    print("✅ スコア可視化モジュール読み込み成功")
except ImportError as e:
    print(f"❌ スコア可視化モジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/score_visualization.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.score_analysis import generate_top_stocks_report, generate_rsi35_below_report
    print("✅ スコア分析モジュール読み込み成功")
except ImportError as e:
    print(f"❌ スコア分析モジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/score_analysis.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.enhanced_score_analysis import generate_enhanced_top_stocks_report, generate_enhanced_rsi35_report
    print("✅ 拡張スコア分析モジュール読み込み成功")
except ImportError as e:
    print(f"❌ 拡張スコア分析モジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/enhanced_score_analysis.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.watchlist_report import generate_watchlist_report_html, generate_dynamic_watchlist_html, update_watchlist_performance_data, generate_mini_chart_watchlist_html
    print("✅ ウォッチリストレポートモジュール読み込み成功")
except ImportError as e:
    print(f"❌ ウォッチリストレポートモジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/watchlist_report.py の実装を確認してください")
    sys.exit(1)

try:
    from investment_analysis.analysis.daily_ranking_report import generate_daily_ranking_html
    print("✅ 日次ランキングレポートモジュール読み込み成功")
except ImportError as e:
    print(f"❌ 日次ランキングレポートモジュールの読み込み失敗: {e}")
    print("🔧 src/analysis/daily_ranking_report.py の実装を確認してください")
    sys.exit(1)

# trade_journal_report の機能は portfolio_alltime.html と market_score_report.html に統合されました

try:
    from investment_analysis.scoring.validation import ScoringValidator
    print("✅ スコアリング検証モジュール読み込み成功")
except ImportError as e:
    print(f"❌ スコアリング検証モジュールの読み込み失敗: {e}")
    print("🔧 src/scoring/validation.py の実装を確認してください")
    # 検証機能は必須ではないので処理を継続
    ScoringValidator = None

print("🎉 全モジュールのインポート完了")

def build_html(fig, explanation):
    """Plotly 図と説明文を 1 枚の HTML にまとめて返す"""
    # --- 図の本体を取得 ---
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # --- 静的テンプレート ---
    template = (
        """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <style>
    body{font-family:system-ui,sans-serif;margin:0;padding:1rem;}
    .chart-container{max-width:1000px;margin:0 auto;}
    .explanation{font-size:.9rem;line-height:1.6;margin-top:.8rem;color:#333;}
  </style>
</head>
<body>
  <div class="chart-container">
    {plot}
    <p class="explanation">{exp}</p>
  </div>
</body>
</html>"""
    )

    # --- プレースホルダー置換 ---
    return (
        template
        .replace("{plot}", plot_html)
        .replace("{exp}", explanation.strip())
    )


def fetch_top10_appearance_counts(engine, symbols_markets_list):
    """
    指定された銘柄のTop10入り回数を一括取得

    Parameters:
    -----------
    engine : sqlalchemy.engine.Engine
        データベース接続エンジン
    symbols_markets_list : list of tuple
        [(symbol, market), ...] の形式
        例: [('AAPL', 'us'), ('7203.T', 'jp')]

    Returns:
    --------
    dict
        {(symbol, market): count, ...} の形式
        例: {('AAPL', 'us'): 45, ('7203.T', 'jp'): 23}
    """
    from sqlalchemy import text

    if not symbols_markets_list:
        return {}

    # IN句用のプレースホルダーを構築
    placeholders = []
    params = {}
    for idx, (symbol, market) in enumerate(symbols_markets_list):
        placeholders.append(f"(:symbol_{idx}, :market_{idx})")
        params[f'symbol_{idx}'] = symbol
        params[f'market_{idx}'] = market

    in_clause = ', '.join(placeholders)

    query = text(f"""
        SELECT
            symbol,
            market,
            COUNT(*) as top10_count
        FROM backtest_results.score_rankings_v2
        WHERE ranking_scope = 'daily'
          AND ranking_method = 'total_score'
          AND rank <= 10
          AND (symbol, market) IN ({in_clause})
        GROUP BY symbol, market
    """)

    with engine.connect() as conn:
        df_counts = pd.read_sql(query, conn, params=params)

    # DataFrameから辞書に変換
    counts = {(row['symbol'], row['market']): row['top10_count']
              for _, row in df_counts.iterrows()}

    return counts


def _build_ranking_query(market: str) -> str:
    """
    ランキング取得用のSQLクエリを生成（DRY原則）

    Parameters:
    -----------
    market : str
        'jp' または 'us'

    Returns:
    --------
    str
        SQLクエリ文字列
    """
    return text("""
        SELECT
            sr.symbol,
            sr.score,
            sr.percentile,
            sr.universe_size,
            sr.market,
            sr.rank as original_rank,
            COALESCE(cp.company_name, sr.symbol) as company_name
        FROM backtest_results.score_rankings_v2 sr
        LEFT JOIN (
            SELECT DISTINCT ON (symbol)
                symbol, company_name
            FROM fmp_data.company_profile
            ORDER BY symbol, date DESC
        ) cp ON sr.symbol = cp.symbol
        WHERE sr.ranking_scope = 'daily'
        AND sr.ranking_method = 'total_score'
        AND sr.market = :market
        AND sr.rank_date = :rank_date
        ORDER BY sr.score DESC
        LIMIT 10
    """)


def _get_latest_available_ranking_date(engine):
    """
    最新の利用可能なランキング日付を取得

    Parameters:
    -----------
    engine : sqlalchemy.engine.Engine
        データベース接続エンジン

    Returns:
    --------
    str or None
        最新の日付（YYYY-MM-DD形式）、データがない場合はNone
    """
    query = text("""
        SELECT MAX(rank_date) as latest_date
        FROM backtest_results.score_rankings_v2
        WHERE ranking_scope = 'daily'
        AND ranking_method = 'total_score'
    """)

    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        if result and result[0]:
            return result[0].strftime('%Y-%m-%d')
    return None


def _fetch_market_ranking_data(engine, market: str, rank_date: str) -> pd.DataFrame:
    """
    特定市場のランキングデータを取得

    Parameters:
    -----------
    engine : sqlalchemy.engine.Engine
        データベース接続エンジン
    market : str
        'jp' または 'us'
    rank_date : str
        取得する日付（YYYY-MM-DD形式）

    Returns:
    --------
    pd.DataFrame
        ランキングデータ
    """
    query = _build_ranking_query(market)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={'market': market, 'rank_date': rank_date})


def fetch_daily_top10_rankings(engine, rank_date=None):
    """
    backtest_results.score_rankings_v2 から日次トップ10ランキングを取得（日米統合）
    時差を考慮して、実行時刻に応じて米国株と日本株で異なる日付を参照

    Parameters:
    -----------
    engine : sqlalchemy.engine.Engine
        データベース接続エンジン
    rank_date : str, optional
        取得する日付 (YYYY-MM-DD形式)。Noneの場合は実行時刻に応じて自動設定

    Returns:
    --------
    dict
        {'combined': DataFrame, 'jp_date': str, 'us_date': str}
        DataFrameには 'rank', 'symbol', 'score', 'percentile', 'universe_size', 'market' が含まれる
    """
    from datetime import datetime, timedelta

    # 日本時間の現在時刻を取得
    now_jst = datetime.now()
    current_hour = now_jst.hour

    # rank_dateが指定されていない場合、実行時刻に応じて日付を決定
    if rank_date is None:
        # 夕方実行（14時～23時59分）：日本株=当日、米国株=前日
        # 朝実行（0時～13時59分）：日本株=前日、米国株=前日（FMP取得後のため前営業日データは既に存在）
        if 14 <= current_hour <= 23:
            # 夕方実行：日本市場は終了しているが米国市場はまだ
            jp_date = now_jst.strftime('%Y-%m-%d')  # 当日
            us_date = (now_jst - timedelta(days=1)).strftime('%Y-%m-%d')  # 前日
            print(f"  📅 夕方実行モード（{current_hour}時）: 日本株={jp_date}, 米国株={us_date}")
        else:
            # 朝実行：日本株=前日、米国株=前日
            # FMP取得後にレポートが出力されるため、前営業日のデータは既に存在する
            previous_day = now_jst - timedelta(days=1)
            jp_date = previous_day.strftime('%Y-%m-%d')
            us_date = previous_day.strftime('%Y-%m-%d')  # 米国株も前日
            print(f"  📅 朝実行モード（{current_hour}時）: 日本株={jp_date}, 米国株={us_date}")
    else:
        # rank_dateが指定されている場合は両方同じ日付
        jp_date = rank_date
        us_date = rank_date
        print(f"  📅 日付指定モード: 日本株={jp_date}, 米国株={us_date}")

    # 日本株と米国株のデータを取得
    df_jp = _fetch_market_ranking_data(engine, 'jp', jp_date)
    df_us = _fetch_market_ranking_data(engine, 'us', us_date)

    # データが0件の場合は、最新の利用可能な日付にフォールバック
    if df_jp.empty and df_us.empty:
        print(f"  ⚠️ 指定日（日本株={jp_date}, 米国株={us_date}）のデータがありません。最新データにフォールバック...")

        fallback_date = _get_latest_available_ranking_date(engine)
        if fallback_date:
            print(f"  📅 フォールバック日付: {fallback_date}")
            jp_date = fallback_date
            us_date = fallback_date

            # フォールバック日付でデータを再取得
            df_jp = _fetch_market_ranking_data(engine, 'jp', jp_date)
            df_us = _fetch_market_ranking_data(engine, 'us', us_date)

    # 日米のデータを統合してスコア順にソート
    df_combined = pd.concat([df_jp, df_us], ignore_index=True)

    # 同じsymbolが複数存在する場合は、スコアが高い方のみを保持（重複を排除）
    # これは防御的プログラミングとして実装（データベースが正しければ発生しないが、万が一に備える）
    df_combined = df_combined.sort_values('score', ascending=False)
    df_combined = df_combined.drop_duplicates(subset=['symbol'], keep='first')

    df_combined = df_combined.head(10).reset_index(drop=True)
    df_combined['rank'] = range(1, len(df_combined) + 1)

    # Top10入り回数を取得して追加
    symbols_markets = [(row['symbol'], row['market']) for _, row in df_combined.iterrows()]
    top10_counts = fetch_top10_appearance_counts(engine, symbols_markets)
    df_combined['top10_count'] = df_combined.apply(
        lambda row: top10_counts.get((row['symbol'], row['market']), 0),
        axis=1
    )

    return {
        'combined': df_combined,
        'jp_date': jp_date,
        'us_date': us_date
    }


# レポート出力先ディレクトリ
REPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "reports"
GRAPHS_DIR = REPORT_DIR / "graphs"

# iCloud用のレポート出力先ディレクトリ
ICLOUD_REPORT_DIR = Path("/Users/HOME/Library/Mobile Documents/com~apple~CloudDocs/reports")
ICLOUD_GRAPHS_DIR = ICLOUD_REPORT_DIR / "graphs"

# サンプルデータを生成するための開始日(データがない場合のフォールバック用)
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')


def connect_to_database():
    """データベースに接続するための SQLAlchemy エンジンを取得"""
    # 接続文字列の構築
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # エンジンの作成
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    
    return engine


def ensure_api_alive() -> bool:
    """
    APIサーバーの生存確認
    
    Returns:
        bool: APIサーバーが正常に動作している場合True
    """
    try:
        response = requests.get("http://127.0.0.1:5001/api/health", timeout=3)
        response.raise_for_status()
        print("  ✅ ウォッチリストAPIサーバーは正常に動作中")
        return True
    except requests.exceptions.ConnectionError:
        print("  ⚠️ ウォッチリストAPIサーバーに接続できません")
        print("  💡 解決方法: 'python start_watchlist_api.py' でサーバーを起動してください")
        return False
    except requests.exceptions.Timeout:
        print("  ⚠️ ウォッチリストAPIサーバーが応答しません（タイムアウト）")
        return False
    except Exception as e:
        print(f"  ⚠️ ウォッチリストAPIサーバーのヘルスチェックでエラー: {e}")
        return False


def generate_fallback_watchlist_html() -> str:
    """APIサーバー未接続時のフォールバック用ウォッチリストHTML"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>ウォッチリスト - API未接続</title>
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .error-card {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }}
            .solution-card {{
                background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .command {{
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 10px 15px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                margin: 10px 0;
                font-size: 14px;
            }}
            .icon {{
                font-size: 2em;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 ウォッチリスト追跡レポート</h1>
            <p>最終更新: {current_time}</p>
            
            <div class="error-card">
                <div class="icon">❌</div>
                <h2>APIサーバーに接続できません</h2>
                <p>ウォッチリスト機能を使用するにはAPIサーバーを起動する必要があります。</p>
            </div>
            
            <div class="solution-card">
                <h3>🚀 解決方法</h3>
                
                <h4>1️⃣ APIサーバーを手動起動（すぐ試す）</h4>
                <p>新しいターミナルで以下を実行:</p>
                <div class="command">python start_watchlist_api.py</div>
                
                <h4>2️⃣ バックグラウンド起動（推奨）</h4>
                <p>サーバーを常駐させる場合:</p>
                <div class="command">nohup python start_watchlist_api.py > watchlist_api.log 2>&1 &</div>
                
                <h4>3️⃣ macOS自動起動設定（高度）</h4>
                <p>システム起動時に自動で立ち上げる場合、launchdを使用してください。</p>
                
                <h4>4️⃣ 起動確認</h4>
                <p>サーバー起動後、このページを再読み込みしてください。</p>
                <div class="command">curl http://127.0.0.1:5001/api/health</div>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
                <h3>📝 備考</h3>
                <ul>
                    <li>APIサーバーがないとウォッチリスト機能は使用できませんが、他のレポートは正常に生成されます</li>
                    <li>サーバーを起動した後は、ブラウザでこのページを再読み込みしてください</li>
                    <li>サーバーのメモリ使用量は30-50MB程度で、CPU負荷もほとんどありません</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


def fetch_stock_data(engine, start_date='2010-01-01'):
    """株価指数データを取得"""
    query = text("""
    SELECT symbol, date, close 
    FROM fmp_data.daily_prices
    WHERE symbol IN ('^VIX', '^GSPC', '^DJI', '^IXIC', '^N225', 'GCUSD', 'GOLD')
      AND date >= :start_date
    ORDER BY date;
    """)
    
    # SQLを実行してデータ取得
    with engine.connect() as conn:
        df_stocks = pd.read_sql_query(query, conn, params={"start_date": start_date})
    
    # データがない場合はサンプルデータを生成
    if df_stocks.empty or df_stocks['close'].isna().all():
        raise ValueError("株価データが取得できませんでした。")
    
    # ピボットテーブルを作成(カラムが [date, ^VIX, ^GSPC, ...] の形に)
    df_stocks_pivot = df_stocks.pivot(index='date', columns='symbol', values='close').reset_index()
    
    return df_stocks_pivot


def fetch_forex_data(engine, start_date='2010-01-01'):
    """為替データを取得"""
    query = text("""
    SELECT symbol, date, price 
    FROM fmp_data.forex
    WHERE symbol IN (
      'USDJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 
      'AUDUSD', 'USDCHF', 'EURJPY', 'GBPJPY'
    )
      AND date >= :start_date
    ORDER BY date;
    """)
    
    # SQLを実行してデータ取得
    with engine.connect() as conn:
        df_forex = pd.read_sql_query(query, conn, params={"start_date": start_date})
    
    # データがない場合はサンプルデータを生成
    if df_forex.empty or df_forex['price'].isna().all():
        print("警告: 為替データが取得できませんでした。サンプルデータを生成します。")
        df_forex = generate_sample_forex_data()
    
    # ピボットテーブルを作成
    df_forex_pivot = df_forex.pivot(index='date', columns='symbol', values='price').reset_index()
    
    return df_forex_pivot


def fetch_fred_data(engine, start_date='2010-01-01'):
    """FRED経済指標データを取得"""
    query = text("""
    SELECT indicator_name, date, value
    FROM fred_data.economic_indicators
    WHERE indicator_name IN (
      'FEDFUNDS', 'DGS10', 'BAA10Y', 'TWEXBGSMTH',
      'CPIAUCSL', 'CPILEGSL', 'CPILFESL', 'PCEPI',
      'GDP', 'UNRATE', 'yield_difference'
    )
      AND date >= :start_date
    ORDER BY date;
    """)
    
    # SQLを実行してデータ取得
    with engine.connect() as conn:
        df_fred = pd.read_sql_query(query, conn, params={"start_date": start_date})
    
    # データがない場合はサンプルデータを生成
    if df_fred.empty or df_fred['value'].isna().all():
        print("警告: FRED経済指標データが取得できませんでした。サンプルデータを生成します。")
        df_fred = generate_sample_fred_data()
    
    # ピボットテーブルを作成
    df_fred_pivot = df_fred.pivot(index='date', columns='indicator_name', values='value').reset_index()
    
    return df_fred_pivot


def generate_sample_forex_data():
    """サンプルの為替データを生成"""
    # 日付範囲の作成
    dates = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE)
    symbols = ['USDJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF', 'EURJPY', 'GBPJPY']
    
    # 初期値の設定
    base_values = {
        'USDJPY': 110,
        'EURUSD': 1.1,
        'GBPUSD': 1.3,
        'USDCAD': 1.3,
        'AUDUSD': 0.7,
        'USDCHF': 0.9,
        'EURJPY': 130,
        'GBPJPY': 145,
    }
    
    # データフレームの作成準備
    rows = []
    
    # ランダムウォークでデータ生成
    np.random.seed(43)  # 再現性のため
    
    for symbol in symbols:
        value = base_values[symbol]
        for date in dates:
            # 価格変動のシミュレーション
            change = np.random.normal(0, 0.005) * value
            value += change
            
            rows.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'price': value
            })
    
    return pd.DataFrame(rows)


def generate_sample_fred_data():
    """サンプルのFRED経済指標データを生成"""
    # 日付範囲の作成(月次データ)
    dates = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE, freq='MS')
    indicators = [
        'FEDFUNDS', 'DGS10', 'BAA10Y', 'TWEXBGSMTH',
        'CPIAUCSL', 'CPILEGSL', 'CPILFESL', 'PCEPI',
        'GDP', 'UNRATE', 'yield_difference'
    ]
    
    # 初期値の設定
    base_values = {
        'FEDFUNDS': 1.5,
        'DGS10': 2.5,
        'BAA10Y': 3.5,
        'TWEXBGSMTH': 110,
        'CPIAUCSL': 260,
        'CPILEGSL': 270,
        'CPILFESL': 265,
        'PCEPI': 110,
        'GDP': 20000,
        'UNRATE': 4.5,
        'yield_difference': 1.0
    }
    
    # 変動パターンの設定
    trend_factors = {
        'FEDFUNDS': 0.05,        # 上昇トレンド
        'DGS10': 0.03,           # 上昇トレンド
        'BAA10Y': 0.02,          # 上昇トレンド
        'TWEXBGSMTH': -0.01,     # 下降トレンド
        'CPIAUCSL': 0.2,         # インフレ上昇
        'CPILEGSL': 0.2,
        'CPILFESL': 0.15,
        'PCEPI': 0.1,
        'GDP': 50,               # GDP成長
        'UNRATE': -0.05,         # 失業率低下
        'yield_difference': -0.02 # イールドカーブフラット化
    }
    
    # データフレームの作成準備
    rows = []
    
    # データ生成
    np.random.seed(44)  # 再現性のため
    
    for indicator in indicators:
        value = base_values[indicator]
        trend = trend_factors[indicator]
        
        for date in dates:
            # 基本トレンドに沿った変動
            change = trend + np.random.normal(0, abs(trend))
            value += change
            
            if indicator == 'UNRATE':
                value = max(3.0, min(10.0, value))  # 3%～10%に制限
            elif indicator == 'yield_difference':
                value = max(-1.0, min(3.0, value))  # -1%～3%に制限
            
            rows.append({
                'indicator_name': indicator,
                'date': date.strftime('%Y-%m-%d'),
                'value': value
            })
    
    return pd.DataFrame(rows)


def prepare_merged_dataframes(df_stocks, df_forex, df_fred):
    """各データフレームを結合して分析用データセットを作成"""
    # 日付型に変換
    df_stocks['date'] = pd.to_datetime(df_stocks['date'])
    df_forex['date'] = pd.to_datetime(df_forex['date'])
    df_fred['date'] = pd.to_datetime(df_fred['date'])
    
    # マージ(日足ベース)
    # まず株価と為替をマージ
    df_daily = pd.merge(df_stocks, df_forex, on='date', how='outer')
    
    # FRED指標(月次・四半期)は日足データにマージする前に日次補間が必要
    # 日付インデックスに変換
    df_fred_indexed = df_fred.set_index('date')
    
    # 日次データの日付範囲を取得
    min_date = min(df_daily['date'])
    max_date = max(df_daily['date'])
    
    # 新しい日次インデックスを作成
    daily_index = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # FRED指標を日次にリサンプリング (前方埋め)
    df_fred_daily = df_fred_indexed.reindex(daily_index).ffill()
    df_fred_daily = df_fred_daily.reset_index()
    df_fred_daily = df_fred_daily.rename(columns={'index': 'date'})
    
    # 最終的なマージデータセット
    df_merged = pd.merge(df_daily, df_fred_daily, on='date', how='outer')
    
    # 欠損値処理
    df_merged = df_merged.ffill().bfill().dropna(how='all')
    
    return df_merged


def generate_reports(engine, df_merged, df_scored, df_scored_long, *, offline: bool = False):
    """各種グラフを生成してレポートとして保存
    
    Args:
        engine: DB接続エンジン
        df_merged: マージ済み市場データ
        df_scored: 短期スコア
        df_scored_long: 長期スコア
        offline: True の場合、APIサーバー依存機能をフォールバックに切替
    """
    # ディレクトリ作成
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # iCloudディレクトリも作成
    os.makedirs(ICLOUD_REPORT_DIR, exist_ok=True)
    os.makedirs(ICLOUD_GRAPHS_DIR, exist_ok=True)

    # 1. 株価指数推移(正規化)
    fig_indices, exp_indices = plot_normalized_indices(df_merged)
    html_content = build_html(fig_indices, exp_indices)
    (GRAPHS_DIR / "normalized_indices.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "normalized_indices.html").write_text(html_content, encoding="utf-8")
    
    # 1‑b. 株価指数 (直近 6 W)
    fig_idx_6w, exp_idx_6w = plot_normalized_indices_6w(df_merged)
    html_content = build_html(fig_idx_6w, exp_idx_6w)
    (GRAPHS_DIR / "normalized_indices_6w.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "normalized_indices_6w.html").write_text(html_content, encoding="utf-8")
    
    # 1‑c. 株価指数 (直近 3 M)
    fig_idx_3m, exp_idx_3m = plot_normalized_indices_3m(df_merged)
    html_content = build_html(fig_idx_3m, exp_idx_3m)
    (GRAPHS_DIR / "normalized_indices_3m.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "normalized_indices_3m.html").write_text(html_content, encoding="utf-8")
    
    # 2. VIX vs S&P500(2軸)
    fig_vix_sp500, exp_vix_sp500 = plot_vix_vs_sp500(df_merged)
    html_content = build_html(fig_vix_sp500, exp_vix_sp500)
    (GRAPHS_DIR / "vix_vs_sp500.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "vix_vs_sp500.html").write_text(html_content, encoding="utf-8")

    # 3. 金・ドル・為替(正規化)
    fig_gold_dollar, exp_gold_dollar = plot_gold_dollar_yen(df_merged)
    html_content = build_html(fig_gold_dollar, exp_gold_dollar)
    (GRAPHS_DIR / "gold_dollar_yen.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "gold_dollar_yen.html").write_text(html_content, encoding="utf-8")
    
    # --- 金・ドル・為替 (6W / 3M) ---
    fig_gdy_6w, exp_gdy_6w = plot_gold_dollar_yen_6w(df_merged)
    html_content = build_html(fig_gdy_6w, exp_gdy_6w)
    (GRAPHS_DIR / "gold_dollar_yen_6w.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "gold_dollar_yen_6w.html").write_text(html_content, encoding="utf-8")
    
    fig_gdy_3m, exp_gdy_3m = plot_gold_dollar_yen_3m(df_merged)
    html_content = build_html(fig_gdy_3m, exp_gdy_3m)
    (GRAPHS_DIR / "gold_dollar_yen_3m.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "gold_dollar_yen_3m.html").write_text(html_content, encoding="utf-8")

    # 4. 通貨ペア(対ドル／クロス円)
    #   plot_currency_pairs の戻り値は
    #   (fig_usd_pairs, fig_jpy_pairs, explanation) を想定
    res_currency = plot_currency_pairs(df_merged)
    if len(res_currency) == 3:
        fig_usd_pairs, fig_jpy_pairs, exp_currency = res_currency
    else:
        # 旧仕様との後方互換
        fig_usd_pairs, fig_jpy_pairs = res_currency
        exp_currency = ""
    
    html_content = build_html(fig_usd_pairs, exp_currency)
    (GRAPHS_DIR / "usd_currency_pairs.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "usd_currency_pairs.html").write_text(html_content, encoding="utf-8")
    
    html_content = build_html(fig_jpy_pairs, exp_currency)
    (GRAPHS_DIR / "jpy_currency_pairs.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "jpy_currency_pairs.html").write_text(html_content, encoding="utf-8")
    
    # --- 通貨ペア (6W / 3M) ---
    fig_usd_6w, fig_jpy_6w, exp_cur_6w = plot_currency_pairs_6w(df_merged)
    html_content = build_html(fig_usd_6w, exp_cur_6w)
    (GRAPHS_DIR / "usd_pairs_6w.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "usd_pairs_6w.html").write_text(html_content, encoding="utf-8")
    
    html_content = build_html(fig_jpy_6w, exp_cur_6w)
    (GRAPHS_DIR / "jpy_pairs_6w.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "jpy_pairs_6w.html").write_text(html_content, encoding="utf-8")
    
    fig_usd_3m, fig_jpy_3m, exp_cur_3m = plot_currency_pairs_3m(df_merged)
    html_content = build_html(fig_usd_3m, exp_cur_3m)
    (GRAPHS_DIR / "usd_pairs_3m.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "usd_pairs_3m.html").write_text(html_content, encoding="utf-8")
    
    html_content = build_html(fig_jpy_3m, exp_cur_3m)
    (GRAPHS_DIR / "jpy_pairs_3m.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "jpy_pairs_3m.html").write_text(html_content, encoding="utf-8")
    
    # 5. 金利推移
    fig_interest, exp_interest = plot_interest_rates(df_merged)
    html_content = build_html(fig_interest, exp_interest)
    (GRAPHS_DIR / "interest_rates.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "interest_rates.html").write_text(html_content, encoding="utf-8")

    # 6. インフレ指標
    fig_inflation, exp_inflation = plot_inflation(df_merged)
    html_content = build_html(fig_inflation, exp_inflation)
    (GRAPHS_DIR / "inflation.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "inflation.html").write_text(html_content, encoding="utf-8")

    # 7. 経済スコア
    print("DEBUG: About to generate economic score (short-term)...")
    fig_score, exp_score = plot_economic_score(df_scored)
    html_content = build_html(fig_score, exp_score)
    (GRAPHS_DIR / "economic_score.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "economic_score.html").write_text(html_content, encoding="utf-8")
    print("DEBUG: Economic score (short-term) completed!")
    
    # 7‑b. 経済スコア (長期 12M カナリア式)
    print("DEBUG: About to generate economic score (long-term)...")
    fig_score_long, exp_score_long = plot_economic_score(df_scored_long)
    html_content = build_html(fig_score_long, "【12M カナリア式】<br>" + exp_score_long)
    (GRAPHS_DIR / "economic_score_long.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "economic_score_long.html").write_text(html_content, encoding="utf-8")
    print("DEBUG: Economic score (long-term) completed!")

    # 8. 経済指標サブプロット
    print("DEBUG: About to generate economic subplots...")
    try:
        fig_subplots, exp_subplots = plot_economic_subplots(df_merged)
        html_content = build_html(fig_subplots, exp_subplots)
        (GRAPHS_DIR / "economic_subplots.html").write_text(html_content, encoding="utf-8")
        (ICLOUD_GRAPHS_DIR / "economic_subplots.html").write_text(html_content, encoding="utf-8")
        print("DEBUG: Economic subplots completed!")
    except Exception as e:
        print(f"ERROR: Economic subplots failed: {e}")
        import traceback
        traceback.print_exc()
        print("DEBUG: Skipping economic subplots and continuing...")
    
    # 9. 市場スコア評価（新機能）
    print("DEBUG: Starting market score evaluation...")
    print("市場スコア評価を生成中...")
    # マクロスコアとミクロスコアを計算
    print("DEBUG: About to call calculate_combined_score...")
    df_macro, macro_components, df_micro = calculate_combined_score(engine, df_merged)
    print("DEBUG: calculate_combined_score completed!")
    
    # ポートフォリオの銘柄リストを取得（確認用）
    print("DEBUG: About to get portfolio symbols...")
    symbols = get_portfolio_symbols(engine)
    print("DEBUG: Got portfolio symbols!")
    print(f"ポートフォリオ内の銘柄: {symbols}")
    
    # df_microに含まれる銘柄と期待する銘柄の差分を確認
    if not df_micro.empty:
        micro_symbols = df_micro['symbol'].unique().tolist()
        print(f"スコア計算された銘柄: {micro_symbols}")
        
        missing_symbols = [s for s in symbols if s not in micro_symbols]
        if missing_symbols:
            print(f"警告: スコア計算に含まれていない銘柄があります: {missing_symbols}")
    
    # スパークラインデータを生成（3ヶ月分、購入日対応）
    print("DEBUG: About to call create_score_sparklines...")
    sparkline_data = create_score_sparklines(df_micro, engine, days_back=90)
    print("DEBUG: create_score_sparklines completed!")
    print("  ✅ スパークラインデータ生成完了")
    
    # マクロスナップショットを生成（Fail-Fast方式）
    asof_date = datetime.now().date()
    output_dir = str(GRAPHS_DIR)
    
    print("DEBUG: About to call build_macro_snapshot")
    print("=== [MACRO] build_macro_snapshot: ENTER ===")
    try:
        # SSOT(既存HTML)方式なら DB不要。必要なら asof_date と output_dir だけ渡す。
        print("DEBUG: Importing build_macro_snapshot...")
        from investment_analysis.analysis.score_visualization import build_macro_snapshot
        import inspect
        
        print("DEBUG: Import successful")
        print("[MACRO] imported module file:", build_macro_snapshot.__module__)
        print("[MACRO] build_macro_snapshot defined in:", inspect.getsourcefile(build_macro_snapshot))
        
        macro_snapshot = build_macro_snapshot(asof_date=asof_date, output_dir=output_dir)
        if not macro_snapshot or "kpis" not in macro_snapshot:
            raise RuntimeError("[MACRO] snapshot is empty or missing 'kpis'")
        
        # 可視化ログ（最初の3件だけ例示）
        for k in list(macro_snapshot["kpis"])[:3]:
            v = macro_snapshot["kpis"][k]
            print(f"[MACRO] KPI {k} value={v.get('value')} series_len={len(v.get('series', []))}")
        
        print("=== [MACRO] build_macro_snapshot: OK ===")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise  # ← 握り潰さず必ず止める

    # 生成直後の強制ログ（12キーすべて）
    print("SNAPSHOT_KEYS", sorted(macro_snapshot["kpis"].keys()))
    for k, v in macro_snapshot["kpis"].items():
        print(f"KPI {k} value={v.get('value')} series_len={len(v.get('series', []))} source={v.get('source')} trace={v.get('trace')}")
    
    # 市場スコアレポート
    html_content = generate_market_score_html(df_macro, macro_components, df_micro, sparkline_data, macro_snapshot, engine)
    
    # HTMLへの埋め込みを"毎回アサート"（正規表現で既存ブロック全体を置換）
    import re
    import json
    
    payload = json.dumps(macro_snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    
    # 既存の <script id="macro-snapshot"...> ブロックを"丸ごと"置換
    html_content = re.sub(
        r'(<script id="macro-snapshot"[^>]*>)(.*?)(</script>)',
        r'\1' + payload + r'\3',
        html_content,
        flags=re.DOTALL
    )
    
    # 念のため、置換結果の健全性チェック
    if '"kpis":' not in html_content or '"asof":' not in html_content:
        raise RuntimeError("[MACRO] failed to inject snapshot json")
    
    (GRAPHS_DIR / "market_score_report.html").write_text(html_content, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "market_score_report.html").write_text(html_content, encoding="utf-8")
    
    # 埋め込みJSONの存在を保証
    print("  🔍 埋め込みJSONの存在確認中...")
    assert 'id="macro-snapshot"' in html_content, "macro-snapshot埋め込みJSONが見つかりません"
    print("  ✅ 埋め込みJSON確認完了")
    
    # 埋め込みJSONの内容確認
    if 'id="macro-snapshot"' in html_content:
        start_idx = html_content.find('id="macro-snapshot"')
        end_idx = html_content.find('</script>', start_idx)
        if end_idx != -1:
            json_content = html_content[start_idx:end_idx]
            print(f"  🔍 埋め込みJSON内容確認: {len(json_content)} characters")
            if '"kpis"' in json_content:
                print("  ✅ 埋め込みJSONにkpisキーが存在")
            else:
                print("  ⚠️ 埋め込みJSONにkpisキーが存在しません")
    
    # ポートフォリオ情報を取得
    print("ポートフォリオ情報を取得中...")
    portfolio_html, portfolio_figs = build_portfolio_section(engine)

    # ポートフォリオグラフをそれぞれ保存
    
    # 株価/テクニカルチャートを保存 (portfolio_figs[1:])
    # チャートを一つのHTMLにまとめる
    tech_html = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <style>
    body{font-family:system-ui,sans-serif;margin:0;padding:1rem;}
    .chart-container{max-width:1000px;margin:0 auto;}
    .chart-item{margin-bottom:30px;}
    .explanation{font-size:.9rem;line-height:1.6;margin-top:.8rem;color:#333;}
    h2 {margin-top: 30px; color: #333;}
    h3 {margin-top: 20px; color: #555; border-left: 4px solid #3498db; padding-left: 10px;}
    code {background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px;}
    ul {margin-left: 20px;}
    li {margin-bottom: 8px;}
  </style>
</head>
<body>
  <div class="chart-container">
    <h2>ポートフォリオ銘柄分析</h2>
"""

    # テクニカルチャート群を抽出（インデックス1からロット別チャートの始まる前まで）
    tech_charts_count = len([fig for fig in portfolio_figs if hasattr(fig.layout, 'title') and 
                           hasattr(fig.layout.title, 'text') and 
                           "Technical Chart" in fig.layout.title.text])
    
    # テクニカルチャートのみ抽出
    tech_charts = [fig for fig in portfolio_figs[1:tech_charts_count+1] 
                  if hasattr(fig.layout, 'title') and 
                  hasattr(fig.layout.title, 'text') and 
                  "Technical Chart" in fig.layout.title.text]
    
    # 各テクニカルチャートをHTMLに追加
    for i, fig in enumerate(tech_charts):
        plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        tech_html += f'    <div class="chart-item">{plot_div}</div>\n'
    
    # HTMLの終了タグを追加
    tech_html += """
    <div class="explanation">
      <h2>チャートの読み方と活用例</h2>
      
      <!-- 1. 価格 & 移動平均線 -->
      <h3>① 価格 ＋ 移動平均線（SMA）</h3>
      <p>
      <strong>何を見る？</strong><br>
      値動きそのものと、短期・中期の平均価格を比較してトレンドを把握します。<br>
      日本株・米国株ともに <code>SMA20 / SMA40</code> を採用。
      </p>
      <ul>
      <li><em>ゴールデンクロス</em>（短期SMAが長期SMAを上抜け） → 上昇トレンド入りシグナル</li>
      <li><em>デッドクロス</em>（短期SMAが長期SMAを下抜け） → 下降トレンド入りシグナル</li>
      <li>価格が<strong>SMAより上</strong> → 強気地合い／サポート<br>
          価格が<strong>SMAより下</strong> → 弱気地合い／レジスタンス</li>
      </ul>

      <!-- 2. Volume -->
      <h3>② 出来高（Volume）</h3>
      <p>
      <strong>何を見る？</strong><br>
      値動きの「信頼度」。<br>
      価格変動と同方向に出来高が増加していれば、トレンドの裏付けが強いと判断します。
      </p>

      <!-- 3. MACD Histogram -->
      <h3>③ MACD ヒストグラム</h3>
      <p>
      <strong>何を見る？</strong><br>
      トレンドの<strong>勢い（モメンタム）</strong>。0 ラインより上は上昇モメンタム、下は下降モメンタムを示します。
      </p>
      <ul>
      <li>ヒストグラムが<strong>プラスからマイナスへ転換</strong> → 上昇勢いの弱まり・反転警戒</li>
      <li>山（ピーク）や谷（ボトム）が連続して小さくなる → 勢いの減速</li>
      </ul>

      <!-- 4. RSI (14) -->
      <h3>④ RSI (14)</h3>
      <p>
      <strong>何を見る？</strong><br>
      買われ過ぎ・売られ過ぎの<strong>過熱感</strong>。0〜100 のスケールで表示します。
      </p>
      <ul>
      <li>70 以上 → <em>Overbought</em>（利確・押し目待ちを検討）</li>
      <li>30 以下 → <em>Oversold</em>（リバウンド・仕込み場を検討）</li>
      <li>ダイバージェンス（価格が高値更新なのに RSI が下落 etc.）はトレンド転換のシグナル候補</li>
      </ul>

      <!-- 5. ATR (14) -->
      <h3>⑤ ATR (14)</h3>
      <p>
      <strong>何を見る？</strong><br>
      過去 14 本分の<strong>平均的な価格変動幅（ボラティリティ）</strong>。方向性は示しません。
      </p>
      <ul>
      <li><strong>ATR が大きい</strong> → 相場が荒い・イベント直後。<br>
          ⇒ ストップ幅を広げる／ポジションサイズを減らす</li>
      <li><strong>ATR が小さい</strong> → レンジ相場・エネルギー蓄積期。<br>
          ⇒ ブレイクアウト前の静けさを示す可能性</li>
      <li>リスク管理での活用例：<br>
          逆指値 = エントリー価格 − (2 × ATR) など、変動幅に応じて調整</li>
      </ul>
    </div>
  </div>
</body>
</html>"""
    
    # 作成したHTMLファイルを保存
    (GRAPHS_DIR / "portfolio_history.html").write_text(tech_html, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "portfolio_history.html").write_text(tech_html, encoding="utf-8")
    
    # ロット別含み損益グラフを保存
    # 通貨ごとに別々のdivで複数のグラフを表示するHTMLを作成
    lot_html = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <style>
    body{font-family:system-ui,sans-serif;margin:0;padding:1rem;}
    .chart-container{max-width:1000px;margin:0 auto;}
    .explanation{font-size:.9rem;line-height:1.6;margin-top:.8rem;color:#333;}
  </style>
</head>
<body>
  <div class="chart-container">
"""
    
    # ロット別チャートの位置を特定
    lot_start_index = tech_charts_count + 1
    
    # 各通貨のグラフをHTMLに追加
    for i, fig in enumerate(portfolio_figs[lot_start_index:]):
        plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        lot_html += f"    <div>{plot_div}</div>\n"
    
    # テーブル部分とHTMLの終了タグを追加
    lot_html += f"""    <p class="explanation">{portfolio_html.replace("<h2>My Portfolio</h2>", "")}</p>
  </div>
</body>
</html>"""
    
    # 作成したHTMLファイルを保存
    (GRAPHS_DIR / "portfolio_lot.html").write_text(lot_html, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "portfolio_lot.html").write_text(lot_html, encoding="utf-8")
    
    # 全期間損益実績ページを保存
    alltime_html = build_alltime_portfolio_section(engine)
    (GRAPHS_DIR / "portfolio_alltime.html").write_text(alltime_html, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "portfolio_alltime.html").write_text(alltime_html, encoding="utf-8")
    
    # 10. スコア上位銘柄分析レポート（新機能）
    print("スコア上位銘柄分析レポートを生成中...")
    top_stocks_html = generate_top_stocks_report(engine)
    (GRAPHS_DIR / "top_stocks_analysis.html").write_text(top_stocks_html, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "top_stocks_analysis.html").write_text(top_stocks_html, encoding="utf-8")
    
    # 11. RSI35以下分析レポート（新機能）
    print("RSI35以下分析レポートを生成中...")
    rsi35_html = generate_rsi35_below_report(engine)
    (GRAPHS_DIR / "rsi35_below_analysis.html").write_text(rsi35_html, encoding="utf-8")
    (ICLOUD_GRAPHS_DIR / "rsi35_below_analysis.html").write_text(rsi35_html, encoding="utf-8")
    
    # 12. ミニチャート用JSON生成
    print("ミニチャート用JSONを生成中...")
    try:
        subprocess.run([sys.executable, "src/analysis/generate_mini_json.py"], 
                      check=True, cwd=project_root)
        print("  ✅ ミニチャート用JSON生成完了")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ ミニチャート用JSON生成エラー: {e}")
    except Exception as e:
        print(f"  ⚠️ ミニチャート用JSON生成で予期しないエラー: {e}")
    
    # 13. ウォッチリストレポート（動的更新版 + API生存確認 + ミニチャート機能）
    print("ウォッチリスト追跡レポートを生成中...")
    
    # API生存確認
    if (not offline) and ensure_api_alive():
        try:
            # ウォッチリストパフォーマンスデータ更新（軽量化のため一時的にスキップ）
            print("  ⚠️ パフォーマンスデータ更新をスキップ（処理軽量化のため）")
            # update_watchlist_performance_data(engine)
            
            # ウォッチリストレポート生成（ミニチャート版を正式採用）
            print("  📈 ウォッチリストレポート生成中...")
            mini_chart_html = generate_mini_chart_watchlist_html(engine)
            (GRAPHS_DIR / "watchlist_report_mini.html").write_text(mini_chart_html, encoding="utf-8")
            (ICLOUD_GRAPHS_DIR / "watchlist_report_mini.html").write_text(mini_chart_html, encoding="utf-8")
            
            print("  ✅ ウォッチリストレポート生成完了")
        except Exception as e:
            print(f"  ⚠️ ウォッチリストレポート生成エラー: {e}")
            # エラー時は空のレポートを生成
            from investment_analysis.analysis.watchlist_report import generate_empty_watchlist_html
            empty_watchlist_html = generate_empty_watchlist_html()
            (GRAPHS_DIR / "watchlist_report_mini.html").write_text(empty_watchlist_html, encoding="utf-8")
            (ICLOUD_GRAPHS_DIR / "watchlist_report_mini.html").write_text(empty_watchlist_html, encoding="utf-8")
    else:
        # APIサーバー未接続時はフォールバック用レポートを生成
        print("  🔄 APIサーバー未接続のため、フォールバック用レポートを生成中...")
        fallback_html = generate_fallback_watchlist_html()
        (GRAPHS_DIR / "watchlist_report_mini.html").write_text(fallback_html, encoding="utf-8")
        (ICLOUD_GRAPHS_DIR / "watchlist_report_mini.html").write_text(fallback_html, encoding="utf-8")
        print("  ✅ フォールバック用ウォッチリストレポート生成完了")
        print("  💡 APIサーバーを起動後、レポートを再生成してください")
    
    # 14. 日次株価上昇率ランキングレポート（新機能）
    print("日次株価上昇率ランキングレポートを生成中...")
    try:
        ranking_html = generate_daily_ranking_html(engine)
        (GRAPHS_DIR / "daily_ranking_report.html").write_text(ranking_html, encoding="utf-8")
        (ICLOUD_GRAPHS_DIR / "daily_ranking_report.html").write_text(ranking_html, encoding="utf-8")
        print("  ✅ 日次ランキングレポート生成完了")
    except Exception as e:
        print(f"  ⚠️ 日次ランキングレポート生成エラー: {e}")
        # エラー時は空のレポートを生成
        from investment_analysis.analysis.daily_ranking_report import generate_empty_ranking_html
        empty_ranking_html = generate_empty_ranking_html()
        (GRAPHS_DIR / "daily_ranking_report.html").write_text(empty_ranking_html, encoding="utf-8")
        (ICLOUD_GRAPHS_DIR / "daily_ranking_report.html").write_text(empty_ranking_html, encoding="utf-8")

    # 15. スコアリング品質検証レポート（新機能）
    print("スコアリング品質検証レポートを生成中...")
    try:
        if ScoringValidator is not None:
            validator = ScoringValidator()
            current_date = datetime.now().strftime("%Y-%m-%d")
            daily_results = validator.run_daily_validation(current_date)
            
            # 検証HTMLセクションを生成
            validation_html_content = validator.generate_daily_html_section(daily_results)
            
            # スタンドアロンHTML として完成
            standalone_validation_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <title>スコアリング品質チェック - {current_date}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .metric-card {{ background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 10px; text-align: center; }}
        .metric-card h4 {{ margin: 0 0 10px 0; color: #495057; font-size: 0.9rem; font-weight: 600; }}
        .metric-value {{ display: block; font-size: 1.5rem; font-weight: bold; margin: 5px 0; }}
        .text-success {{ color: #28a745; }}
        .text-warning {{ color: #ffc107; }}
        .text-danger {{ color: #dc3545; }}
        .font-weight-bold {{ font-weight: bold; }}
        .row {{ display: flex; flex-wrap: wrap; margin: -10px; }}
        .col-md-3 {{ flex: 0 0 25%; max-width: 25%; }}
        .alert-summary {{ background: #e9ecef; padding: 15px; border-radius: 8px; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 0.8rem; }}
        .badge-success {{ background-color: #28a745; }}
        .badge-warning {{ background-color: #ffc107; }}
        .badge-danger {{ background-color: #dc3545; }}
        .validation-details {{ margin-top: 20px; }}
        .pillar-scores {{ list-style: none; padding: 0; }}
        .pillar-scores li {{ padding: 5px 0; border-bottom: 1px solid #eee; }}
        .driver-distribution {{ list-style: none; padding: 0; }}
        .driver-distribution li {{ padding: 3px 0; }}
        .alert-item {{ margin: 8px 0; padding: 8px; border-left: 4px solid #007bff; background: #f8f9fa; }}
        .alert-list {{ list-style: none; padding: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 スコアリングシステム品質検証レポート</h1>
        <p>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {validation_html_content}
        
        <div class="footer" style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d;">
            <p>🤖 This report was automatically generated by the Scoring Validation System</p>
        </div>
    </div>
</body>
</html>"""
            
            (GRAPHS_DIR / "scoring_validation_report.html").write_text(standalone_validation_html, encoding="utf-8")
            (ICLOUD_GRAPHS_DIR / "scoring_validation_report.html").write_text(standalone_validation_html, encoding="utf-8")
            
            # 検証結果をログに記録
            validator.log_validation_results(daily_results)
            print("  ✅ スコアリング品質検証レポート生成完了")
            
        else:
            # 検証モジュールが利用できない場合のフォールバック
            fallback_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <title>スコアリング品質チェック - 利用不可</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; line-height: 1.6; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .error-card {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 スコアリング品質チェック</h1>
        <div class="error-card">
            <h2>⚠️ 検証機能が利用できません</h2>
            <p>スコアリング検証モジュールが正しくインポートされていません。</p>
            <p>src/scoring/validation.py の実装を確認してください。</p>
        </div>
    </div>
</body>
</html>"""
            (GRAPHS_DIR / "scoring_validation_report.html").write_text(fallback_html, encoding="utf-8")
            (ICLOUD_GRAPHS_DIR / "scoring_validation_report.html").write_text(fallback_html, encoding="utf-8")
            print("  ⚠️ スコアリング検証モジュールが利用できません")
        
    except Exception as e:
        print(f"  ⚠️ スコアリング品質検証レポート生成エラー: {e}")
        # エラー時は基本的なエラーレポートを生成
        error_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="utf-8">
    <title>スコアリング品質チェック - エラー</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; line-height: 1.6; }}
        .container {{ max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .error-card {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 スコアリング品質チェック</h1>
        <div class="error-card">
            <h2>❌ エラーが発生しました</h2>
            <p>検証処理中にエラーが発生しました:</p>
            <p>{str(e)}</p>
        </div>
    </div>
</body>
</html>"""
        (GRAPHS_DIR / "scoring_validation_report.html").write_text(error_html, encoding="utf-8")
        (ICLOUD_GRAPHS_DIR / "scoring_validation_report.html").write_text(error_html, encoding="utf-8")
    
    # 16. 売買記録分析は portfolio_alltime.html と market_score_report.html に統合されました
    print("  ✅ 売買記録分析機能は他のレポートに統合済み")

    # ダッシュボード HTML
    create_dashboard_html()

    # 更新日時
    update_text = f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    with open(REPORT_DIR / "last_update.txt", "w", encoding="utf-8") as f:
        f.write(update_text)
    with open(ICLOUD_REPORT_DIR / "last_update.txt", "w", encoding="utf-8") as f:
        f.write(update_text)

    print(f"レポートを {REPORT_DIR} と {ICLOUD_REPORT_DIR} に保存しました。")
    print(f"グラフファイルは {GRAPHS_DIR} と {ICLOUD_GRAPHS_DIR} に保存しました。")


def create_dashboard_html():
    """
    ダッシュボードHTMLファイルを生成
    """
    # 現在時刻
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # グラフグループの定義
    groups = {
        '株価指数': [
            '株価指数(正規化)',
            '株価指数(直近 3M)',
            '株価指数(直近 6W)',
            'VIX vs S&P500'
        ],
        '金・ドル・為替': [
            '金・ドル・為替',
            '金・ドル・為替(直近 3M)',
            '金・ドル・為替(直近 6W)'
        ],
        '通貨ペア': [
            '対ドル通貨ペア',
            '対ドル通貨ペア(3M)',
            '対ドル通貨ペア(6W)',
            'クロス円通貨ペア',
            'クロス円通貨ペア(3M)',
            'クロス円通貨ペア(6W)'
        ],
        '金融指標': [
            '金利推移',
            'インフレ指標'
        ],
        '総合評価': [
            '経済スコア(3M 短期)',
            '経済スコア(12M カナリア式)',
            '経済指標サブプロット',
            '市場スコアレポート',
        ],
        'Portfolio': [
            '株価推移',
            'ロット別含み損益',
            '全期間損益実績'
        ],
        'ランキング': [
            '日次上昇率ランキング'
        ],
        'スコア分析': [
            'スコア上位銘柄分析',
            'RSI35以下',
            'ウォッチリスト'
        ],
        '品質検証': [
            'スコアリング品質チェック'
        ]
    }
    
    # グループごとの色定義
    group_colors = {
        '株価指数': '#d4e6f1',      # 薄い青
        '金・ドル・為替': '#d5f5e3', # 薄い緑
        '通貨ペア': '#fdebd0',       # 薄いオレンジ
        '金融指標': '#ebdef0',       # 薄い紫
        '総合評価': '#f9e79f',       # 薄い黄色
        'Portfolio': '#d6eaf8',   # 薄い水色
        'ランキング': '#ffe6e6',      # 薄い赤
        'スコア分析': '#fce4ec',      # 薄いピンク
        '品質検証': '#e8f5e8',       # 薄い緑
    }
    
    # HTMLパスのマッピング
    html_paths = {
        '株価指数(正規化)': "graphs/normalized_indices.html",
        '株価指数(直近 3M)': "graphs/normalized_indices_3m.html",
        '株価指数(直近 6W)': "graphs/normalized_indices_6w.html",
        'VIX vs S&P500': "graphs/vix_vs_sp500.html",
        '金・ドル・為替': "graphs/gold_dollar_yen.html",
        '金・ドル・為替(直近 3M)': "graphs/gold_dollar_yen_3m.html",
        '金・ドル・為替(直近 6W)': "graphs/gold_dollar_yen_6w.html",
        '対ドル通貨ペア': "graphs/usd_currency_pairs.html",
        '対ドル通貨ペア(3M)': "graphs/usd_pairs_3m.html",
        '対ドル通貨ペア(6W)': "graphs/usd_pairs_6w.html",
        'クロス円通貨ペア': "graphs/jpy_currency_pairs.html",
        'クロス円通貨ペア(3M)': "graphs/jpy_pairs_3m.html",
        'クロス円通貨ペア(6W)': "graphs/jpy_pairs_6w.html",
        '金利推移': "graphs/interest_rates.html",
        'インフレ指標': "graphs/inflation.html",
        '経済スコア(3M 短期)': "graphs/economic_score.html",
        '経済スコア(12M カナリア式)': "graphs/economic_score_long.html",
        '経済指標サブプロット': "graphs/economic_subplots.html",
        '市場スコアレポート': "graphs/market_score_report.html",
        '株価推移': "graphs/portfolio_history.html",
        'ロット別含み損益': "graphs/portfolio_lot.html",
        '全期間損益実績': "graphs/portfolio_alltime.html",
        '日次上昇率ランキング': "graphs/daily_ranking_report.html",
        'スコア上位銘柄分析': "graphs/top_stocks_analysis.html",
        'RSI35以下': "graphs/rsi35_below_analysis.html",
        'ウォッチリスト': "graphs/watchlist_report_mini.html",
        'スコアリング品質チェック': "graphs/scoring_validation_report.html"
    }
    
    # ナビゲーションHTML生成
    nav_html = ""
    for group, items in groups.items():
        nav_html += f"<div class=\"nav-group\"><div class=\"group-label\">{group}</div>"
        for item in items:
            color = group_colors.get(group, '#f0f0f0')
            nav_html += f"<div class=\"nav-item\" data-target=\"{html_paths[item]}\" style=\"background-color: {color};\">{item}</div>"
        nav_html += "</div>"
    
    # HTML全体を生成
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Economic Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        
        /* ヘッダー部分のレイアウト */
        .header {{
            margin-bottom: 20px;
        }}
        
        .title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding: 0 20px;
        }}
        
        .title-main {{
            flex: 1;
            text-align: left;
        }}
        
        .title-main h1 {{
            margin: 0;
            font-size: 2rem;
            color: #2c3e50;
            font-weight: 600;
        }}
        
        .title-info {{
            flex: 1;
            text-align: right;
            color: #6c757d;
            font-size: 0.9rem;
        }}
        
        /* 折りたたみ制御 */
        .nav-controls {{
            text-align: center;
            margin-bottom: 15px;
        }}
        
        .toggle-nav {{
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,123,255,0.3);
        }}
        
        .toggle-nav:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,123,255,0.4);
        }}
        
        .toggle-nav:active {{
            transform: translateY(0);
        }}
        
        /* ナビゲーション部分 */
        .nav-container {{
            overflow: hidden;
            transition: max-height 0.5s ease;
            margin-bottom: 20px;
        }}
        
        .nav-container.collapsed {{
            max-height: 0;
            margin-bottom: 0;
        }}
        
        .nav-container.expanded {{
            max-height: 500px;
        }}
        
        .nav {{ 
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 15px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .nav-group {{
            display: flex;
            flex-direction: column;
            margin-right: 15px;
            margin-bottom: 10px;
        }}
        
        .group-label {{
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 0.9rem;
            color: #495057;
            text-shadow: 0 1px 2px rgba(255,255,255,0.8);
        }}
        
        .nav-item {{ 
            padding: 8px 12px; 
            margin-bottom: 5px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            border: 1px solid rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .nav-item:hover {{ 
            filter: brightness(0.95);
            transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        }}
        
        .nav-item.active {{ 
            color: white;
            font-weight: bold;
            filter: brightness(0.9);
            box-shadow: 0 3px 12px rgba(0,0,0,0.25);
            transform: translateY(-1px);
        }}
        
        iframe {{ 
            width: 100%; 
            height: 1080px; 
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            transition: opacity 0.3s ease;
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 0.8em;
            color: #6c757d;
        }}
        
        /* スマートフォン対応 */
        @media (max-width: 768px) {{
            .title-row {{
                flex-direction: column;
                text-align: center;
                padding: 0 10px;
            }}
            
            .title-main {{
                text-align: center;
                margin-bottom: 10px;
            }}
            
            .title-main h1 {{
                font-size: 1.6rem;
            }}
            
            .title-info {{
                text-align: center;
            }}
            
            .nav {{
                flex-direction: column;
            }}
            
            .nav-group {{
                margin-right: 0;
            }}
            
            iframe {{
                height: 600px;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <div class="title-row">
                <div class="title-main">
                    <h1>Economic Dashboard</h1>
                </div>
                <div class="title-info">
                    Report Generated: {now}
                </div>
            </div>
            
            <div class="nav-controls">
                <button class="toggle-nav" id="toggle-nav">
                    📊 ナビゲーションを表示 / 非表示
                </button>
            </div>
        </div>
        
        <div class="nav-container expanded" id="nav-container">
            <div class="nav" id="nav">
                {nav_html}
            </div>
        </div>

        <iframe id="graph-frame" src="graphs/j"></iframe>
        
        <div class="footer">
            <p>データは日次で自動更新されます。データソース: FMP API, FRED API</p>
        </div>
    </div>
    
    <script>
        // ナビゲーションの切り替え機能
        const navItems = document.querySelectorAll('.nav-item');
        const frame = document.getElementById('graph-frame');
        
        // ナビゲーション折りたたみ機能
        const toggleBtn = document.getElementById('toggle-nav');
        const navContainer = document.getElementById('nav-container');
        
        // 折りたたみ状態をlocalStorageから復元
        const navCollapsed = localStorage.getItem('nav-collapsed') === 'true';
        if (navCollapsed) {{
            navContainer.classList.remove('expanded');
            navContainer.classList.add('collapsed');
        }}
        
        toggleBtn.addEventListener('click', () => {{
            const isCollapsed = navContainer.classList.contains('collapsed');
            
            if (isCollapsed) {{
                navContainer.classList.remove('collapsed');
                navContainer.classList.add('expanded');
                localStorage.setItem('nav-collapsed', 'false');
            }} else {{
                navContainer.classList.remove('expanded');
                navContainer.classList.add('collapsed');
                localStorage.setItem('nav-collapsed', 'true');
            }}
        }});
        
        // 初期状態で「市場スコアレポート」をアクティブに
        const marketScoreItem = Array.from(navItems).find(item => item.textContent === '市場スコアレポート');
        if (marketScoreItem) {{
            // 他のアクティブ状態をクリア
            navItems.forEach(i => i.classList.remove('active'));
            // 市場スコアレポートをアクティブに
            marketScoreItem.classList.add('active');
            // iframe のソースも市場スコアレポートに設定
            frame.src = marketScoreItem.dataset.target;
        }} else {{
            // 市場スコアレポートが見つからない場合は経済スコアを探す
            const economicScoreItem = Array.from(navItems).find(item => item.textContent === '経済スコア(3M 短期)');
            if (economicScoreItem) {{
                navItems.forEach(i => i.classList.remove('active'));
                economicScoreItem.classList.add('active');
                frame.src = economicScoreItem.dataset.target;
            }} else {{
                // どちらも見つからない場合はデフォルト設定
                const defaultItem = navItems[0];
                if (defaultItem) {{
                    defaultItem.classList.add('active');
                    frame.src = defaultItem.dataset.target;
                }}
            }}
        }}
        
        navItems.forEach(item => {{
            item.addEventListener('click', () => {{
                // アクティブクラスの切り替え
                navItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                
                // iframeのソース変更（フェードエフェクト付き）
                frame.style.opacity = '0.7';
                setTimeout(() => {{
                    frame.src = item.dataset.target;
                    frame.style.opacity = '1';
                }}, 150);
            }});
        }});
        
        // iframeの読み込み完了時のエフェクト
        frame.addEventListener('load', () => {{
            frame.style.opacity = '1';
        }});
    </script>
</body>
</html>
"""
    
    # HTMLファイルを書き込み
    dashboard_path = REPORT_DIR / "dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # iCloudにもコピー
    icloud_dashboard_path = ICLOUD_REPORT_DIR / "dashboard.html"
    with open(icloud_dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"ダッシュボードHTMLを生成しました: {dashboard_path}")
    return dashboard_path


# APIサーバー自動起動関連の関数を削除（常駐サービス化のため）
# 手動でAPIサーバーを起動してください: python start_watchlist_api.py


def check_port_available(port: int) -> bool:
    """指定されたポートが使用可能かチェック"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            return result != 0  # 0でない場合は接続できない（使用可能）
    except Exception:
        return True  # エラーの場合は使用可能とみなす


def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """使用可能なポートを探す（改良版）"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_available(port):
            # 二重チェック（レースコンディション対策）
            time.sleep(0.1)  # 短時間待機
            if check_port_available(port):
                return port
    return start_port  # 見つからない場合はデフォルトを返す


def start_http_server(port: int, directory: str = "reports") -> Optional[subprocess.Popen]:
    """HTTPサーバーを起動する"""
    try:
        # ディレクトリの存在確認
        if not os.path.exists(directory):
            print(f"⚠️ ディレクトリが存在しません: {directory}")
            return None
        
        print(f"🔧 HTTPサーバー起動コマンド: python -m http.server {port} --bind 127.0.0.1 --directory {directory}")
        
        # サーバー起動（127.0.0.1でバインド）
        # stdout/stderrをDEVNULLにリダイレクトしてバッファブロックを防ぐ
        process = subprocess.Popen([
            sys.executable, "-m", "http.server", str(port),
            "--bind", "127.0.0.1",
            "--directory", directory
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 少し待ってからプロセスの状態を確認
        time.sleep(3)  # 3秒待機（より確実に）
        if process.poll() is None:  # まだ実行中
            print(f"✅ HTTPサーバーをポート {port} で起動しました（127.0.0.1:{port}）")
            print(f"📋 プロセスID: {process.pid}")
            return process
        else:
            # プロセスが終了していた場合
            print(f"⚠️ HTTPサーバー起動に失敗: プロセスが予期せず終了しました")
            print(f"📋 終了コード: {process.returncode}")
            return None
    except Exception as e:
        print(f"⚠️ HTTPサーバー起動エラー: {e}")
        import traceback
        print(f"🔍 詳細エラー:\n{traceback.format_exc()}")
        return None


def ensure_http_server(preferred_port: int = 8080) -> tuple[str, Optional[subprocess.Popen]]:
    """
    HTTPサーバーが動作していることを確認し、必要に応じて起動する
    
    Returns:
        tuple: (dashboard_url, server_process or None)
    """
    # まず既存のサーバーがあるかチェック（127.0.0.1で統一）
    for port in range(preferred_port, preferred_port + 5):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/dashboard.html", timeout=5)
            if response.status_code == 200:
                print(f"✅ 既存のHTTPサーバーを発見（127.0.0.1:{port}）")
                return f"http://127.0.0.1:{port}/dashboard.html", None
        except requests.exceptions.RequestException:
            continue
    
    # 既存のサーバーが見つからない場合、新しく起動
    print("🚀 HTTPサーバーを起動中...")
    available_port = find_available_port(preferred_port)
    
    if available_port != preferred_port:
        print(f"💡 ポート {preferred_port} は使用中のため、ポート {available_port} を使用します")
    
    server_process = start_http_server(available_port)
    if server_process:
        # サーバー起動後の確認（127.0.0.1で統一）
        dashboard_url = f"http://127.0.0.1:{available_port}/dashboard.html"
        
        # 最大15回確認を試行（より確実に）
        print("⏳ HTTPサーバーの起動確認中...")
        for i in range(15):
            try:
                time.sleep(1.5)  # 1.5秒待機（より確実に）
                response = requests.get(dashboard_url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ HTTPサーバー起動完了（127.0.0.1:{available_port}）")
                    # 追加で少し待機してからブラウザを開く
                    time.sleep(1)
                    return dashboard_url, server_process
                else:
                    print(f"⏳ 起動確認中... ({i+1}/15) - Status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⏳ 起動確認中... ({i+1}/15) - Error: {type(e).__name__}")
                if i == 14:  # 最後の試行
                    print("⚠️ HTTPサーバーの起動確認に失敗しました")
                    print(f"⚠️ エラー: {e}")
                    
                    # サーバープロセスの状態を確認
                    if server_process.poll() is not None:
                        print("❌ HTTPサーバープロセスが終了しています")
                        print(f"📋 終了コード: {server_process.returncode}")
                    else:
                        print("🔄 サーバープロセスはまだ実行中です")
                        # プロセスが生きている場合は強制終了
                        print("🛑 応答しないサーバープロセスを強制終了します")
                        try:
                            server_process.terminate()
                            server_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            print("⚠️ プロセスが応答しないため、強制kill実行")
                            server_process.kill()
                            server_process.wait()
        
        # 起動確認に失敗した場合でも、ファイル直接アクセスとして返す
        dashboard_file = REPORT_DIR / "dashboard.html"
        if dashboard_file.exists():
            print("🔄 HTTPサーバーの確認に失敗したため、ファイル直接アクセスを使用します")
            return f"file://{dashboard_file.absolute()}", None  # プロセスは既に終了済み
    
    # サーバー起動に失敗した場合
    print("❌ HTTPサーバーの起動に失敗しました")
    dashboard_file = REPORT_DIR / "dashboard.html"
    if dashboard_file.exists():
        print("💡 ファイル直接アクセスを使用してください")
        return f"file://{dashboard_file.absolute()}", None
    
    return "", None


def open_dashboard_safely(dashboard_url: str):
    """ダッシュボードを安全に開く（代替手段も含む）"""
    if not dashboard_url:
        # HTTPサーバーが使用できない場合はファイル直接開きを試行
        dashboard_file = REPORT_DIR / "dashboard.html"
        if dashboard_file.exists():
            print("🔄 HTTPサーバーが利用できないため、ファイルを直接開きます")
            dashboard_url = f"file://{dashboard_file.absolute()}"
        else:
            print("❌ ダッシュボードファイルが見つかりません")
            return
    
    print(f"🌐 ダッシュボードをブラウザで開いています: {dashboard_url}")
    
    # ブラウザで開く（複数の方法を試行）
    if sys.platform == 'darwin':
        # macOSの場合
        try:
            # 方法1: open コマンド
            result = subprocess.run(['open', dashboard_url], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ Safariでダッシュボードを開きました")
                return
            else:
                print(f"⚠️ openコマンドが失敗: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠️ openコマンドがタイムアウトしました")
        except Exception as e:
            print(f"⚠️ openコマンドでエラー: {e}")
        
        # 方法2: webbrowserモジュールを使用
        try:
            webbrowser.open(dashboard_url)
            print("✅ デフォルトブラウザでダッシュボードを開きました")
            return
        except Exception as e:
            print(f"⚠️ webbrowserでエラー: {e}")
        
        # 方法3: 特定のブラウザを指定
        browsers = [
            'safari',
            'google chrome', 
            'firefox'
        ]
        
        for browser in browsers:
            try:
                subprocess.run(['open', '-a', browser, dashboard_url], 
                             check=True, capture_output=True, timeout=5)
                print(f"✅ {browser}でダッシュボードを開きました")
                return
            except:
                continue
        
        print("⚠️ 全ての方法でブラウザの起動に失敗しました")
    else:
        # 他のプラットフォーム
        try:
            webbrowser.open(dashboard_url)
            print("✅ デフォルトブラウザでダッシュボードを開きました")
            return
        except Exception as e:
            print(f"⚠️ ブラウザでの表示に失敗: {e}")
    
    # 全ての自動開きが失敗した場合
    print("❌ 自動でブラウザを開くことができませんでした")
    print("💡 手動で以下のURLをブラウザで開いてください:")
    print(f"   📋 {dashboard_url}")
    
    # クリップボードにコピーを試行（macOSのみ）
    if sys.platform == 'darwin':
        try:
            subprocess.run(['pbcopy'], input=dashboard_url, text=True, check=True)
            print("📋 URLをクリップボードにコピーしました")
        except:
            pass


def main():
    """メイン処理（堅牢なエラーハンドリング付き）"""
    print("🚀 エコノミックダッシュボード生成を開始します...")
    print(f"📅 実行開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 実行オプション
    args = sys.argv[1:]
    no_browser = "--no-browser" in args
    no_server = "--no-server" in args
    offline = "--offline" in args
    batch_mode = "--batch-mode" in args  # バッチ実行モード用フラグ
    keep_server = "--keep-server" in args  # サーバー常駐モード用フラグ
    start_server = "--start-server" in args  # 明示的なサーバー起動フラグ
    
    # バッチ実行時は自動的にサーバー起動を無効化（分離のため）
    if batch_mode:
        print("🏭 バッチ実行モードを検出しました")
        print("   📋 HTTPサーバーは外部で管理してください（scripts/start_dashboard_server.sh）")
        if "--no-browser" not in args:
            print("   🌐 レポート生成後にブラウザを自動起動します")
        no_server = True   # バッチモードではサーバー起動しない
    
    # 明示的なサーバー起動フラグがある場合は有効化
    if start_server:
        no_server = False
    
    try:
        # データベース接続
        print("\n🔌 データベースに接続中...")
        try:
            engine = connect_to_database()
            # 接続テスト
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test")).fetchone()
                print(f"   ✅ データベース接続成功: {result.test}")
        except Exception as db_error:
            print(f"❌ データベース接続エラー: {db_error}")
            print("🔧 解決方法:")
            print("   1. PostgreSQLサーバーが起動していることを確認")
            print("   2. src/utilities/config.py の接続設定を確認")
            print("   3. データベースユーザーの権限を確認")
            return
        
        # ポートフォリオ情報を取得
        print("\n📊 ポートフォリオ情報を取得中...")
        try:
            portfolio_html, portfolio_figs = build_portfolio_section(engine)
            print(f"   ✅ ポートフォリオデータ取得成功: {len(portfolio_figs)}個のグラフ")
        except Exception as portfolio_error:
            print(f"⚠️ ポートフォリオ取得エラー: {portfolio_error}")
            print("   💡 ポートフォリオデータがなくても処理は続行します")
            portfolio_html, portfolio_figs = "", []
        
        # データ取得(2010年以降)
        print("\n📈 市場データを取得中...")
        
        # 株価指数データ
        try:
            print("   📊 株価指数データを取得中...")
            df_stocks = fetch_stock_data(engine)
            print(f"   ✅ 株価指数データ取得成功: {len(df_stocks)}件")
        except Exception as stocks_error:
            print(f"❌ 株価指数データ取得エラー: {stocks_error}")
            print("🔧 対象テーブル: fmp_data.daily_prices")
            return
        
        # 為替データ
        try:
            print("   💱 為替データを取得中...")
            df_forex = fetch_forex_data(engine)
            print(f"   ✅ 為替データ取得成功: {len(df_forex)}件")
        except Exception as forex_error:
            print(f"❌ 為替データ取得エラー: {forex_error}")
            print("🔧 対象テーブル: fmp_data.forex")
            return
        
        # 経済指標データ
        try:
            print("   📉 経済指標データを取得中...")
            df_fred = fetch_fred_data(engine)
            print(f"   ✅ 経済指標データ取得成功: {len(df_fred)}件")
        except Exception as fred_error:
            print(f"❌ 経済指標データ取得エラー: {fred_error}")
            print("🔧 対象テーブル: fred_data.economic_indicators")
            return
        
        # データ整形
        print("\n🔄 データを整形・結合中...")
        try:
            df_merged = prepare_merged_dataframes(df_stocks, df_forex, df_fred)
            print(f"   ✅ データ結合成功: {len(df_merged)}件、{len(df_merged.columns)}カラム")
            
            # データ品質チェック
            if df_merged.empty:
                raise ValueError("結合後のデータが空です")
            
            missing_pct = (df_merged.isnull().sum().sum() / (df_merged.shape[0] * df_merged.shape[1])) * 100
            print(f"   📊 データ品質: 欠損率 {missing_pct:.1f}%")
            
        except Exception as merge_error:
            print(f"❌ データ整形エラー: {merge_error}")
            return
        
        # スコア計算
        print("\n🧮 経済スコアを計算中...")
        try:
            print("   📊 短期スコア（3M）を計算中...")
            df_scored = evaluate_economic_indicators(df_merged)
            print(f"   ✅ 短期スコア計算成功: {len(df_scored)}件")
            
            print("   📊 長期スコア（12M）を計算中...")
            df_scored_long = evaluate_economic_indicators_long(df_merged)
            print(f"   ✅ 長期スコア計算成功: {len(df_scored_long)}件")
            
        except Exception as score_error:
            print(f"❌ スコア計算エラー: {score_error}")
            print("🔧 scoring_functions モジュールの実装を確認してください")
            return
        
        # レポート生成
        print("\n📄 レポートを生成中...")
        try:
            generate_reports(engine, df_merged, df_scored, df_scored_long, offline=offline)
            print("   ✅ 全レポート生成完了")
            
            # 生成されたファイルの確認
            report_files = list(GRAPHS_DIR.glob("*.html"))
            print(f"   📁 生成ファイル数: {len(report_files)}個")
            
        except Exception as report_error:
            print(f"❌ レポート生成エラー: {report_error}")
            print(f"🔧 出力先ディレクトリ: {REPORT_DIR}")
            print("   ディスク容量と書き込み権限を確認してください")
            import traceback
            print(f"🔍 詳細エラー:\n{traceback.format_exc()}")
            return
        
        print("\n✅ 処理が正常に完了しました！")
        print(f"📅 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n⚡ ユーザーによる中断（Ctrl+C）")
        return
    except Exception as unexpected_error:
        print(f"\n❌ 予期しないエラーが発生しました: {unexpected_error}")
        import traceback
        print(f"🔍 詳細エラー:\n{traceback.format_exc()}")
        return
    
    # HTTPサーバーの確認・起動とダッシュボードの表示
    try:
        dashboard_file = REPORT_DIR / "dashboard.html"
        
        # サーバー起動をスキップ
        if no_server:
            print("\n🌐 HTTPサーバー起動はスキップされました（--no-server 指定）")
            print(f"📄 ダッシュボードファイル: {dashboard_file}")
            if not no_browser:
                open_dashboard_safely(f"file://{dashboard_file.absolute()}")
            return

        # ブラウザ起動をスキップ
        if no_browser and not no_server:
            print("\n🌐 HTTPサーバーのみ起動中...")
            url, proc = ensure_http_server()
            if url:
                print(f"🌐 ダッシュボードURL: {url}")
                print("💡 ブラウザで上記URLにアクセスしてダッシュボードを確認してください")
            else:
                print(f"📄 ダッシュボードファイル: {dashboard_file}")
                print("💡 ファイルを直接ブラウザで開いてください")
            return

        print("\n🌐 ダッシュボード表示準備中...")
        dashboard_url, server_process = ensure_http_server()
        
        if not dashboard_url:
            print("❌ HTTPサーバーの起動に失敗しました")
            print(f"📄 ダッシュボードファイルを直接開いてください: {dashboard_file}")
            return
        
        # ダッシュボードを開く
        open_dashboard_safely(dashboard_url)
        
        # サーバープロセスの情報を表示
        if server_process:
            print(f"📋 HTTPサーバー情報:")
            print(f"   PID: {server_process.pid}")
            print(f"   URL: {dashboard_url}")
            print(f"   停止方法: kill {server_process.pid}")
            print(f"             または Ctrl+C でこのスクリプト終了時に自動停止")
            
            # スクリプト終了時にサーバーも停止するよう設定（keep_serverが指定されていない場合のみ）
            if not keep_server:
                import atexit
                def cleanup():
                    if server_process and server_process.poll() is None:
                        print("\n🛑 HTTPサーバーを停止中...")
                        try:
                            server_process.terminate()
                            server_process.wait(timeout=10)
                            print("✅ HTTPサーバーを停止しました")
                        except subprocess.TimeoutExpired:
                            print("⚠️ サーバーが応答しないため、強制kill実行")
                            server_process.kill()
                            server_process.wait()
                            print("✅ HTTPサーバーを強制停止しました")
                        except Exception as e:
                            print(f"⚠️ サーバー停止中にエラー: {e}")
                
                atexit.register(cleanup)
            
            # バッチ実行時またはサーバー常駐モード時は待機しない
            if batch_mode or keep_server:
                print("\n" + "="*60)
                print("🌐 ダッシュボードがブラウザで開かれました")
                print("📊 レポートを確認してください")
                print("="*60)
                if batch_mode:
                    print("💡 バッチ実行モードのため、HTTPサーバーは常駐します")
                else:
                    print("💡 サーバー常駐モードのため、HTTPサーバーは常駐します")
                print(f"   📋 サーバー情報: PID {server_process.pid}, URL {dashboard_url}")
                print("   🛑 停止する場合: kill {server_process.pid}")
                print("="*60)
                return
            
            # ユーザーがダッシュボードを確認できるように待機
            print("\n" + "="*60)
            print("🌐 ダッシュボードがブラウザで開かれました")
            print("📊 レポートを確認してください")
            print("="*60)
            print("💡 確認が完了したら Enter を押してスクリプトを終了してください")
            print("   または Ctrl+C で強制終了できます")
            print("="*60)
            
            try:
                input()  # Enterキーが押されるまで待機
            except KeyboardInterrupt:
                print("\n⚡ Ctrl+C が押されました")
            
            print("🔄 スクリプトを終了します...")
        else:
            print("💡 既存のHTTPサーバーを使用しています")
            print("🌐 ブラウザでダッシュボードを確認してください")
            print(f"   URL: {dashboard_url}")
            
            # 既存サーバーの場合は待機なしで終了
            print("📝 レポートの確認が完了したら、このウィンドウを閉じてください")
            
    except Exception as dashboard_error:
        print(f"⚠️ ダッシュボード表示エラー: {dashboard_error}")
        print("💡 レポートファイルは正常に生成されています")
        print(f"🌐 手動でブラウザから以下を開いてください: {REPORT_DIR}/dashboard.html")


if __name__ == "__main__":
    # ヘルプ表示
    if "--help" in sys.argv or "-h" in sys.argv:
        print("📊 Economic Dashboard Generator")
        print("")
        print("使用方法:")
        print("  python src/analysis/daily_report.py [オプション]")
        print("")
        print("オプション:")
        print("  --batch-mode      バッチ実行モード（サーバー起動・ブラウザ起動無効）")
        print("  --start-server    明示的にHTTPサーバーを起動")
        print("  --keep-server     サーバーを常駐させる")
        print("  --no-server       HTTPサーバーを起動しない")
        print("  --no-browser      ブラウザを起動しない")
        print("  --offline         オフラインモード")
        print("  --log-file FILE   ログファイルを指定")
        print("")
        print("推奨使用方法:")
        print("  1. バッチ処理: python src/analysis/daily_report.py --batch-mode")
        print("  2. サーバー管理: scripts/start_dashboard_server.sh start")
        print("  3. ダッシュボード閲覧: scripts/start_dashboard_server.sh open")
        print("")
        print("ダッシュボードHTTPサーバーの管理は scripts/start_dashboard_server.sh を使用してください。")
        sys.exit(0)
    
    main() 
