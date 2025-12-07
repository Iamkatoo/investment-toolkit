#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ウォッチリスト銘柄のミニチャート用JSON生成スクリプト
Plotly staticPlot用の軽量なJSONデータを生成

実行方法:
    python src/analysis/generate_mini_json.py

出力:
    reports/mini_json/ ディレクトリに各銘柄のJSONファイルが保存される
"""

import os
import sys
from datetime import date, timedelta
import pandas as pd
import numpy as np
import json
import time
import logging
from pathlib import Path
from sqlalchemy import create_engine, text

# TALibの代替として独自計算を使用
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("警告: TALibが利用できません。独自計算を使用します。")

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内のモジュールをインポート
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

# ログ設定
LOG_DIR = project_root / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "mini_json.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
LOG = logging.getLogger("mini_json")

# 設定
LOOKBACK = 120  # 足数（約半年）
OUT_DIR = project_root / "reports" / "mini_json"
OUT_DIR.mkdir(exist_ok=True)


def connect_to_database():
    """データベースに接続するための SQLAlchemy エンジンを取得"""
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    return engine


def fetch_watchlist_symbols(engine):
    """ウォッチリスト銘柄と日次ランキング銘柄を取得"""
    # ウォッチリスト銘柄を取得
    watchlist_query = text("""
        SELECT DISTINCT symbol 
        FROM watchlist.tracked_stocks 
        WHERE is_active = true
        ORDER BY symbol
    """)
    
    # 日次ランキング上位銘柄を取得（トップ10）
    ranking_query = text("""
        WITH latest_date AS (
            SELECT MAX(date) as max_date
            FROM fmp_data.daily_prices
            WHERE change_percent IS NOT NULL
        )
        SELECT dp.symbol, dp.change_percent
        FROM fmp_data.daily_prices dp
        INNER JOIN latest_date ld ON dp.date = ld.max_date
        WHERE dp.change_percent IS NOT NULL
            AND dp.change_percent > 0
            AND dp.volume > 10000
            AND dp.close > 1.0
        ORDER BY dp.change_percent DESC
        LIMIT 10
    """)
    
    symbols = set()
    
    with engine.connect() as conn:
        # ウォッチリスト銘柄
        result = conn.execute(watchlist_query)
        watchlist_symbols = [row[0] for row in result.fetchall()]
        symbols.update(watchlist_symbols)
        
        # ランキング銘柄
        result = conn.execute(ranking_query)
        ranking_symbols = [row[0] for row in result.fetchall()]
        symbols.update(ranking_symbols)
    
    LOG.info(f"ウォッチリスト銘柄: {len(watchlist_symbols)}件")
    LOG.info(f"ランキング銘柄: {len(ranking_symbols)}件")
    LOG.info(f"合計対象銘柄: {len(symbols)}件（重複除去後）")
    
    return list(symbols)


def fetch_watchlist_added_dates(engine):
    """ウォッチリスト銘柄の追加日情報を取得"""
    query = text("""
        SELECT symbol, added_date, analysis_type
        FROM watchlist.tracked_stocks 
        WHERE is_active = true
        ORDER BY symbol, added_date
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        watchlist_info = {}
        for row in result.fetchall():
            symbol = row[0]
            added_date = row[1].strftime('%Y-%m-%d') if row[1] else None
            analysis_type = row[2]
            
            if symbol not in watchlist_info:
                watchlist_info[symbol] = []
            
            watchlist_info[symbol].append({
                'added_date': added_date,
                'analysis_type': analysis_type
            })
    
    return watchlist_info


def fetch_prices_with_indicators(engine, symbols):
    """株価データとテクニカル指標を取得（データベースから直接）"""
    if not symbols:
        return pd.DataFrame()
    
    since = date.today() - timedelta(days=LOOKBACK * 2)  # 休日補正で余裕を見る
    
    # 株価データとテクニカル指標を結合して取得
    query = text("""
        SELECT 
            p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
            ti.sma_20, ti.sma_40, ti.rsi_14, ti.macd_hist
        FROM fmp_data.daily_prices p
        LEFT JOIN calculated_metrics.technical_indicators ti 
            ON p.symbol = ti.symbol AND p.date = ti.date
        WHERE p.symbol = ANY(:symbols) AND p.date >= :since
        ORDER BY p.symbol, p.date
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"symbols": symbols, "since": since})
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['symbol', 'date'])
    
    return df


def calculate_sma(prices, period):
    """単純移動平均を計算"""
    if TALIB_AVAILABLE:
        return ta.SMA(prices, period)
    else:
        # pandas rolling mean
        return pd.Series(prices).rolling(window=period, min_periods=1).mean().values


def calculate_rsi(prices, period=14):
    """RSIを計算"""
    if TALIB_AVAILABLE:
        return ta.RSI(prices, period)
    else:
        # 独自RSI計算
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACDを計算"""
    if TALIB_AVAILABLE:
        macd, signal_line, histogram = ta.MACD(prices, fast, slow, signal)
        return macd, signal_line, histogram
    else:
        # 独自MACD計算
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd.values, signal_line.values, histogram.values


def get_indicators_from_db(df_sym):
    """データベースから取得したテクニカル指標を整理"""
    if df_sym.empty:
        return None, None, None, None
    
    try:
        # データベースから取得した指標を使用
        rsi = df_sym['rsi_14'].values if 'rsi_14' in df_sym.columns else None
        macd_hist = df_sym['macd_hist'].values if 'macd_hist' in df_sym.columns else None
        sma20 = df_sym['sma_20'].values if 'sma_20' in df_sym.columns else None
        sma40 = df_sym['sma_40'].values if 'sma_40' in df_sym.columns else None
        
        # データベースに指標がない場合はフォールバック計算
        if rsi is None or pd.isna(rsi).all():
            LOG.info(f"RSIデータがないため計算中...")
            rsi = calculate_rsi(df_sym['close'].values, 14)
        
        if macd_hist is None or pd.isna(macd_hist).all():
            LOG.info(f"MACDデータがないため計算中...")
            _, _, macd_hist = calculate_macd(df_sym['close'].values, 12, 26, 9)
        
        if sma20 is None or pd.isna(sma20).all():
            LOG.info(f"SMA20データがないため計算中...")
            sma20 = calculate_sma(df_sym['close'].values, 20)
        
        if sma40 is None or pd.isna(sma40).all():
            LOG.info(f"SMA40データがないため計算中...")
            sma40 = calculate_sma(df_sym['close'].values, 40)
        
        return rsi, macd_hist, sma20, sma40
    except Exception as e:
        LOG.warning(f"指標取得エラー: {e}")
        return None, None, None, None


def export_symbol(symbol, df, watchlist_info):
    """シンボルのJSONを出力"""
    try:
        # 最新のLOOKBACK足のデータに限定
        df = df.tail(LOOKBACK).reset_index(drop=True)
        
        if df.empty:
            LOG.warning(f"{symbol}: データが不足しています")
            return False
        
        # 指標をデータベースから取得
        rsi, hist, sma20, sma40 = get_indicators_from_db(df)
        
        if rsi is None:
            LOG.warning(f"{symbol}: 指標計算に失敗しました")
            return False
        
        # NaN値を処理
        def clean_data(data):
            if data is None:
                return []
            cleaned = []
            for x in data:
                if pd.isna(x) or np.isnan(x) or np.isinf(x):
                    cleaned.append(None)
                else:
                    cleaned.append(float(x))
            return cleaned
        
        # JSONペイロードを作成
        payload = {
            "symbol": symbol,
            "date": df['date'].dt.strftime("%Y-%m-%d").tolist(),
            "open": clean_data(df['open'].values),
            "high": clean_data(df['high'].values),
            "low": clean_data(df['low'].values),
            "close": clean_data(df['close'].values),
            "volume": clean_data(df['volume'].values),
            "sma20": clean_data(sma20),
            "sma40": clean_data(sma40),
            "rsi": clean_data(rsi),
            "macd_hist": clean_data(hist),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "watchlist_info": watchlist_info  # ウォッチリスト追加日情報を追加
        }
        
        # ファイルに保存
        output_file = OUT_DIR / f"{symbol}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))
        
        file_size = output_file.stat().st_size
        LOG.info(f"{symbol}: JSON生成完了 ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        LOG.error(f"{symbol}: JSONエクスポートエラー - {e}")
        return False


def cleanup_old_files(symbols):
    """不要な古いJSONファイルを削除"""
    existing_files = list(OUT_DIR.glob("*.json"))
    active_files = {f"{symbol}.json" for symbol in symbols}
    
    deleted_count = 0
    for file_path in existing_files:
        if file_path.name not in active_files:
            try:
                file_path.unlink()
                deleted_count += 1
                LOG.info(f"古いファイルを削除: {file_path.name}")
            except Exception as e:
                LOG.warning(f"ファイル削除エラー: {file_path.name} - {e}")
    
    if deleted_count > 0:
        LOG.info(f"古いファイル {deleted_count} 件を削除しました")


def main():
    """メイン処理"""
    start_time = time.perf_counter()
    LOG.info("=== ミニチャート用JSON生成開始 ===")
    
    try:
        # データベース接続
        engine = connect_to_database()
        
        # ウォッチリスト銘柄を取得
        symbols = fetch_watchlist_symbols(engine)
        if not symbols:
            LOG.warning("アクティブなウォッチリスト銘柄が見つかりません")
            return
        
        LOG.info(f"対象銘柄数: {len(symbols)}")
        LOG.info(f"対象銘柄: {', '.join(symbols)}")
        
        # ウォッチリスト追加日情報を取得
        watchlist_added_dates = fetch_watchlist_added_dates(engine)
        
        # 株価データとテクニカル指標を取得
        LOG.info("株価データとテクニカル指標を取得中...")
        df_all = fetch_prices_with_indicators(engine, symbols)
        
        if df_all.empty:
            LOG.warning("株価データが取得できませんでした")
            return
        
        # 銘柄ごとにJSONを生成
        success_count = 0
        total_size = 0
        
        for symbol in symbols:
            symbol_start = time.perf_counter()
            
            # 銘柄データを抽出
            df_sym = df_all[df_all['symbol'] == symbol].copy()
            
            if df_sym.empty:
                LOG.warning(f"{symbol}: データが見つかりません")
                continue
            
            # ウォッチリスト追加日情報を取得
            watchlist_info = watchlist_added_dates.get(symbol, [])
            
            # JSONを生成
            if export_symbol(symbol, df_sym, watchlist_info):
                success_count += 1
                # ファイルサイズを集計
                json_file = OUT_DIR / f"{symbol}.json"
                if json_file.exists():
                    total_size += json_file.stat().st_size
            
            symbol_time = time.perf_counter() - symbol_start
            LOG.info(f"{symbol}: 処理完了 ({symbol_time:.3f}s)")
        
        # 古いファイルをクリーンアップ
        cleanup_old_files(symbols)
        
        # 結果をログ出力
        total_time = time.perf_counter() - start_time
        LOG.info("=== 処理結果 ===")
        LOG.info(f"成功: {success_count}/{len(symbols)} 銘柄")
        LOG.info(f"総ファイルサイズ: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        LOG.info(f"処理時間: {total_time:.3f}s")
        LOG.info(f"平均処理時間/銘柄: {total_time/len(symbols):.3f}s")
        
        if success_count > 0:
            LOG.info(f"ミニチャート用JSONファイルを {OUT_DIR} に保存しました")
        
    except Exception as e:
        LOG.error(f"処理中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main() 