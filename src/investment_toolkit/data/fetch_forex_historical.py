#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMP APIから履歴為替データを取得し、PostgreSQLデータベースに保存するスクリプト

実行方法:
    python src/data/fetch_forex_historical.py

取得対象:
    - config/forex_config.jsonに記載された通貨ペア
    - 2010-01-01から2025-02-26までの日次データ

データベース:
    - スキーマ: fmp_data
    - テーブル: forex
    - カラム: symbol, date, price, volume
"""

import os
import json
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
import time
import logging

# プロジェクト内のモジュールをインポート
from investment_toolkit.utilities.config import FMP_API_KEY_PRIMARY, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fetch_forex_historical')

# FMP API設定
BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/light"
START_DATE = "2010-01-01"
END_DATE = "2025-02-26"

# データベース接続
def connect_to_database():
    """データベースに接続するための SQLAlchemy エンジンを取得"""
    # 接続文字列の構築
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # エンジンの作成
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    
    return engine

def load_forex_symbols():
    """設定ファイルから通貨ペアのリストを読み込む"""
    try:
        # プロジェクトルートディレクトリ
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = root_dir / "config" / "forex_config.json"
        
        # 設定ファイルが存在しない場合はデフォルト値を使用
        if not config_path.exists():
            logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return ['USDJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF', 'EURJPY', 'GBPJPY']
        
        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 通貨ペアリストを取得
        if 'forex_pairs' in config and isinstance(config['forex_pairs'], list):
            return config['forex_pairs']
        else:
            logger.warning("設定ファイルから通貨ペアリストを読み込めませんでした。デフォルト値を使用します。")
            return ['USDJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF', 'EURJPY', 'GBPJPY']
    except Exception as e:
        logger.error(f"設定ファイルの読み込みエラー: {e}")
        # エラー時はデフォルト値を返す
        return ['USDJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF', 'EURJPY', 'GBPJPY']

def fetch_forex_data(symbol, from_date, to_date, api_key):
    """FMP APIから為替データを取得"""
    logger.info(f"{symbol}の為替データを取得中: {from_date}～{to_date}")
    
    # APIリクエストURL
    url = f"{BASE_URL}?symbol={symbol}&from={from_date}&to={to_date}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # エラーチェック
        
        data = response.json()
        
        if not data:
            logger.warning(f"{symbol}のデータが取得できませんでした")
            return pd.DataFrame()
        
        # データフレームに変換
        df = pd.DataFrame(data)
        
        # カラム名の変更と必要なカラムのみ選択
        df = df.rename(columns={
            'date': 'date', 
            'close': 'price',
            'volume': 'volume'
        })
        
        # symbolカラム追加
        df['symbol'] = symbol
        
        # データ型変換
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # volumeがないケースもあるため、存在確認
        if 'volume' not in df.columns:
            df['volume'] = None
        else:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 必要なカラムのみ選択
        df = df[['symbol', 'date', 'price', 'volume']]
        
        logger.info(f"{symbol}: {len(df)}行のデータを取得しました")
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"APIリクエストエラー ({symbol}): {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"データ処理エラー ({symbol}): {e}")
        return pd.DataFrame()

def save_to_database(engine, df):
    """データフレームをデータベースに保存"""
    if df.empty:
        logger.warning("保存するデータがありません")
        return
    
    try:
        # テーブル存在確認クエリ
        check_query = text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'fmp_data' AND table_name = 'forex'
        )
        """)
        
        with engine.connect() as conn:
            result = conn.execute(check_query).scalar()
            
            if not result:
                # テーブルが存在しない場合は作成
                create_query = text("""
                CREATE TABLE IF NOT EXISTS fmp_data.forex (
                    symbol VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    volume BIGINT,
                    PRIMARY KEY (symbol, date)
                )
                """)
                conn.execute(create_query)
                conn.commit()
                logger.info("テーブル fmp_data.forex を作成しました")
        
        # SQLAlchemyのto_sqlでは、主キー制約があるとエラーになるため、
        # 以下では一時テーブルを作成して、UPSERTを実行
        with engine.begin() as conn:
            # 一時テーブル名
            temp_table_name = 'temp_forex_import'
            
            # 一時テーブルにデータを挿入
            df.to_sql(temp_table_name, conn, schema='fmp_data', if_exists='replace', index=False)
            
            # UPSERTクエリの実行
            upsert_query = text(f"""
            INSERT INTO fmp_data.forex (symbol, date, price, volume)
            SELECT symbol, date, price, volume
            FROM fmp_data.{temp_table_name}
            ON CONFLICT (symbol, date) DO UPDATE 
            SET price = EXCLUDED.price, volume = EXCLUDED.volume
            """)
            
            conn.execute(upsert_query)
            
            # 一時テーブルの削除
            drop_query = text(f"DROP TABLE fmp_data.{temp_table_name}")
            conn.execute(drop_query)
        
        logger.info(f"{len(df)}行のデータをデータベースに保存しました")
    
    except Exception as e:
        logger.error(f"データベース保存エラー: {e}")

def get_earliest_date(engine, symbol):
    """指定された通貨ペアの最も古いデータの日付を取得"""
    try:
        query = text("""
        SELECT MIN(date) FROM fmp_data.forex
        WHERE symbol = :symbol
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol}).scalar()
            
        return result
    except Exception as e:
        logger.error(f"日付取得エラー: {e}")
        return None

def main():
    """メイン処理"""
    logger.info("FMP APIからの為替履歴データ取得を開始します...")
    
    # DB接続
    engine = connect_to_database()
    
    # 通貨ペアリスト取得
    symbols = load_forex_symbols()
    
    # デバッグ情報
    logger.info(f"取得する通貨ペア: {symbols}")
    
    # 各通貨ペアでデータ取得
    for symbol in symbols:
        try:
            # すでに存在している最古の日付を取得
            earliest_date = get_earliest_date(engine, symbol)
            
            if earliest_date:
                # 日付オブジェクトを文字列に変換
                if isinstance(earliest_date, datetime):
                    earliest_date = earliest_date.strftime('%Y-%m-%d')
                else:
                    earliest_date = earliest_date.isoformat()
                    
                logger.info(f"{symbol}の最古のデータ日付: {earliest_date}")
                
                # 既存データより古いデータのみ取得
                if earliest_date <= START_DATE:
                    logger.info(f"{symbol}は既に十分な履歴データが存在します。処理をスキップします。")
                    continue
                
                # 取得終了日を既存データの開始日の前日に設定
                to_date = (datetime.strptime(earliest_date, '%Y-%m-%d')).strftime('%Y-%m-%d')
            else:
                # データが存在しない場合は全期間取得
                to_date = END_DATE
            
            # データ取得
            df = fetch_forex_data(symbol, START_DATE, to_date, FMP_API_KEY_PRIMARY)
            
            if not df.empty:
                # データベースに保存
                save_to_database(engine, df)
            
            # API制限を考慮して少し待機
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"{symbol}の処理中にエラーが発生: {e}")
    
    logger.info("すべての通貨ペアの履歴データ取得が完了しました")

if __name__ == "__main__":
    main() 