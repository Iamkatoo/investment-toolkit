import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from investment_toolkit.utilities.config import (
    FRED_API_KEY,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME
)

class FREDAPI:
    """
    Federal Reserve Economic Data (FRED) APIとの通信を扱うクラス
    主な機能:
        - APIからの経済指標データ取得
        - データの前処理
        - エラーハンドリング
    """
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, debug=False):
        """
        FREDAPIクラスの初期化
        
        パラメータ:
            debug (bool): デバッグモードを有効にするかどうか
        """
        self.logger = logging.getLogger('src.data.fred_api')
        if debug:
            self.logger.setLevel(logging.DEBUG)
            
        self.api_key = FRED_API_KEY
        self.db_engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
    def fetch_data(self, endpoint, params=None, max_retries=3, retry_delay=5):
        """
        FRED APIからデータを取得する汎用関数
        
        パラメータ:
            endpoint (str): APIエンドポイント
            params (dict): 追加のクエリパラメータ（オプション）
            max_retries (int): リトライ最大回数
            retry_delay (int): リトライ間隔（秒）
            
        戻り値:
            dict: JSON形式のレスポンス
        """
        if params is None:
            params = {}
            
        # APIキーをパラメータに追加
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        # デバッグ情報をログに出力
        debug_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items() if k != 'api_key'])}"
        self.logger.info(f"APIリクエスト: {debug_url}")
        
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.error(f"APIエラー: ステータス {response.status_code}, レスポンス: {response.text[:500]}")
                    retries += 1
                    if retries < max_retries:
                        self.logger.info(f"リトライします（{retries}/{max_retries}）... {retry_delay}秒待機")
                        time.sleep(retry_delay)
                    else:
                        self.logger.error(f"最大リトライ回数に達しました。エンドポイント: {endpoint}")
                        return None
            except Exception as e:
                self.logger.error(f"API呼び出し中にエラーが発生しました: {str(e)}")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"最大リトライ回数に達しました。エンドポイント: {endpoint}")
                    return None
                    
        return None
    
    def get_series_data(self, series_id, observation_start=None, observation_end=None):
        """
        FRED経済指標データシリーズを取得
        
        パラメータ:
            series_id (str): シリーズID（例: 'GDP', 'CPIAUCSL'）
            observation_start (str): 開始日（YYYY-MM-DD形式）
            observation_end (str): 終了日（YYYY-MM-DD形式）
            
        戻り値:
            DataFrame: 経済指標データ
        """
        endpoint = f"series/observations"
        params = {
            'series_id': series_id,
        }
        
        # データベースの最新日付を確認
        latest_date = self._get_latest_date(series_id)
        
        # 開始日が指定されていない場合
        if observation_start is None:
            if latest_date:
                # 翌日から今日までのデータを取得
                observation_start = (pd.to_datetime(latest_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                self.logger.info(f"{series_id}の最新日付（{latest_date}）以降のデータを取得します")
            else:
                # データがない場合は過去5年間のデータを取得
                observation_start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
                self.logger.info(f"{series_id}のデータが存在しないため、過去5年間のデータを取得します")
                
        params['observation_start'] = observation_start
        
        # 終了日が指定されていない場合は今日を設定
        if observation_end is None:
            observation_end = datetime.now().strftime('%Y-%m-%d')
        params['observation_end'] = observation_end
        
        data = self.fetch_data(endpoint, params)
        
        if not data or 'observations' not in data:
            self.logger.warning(f"{series_id}のデータが取得できませんでした")
            return pd.DataFrame()
            
        # DataFrameに変換
        df = pd.DataFrame(data['observations'])
        
        if df.empty:
            self.logger.warning(f"{series_id}のデータが空です")
            return df
            
        # カラム名を変換
        df = df.rename(columns={
            'date': 'date',
            'value': 'value'
        })
        
        # データ型を変換
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # indicator_nameカラムを追加
        df['indicator_name'] = series_id
        
        # 頻度を取得（シリーズ情報を取得）
        series_info = self.get_series_info(series_id)
        frequency = 'daily'  # デフォルト
        if series_info and 'frequency_short' in series_info:
            freq = series_info['frequency_short']
            if freq == 'Q':
                frequency = 'quarterly'
            elif freq in ['M', 'BM']:
                frequency = 'monthly'
            elif freq in ['D', 'BD']:
                frequency = 'daily'
            elif freq == 'A':
                # 年次データはquarterlyとして扱う（データベースの制約に合わせる）
                frequency = 'quarterly'  # 'annual'から'quarterly'に変更
                self.logger.info(f"{series_id}は年次データですが、quarterly頻度として保存します")
                
        df['frequency'] = frequency
        
        # 不要なカラムを削除
        if 'realtime_start' in df.columns:
            df = df.drop(columns=['realtime_start'])
        if 'realtime_end' in df.columns:
            df = df.drop(columns=['realtime_end'])
            
        self.logger.info(f"{series_id}のデータを取得しました（{len(df)}件）")
        
        return df
    
    def get_series_info(self, series_id):
        """
        FREDシリーズの詳細情報を取得
        
        パラメータ:
            series_id (str): シリーズID
            
        戻り値:
            dict: シリーズ情報
        """
        endpoint = f"series"
        params = {
            'series_id': series_id
        }
        
        data = self.fetch_data(endpoint, params)
        
        if not data or 'seriess' not in data or not data['seriess']:
            self.logger.warning(f"{series_id}のシリーズ情報が取得できませんでした")
            return None
            
        return data['seriess'][0]
    
    def _get_latest_date(self, indicator_name, schema='fred_data'):
        """
        指定した経済指標の最新日付を取得する内部メソッド
        
        パラメータ:
            indicator_name (str): 経済指標名
            schema (str): スキーマ名
            
        戻り値:
            str: 最新日付（YYYY-MM-DD形式）
        """
        query = f"""
            SELECT MAX(date) as latest_date
            FROM {schema}.economic_indicators
            WHERE indicator_name = :indicator_name
        """
        
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), {"indicator_name": indicator_name})
                row = result.fetchone()
                latest_date = row[0] if row and row[0] else None
                
                if latest_date:
                    return latest_date.strftime('%Y-%m-%d')
                return None
        except Exception as e:
            self.logger.error(f"最新日付取得エラー: {e}")
            return None 