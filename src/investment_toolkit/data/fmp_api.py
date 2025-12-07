import requests
import pandas as pd
import json
import os
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from investment_toolkit.utilities.config import (
    FMP_API_KEY_PRIMARY as API_KEY,
    FMP_API_KEY_SECONDARY as API_KEY_SECONDARY,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME
)
from .database_utils import preprocess_dataframe, check_fmp_bandwidth_limit, camel_to_snake
import traceback
import re
import sys
from sqlalchemy.exc import SQLAlchemyError

class APIRateLimiter:
    def __init__(self, max_requests_per_minute=290):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.cache = {}
        self.cache_ttl = 300  # キャッシュの有効期限（秒）
        
    def wait_if_needed(self):
        """レート制限に基づいて待機時間を計算"""
        now = time.time()
        
        # 1分以上前のリクエストを削除
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            # 最も古いリクエストから1分経過するまでの時間を計算
            wait_time = 60 - (now - self.requests[0])
            if wait_time > 0:
                time.sleep(wait_time)
                # 待機後に再度チェック
                return self.wait_if_needed()
        
        self.requests.append(now)
        
    def get_cached_response(self, key):
        """キャッシュからレスポンスを取得"""
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                del self.cache[key]
        return None
        
    def cache_response(self, key, data):
        """レスポンスをキャッシュに保存"""
        self.cache[key] = (time.time(), data)
        
    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache.clear()

class FMPAPI:
    """
    Financial Modeling Prep (FMP) APIとの通信を扱うクラス
    主な機能:
        - APIからの財務データ取得
        - データの前処理
        - エラーハンドリング（APIキー切り替え、リトライなど）
    """
    BASE_URL_STABLE = "https://financialmodelingprep.com/stable"
    BASE_URL_V4 = "https://financialmodelingprep.com/api/v4"
    
    def __init__(self, debug=False):
        """
        FMPAPIクラスの初期化
        
        パラメータ:
            debug (bool): デバッグモードを有効にするかどうか
        """
        self.logger = logging.getLogger('src.data.fmp_api')
        # 常に詳細なログを出力するようにDEBUGレベルを設定
        self.logger.setLevel(logging.DEBUG)
        
        # ログフォーマットの設定
        for handler in self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
        
        # APIキーの設定
        self.api_keys = []
        if API_KEY:
            self.api_keys.append(API_KEY)
        if API_KEY_SECONDARY:
            self.api_keys.append(API_KEY_SECONDARY)
        self.current_key_index = 0
        self.db_engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 問題のある銘柄を追跡するリスト
        self.problematic_symbols = set(['PACI-WT'])  # 初期値として問題が知られている銘柄を追加
        
        # API使用量の追跡
        self.total_data_size = 0
        self.request_count = 0
        
        # エラー統計の初期化
        self.error_stats = {
            'rate_limit': 0,
            'not_found': 0,
            'server_error': 0,
            'client_error': 0,
            'network_error': 0,
            'parse_error': 0,
            'other_error': 0
        }
        
        # エラー履歴の保持（直近100件）
        self.error_history = []
        
        self.rate_limiter = APIRateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
    
    @property
    def current_api_key(self):
        """現在使用中のAPIキーを返す"""
        return self.api_keys[self.current_key_index]
    
    def switch_api_key(self):
        """APIキーを切り替える"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.logger.info(f"APIキーを切り替えました: index={self.current_key_index}")
        return self.current_api_key
    
    def _calculate_data_size(self, data):
        """
        データのサイズをバイト単位で計算
        
        パラメータ:
            data: サイズを測定するデータ
            
        戻り値:
            int: データサイズ（バイト）
        """
        if isinstance(data, (str, bytes)):
            return len(data)
        else:
            # 文字列以外の場合はJSONに変換してサイズを測定
            try:
                json_str = json.dumps(data)
                return len(json_str.encode('utf-8'))
            except:
                # 変換できない場合は近似値を返す
                return sys.getsizeof(str(data))
    
    def _format_size(self, size_bytes):
        """
        バイトサイズを人間が読みやすい形式に変換
        
        パラメータ:
            size_bytes (int): バイト単位のサイズ
            
        戻り値:
            str: 読みやすい形式のサイズ（KB、MB、GBなど）
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"
    
    def _calculate_backoff_time(self, retry_count, base_delay=1):
        """
        指数バックオフ時間を計算
        
        パラメータ:
            retry_count (int): 現在のリトライ回数
            base_delay (int): 基本待機時間（秒）
            
        戻り値:
            float: 待機時間（秒）
        """
        # 最大待機時間を30秒に制限
        return min(base_delay * (2 ** retry_count), 30)
    
    def _log_error(self, error_type, symbol, error_details):
        """
        エラー情報を構造化して記録
        
        パラメータ:
            error_type (str): エラーの種類
            symbol (str): 対象のシンボル
            error_details (dict): エラーの詳細情報
        """
        # エラー統計を更新
        if error_type in self.error_stats:
            self.error_stats[error_type] += 1
            
        # エラー履歴に追加
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'symbol': symbol,
            'details': error_details
        }
        self.error_history.append(error_entry)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
            
        # エラーログを出力
        self.logger.error(f"エラー発生: {error_type} - シンボル: {symbol}")
        self.logger.error(f"エラー詳細: {json.dumps(error_details, indent=2)}")
        
    def _make_request(self, endpoint, params=None, use_cache=True):
        """APIリクエストを実行（キャッシュ対応）"""
        if params is None:
            params = {}
            
        params['apikey'] = self.current_api_key
        url = f"{self.BASE_URL_V4}/{endpoint}"
        
        # キャッシュキーの生成
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        
        # キャッシュからデータを取得
        if use_cache:
            cached_data = self.rate_limiter.get_cached_response(cache_key)
            if cached_data is not None:
                return cached_data
        
        # レート制限のチェック
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # レスポンスをキャッシュ
            if use_cache:
                self.rate_limiter.cache_response(cache_key, data)
                
            return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"APIリクエストエラー: {e}")
            return None

    def fetch_data(self, endpoint, params=None, max_retries=3, retry_delay=5, use_v4=False, use_legacy=False):
        """
        FMP APIからデータを取得する汎用関数
        
        パラメータ:
            endpoint (str): APIエンドポイント（例: '/income-statement/AAPL'）
            params (dict): 追加のクエリパラメータ（オプション）
            max_retries (int): リトライ最大回数
            retry_delay (int): リトライ間隔（秒）
            use_v4 (bool): API v4を使用するかどうか
            use_legacy (bool): レガシーエンドポイント形式を使用する
            
        戻り値:
            dict: JSON形式のレスポンス
        """
        if params is None:
            params = {}
        
        # シンボルが問題のある銘柄リストに含まれている場合は即座にスキップ
        symbol = params.get('symbol', '')
        if symbol in self.problematic_symbols:
            self.logger.warning(f"問題のある銘柄 {symbol} はスキップします（過去のエラー履歴あり）")
            return None
        
        # APIキーをパラメータに追加
        params['apikey'] = self.current_api_key
        
        # 詳細ログ: APIキーの状態を常に記録
        self.logger.info(f"APIキー値: {self.current_api_key[:4]}...{self.current_api_key[-4:]}")
        self.logger.info(f"パラメータ一覧: {', '.join([f'{k}={v}' for k, v in params.items() if k != 'apikey'])}")
        
        # リクエストのタイムアウト値（秒）- 短めに設定
        timeout = 10
        
        retries = 0
        while retries < max_retries:
            request_start_time = datetime.now()
            try:
                # API v4かstableかによってベースURLを切り替え
                if use_legacy:
                    base_url = "https://financialmodelingprep.com"
                    url = f"{base_url}{endpoint}"
                else:
                    base_url = self.BASE_URL_V4 if use_v4 else self.BASE_URL_STABLE
                    url = f"{base_url}{endpoint}"
                
                # URLとパラメータを常に詳細ログに出力
                masked_params = params.copy()
                if 'apikey' in masked_params:
                    masked_key = self.current_api_key[:4] + "..." + self.current_api_key[-4:]
                    masked_params['apikey'] = masked_key
                
                debug_params_str = '&'.join([f'{k}={v}' for k, v in masked_params.items()])
                full_debug_url = f"{url}?{debug_params_str}"
                self.logger.info(f"APIリクエスト: {full_debug_url}")
                
                # 本物のパラメータを使ってリクエストを実行（タイムアウト付き）
                self.logger.info(f"API呼び出し開始: {request_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} （タイムアウト: {timeout}秒）")
                response = requests.get(url, params=params, timeout=timeout)
                request_end_time = datetime.now()
                request_duration = (request_end_time - request_start_time).total_seconds()
                self.logger.info(f"API呼び出し完了: {request_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} （所要時間: {request_duration:.3f}秒）")
                
                # 実際に送信されたURLと応答ステータスを常に詳細ログに出力
                self.logger.info(f"送信されたURL: {response.url}")
                self.logger.info(f"応答ステータス: {response.status_code} {response.reason}")
                
                # リクエスト回数をカウント
                self.request_count += 1
                
                # レスポンスヘッダを表示
                self.logger.debug("レスポンスヘッダ:")
                for key, value in response.headers.items():
                    self.logger.debug(f"  {key}: {value}")
                
                # データサイズを計測
                content_length = response.headers.get('Content-Length')
                if content_length:
                    data_size = int(content_length)
                else:
                    data_size = len(response.content)
                
                # データサイズをログに記録
                human_readable_size = self._format_size(data_size)
                self.logger.info(f"レスポンスデータサイズ: {human_readable_size} ({data_size} bytes)")
                
                # 合計データサイズを更新
                self.total_data_size += data_size
                total_human_readable = self._format_size(self.total_data_size)
                self.logger.info(f"累計データ通信量: {total_human_readable} ({self.request_count}リクエスト)")
                
                # レスポンスのステータスコードをチェック
                if response.status_code == 200:
                    # 正常なレスポンスを処理
                    pass
                elif response.status_code == 429:
                    # レート制限に達した場合
                    error_details = {
                        'status_code': response.status_code,
                        'response_text': response.text,
                        'retry_count': retries,
                        'max_retries': max_retries
                    }
                    self._log_error('rate_limit', symbol, error_details)
                    
                    # APIキーを切り替え
                    if len(self.api_keys) > 1:
                        prev_key = self.current_api_key
                        new_key = self.switch_api_key()
                        params['apikey'] = new_key
                        self.logger.info(f"APIキーを切り替えました: {prev_key[:4]}... → {new_key[:4]}...")
                    
                    retries += 1
                    if retries < max_retries:
                        backoff_time = self._calculate_backoff_time(retries, retry_delay)
                        self.logger.info(f"リトライします（{retries}/{max_retries}）... {backoff_time}秒待機")
                        time.sleep(backoff_time)
                        continue
                    else:
                        self.logger.error(f"最大リトライ回数に達しました。エンドポイント: {endpoint}")
                        return None
                
                # クライアント側のエラー（4xx）またはサーバー側のエラー（5xx）
                elif response.status_code >= 400:
                    error_type = 'not_found' if response.status_code == 404 else 'client_error'
                    if response.status_code >= 500:
                        error_type = 'server_error'
                        
                    error_details = {
                        'status_code': response.status_code,
                        'response_text': response.text,
                        'retry_count': retries,
                        'max_retries': max_retries
                    }
                    self._log_error(error_type, symbol, error_details)
                    
                    # 銘柄が問題の原因の場合、問題のある銘柄リストに追加
                    if response.status_code == 404 and symbol:
                        self.problematic_symbols.add(symbol)
                        self.logger.warning(f"銘柄 {symbol} を問題のある銘柄リストに追加しました（404エラー）")
                    
                    # サーバーエラーの場合はリトライ
                    if response.status_code >= 500:
                        retries += 1
                        if retries < max_retries:
                            backoff_time = self._calculate_backoff_time(retries, retry_delay)
                            self.logger.info(f"リトライします（{retries}/{max_retries}）... {backoff_time}秒待機")
                            time.sleep(backoff_time)
                            continue
                    
                    return None
                
                try:
                    # 正常なレスポンスの場合はJSONを返す
                    json_data = response.json()
                    
                    # レスポンスの概要を常に詳細ログに出力
                    if isinstance(json_data, list):
                        self.logger.info(f"レスポンス: リスト（{len(json_data)}件）")
                        if len(json_data) > 0:
                            sample = json_data[0]
                            if isinstance(sample, dict):
                                self.logger.info(f"サンプル項目のキー: {', '.join(list(sample.keys())[:5])}...")
                    elif isinstance(json_data, dict):
                        self.logger.info(f"レスポンス: 辞書（{len(json_data)}キー）")
                        self.logger.info(f"レスポンスのキー: {', '.join(list(json_data.keys())[:5])}...")
                    
                    return json_data
                
                except json.JSONDecodeError as e:
                    # JSONの解析エラー
                    error_details = {
                        'error': str(e),
                        'response_text': response.text[:100] + '...' if len(response.text) > 100 else response.text
                    }
                    self._log_error('parse_error', symbol, error_details)
                    
                    # 空のレスポンスの場合は空のリストを返す
                    if not response.text.strip():
                        self.logger.warning("空のレスポンスを受信しました")
                        return []
                    
                    return None
            
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # タイムアウトまたは接続エラーの場合はリトライ
                error_details = {
                    'error': str(e),
                    'retry_count': retries,
                    'max_retries': max_retries
                }
                self._log_error('network_error', symbol, error_details)
                
                retries += 1
                if retries < max_retries:
                    backoff_time = self._calculate_backoff_time(retries, retry_delay)
                    self.logger.info(f"リトライします（{retries}/{max_retries}）... {backoff_time}秒待機")
                    time.sleep(backoff_time)
                    continue
                else:
                    self.logger.error(f"最大リトライ回数に達しました。エンドポイント: {endpoint}")
                    if symbol:
                        self.problematic_symbols.add(symbol)
                        self.logger.warning(f"銘柄 {symbol} を問題のある銘柄リストに追加しました")
                    return None
            
            except Exception as e:
                # その他の例外
                error_details = {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'retry_count': retries,
                    'max_retries': max_retries
                }
                self._log_error('other_error', symbol, error_details)
                
                if symbol:
                    self.problematic_symbols.add(symbol)
                    self.logger.warning(f"銘柄 {symbol} を問題のある銘柄リストに追加しました")
                return None
                
        return None

    def to_dataframe(self, data):
        """
        APIレスポンスをDataFrameに変換
        
        パラメータ:
            data: APIから取得したデータ（辞書またはリスト）
            
        戻り値:
            DataFrame: 変換後のDataFrame
        """
        import pandas as pd
        
        if data is None:
            return pd.DataFrame()
            
        if isinstance(data, dict):
            # エラーメッセージが含まれている場合
            if "Error Message" in data:
                self.logger.warning(f"APIエラー: {data['Error Message']}")
                return pd.DataFrame()
                
            # 単一オブジェクトの場合はリストに変換
            data = [data]
            
        # 空のリストまたは空の辞書の場合
        if not data:
            return pd.DataFrame()
        
        # リストをDataFrameに変換
        try:
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            self.logger.error(f"DataFrameへの変換エラー: {e}")
            return pd.DataFrame()

    def get_income_statements(self, symbol, period='annual', limit=None, is_japanese_stock=False):
        """
        損益計算書を取得
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            limit (int): 取得するデータの制限数（Noneの場合は最新日付から計算）
            is_japanese_stock (bool): 日本株かどうか
            
        戻り値:
            list or DataFrame: 損益計算書データ
        """
        if limit is None:
            # データベースから最新日付を取得
            latest_date = self._get_latest_date(symbol, 'income_statements', period_type=period)
            if latest_date:
                latest_date = pd.to_datetime(latest_date)
                today = pd.Timestamp.now()
                
                if period == 'annual':
                    # 年次データの場合、年数に1を加えて余裕を持たせる
                    years_diff = (today.year - latest_date.year) + 1
                    limit = max(1, years_diff + 1)  # 最低1年分は取得
                else:  # quarter
                    # 四半期データの場合、四半期数に2を加えて余裕を持たせる
                    quarters_diff = ((today.year - latest_date.year) * 4 + 
                                   (today.quarter - latest_date.quarter)) + 2
                    limit = max(4, quarters_diff + 2)  # 最低4四半期分は取得
            else:
                # データがない場合はデフォルト値を設定
                limit = 4 if period == 'quarter' else 3
        
        # 日本株（.Tで終わる銘柄）または明示的に指定された場合は旧エンドポイントを使用
        if is_japanese_stock or symbol.endswith('.T'):
            self.logger.info(f"{symbol}は日本株のため、財務諸表データには旧エンドポイントを使用します")
            legacy_endpoint = f"/api/v3/income-statement/{symbol}"
            legacy_params = {
                'period': period,
                'limit': limit
            }
            return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
        else:
            # 通常の銘柄は標準エンドポイントを使用
            endpoint = f"/income-statement/{symbol}"
            params = {
                'period': period,
                'limit': limit
            }
            
            data = self.fetch_data(endpoint, params)
            
            # サブスクリプションエラーの場合、旧エンドポイントを試す
            if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
                self.logger.warning(f"{symbol}の損益計算書データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
                legacy_endpoint = f"/api/v3/income-statement/{symbol}"
                legacy_params = {
                    'period': period,
                    'limit': limit
                }
                return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
            return data

    def get_balance_sheets(self, symbol, period='annual', limit=None, is_japanese_stock=False):
        """
        貸借対照表を取得
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            limit (int): 取得するデータの制限数（Noneの場合は最新日付から計算）
            is_japanese_stock (bool): 日本株かどうか
            
        戻り値:
            list or DataFrame: 貸借対照表データ
        """
        if limit is None:
            # データベースから最新日付を取得
            latest_date = self._get_latest_date(symbol, 'balance_sheets', period_type=period)
            if latest_date:
                latest_date = pd.to_datetime(latest_date)
                today = pd.Timestamp.now()
                
                if period == 'annual':
                    # 年次データの場合、年数に1を加えて余裕を持たせる
                    years_diff = (today.year - latest_date.year) + 1
                    limit = max(1, years_diff + 1)  # 最低1年分は取得
                else:  # quarter
                    # 四半期データの場合、四半期数に2を加えて余裕を持たせる
                    quarters_diff = ((today.year - latest_date.year) * 4 + 
                                   (today.quarter - latest_date.quarter)) + 2
                    limit = max(4, quarters_diff + 2)  # 最低4四半期分は取得
            else:
                # データがない場合はデフォルト値を設定
                limit = 4 if period == 'quarter' else 3
        
        # 日本株（.Tで終わる銘柄）または明示的に指定された場合は旧エンドポイントを使用
        if is_japanese_stock or symbol.endswith('.T'):
            self.logger.info(f"{symbol}は日本株のため、財務諸表データには旧エンドポイントを使用します")
            legacy_endpoint = f"/api/v3/balance-sheet-statement/{symbol}"
            legacy_params = {
                'period': period,
                'limit': limit
            }
            return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
        else:
            # 通常の銘柄は標準エンドポイントを使用
            endpoint = f"/balance-sheet-statement/{symbol}"
            params = {
                'period': period,
                'limit': limit
            }
            
            data = self.fetch_data(endpoint, params)
            
            # サブスクリプションエラーの場合、旧エンドポイントを試す
            if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
                self.logger.warning(f"{symbol}の貸借対照表データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
                legacy_endpoint = f"/api/v3/balance-sheet-statement/{symbol}"
                legacy_params = {
                    'period': period,
                    'limit': limit
                }
                return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
            return data

    def get_cash_flows(self, symbol, period='annual', limit=None, is_japanese_stock=False):
        """
        キャッシュフロー計算書を取得
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            limit (int): 取得するデータの制限数（Noneの場合は最新日付から計算）
            is_japanese_stock (bool): 日本株かどうか
            
        戻り値:
            list or DataFrame: キャッシュフロー計算書データ
        """
        if limit is None:
            # データベースから最新日付を取得
            latest_date = self._get_latest_date(symbol, 'cash_flows', period_type=period)
            if latest_date:
                latest_date = pd.to_datetime(latest_date)
                today = pd.Timestamp.now()
                
                if period == 'annual':
                    # 年次データの場合、年数に1を加えて余裕を持たせる
                    years_diff = (today.year - latest_date.year) + 1
                    limit = max(1, years_diff + 1)  # 最低1年分は取得
                else:  # quarter
                    # 四半期データの場合、四半期数に2を加えて余裕を持たせる
                    quarters_diff = ((today.year - latest_date.year) * 4 + 
                                   (today.quarter - latest_date.quarter)) + 2
                    limit = max(4, quarters_diff + 2)  # 最低4四半期分は取得
            else:
                # データがない場合はデフォルト値を設定
                limit = 4 if period == 'quarter' else 3
        
        # 日本株（.Tで終わる銘柄）または明示的に指定された場合は旧エンドポイントを使用
        if is_japanese_stock or symbol.endswith('.T'):
            self.logger.info(f"{symbol}は日本株のため、財務諸表データには旧エンドポイントを使用します")
            legacy_endpoint = f"/api/v3/cash-flow-statement/{symbol}"
            legacy_params = {
                'period': period,
                'limit': limit
            }
            return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
        else:
            # 通常の銘柄は標準エンドポイントを使用
            endpoint = f"/cash-flow-statement/{symbol}"
            params = {
                'period': period,
                'limit': limit
            }
            
            data = self.fetch_data(endpoint, params)
            
            # サブスクリプションエラーの場合、旧エンドポイントを試す
            if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
                self.logger.warning(f"{symbol}のキャッシュフロー計算書データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
                legacy_endpoint = f"/api/v3/cash-flow-statement/{symbol}"
                legacy_params = {
                    'period': period,
                    'limit': limit
                }
                return self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
            return data

    def get_company_profile(self, symbol):
        """
        会社プロファイルデータを取得
        
        パラメータ:
            symbol (str): 証券コード
            
        戻り値:
            DataFrame: 会社プロファイルデータ
        """
        endpoint = f"/profile"
        params = {
            'symbol': symbol
        }
        data = self.fetch_data(endpoint, params)
        
        # サブスクリプションエラーの場合、旧エンドポイントを試す
        if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
            self.logger.warning(f"{symbol}の会社プロファイルデータ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
            # 旧エンドポイント（APIv3）を使用して再試行
            legacy_endpoint = f"/api/v3/profile/{symbol}"
            data = self.fetch_data(legacy_endpoint, {}, use_legacy=True)
            
        return self.to_dataframe(data)
    
    def get_historical_price(self, symbol, from_date=None, to_date=None):
        """
        株価の履歴データを取得
        
        パラメータ:
            symbol (str): 証券コード
            from_date (str): 開始日（YYYY-MM-DD形式）
            to_date (str): 終了日（YYYY-MM-DD形式）
            
        戻り値:
            DataFrame: 株価履歴データ
        """
        # データベースの最新日付を確認
        if from_date is None:
            latest_date = self._get_latest_date(symbol, 'daily_prices')
            if latest_date:
                # 翌日から今日までのデータを取得
                from_date = (pd.to_datetime(latest_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                self.logger.info(f"{symbol}の株価データ: 最新日付（{latest_date}）以降のデータを取得します")
            else:
                # データがない場合は直近1年間のデータを取得
                from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                self.logger.info(f"{symbol}の株価データ: 過去1年間のデータを取得します")
        
        # 終了日が指定されていない場合は今日を設定
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
            
        # 日付の前後関係を確認
        if from_date and to_date and pd.to_datetime(from_date) > pd.to_datetime(to_date):
            self.logger.warning(f"開始日({from_date})が終了日({to_date})より後になっています。データは取得されません。")
            return pd.DataFrame()
            
        # 開始日と終了日が同じ場合は警告
        if from_date and to_date and from_date == to_date:
            self.logger.warning(f"開始日と終了日が同じ({from_date})です。データが少ない可能性があります。")
        
        # 日本株（.Tで終わる銘柄）は直接旧エンドポイントを使用
        use_legacy_first = symbol.endswith('.T')
        
        if use_legacy_first:
            self.logger.info(f"{symbol}は日本株のため、直接旧エンドポイントを使用します")
            legacy_endpoint = f"/api/v3/historical-price-full/{symbol}"
            legacy_params = {}
            if from_date:
                legacy_params['from'] = from_date
            if to_date:
                legacy_params['to'] = to_date
                
            data = self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
        else:
            # 通常の銘柄は標準エンドポイントを試す
            endpoint = f"/historical-price-eod/full"
            params = {
                'symbol': symbol
            }
            
            # from_dateとto_dateが指定されている場合は追加
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
                
            data = self.fetch_data(endpoint, params)
            
            # サブスクリプションエラーの場合、旧エンドポイントを試す
            if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
                self.logger.warning(f"{symbol}の株価履歴データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
                # 旧エンドポイント（APIv3）を使用して再試行
                legacy_endpoint = f"/api/v3/historical-price-full/{symbol}"
                legacy_params = {}
                if from_date:
                    legacy_params['from'] = from_date
                if to_date:
                    legacy_params['to'] = to_date
                    
                data = self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
        # データがない場合は空のDataFrameを返す
        if not data:
            self.logger.warning(f"{symbol}の株価履歴データが取得できませんでした")
            return pd.DataFrame()
            
        # データ形式の違いを吸収
        if isinstance(data, dict):
            if 'historical' in data:
                historical_data = data['historical']
                symbol_info = data.get('symbol', symbol)
                
                # DataFrameに変換
                df = pd.DataFrame(historical_data)
                
                # symbolカラムが存在しない場合は追加
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol_info
            else:
                # historical形式でない場合は空のDataFrameを返す
                self.logger.warning(f"{symbol}の株価履歴データが不正な形式です")
                return pd.DataFrame()
        else:
            # 直接リストが返ってきた場合
            df = pd.DataFrame(data)
            
            # symbolカラムが存在しない場合は追加
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
                
        # 空のDataFrameのチェック
        if df.empty:
            self.logger.warning(f"{symbol}の株価履歴データが空です")
            return df
            
        # カラム名を標準化
        column_mapping = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'adjClose': 'adj_close',
            'volume': 'volume',
            'unadjustedVolume': 'unadjusted_volume',
            'change': 'change',
            'changePercent': 'change_percent',
            'vwap': 'vwap',
            'label': 'label',
            'changeOverTime': 'change_over_time'
        }
        
        # 存在するカラムだけをマッピング
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df[new_name] = df[old_name]
                df = df.drop(columns=[old_name])
                
        # 日付カラムを文字列に変換（DB保存用）
        if 'date' in df.columns and df['date'].dtype != 'object':
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
        # 取得件数を報告
        if not df.empty:
            self.logger.info(f"{symbol}の株価データ: {len(df)}件取得（期間: {df['date'].min()} ～ {df['date'].max()}）")
        else:
            self.logger.info(f"{symbol}の株価データ: 新しいデータはありません")
            
        return df
    
    def get_news(self, symbol=None, limit=50, page=0):
        """
        ニュースデータを取得
        
        パラメータ:
            symbol (str): 証券コード（Noneなら全銘柄）
            limit (int): 取得するニュース数
            page (int): ページ番号
            
        戻り値:
            DataFrame: ニュースデータ
        """
        endpoint = f"/news/stock-latest"
        params = {
            'limit': limit,
            'page': page
        }
        if symbol:
            params['tickers'] = symbol
        
        data = self.fetch_data(endpoint, params)
        return self.to_dataframe(data)
    
    def get_sentiment_news(self, page=0):
        """
        センチメント分析付きニュースデータを取得（API v4）
        
        パラメータ:
            page (int): ページ番号
            
        戻り値:
            DataFrame: センチメント分析付きニュースデータ
        """
        endpoint = f"/stock-news-sentiments-rss-feed"
        params = {
            'page': page
        }
        
        data = self.fetch_data(endpoint, params, use_v4=True)
        return self.to_dataframe(data)
    
    def get_sentiment_news_raw(self, page=0):
        """
        センチメント分析付きニュースデータを生のJSON形式で取得（API v4）
        
        パラメータ:
            page (int): ページ番号
            
        戻り値:
            list: センチメント分析付きニュースデータ (JSON配列)
        """
        endpoint = f"/stock-news-sentiments-rss-feed"
        params = {
            'page': page
        }
        
        data = self.fetch_data(endpoint, params, use_v4=True)
        return data
    
    def get_shares_float(self, symbol, from_date=None):
        """
        株式数データを取得（API v4）
        
        パラメータ:
            symbol (str): 証券コード
            from_date (str): 開始日（YYYY-MM-DD形式）
            
        戻り値:
            DataFrame: 株式数データ
        """
        endpoint = f"/historical/shares_float"
        params = {
            'symbol': symbol
        }
        
        # from_dateが指定されている場合は追加
        if from_date:
            params['from'] = from_date
        
        data = self.fetch_data(endpoint, params, use_v4=True)
        
        # サブスクリプションエラーの場合、旧エンドポイントを試す
        if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
            self.logger.warning(f"{symbol}の株式数データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
            # 旧エンドポイント（APIv4）を使用して再試行 - 同じエンドポイントだが直接パスを指定
            legacy_endpoint = f"/api/v4/historical/shares_float"
            legacy_params = {
                'symbol': symbol
            }
            if from_date:
                legacy_params['from'] = from_date
                
            data = self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
        return self.to_dataframe(data)
    
    def get_employee_count(self, symbol, limit=None):
        """
        従業員数データを取得
        
        パラメータ:
            symbol (str): 証券コード
            limit (int): 取得するデータ数（過去何年分か指定）
            
        戻り値:
            DataFrame: 従業員数データ
        """
        # 最新APIエンドポイント
        endpoint = f"/employee-count"
        params = {
            'symbol': symbol
        }
        
        if limit:
            params['limit'] = limit
        
        data = self.fetch_data(endpoint, params)
        
        # サブスクリプションエラーの場合、旧エンドポイントを試す
        if not data or (isinstance(data, dict) and data.get("Error Message", "").startswith("Premium Query Parameter")):
            self.logger.warning(f"{symbol}の従業員数データ取得でサブスクリプションエラーが発生しました。旧エンドポイントを試行します。")
            # 旧エンドポイント（APIv4）を使用して再試行
            legacy_endpoint = f"/api/v4/historical/employee_count"
            legacy_params = {
                'symbol': symbol
            }
            if limit:
                legacy_params['limit'] = limit
                
            data = self.fetch_data(legacy_endpoint, legacy_params, use_legacy=True)
            
        return self.to_dataframe(data)
    
    def get_forex(self, symbol, from_date=None, to_date=None):
        """
        外国為替レートの履歴データを取得
        
        パラメータ:
            symbol (str): 通貨ペア（例: 'USDJPY'）
            from_date (str): 開始日（YYYY-MM-DD形式）
            to_date (str): 終了日（YYYY-MM-DD形式）
            
        戻り値:
            DataFrame: 外国為替レートデータ
        """
        # STABLE APIエンドポイントを使用
        endpoint = "historical-price-eod/light"
        params = {
            'symbol': symbol
        }
        
        # from_dateとto_dateが指定されている場合は追加
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        # STABLE APIを使用
        url = f"{self.BASE_URL_STABLE}/{endpoint}"
        
        # キャッシュキーの生成
        cache_key = f"{url}:{json.dumps(params, sort_keys=True)}"
        
        # キャッシュからデータを取得
        cached_data = self.rate_limiter.get_cached_response(cache_key)
        if cached_data is not None:
            return self.to_dataframe(cached_data)
        
        # レート制限のチェック
        self.rate_limiter.wait_if_needed()
        
        # プライマリAPIキーを使用
        for i, api_key in enumerate(self.api_keys):
            try:
                params['apikey'] = api_key
                response = self.session.get(url, params=params)
                self.request_count += 1
                
                # エラーチェック
                if response.status_code != 200:
                    self.logger.warning(f"為替データ取得エラー ({i+1}/{len(self.api_keys)}番目のAPIキー): ステータスコード {response.status_code}")
                    continue
                
                data = response.json()
                
                if not data:
                    self.logger.warning(f"為替データが空 ({i+1}/{len(self.api_keys)}番目のAPIキー)")
                    continue
                
                # レスポンスをキャッシュ
                self.rate_limiter.cache_response(cache_key, data)
                
                # データフレームに変換して返す
                df = self.to_dataframe(data)
                
                # 現在時刻を追加
                if not df.empty:
                    df['retrieved_at'] = datetime.now()
                
                return df
                
            except Exception as e:
                self.logger.error(f"為替データ取得中にエラー ({i+1}/{len(self.api_keys)}番目のAPIキー): {str(e)}")
                self.logger.debug(f"エラー詳細: {traceback.format_exc()}")
        
        # すべてのAPIキーで失敗した場合は空のDataFrameを返す
        self.logger.error(f"すべてのAPIキーで為替データ取得に失敗: {symbol}")
        return pd.DataFrame()
    
    def get_earnings_calendar(self, from_date, to_date=None):
        """
        決算発表カレンダーを取得
        
        パラメータ:
            from_date (str): 開始日（YYYY-MM-DD形式）
            to_date (str): 終了日（YYYY-MM-DD形式）、Noneの場合は開始日と同じ
            
        戻り値:
            list or DataFrame: 決算発表カレンダーデータ（APIからの生データ）
        """
        if to_date is None:
            to_date = from_date
            
        endpoint = f"/earnings-calendar"  # 修正: 正しいエンドポイント名
        params = {
            'from': from_date,
            'to': to_date
        }
        
        data = self.fetch_data(endpoint, params)
        
        # 空のデータの場合は空のリストを返す
        if not data:
            self.logger.warning("決算カレンダーデータが空です")
            return []
            
        # APIレスポンスをそのまま返す
        return data

    def save_earnings_calendar(self, data, save_to_db=True):
        """
        決算カレンダーのデータをDataFrameとして処理し、オプションでデータベースに保存する
        """
        self.logger.debug("決算カレンダーデータを処理します")
        
        # データの空チェック
        if data is None:
            self.logger.warning("決算カレンダーデータがNoneです")
            return None
        if isinstance(data, list) and not data:
            self.logger.warning("決算カレンダーデータのリストが空です")
            return None
        if hasattr(data, 'empty') and data.empty:
            self.logger.warning("決算カレンダーデータのDataFrameが空です")
            return None

        # データをDataFrameに変換
        if isinstance(data, list):
            # リストの場合は新しいDataFrameを作成
            df = pd.DataFrame(data)
        elif hasattr(data, 'empty'):  # DataFrameの場合
            # データフレームの場合はコピーを作成
            df = data.copy()
        else:
            self.logger.warning(f"不明な形式の決算カレンダーデータです: {type(data)}")
            return None
            
        # まず全てのデータを表示してデバッグ
        self.logger.debug(f"元データのカラム: {df.columns.tolist()}")
        if len(df) > 0:
            self.logger.debug(f"最初の行のサンプル: {df.iloc[0].to_dict()}")

        # dateカラムが存在しない場合はreport_dateをdateにコピー
        if 'date' not in df.columns and 'report_date' in df.columns:
            df['date'] = df['report_date']
            self.logger.debug("report_dateカラムをdateカラムにコピーしました")

        # symbolカラムが存在しない場合はエラー
        if 'symbol' not in df.columns:
            self.logger.error("'symbol'カラムが決算カレンダーデータに存在しません")
            self.logger.debug(f"利用可能なカラム: {df.columns.tolist()}")
            return None

        # 必要なdateカラムがない場合はエラー
        if 'date' not in df.columns:
            self.logger.error("'date'カラムが決算カレンダーデータに存在しません")
            self.logger.debug(f"利用可能なカラム: {df.columns.tolist()}")
            return None

        # カラム名を変更 (キャメルケースからスネークケースへ)
        column_mapping = {
            'symbol': 'symbol',
            'date': 'report_date',
            'epsEstimated': 'eps_estimated',
            'epsActual': 'eps_actual',
            'revenueEstimated': 'revenue_estimated',
            'revenueActual': 'revenue_actual',
            'lastUpdated': 'last_updated'
        }
        
        # カラム名を変更
        df.rename(columns=column_mapping, inplace=True)

        # データ型の変換
        try:
            # 日付カラムの処理
            df['report_date'] = pd.to_datetime(df['report_date']).dt.date
        except Exception as e:
            self.logger.error(f"report_date変換中にエラーが発生しました: {e}")
            # エラーが発生しても処理を続行

        try:
            # last_updatedカラムの処理
            if 'last_updated' in df.columns:
                df['last_updated'] = pd.to_datetime(df['last_updated']).dt.date
        except Exception as e:
            self.logger.error(f"last_updated変換中にエラーが発生しました: {e}")
            # エラーが発生しても処理を続行

        # NaNを処理
        numeric_cols = ['eps_estimated', 'eps_actual', 'revenue_estimated', 'revenue_actual']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 重複データの削除（DataFrame内部の重複のみ）
        df = df.drop_duplicates(subset=['symbol', 'report_date'], keep='first')

        # 現在時刻を追加
        df['retrieved_at'] = datetime.now()
        df['status'] = 'pending'  # 初期ステータスはpending

        # データベースへの保存
        if save_to_db and self.db_engine:
            try:
                self.logger.debug(f"DataFrame情報: {len(df)}行, カラム: {df.columns.tolist()}")
                
                # earningsテーブルに保存 - UPSERTを使用
                table_name = 'earnings'
                schema = 'fmp_data'
                
                # DataFrameをレコードのリストに変換
                records = df.to_dict('records')
                
                if records:
                    with self.db_engine.begin() as conn:
                        # バッチ処理のためのカウンタとバッチサイズ
                        counter = 0
                        batch_size = 100
                        total_updated = 0
                        
                        for record in records:
                            # ON CONFLICT DO UPDATEを使用したupsertクエリ
                            # statusカラムは既存のレコードの値を維持
                            upsert_query = text("""
                                INSERT INTO fmp_data.earnings 
                                (symbol, report_date, eps_actual, eps_estimated, 
                                revenue_actual, revenue_estimated, last_updated, 
                                retrieved_at, status) 
                                VALUES 
                                (:symbol, :report_date, :eps_actual, :eps_estimated, 
                                :revenue_actual, :revenue_estimated, :last_updated, 
                                :retrieved_at, :status)
                                ON CONFLICT (symbol, report_date) 
                                DO UPDATE SET 
                                eps_actual = EXCLUDED.eps_actual,
                                eps_estimated = EXCLUDED.eps_estimated,
                                revenue_actual = EXCLUDED.revenue_actual,
                                revenue_estimated = EXCLUDED.revenue_estimated,
                                last_updated = EXCLUDED.last_updated,
                                retrieved_at = EXCLUDED.retrieved_at
                                -- statusカラムは更新しない（ただしstatusがpendingの場合のみ更新）
                                WHERE fmp_data.earnings.status = 'pending'
                            """)
                            
                            conn.execute(upsert_query, record)
                            counter += 1
                            
                            # バッチサイズに達したらカウンタをリセット
                            if counter >= batch_size:
                                self.logger.debug(f"{batch_size}件のレコードを処理しました")
                                total_updated += counter
                                counter = 0
                        
                        # 残りのレコードを計上
                        if counter > 0:
                            total_updated += counter
                    
                    self.logger.debug(f"{total_updated}行のデータをfmp_data.{table_name}に保存しました")
                    self.logger.info(f"決算カレンダーデータを保存しました: {total_updated}件")
                
            except Exception as e:
                self.logger.error(f"決算カレンダーの保存中にエラーが発生しました: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return df

        return df

    def _get_latest_date(self, symbol, table_name, schema='fmp_data', period_type=None):
        """
        指定したテーブルの指定した銘柄の最新日付を取得する内部メソッド
        
        パラメータ:
            symbol (str): 証券コード
            table_name (str): テーブル名
            schema (str): スキーマ名
            period_type (str): 'annual'または'quarterly'（財務データの場合）
            
        戻り値:
            str: 最新日付（YYYY-MM-DD形式）
        """
        query = f"""
            SELECT MAX(date) as latest_date
            FROM {schema}.{table_name}
            WHERE symbol = :symbol
        """
        
        # 財務データの場合はperiod_typeでフィルタリング
        if period_type and table_name in ['income_statements', 'balance_sheets', 'cash_flows']:
            query += " AND period_type = :period_type"
            self.logger.debug(f"最新日付取得クエリ（period_type={period_type}）: {query}")
        else:
            self.logger.debug(f"最新日付取得クエリ: {query}")
        
        try:
            with self.db_engine.connect() as conn:
                params = {"symbol": symbol}
                if period_type and table_name in ['income_statements', 'balance_sheets', 'cash_flows']:
                    params["period_type"] = period_type
                    
                result = conn.execute(text(query), params)
                row = result.fetchone()
                latest_date = row[0] if row and row[0] else None
                
                if latest_date:
                    self.logger.debug(f"{symbol}の{table_name}（{period_type if period_type else '全て'}）の最新日付: {latest_date}")
                    return latest_date.strftime('%Y-%m-%d')
                else:
                    self.logger.debug(f"{symbol}の{table_name}（{period_type if period_type else '全て'}）の最新日付が見つかりません")
                return None
        except Exception as e:
            self.logger.error(f"最新日付取得エラー: {e}")
            return None

    def get_api_usage_summary(self):
        """
        API使用量の概要を取得
        
        戻り値:
            dict: API使用量の概要情報
        """
        return {
            "request_count": self.request_count,
            "total_data_size_bytes": self.total_data_size,
            "total_data_size_formatted": self._format_size(self.total_data_size),
            "average_request_size": self._format_size(self.total_data_size / max(1, self.request_count))
        }
        
    def log_api_usage_summary(self):
        """APIの使用量概要をログに出力"""
        summary = self.get_api_usage_summary()
        self.logger.info("=== FMP API 使用量サマリー ===")
        self.logger.info(f"リクエスト総数: {summary['request_count']}回")
        self.logger.info(f"データ総量: {summary['total_data_size_formatted']} ({summary['total_data_size_bytes']} bytes)")
        self.logger.info(f"リクエスト平均サイズ: {summary['average_request_size']}")
        self.logger.info("==============================")
        return summary

    def fetch_and_store_daily_prices(self, symbol, start_date=None, end_date=None):
        """日次株価データを取得してデータベースに保存する"""
        try:
            # 株価データを取得
            prices = self.fetch_daily_prices(symbol, start_date, end_date)
            
            if not prices:
                # 株価データが取得できない場合は、シンボルのステータスを非アクティブに更新
                self.update_symbol_status(symbol, is_active=False)
                self.logger.warning(f"{symbol}の株価データが取得できませんでした。ステータスを非アクティブに更新します。")
                return False
            
            # データベースに保存
            self.store_daily_prices(symbol, prices)
            return True
        
        except Exception as e:
            self.logger.error(f"{symbol}の株価データ取得中にエラーが発生: {e}")
            return False

    def update_symbol_status(self, symbol, is_active=False):
        """シンボルのステータスを更新する"""
        try:
            with self.db_engine.connect() as conn:
                # is_activeをFalseに設定する場合はmanually_deactivatedもTrueに設定
                manually_deactivated = not is_active
                
                query = text("""
                    UPDATE fmp_data.symbol_status 
                    SET is_active = :is_active,
                        manually_deactivated = :manually_deactivated,
                        last_updated = NOW()
                    WHERE symbol = :symbol
                """)
                conn.execute(query, {
                    "symbol": symbol, 
                    "is_active": is_active,
                    "manually_deactivated": manually_deactivated
                })
                conn.commit()
        except Exception as e:
            self.logger.error(f"{symbol}のステータス更新中にエラーが発生: {e}")

    def get_error_statistics(self):
        """
        エラー統計情報を取得
        
        戻り値:
            dict: エラー統計情報
        """
        return {
            'stats': self.error_stats,
            'total_requests': self.request_count,
            'total_data_size': self._format_size(self.total_data_size),
            'problematic_symbols_count': len(self.problematic_symbols),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def print_error_report(self):
        """
        エラー統計情報をログに出力
        """
        stats = self.get_error_statistics()
        
        self.logger.info("=== FMP API エラー統計レポート ===")
        self.logger.info(f"総リクエスト数: {stats['total_requests']}")
        self.logger.info(f"総データ通信量: {stats['total_data_size']}")
        self.logger.info(f"問題のある銘柄数: {stats['problematic_symbols_count']}")
        
        self.logger.info("\nエラー種別ごとの発生回数:")
        for error_type, count in stats['stats'].items():
            if count > 0:
                self.logger.info(f"  {error_type}: {count}回")
        
        if stats['recent_errors']:
            self.logger.info("\n直近のエラー（最新10件）:")
            for error in stats['recent_errors']:
                self.logger.info(f"  [{error['timestamp']}] {error['type']} - {error['symbol']}")
                self.logger.info(f"    詳細: {json.dumps(error['details'], indent=2)}")
        
        self.logger.info("================================")
    
    def reset_error_statistics(self):
        """
        エラー統計情報をリセット
        """
        self.error_stats = {
            'rate_limit': 0,
            'not_found': 0,
            'server_error': 0,
            'client_error': 0,
            'network_error': 0,
            'parse_error': 0,
            'other_error': 0
        }
        self.error_history = []
        self.request_count = 0
        self.total_data_size = 0
        self.problematic_symbols = set()
        self.logger.info("エラー統計情報をリセットしました")

    def analyze_error_patterns(self):
        """
        エラーパターンを分析し、問題の傾向を特定する
        
        戻り値:
            dict: エラーパターン分析結果
        """
        if not self.error_history:
            return {"message": "エラー履歴がありません"}
        
        # エラータイプごとの集計
        error_type_counts = {}
        symbol_error_counts = {}
        time_based_errors = {}
        
        for error in self.error_history:
            # エラータイプの集計
            error_type = error['type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            # 銘柄ごとのエラー集計
            symbol = error['symbol']
            symbol_error_counts[symbol] = symbol_error_counts.get(symbol, 0) + 1
            
            # 時間帯ごとのエラー集計（1時間単位）
            hour = error['timestamp'].split(' ')[1][:2]
            time_based_errors[hour] = time_based_errors.get(hour, 0) + 1
        
        # 最も問題の多い銘柄を特定
        problematic_symbols = sorted(
            symbol_error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 最もエラーが多い時間帯を特定
        peak_error_hours = sorted(
            time_based_errors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            "error_type_distribution": error_type_counts,
            "top_problematic_symbols": problematic_symbols,
            "peak_error_hours": peak_error_hours,
            "total_errors": len(self.error_history),
            "unique_symbols_with_errors": len(symbol_error_counts)
        }

    def classify_problematic_symbols(self):
        """
        問題のある銘柄を分類し、問題の種類ごとにグループ化する
        
        戻り値:
            dict: 銘柄分類結果
        """
        if not self.error_history:
            return {"message": "エラー履歴がありません"}
        
        # 銘柄ごとのエラー情報を収集
        symbol_errors = {}
        
        for error in self.error_history:
            symbol = error['symbol']
            if symbol not in symbol_errors:
                symbol_errors[symbol] = {
                    'error_types': set(),
                    'error_count': 0,
                    'last_error': None,
                    'first_error': None
                }
            
            symbol_errors[symbol]['error_types'].add(error['type'])
            symbol_errors[symbol]['error_count'] += 1
            
            # 最初と最後のエラーを記録
            if not symbol_errors[symbol]['first_error']:
                symbol_errors[symbol]['first_error'] = error['timestamp']
            symbol_errors[symbol]['last_error'] = error['timestamp']
        
        # 問題の種類ごとに銘柄を分類
        classification = {
            'rate_limit_issues': [],
            'data_availability_issues': [],
            'server_issues': [],
            'other_issues': []
        }
        
        for symbol, error_info in symbol_errors.items():
            error_types = error_info['error_types']
            
            if 'rate_limit' in error_types:
                classification['rate_limit_issues'].append({
                    'symbol': symbol,
                    'error_count': error_info['error_count'],
                    'last_error': error_info['last_error']
                })
            elif 'not_found' in error_types:
                classification['data_availability_issues'].append({
                    'symbol': symbol,
                    'error_count': error_info['error_count'],
                    'last_error': error_info['last_error']
                })
            elif 'server_error' in error_types or 'timeout' in error_types:
                classification['server_issues'].append({
                    'symbol': symbol,
                    'error_count': error_info['error_count'],
                    'last_error': error_info['last_error']
                })
            else:
                classification['other_issues'].append({
                    'symbol': symbol,
                    'error_count': error_info['error_count'],
                    'last_error': error_info['last_error']
                })
        
        # 各カテゴリ内でエラー数が多い順にソート
        for category in classification:
            classification[category] = sorted(
                classification[category],
                key=lambda x: x['error_count'],
                reverse=True
            )
        
        return classification

    def generate_detailed_error_report(self):
        """
        詳細なエラーレポートを生成する
        
        戻り値:
            dict: 詳細なエラーレポート
        """
        # 基本的な統計情報
        basic_stats = self.get_error_statistics()
        
        # エラーパターン分析
        pattern_analysis = self.analyze_error_patterns()
        
        # 問題銘柄の分類
        symbol_classification = self.classify_problematic_symbols()
        
        # レポートの生成
        report = {
            "summary": {
                "total_requests": basic_stats['total_requests'],
                "total_data_size": basic_stats['total_data_size'],
                "total_errors": len(self.error_history),
                "problematic_symbols_count": basic_stats['problematic_symbols_count']
            },
            "error_patterns": pattern_analysis,
            "problematic_symbols": symbol_classification,
            "recent_errors": basic_stats['recent_errors'],
            "recommendations": self._generate_recommendations(pattern_analysis, symbol_classification)
        }
        
        return report

    def _generate_recommendations(self, pattern_analysis, symbol_classification):
        """
        エラー分析に基づいて推奨事項を生成する
        
        パラメータ:
            pattern_analysis: エラーパターン分析結果
            symbol_classification: 問題銘柄の分類結果
            
        戻り値:
            list: 推奨事項のリスト
        """
        recommendations = []
        
        # レート制限に関する推奨事項
        if 'rate_limit' in pattern_analysis.get('error_type_distribution', {}):
            rate_limit_count = pattern_analysis['error_type_distribution']['rate_limit']
            if rate_limit_count > 10:
                recommendations.append({
                    "type": "rate_limit",
                    "message": "レート制限エラーが頻発しています。APIリクエストの間隔を広げるか、バッチサイズを小さくすることを検討してください。",
                    "action": "APIリクエストの間隔を調整する"
                })
        
        # サーバーエラーに関する推奨事項
        if 'server_error' in pattern_analysis.get('error_type_distribution', {}):
            server_error_count = pattern_analysis['error_type_distribution']['server_error']
            if server_error_count > 5:
                recommendations.append({
                    "type": "server_error",
                    "message": "サーバーエラーが発生しています。リトライ間隔を長くするか、一時停止してから再開することを検討してください。",
                    "action": "リトライ戦略を調整する"
                })
        
        # 特定の銘柄に関する推奨事項
        problematic_symbols = symbol_classification.get('rate_limit_issues', [])[:5]
        if problematic_symbols:
            symbols_str = ", ".join([item['symbol'] for item in problematic_symbols])
            recommendations.append({
                "type": "problematic_symbols",
                "message": f"以下の銘柄で多くのエラーが発生しています: {symbols_str}",
                "action": "これらの銘柄を個別に処理するか、スキップすることを検討してください"
            })
        
        # 時間帯に関する推奨事項
        peak_hours = pattern_analysis.get('peak_error_hours', [])
        if peak_hours:
            hours_str = ", ".join([f"{hour}時 ({count}回)" for hour, count in peak_hours])
            recommendations.append({
                "type": "peak_hours",
                "message": f"以下の時間帯でエラーが集中しています: {hours_str}",
                "action": "これらの時間帯を避けて処理を実行することを検討してください"
            })
        
        return recommendations

    def get_actively_trading_list(self):
        """
        /stable/actively-trading-list エンドポイントからアクティブ取引銘柄リストを取得
        
        戻り値:
            list: アクティブ取引銘柄のリスト
        """
        endpoint = "/actively-trading-list"
        data = self.fetch_data(endpoint, params={})
        
        if data is None:
            self.logger.warning("actively-trading-listデータが取得できませんでした")
            return []
        
        if isinstance(data, dict) and "Error Message" in data:
            self.logger.error(f"actively-trading-listエラー: {data['Error Message']}")
            return []
        
        self.logger.info(f"actively-trading-listから{len(data) if isinstance(data, list) else 0}件の銘柄を取得しました")
        return data if isinstance(data, list) else []

    def get_stable_company_profile(self, symbol):
        """
        /stable/profile エンドポイントから会社プロファイルを取得
        
        パラメータ:
            symbol (str): 証券コード
            
        戻り値:
            dict or None: 会社プロファイルデータ
        """
        endpoint = "/profile"
        params = {"symbol": symbol}
        
        data = self.fetch_data(endpoint, params)
        
        if data is None:
            self.logger.warning(f"{symbol}の会社プロファイルデータが取得できませんでした")
            return None
        
        if isinstance(data, dict) and "Error Message" in data:
            self.logger.error(f"{symbol}の会社プロファイルエラー: {data['Error Message']}")
            return None
        
        # リストの場合は最初の要素を返す
        if isinstance(data, list):
            return data[0] if len(data) > 0 else None
        
        return data

    def get_stable_employee_count(self, symbol):
        """
        /stable/employee-count エンドポイントから従業員数データを取得
        
        パラメータ:
            symbol (str): 証券コード
            
        戻り値:
            list: 従業員数データ
        """
        endpoint = "/employee-count"
        params = {"symbol": symbol}
        
        data = self.fetch_data(endpoint, params)
        
        if data is None:
            self.logger.warning(f"{symbol}の従業員数データが取得できませんでした")
            return []
        
        if isinstance(data, dict) and "Error Message" in data:
            self.logger.error(f"{symbol}の従業員数エラー: {data['Error Message']}")
            return []
        
        if not isinstance(data, list):
            return [data] if data else []
        
        return data

    def _camel_to_snake(self, camel_str):
        """
        キャメルケースの文字列をスネークケースに変換します。
        例: "camelCase" -> "camel_case"
        """
        # 既にスネークケースの場合は変換しない
        if '_' in camel_str:
            return camel_str
            
        # 最初の大文字の前に_を追加
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
        # 残りの大文字の前に_を追加し、小文字に変換
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        return s2
