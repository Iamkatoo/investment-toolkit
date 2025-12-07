"""
J-Quants APIクライアント

認証を含むAPI呼び出しの基盤クラスを提供します。
レート制限やリトライ機能も含みます。
"""

import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, date

from .auth import get_auth

logger = logging.getLogger(__name__)

class JQuantsAPIClient:
    """J-Quants APIクライアント"""
    
    BASE_URL = "https://api.jquants.com/v1"
    
    def __init__(self):
        self.auth = get_auth()
        self.last_request_time: Optional[float] = None
        self.min_request_interval = 0.1  # 100ms間隔でリクエスト制限
    
    def _wait_for_rate_limit(self):
        """レート制限のための待機"""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                wait_time = self.min_request_interval - elapsed
                logger.debug(f"レート制限のため{wait_time:.2f}秒待機中...")
                time.sleep(wait_time)
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        API リクエストを実行
        
        Args:
            endpoint: APIエンドポイント（/v1/ 以降の部分）
            params: リクエストパラメータ
            max_retries: 最大リトライ回数
            
        Returns:
            Dict[str, Any]: APIレスポンス
            
        Raises:
            Exception: API呼び出しエラー
        """
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        for attempt in range(max_retries + 1):
            try:
                # レート制限チェック
                self._wait_for_rate_limit()
                
                # 認証ヘッダー取得
                headers = self.auth.get_auth_headers()
                
                logger.debug(f"API呼び出し: {url} (試行 {attempt + 1}/{max_retries + 1})")
                
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=60
                )
                
                self.last_request_time = time.time()
                
                response.raise_for_status()
                
                response_data = response.json()
                logger.debug(f"API呼び出し成功: {len(response_data)} 件のデータを取得")
                
                return response_data
                
            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP エラー (試行 {attempt + 1}): {e}")
                if e.response.status_code == 401:
                    # 認証エラーの場合は認証をリセット
                    logger.info("認証エラーのため認証情報をリセットします")
                    self.auth.refresh_token = None
                    self.auth.refresh_token_acquired_at = None
                    
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 指数バックオフ
                    logger.info(f"{wait_time}秒後にリトライします...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries + 1} attempts: {e}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"リクエストエラー (試行 {attempt + 1}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"{wait_time}秒後にリトライします...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries + 1} attempts: {e}")
                    
            except Exception as e:
                logger.error(f"予期しないエラー: {e}")
                raise
        
        raise Exception("Should not reach here")
    
    def get_daily_quotes(self, target_date: date, code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        日足データを取得
        
        Args:
            target_date: 取得対象日
            code: 銘柄コード（指定しない場合は全銘柄）
            
        Returns:
            List[Dict[str, Any]]: 日足データのリスト
        """
        params = {
            "date": target_date.strftime("%Y-%m-%d")
        }
        
        if code:
            params["code"] = code
        
        logger.info(f"日足データ取得開始: {target_date.strftime('%Y-%m-%d')}" + 
                   (f" 銘柄: {code}" if code else " 全銘柄"))
        
        try:
            response = self._make_request("prices/daily_quotes", params)
            
            # レスポンスが辞書でdaily_quotesキーを持つ場合
            if isinstance(response, dict) and "daily_quotes" in response:
                data = response["daily_quotes"]
            # レスポンスがリストの場合
            elif isinstance(response, list):
                data = response
            else:
                # レスポンスが辞書だが、直接データが含まれている場合
                data = [response] if isinstance(response, dict) else []
            
            logger.info(f"日足データ取得完了: {len(data)} 件")
            return data
            
        except Exception as e:
            logger.error(f"日足データ取得エラー: {e}")
            raise
    
    def get_financial_statements(self, code: str) -> Dict[str, Any]:
        """
        財務諸表データを取得
        
        Args:
            code: 銘柄コード
            
        Returns:
            Dict[str, Any]: 財務諸表データ
        """
        params = {"code": code}
        
        logger.info(f"財務諸表データ取得開始: {code}")
        
        try:
            response = self._make_request("fins/statements", params)
            logger.info(f"財務諸表データ取得完了: {code}")
            return response
            
        except Exception as e:
            logger.error(f"財務諸表データ取得エラー ({code}): {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        接続テスト
        
        Returns:
            bool: 接続成功時True
        """
        try:
            # 今日の日付で少量のデータを取得してテスト
            today = date.today()
            self.get_daily_quotes(today)
            logger.info("API接続テスト成功")
            return True
            
        except Exception as e:
            logger.error(f"API接続テスト失敗: {e}")
            return False 