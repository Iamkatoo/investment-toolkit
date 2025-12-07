"""
J-Quants API認証モジュール

REFRESH_TOKENとidTokenの取得・管理を行います。
セキュリティのため、REFRESH_TOKENは1日1回取得し、
idTokenはAPI呼び出しの都度取得します。
"""

import requests
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time

from investment_analysis.utilities.config import JQUANTS_EMAIL, JQUANTS_PASSWORD

logger = logging.getLogger(__name__)

class JQuantsAuth:
    """J-Quants API認証クラス"""
    
    BASE_URL = "https://api.jquants.com/v1"
    
    def __init__(self):
        self.refresh_token: Optional[str] = None
        self.refresh_token_acquired_at: Optional[datetime] = None
        self.email = JQUANTS_EMAIL
        self.password = JQUANTS_PASSWORD
        
        if not self.email or not self.password:
            raise ValueError("JQUANTS_EMAIL and JQUANTS_PASSWORD must be set in environment variables")
    
    def _should_refresh_token(self) -> bool:
        """REFRESH_TOKENを更新すべきかどうかを判定"""
        if not self.refresh_token or not self.refresh_token_acquired_at:
            return True
        
        # 1日経過していれば更新
        return datetime.now() - self.refresh_token_acquired_at > timedelta(days=1)
    
    def get_refresh_token(self) -> str:
        """
        REFRESH_TOKENを取得
        
        Returns:
            str: REFRESH_TOKEN
            
        Raises:
            Exception: 認証エラーが発生した場合
        """
        if not self._should_refresh_token():
            logger.debug("REFRESH_TOKENは有効期限内のため、再利用します")
            return self.refresh_token
        
        logger.info("新しいREFRESH_TOKENを取得中...")
        
        auth_url = f"{self.BASE_URL}/token/auth_user"
        
        auth_data = {
            "mailaddress": self.email,
            "password": self.password
        }
        
        try:
            response = requests.post(
                auth_url,
                data=json.dumps(auth_data),
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            if "refreshToken" not in response_data:
                raise Exception(f"REFRESH_TOKEN not found in response: {response_data}")
            
            self.refresh_token = response_data["refreshToken"]
            self.refresh_token_acquired_at = datetime.now()
            
            logger.info("REFRESH_TOKENを正常に取得しました")
            return self.refresh_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"REFRESH_TOKEN取得時のHTTPエラー: {e}")
            raise Exception(f"Failed to get refresh token: {e}")
        except Exception as e:
            logger.error(f"REFRESH_TOKEN取得時のエラー: {e}")
            raise
    
    def get_id_token(self) -> str:
        """
        idTokenを取得
        
        Returns:
            str: idToken
            
        Raises:
            Exception: 認証エラーが発生した場合
        """
        # REFRESH_TOKENが必要な場合は先に取得
        refresh_token = self.get_refresh_token()
        
        logger.debug("idTokenを取得中...")
        
        token_url = f"{self.BASE_URL}/token/auth_refresh"
        
        try:
            response = requests.post(
                f"{token_url}?refreshtoken={refresh_token}",
                timeout=30
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            if "idToken" not in response_data:
                raise Exception(f"idToken not found in response: {response_data}")
            
            id_token = response_data["idToken"]
            logger.debug("idTokenを正常に取得しました")
            
            return id_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"idToken取得時のHTTPエラー: {e}")
            # REFRESH_TOKENが無効になった可能性があるため、次回は再取得する
            self.refresh_token = None
            self.refresh_token_acquired_at = None
            raise Exception(f"Failed to get id token: {e}")
        except Exception as e:
            logger.error(f"idToken取得時のエラー: {e}")
            raise
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        認証ヘッダーを取得
        
        Returns:
            Dict[str, str]: Authorization ヘッダー
        """
        id_token = self.get_id_token()
        return {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }


# シングルトンインスタンス
_auth_instance: Optional[JQuantsAuth] = None

def get_auth() -> JQuantsAuth:
    """
    認証インスタンスを取得（シングルトンパターン）
    
    Returns:
        JQuantsAuth: 認証インスタンス
    """
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = JQuantsAuth()
    return _auth_instance 