#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非同期APIレート制限モジュール
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """非同期APIリクエストのレート制限を管理するクラス"""

    def __init__(self, rate_limit: int, period: float = 60.0, burst_limit: int = None, burst_period: float = 10.0):
        """
        初期化

        Args:
            rate_limit (int): 期間あたりの最大リクエスト数
            period (float): 制限期間（秒）
            burst_limit (int, optional): バースト時の最大リクエスト数、指定がなければrate_limitの30%
            burst_period (float): バースト判定期間（秒）
        """
        self.rate_limit = rate_limit
        self.period = period
        self.burst_limit = burst_limit if burst_limit is not None else max(1, int(rate_limit * 0.3))
        self.burst_period = burst_period
        
        # レート制限用のトークンバケット
        self.tokens = rate_limit
        self.last_refill_time = time.time()
        
        # リクエスト履歴
        self.request_history = []
        
        # 保護用のロック
        self._lock = asyncio.Lock()
        
        logger.debug(
            f"レート制限を初期化: {rate_limit}リクエスト/{period}秒, "
            f"バースト制限: {self.burst_limit}リクエスト/{burst_period}秒"
        )

    async def acquire(self):
        """
        リクエストのトークンを取得

        Returns:
            bool: トークンが取得できればTrue、制限超過ならFalse
        """
        async with self._lock:
            now = time.time()
            
            # 古いリクエスト履歴をクリーンアップ
            self._clean_old_requests(now)
            
            # トークンを補充
            await self._refill_tokens(now)
            
            # トークンが残っているか確認
            if self.tokens <= 0:
                # 制限超過
                refill_time = self._time_until_next_refill()
                logger.warning(
                    f"レート制限に達しました。残りトークン: {self.tokens}, "
                    f"次の補充まで: {refill_time:.2f}秒"
                )
                return False
            
            # バースト制限をチェック
            recent_requests = [
                req for req in self.request_history 
                if now - req < self.burst_period
            ]
            
            if len(recent_requests) >= self.burst_limit:
                logger.warning(
                    f"バースト制限に達しました。直近{self.burst_period}秒間に"
                    f"{len(recent_requests)}リクエスト（上限: {self.burst_limit}）"
                )
                return False
            
            # トークンを消費して履歴に追加
            self.tokens -= 1
            self.request_history.append(now)
            
            remaining = await self.remaining_tokens()
            logger.debug(f"トークンを取得しました。残り: {remaining}")
            
            return True

    def _clean_old_requests(self, now):
        """
        期間外の古いリクエスト履歴を削除

        Args:
            now (float): 現在時刻（UNIX時間）
        """
        cutoff_time = now - self.period
        self.request_history = [
            req for req in self.request_history 
            if req > cutoff_time
        ]

    async def _refill_tokens(self, now=None):
        """
        経過時間に応じてトークンを補充

        Args:
            now (float, optional): 現在時刻、指定がなければ現在時刻を使用
        """
        if now is None:
            now = time.time()
        
        # 前回の補充からの経過時間
        elapsed = now - self.last_refill_time
        
        if elapsed > 0:
            # 補充するトークン数を計算
            refill_amount = (elapsed / self.period) * self.rate_limit
            
            if refill_amount > 0:
                self.tokens = min(self.rate_limit, self.tokens + refill_amount)
                self.last_refill_time = now

    def _time_until_next_refill(self):
        """
        次のトークン補充までの時間を計算

        Returns:
            float: 次の補充までの時間（秒）
        """
        if self.tokens >= self.rate_limit:
            return 0
        
        # トークン1つ分の補充に必要な時間
        token_refill_time = self.period / self.rate_limit
        
        # 1トークン分の補充にかかる時間
        return token_refill_time

    async def remaining_tokens(self):
        """
        現在の残りトークン数を取得

        Returns:
            float: 残りトークン数
        """
        async with self._lock:
            now = time.time()
            await self._refill_tokens(now)
            return self.tokens 