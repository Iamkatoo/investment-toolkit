#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非同期データベース接続管理モジュール
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import asyncpg

# 環境変数を読み込む
load_dotenv()

# ログ設定
logger = logging.getLogger(__name__)

# 環境変数からデータベース接続情報を取得
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "investment")


class AsyncDBConnectionPool:
    """非同期データベース接続プール"""

    def __init__(self, dsn=None, min_size=2, max_size=10):
        """
        初期化

        Args:
            dsn (str, optional): データソース名。指定がない場合は環境変数から構築
            min_size (int): プールの最小接続数
            max_size (int): プールの最大接続数
        """
        if dsn is None:
            # 接続文字列の作成
            self.dsn = f"postgres://{DB_USER}"
            
            if DB_PASSWORD:
                self.dsn += f":{DB_PASSWORD}"
                
            self.dsn += f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        else:
            self.dsn = dsn
            
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """接続プールの初期化"""
        async with self._init_lock:
            if self.pool is None:
                try:
                    logger.debug(f"DB接続プールを初期化: min_size={self.min_size}, max_size={self.max_size}")
                    self.pool = await asyncpg.create_pool(
                        dsn=self.dsn,
                        min_size=self.min_size,
                        max_size=self.max_size
                    )
                    return self.pool
                except Exception as e:
                    logger.error(f"DB接続プールの初期化エラー: {e}")
                    raise
            return self.pool

    async def close(self):
        """接続プールをクローズ"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.debug("DB接続プールをクローズしました")

    @asynccontextmanager
    async def acquire(self):
        """接続をプールから取得"""
        if self.pool is None:
            await self.initialize()
            
        async with self.pool.acquire() as connection:
            yield connection

    async def execute(self, query, *args, timeout=None):
        """
        SQLクエリを実行

        Args:
            query (str): 実行するSQLクエリ
            *args: クエリパラメータ
            timeout (float, optional): タイムアウト（秒）

        Returns:
            str: 実行結果のステータスタグ
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)

    async def fetch(self, query, *args, timeout=None):
        """
        SQLクエリを実行して全ての結果を取得

        Args:
            query (str): 実行するSQLクエリ
            *args: クエリパラメータ
            timeout (float, optional): タイムアウト（秒）

        Returns:
            List[asyncpg.Record]: 取得したデータのリスト
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    async def fetchrow(self, query, *args, timeout=None):
        """
        SQLクエリを実行して一つの行を取得

        Args:
            query (str): 実行するSQLクエリ
            *args: クエリパラメータ
            timeout (float, optional): タイムアウト（秒）

        Returns:
            asyncpg.Record: 取得したデータの行
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)

    async def fetchval(self, query, *args, column=0, timeout=None):
        """
        SQLクエリを実行して一つの値を取得

        Args:
            query (str): 実行するSQLクエリ
            *args: クエリパラメータ
            column (int): 取得する列の位置
            timeout (float, optional): タイムアウト（秒）

        Returns:
            Any: 取得した値
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)

    async def copy_records_to_table(
        self, table_name, records=None, columns=None, schema_name=None, timeout=None
    ):
        """
        レコードをデータベーステーブルにコピー

        Args:
            table_name (str): 対象テーブル名
            records (List[Tuple]): コピーするレコードのリスト
            columns (List[str], optional): 対象カラムのリスト
            schema_name (str, optional): スキーマ名
            timeout (float, optional): タイムアウト（秒）

        Returns:
            str: 実行結果のステータスタグ
        """
        async with self.acquire() as conn:
            return await conn.copy_records_to_table(
                table_name=table_name,
                records=records,
                columns=columns,
                schema_name=schema_name,
                timeout=timeout
            )


class AsyncDBManager:
    """非同期データベース操作を管理するクラス"""
    
    def __init__(self, pool=None):
        """
        初期化

        Args:
            pool (AsyncDBConnectionPool, optional): DB接続プール
        """
        self.pool = pool if pool else AsyncDBConnectionPool()
        
    async def initialize(self):
        """初期化処理"""
        if self.pool:
            await self.pool.initialize()
    
    async def close(self):
        """終了処理"""
        if self.pool:
            await self.pool.close()
    
    async def execute_transaction(self, *queries_with_params):
        """
        トランザクション内で複数のクエリを実行

        Args:
            queries_with_params: (query, params) のタプルのリスト

        Returns:
            List[Any]: 各クエリの実行結果
        """
        if self.pool is None:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                results = []
                for query, params in queries_with_params:
                    if params:
                        result = await conn.execute(query, *params)
                    else:
                        result = await conn.execute(query)
                    results.append(result)
                return results 