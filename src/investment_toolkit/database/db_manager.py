#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
データベース接続マネージャ
"""

import os
import logging
import psycopg2
import psycopg2.extras

# ログ設定
logger = logging.getLogger(__name__)

def get_db_connection():
    """データベース接続を取得"""
    # 環境変数から接続情報を取得
    db_name = os.environ.get("DB_NAME", "investment")
    db_user = os.environ.get("DB_USER", "postgres")
    db_password = os.environ.get("DB_PASSWORD", "postgres")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    
    try:
        # データベース接続
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        
        return conn
    except Exception as e:
        logger.error(f"データベース接続エラー: {e}")
        raise

class DatabaseManager:
    """データベース操作マネージャクラス"""
    
    def __init__(self):
        """コンストラクタ"""
        self.conn = None
    
    def __enter__(self):
        """コンテキストマネージャ開始時の処理"""
        self.conn = get_db_connection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ終了時の処理"""
        if self.conn:
            if exc_type is not None:
                # 例外発生時はロールバック
                self.conn.rollback()
            else:
                # 正常終了時はコミット
                self.conn.commit()
            
            self.conn.close()
    
    def commit(self):
        """トランザクションのコミット"""
        if self.conn:
            self.conn.commit()
    
    def rollback(self):
        """トランザクションのロールバック"""
        if self.conn:
            self.conn.rollback()
    
    def fetchone(self, query, params=None):
        """
        1件のデータを取得
        
        Args:
            query (str): SQL文
            params (tuple): パラメータ
            
        Returns:
            tuple: 取得したデータ（1行）
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def fetchall(self, query, params=None):
        """
        複数件のデータを取得
        
        Args:
            query (str): SQL文
            params (tuple): パラメータ
            
        Returns:
            list: 取得したデータ（複数行）
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute(self, query, params=None):
        """
        SQLを実行
        
        Args:
            query (str): SQL文
            params (tuple): パラメータ
            
        Returns:
            int: 影響を受けた行数
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def executemany(self, query, params_list):
        """
        複数のSQLを一括実行
        
        Args:
            query (str): SQL文
            params_list (list): パラメータのリスト
            
        Returns:
            int: 影響を受けた行数
        """
        with self.conn.cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def copy_from(self, file, table, sep='\t', null='\\N', columns=None):
        """
        COPY FROM でファイルからデータをインポート
        
        Args:
            file (file): コピー元ファイル（open済みのファイルオブジェクト）
            table (str): テーブル名
            sep (str): 区切り文字
            null (str): NULL値の表現
            columns (list): カラム名のリスト
        """
        with self.conn.cursor() as cursor:
            cursor.copy_from(file, table, sep, null, columns)
    
    def copy_expert(self, sql, file):
        """
        COPY コマンドを実行
        
        Args:
            sql (str): COPY SQL文
            file (file): ファイルオブジェクト
        """
        with self.conn.cursor() as cursor:
            cursor.copy_expert(sql, file) 