"""
Database validation utilities for migration testing

This module provides database write validation and test table management
for safe migration testing during Phase 4 (repository restructuring).
"""

import os
import logging
from typing import Optional, List, Dict
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class DBTestValidator:
    """データベース書き込みテスト用バリデータ"""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.db_test_mode = os.getenv("DB_TEST_MODE", "false").lower() == "true"
        self.skip_db_write = os.getenv("SKIP_DB_WRITE", "false").lower() == "true"
        self.test_tables = self._parse_test_tables()

    def _parse_test_tables(self) -> Dict[str, str]:
        """環境変数からテストテーブルを解析"""
        tables_str = os.getenv("DB_TEST_TABLES", "")
        if not tables_str:
            return {}

        # カンマ区切りでテーブル名を取得
        table_list = [t.strip() for t in tables_str.split(",")]

        # テスト用テーブル名マッピング
        test_mapping = {}
        for table in table_list:
            # スキーマ.テーブル形式に対応
            if "." in table:
                schema, table_name = table.split(".", 1)
                test_mapping[table] = f"{schema}.{table_name}_test"
            else:
                test_mapping[table] = f"{table}_test"

        return test_mapping

    def get_target_table(self, original_table: str) -> str:
        """
        保存先テーブル名を取得

        Args:
            original_table: 元のテーブル名

        Returns:
            テストモード時はテスト用テーブル、それ以外は元のテーブル
        """
        if self.db_test_mode and original_table in self.test_tables:
            test_table = self.test_tables[original_table]
            logger.info(f"🧪 [TEST MODE] {original_table} → {test_table}")
            return test_table

        return original_table

    def ensure_test_table_exists(self, original_table: str, test_table: str) -> bool:
        """
        テスト用テーブルが存在するか確認、なければ作成

        Args:
            original_table: 元のテーブル名
            test_table: テスト用テーブル名

        Returns:
            成功時True
        """
        try:
            inspector = inspect(self.engine)

            # スキーマとテーブル名を分割
            if "." in original_table:
                schema, table = original_table.split(".", 1)
            else:
                schema, table = None, original_table

            if "." in test_table:
                test_schema, test_table_name = test_table.split(".", 1)
            else:
                test_schema, test_table_name = schema, test_table

            # テスト用テーブルが既に存在するか確認
            existing_tables = inspector.get_table_names(schema=test_schema)

            if test_table_name in existing_tables:
                logger.info(f"✅ Test table exists: {test_table}")
                return True

            # 元のテーブルのスキーマをコピーしてテスト用テーブルを作成
            logger.info(f"🔨 Creating test table: {test_table}")

            with self.engine.begin() as conn:
                # PostgreSQLの場合
                if schema:
                    create_sql = f"""
                    CREATE TABLE {test_schema}.{test_table_name}
                    (LIKE {schema}.{table} INCLUDING ALL)
                    """
                else:
                    create_sql = f"""
                    CREATE TABLE {test_table_name}
                    (LIKE {table} INCLUDING ALL)
                    """

                conn.execute(text(create_sql))
                logger.info(f"✅ Test table created: {test_table}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to create test table {test_table}: {e}")
            return False

    def validate_write_capability(self, table_name: str, sample_data: dict) -> bool:
        """
        書き込み可能性を検証（ロールバック）

        Args:
            table_name: テーブル名
            sample_data: サンプルデータ

        Returns:
            書き込み可能ならTrue
        """
        try:
            with self.engine.begin() as conn:
                # サンプルデータをINSERT
                columns = ", ".join(sample_data.keys())
                placeholders = ", ".join([f":{k}" for k in sample_data.keys()])

                insert_sql = f"""
                INSERT INTO {table_name} ({columns})
                VALUES ({placeholders})
                """

                conn.execute(text(insert_sql), sample_data)

                # 強制ロールバック
                raise Exception("Test rollback")

        except Exception as e:
            if "Test rollback" in str(e):
                logger.info(f"✅ Write capability validated (rolled back): {table_name}")
                return True
            else:
                logger.error(f"❌ Write validation failed: {table_name} - {e}")
                return False

    def save_with_validation(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = 'append',
        index: bool = False,
        **kwargs
    ) -> bool:
        """
        データフレームをDBに保存（検証付き）

        Args:
            df: 保存するDataFrame
            table_name: テーブル名
            if_exists: 'append', 'replace', 'fail'
            index: インデックスを保存するか
            **kwargs: pandas.to_sqlへの追加引数

        Returns:
            成功時True
        """
        # SKIP_DB_WRITEが有効な場合はスキップ
        if self.skip_db_write:
            logger.info(f"🔵 [SKIP] DB write skipped: {table_name} ({len(df)} rows)")
            return True

        # テストモード時はテーブル名を変換
        target_table = self.get_target_table(table_name)

        # テスト用テーブルの場合、存在確認・作成
        if target_table != table_name:
            if not self.ensure_test_table_exists(table_name, target_table):
                logger.error(f"❌ Failed to prepare test table: {target_table}")
                return False

        try:
            # スキーマとテーブル名を分離
            if "." in target_table:
                schema, table_only = target_table.split(".", 1)
            else:
                schema, table_only = None, target_table

            # データ保存
            df.to_sql(
                table_only,
                self.engine,
                schema=schema,
                if_exists=if_exists,
                index=index,
                **kwargs
            )

            logger.info(f"✅ Data saved: {target_table} ({len(df)} rows)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save data: {target_table} - {e}")
            return False

    def execute_with_validation(
        self,
        sql: str,
        params: Optional[dict] = None,
        skip_on_dry_run: bool = True
    ) -> bool:
        """
        SQL文を実行（検証付き）

        Args:
            sql: 実行するSQL文
            params: SQLパラメータ
            skip_on_dry_run: Dry-Runモード時にスキップするか

        Returns:
            成功時True
        """
        if self.skip_db_write and skip_on_dry_run:
            logger.info(f"🔵 [SKIP] SQL execution skipped (dry-run mode)")
            return True

        try:
            with self.engine.begin() as conn:
                if params:
                    conn.execute(text(sql), params)
                else:
                    conn.execute(text(sql))

            logger.info(f"✅ SQL executed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ SQL execution failed: {e}")
            return False


# グローバルインスタンス
_db_validator = None


def get_db_validator(engine: Engine) -> DBTestValidator:
    """
    DBバリデータのシングルトン取得

    Args:
        engine: SQLAlchemyエンジン

    Returns:
        DBTestValidator インスタンス
    """
    global _db_validator
    if _db_validator is None:
        _db_validator = DBTestValidator(engine)
    return _db_validator


def reset_db_validator():
    """DBバリデータをリセット（テスト用）"""
    global _db_validator
    _db_validator = None
