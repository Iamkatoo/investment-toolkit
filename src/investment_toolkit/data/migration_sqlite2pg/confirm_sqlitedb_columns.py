import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import re

# SQLite & PostgreSQL の接続情報
sqlite_db_path = "/Volumes/Sandisk SSD/Codes/Investments/db/f.db"

# SQLite3 に接続
sqlite_conn = sqlite3.connect(sqlite_db_path)

# SQLite → PostgreSQL 移行用テーブルリスト
tables = {
    "company_profile": ("fmp_data", "company_profile"),
    "daily_prices": ("fmp_data", "daily_prices"),
    "employee_count": ("fmp_data", "employee_count"),
    "news": ("fmp_data", "news"),
    "shares": ("fmp_data", "shares"),
}

# 各テーブルのカラム名を取得する関数
def get_column_names(table_name):
    """SQLiteのテーブルからカラム名を取得"""
    query = f"PRAGMA table_info({table_name})"
    df_columns = pd.read_sql(query, sqlite_conn)
    return df_columns["name"].tolist()

def camel_to_snake(name):
    """UpperCamelCase → lower_snake_case に変換"""
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

# 各テーブルのカラム名を取得
for sqlite_table, (pg_schema, pg_table) in tables.items():
    print(f"Processing {sqlite_table} → {pg_table}")

    # カラム名を取得
    column_names = get_column_names(sqlite_table)
    print(f"Columns in {sqlite_table}: {column_names}")
    print(f"snake name: {[camel_to_snake(col) for col in column_names]}")