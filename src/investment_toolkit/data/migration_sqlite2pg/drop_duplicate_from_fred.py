import sqlite3
import pandas as pd

# SQLite のデータベースパス
sqlite_fred_db_path = "/Volumes/Sandisk SSD/Codes/Investments/db/fred.db"
conn = sqlite3.connect(sqlite_fred_db_path)

# 特殊なカラム名のテーブル
special_columns = {"yield_difference": "yield_diff"}

def remove_duplicates_from_fred():
    """SQLite (fred.db) の全テーブルで `date` の重複を削除"""
    # 全テーブルを取得
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(tables_query, conn)["name"].tolist()

    for table in tables:
        print(f"Processing {table}...")

        # 特殊カラム名がある場合は適用
        value_column = special_columns.get(table, table)  # デフォルトでテーブル名をカラム名とする

        # データ取得
        df = pd.read_sql(f"SELECT * FROM {table}", conn)

        # `date` の重複を削除（最初の1件のみ保持）
        if "date" in df.columns:
            df = df.sort_values(by=["date"]).drop_duplicates(subset=["date"], keep="first")

            # クリーンなデータを再保存（既存データを削除してから挿入）
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"Cleaned {table}: {len(df)} records remaining.")

# 実行
remove_duplicates_from_fred()

# SQLite 接続を閉じる
conn.close()