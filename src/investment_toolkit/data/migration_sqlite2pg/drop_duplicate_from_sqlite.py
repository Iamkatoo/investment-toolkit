import sqlite3
import pandas as pd

# SQLite の接続
sqlite_db_path = "/Volumes/Sandisk SSD/Codes/Investments/db/f.db"
conn = sqlite3.connect(sqlite_db_path)

def remove_duplicates(table_name):
    """symbol, date の重複データを削除"""
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    # 重複を削除（最初の1件のみ保持）
    #df = df.sort_values(by=["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="first")
    #df = df.sort_values(by=["symbol", "filingDate"]).drop_duplicates(subset=["symbol", "filingDate"], keep="first")
    df = df.sort_values(by=["symbol", "publishedDate"]).drop_duplicates(subset=["symbol", "publishedDate"], keep="first")

    # クリーンなデータを再保存（既存データを削除してから挿入）
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"Cleaned {table_name}: {len(df)} records remaining.")

# CF_q と CF_y の重複を削除
#remove_duplicates("CF_q")
#remove_duplicates("CF_y")
#remove_duplicates("company_profile")
#remove_duplicates("daily_prices")
#remove_duplicates("employee_count")
remove_duplicates("news")
#remove_duplicates("shares")

conn.close()