import sqlite3
import pandas as pd
from sqlalchemy import create_engine

# postgreの接続情報
postgres_conn_str = "postgresql://HOME@localhost:5432/investment"

# PostgreSQL エンジン作成
pg_engine = create_engine(postgres_conn_str)

sqlite_fred_db_path = "/Volumes/Sandisk SSD/Codes/Investments/db/fred.db"
sqlite_conn = sqlite3.connect(sqlite_fred_db_path)

# FRED の全テーブルリスト
fred_tables = ["BAA10Y", "CPIAUCSL", "CPILEGSL", "CPILFESL", "DGS10", "FEDFUNDS", 
               "GDP", "PCEPI", "TWEXBGSMTH", "UNRATE", "yield_difference"]

dfs = []

for table in fred_tables:
    print(f"Processing {table}")

    # SQLite からデータを取得
    df = pd.read_sql(f"SELECT * FROM {table}", sqlite_conn)

    # `indicator_name` カラムを追加
    df["indicator_name"] = table
    
    if 'yield_diff' in df.columns:
        df.rename(columns={'yield_diff':'value'}, inplace=True)
    else:
        df.rename(columns={table:'value'}, inplace=True)

    # 頻度を設定
    if table == "GDP":
        df["frequency"] = "quarterly"
    elif table in ["CPIAUCSL", "CPIEGSL", "CPILFESL", "PCEPI", "UNRATE"]:
        df["frequency"] = "monthly"
    else:
        df["frequency"] = "daily"

    dfs.append(df)

# すべての FRED データを統合
fred_df = pd.concat(dfs, ignore_index=True)

# PostgreSQL に保存
fred_df.to_sql("economic_indicators", pg_engine, schema="fred_data", if_exists="append", index=False)

sqlite_conn.close()
print("FRED data migration completed successfully!")