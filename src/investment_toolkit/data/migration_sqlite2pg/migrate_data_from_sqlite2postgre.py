import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import re

def camel_to_snake(name):
    """UpperCamelCase → lower_snake_case に変換"""
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

def convert_dataframe_columns(df):
    """DataFrame のカラム名を一括変換（キャメルケース → スネークケース）"""
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df

def fix_common_typos(df):
    """特定のカラム名のタイポを修正"""
    typo_mapping = {
        "filling_date": "filing_date",  # タイポ修正
        "ebitdaratio": "ebitda_ratio",
        "epsdiluted": "eps_diluted",
        "othertotal_stockholders_equity": "other_total_stockholders_equity",
        "other_investing_activites": "other_investing_activities",
        "net_cash_used_for_investing_activites": "net_cash_used_for_investing_activities",
        "other_financing_activites": "other_financing_activities",
    }
    df.rename(columns=typo_mapping, inplace=True)
    return df

def add_period_type(df, period):
    """period_type を自動追加する関数"""
    if period and "period_type" not in df.columns:
        df["period_type"] = period
    return df

def preprocess_dataframe(df, period=None):
    """カラム名の変換 + タイポ修正 + period_type 追加"""
    df = convert_dataframe_columns(df)  # キャメルケース → スネークケース
    df = fix_common_typos(df)  # タイポ修正
    df = add_period_type(df, period)  # period_type 追加
    
    # period が ''（空文字）なら NULL に変換
    if "period" in df.columns:
        df["period"].replace("", None, inplace=True)
    
    return df

# SQLite & PostgreSQL の接続情報
sqlite_db_path = "/Volumes/Sandisk SSD/Codes/Investments/db/f.db"
#sqlite_db_path = "/Users/HOME/Codes/Investment/db/f.db"
postgres_conn_str = "postgresql://HOME@localhost:5432/investment"

# PostgreSQL エンジン作成
pg_engine = create_engine(postgres_conn_str)

# SQLite3 に接続
sqlite_conn = sqlite3.connect(sqlite_db_path)

# SQLite → PostgreSQL 移行用テーブルリスト
'''tables = {
    #"PL_q": ("fmp_data", "income_statements", "quarterly"),
    #"PL_y": ("fmp_data", "income_statements", "annual"),
    #"BS_q": ("fmp_data", "balance_sheets", "quarterly"),
    #"BS_y": ("fmp_data", "balance_sheets", "annual"),
    "CF_q": ("fmp_data", "cash_flows", "quarterly"),
    "CF_y": ("fmp_data", "cash_flows", "annual"),
}
'''
tables = {
    #"company_profile": ("fmp_data", "company_profile"),
    #"daily_prices": ("fmp_data", "daily_prices"),
    "employee_count": ("fmp_data", "employee_count"),
    "news": ("fmp_data", "news"),
    "shares": ("fmp_data", "shares"),
}

# 各テーブルごとにデータを取得し、PostgreSQL に保存
#for sqlite_table, (pg_schema, pg_table, period) in tables.items():
for sqlite_table, (pg_schema, pg_table) in tables.items():
    print(f"Processing {sqlite_table} → {pg_table}")

    # SQLite からデータを取得
    df = pd.read_sql(f"SELECT * FROM {sqlite_table}", sqlite_conn)

    # `preprocess_dataframe()` を適用
    #df = preprocess_dataframe(df, period)
    df = convert_dataframe_columns(df)
    
    if sqlite_table == "employee_count":
        df = df.drop(columns=['index'], errors='ignore')
    if sqlite_table == 'company_profile':
        df['ipo_date'] = df["ipo_date"].replace('', None)
        df['full_time_employees'] = df["full_time_employees"].replace('', None)

    # PostgreSQL にデータを追加
    df.to_sql(pg_table, pg_engine, schema=pg_schema, if_exists="append", index=False)

sqlite_conn.close()
print("Data migration completed successfully!")

f
# next fmp_apiクラスを作り込む
    # database_utilsの中身をどっちに入れるべきか
    # APIキーを二つで運用する感じで。エラー出たら2個目の使う
    # lower snake化して、間違った名前のcolun名は直すとかちょっと手直しが必要
