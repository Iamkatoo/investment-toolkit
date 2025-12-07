import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

def connect_to_postgresql(host=None, database=None, user=None, password=None, port=None):
    """PostgreSQLデータベースに接続する"""
    # .envファイルから環境変数を取得、指定がなければデフォルト値を使用
    host = host or os.getenv('DB_HOST', 'localhost')
    database = database or os.getenv('DB_NAME', 'finance_db')
    user = user or os.getenv('DB_USER', 'HOME')
    password = password or os.getenv('DB_PASSWORD', '')
    port = port or os.getenv('DB_PORT', '5432')
    
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    print(f"接続文字列: {connection_string}")  # 接続情報を表示（パスワードは本番環境では削除してください）
    engine = create_engine(connection_string)
    return engine

def check_database_tables(engine):
    """データベース内のテーブルとサンプルデータを確認する"""
    try:
        # スキーマ情報の取得
        with engine.connect() as connection:
            # テーブル一覧を取得
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            tables = pd.read_sql_query(text(tables_query), connection)
            print("データベーステーブル一覧:")
            print(tables)
            
            # 各テーブルのサンプルデータを取得
            for table in tables['table_name']:
                try:
                    sample_query = f"SELECT * FROM {table} LIMIT 5"
                    sample_data = pd.read_sql_query(text(sample_query), connection)
                    print(f"\n{table}のサンプルデータ:")
                    print(f"行数: {len(sample_data)}")
                    print(f"列: {', '.join(sample_data.columns)}")
                    if not sample_data.empty:
                        print(sample_data.head(2))
                except Exception as e:
                    print(f"{table}のデータ取得中にエラー: {e}")
                    
            # income_statementsとsymbol_statusの件数確認
            count_query = """
            SELECT 
                (SELECT COUNT(*) FROM income_statements WHERE period_type = 'FY') AS income_statements_count,
                (SELECT COUNT(*) FROM symbol_status WHERE is_active = true) AS active_symbols_count,
                (SELECT COUNT(*) FROM balance_sheets WHERE period_type = 'FY') AS balance_sheets_count,
                (SELECT COUNT(*) FROM cash_flows WHERE period_type = 'FY') AS cash_flows_count
            """
            try:
                counts = pd.read_sql_query(text(count_query), connection)
                print("\nテーブル件数:")
                print(counts)
            except Exception as e:
                print(f"件数取得中にエラー: {e}")
                
    except Exception as e:
        print(f"データベース確認中にエラー: {e}")

def test_simple_join(engine):
    """シンプルなJOINクエリでデータ取得をテスト"""
    try:
        with engine.connect() as connection:
            # 各テーブルが正しく結合できるか確認するシンプルなクエリ
            simple_query = """
            SELECT 
                s.symbol, s.name, i.calendar_year, i.revenue, i.net_income
            FROM 
                symbol_status s
            JOIN 
                income_statements i ON s.symbol = i.symbol
            WHERE 
                i.period_type = 'FY'
                AND s.is_active = true
            LIMIT 10
            """
            result = pd.read_sql_query(text(simple_query), connection)
            print("\nシンプルJOINクエリのテスト結果:")
            print(f"取得行数: {len(result)}")
            if not result.empty:
                print(result.head())
            else:
                print("データが取得できませんでした。")
    except Exception as e:
        print(f"シンプルJOINクエリ実行中にエラー: {e}")

def screen_stable_growth_stocks(engine, min_roa=0, min_roe=0, min_equity_ratio=0, 
                                max_pbr=10, min_revenue_growth=0):
    """
    安定成長株をスクリーニングする関数
    条件を最低限に設定してデバッグ
    """
    
    # SQLクエリの構築 - デバッグ用に簡略化
    query = """
    WITH 
    -- 直近の財務データを取得
    latest_financials AS (
        SELECT 
            i.symbol,
            i.calendar_year,
            i.revenue,
            i.net_income,
            b.total_assets,
            b.total_stockholders_equity,
            c.free_cash_flow
        FROM income_statements i
        JOIN balance_sheets b ON i.symbol = b.symbol AND i.calendar_year = b.calendar_year AND i.period_type = b.period_type
        JOIN cash_flows c ON i.symbol = c.symbol AND i.calendar_year = c.calendar_year AND i.period_type = c.period_type
        WHERE i.period_type = 'FY'
        AND b.period_type = 'FY'
        AND c.period_type = 'FY'
    ),
    
    -- 直近年度のデータを取得
    latest_year_data AS (
        SELECT 
            f.symbol,
            f.calendar_year,
            f.revenue,
            f.net_income,
            f.total_assets,
            f.total_stockholders_equity,
            f.free_cash_flow,
            (f.net_income / NULLIF(f.total_assets, 0)) * 100 AS roa,
            (f.net_income / NULLIF(f.total_stockholders_equity, 0)) * 100 AS roe,
            (f.total_stockholders_equity / NULLIF(f.total_assets, 0)) * 100 AS equity_ratio
        FROM latest_financials f
        INNER JOIN (
            SELECT symbol, MAX(calendar_year) as max_year
            FROM latest_financials
            GROUP BY symbol
        ) m ON f.symbol = m.symbol AND f.calendar_year = m.max_year
    ),
    
    -- 5年間の売上高データを取得して成長率を計算
    revenue_growth AS (
        SELECT 
            symbol,
            (POWER(
                (MAX(CASE WHEN rnk = 1 THEN revenue END) / 
                 NULLIF(MAX(CASE WHEN rnk = 5 THEN revenue END), 0)), 
                0.25) - 1) * 100 AS avg_annual_growth
        FROM (
            SELECT 
                symbol, 
                calendar_year, 
                revenue,
                RANK() OVER (PARTITION BY symbol ORDER BY calendar_year DESC) as rnk
            FROM latest_financials
        ) ranked_data
        GROUP BY symbol
        HAVING COUNT(DISTINCT calendar_year) >= 5
    )
    
    -- 最終的なスクリーニング結果 - 条件を最小限に
    SELECT 
        s.name, 
        l.symbol, 
        s.exchange,
        l.roa,
        l.roe,
        l.equity_ratio,
        l.free_cash_flow,
        r.avg_annual_growth AS revenue_growth_5yr
    FROM latest_year_data l
    JOIN symbol_status s ON l.symbol = s.symbol
    JOIN revenue_growth r ON l.symbol = r.symbol
    WHERE s.is_active = true
    AND l.roa > :min_roa
    AND l.roe > :min_roe
    AND l.equity_ratio > :min_equity_ratio
    AND r.avg_annual_growth > :min_revenue_growth
    ORDER BY r.avg_annual_growth DESC
    LIMIT 100;
    """
    
    # SQLパラメータの設定
    params = {
        "min_roa": min_roa,
        "min_roe": min_roe,
        "min_equity_ratio": min_equity_ratio,
        "min_revenue_growth": min_revenue_growth
    }
    
    try:
        # クエリの実行とデータフレームへの変換
        print(f"実行するクエリパラメータ: {params}")
        with engine.connect() as connection:
            result = pd.read_sql_query(text(query), connection, params=params)
        
        print(f"取得件数: {len(result)}")
        return result
    except Exception as e:
        print(f"スクリーニングクエリ実行中にエラー: {e}")
        return pd.DataFrame()

def main():
    # データベース接続
    engine = connect_to_postgresql()
    
    # データベースのテーブルとサンプルデータを確認
    print("\n==== データベース構造の確認 ====")
    check_database_tables(engine)
    
    # シンプルなJOINクエリでテスト
    print("\n==== シンプルなJOINクエリのテスト ====")
    test_simple_join(engine)
    
    # スクリーニング条件（デバッグ用に最小限の条件）
    min_roa = 0.0  # すべての条件を0に設定して
    min_roe = 0.0  # まずデータが取得できるか確認
    min_equity_ratio = 0.0
    max_pbr = 10.0
    min_revenue_growth = 0.0
    
    # スクリーニングの実行
    print("\n==== スクリーニングクエリのテスト（最小限の条件） ====")
    result_df = screen_stable_growth_stocks(
        engine,
        min_roa=min_roa,
        min_roe=min_roe,
        min_equity_ratio=min_equity_ratio,
        max_pbr=max_pbr,
        min_revenue_growth=min_revenue_growth
    )
    
    # 結果の表示
    if not result_df.empty:
        print(f"スクリーニング結果：{len(result_df)}銘柄")
        print(result_df.head())
    else:
        print("条件に合致する銘柄は見つかりませんでした。データベース構造を確認してください。")

if __name__ == "__main__":
    main()