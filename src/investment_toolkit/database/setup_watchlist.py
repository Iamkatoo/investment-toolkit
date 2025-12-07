#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ウォッチリスト機能のデータベーススキーマ作成スクリプト
既存のデータベースには一切影響せず、新しいスキーマのみを作成
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
import subprocess

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# プロジェクト内のモジュールをインポート
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME


def create_database_connection():
    """データベース接続を作成"""
    try:
        SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(SQLALCHEMY_DATABASE_URI)
        return engine
    except Exception as e:
        print(f"データベース接続エラー: {e}")
        return None


def check_existing_watchlist_schema(engine):
    """既存のウォッチリストスキーマをチェック"""
    try:
        query = text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'watchlist'")
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
            return result is not None
    except Exception as e:
        print(f"スキーマチェックエラー: {e}")
        return False


def execute_sql_file(engine, sql_file_path):
    """SQLファイルを実行"""
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
            
        # SQLを一括実行（関数定義などを考慮）
        with engine.begin() as conn:
            conn.execute(text(sql_content))
            print("SQLファイルの実行が完了しました")
            return True
                
    except Exception as e:
        print(f"SQL実行エラー: {e}")
        return False


def verify_schema_creation(engine):
    """スキーマ作成の確認"""
    try:
        # スキーマの存在確認
        schema_query = text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'watchlist'")
        
        # テーブルの存在確認
        tables_query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'watchlist'
        ORDER BY table_name
        """)
        
        # ビューの存在確認
        views_query = text("""
        SELECT table_name 
        FROM information_schema.views 
        WHERE table_schema = 'watchlist'
        """)
        
        with engine.connect() as conn:
            # スキーマ確認
            schema_result = conn.execute(schema_query).fetchone()
            if not schema_result:
                print("❌ ウォッチリストスキーマが作成されていません")
                return False
            
            # テーブル確認
            tables_result = conn.execute(tables_query).fetchall()
            expected_tables = {'tracked_stocks', 'performance_tracking', 'analysis_performance'}
            actual_tables = {row[0] for row in tables_result}
            
            if not expected_tables.issubset(actual_tables):
                missing_tables = expected_tables - actual_tables
                print(f"❌ 不足しているテーブル: {missing_tables}")
                return False
            
            # ビュー確認
            views_result = conn.execute(views_query).fetchall()
            actual_views = {row[0] for row in views_result}
            
            if 'vw_current_watchlist' not in actual_views:
                print("❌ ビュー 'vw_current_watchlist' が作成されていません")
                return False
            
            print("✅ ウォッチリストスキーマが正常に作成されました")
            print(f"   - スキーマ: watchlist")
            print(f"   - テーブル: {', '.join(sorted(actual_tables))}")
            print(f"   - ビュー: {', '.join(sorted(actual_views))}")
            return True
            
    except Exception as e:
        print(f"スキーマ確認エラー: {e}")
        return False


def main():
    """メイン処理"""
    print("🚀 ウォッチリスト機能のデータベーススキーマ作成を開始します...")
    
    # データベース接続
    print("📊 データベースに接続中...")
    engine = create_database_connection()
    if not engine:
        print("❌ データベース接続に失敗しました")
        return False
    
    # 既存スキーマのチェック
    print("🔍 既存のウォッチリストスキーマをチェック中...")
    schema_exists = check_existing_watchlist_schema(engine)
    
    if schema_exists:
        print("⚠️  ウォッチリストスキーマが既に存在します")
        response = input("既存のスキーマを再作成しますか？ (y/N): ").lower()
        if response != 'y':
            print("処理を中止しました")
            return False
    
    # SQLファイルのパス
    sql_file_path = Path(__file__).parent / "create_watchlist_schema.sql"
    
    if not sql_file_path.exists():
        print(f"❌ SQLファイルが見つかりません: {sql_file_path}")
        return False
    
    # SQLファイル実行
    print("🛠️  ウォッチリストスキーマを作成中...")
    success = execute_sql_file(engine, sql_file_path)
    
    if not success:
        print("❌ スキーマの作成に失敗しました")
        return False
    
    # 作成確認
    print("✅ スキーマ作成の確認中...")
    verification_success = verify_schema_creation(engine)
    
    if verification_success:
        print("🎉 ウォッチリスト機能のデータベースセットアップが完了しました！")
        print("\n次のステップ:")
        print("1. レポートにチェックボックス機能を追加")
        print("2. ウォッチリスト管理API を実装")
        print("3. ウォッチリスト専用レポートを作成")
        return True
    else:
        print("❌ スキーマの確認に失敗しました")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✨ セットアップ完了")
            sys.exit(0)
        else:
            print("\n💥 セットアップ失敗")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        sys.exit(1) 