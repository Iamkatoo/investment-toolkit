#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Energyセクターの紐付け問題を修正するスクリプト

このスクリプトは以下の処理を行います：
1. raw_sector = 'Energy' の企業を特定
2. 正しいセクターID（sector_id = 4）を明示的に割り当て
3. 修正前後の統計を出力
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import psycopg2

# スクリプトのあるディレクトリの親ディレクトリをパスに追加
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# ロギングを設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# データベース接続情報を取得
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

def get_db_connection():
    """データベースへの接続を取得"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"データベース接続エラー: {e}")
        raise

def check_current_state():
    """現在のEnergyセクターの状態をチェック"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 'Energy'のraw_sectorを持つレコード数を確認
            cur.execute("""
            SELECT COUNT(*) 
            FROM reference.company_gics 
            WHERE raw_sector = 'Energy'
            """)
            raw_energy_count = cur.fetchone()[0]
            
            # セクターIDが4のレコード数を確認
            cur.execute("""
            SELECT COUNT(*) 
            FROM reference.company_gics 
            WHERE sector_id = 4
            """)
            energy_id_count = cur.fetchone()[0]
            
            # セクターIDとraw_sectorの不一致を確認
            cur.execute("""
            SELECT COUNT(*) 
            FROM reference.company_gics 
            WHERE raw_sector = 'Energy' AND sector_id != 4
            """)
            mismatch_count = cur.fetchone()[0]
            
            return {
                "raw_energy_count": raw_energy_count,
                "energy_id_count": energy_id_count,
                "mismatch_count": mismatch_count
            }

def fix_energy_sector():
    """Energyセクターの紐付けを修正"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 現在の状態を記録
        current_state = check_current_state()
        logger.info(f"修正前の状態: {current_state}")
        
        # raw_sector = 'Energy'に対して、正しいセクターID（4）を設定
        cur.execute("""
        UPDATE reference.company_gics
        SET sector_id = 4
        WHERE raw_sector = 'Energy'
        """)
        
        update_count = cur.rowcount
        logger.info(f"{update_count}件のEnergyセクターレコードを修正しました")
        
        # マスターテーブルのセクター情報を確認
        cur.execute("""
        SELECT sector_id, sector_name
        FROM reference.gics_sector
        WHERE sector_id = 4
        """)
        energy_sector = cur.fetchone()
        if energy_sector:
            logger.info(f"Energyセクターのマスター情報: ID={energy_sector[0]}, 名前={energy_sector[1]}")
        else:
            logger.warning("セクターID=4のレコードがマスターテーブルに存在しません")
        
        # 修正をコミット
        conn.commit()
        
        # 修正後の状態を確認
        after_state = check_current_state()
        logger.info(f"修正後の状態: {after_state}")
        
        return True
    except Exception as e:
        logger.error(f"Energy セクター修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def check_sector_ids():
    """マスターテーブルとマッピングの整合性を確認"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # セクターマスターの全レコードを確認
            cur.execute("""
            SELECT sector_id, sector_name
            FROM reference.gics_sector
            ORDER BY sector_id
            """)
            sectors = cur.fetchall()
            logger.info("セクターマスター一覧:")
            for sector in sectors:
                logger.info(f"  ID={sector[0]}, 名前={sector[1]}")
            
            # 各raw_sectorとセクターIDの組み合わせを確認
            cur.execute("""
            SELECT raw_sector, sector_id, COUNT(*)
            FROM reference.company_gics
            GROUP BY raw_sector, sector_id
            ORDER BY raw_sector, sector_id
            """)
            mappings = cur.fetchall()
            logger.info("現在のマッピング状況:")
            for mapping in mappings:
                logger.info(f"  raw_sector={mapping[0]}, sector_id={mapping[1]}, 件数={mapping[2]}")

def main():
    parser = argparse.ArgumentParser(description='Energyセクターの紐付け問題を修正')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細なログを出力する')
    parser.add_argument('--check-only', '-c', action='store_true', help='現状の確認のみを行い、修正は行わない')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 現状の確認
    check_sector_ids()
    
    if args.check_only:
        logger.info("確認のみのモードで実行しました。修正は行いません。")
        return
    
    # Energy セクターの修正
    logger.info("Energyセクターの修正処理を開始します...")
    result = fix_energy_sector()
    
    if result:
        logger.info("Energyセクターの修正処理が完了しました")
        
        # 修正後の確認（テーブル全体の整合性確認）
        logger.info("すべてのセクターIDのマッピング状況を確認します...")
        check_sector_ids()
    else:
        logger.error("Energyセクターの修正処理に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main() 