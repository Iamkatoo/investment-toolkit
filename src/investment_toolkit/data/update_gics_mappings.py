#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GICSセクター・業種マッピングの一括修正スクリプト

このスクリプトは以下の処理を統合します：
1. GICSテーブルのアクティブステータス更新
2. セクターIDがないレコードの修正
3. Energyセクターの業種IDマッピング修正
4. Consumer Discretionaryセクターの特殊ケース修正
5. Materialsセクターの修正
6. 業種（インダストリー）マッピングの修正
"""

import os
import sys
import logging
import argparse
import re
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from typing import Dict, List, Tuple, Optional, Set

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
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from investment_analysis.utilities import industry_mapper

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

def update_gics_active_status():
    """
    GICSテーブルのアクティブステータスを更新
    - 最新のプライスデータがある企業をアクティブとしてマーク
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("GICSテーブルのアクティブステータス更新を開始します...")
        
        # アクティブステータスをリセット
        cur.execute("UPDATE reference.company_gics SET is_active = FALSE")
        logger.info("すべての企業のアクティブステータスをリセットしました")
        
        # 最新のプライスデータがある企業をアクティブとしてマーク
        cur.execute("""
        UPDATE reference.company_gics cg
        SET is_active = TRUE
        FROM (
            SELECT DISTINCT symbol 
            FROM fmp_data.daily_prices 
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        ) dp
        WHERE cg.symbol = dp.symbol
        """)
        
        update_count = cur.rowcount
        logger.info(f"{update_count}件の企業をアクティブとしてマークしました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"アクティブステータス更新中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def repair_gics_sector_mapping():
    """
    セクターIDがないレコードを修正
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("セクターマッピングの修復を開始します...")
        
        # セクターIDがないレコード数
        cur.execute("SELECT COUNT(*) FROM reference.company_gics WHERE sector_id IS NULL")
        null_sector_count = cur.fetchone()[0]
        logger.info(f"セクターIDがないレコード数: {null_sector_count}")
        
        if null_sector_count > 0:
            # raw_sectorに基づいてセクターIDを割り当て
            cur.execute("""
            UPDATE reference.company_gics cg
            SET sector_id = gs.sector_id
            FROM reference.gics_sector gs
            WHERE cg.raw_sector = gs.sector_name AND cg.sector_id IS NULL
            """)
            
            update_count = cur.rowcount
            logger.info(f"{update_count}件のレコードにセクターIDを割り当てました")
            
            # 残りのnullレコードを確認
            cur.execute("SELECT COUNT(*) FROM reference.company_gics WHERE sector_id IS NULL")
            remaining_null = cur.fetchone()[0]
            
            if remaining_null > 0:
                logger.info(f"セクターIDがまだ割り当てられていないレコード: {remaining_null}件")
                
                # 残りのnullレコードの詳細を表示
                cur.execute("""
                SELECT raw_sector, COUNT(*) 
                FROM reference.company_gics 
                WHERE sector_id IS NULL 
                GROUP BY raw_sector 
                ORDER BY COUNT(*) DESC
                """)
                
                for raw_sector, count in cur.fetchall():
                    logger.info(f"  raw_sector='{raw_sector}': {count}件")
                
                # セクターマッピングの拡張
                sector_map_extended = {
                    "Energy": "Energy",
                    "Materials": "Materials",
                    "Industrials": "Industrials",
                    "Consumer Discretionary": "Consumer Discretionary",
                    "Consumer Staples": "Consumer Staples",
                    "Health Care": "Health Care",
                    "Financials": "Financials",
                    "Information Technology": "Information Technology",
                    "Communication Services": "Communication Services",
                    "Utilities": "Utilities",
                    "Real Estate": "Real Estate",
                    "Unclassified": "Unclassified",
                    # 以下、variation
                    "Technology": "Information Technology",
                    "Communications": "Communication Services",
                    "Financial": "Financials",
                    "Healthcare": "Health Care",
                    "Industrial": "Industrials",
                    "Utility": "Utilities",
                    "Consumer Cyclical": "Consumer Discretionary",
                    "Consumer Defensive": "Consumer Staples",
                    "Basic Materials": "Materials",
                    "Real-Estate": "Real Estate",
                }
                
                # 拡張マッピングに基づいて更新
                for raw_sector, standard_sector in sector_map_extended.items():
                    cur.execute("""
                    UPDATE reference.company_gics cg
                    SET sector_id = gs.sector_id
                    FROM reference.gics_sector gs
                    WHERE cg.raw_sector = %s AND gs.sector_name = %s AND cg.sector_id IS NULL
                    """, (raw_sector, standard_sector))
                    
                    update_count = cur.rowcount
                    if update_count > 0:
                        logger.info(f"{update_count}件の'{raw_sector}'を'{standard_sector}'に修正しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"セクターマッピング修復中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def fix_energy_industry_mapping():
    """Energyセクターの企業に適切なindustry_idを割り当て"""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("Energyセクターの業種マッピング修正を開始します...")
        
        # Energyセクターの企業数を確認
        cur.execute("SELECT COUNT(*) FROM reference.company_gics WHERE sector_id = 4")
        energy_count = cur.fetchone()[0]
        logger.info(f"Energyセクター（ID=4）の企業数: {energy_count}")
        
        # Energyセクターの業種IDを取得
        cur.execute("""
        SELECT industry_id, industry_name
        FROM reference.gics_industry
        WHERE sector_id = 4
        """)
        energy_industries = {name: id for id, name in cur.fetchall()}
        
        # マッピングを定義
        industry_pattern_map = {
            "Oil & Gas Equipment & Services": "Energy Equipment & Services",
            "Energy Equipment & Services": "Energy Equipment & Services",
            "Oil & Gas - E&P": "Oil, Gas & Consumable Fuels",
            "Oil & Gas Integrated": "Oil, Gas & Consumable Fuels",
            "Oil & Gas Midstream": "Oil, Gas & Consumable Fuels",
            "Oil & Gas Refining & Marketing": "Oil, Gas & Consumable Fuels",
            "Oil & Gas Drilling": "Oil, Gas & Consumable Fuels",
            "Uranium": "Oil, Gas & Consumable Fuels",
            "Coal": "Oil, Gas & Consumable Fuels",
        }
        
        # 未分類の企業を修正
        updated_count = 0
        for raw_industry, standard_industry in industry_pattern_map.items():
            industry_id = energy_industries.get(standard_industry)
            if industry_id:
                cur.execute("""
                UPDATE reference.company_gics
                SET industry_id = %s
                WHERE sector_id = 4 AND raw_industry = %s AND (industry_id = 47 OR industry_id IS NULL)
                """, (industry_id, raw_industry))
                
                cur_update_count = cur.rowcount
                if cur_update_count > 0:
                    logger.info(f"{cur_update_count}件の'{raw_industry}'を'{standard_industry}'(ID={industry_id})に修正しました")
                    updated_count += cur_update_count
        
        # より広いパターンマッチによる修正
        pattern_matches = [
            ("%Oil%Gas%", "Oil, Gas & Consumable Fuels"),
            ("%Energy%", "Energy Equipment & Services"),
            ("%Coal%", "Oil, Gas & Consumable Fuels"),
        ]
        
        for pattern, standard_industry in pattern_matches:
            industry_id = energy_industries.get(standard_industry)
            if industry_id:
                cur.execute("""
                UPDATE reference.company_gics
                SET industry_id = %s
                WHERE sector_id = 4 AND raw_industry LIKE %s AND (industry_id = 47 OR industry_id IS NULL)
                """, (industry_id, pattern))
                
                cur_update_count = cur.rowcount
                if cur_update_count > 0:
                    logger.info(f"{cur_update_count}件のパターン'{pattern}'を'{standard_industry}'(ID={industry_id})に修正しました")
                    updated_count += cur_update_count
        
        logger.info(f"合計{updated_count}件のEnergy企業の業種IDを修正しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Energy業種マッピング修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def fix_consumer_discretionary():
    """
    Consumer Discretionaryセクターの問題を修正
    - Construction & Engineeringが未分類
    - Automobilesが未分類
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("Consumer Discretionaryセクターの修正を開始します...")
        
        # Consumer Discretionaryのセクターを確認
        cur.execute("""
        SELECT sector_id, sector_name
        FROM reference.gics_sector
        WHERE sector_name = 'Consumer Discretionary'
        """)
        consumer_sector = cur.fetchone()
        consumer_sector_id = consumer_sector[0] if consumer_sector else None
        
        if consumer_sector_id is None:
            logger.error("Consumer Discretionaryセクターが見つかりません")
            return False
        
        logger.info(f"正しいConsumer DiscretionaryのセクターID: {consumer_sector_id}")
        
        # 修正対象の業種を確認
        target_industries = ["Construction & Engineering", "Automobiles", "Textiles, Apparel & Luxury Goods"]
        
        for industry_name in target_industries:
            # 業種IDを確認
            cur.execute("""
            SELECT industry_id, industry_name, sector_id
            FROM reference.gics_industry
            WHERE industry_name = %s
            """, (industry_name,))
            industry_info = cur.fetchone()
            
            if industry_info:
                industry_id = industry_info[0]
                current_sector_id = industry_info[2]
                
                logger.info(f"業種 '{industry_name}' (ID: {industry_id})、現在のセクターID: {current_sector_id}")
                
                # 業種のセクターIDが Consumer Discretionary でない場合は修正
                if industry_name == "Construction & Engineering" and current_sector_id != 3:  # Industrials
                    # Construction & Engineering は Industrials セクターに属する
                    cur.execute("""
                    UPDATE reference.gics_industry
                    SET sector_id = 3
                    WHERE industry_id = %s
                    """, (industry_id,))
                    logger.info(f"{industry_name}のセクターIDをIndustrials(3)に更新しました")
                elif industry_name != "Construction & Engineering" and current_sector_id != consumer_sector_id:
                    # その他の対象業種は Consumer Discretionary に修正
                    cur.execute("""
                    UPDATE reference.gics_industry
                    SET sector_id = %s
                    WHERE industry_id = %s
                    """, (consumer_sector_id, industry_id))
                    logger.info(f"{industry_name}のセクターIDを{current_sector_id}から{consumer_sector_id}に更新しました")
            else:
                logger.warning(f"業種 '{industry_name}' が見つかりません")
            
            # 対応するraw_industryの企業を修正
            if industry_name == "Construction & Engineering":
                # Construction & Engineering は Industrials セクターに修正
                cur.execute("""
                UPDATE reference.company_gics
                SET 
                    sector_id = 3,
                    industry_id = COALESCE(
                        (SELECT industry_id FROM reference.gics_industry WHERE industry_name = %s),
                        industry_id
                    )
                WHERE raw_industry LIKE %s OR raw_industry LIKE %s
                """, (industry_name, "Construction%", "%Engineering%"))
            else:
                cur.execute("""
                UPDATE reference.company_gics
                SET 
                    sector_id = %s,
                    industry_id = COALESCE(
                        (SELECT industry_id FROM reference.gics_industry WHERE industry_name = %s),
                        industry_id
                    )
                WHERE raw_industry LIKE %s
                """, (consumer_sector_id, industry_name, f"%{industry_name.split(' ')[0]}%"))
            
            update_count = cur.rowcount
            logger.info(f"{update_count}件の{industry_name}関連企業を更新しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Consumer Discretionary修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def fix_materials_sector():
    """
    Materialsセクターの問題を修正
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("Materialsセクターの修正を開始します...")
        
        # Materialsのセクターを確認
        cur.execute("""
        SELECT sector_id, sector_name
        FROM reference.gics_sector
        WHERE sector_name = 'Materials'
        """)
        materials_sector = cur.fetchone()
        materials_sector_id = materials_sector[0] if materials_sector else None
        
        if materials_sector_id is None:
            logger.error("Materialsセクターが見つかりません")
            return False
        
        logger.info(f"正しいMaterialsのセクターID: {materials_sector_id}")
        
        # raw_sector = 'Materials'のすべての企業を修正
        cur.execute("""
        UPDATE reference.company_gics
        SET sector_id = %s
        WHERE raw_sector = 'Materials'
        """, (materials_sector_id,))
        
        update_count = cur.rowcount
        logger.info(f"{update_count}件のMaterials企業のセクターIDを{materials_sector_id}に修正しました")
        
        # Materialsセクター内の特定の業種問題を修正
        problem_industries = {
            "Metals & Mining": "Metals & Mining",
            "Containers & Packaging": "Containers & Packaging",
            "Materials (Misc)": "Chemicals",
        }
        
        # 業種IDマッピングを取得
        cur.execute("""
        SELECT industry_name, industry_id
        FROM reference.gics_industry
        WHERE sector_id = %s
        """, (materials_sector_id,))
        materials_industries = {name: id for name, id in cur.fetchall()}
        
        # 特定の業種問題を修正
        for raw_industry, standard_industry in problem_industries.items():
            industry_id = materials_industries.get(standard_industry)
            if industry_id:
                cur.execute("""
                UPDATE reference.company_gics
                SET industry_id = %s
                WHERE sector_id = %s AND raw_industry = %s AND (industry_id = 47 OR industry_id IS NULL)
                """, (industry_id, materials_sector_id, raw_industry))
                
                cur_update_count = cur.rowcount
                if cur_update_count > 0:
                    logger.info(f"{cur_update_count}件の'{raw_industry}'を'{standard_industry}'(ID={industry_id})に修正しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Materials修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_industry_mappings() -> Dict[str, Dict[str, int]]:
    """
    セクター別の業種マッピングを取得
    
    戻り値: 
        {
            'セクター名': {
                '業種名': 業種ID,
                ...
            },
            ...
        }
    """
    industry_mappings = {}
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # セクターとその下の業種IDを取得
            cur.execute("""
            SELECT s.sector_name, i.industry_name, i.industry_id
            FROM reference.gics_sector s
            JOIN reference.gics_industry i ON s.sector_id = i.sector_id
            ORDER BY s.sector_name, i.industry_name
            """)
            
            for sector_name, industry_name, industry_id in cur.fetchall():
                if sector_name not in industry_mappings:
                    industry_mappings[sector_name] = {}
                industry_mappings[sector_name][industry_name] = industry_id
    
    return industry_mappings

def fix_industry_mappings():
    """
    業種マッピングの問題を修正
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # セクター別の業種マッピングを取得
        industry_mappings = get_industry_mappings()
        
        logger.info("業種マッピングの修正を開始します...")
        
        # 追加の手動マッピング（特定の問題を修正するため）
        manual_mappings = {
            # Communication Services
            "Diversified Telecommunication Services": ("Diversified Telecommunication Services", "Communication Services"),
            
            # Materials
            "Metals & Mining": ("Metals & Mining", "Materials"),
            "Containers & Packaging": ("Containers & Packaging", "Materials"),
            "Materials (Misc)": ("Chemicals", "Materials"),
            
            # Consumer Staples
            "Food & Staples Retailing": ("Food & Staples Retailing", "Consumer Staples"),
            
            # Financials
            "Consumer Finance": ("Consumer Finance", "Financials"),
            "Insurance Brokers": ("Insurance", "Financials"),
            "Life & Health Insurance": ("Insurance", "Financials"),
            "Reinsurance": ("Insurance", "Financials"),
            "Property & Casualty Insurance": ("Insurance", "Financials"),
            "Multi-line Insurance": ("Insurance", "Financials"),
            "Financial Exchanges & Data": ("Financial Exchanges & Data", "Financials"),
            
            # Utilities
            "Independent Power & Renewable Electricity Producers": ("Independent Power & Renewable Electricity Producers", "Utilities"),
        }
        
        # 直接マッピング
        updated_count = 0
        for raw_industry, (industry_name, sector_name) in manual_mappings.items():
            if sector_name in industry_mappings and industry_name in industry_mappings[sector_name]:
                industry_id = industry_mappings[sector_name][industry_name]
                
                cur.execute("""
                UPDATE reference.company_gics
                SET industry_id = %s
                WHERE raw_industry = %s AND industry_id = 47
                """, (industry_id, raw_industry))
                
                cur_update_count = cur.rowcount
                if cur_update_count > 0:
                    logger.info(f"{cur_update_count}件のraw_industry='{raw_industry}'企業の業種IDを{industry_id}({industry_name})に修正しました")
                    updated_count += cur_update_count
        
        # パターンマッチングによる修正
        pattern_mappings = [
            (r"Diversified.*Telecom", "Diversified Telecommunication Services", "Communication Services"),
            (r"Metal.*Mining", "Metals & Mining", "Materials"),
            (r"Container.*Package", "Containers & Packaging", "Materials"),
            (r"Material.*Misc", "Chemicals", "Materials"),
            (r"Food.*Retail", "Food & Staples Retailing", "Consumer Staples"),
            (r"Consumer.*Finance", "Consumer Finance", "Financials"),
            (r"Insurance.*Broker", "Insurance", "Financials"),
            (r"Life.*Health.*Insurance", "Insurance", "Financials"),
            (r"Reinsurance", "Insurance", "Financials"),
            (r"Property.*Casualty.*Insurance", "Insurance", "Financials"),
            (r"Multi.*line.*Insurance", "Insurance", "Financials"),
            (r"Financial.*Exchange", "Financial Exchanges & Data", "Financials"),
            (r"Independent.*Power|Renewable.*Electric", "Independent Power & Renewable Electricity Producers", "Utilities"),
        ]
        
        # raw_industryを取得
        cur.execute("""
        SELECT raw_industry, COUNT(*)
        FROM reference.company_gics
        WHERE industry_id = 47
        GROUP BY raw_industry
        """)
        
        raw_industries = cur.fetchall()
        
        for pattern_str, industry_name, sector_name in pattern_mappings:
            if sector_name in industry_mappings and industry_name in industry_mappings[sector_name]:
                industry_id = industry_mappings[sector_name][industry_name]
                pattern = re.compile(pattern_str, re.IGNORECASE)
                
                for raw_industry, _ in raw_industries:
                    if pattern.search(raw_industry):
                        cur.execute("""
                        UPDATE reference.company_gics
                        SET industry_id = %s
                        WHERE raw_industry = %s AND industry_id = 47
                        """, (industry_id, raw_industry))
                        
                        cur_update_count = cur.rowcount
                        if cur_update_count > 0:
                            logger.info(f"{cur_update_count}件のraw_industry='{raw_industry}'企業の業種IDを{industry_id}({industry_name})に修正しました（パターンマッチ）")
                            updated_count += cur_update_count
        
        logger.info(f"合計{updated_count}件の企業の業種IDを修正しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"業種マッピング修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def fix_raw_sector_mappings():
    """
    raw_sectorに基づいてsector_idを修正する
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        logger.info("raw_sectorに基づくセクターID修正を開始します...")
        
        # セクター名とIDのマッピングを取得
        cur.execute("""
        SELECT sector_name, sector_id
        FROM reference.gics_sector
        WHERE sector_name != 'Unclassified'
        """)
        sector_mapping = {row[0]: row[1] for row in cur.fetchall()}
        
        # raw_sectorと一致しないsector_idを調査
        for sector_name, sector_id in sector_mapping.items():
            # SECTOR_MAPから逆引き用のマッピングを作成
            raw_sector_variants = [k for k, v in industry_mapper.SECTOR_MAP.items() if v == sector_name]
            raw_sector_variants.append(sector_name)  # 標準名も含める
            
            for raw_sector in raw_sector_variants:
                # raw_sectorが一致するのにsector_idが異なる企業を修正
                cur.execute("""
                UPDATE reference.company_gics
                SET sector_id = %s
                WHERE raw_sector = %s AND sector_id != %s
                """, (sector_id, raw_sector, sector_id))
                
                update_count = cur.rowcount
                if update_count > 0:
                    logger.info(f"{update_count}件のraw_sector='{raw_sector}'企業のセクターIDを{sector_id}({sector_name})に修正しました")
        
        # 変更をコミット
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"raw_sectorマッピング修正中にエラーが発生: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='GICSセクター・業種マッピングの一括修正')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細なログを出力する')
    parser.add_argument('--check-only', '-c', action='store_true', help='現状の確認のみを行い、修正は行わない')
    parser.add_argument('--log-file', type=str, help='ログファイルのパス')
    parser.add_argument('--skip-active-update', action='store_true', help='アクティブステータス更新をスキップ')
    parser.add_argument('--skip-sector-repair', action='store_true', help='セクターID修復をスキップ')
    parser.add_argument('--skip-energy', action='store_true', help='Energyセクター修正をスキップ')
    parser.add_argument('--skip-consumer', action='store_true', help='Consumer Discretionary修正をスキップ')
    parser.add_argument('--skip-materials', action='store_true', help='Materials修正をスキップ')
    parser.add_argument('--skip-industry', action='store_true', help='業種マッピング修正をスキップ')
    parser.add_argument('--skip-raw-sector', action='store_true', help='raw_sectorマッピング修正をスキップ')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 追加のログファイルが指定された場合
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    # 確認のみモードの場合は終了
    if args.check_only:
        logger.info("確認のみのモードで実行しました。修正は行いません。")
        return
    
    # 各ステップを実行
    steps = [
        ("アクティブステータス更新", update_gics_active_status, args.skip_active_update),
        ("セクターID修復", repair_gics_sector_mapping, args.skip_sector_repair),
        ("Energyセクター修正", fix_energy_industry_mapping, args.skip_energy),
        ("raw_sectorマッピング修正", fix_raw_sector_mappings, args.skip_raw_sector),
        ("Consumer Discretionary修正", fix_consumer_discretionary, args.skip_consumer),
        ("Materials修正", fix_materials_sector, args.skip_materials),
        ("業種マッピング修正", fix_industry_mappings, args.skip_industry),
    ]
    
    success = True
    for step_name, step_func, skip in steps:
        if skip:
            logger.info(f"{step_name}はスキップします")
            continue
            
        logger.info(f"=== {step_name}を開始します ===")
        step_success = step_func()
        logger.info(f"=== {step_name}を終了しました (成功: {step_success}) ===\n")
        
        if not step_success:
            success = False
    
    if success:
        logger.info("GICSセクター・業種マッピングの一括修正が完了しました")
    else:
        logger.error("GICSセクター・業種マッピングの一括修正中にエラーが発生しました")
        sys.exit(1)

if __name__ == "__main__":
    main() 