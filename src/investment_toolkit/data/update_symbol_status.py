#!/usr/bin/env python
"""
シンボルステータスの更新を実行するスクリプト。

シンボルリストを取得し、現在上場中かどうかを判定して
fmp_data.symbol_statusテーブルを作成・更新します。
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# スクリプトのあるディレクトリの親ディレクトリをパスに追加
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# ロギングを設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'log', 'symbol_status_update.log'))
    ]
)

logger = logging.getLogger(__name__)

# symbol_status_managerをインポート
from investment_analysis.data.symbol_status_manager import update_symbol_status

def main():
    parser = argparse.ArgumentParser(description='シンボルステータスの更新を実行')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細なログを出力する')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger('src.data.symbol_status_manager').setLevel(logging.DEBUG)
    
    logger.info("シンボルステータスの更新を開始します...")
    
    result = update_symbol_status()
    
    if result:
        logger.info("シンボルステータスの更新が完了しました。")
    else:
        logger.error("シンボルステータスの更新に失敗しました。")
        sys.exit(1)

if __name__ == "__main__":
    main() 