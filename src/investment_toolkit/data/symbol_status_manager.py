'''
直接実行時：update_symbol_status()を実行しsymbol_statusテーブルを更新。しかしsymbols.txtなどテキストファイルは生成しない
月次更新時のみテキスト生成
'''

import requests
import pandas as pd
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
import os
import sys
import json
import re
from pathlib import Path

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 設定を読み込む
from investment_toolkit.utilities.config import (
    FMP_API_KEY_PRIMARY,
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME
)

# ロガーの設定
logger = logging.getLogger('src.data.symbol_status_manager')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# SQLAlchemyの設定
Base = declarative_base()
db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(db_url)
metadata = MetaData(schema='fmp_data')

# symbol_statusテーブルの定義
class SymbolStatus(Base):
    __tablename__ = 'symbol_status'
    __table_args__ = {'schema': 'fmp_data'}
    
    symbol = Column(String, primary_key=True)
    name = Column(String)
    exchange = Column(String)
    type = Column(String)
    is_active = Column(Boolean, default=True)
    manually_deactivated = Column(Boolean, default=False)
    last_updated = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<SymbolStatus(symbol='{self.symbol}', name='{self.name}', is_active={self.is_active})>"

def create_schema_if_not_exists():
    """fmp_dataスキーマが存在しない場合は作成する"""
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS fmp_data"))
        conn.commit()

def create_symbol_status_table():
    """symbol_statusテーブルを作成する"""
    create_schema_if_not_exists()
    Base.metadata.create_all(engine)
    logger.info("symbol_statusテーブルを作成しました")

def get_active_symbols_from_api():
    """Financial Modeling Prepから上場中のシンボル一覧を取得する"""
    url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY_PRIMARY}"
    logger.info("APIから上場中シンボルの一覧を取得しています...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"{len(data)}件のシンボル情報を取得しました")
            return data
        else:
            logger.error(f"APIエラー: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"APIリクエスト中にエラーが発生しました: {e}")
        return []

def get_existing_symbols_from_db():
    """データベースから既存の全シンボルを取得する"""
    try:
        with engine.connect() as conn:
            # 既存のシンボルをすべてのテーブルから収集
            tables = [
                'income_statements', 
                'balance_sheets', 
                'cash_flows', 
                'daily_prices', 
                'shares', 
                'company_profile'
            ]
            
            all_symbols = set()
            
            for table in tables:
                try:
                    query = text(f"SELECT DISTINCT symbol FROM fmp_data.{table}")
                    result = conn.execute(query)
                    symbols = {row[0] for row in result}
                    logger.info(f"{table}テーブルから{len(symbols)}件のシンボルを取得しました")
                    all_symbols.update(symbols)
                except Exception as table_error:
                    logger.warning(f"{table}テーブルの読み取り中にエラーが発生しました: {table_error}")
            
            logger.info(f"データベースから合計{len(all_symbols)}件のユニークなシンボルを取得しました")
            return all_symbols
    
    except Exception as e:
        logger.error(f"データベースからのシンボル取得中にエラーが発生しました: {e}")
        return set()

def load_markets_config(config_file=None):
    """市場設定ファイルを読み込む"""
    if config_file is None:
        # デフォルトはプロジェクトルートのconfig/markets.json
        config_file = os.path.join(
            Path(__file__).resolve().parent.parent.parent,
            'config',
            'markets.json'
        )
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"市場設定ファイルの読み込みエラー: {e}")
        # 最小限の設定を返す
        return {
            "markets": [
                {
                    "id": "jp",
                    "name": "日本",
                    "exchanges": ["Tokyo"],
                    "symbol_pattern": "\\.T$",
                    "enabled": True
                },
                {
                    "id": "us",
                    "name": "米国",
                    "exchanges": ["NYSE", "NASDAQ", "AMEX"],
                    "symbol_pattern": "^[A-Z]+$",
                    "enabled": True
                }
            ]
        }

def update_symbol_status():
    """シンボルのステータステーブルを更新する"""
    # テーブルが存在しない場合は作成
    create_symbol_status_table()
    
    # APIから現在上場中のシンボルを取得
    active_symbols_data = get_active_symbols_from_api()
    active_symbols = {item['symbol'] for item in active_symbols_data}
    
    # DBから既存のシンボルを取得
    existing_symbols = get_existing_symbols_from_db()
    
    # 日足データの取得状況を確認
    logger.info("既存銘柄の日足データ取得状況を確認しています...")
    symbols_with_data_status = check_daily_data_availability(list(existing_symbols))
    
    # シンボルのステータスを更新
    symbols_to_update = []
    
    # 既存のシンボルをすべて処理
    for symbol in existing_symbols:
        # 現在のデータベースの状態を確認
        current_status = get_current_symbol_status(symbol)
        manually_deactivated = current_status.get('manually_deactivated', False)
        
        is_active_in_api = symbol in active_symbols
        symbol_data = next((item for item in active_symbols_data if item['symbol'] == symbol), None)
        
        # 日足データの取得状況
        has_recent_daily_data = symbols_with_data_status.get(symbol, {}).get('has_recent_data', True)
        
        # 手動で無効化されていた場合は、APIの状態に関わらず非アクティブのままにする
        # APIにアクティブで存在し、かつ最近の日足データがある場合のみアクティブとする
        is_active = is_active_in_api and not manually_deactivated and has_recent_daily_data
        
        if symbol_data:
            # APIに存在する場合はデータを取得
            symbols_to_update.append({
                'symbol': symbol,
                'name': symbol_data.get('name', ''),
                'exchange': symbol_data.get('exchange', ''),
                'type': symbol_data.get('type', ''),
                'is_active': is_active,
                'manually_deactivated': manually_deactivated,
                'last_updated': datetime.now()
            })
        else:
            # APIに存在しない場合は廃止として登録
            # manually_deactivatedはそのまま保持
            symbols_to_update.append({
                'symbol': symbol,
                'name': '',
                'exchange': '',
                'type': '',
                'is_active': False,
                'manually_deactivated': manually_deactivated,
                'last_updated': datetime.now()
            })
    
    # APIに存在するが、既存テーブルには登録されていないシンボルを追加
    for symbol_data in active_symbols_data:
        symbol = symbol_data['symbol']
        if symbol not in existing_symbols:
            symbols_to_update.append({
                'symbol': symbol,
                'name': symbol_data.get('name', ''),
                'exchange': symbol_data.get('exchange', ''),
                'type': symbol_data.get('type', ''),
                'is_active': True,
                'manually_deactivated': False,
                'last_updated': datetime.now()
            })
    
    # 更新データをPandasデータフレームに変換
    df = pd.DataFrame(symbols_to_update)
    
    # 重複を削除（symbolをキーとして）
    df = df.drop_duplicates(subset=['symbol'])
    
    try:
        # データベースに一括更新
        # 既存のレコードを削除して、新しいデータを追加する
        # autocommit=Trueでエンジンを作成し直す
        engine_autocommit = create_engine(db_url, isolation_level="AUTOCOMMIT")
        with engine_autocommit.connect() as conn:
            # 既存のsymbol_statusテーブルが存在するか確認
            check_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'fmp_data' 
                    AND table_name = 'symbol_status'
                )
            """)
            table_exists = conn.execute(check_query).scalar()
            
            if table_exists:
                # 既存のシンボルステータスを削除
                delete_query = text("DELETE FROM fmp_data.symbol_status")
                conn.execute(delete_query)
                
                # 直接SQLを使って挿入する方法に変更
                # メモリに入るサイズのバッチで処理
                batch_size = 1000
                total_inserted = 0
                
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    
                    # バッチ内の各行を挿入
                    for _, row in batch_df.iterrows():
                        insert_query = text("""
                            INSERT INTO fmp_data.symbol_status 
                            (symbol, name, exchange, type, is_active, manually_deactivated, last_updated)
                            VALUES (:symbol, :name, :exchange, :type, :is_active, :manually_deactivated, :last_updated)
                        """)
                        
                        conn.execute(insert_query, {
                            'symbol': row['symbol'],
                            'name': row['name'],
                            'exchange': row['exchange'],
                            'type': row['type'],
                            'is_active': row['is_active'],
                            'manually_deactivated': row['manually_deactivated'],
                            'last_updated': row['last_updated']
                        })
                        
                        total_inserted += 1
                
                logger.info(f"{total_inserted}件のシンボルステータスをデータベースに更新しました")
            else:
                # テーブルが存在しない場合は新規作成（SQLAlchemyのDDLを使用）
                create_symbol_status_table()
                
                # 一行ずつ挿入
                inserted = 0
                for _, row in df.iterrows():
                    insert_query = text("""
                        INSERT INTO fmp_data.symbol_status 
                        (symbol, name, exchange, type, is_active, manually_deactivated, last_updated)
                        VALUES (:symbol, :name, :exchange, :type, :is_active, :manually_deactivated, :last_updated)
                    """)
                    
                    conn.execute(insert_query, {
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'exchange': row['exchange'],
                        'type': row['type'],
                        'is_active': row['is_active'],
                        'manually_deactivated': row['manually_deactivated'],
                        'last_updated': row['last_updated']
                    })
                    
                    inserted += 1
            
                logger.info(f"{inserted}件のシンボルステータスをデータベースに新規作成しました")
        
        # インデックスを作成
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_symbol_status_symbol 
                ON fmp_data.symbol_status (symbol)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_symbol_status_is_active 
                ON fmp_data.symbol_status (is_active)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_symbol_status_exchange 
                ON fmp_data.symbol_status (exchange)
            """))
            conn.commit()
        
        return True
    except Exception as e:
        logger.error(f"シンボルステータステーブル更新中にエラーが発生しました: {e}")
        import traceback
        logger.error(f"エラーの詳細: {traceback.format_exc()}")
        return False

def check_daily_data_availability(symbols, days_threshold=90):
    """
    銘柄の日足データの取得状況を確認する
    
    パラメータ:
        symbols (list): 確認する銘柄リスト
        days_threshold (int): データが存在しないと判断する日数のしきい値
        
    戻り値:
        dict: 銘柄ごとの日足データ状況 {symbol: {'has_recent_data': bool, 'latest_date': str}}
    """
    result = {}
    
    try:
        # 現在の日付を取得
        current_date = datetime.now().date()
        threshold_date = current_date - timedelta(days=days_threshold)
        
        # バッチで処理（大量の銘柄がある場合のパフォーマンス対策）
        batch_size = 500
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            
            # プレースホルダーを作成
            placeholders = ', '.join([f"'{symbol}'" for symbol in batch_symbols])
            
            # 各銘柄の最新日付を取得
            query = text(f"""
                SELECT symbol, MAX(date) as latest_date
                FROM fmp_data.daily_prices
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            """)
            
            with engine.connect() as conn:
                rows = conn.execute(query)
                
                for row in rows:
                    symbol = row[0]
                    latest_date = row[1]
                    
                    # 最新の日付が閾値以内かどうかを判定
                    has_recent_data = False
                    if latest_date:
                        has_recent_data = latest_date >= threshold_date
                    
                    result[symbol] = {
                        'has_recent_data': has_recent_data,
                        'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None
                    }
            
            # バッチ処理の進捗を記録
            logger.info(f"日足データ確認: {i+len(batch_symbols)}/{len(symbols)}銘柄を処理しました")
        
        # 結果集計
        total_checked = len(result)
        active_count = sum(1 for data in result.values() if data['has_recent_data'])
        inactive_count = total_checked - active_count
        
        logger.info(f"日足データ確認結果: 合計={total_checked}銘柄, アクティブ={active_count}銘柄, 非アクティブ={inactive_count}銘柄")
        
        # データが存在しない銘柄については結果に含めない
        for symbol in symbols:
            if symbol not in result:
                result[symbol] = {
                    'has_recent_data': False,
                    'latest_date': None
                }
                
        return result
        
    except Exception as e:
        logger.error(f"日足データの取得状況確認中にエラーが発生しました: {e}")
        import traceback
        logger.error(f"エラーの詳細: {traceback.format_exc()}")
        
        # エラーの場合はデフォルトですべてアクティブとして返す
        return {symbol: {'has_recent_data': True, 'latest_date': None} for symbol in symbols}

def get_filtered_symbols(exchanges=None, active_only=True, output_file=None, security_types=None):
    """
    条件を指定してシンボルをフィルタリングする
    
    パラメータ:
        exchanges (list): 対象の取引所リスト ('NASDAQ Global Select', 'New York Stock Exchange', 'Tokyo' など)
        active_only (bool): 上場中の銘柄のみを対象とするか
        output_file (str): 結果を出力するファイルパス (None の場合は出力しない)
        security_types (list): 対象の銘柄タイプリスト ('stock', 'etf', 'trust' など)
        
    戻り値:
        list: フィルタリングされたシンボルのリスト
    """
    try:
        # 基本的なクエリ
        query = """
            SELECT s.symbol, s.name, s.exchange, s.type, s.is_active
            FROM fmp_data.symbol_status s
            WHERE 1=1
        """
        
        params = {}
        
        # is_activeでフィルタリング
        if active_only:
            query += " AND s.is_active = :active"
            params['active'] = True
        
        # 取引所でフィルタリング
        if exchanges and len(exchanges) > 0:
            placeholders = []
            for i, exchange in enumerate(exchanges):
                param_name = f"exchange_{i}"
                placeholders.append(f":{param_name}")
                params[param_name] = exchange
            
            query += f" AND s.exchange IN ({', '.join(placeholders)})"
        
        # 銘柄タイプでフィルタリング
        if security_types and len(security_types) > 0:
            placeholders = []
            for i, sec_type in enumerate(security_types):
                param_name = f"type_{i}"
                placeholders.append(f":{param_name}")
                params[param_name] = sec_type
            
            query += f" AND s.type IN ({', '.join(placeholders)})"
        
        # クエリ実行
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            # 辞書リストに変換
            symbols_data = []
            for row in result:
                symbol_data = {
                    'symbol': row[0],
                    'name': row[1],
                    'exchange': row[2],
                    'type': row[3],
                    'is_active': row[4]
                }
                symbols_data.append(symbol_data)
            
        logger.info(f"条件に一致する銘柄を{len(symbols_data)}件取得しました")
        
        # 結果をデータフレームに変換
        df = pd.DataFrame(symbols_data)
        
        # ファイルに出力
        if output_file and not df.empty:
            # シンボルだけをファイルに出力
            symbol_list = df['symbol'].tolist()
            
            # プロジェクトのルートディレクトリを取得
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = project_root / output_file
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # ファイルに書き込み
            with open(output_path, 'w') as f:
                # ヘッダーコメントを追加
                f.write("# Filtered symbols - Generated by symbol_status_manager.py\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                types_str = "N/A" if not security_types else ", ".join(security_types)
                f.write(f"# Filters: active_only={active_only}, exchanges={exchanges}, security_types={types_str}\n#\n")
                
                # シンボルを1行ずつ書き込み
                for symbol in symbol_list:
                    f.write(f"{symbol}\n")
            
            logger.info(f"フィルタリングされたシンボルを {output_file} に出力しました（{len(symbol_list)}件）")
        
        return df['symbol'].tolist() if not df.empty else []
    
    except Exception as e:
        logger.error(f"シンボルフィルタリング中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def update_filtered_symbols_file(output_path="config/symbols.txt", exchanges=None, is_actively_trading=True, security_types=None):
    """
    指定された条件に基づいてフィルタリングされたシンボルリストを更新する
    
    パラメータ:
        output_path (str): 出力先ファイル名
        exchanges (list): 対象の取引所リスト
        is_actively_trading (bool): 上場中かどうか
        security_types (list): 対象の銘柄タイプリスト
        
    戻り値:
        bool: 更新成功ならTrue、失敗ならFalse
    """
    # デフォルトの取引所
    if exchanges is None:
        exchanges = ['NYSE', 'NASDAQ', 'AMEX', 'Tokyo']
    
    try:
        # フィルタリングされたシンボルを取得して出力
        filtered_symbols = get_filtered_symbols(
            exchanges=exchanges,
            active_only=is_actively_trading,
            output_file=output_path,
            security_types=security_types
        )
        
        return len(filtered_symbols) > 0
    
    except Exception as e:
        logger.error(f"銘柄リストファイル更新中にエラーが発生しました: {e}")
        return False

def update_filtered_symbols_by_market(market_id=None, config_file=None, security_types=None):
    """
    指定された市場IDに基づいて銘柄リストを更新する
    
    パラメータ:
        market_id (str): 市場ID（None の場合は全市場を更新）
        config_file (str): 市場設定ファイルのパス
        security_types (list): 対象の銘柄タイプリスト ('stock', 'etf', 'trust' など)
        
    戻り値:
        dict: 市場IDごとの更新結果 {market_id: True/False}
    """
    # 市場設定を読み込む
    markets_config = load_markets_config(config_file)
    
    # コマンドラインからsecurity_typesが指定されたかどうか
    command_line_types = security_types is not None
    
    if command_line_types:
        logger.info(f"コマンドラインで指定された銘柄タイプ {', '.join(security_types)} を使用します")
    else:
        logger.info(f"市場設定ファイルから銘柄タイプを読み込みます")
    
    results = {}
    markets_to_update = []
    
    # 更新対象の市場を決定
    if market_id:
        # 指定された市場IDのみ
        market = next((m for m in markets_config.get("markets", []) if m.get("id") == market_id), None)
        if market:
            markets_to_update.append(market)
        else:
            logger.error(f"指定された市場ID '{market_id}' が見つかりません")
            return {market_id: False}
    else:
        # 全市場
        markets_to_update = [m for m in markets_config.get("markets", []) if m.get("enabled", True)]
    
    # 各市場の銘柄リストを更新
    for market in markets_to_update:
        market_id = market.get("id")
        market_name = market.get("name", market_id)
        
        # コマンドライン引数で指定がなければ、市場設定から銘柄タイプを取得
        market_security_types = security_types
        if not command_line_types:
            market_security_types = market.get("security_types", ["stock"])
            logger.info(f"{market_name}市場の設定された銘柄タイプ: {', '.join(market_security_types)}")
        
        logger.info(f"{market_name}市場の銘柄リストを更新します（対象タイプ: {', '.join(market_security_types)}）")
        
        try:
            # 市場情報から設定を抽出
            exchanges = market.get("exchanges", [])
            symbols_file = market.get("symbols_file")
            
            # シンボルをフィルタリングしてファイルに出力
            filtered_symbols = get_filtered_symbols(
                exchanges=exchanges,
                active_only=True,
                output_file=symbols_file,
                security_types=market_security_types
            )
            
            success = len(filtered_symbols) > 0
            results[market_id] = success
            
            if success:
                logger.info(f"{market_name}市場の銘柄リスト更新が完了しました（{len(filtered_symbols)}銘柄）")
            else:
                logger.warning(f"{market_name}市場の銘柄リスト更新に失敗しました")
                
        except Exception as e:
            logger.error(f"{market_name}市場の銘柄リスト更新中にエラーが発生しました: {e}")
            results[market_id] = False
    
    # 各市場の統合リストも生成
    try:
        # 全市場のシンボルを統合したリストを作成
        all_symbols = []
        
        for market in markets_to_update:
            symbols_file = market.get("symbols_file")
            if symbols_file:
                try:
                    # 各市場のシンボルファイルを読み込み
                    project_root = Path(__file__).resolve().parent.parent.parent
                    full_path = project_root / symbols_file
                    
                    if os.path.exists(full_path):
                        with open(full_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    all_symbols.append(line)
                except Exception as read_error:
                    logger.error(f"シンボルファイル {symbols_file} の読み込みエラー: {read_error}")
        
        # 統合リストをファイルに出力
        if all_symbols:
            output_path = os.path.join(
                Path(__file__).resolve().parent.parent.parent,
                'config',
                'symbols.txt'
            )
            
            with open(output_path, 'w') as f:
                # ヘッダーコメントを追加
                f.write("# Filtered symbols - Generated by symbol_status_manager.py\n")
                f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Integrated from all market symbol files\n#\n")
                
                # シンボルを1行ずつ書き込み
                for symbol in sorted(set(all_symbols)):
                    f.write(f"{symbol}\n")
                    
            logger.info(f"統合シンボルリストを生成しました（{len(all_symbols)}銘柄）")
            
    except Exception as e:
        logger.error(f"統合シンボルリストの生成中にエラーが発生しました: {e}")
    
    return results

def get_current_symbol_status(symbol):
    """
    データベースから特定のシンボルの現在のステータス情報を取得する
    
    パラメータ:
        symbol (str): 対象のシンボル
        
    戻り値:
        dict: シンボルのステータス情報の辞書
            {
                'is_active': bool,
                'manually_deactivated': bool,
                'last_updated': datetime
            }
    """
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT is_active, manually_deactivated, last_updated
                FROM fmp_data.symbol_status
                WHERE symbol = :symbol
            """)
            
            result = conn.execute(query, {"symbol": symbol})
            row = result.fetchone()
            
            if row:
                return {
                    'is_active': row[0],
                    'manually_deactivated': row[1],
                    'last_updated': row[2]
                }
            else:
                return {
                    'is_active': True,
                    'manually_deactivated': False,
                    'last_updated': None
                }
    except Exception as e:
        logger.error(f"{symbol}のステータス情報取得中にエラーが発生: {e}")
        return {
            'is_active': True,
            'manually_deactivated': False,
            'last_updated': None
        }

if __name__ == "__main__":
    update_symbol_status() 