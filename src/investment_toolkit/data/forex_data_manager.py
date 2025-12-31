import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from .fred_api import FREDAPI


class ForexDataManager:
    """
    FRED APIから為替データを取得・管理するクラス

    主な機能:
    - FRED APIから直接ペア（USDJPY等）の為替データを取得
    - クロスレート（EURJPY, GBPJPY）の計算
    - fred_data.forexテーブルへのデータ保存
    - FMPデータからFREDへの一括移行
    """

    # FRED為替シリーズマッピング
    FRED_SERIES_MAPPING = {
        'USDJPY': 'DEXJPUS',
        'EURUSD': 'DEXUSEU',
        'GBPUSD': 'DEXUSUK',
        'USDCAD': 'DEXCAUS',
        'AUDUSD': 'DEXUSAL',
        'USDCHF': 'DEXSZUS',
        'EURJPY': {
            'base_pairs': ['DEXUSEU', 'DEXJPUS'],
            'operation': 'multiply'
        },
        'GBPJPY': {
            'base_pairs': ['DEXUSUK', 'DEXJPUS'],
            'operation': 'multiply'
        }
    }

    def __init__(self, db_conn_string=None):
        """初期化"""
        if db_conn_string is None:
            db_conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

        self.db_engine = create_engine(db_conn_string)
        self.api = FREDAPI()
        self.logger = logging.getLogger(__name__)

    def ensure_forex_schema_exists(self):
        """fred_data.forexテーブルの存在確認・作成"""
        try:
            with self.db_engine.begin() as conn:
                # fred_dataスキーマ作成
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS fred_data"))

                # forex テーブル作成
                create_table_query = text("""
                CREATE TABLE IF NOT EXISTS fred_data.forex (
                    symbol VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    volume BIGINT,
                    source VARCHAR(20) DEFAULT 'FRED',
                    PRIMARY KEY (symbol, date),
                    CONSTRAINT forex_valid_symbol CHECK (
                        symbol IN ('USDJPY','EURUSD','GBPUSD','USDCAD',
                                   'AUDUSD','USDCHF','EURJPY','GBPJPY')
                    )
                )
                """)
                conn.execute(create_table_query)

            self.logger.info("fred_data.forexテーブルの確認・作成が完了しました")
            return True

        except Exception as e:
            self.logger.error(f"テーブル作成エラー: {e}", exc_info=True)
            raise

    @staticmethod
    def get_fred_series_mapping():
        """為替ペアとFREDシリーズIDのマッピングを返す"""
        return ForexDataManager.FRED_SERIES_MAPPING.copy()

    def fetch_direct_pair(self, pair_symbol, observation_start=None, observation_end=None):
        """
        直接ペアのデータをFRED APIから取得

        パラメータ:
            pair_symbol: 為替ペア（例: 'USDJPY'）
            observation_start: 開始日（YYYY-MM-DD形式）
            observation_end: 終了日（YYYY-MM-DD形式）

        戻り値:
            DataFrame: columns = [symbol, date, price, volume]
        """
        try:
            # FREDシリーズIDを取得
            series_id = self.FRED_SERIES_MAPPING.get(pair_symbol)

            if not series_id or isinstance(series_id, dict):
                self.logger.error(f"{pair_symbol}は直接ペアではありません")
                return pd.DataFrame()

            self.logger.info(f"{pair_symbol}のデータを取得中（FRED series: {series_id}）")

            # FRED APIからデータ取得
            df = self.api.get_series_data(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end
            )

            if df.empty:
                self.logger.warning(f"{pair_symbol}のデータが空です")
                return df

            # カラムリネーム: indicator_name → symbol, value → price
            df = df.rename(columns={
                'indicator_name': 'symbol',
                'value': 'price'
            })

            # symbolをペア名に置き換え
            df['symbol'] = pair_symbol

            # volumeカラム追加（FREDにはないのでNULL）
            df['volume'] = None

            # 必要なカラムのみ選択
            df = df[['symbol', 'date', 'price', 'volume']]

            self.logger.info(f"{pair_symbol}: {len(df)}件のデータを取得しました")
            return df

        except Exception as e:
            self.logger.error(f"{pair_symbol}の取得エラー: {e}", exc_info=True)
            return pd.DataFrame()

    def calculate_cross_rate(self, pair_symbol, observation_start=None, observation_end=None):
        """
        クロスレート（EURJPY, GBPJPY）を計算

        パラメータ:
            pair_symbol: クロスレートペア（'EURJPY' or 'GBPJPY'）
            observation_start: 開始日
            observation_end: 終了日

        戻り値:
            DataFrame: columns = [symbol, date, price, volume]
        """
        try:
            # クロスレート設定を取得
            cross_config = self.FRED_SERIES_MAPPING.get(pair_symbol)

            if not isinstance(cross_config, dict):
                self.logger.error(f"{pair_symbol}はクロスレートではありません")
                return pd.DataFrame()

            base_pairs = cross_config['base_pairs']
            operation = cross_config['operation']

            self.logger.info(f"{pair_symbol}を計算中（{base_pairs[0]} {operation} {base_pairs[1]}）")

            # 2つのベースシリーズを取得
            df1 = self.api.get_series_data(base_pairs[0], observation_start, observation_end)
            df2 = self.api.get_series_data(base_pairs[1], observation_start, observation_end)

            if df1.empty or df2.empty:
                self.logger.warning(f"{pair_symbol}のベースデータが取得できませんでした")
                return pd.DataFrame()

            # DataFrameをdateでマージ（inner join: 両方存在する日付のみ）
            df1 = df1.rename(columns={'value': 'value1'})
            df2 = df2.rename(columns={'value': 'value2'})

            df_merged = pd.merge(
                df1[['date', 'value1']],
                df2[['date', 'value2']],
                on='date',
                how='inner'
            )

            # 日付不一致の警告
            total_dates = len(set(df1['date'].tolist()) | set(df2['date'].tolist()))
            matched_dates = len(df_merged)
            if total_dates > matched_dates:
                missing_count = total_dates - matched_dates
                self.logger.warning(f"{pair_symbol}: {missing_count}日分のデータが片方にしか存在しません")

            if df_merged.empty:
                self.logger.warning(f"{pair_symbol}: 共通する日付が見つかりませんでした")
                return pd.DataFrame()

            # クロスレート計算
            if operation == 'multiply':
                df_merged['price'] = df_merged['value1'] * df_merged['value2']
            else:
                self.logger.error(f"未対応の演算: {operation}")
                return pd.DataFrame()

            # 最終DataFrame作成
            df_result = pd.DataFrame({
                'symbol': pair_symbol,
                'date': df_merged['date'],
                'price': df_merged['price'],
                'volume': None
            })

            self.logger.info(f"{pair_symbol}: {len(df_result)}件のクロスレートを計算しました")
            return df_result

        except Exception as e:
            self.logger.error(f"{pair_symbol}の計算エラー: {e}", exc_info=True)
            return pd.DataFrame()

    def _get_latest_date(self, pair_symbol):
        """データベースから指定ペアの最新日付を取得"""
        try:
            query = text("""
                SELECT MAX(date) as latest_date
                FROM fred_data.forex
                WHERE symbol = :symbol
            """)

            with self.db_engine.connect() as conn:
                result = conn.execute(query, {"symbol": pair_symbol})
                row = result.fetchone()
                latest_date = row[0] if row and row[0] else None

                if latest_date:
                    return latest_date.strftime('%Y-%m-%d')
                return None

        except Exception as e:
            self.logger.error(f"{pair_symbol}の最新日付取得エラー: {e}")
            return None

    def update_forex_pair(self, pair_symbol, force_full=False):
        """
        1つの為替ペアを更新

        パラメータ:
            pair_symbol: 為替ペア名
            force_full: True=全期間取得、False=増分更新

        戻り値:
            bool: 成功ならTrue
        """
        try:
            self.logger.info(f"=== {pair_symbol}の更新開始 ===")

            # 直接ペアかクロスレートか判定
            series_info = self.FRED_SERIES_MAPPING.get(pair_symbol)
            is_cross_rate = isinstance(series_info, dict)

            # 最新日取得
            latest_date = self._get_latest_date(pair_symbol)

            # 開始日決定
            if force_full or latest_date is None:
                observation_start = '2010-01-01'
                self.logger.info(f"{pair_symbol}: 全期間取得（2010-01-01〜）")
            else:
                # 最新日の翌日から取得
                next_day = pd.to_datetime(latest_date) + timedelta(days=1)
                observation_start = next_day.strftime('%Y-%m-%d')
                self.logger.info(f"{pair_symbol}: 増分更新（{observation_start}〜）")

            # 終了日
            observation_end = datetime.now().strftime('%Y-%m-%d')

            # データ取得
            if is_cross_rate:
                df = self.calculate_cross_rate(pair_symbol, observation_start, observation_end)
            else:
                df = self.fetch_direct_pair(pair_symbol, observation_start, observation_end)

            if df.empty:
                self.logger.info(f"{pair_symbol}: 新規データなし")
                return True  # データがないのは正常（週末等）

            # データベース保存
            success = self.save_to_database(df)

            if success:
                self.logger.info(f"{pair_symbol}: 更新成功（{len(df)}件）")
            else:
                self.logger.warning(f"{pair_symbol}: 更新失敗")

            return success

        except Exception as e:
            self.logger.error(f"{pair_symbol}の更新エラー: {e}", exc_info=True)
            return False

    def update_all_forex(self, pairs=None):
        """
        全為替ペアを更新

        パラメータ:
            pairs: 更新するペアのリスト（Noneなら全ペア）

        戻り値:
            dict: {pair_symbol: success_bool}
        """
        if pairs is None:
            pairs = list(self.FRED_SERIES_MAPPING.keys())

        # 直接ペアとクロスレートに分離
        direct_pairs = [p for p in pairs
                        if not isinstance(self.FRED_SERIES_MAPPING.get(p), dict)]
        cross_pairs = [p for p in pairs
                       if isinstance(self.FRED_SERIES_MAPPING.get(p), dict)]

        results = {}

        self.logger.info(f"為替データ更新開始: 直接{len(direct_pairs)}ペア、クロス{len(cross_pairs)}ペア")

        # 直接ペアを先に更新（クロスレートの依存関係のため）
        for pair in direct_pairs:
            results[pair] = self.update_forex_pair(pair)

        # クロスペアを後で更新
        for pair in cross_pairs:
            results[pair] = self.update_forex_pair(pair)

        success_count = sum(1 for v in results.values() if v)
        self.logger.info(f"為替データ更新完了: {success_count}/{len(pairs)}ペア成功")

        return results

    def save_to_database(self, df, table_name='forex', schema='fred_data'):
        """
        DataFrameをfred_data.forexテーブルに保存

        パラメータ:
            df: 保存するDataFrame
            table_name: テーブル名（デフォルト: forex）
            schema: スキーマ名（デフォルト: fred_data）

        戻り値:
            bool: 成功ならTrue
        """
        if df.empty:
            self.logger.warning(f"空のDataFrameのため、{schema}.{table_name}への保存をスキップします")
            return False

        try:
            # 日付型の処理
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date

            # NULL価格の行を削除
            if 'price' in df.columns and df['price'].isnull().any():
                null_count = df['price'].isnull().sum()
                self.logger.warning(f"NULL価格が{null_count}件含まれています")
                df = df.dropna(subset=['price'])
                self.logger.info(f"NULL価格を含む{null_count}行を削除しました")

            if df.empty:
                self.logger.warning("データ検証後にDataFrameが空になりました")
                return False

            # トランザクション内で削除と挿入を実行（UPSERT）
            if 'symbol' in df.columns and 'date' in df.columns:
                unique_keys = df[['symbol', 'date']].drop_duplicates()

                self.logger.debug(f"トランザクション開始: {schema}.{table_name}のデータ更新")
                with self.db_engine.begin() as conn:
                    # 既存データの削除
                    for _, row in unique_keys.iterrows():
                        symbol = row['symbol']
                        date_val = row['date']

                        # 日付を文字列に変換
                        date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)

                        # 削除クエリを実行
                        delete_query = text(f"""
                        DELETE FROM {schema}.{table_name}
                        WHERE symbol = :symbol AND date = :date
                        """)

                        conn.execute(delete_query, {"symbol": symbol, "date": date_str})

                    # データを直接挿入
                    df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False)

                self.logger.info(f"トランザクション完了: {len(df)}行のデータを{schema}.{table_name}に保存しました")
                return True
            else:
                # 標準的な保存方法（symbolまたはdateカラムがない場合）
                df.to_sql(table_name, self.db_engine, schema=schema, if_exists='append', index=False)
                self.logger.info(f"{len(df)}行のデータを{schema}.{table_name}に保存しました")
                return True

        except Exception as e:
            self.logger.error(f"データベースへの保存中にエラーが発生しました: {str(e)}", exc_info=True)
            return False

    def migrate_from_fmp(self, verify_only=False, drop_source=False):
        """
        fmp_data.forexからfred_data.forexへデータを移行

        パラメータ:
            verify_only: True=検証のみ（移行実行しない）
            drop_source: True=移行後にfmp_data.forexを削除

        戻り値:
            dict: 移行レポート
        """
        try:
            self.logger.info("=== FMP→FRED移行開始 ===")

            # fred_data.forexテーブル確保
            self.ensure_forex_schema_exists()

            # fmp_data.forexから全データ取得
            query = text("""
                SELECT symbol, date, price, volume
                FROM fmp_data.forex
                ORDER BY symbol, date
            """)

            with self.db_engine.connect() as conn:
                df_source = pd.read_sql(query, conn)

            self.logger.info(f"ソースデータ: {len(df_source)}行")

            if df_source.empty:
                self.logger.warning("fmp_data.forexにデータがありません")
                return {
                    'source_records': 0,
                    'migrated_records': 0,
                    'date_range': None,
                    'pairs': [],
                    'verification': {'match': False, 'details': 'No source data'}
                }

            # sourceカラム追加
            df_source['source'] = 'FMP'

            # 日付範囲取得
            date_range = {
                'start': df_source['date'].min(),
                'end': df_source['date'].max()
            }

            # ペアリスト取得
            pairs = df_source['symbol'].unique().tolist()

            # ペア別レコード数（ソース）
            source_counts = df_source.groupby('symbol').size().to_dict()

            # 検証のみモード
            if verify_only:
                self.logger.info("検証モード: 移行は実行されません")
                return {
                    'source_records': len(df_source),
                    'migrated_records': 0,
                    'date_range': date_range,
                    'pairs': pairs,
                    'source_counts': source_counts,
                    'verification': {'match': None, 'details': 'Verify-only mode'}
                }

            # fred_data.forexへ挿入
            self.logger.info(f"fred_data.forexへ{len(df_source)}行を移行中...")
            success = self.save_to_database(df_source)

            if not success:
                self.logger.error("移行に失敗しました")
                return {
                    'source_records': len(df_source),
                    'migrated_records': 0,
                    'date_range': date_range,
                    'pairs': pairs,
                    'verification': {'match': False, 'details': 'Migration failed'}
                }

            # 検証: レコード数比較
            query_verify = text("""
                SELECT symbol, COUNT(*) as count
                FROM fred_data.forex
                WHERE symbol = ANY(:symbols)
                GROUP BY symbol
            """)

            with self.db_engine.connect() as conn:
                df_verify = pd.read_sql(query_verify, conn, params={"symbols": pairs})

            target_counts = df_verify.set_index('symbol')['count'].to_dict()

            # 検証結果
            verification = {'match': True, 'details': {}}
            for pair in pairs:
                source_count = source_counts.get(pair, 0)
                target_count = target_counts.get(pair, 0)

                if source_count != target_count:
                    verification['match'] = False
                    verification['details'][pair] = {
                        'source': source_count,
                        'target': target_count,
                        'diff': target_count - source_count
                    }
                else:
                    verification['details'][pair] = {'match': True, 'count': source_count}

            if verification['match']:
                self.logger.info("検証成功: すべてのペアでレコード数が一致しました")
            else:
                self.logger.warning(f"検証失敗: レコード数不一致 - {verification['details']}")

            # ソーステーブル削除（オプション）
            if drop_source and verification['match']:
                self.logger.warning("fmp_data.forexテーブルを削除します")
                with self.db_engine.begin() as conn:
                    conn.execute(text("DROP TABLE fmp_data.forex"))
                self.logger.info("fmp_data.forexを削除しました")

            return {
                'source_records': len(df_source),
                'migrated_records': len(df_source),
                'date_range': date_range,
                'pairs': pairs,
                'source_counts': source_counts,
                'target_counts': target_counts,
                'verification': verification
            }

        except Exception as e:
            self.logger.error(f"移行エラー: {e}", exc_info=True)
            raise
