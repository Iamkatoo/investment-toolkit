import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from .fred_api import FREDAPI
import os

class FREDDataManager:
    """
    FRED APIから取得したデータをデータベースに保存・管理するクラス
    """
    
    # デフォルトの経済指標リスト
    DEFAULT_INDICATORS = [
        "FEDFUNDS", "DGS10", "BAA10Y", "TWEXBGSMTH", 
        "CPIAUCSL", "CPILEGSL", "CPILFESL", "PCEPI", 
        "GDP", "UNRATE"
    ]
    
    def __init__(self, db_conn_string=None):
        """初期化"""
        if db_conn_string is None:
            db_conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            
        self.db_engine = create_engine(db_conn_string)
        self.api = FREDAPI()
        self.logger = logging.getLogger(__name__)
        
    def save_to_database(self, df, table_name='economic_indicators', schema='fred_data', if_exists='append'):
        """DataFrameをデータベースに保存"""
        if df.empty:
            self.logger.warning(f"空のDataFrameのため、{schema}.{table_name}への保存をスキップします")
            return False
            
        try:
            # 日付型の処理
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
                
            # データの検証
            if 'frequency' in df.columns:
                # frequencyの値が適切か確認
                valid_frequencies = ['daily', 'monthly', 'quarterly', 'annual']
                invalid_freqs = df[~df['frequency'].isin(valid_frequencies)]['frequency'].unique()
                if len(invalid_freqs) > 0:
                    self.logger.warning(f"無効なfrequency値が含まれています: {', '.join(invalid_freqs)}")
                    # 無効な値を'daily'に置き換え
                    df.loc[~df['frequency'].isin(valid_frequencies), 'frequency'] = 'daily'
                    self.logger.info("無効なfrequency値を'daily'に置き換えました")
                
            # NaN値のチェックと処理
            if 'value' in df.columns and df['value'].isnull().any():
                null_count = df['value'].isnull().sum()
                self.logger.warning(f"NULL値が{null_count}件含まれています")
                # NULL値を含む行を削除
                df = df.dropna(subset=['value'])
                self.logger.info(f"NULL値を含む{null_count}行を削除しました")
                
            if df.empty:
                self.logger.warning("データの検証後にDataFrameが空になりました")
                return False
                
            # トランザクション内で削除と挿入を実行
            if 'indicator_name' in df.columns and 'date' in df.columns:
                # インジケータと日付のユニークな組み合わせを取得
                unique_keys = df[['indicator_name', 'date']].drop_duplicates()
                
                # トランザクションを開始
                self.logger.debug(f"トランザクション開始: {schema}.{table_name}のデータ更新")
                with self.db_engine.begin() as conn:
                    # 既存データの削除
                    for _, row in unique_keys.iterrows():
                        indicator = row['indicator_name']
                        date_val = row['date']
                        
                        # 日付を文字列に変換
                        date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)
                        
                        # 削除クエリを実行
                        delete_query = text(f"""
                        DELETE FROM {schema}.{table_name}
                        WHERE indicator_name = :indicator AND date = :date
                        """)
                        
                        conn.execute(delete_query, {"indicator": indicator, "date": date_str})
                
                    # データを直接挿入
                    df.to_sql(table_name, conn, schema=schema, if_exists='append', index=False)
                
                self.logger.info(f"トランザクション完了: {len(df)}行のデータを{schema}.{table_name}に保存しました")
                return True
            else:
                # 標準的な保存方法
                df.to_sql(table_name, self.db_engine, schema=schema, if_exists=if_exists, index=False)
                self.logger.info(f"{len(df)}行のデータを{schema}.{table_name}に保存しました")
                return True
                
        except Exception as e:
            self.logger.error(f"データベースへの保存中にエラーが発生しました: {str(e)}")
            return False
    
    def fetch_and_store_series(self, series_id, observation_start=None, observation_end=None):
        """
        FRED経済指標データを取得してDBに保存
        
        パラメータ:
            series_id (str): シリーズID（例: 'GDP', 'CPIAUCSL'）
            observation_start (str): 開始日（YYYY-MM-DD形式）
            observation_end (str): 終了日（YYYY-MM-DD形式）
            
        戻り値:
            bool: 処理成功ならTrue、失敗ならFalse
        """
        try:
            df = self.api.get_series_data(series_id, observation_start, observation_end)
            
            if df.empty:
                self.logger.warning(f"{series_id}のデータ取得に失敗しました")
                return False
                
            return self.save_to_database(df)
            
        except Exception as e:
            self.logger.error(f"{series_id}のデータ取得・保存中にエラーが発生: {e}")
            return False
    
    def load_indicators_from_file(self, filename=None):
        """
        ファイルから経済指標リストを読み込む
        
        パラメータ:
            filename (str): 指標リストファイルのパス。Noneの場合はデフォルトを使用
            
        戻り値:
            list: 経済指標IDのリスト
        """
        indicators = []
        
        # デフォルトのファイル名を設定
        if filename is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # JSONファイルをまず探す
            json_file = os.path.join(project_root, "config", "indicators.json")
            if os.path.exists(json_file):
                filename = json_file
            else:
                # JSONがなければテキストファイルを使用
                filename = os.path.join(project_root, "config", "indicators.txt")
        
        # 拡張子に応じて読み込み方法を変更
        try:
            _, ext = os.path.splitext(filename)
            
            if os.path.exists(filename):
                if ext.lower() == '.json':
                    # JSONファイルからの読み込み
                    import json
                    with open(filename, 'r') as f:
                        data = json.load(f)
                        if 'fred_indicators' in data and isinstance(data['fred_indicators'], list):
                            indicators = data['fred_indicators']
                            self.logger.info(f"JSONファイル {filename} から {len(indicators)} 件の指標を読み込みました")
                else:
                    # テキストファイルからの読み込み
                    with open(filename, 'r') as f:
                        for line in f:
                            indicator = line.strip()
                            if indicator and not indicator.startswith('#'):
                                indicators.append(indicator)
                        self.logger.info(f"テキストファイル {filename} から {len(indicators)} 件の指標を読み込みました")
            else:
                self.logger.warning(f"指標ファイル {filename} が見つかりません。デフォルト指標を使用します。")
                # ファイルが見つからない場合はデフォルト値を使用
                indicators = self.DEFAULT_INDICATORS
                self.logger.info(f"デフォルト指標リスト {len(indicators)} 件を使用します")
                
        except Exception as e:
            self.logger.error(f"指標リストファイルの読み込みエラー: {e}")
            # エラーが発生した場合はデフォルト値を使用
            indicators = self.DEFAULT_INDICATORS
            self.logger.info(f"デフォルト指標リスト {len(indicators)} 件を使用します")
        
        return indicators
        
    def update_all_indicators(self, indicators=None):
        """
        すべての経済指標を更新
        
        パラメータ:
            indicators (list): 更新する指標のリスト、Noneの場合はファイルから読み込み
            
        戻り値:
            dict: 更新結果
        """
        if indicators is None:
            # ファイルから指標リストを読み込み
            indicators = self.load_indicators_from_file()
            
        results = {}
        self.logger.info(f"更新対象指標数: {len(indicators)}個")
        
        for indicator in indicators:
            self.logger.info(f"{indicator}の経済指標データを更新します")
            try:
                success = self.fetch_and_store_series(indicator)
                results[indicator] = success
            except Exception as e:
                self.logger.error(f"{indicator}の更新中にエラーが発生: {e}")
                results[indicator] = False
                
        return results
        
    def ensure_db_schema_exists(self, schema='fred_data', table_name='economic_indicators'):
        """
        必要なDBスキーマとテーブルが存在することを確認し、なければ作成する
        
        パラメータ:
            schema (str): スキーマ名
            table_name (str): テーブル名
            
        戻り値:
            bool: 成功ならTrue、失敗ならFalse
        """
        try:
            # スキーマの存在確認・作成
            schema_check_query = text(f"""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name = '{schema}'
            """)
            
            with self.db_engine.connect() as conn:
                schema_exists = conn.execute(schema_check_query).fetchone() is not None
                
                if not schema_exists:
                    self.logger.info(f"スキーマ {schema} が存在しないため作成します")
                    conn.execute(text(f"CREATE SCHEMA {schema}"))
                    
                # テーブルの存在確認・作成
                table_check_query = text(f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}' AND table_name = '{table_name}'
                """)
                
                table_exists = conn.execute(table_check_query).fetchone() is not None
                
                if not table_exists:
                    self.logger.info(f"テーブル {schema}.{table_name} が存在しないため作成します")
                    create_table_query = text(f"""
                    CREATE TABLE {schema}.{table_name} (
                        date DATE NOT NULL,
                        value NUMERIC,
                        indicator_name VARCHAR(50) NOT NULL,
                        frequency VARCHAR(20),
                        PRIMARY KEY (date, indicator_name),
                        CONSTRAINT economic_indicators_frequency_check 
                            CHECK (frequency = ANY (ARRAY['daily', 'monthly', 'quarterly', 'annual']))
                    )
                    """)
                    conn.execute(create_table_query)
                    
            return True
        except Exception as e:
            self.logger.error(f"データベーススキーマ/テーブル確認中にエラーが発生: {e}")
            return False 
    
    def calculate_yield_difference(self, start_date=None, end_date=None):
        """
        指定された期間のyield difference (DGS10 - FEDFUNDS)を計算しデータベースに保存する
        
        パラメータ:
            start_date (str): 開始日 ('YYYY-MM-DD'形式)、Noneの場合は最新のyield_differenceの次の日
            end_date (str): 終了日 ('YYYY-MM-DD'形式)、Noneの場合は現在日
            
        戻り値:
            bool: 処理が成功した場合True、失敗した場合False
        """
        self.logger.info("yield_difference (DGS10 - FEDFUNDS)の計算を開始します")
        
        try:
            # 日付範囲を決定
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                self.logger.info(f"終了日が指定されていないため、現在日({end_date})を使用します")
                
            if start_date is None:
                # 最新のyield_differenceの日付を取得
                query = text("""
                    SELECT MAX(date) as latest_date
                    FROM fred_data.economic_indicators
                    WHERE indicator_name = 'yield_difference'
                """)
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(query)
                    row = result.fetchone()
                    
                    if row and row[0]:
                        # 最新日の翌日から開始
                        latest_date = row[0]
                        start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                        self.logger.info(f"最新のyield_differenceの日付: {latest_date}, 開始日: {start_date}")
                    else:
                        # データがない場合は、少し長めの期間を取る
                        start_date = '2022-01-01'
                        self.logger.info(f"yield_differenceのデータが見つからないため、{start_date}から計算します")
            
            # DGS10の日足データを取得
            dgs10_query = text("""
                SELECT date, value
                FROM fred_data.economic_indicators
                WHERE indicator_name = 'DGS10'
                  AND date BETWEEN :start_date AND :end_date
                  AND value IS NOT NULL
                ORDER BY date
            """)
            
            # FEDFUNDSの月足データを取得
            fedfunds_query = text("""
                SELECT date, value
                FROM fred_data.economic_indicators
                WHERE indicator_name = 'FEDFUNDS'
                  AND date <= :end_date
                ORDER BY date DESC
                LIMIT 24  -- 過去2年分のデータを取得
            """)
            
            with self.db_engine.connect() as conn:
                # DGS10データを取得
                dgs10_result = conn.execute(dgs10_query, {"start_date": start_date, "end_date": end_date})
                dgs10_data = [{"date": row[0], "value": row[1]} for row in dgs10_result]
                
                if not dgs10_data:
                    self.logger.warning(f"期間({start_date}～{end_date})のDGS10データが見つかりません")
                    return False
                
                # FEDFUNDSデータを取得
                fedfunds_result = conn.execute(fedfunds_query, {"end_date": end_date})
                fedfunds_data = [{"date": row[0], "value": row[1]} for row in fedfunds_result]
                
                if not fedfunds_data:
                    self.logger.warning("FEDFUNDSデータが見つかりません")
                    return False
                
                # DataFrameに変換
                dgs10_df = pd.DataFrame(dgs10_data)
                fedfunds_df = pd.DataFrame(fedfunds_data)
                
                # FEDFUNDS(月足)を各日付に割り当てるための辞書を作成
                fedfunds_monthly = {}
                for _, row in fedfunds_df.iterrows():
                    # 日付を取得（datetime.dateオブジェクト）
                    date_val = row["date"]
                    year_month = f"{date_val.year}-{date_val.month:02d}"
                    fedfunds_monthly[year_month] = row["value"]
                
                # yield_differenceを計算
                yield_diff_data = []
                missing_fedfunds_days = []
                
                for _, row in dgs10_df.iterrows():
                    date_val = row["date"]
                    dgs10_val = row["value"]
                    
                    # NULL値をスキップ
                    if pd.isna(dgs10_val):
                        continue
                    
                    # DGS10の日付から年月を取得
                    year_month = f"{date_val.year}-{date_val.month:02d}"
                    
                    # 対応するFEDFUNDSの値を取得
                    if year_month in fedfunds_monthly:
                        fedfunds_val = fedfunds_monthly[year_month]
                        yield_diff = dgs10_val - fedfunds_val
                        
                        yield_diff_data.append({
                            "date": date_val,
                            "value": yield_diff,
                            "indicator_name": "yield_difference",
                            "frequency": "daily"
                        })
                    else:
                        missing_fedfunds_days.append(date_val)
                
                if missing_fedfunds_days:
                    self.logger.warning(f"{len(missing_fedfunds_days)}日分のFEDFUNDS対応データが見つかりません")
                    self.logger.debug(f"未対応日: {missing_fedfunds_days[:5]}{'...' if len(missing_fedfunds_days) > 5 else ''}")
                
                if not yield_diff_data:
                    self.logger.warning("計算されたyield_differenceデータがありません")
                    return False
                
                # 結果をDataFrameに変換
                yield_diff_df = pd.DataFrame(yield_diff_data)
                
                # データベースに保存
                if not yield_diff_df.empty:
                    success = self.save_to_database(yield_diff_df)
                    if success:
                        self.logger.info(f"{len(yield_diff_df)}件のyield_differenceデータを保存しました")
                    return success
                else:
                    self.logger.warning("保存するyield_differenceデータがありません")
                    return False
                
        except Exception as e:
            self.logger.error(f"yield_differenceの計算中にエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def update_indicator(self, indicator_id):
        """
        単一の経済指標を更新する
        
        パラメータ:
            indicator_id (str): 更新する指標のID
            
        戻り値:
            bool: 更新に成功した場合はTrue、失敗した場合はFalse
        """
        self.logger.info(f"{indicator_id} の経済指標データを更新します")
        
        try:
            # yield_differenceの場合は専用の関数を呼び出す
            if indicator_id == "yield_difference":
                return self.calculate_yield_difference()
                
            # 最新の日付を取得
            latest_date = None
            try:
                query = text("""
                    SELECT MAX(date) as latest_date
                    FROM fred_data.economic_indicators
                    WHERE indicator_name = :indicator_name
                """)
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(query, {"indicator_name": indicator_id})
                    row = result.fetchone()
                    if row and row[0]:
                        latest_date = row[0]
                        self.logger.info(f"{indicator_id} の最新日付: {latest_date}")
            except Exception as e:
                self.logger.warning(f"{indicator_id} の最新日付取得中にエラー: {e}")
            
            # 日付を計算
            if latest_date:
                # 最新データの翌日から今日まで
                observation_start = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # データがない場合は10年前から
                observation_start = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            observation_end = datetime.now().strftime('%Y-%m-%d')
            
            # データ取得・保存
            success = self.fetch_and_store_series(indicator_id, observation_start, observation_end)
            return success
            
        except Exception as e:
            self.logger.error(f"{indicator_id} の更新中にエラーが発生: {e}")
            return False 