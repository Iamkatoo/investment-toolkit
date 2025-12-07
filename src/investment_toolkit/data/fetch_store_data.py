import datetime
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from investment_toolkit.data.fmp_api import FMPAPI

class FMPDataManager:
    """
    FMP API経由でデータを取得・保存するデータマネージャークラス
    - APIリクエストの管理、レート制限対応
    - 取得したデータの整形、変換
    - データベースへの保存処理
    """
    
    def __init__(self, db_conn_string=None):
        """
        初期化
        
        パラメータ:
            db_conn_string (str): データベース接続文字列
        """
        self.logger = logging.getLogger('fmp_data_manager')
        self.logger.setLevel(logging.INFO)
        
        # ハンドラがなければ追加
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 接続文字列が指定されていない場合はデフォルト値を使用
        if db_conn_string is None:
            db_conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            
        # SQLAlchemyエンジンの作成
        self.db_engine = create_engine(db_conn_string)
        
        # API初期化
        self.api = FMPAPI()
        
        # APIリクエスト数カウンター
        self.request_count = 0
        self.api_request_count = {}  # エンドポイント別リクエスト数
        
        # データ取得量カウンター（バイト）
        self.data_size = 0
    
    def get_latest_date(self, symbol, table_name, period_type=None):
        """
        指定したテーブルの指定した銘柄の最新日付を取得する
        
        パラメータ:
            symbol (str): 証券コード
            table_name (str): テーブル名
            period_type (str): 'annual'または'quarterly'（財務データの場合）
            
        戻り値:
            str: 最新日付（YYYY-MM-DD形式）
        """
        # APIクラスの_get_latest_dateメソッドを利用
        return self.api._get_latest_date(symbol, table_name, 'fmp_data', period_type)
    
    def fetch_and_store_employee_count(self, symbol):
        """従業員数データを取得しDBに保存する"""
        try:
            # 現在のデータの最新年度を取得
            latest_year = None
            try:
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(
                        """
                        SELECT MAX(EXTRACT(YEAR FROM date)) as latest_year
                        FROM fmp_data.employee_counts
                        WHERE symbol = :symbol
                        """),
                        {"symbol": symbol}
                    )
                    row = result.fetchone()
                    if row and row[0]:
                        latest_year = int(row[0])
                        # 現在年から最新年を引いて、必要なlimitを計算
                        current_year = datetime.datetime.now().year
                        limit = current_year - latest_year
                        if limit <= 0:
                            # すでに最新データがある場合は1（最小値）を設定
                            limit = 1
                        else:
                            # 安全のために+1年追加
                            limit = limit + 1
                    else:
                        # データがない場合、デフォルトで5年分取得
                        limit = 5
            except Exception as e:
                self.logger.warning(f"{symbol}の従業員数データの最新年度取得中にエラーが発生しました: {e}")
                latest_year = None
                limit = 5

            self.logger.info(f"{symbol}の従業員数データを取得します（limit={limit}）")
            employee_data = self.api.get_employee_count(symbol, limit=limit)
            
            if not employee_data or len(employee_data) == 0:
                self.logger.warning(f"{symbol}の従業員数データが見つかりませんでした")
                return False
            
            # データをデータフレームに変換
            df = pd.DataFrame(employee_data)
            
            # symbol列がなければ追加
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # データをDBに保存
            return self.save_to_database(df, 'employee_counts')
        except Exception as e:
            self.logger.error(f"{symbol}の従業員数データの取得・保存中にエラーが発生しました: {e}")
            return False
            
    def fetch_and_store_company_profile(self, symbol):
        """会社プロファイルデータを取得しDBに保存する"""
        try:
            # APIからデータ取得
            data = self.api.get_company_profile(symbol)
            
            if not data or (isinstance(data, dict) and not data):
                self.logger.warning(f"{symbol}の会社プロファイル情報が見つかりませんでした")
                return False
                
            # データをデータフレームに変換
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
                
            # symbol列がなければ追加
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
                
            # 日付列を追加
            df['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # データをDBに保存
            return self.save_to_database(df, 'company_profile')
            
        except Exception as e:
            self.logger.error(f"{symbol}の会社プロファイル取得・保存中にエラーが発生: {e}")
            return False
            
    def save_to_database(self, df, table_name, schema="fmp_data"):
        """
        DataFrameをデータベースに保存する
        
        パラメータ:
            df (DataFrame): 保存するデータフレーム
            table_name (str): テーブル名
            schema (str): スキーマ名
            
        戻り値:
            bool: 保存に成功したかどうか
        """
        try:
            if df.empty:
                self.logger.warning(f"保存するデータが空のため、{table_name}へのデータ保存をスキップします")
                return False
                
            # データベースに保存
            with self.db_engine.connect() as conn:
                # daily_pricesテーブルの場合は特別に処理（日付型の問題などがあるため）
                if table_name == 'daily_prices' and 'date' in df.columns:
                    # 直接INSERTを使用
                    success_count = 0
                    for _, row in df.iterrows():
                        # NULLの場合はデフォルト値を使用
                        date_val = f"'{row['date']}'" if pd.notna(row['date']) else 'NULL'
                        symbol_val = f"'{row['symbol']}'" if pd.notna(row['symbol']) else 'NULL'
                        
                        # 各数値カラムの値をチェック、NULLの場合はNULLを使用
                        values = {}
                        for col in ['open', 'high', 'low', 'close', 'volume', 'change', 'change_percent', 'vwap']:
                            if col in row and pd.notna(row[col]):
                                values[col] = str(row[col])
                            else:
                                values[col] = 'NULL'
                        
                        # 明示的なINSERT文を作成
                        sql = f"""
                            INSERT INTO {schema}.{table_name} 
                            (date, symbol, open, high, low, close, volume, change, change_percent, vwap)
                            VALUES 
                            ({date_val}::date, {symbol_val}, {values['open']}, {values['high']}, 
                             {values['low']}, {values['close']}, {values['volume']}, 
                             {values['change']}, {values['change_percent']}, {values['vwap']})
                            ON CONFLICT (symbol, date) DO NOTHING
                        """
                        
                        try:
                            conn.execute(text(sql))
                            success_count += 1
                        except Exception as e:
                            self.logger.error(f"行データの挿入中にエラー: {e}")
                            self.logger.error(f"エラーとなったSQL: {sql}")
                    
                    # トランザクション確定
                    conn.commit()
                    
                    if success_count > 0:
                        self.logger.info(f"{table_name}テーブルに{success_count}行のデータを保存しました")
                        return True
                    else:
                        self.logger.error(f"{table_name}テーブルへのデータ保存に失敗しました")
                        return False
                else:
                    # 一時テーブルの名前を作成（daily_prices以外のテーブル用）
                    temp_table = f"temp_{table_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    if table_name in ['income_statements', 'balance_sheets', 'cash_flows']:
                        # 財務諸表テーブルは(symbol, date, period_type)がユニーク
                        # 一時テーブル作成時に適切な型を指定して日付型変換の問題を解決
                        
                        # 日付型に変換が必要なカラム
                        date_columns = ['filing_date', 'accepted_date']
                        
                        # 一時テーブルを削除（存在する場合）
                        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{temp_table}"))
                        
                        # 一時テーブルを適切な型で作成
                        create_temp_sql = f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM {schema}.{table_name} WHERE 1=0"
                        conn.execute(text(create_temp_sql))
                        
                        # データフレームを一時テーブルに挿入（行ごとに処理）
                        for _, row in df.iterrows():
                            # 各カラムの値を準備
                            values = []
                            placeholders = []
                            for col in df.columns:
                                if col == 'date':
                                    # メインの日付カラム（文字列形式の日付）
                                    values.append(row[col])
                                    placeholders.append(f"CAST(:{col} AS date)")
                                elif col in date_columns:
                                    # 日付カラムの型変換（文字列形式の日付）
                                    if pd.notna(row[col]) and row[col] != '':
                                        # 文字列の日付をそのまま使用
                                        values.append(row[col])
                                        placeholders.append(f"CAST(:{col} AS date)")
                                    else:
                                        values.append(None)
                                        placeholders.append(f":{col}")
                                else:
                                    values.append(row[col])
                                    placeholders.append(f":{col}")
                            
                            # INSERT文を実行
                            insert_sql = f"""
                                INSERT INTO {temp_table} ({", ".join(df.columns)})
                                VALUES ({", ".join(placeholders)})
                            """
                            
                            # パラメータ辞書を作成
                            params = {col: values[i] for i, col in enumerate(df.columns)}
                            
                            try:
                                conn.execute(text(insert_sql), params)
                            except Exception as e:
                                self.logger.error(f"一時テーブルへの行挿入エラー: {e}")
                                self.logger.error(f"SQL: {insert_sql}")
                                self.logger.error(f"パラメータ: {params}")
                                continue
                        
                        # 本テーブルにUPSERT
                        conn.execute(text(f"""
                            INSERT INTO {schema}.{table_name}
                            SELECT * FROM {temp_table}
                            ON CONFLICT (symbol, date, period_type) DO UPDATE SET
                            {", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col not in ['symbol', 'date', 'period_type']])}
                        """))
                        
                        # 一時テーブルを削除
                        conn.execute(text(f"DROP TABLE IF EXISTS {temp_table}"))
                    else:
                        # 通常の方法でデータフレームをテーブルに変換
                        df.to_sql(temp_table, conn, schema=schema, if_exists='replace', index=False)
                        
                        # データベースに保存（UPSERT）
                        if table_name == 'company_profile':
                            # 会社プロファイルテーブルは(symbol)がユニーク
                            conn.execute(text(f"""
                                INSERT INTO {schema}.{table_name}
                                SELECT * FROM {schema}.{temp_table}
                                ON CONFLICT (symbol) DO UPDATE SET
                                {", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col != 'symbol'])}
                            """))
                        elif table_name == 'employee_counts':
                            # 従業員数テーブルは(symbol, date)がユニーク
                            conn.execute(text(f"""
                                INSERT INTO {schema}.{table_name}
                                SELECT * FROM {schema}.{temp_table}
                                ON CONFLICT (symbol, date) DO UPDATE SET
                                {", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col not in ['symbol', 'date']])}
                            """))
                        else:
                            # その他のテーブル
                            conn.execute(text(f"""
                                INSERT INTO {schema}.{table_name}
                                SELECT * FROM {schema}.{temp_table}
                                ON CONFLICT DO NOTHING
                            """))
                        
                        # 一時テーブルを削除
                        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{temp_table}"))
                
                    # トランザクション確定
                    conn.commit()
                    
                    self.logger.info(f"{table_name}テーブルに{len(df)}行のデータを保存しました")
                    return True
        except Exception as e:
            self.logger.error(f"{table_name}テーブルへのデータ保存中にエラーが発生: {e}")
            return False

    def fetch_and_store_income_statements(self, symbol, period='annual'):
        """
        損益計算書データを取得してDBに保存する
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            
        戻り値:
            bool: 保存に成功したかどうか
        """
        try:
            # 日本株かどうかを判断
            is_japanese_stock = symbol.endswith('.T')
            
            self.logger.info(f"{symbol}の損益計算書データ({period})を取得します")
            
            # APIからデータ取得
            data = self.api.get_income_statements(symbol, period, is_japanese_stock=is_japanese_stock)
            
            if not data:
                self.logger.warning(f"{symbol}の損益計算書データ({period})が見つかりませんでした")
                return False
            
            # データの構造を確認・修正
            if isinstance(data, dict) and 'historical' in data:
                # 古い構造（historicalキーあり）
                data = data['historical']
            elif isinstance(data, list):
                # 新しい構造（直接リスト）
                pass
            else:
                self.logger.error(f"{symbol}の損益計算書データの構造が不正です: {type(data)}")
                return False
            
            if not data:
                self.logger.warning(f"{symbol}の損益計算書データ({period})が空です")
                return False
            
            # データをDataFrameに変換
            df = pd.DataFrame(data)
            
            # 必要なカラムを追加・修正
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # period_typeを追加
            df['period_type'] = period
            
            # フィールド名の正規化（APIレスポンスとDBカラム名の違いを修正）
            field_mapping = {
                'ebitdaratio': 'ebitda_ratio',
                'epsdiluted': 'eps_diluted',
                'calendarYear': 'calendar_year',
                'fillingDate': 'filing_date',
                'acceptedDate': 'accepted_date',
                'reportedCurrency': 'reported_currency',
                'costOfRevenue': 'cost_of_revenue',
                'grossProfit': 'gross_profit',
                'grossProfitRatio': 'gross_profit_ratio',
                'researchAndDevelopmentExpenses': 'research_and_development_expenses',
                'generalAndAdministrativeExpenses': 'general_and_administrative_expenses',
                'sellingAndMarketingExpenses': 'selling_and_marketing_expenses',
                'sellingGeneralAndAdministrativeExpenses': 'selling_general_and_administrative_expenses',
                'otherExpenses': 'other_expenses',
                'operatingExpenses': 'operating_expenses',
                'costAndExpenses': 'cost_and_expenses',
                'interestIncome': 'interest_income',
                'interestExpense': 'interest_expense',
                'depreciationAndAmortization': 'depreciation_and_amortization',
                'ebitdaRatio': 'ebitda_ratio',
                'operatingIncome': 'operating_income',
                'operatingIncomeRatio': 'operating_income_ratio',
                'totalOtherIncomeExpensesNet': 'total_other_income_expenses_net',
                'incomeBeforeTax': 'income_before_tax',
                'incomeBeforeTaxRatio': 'income_before_tax_ratio',
                'incomeTaxExpense': 'income_tax_expense',
                'netIncome': 'net_income',
                'netIncomeRatio': 'net_income_ratio',
                'weightedAverageShsOut': 'weighted_average_shs_out',
                'weightedAverageShsOutDil': 'weighted_average_shs_out_dil',
                'finalLink': 'final_link'
            }
            
            # フィールド名を変換
            for old_name, new_name in field_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
                    df.drop(columns=[old_name], inplace=True)
            
            # データをDBに保存
            return self.save_to_database(df, 'income_statements')
            
        except Exception as e:
            self.logger.error(f"{symbol}の損益計算書データ({period})の取得・保存中にエラーが発生: {e}")
            return False

    def fetch_and_store_balance_sheets(self, symbol, period='annual'):
        """
        貸借対照表データを取得してDBに保存する
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            
        戻り値:
            bool: 保存に成功したかどうか
        """
        try:
            # 日本株かどうかを判断
            is_japanese_stock = symbol.endswith('.T')
            
            self.logger.info(f"{symbol}の貸借対照表データ({period})を取得します")
            
            # APIからデータ取得
            data = self.api.get_balance_sheets(symbol, period, is_japanese_stock=is_japanese_stock)
            
            if not data:
                self.logger.warning(f"{symbol}の貸借対照表データ({period})が見つかりませんでした")
                return False
            
            # データの構造を確認・修正
            if isinstance(data, dict) and 'historical' in data:
                # 古い構造（historicalキーあり）
                data = data['historical']
            elif isinstance(data, list):
                # 新しい構造（直接リスト）
                pass
            else:
                self.logger.error(f"{symbol}の貸借対照表データの構造が不正です: {type(data)}")
                return False
            
            if not data:
                self.logger.warning(f"{symbol}の貸借対照表データ({period})が空です")
                return False
            
            # データをDataFrameに変換
            df = pd.DataFrame(data)
            
            # 必要なカラムを追加・修正
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # period_typeを追加
            df['period_type'] = period
            
            # フィールド名の正規化
            field_mapping = {
                'calendarYear': 'calendar_year',
                'fillingDate': 'filing_date',
                'acceptedDate': 'accepted_date',
                'reportedCurrency': 'reported_currency',
                'cashAndCashEquivalents': 'cash_and_cash_equivalents',
                'shortTermInvestments': 'short_term_investments',
                'cashAndShortTermInvestments': 'cash_and_short_term_investments',
                'netReceivables': 'net_receivables',
                'otherCurrentAssets': 'other_current_assets',
                'totalCurrentAssets': 'total_current_assets',
                'propertyPlantEquipmentNet': 'property_plant_equipment_net',
                'intangibleAssets': 'intangible_assets',
                'goodwillAndIntangibleAssets': 'goodwill_and_intangible_assets',
                'longTermInvestments': 'long_term_investments',
                'taxAssets': 'tax_assets',
                'otherNonCurrentAssets': 'other_non_current_assets',
                'totalNonCurrentAssets': 'total_non_current_assets',
                'otherAssets': 'other_assets',
                'totalAssets': 'total_assets',
                'accountPayables': 'account_payables',
                'shortTermDebt': 'short_term_debt',
                'taxPayables': 'tax_payables',
                'deferredRevenue': 'deferred_revenue',
                'otherCurrentLiabilities': 'other_current_liabilities',
                'totalCurrentLiabilities': 'total_current_liabilities',
                'longTermDebt': 'long_term_debt',
                'deferredRevenueNonCurrent': 'deferred_revenue_non_current',
                'deferredTaxLiabilitiesNonCurrent': 'deferred_tax_liabilities_non_current',
                'otherNonCurrentLiabilities': 'other_non_current_liabilities',
                'totalNonCurrentLiabilities': 'total_non_current_liabilities',
                'otherLiabilities': 'other_liabilities',
                'capitalLeaseObligations': 'capital_lease_obligations',
                'totalLiabilities': 'total_liabilities',
                'preferredStock': 'preferred_stock',
                'commonStock': 'common_stock',
                'retainedEarnings': 'retained_earnings',
                'accumulatedOtherComprehensiveIncomeLoss': 'accumulated_other_comprehensive_income_loss',
                'otherTotalStockholdersEquity': 'other_total_stockholders_equity',
                'othertotalStockholdersEquity': 'other_total_stockholders_equity',  # APIの実際のフィールド名
                'totalStockholdersEquity': 'total_stockholders_equity',
                'totalEquity': 'total_equity',
                'totalLiabilitiesAndStockholdersEquity': 'total_liabilities_and_stockholders_equity',
                'minorityInterest': 'minority_interest',
                'totalLiabilitiesAndTotalEquity': 'total_liabilities_and_total_equity',
                'totalInvestments': 'total_investments',
                'totalDebt': 'total_debt',
                'netDebt': 'net_debt',
                'finalLink': 'final_link'
            }
            
            # フィールド名を変換
            for old_name, new_name in field_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
                    df.drop(columns=[old_name], inplace=True)
            
            # データをDBに保存
            return self.save_to_database(df, 'balance_sheets')
            
        except Exception as e:
            self.logger.error(f"{symbol}の貸借対照表データ({period})の取得・保存中にエラーが発生: {e}")
            return False

    def fetch_and_store_cash_flows(self, symbol, period='annual'):
        """
        キャッシュフロー計算書データを取得してDBに保存する
        
        パラメータ:
            symbol (str): 銘柄コード
            period (str): 期間（'annual'または'quarter'）
            
        戻り値:
            bool: 保存に成功したかどうか
        """
        try:
            # 日本株かどうかを判断
            is_japanese_stock = symbol.endswith('.T')
            
            self.logger.info(f"{symbol}のキャッシュフロー計算書データ({period})を取得します")
            
            # APIからデータ取得
            data = self.api.get_cash_flows(symbol, period, is_japanese_stock=is_japanese_stock)
            
            if not data:
                self.logger.warning(f"{symbol}のキャッシュフロー計算書データ({period})が見つかりませんでした")
                return False
            
            # データの構造を確認・修正
            if isinstance(data, dict) and 'historical' in data:
                # 古い構造（historicalキーあり）
                data = data['historical']
            elif isinstance(data, list):
                # 新しい構造（直接リスト）
                pass
            else:
                self.logger.error(f"{symbol}のキャッシュフロー計算書データの構造が不正です: {type(data)}")
                return False
            
            if not data:
                self.logger.warning(f"{symbol}のキャッシュフロー計算書データ({period})が空です")
                return False
            
            # データをDataFrameに変換
            df = pd.DataFrame(data)
            
            # 必要なカラムを追加・修正
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # period_typeを追加
            df['period_type'] = period
            
            # フィールド名の正規化
            field_mapping = {
                'calendarYear': 'calendar_year',
                'fillingDate': 'filing_date',
                'acceptedDate': 'accepted_date',
                'reportedCurrency': 'reported_currency',
                'netIncome': 'net_income',
                'depreciationAndAmortization': 'depreciation_and_amortization',
                'deferredIncomeTax': 'deferred_income_tax',
                'stockBasedCompensation': 'stock_based_compensation',
                'changeInWorkingCapital': 'change_in_working_capital',
                'accountsReceivables': 'accounts_receivables',
                'accountsPayables': 'accounts_payables',
                'otherWorkingCapital': 'other_working_capital',
                'otherNonCashItems': 'other_non_cash_items',
                'netCashProvidedByOperatingActivities': 'net_cash_provided_by_operating_activities',
                'investmentsInPropertyPlantAndEquipment': 'investments_in_property_plant_and_equipment',
                'acquisitionsNet': 'acquisitions_net',
                'purchasesOfInvestments': 'purchases_of_investments',
                'salesMaturitiesOfInvestments': 'sales_maturities_of_investments',
                'otherInvestingActivites': 'other_investing_activities',
                'netCashUsedForInvestingActivites': 'net_cash_used_for_investing_activities',
                'debtRepayment': 'debt_repayment',
                'commonStockIssued': 'common_stock_issued',
                'commonStockRepurchased': 'common_stock_repurchased',
                'dividendsPaid': 'dividends_paid',
                'otherFinancingActivites': 'other_financing_activities',
                'netCashUsedProvidedByFinancingActivities': 'net_cash_used_provided_by_financing_activities',
                'effectOfForexChangesOnCash': 'effect_of_forex_changes_on_cash',
                'netChangeInCash': 'net_change_in_cash',
                'cashAtEndOfPeriod': 'cash_at_end_of_period',
                'cashAtBeginningOfPeriod': 'cash_at_beginning_of_period',
                'operatingCashFlow': 'operating_cash_flow',
                'capitalExpenditure': 'capital_expenditure',
                'freeCashFlow': 'free_cash_flow',
                'finalLink': 'final_link'
            }
            
            # フィールド名を変換
            for old_name, new_name in field_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
                    df.drop(columns=[old_name], inplace=True)
            
            # データをDBに保存
            return self.save_to_database(df, 'cash_flows')
            
        except Exception as e:
            self.logger.error(f"{symbol}のキャッシュフロー計算書データ({period})の取得・保存中にエラーが発生: {e}")
            return False

# SQLの型を取得するヘルパー関数
def get_sql_type(dtype):
    """
    Pandas dtypeからSQLの型を取得
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "NUMERIC"
    elif pd.api.types.is_datetime64_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"

def fetch_daily_prices():
    """
    最新の日次株価データを取得する
    
    Returns:
        pd.DataFrame: 日次株価データ（symbol, date, close）
    """
    try:
        # ログレベルを一時的にDEBUGに設定
        logger = logging.getLogger(__name__)
        original_level = logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)
        
        # データベース接続
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        logger.debug("データベースに接続しました")
        
        # 完全なデータセットを持つ最新の日付を取得
        with engine.connect() as conn:
            # 通常の1日あたりのデータ数を計算（過去30日の平均）
            result = conn.execute(text("""
                WITH daily_counts AS (
                    SELECT date, COUNT(*) as count
                    FROM fmp_data.daily_prices
                    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY date
                    ORDER BY date DESC
                )
                SELECT AVG(count)::integer as avg_count
                FROM daily_counts
            """))
            avg_count = result.fetchone()[0]
            logger.debug(f"1日あたりの平均データ数: {avg_count}")
            
            # 完全なデータセットを持つ最新の日付を取得（平均の90%以上のデータを持つ日）
            result = conn.execute(text("""
                WITH daily_counts AS (
                    SELECT date, COUNT(*) as count
                    FROM fmp_data.daily_prices
                    GROUP BY date
                    HAVING COUNT(*) >= :min_count
                    ORDER BY date DESC
                    LIMIT 1
                )
                SELECT date, count FROM daily_counts
            """), {"min_count": int(avg_count * 0.9)})
            row = result.fetchone()
            if not row:
                logger.error("完全なデータセットを持つ日付が見つかりません")
                raise Exception("完全なデータセットを持つ日付が見つかりません")
                
            latest_date = row[0]
            count = row[1]
            logger.debug(f"最新の完全なデータセット日付: {latest_date} (データ数: {count})")
            
            # 最新日のデータを取得
            query = text("""
            SELECT symbol, date, close
            FROM fmp_data.daily_prices
            WHERE date = :latest_date
            """)
            df = pd.read_sql(query, conn, params={"latest_date": latest_date})
            logger.debug(f"取得したデータ: {len(df)}行")
            logger.debug(f"データのサンプル: \n{df.head()}")
            
            if df.empty:
                logger.error(f"日付 {latest_date} のデータが存在しません")
                raise Exception(f"日付 {latest_date} のデータが存在しません")
            
            # ログレベルを元に戻す
            logger.setLevel(original_level)
            return df
            
    except Exception as e:
        logger.error(f"日次株価データ取得エラー: {e}")
        # ログレベルを元に戻す
        logger.setLevel(original_level)
        return pd.DataFrame(columns=['symbol', 'date', 'close'])

def fetch_daily_financials():
    """
    最新の財務データを取得する
    
    Returns:
        pd.DataFrame: 財務データ
    """
    try:
        # データベース接続
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 最新の日付を取得
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT MAX(date) as latest_date FROM fmp_data.income_statements"
            ))
            latest_date = result.fetchone()[0]
            
            if not latest_date:
                raise Exception("財務データが存在しません")
            
            # 最新日のデータを取得
            query = text("""
            SELECT *
            FROM fmp_data.income_statements
            WHERE date = :latest_date
            """)
            df = pd.read_sql(query, conn, params={"latest_date": latest_date})
            
            return df
            
    except Exception as e:
        logging.error(f"財務データ取得エラー: {e}")
        return pd.DataFrame()

def fetch_market_cap():
    """
    最新の時価総額データを取得する
    株価と発行済み株式数から計算
    
    Returns:
        pd.DataFrame: 時価総額データ（symbol, market_cap）
    """
    try:
        logger = logging.getLogger(__name__)
        logger.debug("時価総額データを計算します")
        
        # データベース接続
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 最新の価格データを取得
        latest_prices = fetch_daily_prices()
        if latest_prices.empty:
            logger.error("価格データが取得できません")
            return pd.DataFrame(columns=['symbol', 'market_cap'])
            
        logger.debug(f"価格データを取得: {len(latest_prices)}件")
        
        # 最新の発行済み株式数を取得
        with engine.connect() as conn:
            shares_query = text("""
            WITH latest_shares AS (
                SELECT symbol, MAX(date) as latest_date
                FROM fmp_data.shares
                GROUP BY symbol
            )
            SELECT s.symbol, s.outstanding_shares
            FROM fmp_data.shares s
            JOIN latest_shares ls ON s.symbol = ls.symbol AND s.date = ls.latest_date
            """)
            shares_df = pd.read_sql(shares_query, conn)
            logger.debug(f"株式数データを取得: {len(shares_df)}件")
            
            if shares_df.empty:
                logger.error("発行済み株式数データが取得できません")
                return pd.DataFrame(columns=['symbol', 'market_cap'])
        
        # 株価と株式数を結合して時価総額を計算
        df = pd.merge(latest_prices[['symbol', 'close']], shares_df, on='symbol', how='inner')
        logger.debug(f"結合後のデータ: {len(df)}件")
        
        # 時価総額を計算 (株価 × 発行済み株式数)
        df['market_cap'] = df['close'] * df['outstanding_shares']
        logger.debug(f"時価総額計算結果: {df['market_cap'].notna().sum()}件")
        
        # 不要なカラムを削除
        df = df[['symbol', 'market_cap']]
        
        # デバッグ用にNaNの数を記録
        nan_count = df['market_cap'].isna().sum()
        if nan_count > 0:
            logger.warning(f"時価総額データにNaNが{nan_count}件含まれています")
        
        return df
            
    except Exception as e:
        logger.error(f"時価総額データ計算エラー: {e}")
        return pd.DataFrame(columns=['symbol', 'market_cap'])

def fetch_cashflow():
    """
    最新のキャッシュフローデータを取得する
    
    Returns:
        pd.DataFrame: キャッシュフローデータ
    """
    try:
        # データベース接続
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # 最新の日付を取得
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT MAX(date) as latest_date FROM fmp_data.cash_flows"
            ))
            latest_date = result.fetchone()[0]
            
            if not latest_date:
                raise Exception("キャッシュフローデータが存在しません")
            
            # 最新日のデータを取得
            query = text("""
            SELECT *
            FROM fmp_data.cash_flows
            WHERE date = :latest_date
            """)
            df = pd.read_sql(query, conn, params={"latest_date": latest_date})
            
            return df
            
    except Exception as e:
        logging.error(f"キャッシュフローデータ取得エラー: {e}")
        return pd.DataFrame()