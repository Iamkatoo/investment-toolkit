#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
データベースチェッカー

テーブルの更新状況をチェックし、期待される件数と実際の件数を比較する
"""

import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional
import yaml

from investment_toolkit.database.db_manager import get_db_connection
from .date_calculator import DateCalculator
from .monitor_result import MonitorResult, TableCheckResult

logger = logging.getLogger(__name__)


class DatabaseChecker:
    """データベース監視チェッカー"""

    def __init__(self, config_path: str):
        """
        初期化

        Args:
            config_path: 監視設定YAMLファイルのパス
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.date_calculator = DateCalculator()

    def _load_config(self) -> Dict:
        """設定ファイルを読み込む"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
            raise

    def _get_exchange_condition(self, market: str) -> str:
        """
        市場に応じたexchange条件を取得

        Args:
            market: 市場 ("us" or "jp")

        Returns:
            SQL WHERE句のexchange条件
        """
        if market == "us":
            return "ss.exchange IN ('NASDAQ', 'NYSE', 'NASDAQ Global Select', 'New York Stock Exchange', 'NASDAQ Stock Exchange')"
        elif market == "jp":
            return "ss.exchange = 'Tokyo'"
        else:
            return "1=1"

    def _get_statement_type_from_table(self, table: str, is_ttm: bool = False) -> Optional[str]:
        """
        テーブル名から決算情報の種類を判定

        Args:
            table: テーブル名
            is_ttm: TTMテーブルかどうか

        Returns:
            決算情報の種類 ("income", "balance", "cash") または None
        """
        prefix = "ttm_" if is_ttm else ""

        if f"{prefix}income_statement" in table:
            return "income"
        elif f"{prefix}balance_sheet" in table:
            return "balance"
        elif f"{prefix}cash_flow" in table:
            return "cash"
        return None

    def get_active_symbols_count(self, market: str) -> int:
        """
        アクティブ銘柄数を取得

        Args:
            market: 市場 ("us" or "jp")

        Returns:
            アクティブ銘柄数
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 市場に応じてexchangeを判定
            exchange_condition = self._get_exchange_condition(market)

            query = f"""
                SELECT COUNT(*)
                FROM fmp_data.symbol_status ss
                WHERE ss.is_active = TRUE
                  AND ss.manually_deactivated = FALSE
                  AND {exchange_condition}
            """

            cursor.execute(query)
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            logger.debug(f"アクティブ銘柄数 ({market}): {count}")
            return count

        except Exception as e:
            logger.error(f"アクティブ銘柄数の取得に失敗しました: {e}")
            return 0

    def get_earnings_processed_count(self, market: str, statement_type: str, reference_date: datetime = None) -> int:
        """
        決算情報の期待件数を取得（本日処理された銘柄数）

        本日 xxx_completed_at が更新された銘柄のうち、
        アクティブかつ手動無効化されていない銘柄の数を返す。
        これが「本日の取得処理で処理された銘柄数」となり、
        監視の分母として使用される。

        Args:
            market: 市場 ("us" or "jp")
            statement_type: 決算情報の種類 ("income", "balance", "cash")
            reference_date: 基準日付（デフォルトは本日）

        Returns:
            本日処理された決算情報の件数
        """
        if reference_date is None:
            reference_date = datetime.now()

        # 決算カラムのタイムスタンプ取得
        timestamp_column_map = {
            "income": "income_completed_at",
            "balance": "balance_completed_at",
            "cash": "cash_completed_at"
        }
        timestamp_column = timestamp_column_map.get(statement_type)
        if not timestamp_column:
            logger.error(f"Unknown statement_type: {statement_type}")
            return 0

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 市場に応じてexchangeを判定
            exchange_condition = self._get_exchange_condition(market)

            # 本日 xxx_completed_at が更新された銘柄を、
            # symbol_statusでアクティブかつ手動無効化されていない銘柄に限定してカウント
            query = f"""
                SELECT COUNT(DISTINCT e.symbol)
                FROM fmp_data.earnings e
                INNER JOIN fmp_data.symbol_status ss ON e.symbol = ss.symbol
                WHERE e.{timestamp_column}::date = %s
                  AND ss.is_active = TRUE
                  AND ss.manually_deactivated = FALSE
                  AND {exchange_condition}
            """

            cursor.execute(query, (reference_date.date(),))
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            logger.debug(f"本日処理された {statement_type} statements ({market}市場): {count}")
            return count

        except Exception as e:
            logger.error(f"処理済み決算情報のカウント取得に失敗しました ({statement_type}): {e}")
            return 0

    def get_ttm_calculated_count(self, market: str, statement_type: str, reference_date: datetime = None) -> int:
        """
        TTM決算情報の実際の取得件数を取得

        本日 fmp_data.earnings.xxx_completed_at が更新された銘柄のうち、
        対応するTTMテーブルに最新の決算データが反映されている銘柄数を返す。

        Args:
            market: 市場 ("us" or "jp")
            statement_type: 決算情報の種類 ("income", "balance", "cash")
            reference_date: 基準日付（デフォルトは本日）

        Returns:
            TTM計算が完了した銘柄数
        """
        if reference_date is None:
            reference_date = datetime.now()

        # テーブル名のマッピング
        source_table_map = {
            "income": "income_statements",
            "balance": "balance_sheets",
            "cash": "cash_flows"
        }
        ttm_table_map = {
            "income": "ttm_income_statements",
            "balance": "ttm_balance_sheets",
            "cash": "ttm_cash_flows"
        }
        timestamp_column_map = {
            "income": "income_completed_at",
            "balance": "balance_completed_at",
            "cash": "cash_completed_at"
        }

        source_table = source_table_map.get(statement_type)
        ttm_table = ttm_table_map.get(statement_type)
        timestamp_column = timestamp_column_map.get(statement_type)

        if not all([source_table, ttm_table, timestamp_column]):
            logger.error(f"Unknown statement_type: {statement_type}")
            return 0

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 市場に応じてexchangeを判定
            exchange_condition = self._get_exchange_condition(market)

            # 本日更新された銘柄のうち、TTMテーブルに最新決算データが反映されている銘柄数を取得
            query = f"""
                SELECT COUNT(DISTINCT ttm.symbol)
                FROM calculated_metrics.{ttm_table} ttm
                INNER JOIN (
                    SELECT
                        src.symbol,
                        MAX(src.date) as latest_date
                    FROM fmp_data.{source_table} src
                    INNER JOIN fmp_data.earnings e ON src.symbol = e.symbol
                    INNER JOIN fmp_data.symbol_status ss ON src.symbol = ss.symbol
                    WHERE e.{timestamp_column}::date = %s
                      AND ss.is_active = TRUE
                      AND ss.manually_deactivated = FALSE
                      AND {exchange_condition}
                    GROUP BY src.symbol
                ) updated_symbols ON ttm.symbol = updated_symbols.symbol
                WHERE ttm.as_of_date = updated_symbols.latest_date
            """

            cursor.execute(query, (reference_date.date(),))
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            logger.debug(f"TTM {statement_type} 計算完了 ({market}市場): {count}")
            return count

        except Exception as e:
            logger.error(f"TTM決算情報のカウント取得に失敗しました ({statement_type}): {e}")
            return 0

    def check_table(
        self,
        schema: str,
        table: str,
        config: Dict,
        market: str,
        reference_date: datetime = None
    ) -> TableCheckResult:
        """
        テーブルをチェック

        Args:
            schema: スキーマ名
            table: テーブル名
            config: テーブル設定
            market: 市場
            reference_date: 基準日付

        Returns:
            TableCheckResult
        """
        frequency = config.get("frequency")
        date_column = config.get("date_column", "date")
        count_column = config.get("count_column", "symbol")
        threshold_pct = config.get("alert_threshold_pct")
        expected_min_rows = config.get("expected_min_rows")
        description = config.get("description", "")
        is_earnings_statement = config.get("is_earnings_statement", False)
        is_ttm_statement = config.get("is_ttm_statement", False)
        require_tuesday_check = config.get("require_tuesday_check", False)
        is_double_count = config.get("is_double_count", False)
        use_current_date = config.get("use_current_date", False)

        # 期待される日付を計算
        if use_current_date:
            # use_current_dateフラグがある場合は、市場に関わらず当日の日付を使用
            expected_date = (reference_date or datetime.now()).strftime("%Y-%m-%d")
        else:
            # 通常の市場別・頻度別の日付計算
            expected_date = self.date_calculator.get_expected_date(
                market=market,
                frequency=frequency,
                reference_date=reference_date
            )

        result = TableCheckResult(
            schema=schema,
            table=table,
            frequency=frequency,
            market=market,
            expected_date=expected_date,
            actual_count=0,
            threshold_pct=threshold_pct,
        )

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # TTMテーブルの特別処理（実際の件数は専用関数で取得）
            if is_ttm_statement:
                # テーブル名から決算情報の種類を判定
                statement_type = self._get_statement_type_from_table(table, is_ttm=True)

                if statement_type:
                    # TTM計算が完了した銘柄数を取得（専用のカウント関数を使用）
                    actual_count = self.get_ttm_calculated_count(
                        market=market,
                        statement_type=statement_type,
                        reference_date=reference_date
                    )
                    result.actual_count = actual_count
                else:
                    result.actual_count = 0
            else:
                # 通常テーブルの実際のレコード数を取得
                # タイムスタンプ型のカラムは日付にキャストして比較
                # (例: retrieved_at, created_at など)
                if '_at' in date_column or 'timestamp' in date_column.lower():
                    date_comparison = f"t.{date_column}::date = %s"
                else:
                    date_comparison = f"t.{date_column} = %s"

                # 市場に応じてexchange条件を取得
                exchange_condition = self._get_exchange_condition(market)

                # count_columnがsymbolの場合は市場別にフィルタリング
                if count_column == "symbol":
                    query = f"""
                        SELECT COUNT(DISTINCT t.{count_column})
                        FROM {schema}.{table} t
                        INNER JOIN fmp_data.symbol_status ss ON t.symbol = ss.symbol
                        WHERE {date_comparison}
                          AND ss.is_active = TRUE
                          AND ss.manually_deactivated = FALSE
                          AND {exchange_condition}
                    """
                else:
                    # symbol以外のcount_columnの場合は市場フィルタなし
                    query = f"""
                        SELECT COUNT(DISTINCT t.{count_column})
                        FROM {schema}.{table} t
                        WHERE {date_comparison}
                    """

                cursor.execute(query, (expected_date,))
                actual_count = cursor.fetchone()[0]
                result.actual_count = actual_count

            # 決算情報テーブルの特別処理
            if is_earnings_statement:
                # テーブル名から決算情報の種類を判定
                statement_type = self._get_statement_type_from_table(table, is_ttm=False)

                if statement_type:
                    # 本日処理された銘柄数を期待値として取得
                    result.expected_count = self.get_earnings_processed_count(
                        market=market,
                        statement_type=statement_type,
                        reference_date=reference_date
                    )
                    result.active_symbols_count = None
            elif is_ttm_statement:
                # TTMテーブルの期待件数処理
                statement_type = self._get_statement_type_from_table(table, is_ttm=True)

                if statement_type:
                    # 本日処理された銘柄数を期待値として取得（元データの処理数）
                    result.expected_count = self.get_earnings_processed_count(
                        market=market,
                        statement_type=statement_type,
                        reference_date=reference_date
                    )
                    result.active_symbols_count = None
            else:
                # 通常のテーブルの期待件数計算
                if expected_min_rows:
                    # 固定の最低件数が設定されている場合
                    result.expected_count = expected_min_rows
                elif threshold_pct:
                    # アクティブ銘柄数ベースの場合
                    active_count = self.get_active_symbols_count(market)

                    # is_double_countフラグがある場合は母数を2倍にする
                    # (例: score_rankings_v2は日次と週次の両方を格納するため)
                    if is_double_count:
                        result.active_symbols_count = active_count * 2
                    else:
                        result.active_symbols_count = active_count

                    result.expected_count = int(result.active_symbols_count * (threshold_pct / 100.0))
                else:
                    # 期待件数なし（存在チェックのみ）
                    result.expected_count = None

            # ステータスを判定
            if require_tuesday_check:
                # 火曜日チェックロジック (FREDのforexデータ用)
                # 火曜日のみデータなしをエラーとし、他の曜日はOK
                weekday = (reference_date or datetime.now()).weekday()  # 0=月曜, 1=火曜, ...
                if actual_count > 0:
                    result.status = "ok"
                    result.message = f"{description}: データ正常 ({actual_count}件)"
                elif weekday == 1:  # 火曜日
                    result.status = "error"
                    result.message = f"{description}: 火曜日にデータなし ({actual_count}件) - FRED更新が必要"
                else:
                    # 月水木金土日はデータがなくてもOK (週次更新のため)
                    result.status = "ok"
                    result.message = f"{description}: データ取得スキップ (週次更新のため) ({actual_count}件)"
            elif is_earnings_statement or is_ttm_statement:
                # 決算情報テーブル・TTMテーブルの特別なステータス判定
                # 分母 (expected_count) は本日処理された銘柄数
                # 分子 (actual_count) は実際に取得できた銘柄数

                if result.expected_count == 0:
                    # 本日処理がない場合は正常（決算は毎日あるわけではない）
                    result.status = "ok"
                    result.message = f"{description}: 本日は処理なし (0/0件)"
                elif actual_count == 0:
                    # 処理はあったのに取得が0件の場合はエラー
                    result.status = "error"
                    result.message = f"{description}: 取得失敗 ({actual_count}/{result.expected_count}件)"
                elif actual_count >= result.expected_count * 0.8:
                    # 80%以上取得できていれば正常
                    coverage = (actual_count / result.expected_count) * 100
                    result.status = "ok"
                    result.message = f"{description}: 取得成功 ({actual_count}/{result.expected_count}件, {coverage:.1f}%)"
                elif actual_count >= result.expected_count * 0.5:
                    # 50%以上80%未満は警告
                    coverage = (actual_count / result.expected_count) * 100
                    result.status = "warning"
                    result.message = f"{description}: 取得率低下 ({actual_count}/{result.expected_count}件, {coverage:.1f}%)"
                else:
                    # 50%未満はエラー
                    coverage = (actual_count / result.expected_count) * 100
                    result.status = "error"
                    result.message = f"{description}: 取得率大幅低下 ({actual_count}/{result.expected_count}件, {coverage:.1f}%)"
            elif result.expected_count:
                # 通常のテーブルの判定ロジック
                if actual_count >= result.expected_count:
                    result.status = "ok"
                    result.message = f"{description}: データ正常 ({actual_count}件)"
                elif actual_count >= result.expected_count * 0.5:  # 50%以上あればwarning
                    result.status = "warning"
                    coverage = (actual_count / result.expected_count) * 100
                    result.message = f"{description}: データ不足 ({actual_count}/{result.expected_count}件, {coverage:.1f}%)"
                else:
                    result.status = "error"
                    coverage = (actual_count / result.expected_count) * 100 if result.expected_count > 0 else 0
                    result.message = f"{description}: データ大幅不足 ({actual_count}/{result.expected_count}件, {coverage:.1f}%)"
            else:
                # 期待件数なしの場合（存在チェックのみ）
                if actual_count > 0:
                    result.status = "ok"
                    result.message = f"{description}: データ存在確認 ({actual_count}件)"
                else:
                    result.status = "error"
                    result.message = f"{description}: データなし"

            result.details = {
                "description": description,
                "date_column": date_column,
                "count_column": count_column,
            }

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"テーブルチェックに失敗 ({schema}.{table}): {e}")
            result.status = "error"
            result.message = f"チェック失敗: {str(e)}"

        return result

    def monitor_tables(
        self,
        monitor_type: Literal["daily_us", "daily_jp", "weekly", "monthly"],
        reference_date: datetime = None
    ) -> MonitorResult:
        """
        テーブル群を監視

        Args:
            monitor_type: 監視タイプ
            reference_date: 基準日付

        Returns:
            MonitorResult
        """
        if reference_date is None:
            reference_date = datetime.now()

        # 監視タイプから市場と頻度を判定
        if monitor_type == "daily_us":
            market = "us"
            frequency = "daily"
        elif monitor_type == "daily_jp":
            market = "jp"
            frequency = "daily"
        elif monitor_type == "weekly":
            market = "us"  # 週次は両市場だが、USで代表
            frequency = "weekly"
        elif monitor_type == "monthly":
            market = "us"  # 月次も両市場だが、USで代表
            frequency = "monthly"
        else:
            raise ValueError(f"Unknown monitor_type: {monitor_type}")

        result = MonitorResult(
            monitor_type=monitor_type,
            execution_time=reference_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # 設定からテーブルを抽出してチェック
        for schema, tables in self.config.items():
            for table_name, table_config in tables.items():
                # 頻度が一致しないテーブルはスキップ
                if table_config.get("frequency") != frequency:
                    continue

                # 市場が一致するかチェック
                table_markets = table_config.get("markets", [])
                if market not in table_markets:
                    continue

                # テーブルチェック実行
                logger.info(f"チェック中: {schema}.{table_name} (market={market}, freq={frequency})")
                table_result = self.check_table(
                    schema=schema,
                    table=table_name,
                    config=table_config,
                    market=market,
                    reference_date=reference_date
                )
                result.add_table_result(table_result)

        # サマリー生成
        result.generate_summary()

        return result

    def get_tables_by_frequency(self, frequency: str) -> List[Dict]:
        """
        頻度別にテーブルリストを取得

        Args:
            frequency: 更新頻度

        Returns:
            テーブル情報のリスト
        """
        tables = []
        for schema, schema_tables in self.config.items():
            for table_name, table_config in schema_tables.items():
                if table_config.get("frequency") == frequency:
                    tables.append({
                        "schema": schema,
                        "table": table_name,
                        "config": table_config,
                    })
        return tables
