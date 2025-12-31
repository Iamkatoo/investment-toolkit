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
            if market == "us":
                exchange_condition = "exchange IN ('NASDAQ', 'NYSE', 'NASDAQ Global Select', 'New York Stock Exchange', 'NASDAQ Stock Exchange')"
            elif market == "jp":
                exchange_condition = "exchange = 'Tokyo'"
            else:
                raise ValueError(f"Unknown market: {market}")

            query = f"""
                SELECT COUNT(*)
                FROM fmp_data.symbol_status
                WHERE is_active = TRUE
                  AND manually_deactivated = FALSE
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

    def get_pending_earnings_count(self, market: str, statement_type: str, reference_date: datetime = None) -> int:
        """
        US市場の決算情報の期待件数を取得（earningsテーブルのpendingステータスから算出）

        Args:
            market: 市場 ("us" のみ対応)
            statement_type: 決算情報の種類 ("income", "balance", "cash")
            reference_date: 基準日付（デフォルトは本日）

        Returns:
            期待される決算情報の件数
        """
        if market != "us":
            # JP市場は対象外（別ロジックで処理）
            return 0

        if reference_date is None:
            reference_date = datetime.now()

        # ステータスカラム名のマッピング
        status_column_map = {
            "income": "income_status",
            "balance": "balance_status",
            "cash": "cash_status"
        }

        status_column = status_column_map.get(statement_type)
        if not status_column:
            logger.error(f"Unknown statement_type: {statement_type}")
            return 0

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 市場に応じてexchangeを判定
            if market == "us":
                exchange_condition = "exchange IN ('NASDAQ', 'NYSE', 'NASDAQ Global Select', 'New York Stock Exchange', 'NASDAQ Stock Exchange')"
            else:
                exchange_condition = "1=1"  # フォールバック

            # 決算カラムのタイムスタンプ取得
            timestamp_column_map = {
                "income": "income_completed_at",
                "balance": "balance_completed_at",
                "cash": "cash_completed_at"
            }
            timestamp_column = timestamp_column_map.get(statement_type, "income_completed_at")

            # 本日よりreport_dateが前で、かつ該当のstatusがpending
            # または過去7日以内にcompleteになったものを、
            # symbol_statusでアクティブかつ手動無効化されていない銘柄に限定してカウント
            query = f"""
                SELECT COUNT(DISTINCT e.symbol)
                FROM fmp_data.earnings e
                INNER JOIN fmp_data.symbol_status ss ON e.symbol = ss.symbol
                WHERE e.report_date < %s
                  AND (e.{status_column} = 'pending'
                       OR (e.{status_column} = 'complete'
                           AND e.{timestamp_column} >= CURRENT_DATE - INTERVAL '7 days'))
                  AND ss.is_active = TRUE
                  AND ss.manually_deactivated = FALSE
                  AND ss.{exchange_condition}
            """

            cursor.execute(query, (reference_date.date(),))
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            logger.debug(f"Pending {statement_type} statements for US market (active symbols only): {count}")
            return count

        except Exception as e:
            logger.error(f"Pending earnings count取得に失敗しました ({statement_type}): {e}")
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
        require_tuesday_check = config.get("require_tuesday_check", False)

        # 期待される日付を計算
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

            # 実際のレコード数を取得
            query = f"""
                SELECT COUNT(DISTINCT {count_column})
                FROM {schema}.{table}
                WHERE {date_column} = %s
            """

            cursor.execute(query, (expected_date,))
            actual_count = cursor.fetchone()[0]
            result.actual_count = actual_count

            # 決算情報テーブルの特別処理
            if is_earnings_statement:
                # テーブル名から決算情報の種類を判定
                statement_type = None
                if "income_statement" in table:
                    statement_type = "income"
                elif "balance_sheet" in table:
                    statement_type = "balance"
                elif "cash_flow" in table:
                    statement_type = "cash"

                if market == "us" and statement_type:
                    # US市場: earningsテーブルのpendingステータスから期待値を取得
                    result.expected_count = self.get_pending_earnings_count(
                        market=market,
                        statement_type=statement_type,
                        reference_date=reference_date
                    )
                    result.active_symbols_count = None
                elif market == "jp":
                    # JP市場: アクティブ銘柄数を期待値として表示
                    active_count = self.get_active_symbols_count(market)
                    result.expected_count = active_count
                    result.active_symbols_count = active_count
            else:
                # 通常のテーブルの期待件数計算
                if expected_min_rows:
                    # 固定の最低件数が設定されている場合
                    result.expected_count = expected_min_rows
                elif threshold_pct:
                    # アクティブ銘柄数ベースの場合
                    active_count = self.get_active_symbols_count(market)
                    result.active_symbols_count = active_count
                    result.expected_count = int(active_count * (threshold_pct / 100.0))
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
            elif is_earnings_statement:
                # 決算情報テーブルの特別なステータス判定
                if actual_count > 0:
                    result.status = "ok"
                    result.message = f"{description}: 今回取得 ({actual_count}/{result.expected_count}件)"
                else:
                    # 0件の場合
                    if market == "us":
                        result.status = "warning"
                        result.message = f"{description}: 今回取得 ({actual_count}/{result.expected_count}件)"
                    else:
                        result.status = "ok"
                        result.message = f"{description}: 今回取得 ({actual_count}/{result.expected_count}件)"
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
