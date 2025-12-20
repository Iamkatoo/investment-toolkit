#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
監視結果を格納するデータクラス
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TableCheckResult:
    """テーブルチェック結果"""

    schema: str
    table: str
    frequency: str
    market: str
    expected_date: str
    actual_count: int
    expected_count: Optional[int] = None
    active_symbols_count: Optional[int] = None
    threshold_pct: Optional[float] = None
    status: str = "unknown"  # "ok", "warning", "error"
    message: str = ""
    details: Dict = field(default_factory=dict)

    @property
    def full_table_name(self) -> str:
        """スキーマ.テーブル名を返す"""
        return f"{self.schema}.{self.table}"

    @property
    def coverage_pct(self) -> Optional[float]:
        """カバレッジ率を返す (actual / expected * 100)"""
        if self.expected_count and self.expected_count > 0:
            return (self.actual_count / self.expected_count) * 100
        return None

    def is_healthy(self) -> bool:
        """健全性チェック"""
        return self.status == "ok"

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "schema": self.schema,
            "table": self.table,
            "full_table_name": self.full_table_name,
            "frequency": self.frequency,
            "market": self.market,
            "expected_date": self.expected_date,
            "actual_count": self.actual_count,
            "expected_count": self.expected_count,
            "active_symbols_count": self.active_symbols_count,
            "threshold_pct": self.threshold_pct,
            "coverage_pct": self.coverage_pct,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MonitorResult:
    """監視全体の結果"""

    monitor_type: str  # "daily_us", "daily_jp", "weekly", "monthly"
    execution_time: str
    table_results: List[TableCheckResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def add_table_result(self, result: TableCheckResult):
        """テーブルチェック結果を追加"""
        self.table_results.append(result)

    def add_error(self, error: str):
        """エラーを追加"""
        self.errors.append(error)

    def get_ok_count(self) -> int:
        """OKステータスの数を返す"""
        return sum(1 for r in self.table_results if r.status == "ok")

    def get_warning_count(self) -> int:
        """WARNINGステータスの数を返す"""
        return sum(1 for r in self.table_results if r.status == "warning")

    def get_error_count(self) -> int:
        """ERRORステータスの数を返す"""
        return sum(1 for r in self.table_results if r.status == "error")

    def is_all_healthy(self) -> bool:
        """全てが健全か"""
        return self.get_error_count() == 0 and len(self.errors) == 0

    def generate_summary(self):
        """サマリーを生成"""
        self.summary = {
            "total_tables": len(self.table_results),
            "ok": self.get_ok_count(),
            "warning": self.get_warning_count(),
            "error": self.get_error_count(),
            "has_errors": len(self.errors) > 0,
            "overall_health": "healthy" if self.is_all_healthy() else "unhealthy",
        }

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "monitor_type": self.monitor_type,
            "execution_time": self.execution_time,
            "summary": self.summary,
            "table_results": [r.to_dict() for r in self.table_results],
            "errors": self.errors,
        }

    def format_notification_message(self) -> str:
        """通知用メッセージを整形"""
        lines = [
            f"=== DB監視レポート ({self.monitor_type}) ===",
            f"実行時刻: {self.execution_time}",
            "",
            f"総合ステータス: {self.summary.get('overall_health', 'unknown').upper()}",
            f"テーブル数: {self.summary.get('total_tables', 0)}",
            f"  OK: {self.summary.get('ok', 0)}",
            f"  WARNING: {self.summary.get('warning', 0)}",
            f"  ERROR: {self.summary.get('error', 0)}",
        ]

        # 全テーブルの詳細を表示
        lines.append("")
        lines.append("【DETAILS】")
        for result in self.table_results:
            status_icon = {
                "ok": "✅",
                "warning": "⚠️",
                "error": "❌"
            }.get(result.status, "❓")

            # 件数表示（期待値があれば表示）
            if result.expected_count is not None:
                count_str = f"{result.actual_count}/{result.expected_count}"
            else:
                count_str = f"{result.actual_count}"

            lines.append(f"{status_icon} {result.full_table_name}: {count_str} {result.status.upper()}")

        # エラーテーブルの詳細メッセージ
        error_results = [r for r in self.table_results if r.status == "error"]
        if error_results:
            lines.append("")
            lines.append("【ERROR DETAILS】")
            for result in error_results:
                lines.append(f"  - {result.full_table_name}")
                lines.append(f"    期待日: {result.expected_date}")
                lines.append(f"    メッセージ: {result.message}")

        # ワーニングテーブルの詳細メッセージ
        warning_results = [r for r in self.table_results if r.status == "warning"]
        if warning_results:
            lines.append("")
            lines.append("【WARNING DETAILS】")
            for result in warning_results:
                lines.append(f"  - {result.full_table_name}")
                lines.append(f"    期待日: {result.expected_date}")
                lines.append(f"    メッセージ: {result.message}")

        # システムエラー
        if self.errors:
            lines.append("")
            lines.append("【システムエラー】")
            for error in self.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)
