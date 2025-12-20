#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日付計算モジュール

市場別・頻度別に期待される日付を計算する
- 米国株: 前日の日付でチェック(早朝実行のため)
- 日本株: 当日の日付でチェック(夕方実行のため)
- 週次: 最新の週末日付
- 月次: 最新の月末日付
"""

from datetime import datetime, timedelta
from typing import Literal


class DateCalculator:
    """市場・頻度別の期待日付を計算するクラス"""

    @staticmethod
    def get_expected_date(
        market: Literal["us", "jp"],
        frequency: Literal["daily", "weekly", "monthly"],
        reference_date: datetime = None
    ) -> str:
        """
        期待される日付を計算して返す

        Args:
            market: 市場 ("us" or "jp")
            frequency: 更新頻度 ("daily", "weekly", "monthly")
            reference_date: 基準日付 (Noneの場合は現在日時)

        Returns:
            期待される日付 (YYYY-MM-DD形式)
        """
        if reference_date is None:
            reference_date = datetime.now()

        if frequency == "daily":
            return DateCalculator._get_daily_date(market, reference_date)
        elif frequency == "weekly":
            return DateCalculator._get_weekly_date(reference_date)
        elif frequency == "monthly":
            return DateCalculator._get_monthly_date(reference_date)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")

    @staticmethod
    def _get_daily_date(market: str, reference_date: datetime) -> str:
        """
        日次データの期待日付を取得

        Args:
            market: 市場 ("us" or "jp")
            reference_date: 基準日付

        Returns:
            期待される日付 (YYYY-MM-DD形式)
        """
        if market == "us":
            # 米国株: 前日の日付 (早朝実行で前日データを取得)
            expected_date = reference_date - timedelta(days=1)
        elif market == "jp":
            # 日本株: 当日の日付 (夕方実行で当日データを取得)
            expected_date = reference_date
        else:
            raise ValueError(f"Unknown market: {market}")

        return expected_date.strftime("%Y-%m-%d")

    @staticmethod
    def _get_weekly_date(reference_date: datetime) -> str:
        """
        週次データの期待日付を取得 (直近の金曜日または土曜日)

        Args:
            reference_date: 基準日付

        Returns:
            期待される日付 (YYYY-MM-DD形式)
        """
        # 曜日を取得 (0=月曜, 4=金曜, 5=土曜, 6=日曜)
        weekday = reference_date.weekday()

        # 直近の週末(金曜または土曜)を計算
        if weekday == 4:  # 金曜日
            expected_date = reference_date
        elif weekday == 5:  # 土曜日
            expected_date = reference_date
        elif weekday == 6:  # 日曜日
            expected_date = reference_date - timedelta(days=1)  # 前日の土曜
        else:  # 月〜木曜日
            expected_date = reference_date - timedelta(days=weekday + 3)  # 前週の金曜

        return expected_date.strftime("%Y-%m-%d")

    @staticmethod
    def _get_monthly_date(reference_date: datetime) -> str:
        """
        月次データの期待日付を取得 (前月末)

        Args:
            reference_date: 基準日付

        Returns:
            期待される日付 (YYYY-MM-DD形式)
        """
        # 今月1日を計算
        first_day_of_current_month = reference_date.replace(day=1)

        # 前月末日を計算
        last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)

        return last_day_of_previous_month.strftime("%Y-%m-%d")

    @staticmethod
    def get_week_range(reference_date: datetime = None) -> tuple[str, str]:
        """
        週の範囲(月曜〜日曜)を取得

        Args:
            reference_date: 基準日付

        Returns:
            (week_start, week_end) のタプル (YYYY-MM-DD形式)
        """
        if reference_date is None:
            reference_date = datetime.now()

        weekday = reference_date.weekday()
        week_start = reference_date - timedelta(days=weekday)
        week_end = week_start + timedelta(days=6)

        return (
            week_start.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d")
        )

    @staticmethod
    def get_month_range(reference_date: datetime = None) -> tuple[str, str]:
        """
        月の範囲(1日〜月末)を取得

        Args:
            reference_date: 基準日付

        Returns:
            (month_start, month_end) のタプル (YYYY-MM-DD形式)
        """
        if reference_date is None:
            reference_date = datetime.now()

        # 月の最初の日
        month_start = reference_date.replace(day=1)

        # 翌月の1日を計算して、そこから1日引くことで月末を得る
        if reference_date.month == 12:
            next_month = reference_date.replace(year=reference_date.year + 1, month=1, day=1)
        else:
            next_month = reference_date.replace(month=reference_date.month + 1, day=1)

        month_end = next_month - timedelta(days=1)

        return (
            month_start.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d")
        )
