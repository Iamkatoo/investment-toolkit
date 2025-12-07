import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import re
import json

def camel_to_snake(name):
    """UpperCamelCase → lower_snake_case に変換"""
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

def convert_dataframe_columns(df):
    """DataFrame のカラム名を一括変換（キャメルケース → スネークケース）"""
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df

def fix_common_typos(df):
    """特定のカラム名のタイポを修正"""
    typo_mapping = {
        "filling_date": "filing_date",  # タイポ修正
        "ebitdaratio": "ebitda_ratio",
        "epsdiluted": "eps_diluted",
        "othertotal_stockholders_equity": "other_total_stockholders_equity",
        "other_investing_activites": "other_investing_activities",
        "net_cash_used_for_investing_activites": "net_cash_used_for_investing_activities",
        "other_financing_activites": "other_financing_activities",
    }
    df.rename(columns=typo_mapping, inplace=True)
    return df

def add_period_type(df, period):
    """period_type を自動追加する関数"""
    if period and "period_type" not in df.columns:
        df["period_type"] = period
    return df

def preprocess_dataframe(df, period=None):
    """カラム名の変換 + タイポ修正 + period_type 追加"""
    df = convert_dataframe_columns(df)  # キャメルケース → スネークケース
    df = fix_common_typos(df)  # タイポ修正
    df = add_period_type(df, period)  # period_type 追加
    
    # period が ''（空文字）なら NULL に変換
    if "period" in df.columns:
        df = df.copy()  # 明示的にコピーを作成
        df["period"] = df["period"].replace("", None)
    
    return df


def check_fmp_bandwidth_limit(response):
    """
    FMP API のレスポンスをチェックし、
    `Bandwidth Limit Reach` のエラーが発生したか判定する。

    パラメータ:
        response (str or dict): FMP API のレスポンス（JSON 文字列 または 辞書）

    戻り値:
        bool: エラーなら False, 正常なら True
    """
    try:
        # JSON 文字列の場合は辞書に変換
        if isinstance(response, str):
            response = json.loads(response)

        # "Error Message" キーがあり、Bandwidth制限エラーなら False を返す
        if "Error Message" in response:
            if "Bandwidth Limit Reach" in response["Error Message"] or "Too Many Requests" in response["Error Message"]:
                return False

    except json.JSONDecodeError:
        # JSON のパースに失敗した場合（想定外のレスポンス）、エラーとはみなさない
        pass

    # 正常なデータなら True を返す
    return True