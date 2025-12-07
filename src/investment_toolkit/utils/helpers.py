#!/usr/bin/env python
"""
ユーティリティ関数を提供するモジュール。
"""
import re
from typing import Optional


def normalize_name(s: Optional[str]) -> str:
    """
    セクターやインダストリー名を正規化します。
    
    Args:
        s: 正規化する文字列
        
    Returns:
        str: 正規化された文字列
        
    正規化プロセス:
    - None、空文字列、空白のみの文字列は 'Unclassified' に変換
    - 先頭と末尾の空白を削除
    - 連続する空白を1つの空白に置換
    - 単語の先頭文字を大文字にする（タイトルケース）
    """
    if not s or s.strip() == '':
        return 'Unclassified'
    
    # 空白のトリムと連続する空白の置換
    normalized = re.sub(r'\s+', ' ', s.strip())
    
    # タイトルケースに変換
    return normalized.title() 