#!/usr/bin/env python3
"""
売買記録分析レポート HTML生成モジュール
manage_trade_journal.pyの分析機能をHTML形式でレポート化
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# プロジェクトのルートディレクトリをPythonのパスに追加
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
except ImportError:
    print("❌ データベース設定の読み込み失敗")
    sys.exit(1)


def connect_to_database():
    """データベースに接続するための SQLAlchemy エンジンを取得"""
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    return engine


def get_trade_journal_data(engine: Engine) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """売買記録データを取得し、基本統計を計算"""
    query = text("""
        SELECT 
            id, symbol, buy_date, buy_price, qty, buy_reason_text,
            buy_rsi, buy_sma20, buy_sma40, buy_macd_hist,
            stop_loss_price, take_profit_price,
            sell_date, sell_price, sell_reason_text,
            value_score_at_buy, momentum_score_at_buy, total_score_at_buy,
            quality_score_at_buy, macro_sector_score_at_buy, growth_score_at_buy,
            per_score_at_buy, roic_score_at_buy, rsi_score_at_buy, macd_hist_score_at_buy,
            created_at, updated_at,
            -- 計算項目
            CASE WHEN sell_date IS NOT NULL 
                THEN (sell_price - buy_price) * qty 
                ELSE NULL 
            END as profit_loss,
            CASE WHEN sell_date IS NOT NULL 
                THEN (sell_price - buy_price) / buy_price * 100 
                ELSE NULL 
            END as profit_rate,
            CASE WHEN sell_date IS NOT NULL 
                THEN sell_date - buy_date 
                ELSE NULL 
            END as holding_days,
            buy_price * qty as total_cost
        FROM user_data.trade_journal
        ORDER BY buy_date DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    
    # 基本統計の計算
    stats = {
        'total_trades': len(df),
        'holding_trades': len(df[df['sell_date'].isna()]),
        'sold_trades': len(df[df['sell_date'].notna()]),
        'total_investment': df[df['sell_date'].isna()]['total_cost'].sum() if len(df[df['sell_date'].isna()]) > 0 else 0,
        'total_profit_loss': df[df['sell_date'].notna()]['profit_loss'].sum() if len(df[df['sell_date'].notna()]) > 0 else 0,
        'avg_profit_rate': df[df['sell_date'].notna()]['profit_rate'].mean() if len(df[df['sell_date'].notna()]) > 0 else 0,
        'win_count': len(df[(df['sell_date'].notna()) & (df['profit_loss'] > 0)]),
        'loss_count': len(df[(df['sell_date'].notna()) & (df['profit_loss'] <= 0)]),
    }
    
    if stats['sold_trades'] > 0:
        stats['win_rate'] = (stats['win_count'] / stats['sold_trades']) * 100
    else:
        stats['win_rate'] = 0
    
    return df, stats


def analyze_scores_data(engine: Engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """スコア別パフォーマンス分析のデータを取得"""
    
    # 総合スコア別分析
    total_score_query = text("""
        SELECT 
            CASE 
                WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                ELSE 'スコア無し'
            END as score_range,
            COUNT(*) as total_count,
            COUNT(CASE WHEN sell_date IS NOT NULL THEN 1 END) as sold_count,
            COALESCE(AVG(CASE WHEN sell_date IS NOT NULL 
                THEN (sell_price - buy_price) / buy_price * 100 END), 0) as avg_return_pct,
            COALESCE(COUNT(CASE WHEN sell_date IS NOT NULL AND (sell_price - buy_price) > 0 THEN 1 END) * 100.0 / 
                NULLIF(COUNT(CASE WHEN sell_date IS NOT NULL THEN 1 END), 0), 0) as win_rate_pct
        FROM user_data.trade_journal
        GROUP BY 
            CASE 
                WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                ELSE 'スコア無し'
            END
        ORDER BY 
            CASE 
                WHEN (CASE 
                    WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                    WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                    WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                    WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                    ELSE 'スコア無し'
                END) = '高スコア(≥0.8)' THEN 1
                WHEN (CASE 
                    WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                    WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                    WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                    WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                    ELSE 'スコア無し'
                END) = '中スコア(0.6-0.8)' THEN 2
                WHEN (CASE 
                    WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                    WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                    WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                    WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                    ELSE 'スコア無し'
                END) = '低スコア(0.4-0.6)' THEN 3
                WHEN (CASE 
                    WHEN total_score_at_buy >= 0.8 THEN '高スコア(≥0.8)'
                    WHEN total_score_at_buy >= 0.6 THEN '中スコア(0.6-0.8)'
                    WHEN total_score_at_buy >= 0.4 THEN '低スコア(0.4-0.6)'
                    WHEN total_score_at_buy IS NOT NULL THEN '最低スコア(<0.4)'
                    ELSE 'スコア無し'
                END) = '最低スコア(<0.4)' THEN 4
                ELSE 5
            END
    """)
    
    # バリュースコア別分析
    value_score_query = text("""
        SELECT 
            CASE 
                WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                ELSE 'スコア無し'
            END as value_range,
            COUNT(*) as total_count,
            COUNT(CASE WHEN sell_date IS NOT NULL THEN 1 END) as sold_count,
            COALESCE(AVG(CASE WHEN sell_date IS NOT NULL 
                THEN (sell_price - buy_price) / buy_price * 100 END), 0) as avg_return_pct,
            COALESCE(COUNT(CASE WHEN sell_date IS NOT NULL AND (sell_price - buy_price) > 0 THEN 1 END) * 100.0 / 
                NULLIF(COUNT(CASE WHEN sell_date IS NOT NULL THEN 1 END), 0), 0) as win_rate_pct
        FROM user_data.trade_journal
        GROUP BY 
            CASE 
                WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                ELSE 'スコア無し'
            END
        ORDER BY 
            CASE 
                WHEN (CASE 
                    WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                    WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                    WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                    WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                    ELSE 'スコア無し'
                END) = '高バリュー(≥0.8)' THEN 1
                WHEN (CASE 
                    WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                    WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                    WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                    WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                    ELSE 'スコア無し'
                END) = '中バリュー(0.6-0.8)' THEN 2
                WHEN (CASE 
                    WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                    WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                    WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                    WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                    ELSE 'スコア無し'
                END) = '低バリュー(0.4-0.6)' THEN 3
                WHEN (CASE 
                    WHEN value_score_at_buy >= 0.8 THEN '高バリュー(≥0.8)'
                    WHEN value_score_at_buy >= 0.6 THEN '中バリュー(0.6-0.8)'
                    WHEN value_score_at_buy >= 0.4 THEN '低バリュー(0.4-0.6)'
                    WHEN value_score_at_buy IS NOT NULL THEN '割高(<0.4)'
                    ELSE 'スコア無し'
                END) = '割高(<0.4)' THEN 4
                ELSE 5
            END
    """)
    
    # モメンタムスコア別分析
    momentum_score_query = text("""
        SELECT 
            CASE 
                WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                ELSE 'スコア無し'
            END as momentum_range,
            COUNT(*) as total_count,
            COUNT(CASE WHEN sell_date IS NOT NULL THEN 1 END) as sold_count,
            COALESCE(AVG(CASE WHEN sell_date IS NOT NULL 
                THEN (sell_price - buy_price) / buy_price * 100 END), 0) as avg_return_pct,
            COALESCE(AVG(CASE WHEN sell_date IS NOT NULL 
                THEN sell_date - buy_date END), 0) as avg_holding_days
        FROM user_data.trade_journal
        GROUP BY 
            CASE 
                WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                ELSE 'スコア無し'
            END
        ORDER BY 
            CASE 
                WHEN (CASE 
                    WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                    WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                    WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                    WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                    ELSE 'スコア無し'
                END) = '強いモメンタム(≥0.7)' THEN 1
                WHEN (CASE 
                    WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                    WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                    WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                    WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                    ELSE 'スコア無し'
                END) = '中モメンタム(0.5-0.7)' THEN 2
                WHEN (CASE 
                    WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                    WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                    WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                    WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                    ELSE 'スコア無し'
                END) = '弱いモメンタム(0.3-0.5)' THEN 3
                WHEN (CASE 
                    WHEN momentum_score_at_buy >= 0.7 THEN '強いモメンタム(≥0.7)'
                    WHEN momentum_score_at_buy >= 0.5 THEN '中モメンタム(0.5-0.7)'
                    WHEN momentum_score_at_buy >= 0.3 THEN '弱いモメンタム(0.3-0.5)'
                    WHEN momentum_score_at_buy IS NOT NULL THEN '下落トレンド(<0.3)'
                    ELSE 'スコア無し'
                END) = '下落トレンド(<0.3)' THEN 4
                ELSE 5
            END
    """)
    
    # 高スコア最優秀銘柄
    top_performers_query = text("""
        SELECT 
            symbol, buy_date, sell_date,
            (sell_price - buy_price) / buy_price * 100 as return_pct,
            total_score_at_buy, value_score_at_buy, momentum_score_at_buy
        FROM user_data.trade_journal
        WHERE sell_date IS NOT NULL AND total_score_at_buy >= 0.6
        ORDER BY (sell_price - buy_price) / buy_price DESC
        LIMIT 5
    """)
    
    with engine.connect() as conn:
        df_total_score = pd.read_sql_query(total_score_query, conn)
        df_value_score = pd.read_sql_query(value_score_query, conn)
        df_momentum_score = pd.read_sql_query(momentum_score_query, conn)
        df_top_performers = pd.read_sql_query(top_performers_query, conn)
    
    return df_total_score, df_value_score, df_momentum_score, df_top_performers


def generate_holdings_html(df: pd.DataFrame, stats: Dict[str, Any]) -> str:
    """保有中の銘柄HTML生成"""
    holdings_df = df[df['sell_date'].isna()].copy()
    
    if holdings_df.empty:
        holdings_table = "<p class='no-data'>保有中の銘柄はありません</p>"
    else:
        holdings_table = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>銘柄</th>
                    <th>購入日</th>
                    <th>価格</th>
                    <th>数量</th>
                    <th>投資額</th>
                    <th>損切</th>
                    <th>利確</th>
                    <th>総合スコア</th>
                    <th>理由</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for _, row in holdings_df.iterrows():
            total_cost = row['total_cost']
            total_score_display = f"{row['total_score_at_buy']:.3f}" if pd.notna(row['total_score_at_buy']) else '-'
            stop_loss_display = row['stop_loss_price'] if pd.notna(row['stop_loss_price']) else '-'
            take_profit_display = row['take_profit_price'] if pd.notna(row['take_profit_price']) else '-'
            buy_reason_display = row['buy_reason_text'] if pd.notna(row['buy_reason_text']) else '-'
            holdings_table += f"""
                <tr>
                    <td class="symbol">{row['symbol']}</td>
                    <td>{row['buy_date']}</td>
                    <td class="price">{row['buy_price']:.2f}</td>
                    <td class="quantity">{row['qty']}</td>
                    <td class="amount">{total_cost:,.0f}円</td>
                    <td class="price">{stop_loss_display}</td>
                    <td class="price">{take_profit_display}</td>
                    <td class="score">{total_score_display}</td>
                    <td class="reason">{buy_reason_display}</td>
                </tr>
            """
        
        holdings_table += f"""
            </tbody>
        </table>
        <div class="summary-stats">
            <strong>保有銘柄数:</strong> {len(holdings_df)}銘柄 | 
            <strong>総投資額:</strong> {stats['total_investment']:,.0f}円
        </div>
        """
    
    return holdings_table


def generate_performance_summary_html(stats: Dict[str, Any]) -> str:
    """パフォーマンスサマリーHTML生成"""
    return f"""
    <div class="summary-cards">
        <div class="summary-card">
            <h4>📊 取引統計</h4>
            <div class="stat-item">
                <span class="stat-label">総取引数:</span>
                <span class="stat-value">{stats['total_trades']}件</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">売却済み:</span>
                <span class="stat-value">{stats['sold_trades']}件</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">保有中:</span>
                <span class="stat-value">{stats['holding_trades']}件</span>
            </div>
        </div>
        
        <div class="summary-card">
            <h4>💰 損益統計</h4>
            <div class="stat-item">
                <span class="stat-label">総損益:</span>
                <span class="stat-value {'profit' if stats['total_profit_loss'] > 0 else 'loss'}">{stats['total_profit_loss']:,.0f}円</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">平均収益率:</span>
                <span class="stat-value {'profit' if stats['avg_profit_rate'] > 0 else 'loss'}">{stats['avg_profit_rate']:.1f}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">勝率:</span>
                <span class="stat-value">{stats['win_rate']:.1f}%</span>
            </div>
        </div>
        
        <div class="summary-card">
            <h4>📈 勝敗内訳</h4>
            <div class="stat-item">
                <span class="stat-label">勝ち:</span>
                <span class="stat-value profit">{stats['win_count']}件</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">負け:</span>
                <span class="stat-value loss">{stats['loss_count']}件</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">現在投資額:</span>
                <span class="stat-value">{stats['total_investment']:,.0f}円</span>
            </div>
        </div>
    </div>
    """


def generate_score_analysis_html(df_total: pd.DataFrame, df_value: pd.DataFrame, 
                                df_momentum: pd.DataFrame, df_top: pd.DataFrame) -> str:
    """スコア別分析HTML生成"""
    
    # 総合スコア別分析テーブル
    total_score_table = """
    <table class="data-table">
        <thead>
            <tr>
                <th>スコア範囲</th>
                <th>総取引</th>
                <th>売却済み</th>
                <th>平均収益率</th>
                <th>勝率</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_total.iterrows():
        score_class = 'high-score' if '高スコア' in row['score_range'] else ('mid-score' if '中スコア' in row['score_range'] else 'low-score')
        total_score_table += f"""
            <tr class="{score_class}">
                <td class="score-range">{row['score_range']}</td>
                <td>{row['total_count']}</td>
                <td>{row['sold_count']}</td>
                <td class="{'profit' if row['avg_return_pct'] > 0 else 'loss'}">{row['avg_return_pct']:.1f}%</td>
                <td>{row['win_rate_pct']:.1f}%</td>
            </tr>
        """
    
    total_score_table += "</tbody></table>"
    
    # バリュースコア別分析テーブル
    value_score_table = """
    <table class="data-table">
        <thead>
            <tr>
                <th>バリュー範囲</th>
                <th>総取引</th>
                <th>売却済み</th>
                <th>平均収益率</th>
                <th>勝率</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_value.iterrows():
        value_score_table += f"""
            <tr>
                <td class="score-range">{row['value_range']}</td>
                <td>{row['total_count']}</td>
                <td>{row['sold_count']}</td>
                <td class="{'profit' if row['avg_return_pct'] > 0 else 'loss'}">{row['avg_return_pct']:.1f}%</td>
                <td>{row['win_rate_pct']:.1f}%</td>
            </tr>
        """
    
    value_score_table += "</tbody></table>"
    
    # モメンタムスコア別分析テーブル
    momentum_score_table = """
    <table class="data-table">
        <thead>
            <tr>
                <th>モメンタム範囲</th>
                <th>総取引</th>
                <th>売却済み</th>
                <th>平均収益率</th>
                <th>平均保有日数</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_momentum.iterrows():
        momentum_score_table += f"""
            <tr>
                <td class="score-range">{row['momentum_range']}</td>
                <td>{row['total_count']}</td>
                <td>{row['sold_count']}</td>
                <td class="{'profit' if row['avg_return_pct'] > 0 else 'loss'}">{row['avg_return_pct']:.1f}%</td>
                <td>{row['avg_holding_days']:.0f}日</td>
            </tr>
        """
    
    momentum_score_table += "</tbody></table>"
    
    # 高スコア最優秀銘柄テーブル
    top_performers_table = """
    <table class="data-table">
        <thead>
            <tr>
                <th>銘柄</th>
                <th>購入日</th>
                <th>売却日</th>
                <th>収益率</th>
                <th>総合</th>
                <th>バリュー</th>
                <th>モメンタム</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_top.iterrows():
        total_score_display = f"{row['total_score_at_buy']:.3f}" if pd.notna(row['total_score_at_buy']) else 'N/A'
        value_score_display = f"{row['value_score_at_buy']:.3f}" if pd.notna(row['value_score_at_buy']) else 'N/A'
        momentum_score_display = f"{row['momentum_score_at_buy']:.3f}" if pd.notna(row['momentum_score_at_buy']) else 'N/A'
        top_performers_table += f"""
            <tr class="top-performer">
                <td class="symbol">{row['symbol']}</td>
                <td>{row['buy_date']}</td>
                <td>{row['sell_date']}</td>
                <td class="profit">{row['return_pct']:.1f}%</td>
                <td class="score">{total_score_display}</td>
                <td class="score">{value_score_display}</td>
                <td class="score">{momentum_score_display}</td>
            </tr>
        """
    
    top_performers_table += "</tbody></table>"
    
    return f"""
    <div class="score-analysis-section">
        <h3>📊 総合スコア別分析</h3>
        {total_score_table}
        
        <h3>💎 バリュースコア別分析</h3>
        {value_score_table}
        
        <h3>🚀 モメンタムスコア別分析</h3>
        {momentum_score_table}
        
        <h3>🏆 高スコア最優秀銘柄 TOP5</h3>
        {top_performers_table}
    </div>
    """


def generate_trade_journal_html(engine: Engine) -> str:
    """売買記録分析レポートのメインHTML生成"""
    try:
        # データ取得
        df, stats = get_trade_journal_data(engine)
        df_total, df_value, df_momentum, df_top = analyze_scores_data(engine)
        
        # 各セクションのHTML生成
        performance_html = generate_performance_summary_html(stats)
        holdings_html = generate_holdings_html(df, stats)
        score_analysis_html = generate_score_analysis_html(df_total, df_value, df_momentum, df_top)
        
        # 現在時刻
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 完全なHTMLを生成（CSSスタイルを別途定義）
        css_styles = """
                body {
                    font-family: 'Helvetica Neue', Arial, 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', Meiryo, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    line-height: 1.6;
                    color: #2c3e50;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 10px;
                    font-size: 2.2em;
                }
                .update-time {
                    text-align: center;
                    color: #7f8c8d;
                    margin-bottom: 30px;
                    font-size: 0.9em;
                }
                h2 {
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 40px;
                    margin-bottom: 20px;"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <title>売買記録分析レポート</title>
            <style>
                {css_styles}
                h3 {{
                    color: #2980b9;
                    margin-top: 30px;
                    margin-bottom: 15px;
                }}
                .summary-cards {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .summary-card h4 {{
                    margin-top: 0;
                    margin-bottom: 15px;
                    font-size: 1.1em;
                }}
                .stat-item {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                    padding: 5px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.2);
                }}
                .stat-label {{
                    font-weight: 500;
                }}
                .stat-value {{
                    font-weight: bold;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .data-table th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                .data-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .data-table tr:hover {{
                    background-color: #e3f2fd;
                }}
                .symbol {{
                    font-weight: bold;
                    color: #2980b9;
                }}
                .price {{
                    text-align: right;
                    font-family: monospace;
                }}
                .quantity {{
                    text-align: right;
                }}
                .amount {{
                    text-align: right;
                    font-weight: bold;
                }}
                .score {{
                    text-align: center;
                    font-family: monospace;
                }}
                .reason {{
                    font-size: 0.9em;
                    color: #7f8c8d;
                    max-width: 200px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }}
                .profit {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .loss {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .high-score {{
                    background-color: #d5f4e6;
                }}
                .mid-score {{
                    background-color: #fff3cd;
                }}
                .low-score {{
                    background-color: #f8d7da;
                }}
                .top-performer {{
                    background-color: #d1ecf1;
                }}
                .score-range {{
                    font-weight: bold;
                }}
                .summary-stats {{
                    margin-top: 15px;
                    padding: 10px;
                    background-color: #ecf0f1;
                    border-radius: 5px;
                    text-align: center;
                }}
                .no-data {{
                    text-align: center;
                    color: #7f8c8d;
                    font-style: italic;
                    padding: 20px;
                }}
                .score-analysis-section {{
                    margin-top: 30px;
                }}
                @media (max-width: 768px) {{
                    .container {{
                        padding: 15px;
                        margin: 10px;
                    }}
                    .summary-cards {{
                        grid-template-columns: 1fr;
                    }}
                    .data-table {{
                        font-size: 0.9em;
                    }}
                    h1 {{
                        font-size: 1.8em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 売買記録分析レポート</h1>
                <p class="update-time">最終更新: {current_time}</p>
                
                <h2>📈 パフォーマンスサマリー</h2>
                {performance_html}
                
                <h2>💼 保有中の銘柄</h2>
                {holdings_html}
                
                <h2>🎯 スコア別パフォーマンス分析</h2>
                {score_analysis_html}
                
                <div style="margin-top: 50px; text-align: center; font-size: 0.8em; color: #7f8c8d;">
                    <p>このレポートは投資判断時のスコアと実際のパフォーマンスを分析しています</p>
                    <p>スコアは purchase_date における backtest_results.daily_scores から取得されています</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        # エラー時のフォールバックHTML
        error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <title>売買記録分析レポート - エラー</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>📊 売買記録分析レポート</h1>
            <p>最終更新: {error_time}</p>
            <div class="error">
                <h3>⚠️ レポート生成エラー</h3>
                <p>売買記録データの取得に失敗しました。</p>
                <p>エラー詳細: {str(e)}</p>
                <p>データベース接続とuser_data.trade_journalテーブルの存在を確認してください。</p>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    # テスト実行
    engine = connect_to_database()
    html = generate_trade_journal_html(engine)
    
    # テスト用出力
    test_output_path = Path(__file__).parent.parent.parent / "reports" / "test_trade_journal.html"
    test_output_path.parent.mkdir(exist_ok=True)
    test_output_path.write_text(html, encoding="utf-8")
    print(f"テスト用レポートを生成しました: {test_output_path}")