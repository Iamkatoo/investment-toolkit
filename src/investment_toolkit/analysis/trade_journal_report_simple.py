#!/usr/bin/env python3
"""
売買記録分析レポート（簡易版）
HTMLフォーマットエラーを回避した簡素版
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text, Engine
from typing import Dict, Any, Tuple
from investment_toolkit.utilities.config import get_connection
import os

def connect_to_database():
    """データベース接続"""
    engine = create_engine(f"postgresql://{os.getenv('DB_USER', 'HOME')}:@localhost:5432/investment")
    return engine

def get_trade_journal_data(engine: Engine) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """売買記録データを取得し、基本統計を計算"""
    query = text("""
        SELECT 
            id, symbol, buy_date, buy_price, qty, buy_reason_text,
            buy_rsi, buy_sma20, buy_sma40, buy_macd_hist,
            stop_loss_price, take_profit_price,
            sell_date, sell_price, sell_reason_text,
            total_score_at_buy, value_score_at_buy, momentum_score_at_buy,
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
            buy_price * qty as total_cost
        FROM user_data.trade_journal
        ORDER BY buy_date DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    
    # 基本統計の計算
    total_trades = len(df)
    holding_trades = len(df[df['sell_date'].isna()])
    sold_trades = len(df[df['sell_date'].notna()])
    total_investment = df['total_cost'].sum()
    total_profit_loss = df['profit_loss'].sum() if sold_trades > 0 else 0
    avg_profit_rate = df['profit_rate'].mean() if sold_trades > 0 else 0
    win_count = len(df[(df['sell_date'].notna()) & (df['profit_loss'] > 0)])
    loss_count = len(df[(df['sell_date'].notna()) & (df['profit_loss'] <= 0)])
    win_rate = (win_count / sold_trades * 100) if sold_trades > 0 else 0
    
    stats = {
        'total_trades': total_trades,
        'holding_trades': holding_trades,
        'sold_trades': sold_trades,
        'total_investment': total_investment,
        'total_profit_loss': total_profit_loss,
        'avg_profit_rate': avg_profit_rate,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate
    }
    
    return df, stats

def generate_simple_html(engine: Engine) -> str:
    """簡易HTMLレポート生成"""
    try:
        df, stats = get_trade_journal_data(engine)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保有銘柄
        holdings_df = df[df['sell_date'].isna()]
        holdings_html = ""
        for _, row in holdings_df.iterrows():
            holdings_html += f"""
            <tr>
                <td>{row['symbol']}</td>
                <td>{row['buy_date']}</td>
                <td>{row['buy_price']:.2f}</td>
                <td>{row['qty']}</td>
                <td>{row['total_cost']:.0f}</td>
                <td>{row['stop_loss_price'] if pd.notna(row['stop_loss_price']) else '-'}</td>
                <td>{row['take_profit_price'] if pd.notna(row['take_profit_price']) else '-'}</td>
                <td>{row['buy_reason_text'] or ''}</td>
            </tr>
            """
        
        # 売却済み銘柄（直近10件）
        sold_df = df[df['sell_date'].notna()].head(10)
        sold_html = ""
        for _, row in sold_df.iterrows():
            profit_class = "profit" if row['profit_loss'] > 0 else "loss"
            sold_html += f"""
            <tr>
                <td>{row['symbol']}</td>
                <td>{row['buy_date']}</td>
                <td>{row['sell_date']}</td>
                <td>{row['buy_price']:.2f}</td>
                <td>{row['sell_price']:.2f}</td>
                <td>{row['qty']}</td>
                <td class="{profit_class}">{row['profit_loss']:.0f}</td>
                <td class="{profit_class}">{row['profit_rate']:.1f}%</td>
            </tr>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="utf-8">
            <title>売買記録分析レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                .update-time {{ text-align: center; color: #666; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .profit {{ color: #27ae60; font-weight: bold; }}
                .loss {{ color: #e74c3c; font-weight: bold; }}
                .symbol {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 売買記録分析レポート</h1>
                <p class="update-time">最終更新: {current_time}</p>
                
                <h2>📈 パフォーマンスサマリー</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_trades']}</div>
                        <div class="stat-label">総取引数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['holding_trades']}</div>
                        <div class="stat-label">保有中</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['total_profit_loss']:,.0f}円</div>
                        <div class="stat-label">総損益</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats['win_rate']:.1f}%</div>
                        <div class="stat-label">勝率</div>
                    </div>
                </div>
                
                <h2>💼 保有中の銘柄 ({stats['holding_trades']}件)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>銘柄</th>
                            <th>購入日</th>
                            <th>価格</th>
                            <th>数量</th>
                            <th>投資額</th>
                            <th>損切</th>
                            <th>利確</th>
                            <th>理由</th>
                        </tr>
                    </thead>
                    <tbody>
                        {holdings_html}
                    </tbody>
                </table>
                
                <h2>💰 売却済み取引 (最新10件)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>銘柄</th>
                            <th>購入日</th>
                            <th>売却日</th>
                            <th>購入価格</th>
                            <th>売却価格</th>
                            <th>数量</th>
                            <th>損益</th>
                            <th>収益率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sold_html}
                    </tbody>
                </table>
                
                <h2>📊 統計情報</h2>
                <ul>
                    <li>総投資額: {stats['total_investment']:,.0f}円</li>
                    <li>平均収益率: {stats['avg_profit_rate']:.1f}%</li>
                    <li>勝ち取引: {stats['win_count']}件</li>
                    <li>負け取引: {stats['loss_count']}件</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
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
    html = generate_simple_html(engine)
    
    # テスト用出力
    test_output_path = Path(__file__).parent.parent.parent / "reports" / "trade_journal_simple.html"
    test_output_path.parent.mkdir(exist_ok=True)
    test_output_path.write_text(html, encoding="utf-8")
    print(f"簡易レポートを生成しました: {test_output_path}")
