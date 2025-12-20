#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ウォッチリストレポート生成モジュール
追跡中の銘柄のパフォーマンス分析と可視化を行う
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import json
import html
from sqlalchemy import text

from investment_toolkit.analysis.watchlist_manager import WatchlistManager
from investment_toolkit.analysis.score_analysis import get_stock_technical_data


def create_watchlist_performance_chart(watchlist_data: pd.DataFrame) -> go.Figure:
    """ウォッチリスト全体のパフォーマンス分析チャート"""
    
    if watchlist_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="ウォッチリストに銘柄がありません",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="ウォッチリストパフォーマンス")
        return fig
    
    # サブプロットを作成
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '価格変化率分布',
            'RSI変化分布', 
            '分析タイプ別パフォーマンス',
            '保有期間別リターン'
        ],
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. 価格変化率のヒストグラム
    fig.add_trace(
        go.Histogram(
            x=watchlist_data['price_change_pct'].dropna(),
            name='価格変化率',
            nbinsx=20,
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # 2. RSI変化のヒストグラム
    fig.add_trace(
        go.Histogram(
            x=watchlist_data['rsi_change'].dropna(),
            name='RSI変化',
            nbinsx=20,
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    
    # 3. 分析タイプ別平均パフォーマンス
    type_performance = watchlist_data.groupby('analysis_type')['price_change_pct'].agg([
        'mean', 'count', 'std'
    ]).reset_index()
    
    fig.add_trace(
        go.Bar(
            x=type_performance['analysis_type'],
            y=type_performance['mean'],
            name='平均リターン',
            marker_color='orange',
            error_y=dict(type='data', array=type_performance['std'])
        ),
        row=2, col=1
    )
    
    # 4. 保有期間とリターンの散布図
    fig.add_trace(
        go.Scatter(
            x=watchlist_data['days_since_added'],
            y=watchlist_data['price_change_pct'],
            mode='markers',
            text=watchlist_data['symbol'],
            name='銘柄パフォーマンス',
            marker=dict(
                size=8,
                color=watchlist_data['price_change_pct'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="リターン%")
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="ウォッチリストパフォーマンス分析",
        height=800,
        showlegend=False
    )
    
    # 軸ラベルの設定
    fig.update_xaxes(title_text="価格変化率(%)", row=1, col=1)
    fig.update_xaxes(title_text="RSI変化", row=1, col=2)
    fig.update_xaxes(title_text="分析タイプ", row=2, col=1)
    fig.update_xaxes(title_text="保有日数", row=2, col=2)
    
    fig.update_yaxes(title_text="頻度", row=1, col=1)
    fig.update_yaxes(title_text="頻度", row=1, col=2)
    fig.update_yaxes(title_text="平均リターン(%)", row=2, col=1)
    fig.update_yaxes(title_text="価格変化率(%)", row=2, col=2)
    
    return fig


def create_individual_stock_chart(engine, symbol: str, watchlist_info: Dict[str, Any]) -> go.Figure:
    """個別銘柄の追跡チャート"""
    
    added_date = watchlist_info['added_date']
    analysis_metadata = json.loads(watchlist_info['analysis_metadata']) if isinstance(watchlist_info['analysis_metadata'], str) else watchlist_info['analysis_metadata']
    
    # 追加日以降のテクニカルデータを取得
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (pd.to_datetime(added_date) - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        technical_data = get_stock_technical_data(engine, symbol, days_back=200)
        
        if technical_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"{symbol} のデータが取得できませんでした",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # 追加日以降のデータをフィルター
        technical_data['date'] = pd.to_datetime(technical_data['date'])
        mask = technical_data['date'] >= pd.to_datetime(added_date)
        post_add_data = technical_data[mask].copy()
        
        # サブプロット作成
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{symbol} 株価推移', 'RSI推移', 'ボリューム'],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # 1. 株価チャート
        fig.add_trace(
            go.Scatter(
                x=post_add_data['date'],
                y=post_add_data['close'],
                name='株価',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 追加時の価格ライン
        if 'price' in analysis_metadata:
            added_price = analysis_metadata['price']
            fig.add_hline(
                y=added_price,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text=f"追加時価格: {added_price:.2f}",
                row=1, col=1
            )
        
        # 追加日の垂直線
        fig.add_vline(
            x=pd.to_datetime(added_date),
            line=dict(color="red", width=2, dash="dash"),
            annotation_text="ウォッチリスト追加",
            row=1, col=1
        )
        
        # 2. RSIチャート
        if 'rsi_14' in post_add_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=post_add_data['date'],
                    y=post_add_data['rsi_14'],
                    name='RSI',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # RSI基準線
            fig.add_hline(y=30, line=dict(color="red", width=1, dash="dot"), row=2, col=1)
            fig.add_hline(y=70, line=dict(color="red", width=1, dash="dot"), row=2, col=1)
            
            # 追加時のRSI
            if 'rsi' in analysis_metadata:
                added_rsi = analysis_metadata['rsi']
                fig.add_hline(
                    y=added_rsi,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"追加時RSI: {added_rsi:.1f}",
                    row=2, col=1
                )
        
        # 3. ボリューム
        if 'volume' in post_add_data.columns:
            fig.add_trace(
                go.Bar(
                    x=post_add_data['date'],
                    y=post_add_data['volume'],
                    name='ボリューム',
                    marker_color='lightgray'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} ウォッチリスト追跡チャート',
            height=800,
            showlegend=False
        )
        
        # 軸ラベル
        fig.update_yaxes(title_text="株価", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="ボリューム", row=3, col=1)
        fig.update_xaxes(title_text="日付", row=3, col=1)
        
        return fig
        
    except Exception as e:
        print(f"チャート作成エラー {symbol}: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"チャート作成エラー: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig


def generate_watchlist_report_html(engine) -> str:
    """ウォッチリストレポートのHTMLを生成（軽量化版）"""
    
    # ウォッチリスト管理インスタンス
    wm = WatchlistManager(engine)
    
    # 現在のウォッチリストを取得（高速化済み）
    print("  ウォッチリストデータを取得中...")
    watchlist_data = wm.get_current_watchlist()
    
    if watchlist_data.empty:
        return generate_empty_watchlist_html()
    
    print(f"  ウォッチリストデータ取得完了: {len(watchlist_data)}件")
    
    # 軽量化: 重いパフォーマンス計算をスキップし、基本情報のみで生成
    try:
        # 基本統計のみ計算（高速）
        avg_return = watchlist_data['price_change_pct'].dropna().mean() if not watchlist_data['price_change_pct'].dropna().empty else 0
        positive_count = (watchlist_data['price_change_pct'] > 0).sum()
        avg_days = watchlist_data['days_since_added'].mean()
        
        print(f"  基本統計計算完了 - 平均リターン: {avg_return:.1f}%, プラス銘柄: {positive_count}件")
        
    except Exception as e:
        print(f"  統計計算エラー（デフォルト値使用）: {e}")
        avg_return = 0
        positive_count = 0
        avg_days = 0
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>ウォッチリスト追跡レポート（軽量版）</title>
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 20px;
            }}
            .header-info {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 10px;
                font-size: 0.9em;
                color: #666;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .summary-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .summary-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .watchlist-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 30px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .watchlist-table th,
            .watchlist-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            .watchlist-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
            }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #7f8c8d; }}
            .analysis-badge {{
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                margin-left: 10px;
            }}
            .performance-badge {{
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.9em;
                font-weight: bold;
            }}
            .performance-positive {{
                background-color: #d5f4e6;
                color: #27ae60;
            }}
            .performance-negative {{
                background-color: #ffeaa7;
                color: #e17055;
            }}
            .refresh-note {{
                background-color: #e8f5e8;
                border: 1px solid #27ae60;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
                text-align: center;
            }}
            .optimization-note {{
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
                text-align: center;
                color: #856404;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 ウォッチリスト追跡レポート（軽量版）</h1>
                <div class="header-info">
                    <span>最終更新: {current_time}</span>
                    <span>追跡中銘柄数: <strong>{len(watchlist_data)}</strong></span>
                </div>
            </div>
            
            <div class="optimization-note">
                <p><strong>⚡ 高速化モード</strong></p>
                <p>パフォーマンス向上のため、重い計算とグラフ生成をスキップして基本情報のみ表示しています。</p>
                <p>詳細な分析が必要な場合は、個別銘柄レポートをご利用ください。</p>
            </div>
            
            <!-- サマリーカード -->
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value">{len(watchlist_data)}</div>
                    <div class="summary-label">追跡中銘柄</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{avg_return:.1f}%</div>
                    <div class="summary-label">平均リターン</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{positive_count}</div>
                    <div class="summary-label">プラス銘柄</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{avg_days:.0f}日</div>
                    <div class="summary-label">平均保有期間</div>
                </div>
            </div>
            
            <!-- ウォッチリストテーブル -->
            <h2>📋 ウォッチリスト一覧</h2>
            <table class="watchlist-table">
                <thead>
                    <tr>
                        <th>銘柄</th>
                        <th>会社名</th>
                        <th>分析タイプ</th>
                        <th>追加日</th>
                        <th>保有期間</th>
                        <th>追加時価格</th>
                        <th>現在価格</th>
                        <th>価格変化</th>
                        <th>RSI変化</th>
                        <th>備考</th>
                    </tr>
                </thead>
                <tbody id="watchlist-tbody">
    """
    
    # 銘柄テーブル行を生成
    for _, row in watchlist_data.iterrows():
        try:
            symbol = row['symbol']
            company_name = row.get('company_name', 'N/A')
            analysis_type = row['analysis_type']
            added_date = row['added_date']
            
            # メタデータから追加時の情報を取得
            try:
                metadata = json.loads(row['analysis_metadata']) if isinstance(row['analysis_metadata'], str) else row['analysis_metadata']
            except:
                metadata = {}
            
            added_price = metadata.get('price', 0)
            current_price = row.get('current_price', 0)
            price_change_pct = row.get('price_change_pct', 0)
            rsi_change = row.get('rsi_change', 0)
            days_since_added = row.get('days_since_added', 0)
            
            # パフォーマンスに応じてクラスを設定
            price_class = 'positive' if price_change_pct and price_change_pct > 0 else 'negative' if price_change_pct and price_change_pct < 0 else 'neutral'
            rsi_class = 'positive' if rsi_change and rsi_change > 0 else 'negative' if rsi_change and rsi_change < 0 else 'neutral'
            
            # テーブル行を追加
            html_content += f"""
                    <tr>
                        <td><strong>{html.escape(str(symbol))}</strong></td>
                        <td>{html.escape(str(company_name))}</td>
                        <td><span class="analysis-badge">{html.escape(str(analysis_type))}</span></td>
                        <td>{added_date}</td>
                        <td>{days_since_added}日</td>
                        <td>{'%.2f' % added_price if added_price else 'N/A'}</td>
                        <td>{'%.2f' % current_price if current_price else 'N/A'}</td>
                        <td class="{price_class}">{'%+.1f%%' % price_change_pct if price_change_pct else 'N/A'}</td>
                        <td class="{rsi_class}">{'%+.1f' % rsi_change if rsi_change else 'N/A'}</td>
                        <td>-</td>
                    </tr>
            """
        except Exception as e:
            print(f"  行生成エラー（スキップ）: {symbol} - {e}")
            continue
    
    html_content += """
                </tbody>
            </table>
            
            <div class="refresh-note">
                <p><strong>💡 ヒント</strong></p>
                <p>このレポートは軽量化されています。詳細な分析やチャートは「スコア上位銘柄分析」レポートをご覧ください。</p>
                <p>データは日次で自動更新されます。手動更新が必要な場合は、レポート生成を再実行してください。</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    print("  ウォッチリストレポートHTML生成完了（軽量版）")
    return html_content


def generate_empty_watchlist_html() -> str:
    """空のウォッチリストの場合のHTML"""
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>ウォッチリスト追跡レポート</title>
        <style>
            body {
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .empty-container {
                background-color: white;
                padding: 60px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 600px;
            }
            .empty-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            .empty-title {
                font-size: 2em;
                color: #2c3e50;
                margin-bottom: 20px;
            }
            .empty-message {
                font-size: 1.1em;
                color: #7f8c8d;
                line-height: 1.6;
                margin-bottom: 30px;
            }
            .empty-actions {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .action-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
                font-size: 1em;
                transition: transform 0.2s;
            }
            .action-btn:hover {
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="empty-container">
            <div class="empty-icon">📋</div>
            <h1 class="empty-title">ウォッチリストが空です</h1>
            <p class="empty-message">
                まだウォッチリストに銘柄が追加されていません。<br>
                スコア分析レポートから気になる銘柄をチェックして、<br>
                ウォッチリストに追加してみましょう。
            </p>
            <div class="empty-actions">
                <a href="../graphs/top_stocks_analysis.html" class="action-btn">
                    📊 スコア上位銘柄を見る
                </a>
                <a href="../graphs/rsi35_below_analysis.html" class="action-btn">
                    🎯 RSI35以下銘柄を見る
                </a>
                <a href="../dashboard.html" class="action-btn">
                    🏠 ダッシュボードに戻る
                </a>
            </div>
        </div>
    </body>
    </html>
    """


def update_watchlist_performance_data(engine) -> bool:
    """ウォッチリストのパフォーマンスデータを更新"""
    try:
        wm = WatchlistManager(engine)
        result = wm.update_performance_tracking()
        return result
    except Exception as e:
        print(f"ウォッチリストパフォーマンス更新エラー: {e}")
        return False


def generate_dynamic_watchlist_html() -> str:
    """動的ウォッチリストレポートのHTMLを生成（空テンプレート + JavaScript動的生成）"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>ウォッチリスト追跡レポート（動的更新版）</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 20px;
            }}
            .header-info {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 10px;
                font-size: 0.9em;
                color: #666;
            }}
            .refresh-controls {{
                background-color: #e8f5e8;
                border: 2px solid #27ae60;
                border-radius: 8px;
                padding: 15px;
                margin: 20px 0;
                text-align: center;
            }}
            .refresh-btn {{
                background: linear-gradient(135deg, #27ae60 0%, #219a52 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 1em;
                margin: 0 5px;
                transition: transform 0.2s;
            }}
            .refresh-btn:hover {{
                transform: translateY(-2px);
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .summary-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .summary-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .watchlist-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 30px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .watchlist-table th,
            .watchlist-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            .watchlist-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
            }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #7f8c8d; }}
            .analysis-badge {{
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                margin-left: 10px;
            }}
            .loading {{
                text-align: center;
                padding: 40px;
                color: #666;
            }}
            .error {{
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
            }}
            .empty-message {{
                text-align: center;
                padding: 60px;
                color: #666;
                font-size: 1.2em;
            }}
            .remove-btn {{
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
                transition: background-color 0.2s;
            }}
            .remove-btn:hover {{
                background-color: #c82333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 ウォッチリスト追跡レポート（動的更新版）</h1>
                <div class="header-info">
                    <span>最終更新: {current_time}</span>
                    <span>追跡中銘柄数: <strong><span id="stock-count">-</span></strong></span>
                </div>
            </div>
            
            <!-- 更新コントロール -->
            <div class="refresh-controls">
                <button id="refresh-btn" class="refresh-btn" onclick="refreshWatchlist()">
                    🔄 ウォッチリストを更新
                </button>
                <button id="auto-refresh-btn" class="refresh-btn" onclick="toggleAutoRefresh()">
                    ⏰ 自動更新: OFF
                </button>
                <span style="margin-left: 20px; color: #666;">
                    💡 チェックボックスの変更は即座に反映されます
                </span>
            </div>
            
            <!-- サマリーカード -->
            <div id="summary-grid" class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value" id="total-stocks">-</div>
                    <div class="summary-label">追跡中銘柄</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="avg-return">-</div>
                    <div class="summary-label">平均リターン</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="positive-stocks">-</div>
                    <div class="summary-label">プラス銘柄</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value" id="avg-days">-</div>
                    <div class="summary-label">平均保有期間</div>
                </div>
            </div>
            
            <!-- ウォッチリストテーブル -->
            <h2>📋 ウォッチリスト一覧</h2>
            <div id="watchlist-container">
                <div class="loading">🔄 ウォッチリストを読み込み中...</div>
            </div>
        </div>
        
        <script>
            let autoRefreshInterval = null;
            let isAutoRefreshEnabled = false;
            
            // ページ読み込み時に初回データ取得
            document.addEventListener('DOMContentLoaded', function() {{
                refreshWatchlist();
            }});
            
            // ウォッチリストを更新
            async function refreshWatchlist() {{
                const refreshBtn = document.getElementById('refresh-btn');
                const container = document.getElementById('watchlist-container');
                
                // ボタンを無効化
                refreshBtn.disabled = true;
                refreshBtn.textContent = '🔄 更新中...';
                
                try {{
                    const response = await fetch('http://127.0.0.1:5001/api/watchlist');
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP エラー: ${{response.status}}`);
                    }}
                    
                    const data = await response.json();
                    
                    if (data.success) {{
                        updateWatchlistDisplay(data.data);
                        updateSummaryCards(data.data);
                        updateLastUpdateTime();
                    }} else {{
                        showError('ウォッチリストの取得に失敗しました: ' + (data.error || '不明なエラー'));
                    }}
                }} catch (error) {{
                    console.error('ウォッチリスト取得エラー:', error);
                    showError('APIサーバーに接続できません。手動でサーバーを起動してください: python start_watchlist_api.py');
                }} finally {{
                    // ボタンを再有効化
                    refreshBtn.disabled = false;
                    refreshBtn.textContent = '🔄 ウォッチリストを更新';
                }}
            }}
            
            // ウォッチリスト表示を更新
            function updateWatchlistDisplay(watchlistData) {{
                const container = document.getElementById('watchlist-container');
                
                if (!watchlistData || watchlistData.length === 0) {{
                    container.innerHTML = `
                        <div class="empty-message">
                            📋 ウォッチリストが空です<br>
                            <small>スコア分析レポートから銘柄を追加してください</small>
                        </div>
                    `;
                    return;
                }}
                
                let tableHTML = `
                    <table class="watchlist-table">
                        <thead>
                            <tr>
                                <th>銘柄</th>
                                <th>会社名</th>
                                <th>分析タイプ</th>
                                <th>追加日</th>
                                <th>追加時価格</th>
                                <th>現在価格</th>
                                <th>価格変化</th>
                                <th>保有期間</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                watchlistData.forEach(stock => {{
                    const priceChange = stock.price_change_pct || 0;
                    const priceClass = priceChange > 0 ? 'positive' : priceChange < 0 ? 'negative' : 'neutral';
                    const daysHeld = stock.days_since_added || 0;
                    
                    tableHTML += `
                        <tr>
                            <td><strong>${{stock.symbol}}</strong></td>
                            <td>${{stock.company_name || 'N/A'}}</td>
                            <td><span class="analysis-badge">${{stock.analysis_type}}</span></td>
                            <td>${{stock.added_date}}</td>
                            <td>${{(stock.added_price || 0).toFixed(2)}}</td>
                            <td>${{(stock.current_price || 0).toFixed(2)}}</td>
                            <td class="${{priceClass}}">${{priceChange > 0 ? '+' : ''}}${{priceChange.toFixed(1)}}%</td>
                            <td>${{daysHeld}}日</td>
                            <td>
                                <button class="remove-btn" onclick="removeFromWatchlist('${{stock.symbol}}', '${{stock.analysis_type}}')">
                                    🗑️ 削除
                                </button>
                            </td>
                        </tr>
                    `;
                }});
                
                tableHTML += `
                        </tbody>
                    </table>
                `;
                
                container.innerHTML = tableHTML;
            }}
            
            // サマリーカードを更新
            function updateSummaryCards(watchlistData) {{
                if (!watchlistData || watchlistData.length === 0) {{
                    document.getElementById('total-stocks').textContent = '0';
                    document.getElementById('avg-return').textContent = '-';
                    document.getElementById('positive-stocks').textContent = '0';
                    document.getElementById('avg-days').textContent = '-';
                    document.getElementById('stock-count').textContent = '0';
                    return;
                }}
                
                const totalStocks = watchlistData.length;
                const positiveStocks = watchlistData.filter(stock => (stock.price_change_pct || 0) > 0).length;
                const avgReturn = watchlistData.reduce((sum, stock) => sum + (stock.price_change_pct || 0), 0) / totalStocks;
                const avgDays = watchlistData.reduce((sum, stock) => sum + (stock.days_since_added || 0), 0) / totalStocks;
                
                document.getElementById('total-stocks').textContent = totalStocks;
                document.getElementById('avg-return').textContent = avgReturn.toFixed(1) + '%';
                document.getElementById('positive-stocks').textContent = positiveStocks;
                document.getElementById('avg-days').textContent = Math.round(avgDays) + '日';
                document.getElementById('stock-count').textContent = totalStocks;
            }}
            
            // ウォッチリストから削除
            async function removeFromWatchlist(symbol, analysisType) {{
                if (!confirm(`${{symbol}} をウォッチリストから削除しますか？`)) {{
                    return;
                }}
                
                try {{
                    const response = await fetch('http://127.0.0.1:5001/api/watchlist/remove', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            symbol: symbol,
                            analysis_type: analysisType,
                            reason: 'ユーザーによる手動削除'
                        }})
                    }});
                    
                    const data = await response.json();
                    
                    if (data.success) {{
                        showTemporaryMessage(`✅ ${{symbol}} をウォッチリストから削除しました`, 'success');
                        // 即座に表示を更新
                        setTimeout(refreshWatchlist, 500);
                    }} else {{
                        showTemporaryMessage(`❌ ${{symbol}} の削除に失敗しました`, 'error');
                    }}
                }} catch (error) {{
                    console.error('削除エラー:', error);
                    showTemporaryMessage(`❌ ${{symbol}} の削除でAPI接続エラー`, 'error');
                }}
            }}
            
            // 自動更新のON/OFF
            function toggleAutoRefresh() {{
                const btn = document.getElementById('auto-refresh-btn');
                
                if (isAutoRefreshEnabled) {{
                    // 自動更新を停止
                    clearInterval(autoRefreshInterval);
                    isAutoRefreshEnabled = false;
                    btn.textContent = '⏰ 自動更新: OFF';
                    btn.style.backgroundColor = '#667eea';
                }} else {{
                    // 自動更新を開始（30秒間隔）
                    autoRefreshInterval = setInterval(refreshWatchlist, 30000);
                    isAutoRefreshEnabled = true;
                    btn.textContent = '⏰ 自動更新: ON (30s)';
                    btn.style.backgroundColor = '#27ae60';
                }}
            }}
            
            // エラー表示
            function showError(message) {{
                const container = document.getElementById('watchlist-container');
                container.innerHTML = `
                    <div class="error">
                        <strong>❌ エラー:</strong> ${{message}}
                    </div>
                `;
            }}
            
            // 一時的なメッセージ表示
            function showTemporaryMessage(message, type = 'success') {{
                const messageDiv = document.createElement('div');
                messageDiv.textContent = message;
                messageDiv.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 15px;
                    border-radius: 5px;
                    color: white;
                    font-weight: bold;
                    z-index: 1000;
                    transition: all 0.3s ease;
                    ${{type === 'success' ? 'background-color: #28a745;' : 
                      type === 'error' ? 'background-color: #dc3545;' : 
                      'background-color: #ffc107; color: #212529;'}}
                `;
                
                document.body.appendChild(messageDiv);
                
                // 3秒後に削除
                setTimeout(() => {{
                    messageDiv.remove();
                }}, 3000);
            }}
            
            // 最終更新時刻を更新
            function updateLastUpdateTime() {{
                const now = new Date();
                const timeString = now.toLocaleString('ja-JP', {{
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                }});
                document.getElementById('last-update').textContent = `最終更新: ${{timeString}}`;
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content


def generate_mini_chart_watchlist_html(engine) -> str:
    """ミニチャート機能付きウォッチリストレポートのHTMLを生成"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # アクティブなウォッチリスト銘柄を取得
    try:
        query = text("""
            SELECT DISTINCT symbol 
            FROM watchlist.tracked_stocks 
            WHERE is_active = true
            ORDER BY symbol
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            symbols = [row[0] for row in result.fetchall()]
        
        if not symbols:
            return generate_empty_watchlist_html()
            
    except Exception as e:
        print(f"ウォッチリストデータ取得エラー: {e}")
        symbols = []
    
    # ウォッチリスト詳細データを取得（テーブル表示用）
    watchlist_data = pd.DataFrame()
    try:
        # ウォッチリスト管理インスタンス
        wm = WatchlistManager(engine)
        watchlist_data = wm.get_current_watchlist()
        print(f"ウォッチリストテーブル用データ取得: {len(watchlist_data)}件")
    except Exception as e:
        print(f"ウォッチリストテーブルデータ取得エラー: {e}")
    
    # 銘柄と会社名のマッピング辞書を作成（チャートヘッダー用）
    symbol_to_company = {}
    if not watchlist_data.empty:
        for _, row in watchlist_data.iterrows():
            symbol = row['symbol']
            company_name = row.get('company_name', 'N/A')
            symbol_to_company[symbol] = company_name
    
    # JSON形式でシンボル配列を作成
    symbols_json = json.dumps(symbols)
    
    # JSONデータを直接HTML内に埋め込む（file://制限回避）
    embedded_data = {}
    mini_json_dir = Path(__file__).parent.parent.parent.parent / "reports" / "mini_json"

    for symbol in symbols:
        json_file = mini_json_dir / f"{symbol}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    embedded_data[symbol] = json.load(f)
                print(f"JSONデータ埋め込み: {symbol}")
            except Exception as e:
                print(f"JSONデータ読み込みエラー ({symbol}): {e}")
        else:
            print(f"JSONファイル未発見: {json_file}")
    
    embedded_data_json = json.dumps(embedded_data, ensure_ascii=False, separators=(',', ':'))
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>ウォッチリスト</title>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <link rel="stylesheet" href="../static/mini.css">
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 20px;
            }}
            .header-info {{
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 10px;
                font-size: 0.9em;
                color: #666;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .summary-value {{
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .summary-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .watchlist-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 30px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .watchlist-table th,
            .watchlist-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: center;
            }}
            .watchlist-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
            }}
            .symbol-link {{
                color: #2c3e50;
                text-decoration: none;
                font-weight: bold;
                transition: color 0.2s;
            }}
            .symbol-link:hover {{
                color: #667eea;
                text-decoration: underline;
            }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #7f8c8d; }}
            .analysis-badge {{
                background-color: #3498db;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                margin-left: 10px;
            }}
            .btn {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8em;
                transition: all 0.2s;
            }}
            .btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .btn-sm {{
                padding: 4px 8px;
                font-size: 0.7em;
            }}
            .btn-danger {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }}
            .btn-danger:hover {{
                background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
            }}
            .info-panel {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: center;
            }}
            .chart-section {{
                margin: 30px 0;
            }}
            .section-title {{
                font-size: 1.3em;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 2px solid #667eea;
            }}
            .help-text {{
                background-color: #e8f4fd;
                border: 1px solid #bee5eb;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
                color: #0c5460;
            }}
            .symbol-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 15px;
                border-radius: 8px 8px 0 0;
                font-weight: bold;
                font-size: 1.1em;
            }}
            .symbol-header-text {{
                flex: 1;
            }}
            .symbol-header-actions {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .btn-mini {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8em;
                transition: all 0.2s;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }}
            .btn-mini:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .btn-mini.btn-danger {{
                background: rgba(231, 76, 60, 0.8);
                border-color: rgba(231, 76, 60, 0.9);
            }}
            .btn-mini.btn-danger:hover {{
                background: rgba(192, 57, 43, 0.9);
                border-color: rgba(192, 57, 43, 1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 ウォッチリスト</h1>
                <div class="header-info">
                    <span>最終更新: {current_time}</span>
                    <span>追跡中銘柄数: <strong>{len(symbols)}</strong></span>
                </div>
            </div>
    """
    
    # ウォッチリストテーブルを追加
    if not watchlist_data.empty:
        # 基本統計の計算
        avg_return = watchlist_data['price_change_pct'].dropna().mean() if not watchlist_data['price_change_pct'].dropna().empty else 0
        positive_count = (watchlist_data['price_change_pct'] > 0).sum()
        avg_days = watchlist_data['days_since_added'].mean()
        
        html_content += f"""
             <!-- サマリーカード -->
             <div class="summary-grid">
                 <div class="summary-card">
                     <div class="summary-value">{avg_return:.1f}%</div>
                     <div class="summary-label">平均リターン</div>
                 </div>
                 <div class="summary-card">
                     <div class="summary-value">{positive_count}</div>
                     <div class="summary-label">プラス銘柄</div>
                 </div>
             </div>
            
            <!-- ウォッチリストテーブル -->
            <h2 class="section-title">📋 ウォッチリスト一覧</h2>
            <table class="watchlist-table">
                <thead>
                    <tr>
                        <th>銘柄</th>
                        <th>会社名</th>
                        <th>分析タイプ</th>
                        <th>追加日</th>
                        <th>追加時価格</th>
                        <th>現在価格</th>
                        <th>価格変化</th>
                        <th>保有期間</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # 銘柄テーブル行を生成
        for _, row in watchlist_data.iterrows():
            try:
                symbol = row['symbol']
                company_name = row.get('company_name', 'N/A')
                analysis_type = row['analysis_type']
                added_date = row['added_date']
                
                # メタデータから追加時の情報を取得
                try:
                    metadata = json.loads(row['analysis_metadata']) if isinstance(row['analysis_metadata'], str) else row['analysis_metadata']
                except:
                    metadata = {}
                
                added_price = metadata.get('price', 0)
                current_price = row.get('current_price', 0)
                price_change_pct = row.get('price_change_pct', 0)
                days_since_added = row.get('days_since_added', 0)
                
                # パフォーマンスに応じてクラスを設定
                price_class = 'positive' if price_change_pct and price_change_pct > 0 else 'negative' if price_change_pct and price_change_pct < 0 else 'neutral'
                
                # テーブル行を追加（銘柄名をチャートへのリンクに）
                html_content += f"""
                        <tr>
                            <td><a href="#row-{symbol}" class="symbol-link">{html.escape(str(symbol))}</a></td>
                            <td>{html.escape(str(company_name))}</td>
                            <td><span class="analysis-badge">{html.escape(str(analysis_type))}</span></td>
                            <td>{added_date}</td>
                            <td>{'%.2f' % added_price if added_price else 'N/A'}</td>
                            <td>{'%.2f' % current_price if current_price else 'N/A'}</td>
                            <td class="{price_class}">{'%+.1f%%' % price_change_pct if price_change_pct else 'N/A'}</td>
                            <td>{days_since_added}日</td>
                            <td>
                                <button class="btn btn-sm" onclick="scrollToChart('{symbol}')">📈 チャート</button>
                                <button class="btn btn-sm btn-danger" onclick="removeFromWatchlist('{symbol}', '{analysis_type}')" style="margin-left: 5px;">🗑️ 削除</button>
                            </td>
                        </tr>
                """
            except Exception as e:
                print(f"  行生成エラー（スキップ）: {symbol} - {e}")
                continue
        
        html_content += """
                </tbody>
            </table>
        """
    
    html_content += f"""
            <div class="chart-section">
                <h2 class="section-title">📊 ミニチャート一覧</h2>
                <div class="mini-chart-grid" id="mini-chart-grid">
    """
    
    # 各銘柄のチャートコンテナを生成
    for symbol in symbols:
        # 銘柄名と会社名を組み合わせたヘッダーを作成
        company_name = symbol_to_company.get(symbol, 'N/A')
        
        # ウォッチリストデータから分析タイプを取得
        analysis_type_list = []
        if not watchlist_data.empty:
            symbol_rows = watchlist_data[watchlist_data['symbol'] == symbol]
            for _, row in symbol_rows.iterrows():
                analysis_type = row.get('analysis_type', '')
                if analysis_type and analysis_type not in analysis_type_list:
                    analysis_type_list.append(analysis_type)
        
        # ヘッダーテキスト作成
        if company_name != 'N/A':
            header_text = f"【{symbol}】{company_name}"
        else:
            header_text = f"【{symbol}】"
        
        # 分析タイプがある場合は追加（正式名称をそのまま使用）
        if analysis_type_list:
            header_text += f" [{'/'.join(analysis_type_list)}]"
        
        # 削除ボタン用の分析タイプを決定（最初の分析タイプを使用）
        primary_analysis_type = analysis_type_list[0] if analysis_type_list else 'unknown'
        
        html_content += f"""
                    <div class="watch-row" id="row-{symbol}">
                        <div class="symbol-header">
                            <div class="symbol-header-text">{html.escape(header_text)}</div>
                            <div class="symbol-header-actions">
                                <button class="btn-mini" onclick="scrollToChart('{symbol}')" title="このチャートにスクロール">
                                    📈 チャート
                                </button>
                                <button class="btn-mini btn-danger" onclick="removeFromWatchlist('{symbol}', '{primary_analysis_type}')" title="ウォッチリストから削除">
                                    🗑️ 削除
                                </button>
                            </div>
                        </div>
                        <div id="price-{symbol}" class="chart">
                            <div class="chart-loading">
                                <div class="loading-spinner"></div>
                                価格チャート読み込み中...
                            </div>
                        </div>
                        <div id="indic-{symbol}" class="chart">
                            <div class="chart-loading">
                                <div class="loading-spinner"></div>
                                指標チャート読み込み中...
                            </div>
                        </div>
                    </div>
        """
    
    html_content += f"""
                </div>
            </div>
            
            <div class="help-text">
                <strong>💡 チャートの見方:</strong><br>
                • <strong>価格チャート（左）:</strong> ローソク足で価格推移、青線=SMA20、オレンジ線=SMA40<br>
                • <strong>指標チャート（右）:</strong> 上部=RSI（0-100、70以上で買われ過ぎ、30以下で売られ過ぎ）、下部=MACDヒストグラム（緑=上昇モメンタム、赤=下降モメンタム）<br>
                • <strong>銘柄名をクリック</strong> または <strong>📈 チャートボタン</strong> で該当チャートにジャンプします<br>
                • <strong>🗑️ 削除ボタン</strong> でチャートを見ながら直接ウォッチリストから削除できます（確認ダイアログが表示されます）
            </div>
        </div>
        
        <script src="../static/mini_draw_embedded.js"></script>
        <script>
            // 銘柄リストをグローバル変数として設定
            const symbols = {symbols_json};
            
            // JSONデータを直接埋め込み（file://制限回避）
            const embeddedChartData = {embedded_data_json};
            
            // チャートスクロール機能
            function scrollToChart(symbol) {{
                const element = document.getElementById('row-' + symbol);
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    // 一時的にハイライト効果
                    element.style.boxShadow = '0 0 20px rgba(102, 126, 234, 0.5)';
                    element.style.transition = 'box-shadow 0.3s ease';
                    setTimeout(() => {{
                        element.style.boxShadow = '';
                    }}, 2000);
                }}
            }}
            
            // エラーハンドリング
            window.addEventListener('error', function(e) {{
                console.error('ページエラー:', e.error);
            }});
            
            // 個別リロード機能
            function reloadSymbol(symbol) {{
                loadMiniChart(symbol);
            }}
            
            // ウォッチリストから削除機能
            async function removeFromWatchlist(symbol, analysisType) {{
                if (!confirm(`銘柄 ${{symbol}} をウォッチリストから削除しますか？`)) {{
                    return;
                }}
                
                try {{
                    const response = await fetch('http://127.0.0.1:5001/api/watchlist/remove', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            symbol: symbol,
                            analysis_type: analysisType,
                            reason: 'ユーザーによる削除'
                        }})
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        alert(`${{symbol}} をウォッチリストから削除しました`);
                        // ページをリロードして最新状態を反映
                        window.location.reload();
                    }} else {{
                        alert(`削除に失敗しました: ${{result.message}}`);
                    }}
                }} catch (error) {{
                    console.error('削除エラー:', error);
                    alert('削除処理でエラーが発生しました。APIサーバーが起動しているか確認してください。');
                }}
            }}
            
            // デバッグ情報表示
            function showDebugInfo() {{
                console.log('=== デバッグ情報 ===');
                console.log('追跡銘柄数:', symbols.length);
                console.log('銘柄リスト:', symbols);
                console.log('JSON格納ディレクトリ:', 'mini_json/');
                console.log('パフォーマンス統計:', chartPerformance);
            }}
            
            // キーボードショートカット
            document.addEventListener('keydown', function(e) {{
                if (e.ctrlKey && e.key === 'r') {{
                    e.preventDefault();
                    refreshMiniCharts();
                }} else if (e.ctrlKey && e.key === 'd') {{
                    e.preventDefault();
                    showDebugInfo();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content 

if __name__ == "__main__":
    """メイン実行部分"""
    try:
        from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
        from sqlalchemy import create_engine
        
        print("🚀 ウォッチリストレポート生成開始")
        
        # データベース接続
        SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(SQLALCHEMY_DATABASE_URI)
        
        # ウォッチリストマネージャー初期化
        manager = WatchlistManager(engine)
        
        # ウォッチリストデータ取得
        print("📊 ウォッチリストデータ取得中...")
        watchlist_data = manager.get_current_watchlist()
        print(f"✅ ウォッチリストデータ取得完了: {len(watchlist_data)} 件")
        
        # レポート生成
        print("📝 HTMLレポート生成中...")
        html_content = generate_mini_chart_watchlist_html(engine)

        # ファイル保存
        from investment_toolkit.utilities.paths import get_or_create_reports_config
        _reports_config = get_or_create_reports_config()
        output_path = _reports_config.graphs_dir / "watchlist_report_mini.html"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        print(f"✅ レポート生成完了: {output_path}")
        print(f"📁 ファイルサイズ: {output_path.stat().st_size:,} バイト")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc() 