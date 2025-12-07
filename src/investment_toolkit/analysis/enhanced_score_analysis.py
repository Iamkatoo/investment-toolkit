#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拡張スコア分析モジュール - ウォッチリスト機能付き
既存のscore_analysis.pyを拡張し、チェックボックス機能を追加
"""

import json
import html
import pandas as pd
from datetime import datetime, date
from sqlalchemy import text

# 元のスコア分析モジュールから必要な関数をインポート
from investment_toolkit.analysis.score_analysis import (
    get_stock_basic_info,
    get_stock_technical_data,
    get_stock_fundamental_data,
    get_stock_score_history,
    get_stock_weekly_data,
    get_stock_financial_metrics,
    get_sector_comparison_data,
    create_enhanced_stock_detail_chart,
    create_basic_fallback_chart,
    generate_investment_recommendation,
    analyze_score_components
)


def add_watchlist_javascript() -> str:
    """ウォッチリスト機能用のJavaScript"""
    return """
    <script>
        // 選択された銘柄を追跡
        let selectedStocks = new Set();
        
        // チェックボックスの変更を処理
        function toggleWatchlistSelection(checkbox) {
            const symbol = checkbox.dataset.symbol;
            const analysisType = checkbox.dataset.analysisType;
            const metadata = JSON.parse(checkbox.dataset.metadata);
            
            if (checkbox.checked) {
                selectedStocks.add({
                    symbol: symbol,
                    analysisType: analysisType,
                    metadata: metadata
                });
            } else {
                // セットから削除
                selectedStocks.forEach(stock => {
                    if (stock.symbol === symbol) {
                        selectedStocks.delete(stock);
                    }
                });
            }
            
            updateButtonStates();
            updateSelectionCounter();
        }
        
        // ボタンの状態を更新
        function updateButtonStates() {
            const addButton = document.getElementById('addToWatchlistBtn');
            const clearButton = document.getElementById('clearSelectionBtn');
            
            if (addButton) {
                addButton.disabled = selectedStocks.size === 0;
            }
            if (clearButton) {
                clearButton.disabled = selectedStocks.size === 0;
            }
        }
        
        // 選択カウンターを更新
        function updateSelectionCounter() {
            const counter = document.getElementById('selectionCounter');
            if (counter) {
                counter.textContent = `${selectedStocks.size}銘柄選択中`;
            }
        }
        
        // ウォッチリストに追加
        function addToWatchlist() {
            if (selectedStocks.size === 0) {
                alert('銘柄が選択されていません');
                return;
            }
            
            const stocksArray = Array.from(selectedStocks);
            console.log('ウォッチリストに追加:', stocksArray);
            
            // 実際のAPI呼び出し
            fetch('http://127.0.0.1:5001/api/watchlist/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(stocksArray)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`✅ ${data.success_count}銘柄をウォッチリストに追加しました！`);
                    if (data.failure_count > 0) {
                        console.log('追加に失敗した銘柄:', data.errors);
                    }
                } else {
                    alert('❌ ウォッチリストへの追加に失敗しました: ' + (data.error || '不明なエラー'));
                }
            })
            .catch(error => {
                console.error('API呼び出しエラー:', error);
            })
            .finally(() => {
                // 選択をクリア
                clearSelection();
            });
        }
        
        // 選択をクリア
        function clearSelection() {
            selectedStocks.clear();
            
            // 全チェックボックスのチェックを外す
            document.querySelectorAll('input[type="checkbox"][data-symbol]').forEach(checkbox => {
                checkbox.checked = false;
            });
            
            updateButtonStates();
            updateSelectionCounter();
        }
        
        // ウォッチリスト表示
        function showWatchlist() {
            // 新しいタブでウォッチリストレポートを開く
            window.open('watchlist_report.html', '_blank');
        }
        
        // ページ読み込み時の初期化
        document.addEventListener('DOMContentLoaded', function() {
            updateButtonStates();
            updateSelectionCounter();
            
            // イベントリスナーを設定
            const addButton = document.getElementById('addToWatchlistBtn');
            const clearButton = document.getElementById('clearSelectionBtn');
            const showButton = document.getElementById('showWatchlistBtn');
            
            if (addButton) {
                addButton.addEventListener('click', addToWatchlist);
            }
            if (clearButton) {
                clearButton.addEventListener('click', clearSelection);
            }
            if (showButton) {
                showButton.addEventListener('click', showWatchlist);
            }
        });
    </script>
    """


def add_watchlist_css() -> str:
    """ウォッチリスト機能用のCSS"""
    return """
    <style>
        /* ウォッチリストコントロール */
        .watchlist-controls {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: white;
            text-align: center;
        }
        
        .watchlist-toolbar {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .btn-watchlist {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
        }
        
        .btn-primary {
            background-color: #28a745;
            color: white;
        }
        
        .btn-primary:hover:not(:disabled) {
            background-color: #218838;
            transform: translateY(-1px);
        }
        
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover:not(:disabled) {
            background-color: #545b62;
        }
        
        .btn-watchlist:disabled {
            background-color: #ccc;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        /* チェックボックスのスタイル改善 */
        .watchlist-checkbox-cell {
            width: 60px;
            text-align: center;
            padding: 8px 4px !important;
            vertical-align: middle;
        }
        
        .watchlist-checkbox {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }
        
        .watchlist-checkbox input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
            margin: 0;
        }
        
        .watchlist-checkbox label {
            font-size: 10px;
            color: #666;
            text-align: center;
            line-height: 1.1;
            margin: 0;
            cursor: pointer;
            width: 50px;
            word-wrap: break-word;
            hyphens: auto;
        }
        
        /* サマリーテーブルの調整 */
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        
        .summary-table th,
        .summary-table td {
            border: 1px solid #ddd;
            padding: 8px 4px;
            text-align: center;
            vertical-align: middle;
        }
        
        .summary-table th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
        
        /* ランキング表示の改善 */
        .rank-1 { background-color: #ffd700; }
        .rank-2 { background-color: #c0c0c0; }
        .rank-3 { background-color: #cd7f32; }
        
        /* 詳細カードのチェックボックス */
        .stock-card .watchlist-checkbox {
            flex-direction: row;
            gap: 8px;
        }
        
        .stock-card .watchlist-checkbox label {
            font-size: 12px;
            width: auto;
            white-space: nowrap;
        }
        
        /* レスポンシブ対応 */
        @media (max-width: 768px) {
            .watchlist-toolbar {
                flex-direction: column;
                gap: 10px;
            }
            
            .summary-table {
                font-size: 12px;
            }
            
            .summary-table th,
            .summary-table td {
                padding: 6px 2px;
            }
            
            .watchlist-checkbox label {
                font-size: 9px;
            }
        }
    </style>
    """


def serialize_metadata_safely(metadata):
    """
    日付オブジェクトを含むメタデータを安全にJSONシリアライズするためのヘルパー関数
    """
    if metadata is None:
        return {}
    
    def convert_dates(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dates(item) for item in obj]
        else:
            return obj
    
    return convert_dates(metadata)


def generate_watchlist_checkbox(symbol, analysis_type, metadata=None):
    """ウォッチリスト追加用のチェックボックスHTMLを生成"""
    # メタデータを安全にシリアライズ
    safe_metadata = serialize_metadata_safely(metadata)
    metadata_json = html.escape(json.dumps(safe_metadata))
    
    checkbox_html = f'''
        <div class="watchlist-checkbox">
            <input type="checkbox" 
                   id="watchlist_{symbol}" 
                   data-symbol="{symbol}" 
                   data-analysis-type="{analysis_type}"
                   data-metadata='{metadata_json}'
                   onchange="toggleWatchlistSelection(this)">
            <label for="watchlist_{symbol}">Watch</label>
        </div>
    '''
    return checkbox_html


def generate_watchlist_controls(analysis_type: str) -> str:
    """ウォッチリストコントロール部分のHTMLを生成"""
    return f"""
        <div class="watchlist-controls">
            <h3>📋 ウォッチリスト機能</h3>
            <p>気になる銘柄にチェックを入れて、ウォッチリストに追加できます</p>
            
            <div class="watchlist-toolbar">
                <div class="selection-info">
                    <span id="selectionCounter">0銘柄選択中</span>
                </div>
                
                <div class="watchlist-buttons">
                    <button id="addToWatchlistBtn" class="btn-watchlist btn-primary" disabled>
                        ➕ 追加
                    </button>
                    <button id="clearSelectionBtn" class="btn-watchlist btn-secondary" disabled>
                        🗑️ クリア
                    </button>
                    <button id="showWatchlistBtn" class="btn-watchlist btn-secondary">
                        👁️ 表示
                    </button>
                </div>
            </div>
        </div>
    """


def generate_enhanced_top_stocks_report(engine, target_date: str = None) -> str:
    """
    ウォッチリスト機能付きのスコア上位銘柄レポートを生成（完全版 - グラフ機能含む）
    """
    if target_date is None:
        # 最新の日付を取得
        query = "SELECT MAX(date) as max_date FROM backtest_results.daily_scores"
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            report_date = result.max_date
    else:
        report_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    analysis_type = "top_score_stocks"
    
    # 上位銘柄を取得
    top_stocks_query = text("""
    SELECT 
        symbol,
        date,
        total_score,
        value_score,
        growth_score,
        quality_score,
        momentum_score,
        macro_sector_score,
        per_score,
        fcf_yield_score,
        ev_ebitda_score,
        eps_cagr_score,
        revenue_cagr_score,
        growth_consistency_score,
        roic_score,
        roe_score,
        debt_equity_score,
        altman_z_score,
        piotroski_f_score,
        cash_conversion_score,
        golden_cross_score,
        rsi_score,
        macd_hist_score,
        vol_adj_momentum_score,
        relative_strength_score,
        tail_wind_score,
        sector_rotation_score
    FROM backtest_results.daily_scores
    WHERE date = :target_date
    ORDER BY total_score DESC
    LIMIT 10
    """)
    
    try:
        top_stocks = pd.read_sql(top_stocks_query, engine, params={"target_date": report_date})
    except Exception as e:
        print(f"スコアデータ取得エラー: {e}")
        return f"<html><body><h1>データ取得エラー: {e}</h1></body></html>"
    
    if top_stocks.empty:
        return "<html><body><h1>データが見つかりません</h1></body></html>"
    
    # HTMLの開始部分
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>スコア上位銘柄分析レポート（ウォッチリスト機能付き）</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        {add_watchlist_css()}
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 20px;
            }}
            .stock-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stock-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .stock-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .total-score {{
                font-size: 2em;
                font-weight: bold;
                color: #e74c3c;
            }}
            .score-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            .score-item {{
                text-align: center;
                padding: 10px;
                border-radius: 6px;
                background-color: #f8f9fa;
            }}
            .score-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .score-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .analysis-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 6px;
            }}
            .strengths {{
                color: #27ae60;
                font-weight: bold;
            }}
            .weaknesses {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .recommendation {{
                background-color: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendation.hold {{
                background-color: #fff3cd;
                border-left-color: #ffc107;
            }}
            .recommendation.sell {{
                background-color: #f8d7da;
                border-left-color: #dc3545;
            }}
            .basic-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 15px 0;
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 6px;
            }}
            .info-item {{
                display: flex;
                justify-content: space-between;
            }}
            .chart-container {{
                margin: 20px 0;
                height: 2300px;
                overflow: visible;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 スコア上位銘柄分析レポート</h1>
                <p>分析日: {report_date} | 上位{len(top_stocks)}銘柄</p>
            </div>
            
            {generate_watchlist_controls(analysis_type)}
            
            <!-- サマリーテーブル -->
            <h2>🏆 ランキング一覧</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>ウォッチ</th>
                        <th>順位</th>
                        <th>銘柄</th>
                        <th>総合スコア</th>
                        <th>Value</th>
                        <th>Growth</th>
                        <th>Quality</th>
                        <th>Momentum</th>
                        <th>Macro</th>
                        <th>推奨アクション</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # サマリーテーブルの行を追加
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        rank_class = f"rank-{i+1}" if i < 3 else ""
        
        # 基本情報を取得
        basic_info = get_stock_basic_info(engine, row['symbol'])
        
        # テクニカルデータを取得
        technical_data = get_stock_technical_data(engine, row['symbol'], days_back=30)
        
        # 投資判断を生成
        recommendation = generate_investment_recommendation(row, technical_data, basic_info)
        
        # ウォッチリスト用メタデータ
        watchlist_metadata = {
            'price': basic_info.get('current_price', 0),
            'rsi': technical_data.get('rsi_14', [0]).iloc[-1] if not technical_data.empty else 0,
            'score': row['total_score'],
            'analysis_date': report_date.isoformat(),  # 文字列として保存
            'rank': i + 1
        }
        
        # チェックボックスを生成
        checkbox_html = generate_watchlist_checkbox(row['symbol'], analysis_type, watchlist_metadata)
        
        html_content += f"""
                    <tr class="{rank_class}">
                        <td class="watchlist-checkbox-cell">{checkbox_html}</td>
                        <td>{i+1}</td>
                        <td><strong>{row['symbol']}</strong><br><small>{basic_info.get('company_name', 'N/A')}</small></td>
                        <td><strong>{row['total_score']:.1f}</strong></td>
                        <td>{row['value_score']:.1f}</td>
                        <td>{row['growth_score']:.1f}</td>
                        <td>{row['quality_score']:.1f}</td>
                        <td>{row['momentum_score']:.1f}</td>
                        <td>{row['macro_sector_score']:.1f}</td>
                        <td><span class="recommendation {recommendation['action'].lower()}">{recommendation['action']}</span></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
            
            <!-- 詳細分析セクション -->
            <h2>📈 詳細分析</h2>
    """
    
    # 各銘柄の詳細分析を追加
    for i, (_, row) in enumerate(top_stocks.iterrows()):
        symbol = row['symbol']
        
        # データ取得
        basic_info = get_stock_basic_info(engine, symbol)
        technical_data = get_stock_technical_data(engine, symbol, days_back=252)
        fundamental_data = get_stock_fundamental_data(engine, symbol, years_back=5)
        score_history = get_stock_score_history(engine, symbol, days_back=365)
        weekly_data = get_stock_weekly_data(engine, symbol, weeks_back=52)
        financial_metrics = get_stock_financial_metrics(engine, symbol, years_back=5)
        sector_comparison = get_sector_comparison_data(engine, symbol, days_back=252)
        
        # 詳細チャートを作成
        try:
            chart_fig = create_enhanced_stock_detail_chart(
                symbol, technical_data, row, weekly_data, financial_metrics,
                sector_comparison, technical_data, basic_info, score_history,
                fundamental_data, engine
            )
            chart_html = chart_fig.to_html(include_plotlyjs=False, div_id=f"chart_{symbol}")
        except Exception as e:
            print(f"チャート作成エラー {symbol}: {e}")
            # フォールバック用のシンプルなチャート
            try:
                fallback_chart = create_basic_fallback_chart(symbol, row, fundamental_data, technical_data, basic_info)
                chart_html = fallback_chart.to_html(include_plotlyjs=False, div_id=f"chart_{symbol}")
            except:
                chart_html = f"<p>チャートの作成に失敗しました: {symbol}</p>"
        
        # 投資判断とスコア分析
        recommendation = generate_investment_recommendation(row, technical_data, basic_info)
        score_analysis = analyze_score_components(row)
        
        # ウォッチリスト用メタデータ
        watchlist_metadata = {
            'price': basic_info.get('current_price', 0),
            'rsi': technical_data.get('rsi_14', [0]).iloc[-1] if not technical_data.empty else 0,
            'score': row['total_score'],
            'analysis_date': report_date.isoformat(),
            'rank': i + 1
        }
        
        # チェックボックス
        detail_checkbox = generate_watchlist_checkbox(symbol, analysis_type, watchlist_metadata)
        
        html_content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div>
                        <span class="stock-title">{symbol} - {basic_info.get('company_name', 'N/A')}</span>
                        {detail_checkbox}
                    </div>
                    <span class="total-score">{row['total_score']:.1f}</span>
                </div>
                
                <!-- 基本情報 -->
                <div class="basic-info">
                    <div class="info-item">
                        <span>現在価格:</span>
                        <strong>${basic_info.get('current_price', 'N/A')}</strong>
                    </div>
                    <div class="info-item">
                        <span>時価総額:</span>
                        <strong>${basic_info.get('market_cap', 'N/A'):,.0f}M</strong>
                    </div>
                    <div class="info-item">
                        <span>セクター:</span>
                        <strong>{basic_info.get('sector', 'N/A')}</strong>
                    </div>
                    <div class="info-item">
                        <span>業界:</span>
                        <strong>{basic_info.get('industry', 'N/A')}</strong>
                    </div>
                </div>
                
                <!-- スコア分析 -->
                <div class="analysis-section">
                    <h4>📊 スコア構成分析</h4>
                    <div class="score-grid">
                        <div class="score-item">
                            <div class="score-label">Value</div>
                            <div class="score-value">{row['value_score']:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Growth</div>
                            <div class="score-value">{row['growth_score']:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Quality</div>
                            <div class="score-value">{row['quality_score']:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Momentum</div>
                            <div class="score-value">{row['momentum_score']:.1f}</div>
                        </div>
                        <div class="score-item">
                            <div class="score-label">Macro</div>
                            <div class="score-value">{row['macro_sector_score']:.1f}</div>
                        </div>
                    </div>
                    
                    <p><span class="strengths">強み:</span> {score_analysis['strengths']}</p>
                    <p><span class="weaknesses">注意点:</span> {score_analysis['weaknesses']}</p>
                </div>
                
                <!-- 投資判断 -->
                <div class="recommendation {recommendation['action'].lower()}">
                    <h4>💡 投資判断: {recommendation['action']}</h4>
                    <p><strong>理由:</strong> {recommendation['reasoning']}</p>
                    <p><strong>リスクレベル:</strong> {recommendation.get('risk_level', 'N/A')}</p>
                    <p><strong>投資期間:</strong> {recommendation.get('time_horizon', 'N/A')}</p>
                    <p><strong>エントリー戦略:</strong> {recommendation.get('entry_strategy', 'N/A')}</p>
                    <p><strong>出口戦略:</strong> {recommendation.get('exit_strategy', 'N/A')}</p>
                </div>
                
                <!-- 詳細チャート -->
                <div class="chart-container">
                    {chart_html}
                </div>
            </div>
        """
    
    html_content += f"""
        </div>
        {add_watchlist_javascript()}
    </body>
    </html>
    """
    
    return html_content


def generate_enhanced_rsi35_report(engine, target_date: str = None) -> str:
    """
    ウォッチリスト機能付きのRSI35以下銘柄レポートを生成
    """
    if target_date is None:
        # 最新の日付を取得
        query = "SELECT MAX(date) as max_date FROM backtest_results.vw_daily_master"
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            report_date = result.max_date
    else:
        report_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    
    analysis_type = "rsi35_below"
    
    # RSI35以下の成長銘柄を取得
    rsi35_query = text("""
    SELECT 
        dm.symbol,
        dm.date,
        dm.close as current_price,
        dm.rsi_14,
        ds.total_score,
        ds.value_score,
        ds.growth_score,
        ds.quality_score,
        ds.momentum_score,
        ds.macro_sector_score,
        dm.eps_cagr_3y,
        dm.revenue_cagr_3y,
        dm.market_cap,
        cp.company_name,
        -- 軽量化：セクター・業界情報を簡素化
        COALESCE(cp.sector, 'N/A') as sector,
        COALESCE(cp.industry, 'N/A') as industry
    FROM backtest_results.vw_daily_master dm
    LEFT JOIN backtest_results.daily_scores ds ON dm.symbol = ds.symbol AND dm.date = ds.date
    LEFT JOIN fmp_data.company_profile cp ON dm.symbol = cp.symbol
    WHERE dm.date = :target_date
        AND dm.rsi_14 <= 35
        AND dm.rsi_14 > 20  -- 極端な値を除外
        AND ds.growth_score > 5  -- 成長性がある銘柄
        AND dm.market_cap > 1000000000  -- 時価総額1B以上
        AND dm.eps_cagr_3y > 0.05  -- EPS成長率5%以上
    ORDER BY ds.total_score DESC, dm.rsi_14 ASC
    LIMIT 20
    """)
    
    try:
        rsi35_stocks = pd.read_sql(rsi35_query, engine, params={"target_date": report_date})
    except Exception as e:
        print(f"RSI35データ取得エラー: {e}")
        return f"<html><body><h1>データ取得エラー: {e}</h1></body></html>"
    
    if rsi35_stocks.empty:
        return f"""
        <html>
        <body>
            <div style="text-align: center; padding: 50px;">
                <h1>📊 RSI35以下の成長銘柄レポート</h1>
                <p>分析日: {report_date}</p>
                <p>現在、条件に合致する銘柄はありません。</p>
                <p>（RSI ≤ 35、成長性スコア > 5、時価総額 > $1B、EPS成長率 > 5%</p>
            </div>
        </body>
        </html>
        """
    
    # HTMLの開始部分
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>RSI35以下の成長銘柄レポート（ウォッチリスト機能付き）</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        {add_watchlist_css()}
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e74c3c;
                padding-bottom: 20px;
            }}
            .alert-info {{
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
            }}
            .stock-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                padding: 20px;
                background-color: #fff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .stock-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .stock-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .rsi-value {{
                font-size: 2em;
                font-weight: bold;
                color: #e74c3c;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            .metric-item {{
                text-align: center;
                padding: 10px;
                border-radius: 6px;
                background-color: #f8f9fa;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .analysis-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 6px;
            }}
            .opportunity {{
                background-color: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 15px;
                margin: 15px 0;
            }}
            .risk {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 15px 0;
            }}
            .basic-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 15px 0;
                padding: 15px;
                background-color: #e3f2fd;
                border-radius: 6px;
            }}
            .info-item {{
                display: flex;
                justify-content: space-between;
            }}
            .chart-container {{
                margin: 20px 0;
                height: 800px;
                overflow: visible;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📉 RSI35以下の成長銘柄レポート</h1>
                <p>分析日: {report_date} | 発見銘柄数: {len(rsi35_stocks)}</p>
            </div>
            
            <div class="alert-info">
                <h4>🎯 逆張り投資機会の発見</h4>
                <p>RSI35以下の過売り状態にある成長銘柄を特定しました。これらの銘柄は短期的に売られすぎの可能性があり、
                成長性の高い企業であれば反発の機会となる可能性があります。</p>
                <p><strong>抽出条件:</strong> RSI ≤ 35、成長スコア > 5、時価総額 > $1B、EPS成長率 > 5%</p>
            </div>
            
            <!-- サマリーテーブル -->
            <h2>📋 発見銘柄一覧</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>ウォッチ</th>
                        <th>順位</th>
                        <th>銘柄</th>
                        <th>RSI</th>
                        <th>現在価格</th>
                        <th>総合スコア</th>
                        <th>成長性</th>
                        <th>EPS成長率</th>
                        <th>投資判断</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # サマリーテーブルの行を追加
    for i, (_, row) in enumerate(rsi35_stocks.iterrows()):
        
        # ウォッチリスト用メタデータ
        watchlist_metadata = {
            'price': row['current_price'],
            'rsi': row['rsi_14'],
            'score': row['total_score'],
            'analysis_date': report_date.isoformat(),
            'rank': i + 1,
            'growth_score': row['growth_score'],
            'eps_cagr_3y': row['eps_cagr_3y']
        }
        
        # チェックボックスを生成
        checkbox_html = generate_watchlist_checkbox(row['symbol'], analysis_type, watchlist_metadata)
        
        # 投資判断を簡単に生成
        if row['rsi_14'] < 25 and row['growth_score'] > 7:
            investment_judgment = "強い買い"
        elif row['rsi_14'] < 30 and row['growth_score'] > 5:
            investment_judgment = "買い"
        else:
            investment_judgment = "様子見"
        
        html_content += f"""
                    <tr>
                        <td class="watchlist-checkbox-cell">{checkbox_html}</td>
                        <td>{i+1}</td>
                        <td><strong>{row['symbol']}</strong><br><small>{row['company_name'] or 'N/A'}</small></td>
                        <td><strong style="color: #e74c3c;">{row['rsi_14']:.1f}</strong></td>
                        <td>${row['current_price']:.2f}</td>
                        <td>{row['total_score']:.1f}</td>
                        <td>{row['growth_score']:.1f}</td>
                        <td>{row['eps_cagr_3y']*100:.1f}%</td>
                        <td><span style="color: #27ae60; font-weight: bold;">{investment_judgment}</span></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
            
            <!-- 詳細分析セクション -->
            <h2>🔍 詳細分析</h2>
    """
    
    # 上位5銘柄の詳細分析を追加
    for i, (_, row) in enumerate(rsi35_stocks.head(5).iterrows()):
        symbol = row['symbol']
        
        # テクニカルデータを取得
        technical_data = get_stock_technical_data(engine, symbol, days_back=252)
        
        # ウォッチリスト用メタデータ
        watchlist_metadata = {
            'price': row['current_price'],
            'rsi': row['rsi_14'],
            'score': row['total_score'],
            'analysis_date': report_date.isoformat(),
            'rank': i + 1,
            'growth_score': row['growth_score'],
            'eps_cagr_3y': row['eps_cagr_3y']
        }
        
        # チェックボックス
        detail_checkbox = generate_watchlist_checkbox(symbol, analysis_type, watchlist_metadata)
        
        # 投資機会と リスクの分析
        opportunity_text = f"RSI {row['rsi_14']:.1f}の過売り状態で、成長スコア{row['growth_score']:.1f}の高成長企業"
        if row['eps_cagr_3y'] > 0.15:
            opportunity_text += f"。特にEPS成長率{row['eps_cagr_3y']*100:.1f}%と高い成長性を示している"
        
        risk_text = "短期的な業績悪化や市場全体の下落リスク。RSI35以下は一時的でない可能性も考慮が必要"
        
        # シンプルなチャートHTML（価格チャートのみ）
        if not technical_data.empty:
            chart_html = f"""
            <div id="chart_{symbol}" style="height: 400px;">
                <p>📈 チャート表示機能は実装中です。現在価格: ${row['current_price']:.2f}, RSI: {row['rsi_14']:.1f}</p>
            </div>
            """
        else:
            chart_html = f"<p>チャートデータが取得できませんでした: {symbol}</p>"
        
        html_content += f"""
            <div class="stock-card">
                <div class="stock-header">
                    <div>
                        <span class="stock-title">{symbol} - {row['company_name'] or 'N/A'}</span>
                        {detail_checkbox}
                    </div>
                    <span class="rsi-value">RSI {row['rsi_14']:.1f}</span>
                </div>
                
                <!-- 基本情報 -->
                <div class="basic-info">
                    <div class="info-item">
                        <span>現在価格:</span>
                        <strong>${row['current_price']:.2f}</strong>
                    </div>
                    <div class="info-item">
                        <span>時価総額:</span>
                        <strong>${row['market_cap']:,.0f}M</strong>
                    </div>
                    <div class="info-item">
                        <span>セクター:</span>
                        <strong>{row['sector'] or 'N/A'}</strong>
                    </div>
                    <div class="info-item">
                        <span>業界:</span>
                        <strong>{row['industry'] or 'N/A'}</strong>
                    </div>
                </div>
                
                <!-- 指標分析 -->
                <div class="analysis-section">
                    <h4>📊 主要指標</h4>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-label">RSI (14日)</div>
                            <div class="metric-value" style="color: #e74c3c;">{row['rsi_14']:.1f}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">総合スコア</div>
                            <div class="metric-value">{row['total_score']:.1f}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">成長スコア</div>
                            <div class="metric-value">{row['growth_score']:.1f}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">EPS成長率</div>
                            <div class="metric-value">{row['eps_cagr_3y']*100:.1f}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">売上成長率</div>
                            <div class="metric-value">{row['revenue_cagr_3y']*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                
                <!-- 投資機会 -->
                <div class="opportunity">
                    <h4>🎯 投資機会</h4>
                    <p>{opportunity_text}</p>
                </div>
                
                <!-- リスク -->
                <div class="risk">
                    <h4>⚠️ 注意点・リスク</h4>
                    <p>{risk_text}</p>
                </div>
                
                <!-- チャート -->
                <div class="chart-container">
                    {chart_html}
                </div>
            </div>
        """
    
    html_content += f"""
        </div>
        {add_watchlist_javascript()}
    </body>
    </html>
    """
    
    return html_content


def generate_no_rsi35_stocks_report(analysis_type: str) -> str:
    """RSI35以下の銘柄がない場合のレポート"""
    return f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>RSI35以下 買い候補分析レポート</title>
        {add_watchlist_css()}
    </head>
    <body>
        <div class="container">
            {generate_watchlist_controls(analysis_type)}
            <div class="header">
                <h1>🎯 RSI35以下 買い候補分析レポート</h1>
                <p>分析日: {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div style="text-align: center; padding: 40px; background-color: #fff3cd; border-radius: 8px;">
                <h2>📋 本日の該当銘柄</h2>
                <p><strong>🔍 該当銘柄数: 0銘柄</strong></p>
                <p>厳格なフィルタリング条件により、本日は投資候補銘柄が見つかりませんでした。</p>
            </div>
        </div>
        {add_watchlist_javascript()}
    </body>
    </html>
    """ 