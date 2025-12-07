#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日次株価上昇率ランキングレポート生成モジュール
最新の取引日での価格上昇率が高い銘柄をトップ10で表示
ウォッチリストと同様のミニチャート表示形式を採用
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any
import json
import html
from sqlalchemy import text


def fetch_daily_ranking_data(engine, market_type='US', limit=10):
    """最新日付での価格上昇率トップ銘柄を取得（市場別）
    
    Args:
        engine: データベースエンジン
        market_type: 'US' (米国株) または 'JP' (日本株)
        limit: 取得件数
    """
    
    # 市場フィルターの条件
    if market_type == 'JP':
        # 日本株: .T で終わるまたは4桁数字のみ
        market_filter = "(dp.symbol LIKE '%.T' OR (dp.symbol ~ '^[0-9]{4}$'))"
        market_name = "日本株"
    else:
        # 米国株: .T で終わらない、かつ4桁数字のみでない
        market_filter = "(dp.symbol NOT LIKE '%.T' AND NOT (dp.symbol ~ '^[0-9]{4}$'))"
        market_name = "米国株"
    
    query = text(f"""
    WITH latest_date AS (
        SELECT MAX(dp.date) as max_date
        FROM fmp_data.daily_prices dp
        WHERE dp.change_percent IS NOT NULL
        AND {market_filter}
    ),
    ranked_stocks AS (
        SELECT 
            dp.symbol,
            dp.date,
            dp.close,
            dp.change,
            dp.change_percent,
            dp.volume,
            COALESCE(cp.company_name, 'N/A') as company_name,
            COALESCE(gics.raw_sector, 'N/A') as sector,
            COALESCE(gics.raw_industry, 'N/A') as industry,
            COALESCE(bm.market_cap, cp.mkt_cap) as market_cap,
            COALESCE(cp.exchange, 'N/A') as exchange,
            '{market_type}' as market_type,
            -- テクニカル指標を追加
            ti.rsi_14,
            ti.macd_hist,
            ti.sma_20,
            ti.sma_40,
            ROW_NUMBER() OVER (ORDER BY dp.change_percent DESC) as rank
        FROM fmp_data.daily_prices dp
        INNER JOIN latest_date ld ON dp.date = ld.max_date
        LEFT JOIN (
            SELECT DISTINCT ON (symbol) 
                symbol, company_name, exchange, mkt_cap
            FROM fmp_data.company_profile
            ORDER BY symbol, date DESC
        ) cp ON dp.symbol = cp.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol)
                symbol, raw_sector, raw_industry
            FROM reference.company_gics
            ORDER BY symbol, updated_at DESC
        ) gics ON dp.symbol = gics.symbol
        LEFT JOIN (
            SELECT DISTINCT ON (symbol)
                symbol, market_cap
            FROM calculated_metrics.basic_metrics
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM calculated_metrics.basic_metrics)
                AND market_cap IS NOT NULL
                AND market_cap > 0
            ORDER BY symbol, as_of_date DESC
        ) bm ON dp.symbol = bm.symbol
        -- テクニカル指標テーブルをJOIN
        LEFT JOIN calculated_metrics.technical_indicators ti ON dp.symbol = ti.symbol AND dp.date = ti.date
        WHERE dp.change_percent IS NOT NULL
            AND dp.change_percent > 0
            AND dp.volume > 10000  -- 最低出来高条件
            AND dp.close > 1.0     -- 最低価格条件（ペニーストック除外）
            AND {market_filter}
        ORDER BY dp.change_percent DESC
    )
    SELECT 
        symbol,
        date,
        close,
        change,
        change_percent,
        volume,
        company_name,
        sector,
        industry,
        market_cap,
        exchange,
        market_type,
        rank,
        -- テクニカル指標を追加
        rsi_14,
        macd_hist,
        sma_20,
        sma_40
    FROM ranked_stocks
    WHERE rank <= :limit
    ORDER BY rank
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"limit": limit})
        data = result.fetchall()
        
        if not data:
            return pd.DataFrame()
        
        columns = [
            'symbol', 'date', 'close', 'change', 'change_percent',
            'volume', 'company_name', 'sector', 'industry', 
            'market_cap', 'exchange', 'market_type', 'rank',
            # テクニカル指標を追加
            'rsi_14', 'macd_hist', 'sma_20', 'sma_40'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        
        # デバッグ用: データの状況を確認
        print(f"  {market_name}取得された銘柄数: {len(df)}")
        market_cap_with_data = df['market_cap'].notna().sum()
        print(f"  時価総額データ有り: {market_cap_with_data}/{len(df)}件")
        
        for idx, row in df.iterrows():
            tech_indicators = []
            if pd.notna(row['rsi_14']):
                tech_indicators.append(f"RSI={row['rsi_14']:.1f}")
            if pd.notna(row['macd_hist']):
                tech_indicators.append(f"MACD={row['macd_hist']:.3f}")
            if pd.notna(row['sma_20']):
                tech_indicators.append(f"SMA20={row['sma_20']:.2f}")
            if pd.notna(row['sma_40']):
                tech_indicators.append(f"SMA40={row['sma_40']:.2f}")
            
            # 時価総額データの状況を表示
            market_cap_status = f"時価総額={format_large_number(row['market_cap'])}" if pd.notna(row['market_cap']) else "時価総額=N/A"
            
            tech_status = f" [{', '.join(tech_indicators)}]" if tech_indicators else " [テクニカル指標なし]"
            print(f"  {row['symbol']}: {row['company_name']} / {row['sector']} / {row['industry']} / {market_cap_status}{tech_status}")
            
        return df


def generate_mini_chart_data_for_ranking(engine, symbols: List[str], lookback_days: int = 60) -> Dict:
    """
    ランキング銘柄用のミニチャートデータを動的に生成
    
    Args:
        engine: データベースエンジン
        symbols: 銘柄コードリスト
        lookback_days: 何日分のデータを取得するか
        
    Returns:
        Dict: 銘柄ごとのチャートデータ
    """
    if not symbols:
        return {}
    
    since = date.today() - timedelta(days=lookback_days * 2)  # 休日補正で余裕を見る
    
    # 株価データとテクニカル指標を結合して取得
    query = text("""
        SELECT 
            p.symbol, p.date, p.open, p.high, p.low, p.close, p.volume,
            ti.sma_20, ti.sma_40, ti.rsi_14, ti.macd_hist
        FROM fmp_data.daily_prices p
        LEFT JOIN calculated_metrics.technical_indicators ti 
            ON p.symbol = ti.symbol AND p.date = ti.date
        WHERE p.symbol = ANY(:symbols) AND p.date >= :since
        ORDER BY p.symbol, p.date
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"symbols": symbols, "since": since})
        
        if df.empty:
            print("  ⚠️ ランキング銘柄のチャートデータが取得できませんでした")
            return {}
        
        df['date'] = pd.to_datetime(df['date'])
        chart_data = {}
        
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date').tail(lookback_days)
            
            if symbol_data.empty:
                print(f"  ⚠️ {symbol}: データなし")
                continue
            
            # チャートデータの形式に変換
            chart_data[symbol] = {
                'symbol': symbol,
                'date': symbol_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'open': symbol_data['open'].fillna(method='ffill').tolist(),
                'high': symbol_data['high'].fillna(method='ffill').tolist(), 
                'low': symbol_data['low'].fillna(method='ffill').tolist(),
                'close': symbol_data['close'].fillna(method='ffill').tolist(),
                'volume': symbol_data['volume'].fillna(0).tolist(),
                'sma20': symbol_data['sma_20'].tolist(),
                'sma40': symbol_data['sma_40'].tolist(),
                'rsi': symbol_data['rsi_14'].tolist(),
                'macd_hist': symbol_data['macd_hist'].tolist(),
                'watchlist_info': []  # ランキングではウォッチリスト情報は空
            }
            
            print(f"  ✅ {symbol}: {len(symbol_data)}日分のデータを生成")
        
        print(f"  📊 ランキング用チャートデータ生成完了: {len(chart_data)}/{len(symbols)}銘柄")
        return chart_data
        
    except Exception as e:
        print(f"  ⚠️ ランキング用チャートデータ生成エラー: {e}")
        return {}


def format_large_number(value):
    """大きな数値を読みやすい形式にフォーマット"""
    if pd.isna(value) or value == 0:
        return 'N/A'
    
    if value >= 1e12:
        return f'{value/1e12:.1f}T'
    elif value >= 1e9:
        return f'{value/1e9:.1f}B'
    elif value >= 1e6:
        return f'{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'{value/1e3:.1f}K'
    else:
        return f'{value:.0f}'


def get_sector_color(sector):
    """セクターごとの色を取得"""
    # セクター色マップ（グラデーション対応）
    sector_colors = {
        'Technology': '#3498db',  # 青（テクノロジー系）
        'Software': '#2980b9',   # 濃い青
        'Information Technology': '#3498db',
        'Communication Services': '#5dade2',  # 薄い青
        'Healthcare': '#e74c3c',  # 赤（ヘルスケア系）
        'Pharmaceuticals': '#c0392b',  # 濃い赤
        'Biotechnology': '#e67e22',  # オレンジ赤
        'Consumer Discretionary': '#9b59b6',  # 紫（消費財系）
        'Consumer Staples': '#8e44ad',  # 濃い紫
        'Retail': '#af7ac5',  # 薄い紫
        'Industrials': '#f39c12',  # オレンジ（工業系）
        'Manufacturing': '#e67e22',  # 濃いオレンジ
        'Aerospace': '#d68910',  # 黄オレンジ
        'Financials': '#27ae60',  # 緑（金融系）
        'Banking': '#229954',  # 濃い緑
        'Insurance': '#58d68d',  # 薄い緑
        'Energy': '#e74c3c',  # 赤（エネルギー系）
        'Oil & Gas': '#c0392b',  # 濃い赤
        'Renewable Energy': '#f1c40f',  # 黄色
        'Materials': '#8d6e63',  # 茶色（素材系）
        'Real Estate': '#795548',  # 濃い茶色
        'Utilities': '#607d8b',  # グレー（公益事業）
        'Transportation': '#ff9800',  # オレンジ（運輸系）
        'Media & Entertainment': '#e91e63',  # ピンク
        'Food & Beverage': '#4caf50',  # ライトグリーン
        'Automotive': '#ff5722',  # 深いオレンジ
        'Telecommunications': '#00bcd4',  # シアン
        'N/A': '#95a5a6'  # グレー（不明）
    }
    
    # 部分マッチングでセクターを判定
    sector_lower = str(sector).lower()
    for key, color in sector_colors.items():
        if key.lower() in sector_lower:
            return color
    
    # デフォルト色
    return sector_colors.get(sector, '#95a5a6')


def get_industry_color(industry, sector):
    """インダストリーごとの色を取得（セクターベースで微調整）"""
    # セクターベースの色を取得
    base_color = get_sector_color(sector)
    
    # インダストリー別の微調整マップ
    industry_modifiers = {
        # テクノロジー系の細分化
        'Software': '#2980b9',
        'Hardware': '#3498db',
        'Semiconductors': '#5dade2',
        'Internet': '#1abc9c',
        'Cloud Computing': '#16a085',
        
        # ヘルスケア系の細分化
        'Pharmaceuticals': '#c0392b',
        'Biotechnology': '#e67e22',
        'Medical Devices': '#f39c12',
        'Healthcare Services': '#e74c3c',
        
        # 金融系の細分化
        'Banking': '#229954',
        'Insurance': '#58d68d',
        'Asset Management': '#27ae60',
        'Fintech': '#1e8449',
        
        # 消費財系の細分化
        'Retail': '#af7ac5',
        'E-commerce': '#8e44ad',
        'Luxury Goods': '#9b59b6',
        'Apparel': '#bb8fce',
        
        # エネルギー系の細分化
        'Oil & Gas': '#c0392b',
        'Renewable Energy': '#f1c40f',
        'Solar': '#f4d03f',
        'Wind': '#f7dc6f',
        
        # 工業系の細分化
        'Manufacturing': '#e67e22',
        'Aerospace': '#d68910',
        'Defense': '#b7950b',
        'Construction': '#f39c12',
        
        # 不明
        'N/A': '#95a5a6'
    }
    
    # インダストリー名での部分マッチング
    industry_lower = str(industry).lower()
    for key, color in industry_modifiers.items():
        if key.lower() in industry_lower:
            return color
    
    # デフォルトはセクター色
    return base_color


def format_currency(value, symbol):
    """通貨に応じた価格フォーマット"""
    # 日本株判定（.T で終わるまたは日本の取引所）
    is_japanese = (str(symbol).endswith('.T') or 
                   str(symbol).endswith('.JP') or
                   # 4桁の数字のみの場合も日本株と判定
                   (str(symbol).isdigit() and len(str(symbol)) == 4))
    
    if is_japanese:
        return f'¥{value:.0f}'  # 日本円は小数点なし
    else:
        return f'${value:.2f}'


def generate_daily_ranking_html(engine) -> str:
    """日次株価上昇率ランキングレポートのHTMLを生成"""
    
    # 米国株と日本株のランキングデータを取得
    print("  日次ランキングデータを取得中...")
    us_ranking_data = fetch_daily_ranking_data(engine, market_type='US', limit=10)
    jp_ranking_data = fetch_daily_ranking_data(engine, market_type='JP', limit=10)
    
    # 両方のデータを結合
    ranking_data = pd.concat([us_ranking_data, jp_ranking_data], ignore_index=True)
    
    if ranking_data.empty:
        return generate_empty_ranking_html()
    
    print(f"  ランキングデータ取得完了: 米国株{len(us_ranking_data)}件、日本株{len(jp_ranking_data)}件")
    
    # 最新日付を取得（米国株・日本株それぞれ）
    us_latest_date = us_ranking_data['date'].iloc[0].strftime('%Y-%m-%d') if not us_ranking_data.empty else '不明'
    jp_latest_date = jp_ranking_data['date'].iloc[0].strftime('%Y-%m-%d') if not jp_ranking_data.empty else '不明'
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 銘柄リストをJSON形式で準備（全銘柄）
    symbols = ranking_data['symbol'].tolist()
    symbols_json = json.dumps(symbols)
    
    # 銘柄情報をJavaScript用に準備（会社名、市場タイプ、ランキング含む）
    symbols_info = {}
    
    # 米国株の情報を追加
    for idx, row in us_ranking_data.iterrows():
        symbols_info[row['symbol']] = {
            'company_name': row['company_name'] if row['company_name'] != 'N/A' else row['symbol'],
            'market_type': 'US',
            'rank': int(row['rank']),
            'market_rank': idx + 1  # 市場内でのランキング（1から開始）
        }
    
    # 日本株の情報を追加
    for idx, row in jp_ranking_data.iterrows():
        symbols_info[row['symbol']] = {
            'company_name': row['company_name'] if row['company_name'] != 'N/A' else row['symbol'],
            'market_type': 'JP',
            'rank': int(row['rank']),
            'market_rank': idx + 1  # 市場内でのランキング（1から開始）
        }
    
    symbols_info_json = json.dumps(symbols_info, ensure_ascii=False)
    
    # ランキング銘柄用のミニチャートデータを動的に生成
    print("  ランキング銘柄用チャートデータを生成中...")
    embedded_data = generate_mini_chart_data_for_ranking(engine, symbols)
    
    # フォールバック: 既存のJSONファイルからも読み込み（データが足りない場合）
    mini_json_dir = Path(__file__).parent.parent.parent.parent / "reports" / "mini_json"
    missing_symbols = [s for s in symbols if s not in embedded_data]
    
    if missing_symbols:
        print(f"  既存JSONファイルから補完: {missing_symbols}")
        for symbol in missing_symbols:
            json_file = mini_json_dir / f"{symbol}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        embedded_data[symbol] = json.load(f)
                    print(f"  既存JSONから補完: {symbol}")
                except Exception as e:
                    print(f"  既存JSON読み込みエラー ({symbol}): {e}")
            else:
                print(f"  JSONファイル未発見: {symbol}")
    
    embedded_data_json = json.dumps(embedded_data, ensure_ascii=False, separators=(',', ':'))
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>日次株価上昇率ランキング</title>
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <link rel="stylesheet" href="../static/mini.css">
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 15px;
                background-color: #f8f9fa;
                line-height: 1.5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 3px solid #e74c3c;
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
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin: 25px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
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
            .ranking-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.85em;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
                max-height: 500px; /* テーブルの最大高さを制限 */
                display: block; /* ブロック要素として表示 */
                overflow-y: auto; /* 縦スクロールを有効化 */
            }}
            .ranking-table thead,
            .ranking-table tbody {{
                display: table; /* テーブル形式を維持 */
                width: 100%;
                table-layout: fixed; /* 列幅を固定 */
            }}
            .ranking-table thead {{
                position: sticky; /* ヘッダーを固定 */
                top: 0;
                z-index: 10;
            }}
            .ranking-table th {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 10px 8px;
                text-align: left;
                font-weight: 600;
                border: none;
                font-size: 0.9em;
            }}
            .ranking-table td {{
                padding: 8px;
                border-bottom: 1px solid #eee;
                vertical-align: middle;
            }}
            .ranking-table tr:hover {{
                background-color: #f8f9fa;
            }}
            .ranking-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .rank-badge {{
                display: inline-block;
                width: 24px;
                height: 24px;
                line-height: 24px;
                text-align: center;
                border-radius: 50%;
                font-weight: bold;
                color: white;
                margin-right: 8px;
                font-size: 0.8em;
            }}
            .rank-1 {{ background: linear-gradient(135deg, #f1c40f, #f39c12); }}
            .rank-2 {{ background: linear-gradient(135deg, #95a5a6, #7f8c8d); }}
            .rank-3 {{ background: linear-gradient(135deg, #e67e22, #d35400); }}
            .rank-other {{ background: linear-gradient(135deg, #34495e, #2c3e50); }}
            .positive {{ color: #27ae60; font-weight: bold; }}
            .negative {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #95a5a6; }}
            .sector-badge {{
                background-color: #3498db;
                color: white;
                padding: 3px 6px;
                border-radius: 10px;
                font-size: 0.75em;
                white-space: nowrap;
            }}
            .industry-badge {{
                background-color: #9b59b6;
                color: white;
                padding: 3px 6px;
                border-radius: 10px;
                font-size: 0.75em;
                white-space: nowrap;
            }}
            .chart-button {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 0.75em;
                margin-right: 3px;
            }}
            .chart-button:hover {{
                background-color: #2980b9;
            }}
            .section-title {{
                font-size: 1.8em;
                color: #2c3e50;
                margin: 40px 0 20px 0;
                padding-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .help-text {{
                background-color: #e8f5e8;
                border: 2px solid #27ae60;
                border-radius: 8px;
                padding: 15px;
                margin: 30px 0;
                font-size: 0.9em;
                line-height: 1.6;
            }}
            .mini-chart-container {{
                margin: 25px 0;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 15px;
            }}
            .chart-card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chart-header {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 15px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .chart-content {{
                padding: 0;
                height: 270px;
            }}
            
            /* ウォッチリスト形式のチャートスタイル */
            .chart-section {{
                margin: 30px 0;
            }}
            .mini-chart-grid {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .watch-row {{
                display: grid;
                grid-template-columns: 1fr 1fr; /* 2列に修正: 価格チャート + 指標チャート */
                gap: 10px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            .watch-row:hover {{
                box-shadow: 0 6px 15px rgba(0,0,0,0.15);
                transform: translateY(-2px);
            }}
            .symbol-header {{
                grid-column: 1 / -1;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 12px 15px;
                font-weight: bold;
                font-size: 1.1em;
            }}
            .symbol-header-text {{
                flex: 1;
            }}
            .symbol-header-actions {{
                display: flex;
                gap: 8px;
            }}
            .back-to-ranking-btn {{
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 6px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.85em;
                transition: all 0.3s ease;
            }}
            .back-to-ranking-btn:hover {{
                background-color: rgba(255, 255, 255, 0.3);
                border-color: rgba(255, 255, 255, 0.5);
                transform: translateY(-1px);
            }}
            .chart {{
                min-height: 270px;
                height: 270px;
                padding: 5px; /* パディングを小さくして表示領域を拡大 */
                background: #fafafa;
                border: 1px solid #eee;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: visible; /* グラフが切れないようにvisibleに変更 */
                position: relative; /* 位置指定を追加 */
            }}
            .chart-loading {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 10px;
                color: #666;
            }}
            .loading-spinner {{
                width: 20px;
                height: 20px;
                border: 2px solid #e74c3c;
                border-top: 2px solid transparent;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            /* レスポンシブ対応：小さい画面では縦並びに */
            @media (max-width: 768px) {{
                .watch-row {{
                    grid-template-columns: 1fr; /* 小画面では1列に */
                }}
                .chart {{
                    height: 200px; /* 小画面では高さを縮小 */
                }}
            }}
            
            /* Plotlyチャート専用スタイル */
            .js-plotly-plot .plotly {{
                width: 100% !important;
                height: 100% !important;
            }}
            
            .js-plotly-plot .svg-container {{
                width: 100% !important;
                height: 100% !important;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 日次株価上昇率ランキング</h1>
                <div class="header-info">
                    <span>米国株: <strong>{us_latest_date}</strong></span>
                    <span>日本株: <strong>{jp_latest_date}</strong></span>
                    <span>最終更新: {current_time}</span>
                    <span>上位: <strong>米国株{len(us_ranking_data)}銘柄・日本株{len(jp_ranking_data)}銘柄</strong></span>
                </div>
            </div>
    """
    
    # サマリーカードを追加
    if not ranking_data.empty:
        avg_gain = ranking_data['change_percent'].mean()
        max_gain = ranking_data['change_percent'].max()
        total_volume = ranking_data['volume'].sum()
        unique_sectors = ranking_data['sector'].nunique()
        
        # 米国株・日本株別の統計
        us_avg_gain = us_ranking_data['change_percent'].mean() if not us_ranking_data.empty else 0
        jp_avg_gain = jp_ranking_data['change_percent'].mean() if not jp_ranking_data.empty else 0
        us_max_gain = us_ranking_data['change_percent'].max() if not us_ranking_data.empty else 0
        jp_max_gain = jp_ranking_data['change_percent'].max() if not jp_ranking_data.empty else 0
        
        html_content += f"""
             <!-- サマリーカード -->
             <div class="summary-grid">
                 <div class="summary-card">
                     <div class="summary-value">{max_gain:.1f}%</div>
                     <div class="summary-label">最大上昇率（全体）</div>
                 </div>
                 <div class="summary-card">
                     <div class="summary-value">{us_max_gain:.1f}%</div>
                     <div class="summary-label">米国株最大上昇率</div>
                 </div>
                 <div class="summary-card">
                     <div class="summary-value">{jp_max_gain:.1f}%</div>
                     <div class="summary-label">日本株最大上昇率</div>
                 </div>
                 <div class="summary-card">
                     <div class="summary-value">{unique_sectors}</div>
                     <div class="summary-label">関与セクター数</div>
                 </div>
             </div>
        """
    
    # 米国株ランキングテーブルを追加
    def create_ranking_table(data, title, market_icon):
        table_html = f"""
            <!-- {title}ランキングテーブル -->
            <h2 class="section-title">{market_icon} {title}</h2>
            <table class="ranking-table">
                <thead>
                    <tr>
                        <th>順位</th>
                        <th>銘柄</th>
                        <th>会社名</th>
                        <th>セクター</th>
                        <th>インダストリー</th>
                        <th>価格</th>
                        <th>変化額</th>
                        <th>変化率</th>
                        <th>出来高</th>
                        <th>時価総額</th>
                        <th>チャート</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # テーブル行を追加
        for _, row in data.iterrows():
            rank = int(row['rank'])
            symbol = row['symbol']
            company_name = row['company_name'] if row['company_name'] != 'N/A' else 'N/A'
            sector = row['sector'] if row['sector'] != 'N/A' else 'N/A'
            industry = row['industry'] if row['industry'] != 'N/A' else 'N/A'
            close = row['close']
            change = row['change']
            change_percent = row['change_percent']
            volume = row['volume']
            market_cap = row['market_cap']
            
            # ランクバッジのクラス
            if rank == 1:
                rank_class = "rank-1"
            elif rank == 2:
                rank_class = "rank-2"
            elif rank == 3:
                rank_class = "rank-3"
            else:
                rank_class = "rank-other"
            
            # 変化率のクラス
            change_class = "positive" if change_percent > 0 else "negative" if change_percent < 0 else "neutral"
            
            # セクター・インダストリーの色を取得
            sector_color = get_sector_color(sector)
            industry_color = get_industry_color(industry, sector)
            
            # 通貨フォーマット
            price_formatted = format_currency(close, symbol)
            change_formatted = format_currency(change, symbol)
            
            table_html += f"""
                        <tr>
                            <td>
                                <span class="rank-badge {rank_class}">{rank}</span>
                            </td>
                            <td>
                                <strong><a href="#chart-{symbol}" style="text-decoration: none; color: #2c3e50;">{html.escape(str(symbol))}</a></strong>
                            </td>
                            <td>{html.escape(str(company_name))}</td>
                            <td><span class="sector-badge" style="background-color: {sector_color};">{html.escape(str(sector))}</span></td>
                            <td><span class="industry-badge" style="background-color: {industry_color};">{html.escape(str(industry))}</span></td>
                            <td><strong>{price_formatted}</strong></td>
                            <td class="{change_class}">{'+'if change >= 0 else ''}{change_formatted}</td>
                            <td class="{change_class}"><strong>{change_percent:+.1f}%</strong></td>
                            <td>{format_large_number(volume)}</td>
                            <td>{format_large_number(market_cap)}</td>
                            <td>
                                <button class="chart-button" onclick="document.getElementById('chart-{symbol}').scrollIntoView({{behavior: 'smooth'}})">
                                    📈 チャート
                                </button>
                            </td>
                        </tr>
            """
        
        table_html += """
                    </tbody>
                </table>
        """
        return table_html
    
    # 米国株テーブル
    if not us_ranking_data.empty:
        html_content += create_ranking_table(us_ranking_data, "米国株上昇率ランキング", "🇺🇸")
    
    # 日本株テーブル
    if not jp_ranking_data.empty:
        html_content += create_ranking_table(jp_ranking_data, "日本株上昇率ランキング", "🇯🇵")
    
    # ミニチャートセクションを追加
    html_content += f"""
            <!-- ミニチャートセクション -->
            <h2 class="section-title">📊 詳細チャート</h2>
            <div class="chart-section">
                <div class="mini-chart-grid" id="mini-chart-grid">
                    <!-- チャートはJavaScriptで動的生成 -->
                </div>
            </div>
            
            <div class="help-text">
                <strong>💡 チャートの見方:</strong><br>
                • <strong>価格チャート（左）:</strong> ローソク足で価格推移、青線=SMA20、オレンジ線=SMA40<br>
                • <strong>指標チャート（右）:</strong> 上部=RSI（0-100、70以上で買われ過ぎ、30以下で売られ過ぎ）、下部=MACDヒストグラム（緑=上昇モメンタム、赤=下降モメンタム）<br>
                • <strong>銘柄名をクリック</strong> または <strong>📈 チャートボタン</strong> で該当チャートにジャンプします<br>
                • <strong>市場別表示:</strong> 🇺🇸米国株・🇯🇵日本株は時差により異なる日付のデータを表示します
            </div>
        </div>
        
        <script src="../static/mini_draw_embedded.js"></script>
        <script>
            // 銘柄リストをグローバル変数として設定
            const symbols = {symbols_json};
            
            // 銘柄情報（会社名、市場タイプ、ランキング）
            const symbolsInfo = {symbols_info_json};
            
            // JSONデータを直接埋め込み（file://制限回避）
            const embeddedChartData = {embedded_data_json};
            
            // ページ読み込み完了時にチャートを生成（ウォッチリスト形式）
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('日次ランキングページ読み込み完了');
                console.log('対象銘柄:', symbols);
                console.log('利用可能なJSONデータ:', Object.keys(embeddedChartData));
                
                // ミニチャートグリッドを生成
                const chartGrid = document.getElementById('mini-chart-grid');
                if (!chartGrid) {{
                    console.error('ミニチャートグリッドが見つかりません');
                    return;
                }}
                
                // 各銘柄のチャートを生成（ウォッチリスト形式）
                symbols.forEach((symbol, index) => {{
                    try {{
                        console.log(`チャート生成中: ${{symbol}}`);
                        
                        // ウォッチリスト形式のチャートコンテナ
                        const chartRow = document.createElement('div');
                        chartRow.className = 'watch-row';
                        chartRow.id = `chart-${{symbol}}`;
                        
                        // 銘柄情報を取得
                        const symbolInfo = symbolsInfo[symbol] || {{
                            company_name: symbol,
                            market_type: 'Unknown',
                            market_rank: index + 1
                        }};
                        
                        // 市場アイコンを設定
                        const marketIcon = symbolInfo.market_type === 'US' ? '🇺🇸' : '🇯🇵';
                        
                        // チャートヘッダーを作成
                        const chartHeader = document.createElement('div');
                        chartHeader.className = 'symbol-header';
                        chartHeader.innerHTML = `
                            <div class="symbol-header-text">
                                ${{marketIcon}} 【${{symbol}}】${{symbolInfo.company_name}} ランキング#${{symbolInfo.market_rank}}
                            </div>
                            <div class="symbol-header-actions">
                                <button class="back-to-ranking-btn" onclick="scrollToTop()">
                                    ⬆️ ランキングへ戻る
                                </button>
                            </div>
                        `;
                        
                        // 価格チャートコンテナ
                        const priceChart = document.createElement('div');
                        priceChart.id = `price-${{symbol}}`;
                        priceChart.className = 'chart';
                        priceChart.style.width = '100%';
                        priceChart.style.height = '270px';
                        priceChart.style.minWidth = '0';  // グリッドで縮小されるのを防ぐ
                        priceChart.innerHTML = `
                            <div class="chart-loading">
                                <div class="loading-spinner"></div>
                                価格チャート読み込み中...
                            </div>
                        `;
                        
                        // 指標チャートコンテナ
                        const indicChart = document.createElement('div');
                        indicChart.id = `indic-${{symbol}}`;
                        indicChart.className = 'chart';
                        indicChart.style.width = '100%';
                        indicChart.style.height = '270px';
                        indicChart.style.minWidth = '0';  // グリッドで縮小されるのを防ぐ
                        indicChart.innerHTML = `
                            <div class="chart-loading">
                                <div class="loading-spinner"></div>
                                指標チャート読み込み中...
                            </div>
                        `;
                        
                        // チャート行に追加
                        chartRow.appendChild(chartHeader);
                        chartRow.appendChild(priceChart);
                        chartRow.appendChild(indicChart);
                        chartGrid.appendChild(chartRow);
                        
                        // ミニチャートを描画（遅延ロード）
                        setTimeout(async () => {{
                            await loadRankingMiniChart(symbol);
                        }}, index * 200); // 0.2秒間隔でロード
                        
                    }} catch (error) {{
                        console.error(`チャート生成エラー (${{symbol}}):`, error);
                    }}
                }});
                
                console.log('全チャート生成完了');
            }});
            
            // ミニチャートロード関数（ウォッチリスト準拠）
            async function loadRankingMiniChart(symbol) {{
                try {{
                    console.log(`ミニチャート読み込み開始: ${{symbol}}`);
                    
                    // JSONデータが存在するかチェック
                    if (!embeddedChartData[symbol]) {{
                        console.warn(`JSONデータが見つかりません: ${{symbol}}`);
                        const priceDiv = document.getElementById(`price-${{symbol}}`);
                        const indicDiv = document.getElementById(`indic-${{symbol}}`);
                        if (priceDiv) priceDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">価格データが利用できません</div>';
                        if (indicDiv) indicDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">指標データが利用できません</div>';
                        return;
                    }}
                    
                    // mini_draw_embedded.jsの関数を使用してチャートを描画
                    // embeddedChartDataをグローバル変数として設定して、loadMiniChart関数から呼び出し
                    if (typeof loadMiniChart === 'function' && window.loadMiniChart !== loadRankingMiniChart) {{
                        console.log(`loadMiniChart関数（mini_draw_embedded.js）を呼び出し: ${{symbol}}`);
                        // 元のloadMiniChart関数を呼び出し（埋め込みデータを使用）
                        window.embeddedChartData = embeddedChartData;
                        window.loadMiniChart(symbol);
                        return;
                    }}
                    
                    // フォールバック: renderMiniChartV2関数を試す
                    if (typeof renderMiniChartV2 === 'function') {{
                        console.log(`renderMiniChartV2関数を呼び出し: ${{symbol}}`);
                        await renderMiniChartV2(symbol, embeddedChartData[symbol]);
                        
                        // Plotlyチャートのサイズを最適化
                        setTimeout(() => {{
                            const priceDiv = document.getElementById(`price-${{symbol}}`);
                            const indicDiv = document.getElementById(`indic-${{symbol}}`);
                            
                            if (priceDiv && window.Plotly) {{
                                window.Plotly.Plots.resize(priceDiv);
                            }}
                            if (indicDiv && window.Plotly) {{
                                window.Plotly.Plots.resize(indicDiv);
                            }}
                            
                            console.log(`Plotlyサイズ調整完了: ${{symbol}}`);
                        }}, 100);
                        
                    }} else {{
                        console.error(`ミニチャート描画関数が見つかりません。利用可能な関数:`, Object.keys(window).filter(k => k.includes('chart') || k.includes('Chart') || k.includes('mini') || k.includes('Mini')));
                        const priceDiv = document.getElementById(`price-${{symbol}}`);
                        const indicDiv = document.getElementById(`indic-${{symbol}}`);
                        if (priceDiv) priceDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">チャート描画関数が見つかりません</div>';
                        if (indicDiv) indicDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">チャート描画関数が見つかりません</div>';
                    }}
                    
                }} catch (error) {{
                    console.error(`ミニチャート読み込みエラー (${{symbol}}):`, error);
                    const priceDiv = document.getElementById(`price-${{symbol}}`);
                    const indicDiv = document.getElementById(`indic-${{symbol}}`);
                    if (priceDiv) priceDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">チャート読み込みエラー</div>';
                    if (indicDiv) indicDiv.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">チャート読み込みエラー</div>';
                }}
            }}
            
            // チャートスクロール機能
            function scrollToChart(symbol) {{
                const element = document.getElementById('chart-' + symbol);
                if (element) {{
                    element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    // 一時的にハイライト効果
                    element.style.boxShadow = '0 0 20px rgba(231, 76, 60, 0.5)';
                    element.style.transition = 'box-shadow 0.3s ease';
                    setTimeout(() => {{
                        element.style.boxShadow = '';
                    }}, 2000);
                }}
            }}
            
            // ランキングテーブルへ戻るスクロール機能
            function scrollToTop() {{
                window.scrollTo({{
                    top: 0,
                    behavior: 'smooth'
                }});
            }}
        </script>
    </body>
    </html>
    """
    
    print("  日次ランキングレポートHTML生成完了")
    return html_content


def generate_empty_ranking_html() -> str:
    """エラー時のフォールバック用空レポート"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>日次株価上昇率ランキング - データ取得エラー</title>
        <style>
            body {{
                font-family: system-ui, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
                line-height: 1.6;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            .error-card {{
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 日次株価上昇率ランキング</h1>
            <p>最終更新: {current_time}</p>
            
            <div class="error-card">
                <h2>❌ データ取得エラー</h2>
                <p>日次ランキングデータの取得に失敗しました。</p>
                <p>データベース接続やクエリに問題がある可能性があります。</p>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
                <h3>📝 トラブルシューティング</h3>
                <ul>
                    <li>データベースサーバーが起動していることを確認してください</li>
                    <li>必要なテーブル（fmp_data.daily_prices, fmp_data.company_profile, reference.company_gics, calculated_metrics.basic_metrics）が存在することを確認してください</li>
                    <li>最新の価格データが取得されていることを確認してください</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """





if __name__ == "__main__":
    # テスト用
    from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    from sqlalchemy import create_engine
    
    # データベース接続
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    
    # レポート生成
    html_content = generate_daily_ranking_html(engine)
    
    # テスト用にファイル出力
    with open("test_daily_ranking.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("テスト用ランキングレポートを test_daily_ranking.html に出力しました") 