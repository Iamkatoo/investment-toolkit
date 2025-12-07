"""
スコアリング結果を可視化するための関数群を実装します。
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import datetime
from investment_analysis.analysis.score_weights import MACRO_SCORE_WEIGHTS
import json
from sqlalchemy import text, create_engine
import math
import re
import base64
import struct
import os


def _clean_json_safe(obj):
    """
    Recursively clean JSON data to prevent NaN/Infinity values that cause JSON parse errors.
    Converts NaN/±inf to None and optionally rounds floats for readability.
    """
    if isinstance(obj, dict):
        return {k: _clean_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        # Round for readability (optional)
        return round(obj, 2)
    return obj


def make_pct_series(levels):
    """
    Convert a series of levels to percentage changes from the first value.
    Used for 6W change rate calculations.
    """
    if not levels or len(levels) < 2:
        return []
    base = levels[0]
    if base is None or base <= 0:
        return []
    return [((p/base)-1.0)*100.0 if p is not None else None for p in levels]


def plot_combined_score(df_macro: pd.DataFrame, macro_components: Dict, df_micro: pd.DataFrame,
                       start_date: str = '2015-01-01') -> go.Figure:
    """
    マクロスコアとミクロスコアを組み合わせた総合レポートを生成します。
    
    Parameters
    ----------
    df_macro : pd.DataFrame
        マクロスコアを含むデータフレーム
    macro_components : Dict
        各マクロ指標の寄与度
    df_micro : pd.DataFrame
        ミクロスコア（個別銘柄評価）を含むデータフレーム
    start_date : str, optional
        表示開始日（デフォルト: '2015-01-01'）
        
    Returns
    -------
    go.Figure
        総合スコアレポートの図表
    """
    # 最新日付
    latest_date = df_macro['date'].max()
    date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
    
    # マクロスコアの最新値
    macro_score_sum = sum(macro_components.values())
    
    # ミクロスコアの平均値
    if not df_micro.empty:
        avg_micro_score = df_micro['total_score'].mean()
    else:
        avg_micro_score = 0
    
    # 1. サマリーカード用の色を決定
    macro_color = 'red' if macro_score_sum < 0 else ('green' if macro_score_sum > 0 else 'grey')
    micro_color = 'red' if avg_micro_score < 0 else ('green' if avg_micro_score > 0 else 'grey')
    total_score = macro_score_sum + avg_micro_score
    total_color = 'red' if total_score < 0 else ('green' if total_score > 0 else 'grey')
    
    # 2. サブプロット作成 (3行1列) - 旧スコアカード行を削除
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.4, 0.2],
        specs=[
            [{"type": "bar"}],       # ミクロブロック
            [{"type": "waterfall"}], # マクロウォーターフォール
            [{"type": "scatter"}]    # グローバルインデックスミニタイムライン
        ],
        subplot_titles=(
            "個別銘柄スコア",
            "マクロ指標寄与度",
            "主要指数推移"
        ),
        vertical_spacing=0.1
    )
    
    # 3. サマリーカード (Indicator) - 削除（要件書に従って旧スコアカードを削除）
    
    # 4. ミクロブロック (銘柄別スコア)
    if not df_micro.empty:
        # シンボル順にソート
        df_micro_sorted = df_micro.sort_values('total_score')
        
        # 横棒グラフで各銘柄のスコアを表示
        for i, row in df_micro_sorted.iterrows():
            symbol = row['symbol']
            score = row['total_score']
            color = 'red' if score < 0 else ('green' if score > 0 else 'grey')
            
            fig.add_trace(
                go.Bar(
                    x=[score],
                    y=[symbol],
                    orientation='h',
                    marker_color=color,
                    name=symbol,
                    text=[f"{score:.1f}"],
                    textposition='outside',
                    hovertemplate=(
                        f"<b>{symbol}</b><br>" +
                        f"セクター: {row['sector']}<br>" +
                        f"変化率: {row['price_change_pct']:.2f}%<br>" +
                        f"セクター乖離: {row['sector_deviation']:.2f}%<br>" +
                        f"インダストリー乖離: {row['industry_deviation']:.2f}%<br>" +
                        f"出来高変化: {row['volume_change_pct']:.2f}%<br>" +
                        f"ATR比: {row['atr_ratio']:.2f}<br>" +
                        f"合計: {score:.1f}"
                    )
                ),
                row=1, col=1
            )
        
        # レイアウト調整
        fig.update_xaxes(title_text="スコア", row=1, col=1)
        fig.update_yaxes(title_text="銘柄", row=1, col=1)
    
    # 5. マクロウォーターフォール (指標寄与度)
    # 寄与度をデータフレームに変換
    components = []
    for key, value in macro_components.items():
        if value != 0:  # 値が0の場合は表示しない
            components.append({'name': key, 'value': value})
    
    if components:
        df_components = pd.DataFrame(components)
        
        # 寄与度の大きい順にソート
        df_components = df_components.sort_values('value')
        
        # ウォーターフォールチャート用のデータ準備
        measure = ['relative'] * len(df_components)
        text = [f"{x:.1f}" for x in df_components['value']]
        
        fig.add_trace(
            go.Waterfall(
                name="マクロ寄与度",
                orientation="h",
                measure=measure,
                y=df_components['name'],
                x=df_components['value'],
                text=text,
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "red"}},
                increasing={"marker": {"color": "green"}},
                hovertemplate="%{y}: %{x:.1f}<extra></extra>"
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="寄与度", row=2, col=1)
    
    # 6. グローバルインデックスミニタイムライン
    # フィルタリング
    df_indices = df_macro[df_macro['date'] >= start_date].copy()
    
    # 主要指数の選択
    indices = ['SPY', 'DGS10', 'TWEXBGSMTH']
    colors = ['blue', 'orange', 'green']
    
    for i, index_name in enumerate(indices):
        if index_name in df_indices.columns:
            # 最初の値で正規化
            norm_values = df_indices[index_name] / df_indices[index_name].iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df_indices['date'],
                    y=norm_values,
                    mode='lines',
                    name=index_name,
                    line=dict(color=colors[i])
                ),
                row=3, col=1
            )
    
    fig.update_xaxes(title_text="日付", row=3, col=1)
    fig.update_yaxes(title_text="正規化値 (指数)", row=3, col=1)
    
    # 図表全体のレイアウト調整
    fig.update_layout(
        height=1000,
        width=1000,
        title=f'市場評価総合スコア ({date_str})',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    
    return fig


def get_score_time_series(engine, symbol: str, start_date: str, end_date: str) -> Dict:
    """
    指定銘柄のスコア時系列データを取得
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
    symbol : str
        銘柄コード
    start_date : str
        開始日
    end_date : str
        終了日
        
    Returns
    -------
    Dict
        スコア時系列データ
    """
    try:
        from sqlalchemy import text
        
        # スコア時系列データを取得
        score_query = text("""
            SELECT 
                date,
                total_score,
                growth_score,
                quality_score,
                momentum_score,
                macro_sector_score
            FROM backtest_results.daily_scores
            WHERE symbol = :symbol
            AND date >= :start_date
            AND date <= :end_date
            ORDER BY date
        """)
        
        with engine.connect() as conn:
            df_scores = pd.read_sql_query(score_query, conn, params={
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            })
            
        if df_scores.empty:
            print(f"{symbol}: スコアデータが見つかりませんでした")
            return {
                'dates': [],
                'total_score': [],
                'growth_score': [],
                'quality_score': [],
                'momentum_score': [],
                'macro_sector_score': []
            }
        
        # データを辞書形式で返す
        # date列をdatetimeに変換
        df_scores['date'] = pd.to_datetime(df_scores['date'])
        
        return {
            'dates': df_scores['date'].dt.strftime('%Y-%m-%d').tolist(),
            'total_score': df_scores['total_score'].fillna(0).tolist(),
            'growth_score': df_scores['growth_score'].fillna(0).tolist(),
            'quality_score': df_scores['quality_score'].fillna(0).tolist(),
            'momentum_score': df_scores['momentum_score'].fillna(0).tolist(),
            'macro_sector_score': df_scores['macro_sector_score'].fillna(0).tolist()
        }
        
    except Exception as e:
        print(f"{symbol}: スコアデータ取得エラー: {e}")
        return {
            'dates': [],
            'total_score': [],
            'growth_score': [],
            'quality_score': [],
            'momentum_score': [],
            'macro_sector_score': []
        }


def create_score_sparklines(df_micro: pd.DataFrame, engine, days_back: int = 90) -> Dict:
    """
    銘柄ごとの過去のスパークライン（小さな時系列チャート）データを生成します。
    3ヶ月分を最低期間とし、ポートフォリオ銘柄については購入日からのデータを表示します。
    
    Parameters
    ----------
    df_micro : pd.DataFrame
        ミクロスコア（個別銘柄評価）を含むデータフレーム
    engine : Engine
        データベース接続エンジン
    days_back : int, optional
        遡る日数（デフォルト: 90日=3ヶ月）
        
    Returns
    -------
    Dict
        銘柄ごとのOHLCチャートデータを含む辞書
    """
    if df_micro.empty:
        return {}
    
    # 最新の日付を取得 - 日付型の互換性を確保
    df_micro['date'] = pd.to_datetime(df_micro['date'])
    latest_date = df_micro['date'].max()
    
    # 銘柄リストを取得
    symbols = df_micro['symbol'].unique().tolist()
    
    # ポートフォリオの購入日情報を取得
    portfolio_purchase_dates = {}
    try:
        from sqlalchemy import text
        query = text("""
            WITH transactions AS (
                SELECT symbol, MIN(date) as purchase_date
                FROM user_data.transactions 
                WHERE transaction_type = 'buy'
                GROUP BY symbol
            )
            SELECT symbol, purchase_date
            FROM transactions
            WHERE symbol = ANY(:symbols)
        """)
        
        with engine.connect() as conn:
            portfolio_df = pd.read_sql_query(query, conn, params={"symbols": symbols})
            for _, row in portfolio_df.iterrows():
                portfolio_purchase_dates[row['symbol']] = pd.to_datetime(row['purchase_date'])
                
        print(f"ポートフォリオ購入日情報を取得: {portfolio_purchase_dates}")
    except Exception as e:
        print(f"ポートフォリオ購入日情報取得エラー: {e}")
    
    # デフォルトの表示期間（最低3ヶ月）
    default_start_date = pd.to_datetime(latest_date) - pd.Timedelta(days=days_back)
    latest_date_str = latest_date.strftime('%Y-%m-%d')
    
    # 売買記録データを取得（購入日・損切り・利確ライン表示用）
    trade_journal_data = {}
    try:
        from sqlalchemy import text
        trade_query = text("""
            SELECT symbol, buy_date, buy_price, stop_loss_price, take_profit_price, sell_date
            FROM user_data.trade_journal
            WHERE symbol = ANY(:symbols)
              AND sell_date IS NULL  -- 保有中の銘柄のみ
            ORDER BY buy_date DESC
        """)
        
        with engine.connect() as conn:
            trade_df = pd.read_sql_query(trade_query, conn, params={"symbols": symbols})
            for _, row in trade_df.iterrows():
                symbol = row['symbol']
                if symbol not in trade_journal_data:
                    trade_journal_data[symbol] = []
                trade_journal_data[symbol].append({
                    'buy_date': pd.to_datetime(row['buy_date']),
                    'buy_price': row['buy_price'],
                    'stop_loss_price': row['stop_loss_price'],
                    'take_profit_price': row['take_profit_price']
                })
                
        print(f"売買記録データを取得: {len(trade_journal_data)}銘柄")
    except Exception as e:
        print(f"売買記録データ取得エラー: {e}")
    
    # 過去の株価データを取得
    sparkline_data = {}
    
    for symbol in symbols:
        try:
            # 銘柄ごとの表示期間を決定
            # まず、trade_journalで該当銘柄があるかを確認
            symbol_in_trade_journal = symbol in trade_journal_data and trade_journal_data[symbol]
            
            if symbol_in_trade_journal:
                # trade_journalに存在する場合、その購入日を使用
                trade_buy_date = trade_journal_data[symbol][0]['buy_date']  # 最新の取引
                months_held = (pd.to_datetime(latest_date) - trade_buy_date).days / 30.44
                
                if months_held >= 3:
                    # 3ヶ月以上保有：購入日からのデータを取得
                    start_date = trade_buy_date
                    print(f"{symbol}: 購入日からの全データ表示 ({trade_buy_date.strftime('%Y-%m-%d')} - {months_held:.1f}ヶ月保有)")
                else:
                    # 3ヶ月未満：3ヶ月分のデータを取得
                    start_date = default_start_date
                    print(f"{symbol}: 3ヶ月間表示 (保有{months_held:.1f}ヶ月 < 3ヶ月)")
            elif symbol in portfolio_purchase_dates:
                # portfolioファイルに存在する場合（フォールバック）
                purchase_date = portfolio_purchase_dates[symbol]
                months_held = (pd.to_datetime(latest_date) - purchase_date).days / 30.44
                if months_held >= 3:
                    start_date = purchase_date
                    print(f"{symbol}: ポートフォリオファイルから購入日表示 ({purchase_date.strftime('%Y-%m-%d')} - {months_held:.1f}ヶ月保有)")
                else:
                    start_date = default_start_date
                    print(f"{symbol}: 3ヶ月間表示 (ポートフォリオ保有{months_held:.1f}ヶ月 < 3ヶ月)")
            else:
                # どちらにも存在しない銘柄は3ヶ月間
                start_date = default_start_date
                print(f"{symbol}: 3ヶ月間表示 (取引記録なし)")
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # 銘柄の通貨情報を取得 - fmp_data.company_profileテーブルを使用
            currency_query = """
            SELECT currency
            FROM fmp_data.company_profile
            WHERE symbol = :symbol
            """
            
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(currency_query), {"symbol": symbol})
                    currency_row = result.fetchone()
                    if currency_row:
                        currency = currency_row[0]
                    else:
                        # テーブルからデータが取得できない場合、シンボルから通貨を推測
                        if symbol.endswith('.T'):
                            # 日本株は.Tで終わる
                            currency = 'JPY'
                        else:
                            # その他はUSDとみなす
                            currency = 'USD'
                        print(f"警告: {symbol}の通貨情報がテーブルで見つかりませんでした。シンボルから推測: {currency}")
            except Exception as e:
                print(f"通貨情報取得中にエラー({symbol}): {e}")
                # シンボルから通貨を推測するバックアップロジック
                if symbol.endswith('.T'):
                    currency = 'JPY'
                else:
                    currency = 'USD'
                print(f"エラーのため、シンボルから通貨を推測: {symbol} -> {currency}")
            
            print(f"銘柄: {symbol}, 通貨: {currency}")
            
            # 株価データの取得（OHLCデータを含む）
            price_query = """
            SELECT 
                date, 
                open,
                high,
                low,
                close,
                volume
            FROM fmp_data.daily_prices
            WHERE symbol = :symbol
            AND date >= :start_date
            AND date <= :end_date
            ORDER BY date
            """
            
            # SMAデータの取得（technical_indicatorsテーブルから）
            sma_query = """
            SELECT 
                date, 
                sma_20 AS sma20,
                sma_40 AS sma40
            FROM calculated_metrics.technical_indicators
            WHERE symbol = :symbol
            AND date >= :start_date
            AND date <= :end_date
            ORDER BY date
            """
            
            # セクター平均とインダストリー平均の取得を削除（シンプル化のため）
            
            # 各データの取得
            with engine.connect() as conn:
                # 株価データ
                df_price = pd.read_sql(
                    text(price_query), 
                    conn, 
                    params={"symbol": symbol, "start_date": start_date_str, "end_date": latest_date_str}
                )
                
                # SMAデータ
                df_sma = pd.read_sql(
                    text(sma_query), 
                    conn, 
                    params={"symbol": symbol, "start_date": start_date_str, "end_date": latest_date_str}
                )
            
            # 日付を共通のキーとしてマージ
            df_price['date'] = pd.to_datetime(df_price['date'])
            
            if not df_price.empty:
                dates = df_price['date'].tolist()
                opens = df_price['open'].tolist()
                highs = df_price['high'].tolist()
                lows = df_price['low'].tolist()
                closes = df_price['close'].tolist()
                volumes = df_price['volume'].tolist()
                
                # SMAデータの処理
                if not df_sma.empty:
                    df_sma['date'] = pd.to_datetime(df_sma['date'])
                    df_merged = pd.merge(df_price[['date']], df_sma, on='date', how='left')
                    sma20s = df_merged['sma20'].ffill().bfill().tolist()
                    sma40s = df_merged['sma40'].ffill().bfill().tolist()
                else:
                    # SMAデータがない場合は単純移動平均を計算
                    sma20s = []
                    sma40s = []
                    for i in range(len(closes)):
                        if i < 20:
                            sma20s.append(sum(closes[:i+1]) / (i+1))
                        else:
                            sma20s.append(sum(closes[i-19:i+1]) / 20)
                        
                        if i < 40:
                            sma40s.append(sum(closes[:i+1]) / (i+1))
                        else:
                            sma40s.append(sum(closes[i-39:i+1]) / 40)
                
                # 売買記録データを追加
                trade_data = trade_journal_data.get(symbol, [])
                
                # スコア推移データを取得
                score_data = get_score_time_series(engine, symbol, start_date_str, latest_date_str)
                
                # データをチャート用に整形（OHLCデータとテクニカル指標、売買記録、スコア）
                sparkline_data[symbol] = {
                    'date': dates,
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes,
                    'sma20': sma20s,
                    'sma40': sma40s,
                    'trade_data': trade_data,
                    'score_data': score_data
                }
                
                print(f"{symbol}: {len(df_price)}日分のデータを取得")
            else:
                print(f"{symbol}: 価格データが見つかりませんでした")
                # ダミーデータの作成（チャートが生成されるように）
                dummy_dates = pd.date_range(start=start_date, end=latest_date, freq='D')
                trade_data = trade_journal_data.get(symbol, [])
                
                # ダミースコアデータ
                dummy_score_data = {
                    'dates': [],
                    'total_score': [],
                    'growth_score': [],
                    'quality_score': [],
                    'momentum_score': [],
                    'macro_sector_score': []
                }
                
                sparkline_data[symbol] = {
                    'date': dummy_dates,
                    'open': [0.0] * len(dummy_dates),
                    'high': [0.0] * len(dummy_dates),
                    'low': [0.0] * len(dummy_dates),
                    'close': [0.0] * len(dummy_dates),
                    'volume': [0.0] * len(dummy_dates),
                    'sma20': [0.0] * len(dummy_dates),
                    'sma40': [0.0] * len(dummy_dates),
                    'trade_data': trade_data,
                    'score_data': dummy_score_data
                }
                print(f"{symbol}: ダミーデータを生成しました ({len(dummy_dates)}日分)")
                
        except Exception as e:
            print(f"{symbol}のチャートデータ取得中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            # エラーの場合はダミーデータを提供
            dummy_dates = pd.date_range(start=default_start_date, end=latest_date, freq='D')
            trade_data = trade_journal_data.get(symbol, [])
            
            # ダミースコアデータ
            dummy_score_data = {
                'dates': [],
                'total_score': [],
                'growth_score': [],
                'quality_score': [],
                'momentum_score': [],
                'macro_sector_score': []
            }
            
            sparkline_data[symbol] = {
                'date': dummy_dates,
                'open': [0.0] * len(dummy_dates),
                'high': [0.0] * len(dummy_dates),
                'low': [0.0] * len(dummy_dates),
                'close': [0.0] * len(dummy_dates),
                'volume': [0.0] * len(dummy_dates),
                'sma20': [0.0] * len(dummy_dates),
                'sma40': [0.0] * len(dummy_dates),
                'trade_data': trade_data,
                'score_data': dummy_score_data
            }
    
    return sparkline_data


def plot_market_score_report(df_macro: pd.DataFrame, macro_components: Dict, df_micro: pd.DataFrame,
                           sparkline_data: Optional[Dict] = None) -> go.Figure:
    """
    市場スコアの詳細レポートを生成します。
    
    Parameters
    ----------
    df_macro : pd.DataFrame
        マクロスコアを含むデータフレーム
    macro_components : Dict
        各マクロ指標の寄与度
    df_micro : pd.DataFrame
        ミクロスコア（個別銘柄評価）を含むデータフレーム
    sparkline_data : Dict, optional
        スパークラインデータ
        
    Returns
    -------
    go.Figure
        市場スコア詳細レポートの図表
    """
    # 最新日付
    latest_date = df_macro['date'].max()
    date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
    
    # 横長のレイアウト (1行2列)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[
            [{"type": "table"}, {"type": "waterfall"}]
        ],
        subplot_titles=(
            "個別銘柄スコア詳細",
            "マクロ指標寄与度"
        )
    )
    
    # 1. 詳細テーブル
    if not df_micro.empty:
        # データ整形
        table_data = df_micro.copy()
        
        # スパークラインが提供されている場合は列を追加（実際のプロットはHTMLで）
        if sparkline_data:
            table_data['sparkline'] = "スパークライン"
        
        # 表示する列を選択・名前変更
        display_cols = {
            'symbol': 'シンボル',
            'sector': 'セクター',
            'price_change_pct': '前日比(%)',
            'sector_deviation': 'セクター乖離(%)',
            'industry_deviation': 'インダストリー乖離(%)',
            'volume_change_pct': '出来高変化(%)',
            'price_change_score': '価格スコア',
            'sector_score': 'セクタースコア',
            'industry_score': 'インダストリースコア',
            'volume_score': '出来高スコア',
            'gc_dc_score': 'GC/DCスコア',
            'atr_score': 'ATRスコア',
            'total_score': '合計スコア'
        }
        
        if 'sparkline' in table_data.columns:
            display_cols['sparkline'] = '1ヶ月チャート'
        
        # 列名変更
        table_data = table_data[list(display_cols.keys())].rename(columns=display_cols)
        
        # 数値列のフォーマット
        for col in table_data.columns:
            if col not in ['シンボル', 'セクター', '1ヶ月チャート']:
                table_data[col] = table_data[col].round(2)
        
        # 合計スコアでソート
        table_data = table_data.sort_values('合計スコア', ascending=False)
        
        # スコア色分け関数
        def score_class(val):
            if val >= 2:
                return "positive strong"
            elif val > 0:
                return "positive"
            elif val <= -2:
                return "negative strong"
            elif val < 0:
                return "negative"
            else:
                return ""

        # テーブル作成
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(table_data.columns),
                    fill_color='lightgrey',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[table_data[col] for col in table_data.columns],
                    fill_color=[
                        ['white'] * len(table_data),  # シンボル
                        ['white'] * len(table_data),  # セクター
                        ['white'] * len(table_data),  # 前日比(%)
                        ['white'] * len(table_data),  # セクター乖離(%)
                        ['white'] * len(table_data),  # インダストリー乖離(%)
                        ['white'] * len(table_data),  # 出来高変化(%)
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['価格スコア']],
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['セクタースコア']],
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['インダストリースコア']],
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['出来高スコア']],
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['GC/DCスコア']],
                        ['rgba(255,150,150,0.2)' if v < 0 else 'rgba(150,255,150,0.2)' if v > 0 else 'white' for v in table_data['ATRスコア']],
                        ['rgba(255,150,150,0.5)' if v < 0 else 'rgba(150,255,150,0.5)' if v > 0 else 'white' for v in table_data['合計スコア']]
                    ],
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=1, col=1
        )
    
    # 2. マクロウォーターフォール
    # 寄与度をデータフレームに変換
    components = []
    for key, value in macro_components.items():
        if value != 0:  # 値が0の場合は表示しない
            components.append({'name': key, 'value': value})
    
    if components:
        df_components = pd.DataFrame(components)
        
        # 値が正のものと負のものに分ける
        positives = df_components[df_components['value'] > 0].sort_values('value', ascending=False)
        negatives = df_components[df_components['value'] < 0].sort_values('value')
        
        # まず正の値から表示
        y_values = positives['name'].tolist() + negatives['name'].tolist()
        x_values = positives['value'].tolist() + negatives['value'].tolist()
        
        # ウォーターフォールチャート用のデータ準備
        measure = ['relative'] * len(y_values)
        text = [f"{x:.1f}" for x in x_values]
        
        fig.add_trace(
            go.Waterfall(
                name="マクロ寄与度",
                orientation="v",
                measure=measure,
                x=y_values,
                y=x_values,
                text=text,
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "red"}},
                increasing={"marker": {"color": "green"}},
                hovertemplate="%{x}: %{y:.1f}<extra></extra>"
            ),
            row=1, col=2
        )
    
    # 図表全体のレイアウト調整
    fig.update_layout(
        height=600,
        width=1200,
        title=f'市場スコア詳細レポート ({date_str})',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def iter_plotly_data_blocks(html_text):
    """HTMLテキストからすべてのPlotly.newPlotのdata配列を抽出"""
    import re
    
    print(f"        🔍 DEBUG: iter_plotly_data_blocks開始 (html_text length={len(html_text)})")
    
    # 1つの確実なパターンで括弧バランシングを使って抽出
    pattern = r'Plotly\.newPlot\s*\(\s*["\'][^"\']*["\'],\s*(\[)'
    
    data_blocks = []
    
    # パターンマッチングで開始位置を見つける
    for match in re.finditer(pattern, html_text):
        start_pos = match.end(1) - 1  # '[' の位置
        
        # 括弧バランシングで配列の終了位置を見つける
        bracket_count = 0
        end_pos = start_pos
        
        for i, char in enumerate(html_text[start_pos:], start_pos):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_pos = i + 1
                    break
        
        if bracket_count == 0:  # 正しく閉じられた配列が見つかった
            data_str = html_text[start_pos:end_pos]
            print(f"        🔍 DEBUG: Pattern found: data_str length={len(data_str)}")
            data_blocks.append(data_str)
        else:
            print(f"        ⚠️ DEBUG: Unbalanced brackets starting at {start_pos}")
    
    return data_blocks


def iter_plotly_data_blocks_old(html_text):
    """HTMLテキストからすべてのPlotly.newPlotのdata配列を抽出（旧版）"""
    import re
    
    print(f"        🔍 DEBUG: iter_plotly_data_blocks開始 (html_text length={len(html_text)})")
    
    # 実際のHTML構造に基づくパターン（UUID文字列の第1引数、第2引数がdata配列）
    patterns = [
        # パターン1: 基本的な形式（第2引数がdata配列）
        r'Plotly\.newPlot\([^,]+,\s*(\[[\s\S]*?\])\s*,',
    ]
    
    data_blocks = []
    for i, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, html_text, re.DOTALL))
        print(f"        🔍 DEBUG: Pattern {i+1}: {len(matches)} matches")
        for j, match in enumerate(matches):
            data_str = match.group(1)
            # JavaScriptのNaN/Infinity/末尾カンマを置換
            data_str = re.sub(r'\bNaN\b', 'null', data_str)
            data_str = re.sub(r'\bInfinity\b', 'null', data_str)
            data_str = re.sub(r',\s*\]', ']', data_str)  # 末尾カンマ除去
            data_blocks.append(data_str)
            print(f"        🔍 DEBUG: Pattern {i+1}, Match {j+1}: data_str length={len(data_str)}")
    
    print(f"        🔍 DEBUG: iter_plotly_data_blocks完了: {len(data_blocks)} blocks found")
    return data_blocks


def find_trace(traces, want, pattern=None):
    """トレース名の検索（完全一致→正規表現）"""
    import re
    
    names = [str(t.get("name", "")) for t in traces]
    
    # 1) 完全一致
    for trace in traces:
        if trace.get('name') == want:
            return trace, names
    
    # 2) 正規表現（パターンが指定されている場合）
    if pattern:
        rx = re.compile(pattern, re.I)
        for trace in traces:
            if rx.search(str(trace.get('name', ''))):
                return trace, names
    
    return None, names


def extract_plotly_data(html_path, trace_name, pattern=None):
    """
    HTMLファイルからPlotlyデータを抽出する（堅牢版）
    
    Args:
        html_path: HTMLファイルのパス
        trace_name: 抽出したいトレース名
        pattern: 正規表現パターン（表記ゆれ対応）
        
    Returns:
        dict: {"x": [...], "y": [...], "found": bool, "available_traces": [...]}
    """
    import re
    import json
    import os
    from pathlib import Path
    
    print(f"    🔍 DEBUG: extract_plotly_data開始 (html_path={html_path}, trace_name={trace_name})")
    
    try:
        # ファイル存在確認（絶対パスで表示）
        html_file = Path(html_path)
        if not html_file.exists():
            abs_path = html_file.resolve()
            print(f"⚠️ HTML file not found: {html_path} (absolute: {abs_path})")
            return {"x": [], "y": [], "found": False, "available_traces": []}
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"      🔍 DEBUG: HTMLファイル読み込み完了 (size={len(content)} bytes)")
        
        # すべてのPlotly.newPlotのdata配列を抽出
        data_blocks = iter_plotly_data_blocks(content)
        print(f"      🔍 DEBUG: Plotly data blocks数: {len(data_blocks)}")
        
        if not data_blocks:
            print(f"⚠️ No Plotly data blocks found in: {html_path}")
            return {"x": [], "y": [], "found": False, "available_traces": []}
        
        # 各data配列をパースしてトレースを検索
        all_traces = []
        for i, data_str in enumerate(data_blocks):
            try:
                plotly_data = json.loads(data_str)
                if isinstance(plotly_data, list):
                    all_traces.extend(plotly_data)
                    print(f"      🔍 DEBUG: Block {i+1}: {len(plotly_data)} traces added")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parse error in block {i+1} of {html_path}: {e}")
                continue
        
        print(f"      🔍 DEBUG: 総トレース数: {len(all_traces)}")
        
        if not all_traces:
            print(f"⚠️ No valid traces found in: {html_path}")
            return {"x": [], "y": [], "found": False, "available_traces": []}
        
        # トレース名を検索
        trace, available_traces = find_trace(all_traces, trace_name, pattern)
        print(f"      🔍 DEBUG: トレース検索結果: found={trace is not None}, available_traces={available_traces}")
        
        if trace:
            x_data = trace.get('x', [])
            y_data_raw = trace.get('y', [])
            
            # Y データの形式チェック（バイナリエンコードされている場合をデコード）
            if isinstance(y_data_raw, dict) and "bdata" in y_data_raw:
                # Plotlyのバイナリエンコードされたデータをデコード
                import base64
                import struct
                
                dtype = y_data_raw.get("dtype", "f8")
                bdata = y_data_raw["bdata"]
                
                # Base64デコード
                binary_data = base64.b64decode(bdata)
                
                # dtypeに基づいてバイナリデータを数値に変換
                if dtype == "f8":  # float64
                    format_char = "d"
                    size = 8
                elif dtype == "f4":  # float32
                    format_char = "f"
                    size = 4
                else:
                    format_char = "d"
                    size = 8
                
                # バイナリデータを数値リストに変換
                num_values = len(binary_data) // size
                y_data = list(struct.unpack(f"<{num_values}{format_char}", binary_data))
                
                print(f"      🔍 DEBUG: Decoded binary data - dtype={dtype}, values={num_values}")
            else:
                y_data = y_data_raw
            
            return {"x": x_data, "y": y_data, "found": True, "available_traces": available_traces}
        else:
            print(f"⚠️ Trace not found: want='{trace_name}' in {html_path} available={available_traces}")
            return {"x": [], "y": [], "found": False, "available_traces": available_traces}
        
    except Exception as e:
        print(f"⚠️ Error extracting data from {html_path}: {e}")
        return {"x": [], "y": [], "found": False, "available_traces": []}


def extract_6w_data(x_data, y_data, anchor_days=50, min_points=20):
    """
    6W期間のデータを抽出し、最低点数を保証する（強化版）
    
    Args:
        x_data: 日付データ（ISO形式）
        y_data: 値データ
        anchor_days: アンカー日数（デフォルト50日で6W+α）
        min_points: 最低点数（デフォルト20点）
        
    Returns:
        dict: {"value": float, "series": list, "change_rate": float}
    """
    print(f"        🔍 DEBUG: extract_6w_data開始 - x_data長={len(x_data) if x_data else 0}, y_data長={len(y_data) if y_data else 0}")
    
    if not x_data or not y_data or len(x_data) != len(y_data):
        print(f"        ⚠️ DEBUG: データ長不一致または空 - x_data={len(x_data) if x_data else 0}, y_data={len(y_data) if y_data else 0}")
        return {"value": None, "series": [], "change_rate": None}
    
    # 日付をdatetimeに変換
    from datetime import datetime, timedelta
    try:
        # Plotlyの日付形式対応（ナノ秒精度を削除）
        dates = []
        for d in x_data:
            # 'YYYY-MM-DDTHH:MM:SS.000000000' -> 'YYYY-MM-DDTHH:MM:SS'
            clean_date = d.split('.')[0] if '.' in d else d
            clean_date = clean_date.replace('Z', '+00:00')
            dates.append(datetime.fromisoformat(clean_date))
        print(f"        🔍 DEBUG: 日付変換完了 - 最初の日付={dates[0] if dates else 'None'}, 最後の日付={dates[-1] if dates else 'None'}")
    except Exception as e:
        print(f"⚠️ Date parsing error in extract_6w_data: {e}")
        return {"value": None, "series": [], "change_rate": None}
    
    # 最新日付から6W前の範囲を抽出（50日で6W+α）
    latest_date = dates[0]
    start_date = latest_date - timedelta(days=anchor_days)
    print(f"        🔍 DEBUG: 期間設定 - latest_date={latest_date}, start_date={start_date}, anchor_days={anchor_days}")
    
    # 範囲内のデータを抽出
    filtered_data = [(d, y) for d, y in zip(dates, y_data) if d >= start_date]
    print(f"        🔍 DEBUG: フィルタリング後 - データ点数={len(filtered_data)}")
    
    if len(filtered_data) < 2:
        print(f"        ⚠️ DEBUG: フィルタリング後データ不足 - {len(filtered_data)}点")
        return {"value": None, "series": [], "change_rate": None}
    
    # 最低点数を保証するため、必要に応じて開始日を後退
    if len(filtered_data) < min_points:
        # より古いデータを取得
        extended_start = latest_date - timedelta(days=anchor_days * 1.5)
        extended_data = [(d, y) for d, y in zip(dates, y_data) if d >= extended_start]
        if len(extended_data) >= min_points:
            filtered_data = extended_data
            print(f"        🔍 DEBUG: 6Wデータ点数保証: {len(filtered_data)}点 (拡張期間使用)")
    
    # 最新値と6W前の値で変化率を計算
    latest_value = filtered_data[0][1]
    period_start_value = filtered_data[-1][1]
    print(f"        🔍 DEBUG: 変化率計算 - latest_value={latest_value}, period_start_value={period_start_value}")
    
    if period_start_value and period_start_value != 0:
        change_rate = ((latest_value / period_start_value) - 1) * 100
        print(f"        🔍 DEBUG: 変化率計算結果 - change_rate={change_rate}")
    else:
        change_rate = None
        print(f"        ⚠️ DEBUG: 変化率計算失敗 - period_start_value={period_start_value}")
    
    # 系列データ（最低点数を保証）
    series = [y for _, y in filtered_data]
    if len(series) < min_points:
        # 最後の値で埋める
        last_value = series[-1] if series else 0
        while len(series) < min_points:
            series.append(last_value)
        print(f"        🔍 DEBUG: 6Wデータ補完: {len(series)}点 (最後の値で埋め込み)")
    
    result = {
        "value": latest_value,
        "series": series[:min_points],
        "change_rate": change_rate
    }
    print(f"        🔍 DEBUG: extract_6w_data完了 - value={result['value']}, series_len={len(result['series'])}, change_rate={result['change_rate']}")
    
    return result


def build_macro_snapshot(asof_date, output_dir):
    """
    マクロ経済指標のスナップショットを構築する（SSOT HTML読み出し版）
    
    Args:
        asof_date: 基準日
        output_dir: 出力ディレクトリ（HTMLファイルの場所）
        
    Returns:
        dict: KPIスナップショット
    """
    import json
    import os
    from pathlib import Path
    from datetime import date
    
    print("DEBUG: build_macro_snapshot function called!")
    print(f"[MACRO] build_macro_snapshot({asof_date=}, {output_dir=}) from {__file__}")
    
    # Define the 12 KPI keys we need (no old keys)
    KPI_KEYS = [
        "vix_6w", "sp500_6w", "eq_norm_6w", "dxy_6w", "usdjpy_6w", "gold_6w",
        "us10y", "yield_spread", "ff_rate", "cpi_yoy", "econ_score_short", "econ_score_long"
    ]
    
    # Check required files exist
    def must_exist(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[MACRO] missing source file: {path}")
    
    required_files = [
        "vix_vs_sp500.html",
        "gold_dollar_yen_6w.html", 
        "economic_subplots.html",
        "interest_rates.html",
        "normalized_indices_6w.html"
    ]
    
    for f in required_files:
        must_exist(os.path.join(output_dir, f))
    
    # Trace name patterns for matching (updated to match actual HTML trace names)
    NAME_PATTERNS = {
        "DXY": r"(?:ドル指数|DXY|Dollar)",
        "USDJPY": r"(?:ドル円|USDJPY|USD/JPY)",
        "GCUSD": r"(?:金価格|Gold|XAU|金)",
        "10Y": r"(?:10年債利回り|^10Y$|10年)",
        "2Y": r"(?:2年債利回り|^2Y$|2年)",
        "FF": r"(?:FF金利|^FF$|Fed|政策金利)",
        "CPI_YOY": r"(?:CPI総合|CPI.*YoY|前年比)",
        "VIX": r"(?:VIX.*恐怖指数.*|^VIX$)",
        "SP500": r"(?:S&P ?500|SPX)",
        "NIKKEI": r"(?:日経平均|Nikkei|N225)",
        "ECON": r"(?:経済スコア|Economic\s*Score)",
        "YIELD_SPREAD": r"(?:イールドスプレッド|Yield\s*Spread|スプレッド)"
    }
    
    # Extract KPI data from HTML files
    kpis = {}
    
    # VIX (level)
    vix_data = extract_plotly_data(os.path.join(output_dir, "vix_vs_sp500.html"), "VIX（恐怖指数）", NAME_PATTERNS["VIX"])
    if vix_data and vix_data.get("found") and vix_data.get("y"):
        vix_series = vix_data["y"]
        vix_value = vix_series[-1] if vix_series else None
        kpis["vix_6w"] = {"value": vix_value, "series": vix_series}
        print(f"[MACRO] KPI vix_6w src=vix_vs_sp500.html match={vix_data.get('matched_trace')} series_len={len(vix_series)} value={vix_value}")
    else:
        kpis["vix_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI vix_6w src=vix_vs_sp500.html match=None series_len=0 value=None")
    
    # S&P500 (6W change rate) - CORRECTED: Use normalized_indices_6w.html
    sp500_data = extract_plotly_data(os.path.join(output_dir, "normalized_indices_6w.html"), "S&P 500", NAME_PATTERNS["SP500"])
    if sp500_data and sp500_data.get("found") and sp500_data.get("y"):
        sp500_series = sp500_data["y"]
        if len(sp500_series) >= 2:
            # The data in normalized_indices_6w.html is already percentage changes (normalized to 0% at start)
            # The last value is the 6W change rate, convert from decimal to percentage
            last_val = sp500_series[-1]
            change_rate = last_val * 100  # Convert from decimal (0.0245) to percentage (2.45%)
            kpis["sp500_6w"] = {"value": change_rate, "series": sp500_series}
            print(f"[MACRO] KPI sp500_6w src=normalized_indices_6w.html match={sp500_data.get('matched_trace')} series_len={len(sp500_series)} value={change_rate}")
        else:
            kpis["sp500_6w"] = {"value": None, "series": []}
            print(f"[MACRO] KPI sp500_6w src=normalized_indices_6w.html match={sp500_data.get('matched_trace')} series_len={len(sp500_series)} value=None (insufficient data)")
    else:
        kpis["sp500_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI sp500_6w src=normalized_indices_6w.html match=None series_len=0 value=None")
    
    # Normalized Stock Price (6W change rate) - Use 日経平均 from normalized_indices_6w.html
    eq_norm_data = extract_plotly_data(os.path.join(output_dir, "normalized_indices_6w.html"), "日経平均", NAME_PATTERNS["NIKKEI"])
    if eq_norm_data and eq_norm_data.get("found") and eq_norm_data.get("y"):
        eq_norm_series = eq_norm_data["y"]
        if len(eq_norm_series) >= 2:
            # The data in normalized_indices_6w.html is already percentage changes (normalized to 0% at start)
            # The last value is the 6W change rate, convert from decimal to percentage
            last_val = eq_norm_series[-1]
            change_rate = last_val * 100  # Convert from decimal (0.0245) to percentage (2.45%)
            kpis["eq_norm_6w"] = {"value": change_rate, "series": eq_norm_series}
            print(f"[MACRO] KPI eq_norm_6w src=normalized_indices_6w.html match={eq_norm_data.get('matched_trace')} series_len={len(eq_norm_series)} value={change_rate}")
        else:
            kpis["eq_norm_6w"] = {"value": None, "series": []}
            print(f"[MACRO] KPI eq_norm_6w src=normalized_indices_6w.html match={eq_norm_data.get('matched_trace')} series_len={len(eq_norm_series)} value=None (insufficient data)")
    else:
        kpis["eq_norm_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI eq_norm_6w src=normalized_indices_6w.html match=None series_len=0 value=None")
    
    # DXY (6W change rate)
    dxy_data = extract_plotly_data(os.path.join(output_dir, "gold_dollar_yen_6w.html"), "ドル指数", NAME_PATTERNS["DXY"])
    if dxy_data and dxy_data.get("found") and dxy_data.get("y"):
        dxy_series = dxy_data["y"]
        if len(dxy_series) >= 1:
            # ドル指数は基本的に0なので、最後の値（raw value）をそのまま使用
            last_val = dxy_series[-1]
            kpis["dxy_6w"] = {"value": last_val, "series": dxy_series}
            print(f"[MACRO] KPI dxy_6w src=gold_dollar_yen_6w.html match={dxy_data.get('matched_trace')} series_len={len(dxy_series)} value={last_val} (raw value - DXY is typically 0)")
        else:
            kpis["dxy_6w"] = {"value": None, "series": []}
            print(f"[MACRO] KPI dxy_6w src=gold_dollar_yen_6w.html match={dxy_data.get('matched_trace')} series_len={len(dxy_series)} value=None (insufficient data)")
    else:
        kpis["dxy_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI dxy_6w src=gold_dollar_yen_6w.html match=None series_len=0 value=None")
    
    # USDJPY (6W change rate)
    usdjpy_data = extract_plotly_data(os.path.join(output_dir, "gold_dollar_yen_6w.html"), "ドル円", NAME_PATTERNS["USDJPY"])
    if usdjpy_data and usdjpy_data.get("found") and usdjpy_data.get("y"):
        usdjpy_series = usdjpy_data["y"]
        if len(usdjpy_series) >= 1:
            # gold_dollar_yen_6w.htmlのデータは既に正規化されたパーセンテージ変化率
            # 最後の値（raw value）を100倍してパーセンテージに変換
            last_val = usdjpy_series[-1]
            percentage_value = last_val * 100  # 0.01 → 1.0%
            kpis["usdjpy_6w"] = {"value": percentage_value, "series": usdjpy_series}
            print(f"[MACRO] KPI usdjpy_6w src=gold_dollar_yen_6w.html match={usdjpy_data.get('matched_trace')} series_len={len(usdjpy_series)} value={percentage_value}% (converted from normalized data)")
        else:
            kpis["usdjpy_6w"] = {"value": None, "series": []}
            print(f"[MACRO] KPI usdjpy_6w src=gold_dollar_yen_6w.html match={usdjpy_data.get('matched_trace')} series_len={len(usdjpy_series)} value=None (insufficient data)")
    else:
        kpis["usdjpy_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI usdjpy_6w src=gold_dollar_yen_6w.html match=None series_len=0 value=None")
    
    # Gold (6W change rate) - try multiple sources
    gold_data = None
    gold_sources = [
        ("gold_dollar_yen_6w.html", "金価格"),  # 6WグラフではGCUSDデータを使用
        ("economic_subplots.html", "Gold"),
        ("gold_dollar_yen.html", "金価格")  # 長期グラフではGOLDデータを使用
    ]
    
    for source_file, trace_name in gold_sources:
        source_path = os.path.join(output_dir, source_file)
        if os.path.exists(source_path):
            gold_data = extract_plotly_data(source_path, trace_name, NAME_PATTERNS["GCUSD"])
            if gold_data and gold_data.get("found") and gold_data.get("y"):
                break
    
    if gold_data and gold_data.get("found") and gold_data.get("y"):
        gold_series = gold_data["y"]
        if len(gold_series) >= 1:
            # gold_dollar_yen_6w.htmlのデータは既に正規化されたパーセンテージ変化率
            # 最後の値（raw value）を100倍してパーセンテージに変換
            last_val = gold_series[-1]
            percentage_value = last_val * 100  # 0.05 → 5.0%
            kpis["gold_6w"] = {"value": percentage_value, "series": gold_series}
            print(f"[MACRO] KPI gold_6w src={gold_data.get('source_file')} match={gold_data.get('matched_trace')} series_len={len(gold_series)} value={percentage_value}% (converted from normalized data)")
        else:
            kpis["gold_6w"] = {"value": None, "series": []}
            print(f"[MACRO] KPI gold_6w src={gold_data.get('source_file')} match={gold_data.get('matched_trace')} series_len={len(gold_series)} value=None (insufficient data)")
    else:
        kpis["gold_6w"] = {"value": None, "series": []}
        print(f"[MACRO] KPI gold_6w src=None match=None series_len=0 value=None (not found)")
    
    # US10Y (level)
    us10y_data = extract_plotly_data(os.path.join(output_dir, "economic_subplots.html"), "10年債利回り", NAME_PATTERNS["10Y"])
    if us10y_data and us10y_data.get("found") and us10y_data.get("y"):
        us10y_series = us10y_data["y"]
        us10y_value = us10y_series[-1] if us10y_series else None
        if us10y_value is not None:
            # Convert to percentage if needed
            us10y_value = round(us10y_value, 2)
        kpis["us10y"] = {"value": us10y_value, "series": us10y_series}
        print(f"[MACRO] KPI us10y src=economic_subplots.html match={us10y_data.get('matched_trace')} series_len={len(us10y_series)} value={us10y_value}")
    else:
        kpis["us10y"] = {"value": None, "series": []}
        print(f"[MACRO] KPI us10y src=economic_subplots.html match=None series_len=0 value=None")
    
    # Yield Spread (level) - extract directly from economic_subplots.html
    yield_spread_data = extract_plotly_data(os.path.join(output_dir, "economic_subplots.html"), "イールドスプレッド", NAME_PATTERNS["YIELD_SPREAD"])
    if yield_spread_data and yield_spread_data.get("found") and yield_spread_data.get("y"):
        yield_spread_series = yield_spread_data["y"]
        yield_spread_value = yield_spread_series[-1] if yield_spread_series else None
        if yield_spread_value is not None:
            yield_spread_value = round(yield_spread_value, 2)
        kpis["yield_spread"] = {"value": yield_spread_value, "series": yield_spread_series}
        print(f"[MACRO] KPI yield_spread src=economic_subplots.html match={yield_spread_data.get('matched_trace')} series_len={len(yield_spread_series)} value={yield_spread_value}")
    else:
        kpis["yield_spread"] = {"value": None, "series": []}
        print(f"[MACRO] KPI yield_spread src=economic_subplots.html match=None series_len=0 value=None")
    
    # FF Rate (level)
    ff_data = extract_plotly_data(os.path.join(output_dir, "economic_subplots.html"), "FF金利", NAME_PATTERNS["FF"])
    if ff_data and ff_data.get("found") and ff_data.get("y"):
        ff_series = ff_data["y"]
        ff_value = ff_series[-1] if ff_series else None
        if ff_value is not None:
            ff_value = round(ff_value, 2)
        kpis["ff_rate"] = {"value": ff_value, "series": ff_series}
        print(f"[MACRO] KPI ff_rate src=economic_subplots.html match={ff_data.get('matched_trace')} series_len={len(ff_series)} value={ff_value}")
    else:
        kpis["ff_rate"] = {"value": None, "series": []}
        print(f"[MACRO] KPI ff_rate src=economic_subplots.html match=None series_len=0 value=None")
    
    # CPI YoY (YoY rate) - Use CPI総合 from inflation.html
    cpi_data = extract_plotly_data(os.path.join(output_dir, "inflation.html"), "CPI総合", r"(?:CPI総合)")
    if cpi_data and cpi_data.get("found") and cpi_data.get("y"):
        cpi_series = cpi_data["y"]
        # Find the latest non-NaN value
        cpi_value = None
        for val in reversed(cpi_series):
            if val is not None and not math.isnan(val):
                cpi_value = val
                break
        if cpi_value is not None:
            cpi_value = round(cpi_value, 2)
        kpis["cpi_yoy"] = {"value": cpi_value, "series": cpi_series}
        print(f"[MACRO] KPI cpi_yoy src=inflation.html match={cpi_data.get('matched_trace')} series_len={len(cpi_series)} value={cpi_value}")
    else:
        kpis["cpi_yoy"] = {"value": None, "series": []}
        print(f"[MACRO] KPI cpi_yoy src=inflation.html match=None series_len=0 value=None")
    
    # Economic Score Short (level) - Use economic_score.html
    econ_short_data = extract_plotly_data(os.path.join(output_dir, "economic_score.html"), "Economic Score", NAME_PATTERNS["ECON"])
    if econ_short_data and econ_short_data.get("found") and econ_short_data.get("y"):
        econ_short_series = econ_short_data["y"]
        econ_short_value = econ_short_series[-1] if econ_short_series else None
        if econ_short_value is not None:
            econ_short_value = round(econ_short_value, 2)
        kpis["econ_score_short"] = {"value": econ_short_value, "series": econ_short_series}
        print(f"[MACRO] KPI econ_score_short src=economic_score.html match={econ_short_data.get('matched_trace')} series_len={len(econ_short_series)} value={econ_short_value}")
    else:
        kpis["econ_score_short"] = {"value": None, "series": []}
        print(f"[MACRO] KPI econ_score_short src=economic_score.html match=None series_len=0 value=None")
    
    # Economic Score Long (level) - Use economic_score_long.html
    econ_long_data = extract_plotly_data(os.path.join(output_dir, "economic_score_long.html"), "Economic Score", NAME_PATTERNS["ECON"])
    if econ_long_data and econ_long_data.get("found") and econ_long_data.get("y"):
        econ_long_series = econ_long_data["y"]
        econ_long_value = econ_long_series[-1] if econ_long_series else None
        if econ_long_value is not None:
            econ_long_value = round(econ_long_value, 2)
        kpis["econ_score_long"] = {"value": econ_long_value, "series": econ_long_series}
        print(f"[MACRO] KPI econ_score_long src=economic_score_long.html match={econ_long_data.get('matched_trace')} series_len={len(econ_long_series)} value={econ_long_value}")
    else:
        kpis["econ_score_long"] = {"value": None, "series": []}
        print(f"[MACRO] KPI econ_score_long src=economic_score_long.html match=None series_len=0 value=None")
    
    # Filter to only include the 12 specified KPI keys
    kpis = {k: kpis.get(k) for k in KPI_KEYS}
    
    # Build snapshot
    snapshot = {
        "asof": asof_date.isoformat(),
        "kpis": kpis,
        "heatmap": []  # Initially empty as requested
    }
    
    # Clean the snapshot to prevent NaN values in JSON
    snapshot = _clean_json_safe(snapshot)
    
    print(f"[MACRO] build_macro_snapshot completed with {len(kpis)} KPIs")
    return snapshot


def render_macro_overview(snapshot):
    """
    マクロ俯瞰サマリーのHTMLを生成する（重要度順の並び順対応）
    
    Args:
        snapshot: build_macro_snapshotで生成されたスナップショット
        
    Returns:
        str: HTMLコンテンツ
    """
    import json
    
    # 重要度順の並び順（要件書に従った提案順）
    KPI_ORDER = [
        "vix_6w", "sp500_6w", "eq_norm_6w", "dxy_6w", "usdjpy_6w", "gold_6w",
        "us10y", "yield_spread", "ff_rate", "cpi_yoy", "econ_score_short", "econ_score_long"
    ]
    
    # KPIタイルの定義（期間表記を明示）
    KPI_META = {
        "vix_6w": {"label": "VIX(6W, 水準)", "href": "vix_vs_sp500.html", "polarity": "neg", "unit": ""},
        "sp500_6w": {"label": "S&P500(6W変化率)", "href": "normalized_indices_6w.html", "polarity": "pos", "unit": "%"},
        "eq_norm_6w": {"label": "日経平均(6W変化率)", "href": "normalized_indices_6w.html", "polarity": "pos", "unit": "%"},
        "dxy_6w": {"label": "ドル指数(6W変化率)", "href": "gold_dollar_yen_6w.html", "polarity": "neg", "unit": "%"},
        "usdjpy_6w": {"label": "USDJPY(6W変化率)", "href": "gold_dollar_yen_6w.html", "polarity": "neg", "unit": "%"},
        "gold_6w": {"label": "金価格(6W変化率)", "href": "gold_dollar_yen_6w.html", "polarity": "neg", "unit": "%"},
        "us10y": {"label": "米国10年債利回り(水準)", "href": "interest_rates.html", "polarity": "neg", "unit": "%"},
        "yield_spread": {"label": "米国イールドスプレッド(水準)", "href": "interest_rates.html", "polarity": "pos", "unit": "%"},
        "ff_rate": {"label": "米国政策金利(水準)", "href": "interest_rates.html", "polarity": "neg", "unit": "%"},
        "cpi_yoy": {"label": "CPI 前年比(前年比)", "href": "inflation.html", "polarity": "neg", "unit": "%"},
        "econ_score_short": {"label": "経済スコア(短期, 水準)", "href": "economic_score.html", "polarity": "pos", "unit": ""},
        "econ_score_long": {"label": "経済スコア(長期, 水準)", "href": "economic_score_long.html", "polarity": "pos", "unit": ""}
    }
    
    # KPIタイルのHTMLを生成（重要度順）
    kpi_tiles_html = ""
    for key in KPI_ORDER:
        if key not in KPI_META:
            continue
            
        config = KPI_META[key]
        kpi_data = snapshot["kpis"].get(key, {})
        value = kpi_data.get("value")
        unit = kpi_data.get("unit", "")
        series = kpi_data.get("series", [])
        polarity = kpi_data.get("polarity", "pos")
        
        # 値の表示（Noneの場合はN/Aに変更）
        if value is not None:
            value_display = f"{value}{unit}"
        else:
            value_display = "N/A"
        
        # スパークラインのID
        spark_id = f"spark-{key}"
        
        kpi_tiles_html += f"""
        <a class="kpi-tile" data-key="{key}" href="{config['href']}">
            <div class="kpi-label">{config['label']}</div>
            <div class="kpi-value">{value_display}</div>
            <div class="kpi-spark" id="{spark_id}"></div>
        </a>
        """
    
    # スナップショットをJSONに変換（HTMLエスケープを避ける）
    snapshot_json = json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))
    
    # HTML全体を生成
    html = f"""
    <section id="macro-overview">
        <h2>マクロ俯瞰サマリー</h2>
        <div class="kpi-grid">
            {kpi_tiles_html}
        </div>
        
        <!-- 簡易ヒートマップ（初期OFF） -->
        <div class="quick-heatmaps" id="heatmap-container" style="display: none;">
            <div class="heatmap-section">
                <h3>セクター動向</h3>
                <div id="heatmap-sectors" class="mini-chart"></div>
            </div>
            <div class="heatmap-section">
                <h3>資産動向</h3>
                <div id="heatmap-assets" class="mini-chart"></div>
            </div>
        </div>
        
        <!-- スナップショットデータを埋め込み -->
        <script id="macro-snapshot" type="application/json">{snapshot_json}</script>
        
        <!-- デバッグ出力 -->
        <script>
        try{{
            const el = document.getElementById('macro-snapshot');
            if(!el) throw new Error('macro-snapshot not found');
            // Fallback for NaN/Infinity values (temporary until generation is fixed)
            const raw = el.textContent.replace(/\\bNaN\\b/g,'null')
                                      .replace(/\\bInfinity\\b/g,'null')
                                      .replace(/\\b-Infinity\\b/g,'null');
            const s = JSON.parse(raw);
            const keys = ["vix_6w","sp500_6w","eq_norm_6w","dxy_6w","usdjpy_6w","gold_6w","us10y","yield_spread","ff_rate","cpi_yoy","econ_score_short","econ_score_long"];
            const missing = keys.filter(k=>!s.kpis[k] || s.kpis[k].value==null || !Array.isArray(s.kpis[k].series));
            if(missing.length) console.warn("[macro] missing kpis:", missing);
            else console.log("[macro] all kpis loaded successfully");
        }}catch(e){{ console.error("[macro] snapshot parse error", e); }}
        </script>
        
        <!-- スパークライン描画JS -->
        <script>
        (function(){{
            let snap;
            try {{
            const el = document.getElementById('macro-snapshot');
                if(!el) throw new Error('macro-snapshot not found');
                // Fallback for NaN/Infinity values (temporary until generation is fixed)
                const raw = el.textContent.replace(/\\bNaN\\b/g,'null')
                                          .replace(/\\bInfinity\\b/g,'null')
                                          .replace(/\\b-Infinity\\b/g,'null');
                snap = JSON.parse(raw);
                if(!snap || !snap.kpis) throw new Error('snapshot.kpis missing');
            }} catch(e) {{
                console.error('[macro] snapshot load error:', e);
                return; // N/Aのまま（原因がconsoleに出る）
            }}
            
            // KPIタイルの更新（重要度順）
            const kpiOrder = ["vix_6w","sp500_6w","eq_norm_6w","dxy_6w","usdjpy_6w","gold_6w","us10y","yield_spread","ff_rate","cpi_yoy","econ_score_short","econ_score_long"];
            
            kpiOrder.forEach(key => {{
                const obj = snap.kpis[key];
                if(!obj) return;
                
                const tile = document.querySelector(`.kpi-tile[data-key="${{key}}"]`);
                const val = tile?.querySelector('.kpi-value');
                if(val) {{
                    val.textContent = (obj.value ?? 'N/A') + (obj.unit || '');
                }}
                
                // トレンドの色分け（値がNoneの場合は色付けしない）
                if(tile && obj.value !== null){{
                    // 特別な色判定
                    let isPositive;
                    if(key === 'dxy_6w') {{
                        // ドル指数：0以上なら緑、0未満なら赤
                        isPositive = obj.value >= 0;
                    }} else if(key === 'gold_6w') {{
                        // 金価格：0.1以上なら赤（危険）、0以下なら緑（安全）
                        isPositive = obj.value <= 0;
                    }} else {{
                        // その他のKPIは通常の判定（プラスなら緑、マイナスなら赤）
                        isPositive = obj.value > 0;
                    }}
                    tile.classList.add(isPositive ? 'positive' : 'negative');
                }}
                
                // スパークライン描画（最低2点以上で描画）
                const sparkId = "spark-" + key;
                const sp = document.getElementById(sparkId);
                if(sp && obj.series && obj.series.length >= 2){{
                    Plotly.newPlot(sp, [{{
                        x: [...Array(obj.series.length).keys()],
                        y: obj.series, 
                        type:'scatter', 
                        mode:'lines', 
                        line:{{width:2, color: (function() {{
                            let isPositive;
                            if(key === 'dxy_6w') {{
                                isPositive = obj.value >= 0;
                            }} else if(key === 'gold_6w') {{
                                isPositive = obj.value <= 0;
                            }} else {{
                                isPositive = obj.value > 0;
                            }}
                            return isPositive ? '#26a69a' : '#ef5350';
                        }})()}}
                    }}], {{
                        margin:{{l:16,r:8,t:6,b:12}}, 
                        xaxis:{{visible:false}}, 
                        yaxis:{{visible:false}}
                    }}, {{displayModeBar:false}});
                }} else if(sp) {{
                    // データ不足時はグレー表示
                    sp.style.backgroundColor = '#f5f5f5';
                    sp.style.border = '1px solid #ddd';
                    sp.innerHTML = '<div style="text-align:center;color:#999;font-size:12px;padding:10px;">データ不足</div>';
                }}
            }});
            
            // ヒートマップ描画
            function heat(target, items, key){{
                if(!items?.length) return;
                const values = items.map(d => d[key] ?? 0);
                const names = items.map(d => d.name);
                
                Plotly.newPlot(target, [{{
                    z:[values], 
                    x:names, 
                    y:[''], 
                    type:'heatmap',
                    colorscale: [[0, '#ef5350'], [0.5, '#fff'], [1, '#26a69a']],
                    showscale: false
                }}], {{
                    margin:{{l:18,r:8,t:10,b:24}}, 
                    xaxis:{{tickangle:-30}}, 
                    yaxis:{{visible:false}}
                }}, {{displayModeBar:false}});
            }}
            
            // ヒートマップ表示制御（データがある場合のみ表示）
            const heatmapContainer = document.getElementById('heatmap-container');
            if(snap.heatmap?.sectors?.length > 0 || snap.heatmap?.assets?.length > 0) {{
                heatmapContainer.style.display = 'block';
            heat('heatmap-sectors', snap.heatmap?.sectors, 'd1');
            heat('heatmap-assets', snap.heatmap?.assets, 'w1');
            }}
        }})();
        </script>
        
        <!-- 将来拡張用LLMコメント（非表示） -->
        <!-- <div id="llm-comment" style="display:none"></div> -->
    </section>
    """
    
    return html


def generate_market_score_html(df_macro: pd.DataFrame, macro_components: Dict, df_micro: pd.DataFrame,
                              sparkline_data: Optional[Dict] = None, macro_snapshot: Optional[Dict] = None,
                              engine = None) -> str:
    """
    市場スコアのHTMLレポートを生成します。

    Parameters
    ----------
    df_macro : pd.DataFrame
        マクロスコアを含むデータフレーム
    macro_components : Dict
        各マクロ指標の寄与度
    df_micro : pd.DataFrame
        ミクロスコア（個別銘柄評価）を含むデータフレーム
    sparkline_data : Dict, optional
        スパークラインデータ
    macro_snapshot : Dict, optional
        マクロスナップショットデータ
    engine : sqlalchemy.engine.Engine, optional
        データベース接続エンジン（ランキング表示用）

    Returns
    -------
    str
        HTMLレポート
    """
    # 最新日付
    latest_date = df_macro['date'].max()
    date_str = pd.to_datetime(latest_date).strftime('%Y-%m-%d')
    
    # 銘柄名とフィルタ情報を取得する関数
    def get_stock_info_with_filters(engine, symbols: List[str], target_date: str = None) -> Dict:
        """銘柄名とフィルタ情報、ランキング情報を取得"""
        from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
        from investment_analysis.analysis.score_analysis import get_market_global_ranking
        
        if not symbols:
            return {}
        
        if target_date is None:
            target_date = date_str
        
        symbols_str = "', '".join(symbols)
        
        query = text(f"""
        SELECT DISTINCT
            ds.symbol,
            COALESCE(cp.company_name, ds.symbol) as company_name,
            ds.is_value_trap_filtered,
            ds.is_quality_growth_filtered
        FROM backtest_results.daily_scores ds
        LEFT JOIN fmp_data.company_profile cp ON ds.symbol = cp.symbol
        WHERE ds.symbol IN ('{symbols_str}')
        AND ds.date = (
            SELECT MAX(date) 
            FROM backtest_results.daily_scores 
            WHERE symbol = ds.symbol
            AND date <= '{target_date}'
        )
        """)
        
        try:
            # SQLAlchemy エンジンを作成
            SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            db_engine = create_engine(SQLALCHEMY_DATABASE_URI)
            
            with db_engine.connect() as conn:
                result = pd.read_sql_query(query, conn)
            
            # 辞書形式で返す
            stock_info = {}
            for _, row in result.iterrows():
                # company_nameがNoneまたは空の場合はシンボルを使用
                company_name = row['company_name'] if row['company_name'] and str(row['company_name']).strip() else row['symbol']
                
                # フラグの意味（反転後）：
                # - is_value_trap_filtered = True → バリュートラップ問題有り、フラグアクティブ（警告表示）
                # - is_quality_growth_filtered = True → 品質グロース問題有り、フラグアクティブ（警告表示）
                # - False = 問題なし、フラグ非アクティブ（グレー表示）
                is_value_problematic = bool(row['is_value_trap_filtered'])
                is_quality_problematic = bool(row['is_quality_growth_filtered'])
                
                # 市場タイプを判定（日本株 vs 米国株）
                symbol = row['symbol']
                if symbol.endswith('.T') or (symbol.isdigit() and len(symbol) == 4):
                    market_type = 'JP'
                else:
                    market_type = 'US'
                
                # ランキング情報を取得
                ranking_info = get_market_global_ranking(db_engine, symbol, market_type, target_date)
                
                stock_info[row['symbol']] = {
                    'company_name': company_name,
                    'is_value_filtered': is_value_problematic,
                    'is_quality_filtered': is_quality_problematic,
                    'market_type': market_type,
                    'rank': ranking_info.get('rank', 0),
                    'total_stocks': ranking_info.get('total_stocks', 0)
                }
            
            # 取得できなかった銘柄については、デフォルト値を設定
            for symbol in symbols:
                if symbol not in stock_info:
                    # 簡単な銘柄名推測（ETFなど）
                    if symbol in ['VTI', 'ETF']:
                        company_name = f"{symbol} ETF"
                    else:
                        company_name = symbol
                    
                    # 市場タイプを判定（日本株 vs 米国株）
                    if symbol.endswith('.T') or (symbol.isdigit() and len(symbol) == 4):
                        market_type = 'JP'
                    else:
                        market_type = 'US'
                    
                    stock_info[symbol] = {
                        'company_name': company_name,
                        'is_value_filtered': False,  # デフォルトは通過（良い状態）
                        'is_quality_filtered': False,
                        'market_type': market_type,
                        'rank': 0,
                        'total_stocks': 0
                    }
            
            return stock_info
        except Exception as e:
            print(f"⚠️ 銘柄情報取得エラー: {e}")
            
            # エラー時は銘柄名を推測してデフォルト値を返す
            stock_info = {}
            for symbol in symbols:
                # 簡単な銘柄名推測
                if symbol == 'VTI':
                    company_name = "Vanguard Total Stock Market ETF"
                elif symbol == 'ADBE':
                    company_name = "Adobe Inc."
                elif symbol == 'COKE':
                    company_name = "Coca-Cola Consolidated Inc."
                elif '.T' in symbol:
                    company_name = f"{symbol.replace('.T', '')} (Japanese Stock)"
                else:
                    company_name = symbol
                
                # 市場タイプを判定（日本株 vs 米国株）
                if symbol.endswith('.T') or (symbol.isdigit() and len(symbol) == 4):
                    market_type = 'JP'
                else:
                    market_type = 'US'
                
                stock_info[symbol] = {
                    'company_name': company_name,
                    'is_value_filtered': False,  # デフォルトは通過（良い状態）
                    'is_quality_filtered': False,
                    'market_type': market_type,
                    'rank': 0,
                    'total_stocks': 0
                }
            
            return stock_info
    
    # マクロスコアの最新値
    macro_score_sum = sum(macro_components.values())
    
    # ミクロスコアの平均値
    if not df_micro.empty:
        avg_micro_score = df_micro['total_score'].mean()
        micro_scores_html = ""
        
        # 銘柄情報を取得
        symbols = df_micro['symbol'].unique().tolist()
        stock_info = get_stock_info_with_filters(None, symbols, date_str)
        
        # プロットを生成するための準備
        chart_scripts = ""
        
        # ミクロスコアテーブルを作成（フラグ用CSSを追加）
        micro_scores_html += """
        <style>
        .micro-table .positive { color: green; font-weight: bold; }
        .micro-table .negative { color: red; font-weight: bold; }
        .micro-table .strong.positive { color: #006400; font-weight: bold; }
        .micro-table .strong.negative { color: #b22222; font-weight: bold; }
        
        /* 銘柄ヘッダー用のスタイル */
        .stock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .stock-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        
        .ranking-info {
            font-size: 0.9em;
            color: #666;
            background-color: #f8f9fa;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            white-space: nowrap;
        }
        
        .filter-flags {
            display: flex;
            gap: 8px;
        }
        
        .flag {
            display: flex;
            align-items: flex-start;
            margin-right: 15px;
            cursor: default;
        }
        
        .flag-pole {
            width: 3px;
            height: 35px;
            background-color: #666;
            border-radius: 1px;
            z-index: 2;
            position: relative;
        }
        
        .flag-cloth {
            width: 80px;
            height: 25px;
            background-color: #ddd;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 8px;
            font-weight: bold;
            color: #999;
            position: relative;
            margin-left: 0;
            /* 波打つ形状をclip-pathで作成 */
            clip-path: polygon(
                0% 0%, 
                85% 0%, 
                95% 15%, 
                85% 30%, 
                95% 45%, 
                85% 60%, 
                95% 75%, 
                85% 90%, 
                100% 100%, 
                0% 100%
            );
            /* グラデーションで立体感を演出 */
            background: linear-gradient(135deg, 
                #ddd 0%, 
                #bbb 50%,
                #ddd 100%
            );
            border: 1px solid rgba(0, 0, 0, 0.1);
            /* 風でなびくアニメーション */
            animation: flagWave 3s ease-in-out infinite;
            transform-origin: left center;
        }

        @keyframes flagWave {
            0%, 100% { 
                transform: rotateY(0deg) rotateZ(0deg);
                clip-path: polygon(
                    0% 0%, 
                    85% 0%, 
                    95% 15%, 
                    85% 30%, 
                    95% 45%, 
                    85% 60%, 
                    95% 75%, 
                    85% 90%, 
                    100% 100%, 
                    0% 100%
                );
            }
            25% { 
                transform: rotateY(5deg) rotateZ(1deg);
                clip-path: polygon(
                    0% 0%, 
                    88% 5%, 
                    92% 20%, 
                    88% 35%, 
                    92% 50%, 
                    88% 65%, 
                    92% 80%, 
                    88% 95%, 
                    100% 100%, 
                    0% 100%
                );
            }
            50% { 
                transform: rotateY(0deg) rotateZ(0deg);
                clip-path: polygon(
                    0% 0%, 
                    90% 0%, 
                    85% 15%, 
                    90% 30%, 
                    85% 45%, 
                    90% 60%, 
                    85% 75%, 
                    90% 90%, 
                    100% 100%, 
                    0% 100%
                );
            }
            75% { 
                transform: rotateY(-3deg) rotateZ(-0.5deg);
                clip-path: polygon(
                    0% 0%, 
                    87% 2%, 
                    97% 18%, 
                    87% 32%, 
                    97% 48%, 
                    87% 62%, 
                    97% 78%, 
                    87% 92%, 
                    100% 100%, 
                    0% 100%
                );
            }
        }
        
        .flag.active .flag-cloth {
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .flag.value-active .flag-cloth {
            background: linear-gradient(135deg, 
                #e74c3c 0%, 
                #c0392b 50%,
                #e74c3c 100%
            );
        }
        
        .flag.quality-active .flag-cloth {
            background: linear-gradient(135deg, 
                #f39c12 0%, 
                #d35400 50%,
                #f39c12 100%
            );
        }
        </style>
        """
        # 表形式は削除し、チャートのみを表示
        # 合計スコアでソート
        df_sorted = df_micro.sort_values('total_score', ascending=False)
        
        # 個別銘柄チャートの生成
        micro_scores_html += "<h2>個別銘柄分析チャート</h2>"
        micro_scores_html += "<div class='stock-charts'>"
        
        for symbol in df_sorted['symbol'].unique():
            # 銘柄の詳細情報を取得
            symbol_data = df_sorted[df_sorted['symbol'] == symbol].iloc[0]
            sector = symbol_data['sector']
            
            # 銘柄情報から会社名とフィルタ状況、ランキング情報を取得
            info = stock_info.get(symbol, {
                'company_name': symbol, 
                'is_value_filtered': False, 
                'is_quality_filtered': False,
                'market_type': 'US',
                'rank': 0,
                'total_stocks': 0
            })
            company_name = info['company_name']
            is_value_filtered = info['is_value_filtered']
            is_quality_filtered = info['is_quality_filtered']
            market_type = info['market_type']
            rank = info['rank']
            total_stocks = info['total_stocks']
            
            # ランキング表示用の文字列を作成
            if rank > 0 and total_stocks > 0:
                ranking_text = f"{rank}/{total_stocks}"
                market_emoji = "🇺🇸" if market_type == 'US' else "🇯🇵"
                ranking_display = f'<div class="ranking-info">{market_emoji} Ranking: {ranking_text}</div>'
            else:
                ranking_display = ''
            
            # Valueフラグのクラス（問題有りの時にアクティブ＝警告表示）
            value_flag_class = "flag value-active active" if is_value_filtered else "flag"
            
            # Qualityフラグのクラス（問題有りの時にアクティブ＝警告表示）
            quality_flag_class = "flag quality-active active" if is_quality_filtered else "flag"
            
            # チャートプレースホルダーを作成
            micro_scores_html += f"<div class='chart-container' id='chart-{symbol}'>"
            micro_scores_html += f"""
            <div class="stock-header">
                <div class="stock-title">【{symbol}】{company_name} - {sector}</div>
                {ranking_display}
                <div class="filter-flags">
                    <div class="{value_flag_class}">
                        <div class="flag-pole"></div>
                        <div class="flag-cloth">VALUE TRAP</div>
                    </div>
                    <div class="{quality_flag_class}">
                        <div class="flag-pole"></div>
                        <div class="flag-cloth">QUALITY TRAP</div>
                    </div>
                </div>
            </div>
            """
            micro_scores_html += f"<div id='ohlc-{symbol}' class='stock-chart'></div>"
            micro_scores_html += "</div>"
            
            # チャート生成のJavaScriptを追加
            if sparkline_data and symbol in sparkline_data:
                data = sparkline_data[symbol]
                dates = [d.strftime('%Y-%m-%d') for d in data['date']]
                opens = data.get('open', [])
                highs = data.get('high', [])
                lows = data.get('low', [])
                closes = data.get('close', [])
                volumes = data.get('volume', [])
                sma20s = data.get('sma20', [])
                sma40s = data.get('sma40', [])
                trade_data = data.get('trade_data', [])
                
                # JavaScript変数名に使用する安全なシンボル名を生成（ピリオドを_に置換）
                js_symbol = symbol.replace('.', '_')
                
                # ローソク足チャートスクリプトを生成
                chart_scripts += f"""
                var candlestick_{js_symbol} = {{
                    x: {json.dumps(dates)},
                    open: {json.dumps(opens)},
                    high: {json.dumps(highs)},
                    low: {json.dumps(lows)},
                    close: {json.dumps(closes)},
                    type: 'candlestick',
                    name: 'Price',
                    increasing: {{line: {{color: '#26a69a'}}}},
                    decreasing: {{line: {{color: '#ef5350'}}}},
                    hovertemplate: '<b>Price</b><br>' +
                                   'Open: %{{open}}<br>' +
                                   'High: %{{high}}<br>' +
                                   'Low: %{{low}}<br>' +
                                   'Close: %{{close}}<extra></extra>'
                }};
                
                var traceSMA20_{js_symbol} = {{
                    x: {json.dumps(dates)},
                    y: {json.dumps(sma20s)},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA20',
                    line: {{ color: 'orange', width: 1.5 }},
                    hovertemplate: '<b>SMA20</b><br>%{{y}}<extra></extra>'
                }};
                
                var traceSMA40_{js_symbol} = {{
                    x: {json.dumps(dates)},
                    y: {json.dumps(sma40s)},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'SMA40',
                    line: {{ color: 'red', width: 1.5 }},
                    hovertemplate: '<b>SMA40</b><br>%{{y}}<extra></extra>'
                }};
                """
                
                # 通貨判定関数
                def format_price_with_currency(price, symbol):
                    """銘柄に基づいて価格を適切な通貨で表示"""
                    if symbol.endswith('.T'):  # 日本株
                        return f"¥{price:.0f}"
                    else:  # 米国株
                        return f"${price:.2f}"
                
                def calculate_percentage(target_price, buy_price):
                    """購入価格に対するパーセンテージを計算"""
                    if buy_price and buy_price > 0:
                        percentage = ((target_price - buy_price) / buy_price) * 100
                        return f"({percentage:+.0f}%)"
                    return ""
                
                # 売買記録トレースを生成
                trade_traces = []
                for i, trade in enumerate(trade_data):
                    buy_date_str = trade['buy_date'].strftime('%Y-%m-%d')
                    stop_loss_price = trade['stop_loss_price']
                    take_profit_price = trade['take_profit_price']
                    buy_price = trade.get('buy_price', 0)  # trade_dataから購入価格を取得
                    
                    # 購入日の縦線（黒い点線）
                    chart_scripts += f"""
                var buyLine_{js_symbol}_{i} = {{
                    x: ['{buy_date_str}', '{buy_date_str}'],
                    y: [0, {max(highs) * 1.1 if highs else 1000}],
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Buy {buy_date_str}',
                    line: {{ color: 'black', width: 2, dash: 'dot' }},
                    showlegend: true,
                    hovertemplate: '<b>Buy Date</b><br>{buy_date_str}<extra></extra>'
                }};
                """
                    trade_traces.append(f"buyLine_{js_symbol}_{i}")
                    
                    # 損切りライン（赤い点線）
                    if stop_loss_price and stop_loss_price > 0:
                        price_str = format_price_with_currency(stop_loss_price, symbol)
                        percentage_str = calculate_percentage(stop_loss_price, buy_price)
                        chart_scripts += f"""
                var stopLossLine_{js_symbol}_{i} = {{
                    x: {json.dumps(dates)},
                    y: {json.dumps([stop_loss_price] * len(dates))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Stop Loss {price_str}{percentage_str}',
                    line: {{ color: 'red', width: 2, dash: 'dash' }},
                    showlegend: true,
                    hovertemplate: '<b>Stop Loss</b><br>{price_str}{percentage_str}<extra></extra>'
                }};
                """
                        trade_traces.append(f"stopLossLine_{js_symbol}_{i}")
                    
                    # 利確ライン（緑の点線）
                    if take_profit_price and take_profit_price > 0:
                        price_str = format_price_with_currency(take_profit_price, symbol)
                        percentage_str = calculate_percentage(take_profit_price, buy_price)
                        chart_scripts += f"""
                var takeProfitLine_{js_symbol}_{i} = {{
                    x: {json.dumps(dates)},
                    y: {json.dumps([take_profit_price] * len(dates))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Take Profit {price_str}{percentage_str}',
                    line: {{ color: 'green', width: 2, dash: 'dash' }},
                    showlegend: true,
                    hovertemplate: '<b>Take Profit</b><br>{price_str}{percentage_str}<extra></extra>'
                }};
                """
                        trade_traces.append(f"takeProfitLine_{js_symbol}_{i}")
                
                # スコアデータの取得
                score_data = sparkline_data.get(symbol, {}).get('score_data', {})
                
                chart_scripts += f"""
                

                
                var traceVolume_{js_symbol} = {{
                    x: {json.dumps(dates)},
                    y: {json.dumps(volumes)},
                    type: 'bar',
                    name: 'Volume',
                    yaxis: 'y2',
                    marker: {{
                        color: 'rgba(0,128,0,0.3)'
                    }},
                    hovertemplate: '<b>Volume</b><br>%{{y:,}}<extra></extra>'
                }};
                
                // スコア推移トレース
                var traceTotalScore_{js_symbol} = {{
                    x: {json.dumps(score_data.get('dates', []))},
                    y: {json.dumps(score_data.get('total_score', []))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Total',
                    line: {{ color: 'purple', width: 2 }},
                    yaxis: 'y3',
                    hovertemplate: '<b>Total</b><br>%{{y:.1f}}<extra></extra>'
                }};
                
                var traceGrowthScore_{js_symbol} = {{
                    x: {json.dumps(score_data.get('dates', []))},
                    y: {json.dumps(score_data.get('growth_score', []))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Growth',
                    line: {{ color: 'green', width: 1.5 }},
                    yaxis: 'y3',
                    hovertemplate: '<b>Growth</b><br>%{{y:.1f}}<extra></extra>'
                }};
                
                var traceQualityScore_{js_symbol} = {{
                    x: {json.dumps(score_data.get('dates', []))},
                    y: {json.dumps(score_data.get('quality_score', []))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Quality',
                    line: {{ color: 'blue', width: 1.5 }},
                    yaxis: 'y3',
                    hovertemplate: '<b>Quality</b><br>%{{y:.1f}}<extra></extra>'
                }};
                
                var traceMomentumScore_{js_symbol} = {{
                    x: {json.dumps(score_data.get('dates', []))},
                    y: {json.dumps(score_data.get('momentum_score', []))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Momentum',
                    line: {{ color: 'orange', width: 1.5 }},
                    yaxis: 'y3',
                    hovertemplate: '<b>Momentum</b><br>%{{y:.1f}}<extra></extra>'
                }};
                
                var traceMacroSectorScore_{js_symbol} = {{
                    x: {json.dumps(score_data.get('dates', []))},
                    y: {json.dumps(score_data.get('macro_sector_score', []))},
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Macro',
                    line: {{ color: 'red', width: 1.5 }},
                    yaxis: 'y3',
                    hovertemplate: '<b>Macro</b><br>%{{y:.1f}}<extra></extra>'
                }};
                """
                
                # Y軸の表示範囲を計算（価格データのMIN/MAX + 10%）
                price_data = highs + lows + closes  # 全価格データを結合
                if price_data:
                    price_min = min(price_data)
                    price_max = max(price_data)
                    price_range = price_max - price_min
                    y_min = price_min - (price_range * 0.1)
                    y_max = price_max + (price_range * 0.1)
                else:
                    y_min, y_max = 0, 1000
                
                chart_scripts += f"""
                var layout_{js_symbol} = {{
                    autosize: true,
                    height: 500,
                    showlegend: true,
                    hovermode: 'x unified',  // 統一ホバー表示（縦並び）
                    xaxis: {{
                        rangeslider: {{ visible: false }},  // スライダーを無効化
                        type: 'date'
                    }},
                    yaxis: {{
                        title: 'Price',
                        domain: [0.5, 1],
                        range: [{y_min:.2f}, {y_max:.2f}]  // Y軸範囲を設定
                    }},
                    yaxis2: {{
                        title: 'Volume',
                        domain: [0.25, 0.45]
                    }},
                    yaxis3: {{
                        title: 'Score',
                        domain: [0, 0.2]
                    }},
                    legend: {{
                        orientation: 'h',
                        y: 1.1
                    }},
                    margin: {{
                        l: 50,
                        r: 50,
                        t: 50,
                        b: 50
                    }}
                }};
                
                Plotly.newPlot('ohlc-{symbol}', [candlestick_{js_symbol}, traceSMA20_{js_symbol}, traceSMA40_{js_symbol}, 
                    traceVolume_{js_symbol}, traceTotalScore_{js_symbol}, traceGrowthScore_{js_symbol}, 
                    traceQualityScore_{js_symbol}, traceMomentumScore_{js_symbol}, traceMacroSectorScore_{js_symbol}, 
                    {', '.join(trade_traces)}], layout_{js_symbol});
                """
            else:
                # データがない場合のエラーメッセージを表示
                micro_scores_html += f"<div class='chart-error'>チャートデータを取得できませんでした</div>"
        
        micro_scores_html += "</div>"
        # 補足説明を追加
        micro_scores_html += """
        <div class='explanation' style='margin-top:20px; color:#555; font-size:0.95em;'>
        <strong>【各項目の算出期間・意味】</strong><br>
        ・<b>Price Change(%)</b>：前日終値との比較（1日変化率）<br>
        ・<b>Sector Dev.(%) / Industry Dev.(%)</b>：当日終値の前日比と、同じセクター/インダストリー内の平均前日比との差（1日変化率の相対値）<br>
        ・<b>Volume Change(%)</b>：当日出来高と直近5営業日の出来高平均との比較（1日出来高 ÷ 5日平均）<br>
        ・<b>GC/DCスコア</b>：ゴールデンクロス/デッドクロス発生からの経過日数と指数減衰で算出<br>
        ・<b>ATR Ratio</b>：当日ATR（14日）÷当日終値（＝14日間の平均的な値動き幅の割合）
        </div>
        """
    else:
        avg_micro_score = 0
        micro_scores_html = "<p>ポートフォリオデータがありません</p>"
        chart_scripts = ""
    
    # マクロ俯瞰サマリーを生成
    if macro_snapshot:
        macro_overview_html = render_macro_overview(macro_snapshot)
    else:
        macro_overview_html = "<div class='macro-overview'><h2>マクロ俯瞰サマリー</h2><p>データを取得中...</p></div>"

    # V2スコアランキング Top10を生成
    ranking_html = ""
    if engine is not None:
        try:
            print("🏆 V2スコアランキングTop10を生成中...")
            from investment_analysis.analysis.daily_report import fetch_daily_top10_rankings

            ranking_data = fetch_daily_top10_rankings(engine)
            df_combined = ranking_data['combined']
            jp_date = ranking_data['jp_date']
            us_date = ranking_data['us_date']

            print(f"  📊 取得件数: {len(df_combined)}件")
            print(f"  📊 カラム: {df_combined.columns.tolist()}")

            # 日付表示用
            if jp_date == us_date:
                ranking_date_display = f"基準日: {jp_date}"
            else:
                ranking_date_display = f"基準日: 🇯🇵{jp_date} / 🇺🇸{us_date}"

            if not df_combined.empty:
                print("  ✅ ランキングデータが存在します。HTML生成開始...")
                rows_html = ""
                for idx, row in df_combined.iterrows():
                    rank_badge_color = "#FFD700" if row['rank'] == 1 else "#C0C0C0" if row['rank'] == 2 else "#CD7F32" if row['rank'] == 3 else "#E8F5E9"
                    market_flag = "🇺🇸" if row['market'] == 'us' else "🇯🇵"
                    top10_count = row.get('top10_count', 0)
                    company_name = row.get('company_name', row['symbol'])
                    row_class = "even-row" if idx % 2 == 1 else ""

                    rows_html += f"""
                    <tr class="ranking-row {row_class}">
                        <td style="background-color: {rank_badge_color}; font-weight: bold; text-align: center;">{row['rank']}</td>
                        <td style="font-weight: 600; color: #2c3e50;">{company_name}</td>
                        <td style="font-weight: bold; font-family: monospace;">{row['symbol']}</td>
                        <td style="text-align: center; font-size: 1.2rem;">{market_flag}</td>
                        <td style="text-align: right; font-weight: 600; color: #2E7D32;">{row['score']:.2f}</td>
                        <td style="text-align: right;">{row['percentile']:.2%}</td>
                        <td style="text-align: right; color: #666;">{row['universe_size']:,}</td>
                        <td style="text-align: center; font-weight: 600; color: #1976D2;">{top10_count}回</td>
                        <td style="text-align: center; padding: 8px;">
                            <button class="detailed-report-btn" onclick="generateDetailedReport('{row['symbol']}')">
                                📊 詳細レポート生成
                            </button>
                        </td>
                    </tr>
                    """

                ranking_html = f"""
                <style>
                    .ranking-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                        font-size: 0.9rem;
                    }}
                    .ranking-table thead {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .ranking-table th {{
                        padding: 12px;
                        text-align: left;
                        font-weight: 600;
                    }}
                    .ranking-table .ranking-row {{
                        transition: all 0.2s ease;
                    }}
                    .ranking-table .ranking-row.even-row {{
                        background-color: #f8f9fa;
                    }}
                    .ranking-table .ranking-row:hover {{
                        background-color: #e3f2fd;
                        transform: translateX(5px);
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .ranking-table td {{
                        padding: 10px 12px;
                        border-bottom: 1px solid #ecf0f1;
                    }}

                    /* 詳細レポートボタンのスタイル */
                    .detailed-report-btn {{
                        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 6px;
                        cursor: pointer;
                        font-size: 0.9em;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        transition: all 0.3s;
                        white-space: nowrap;
                    }}
                    .detailed-report-btn:hover {{
                        background: linear-gradient(135deg, #2980b9 0%, #1e5f8e 100%);
                        transform: translateY(-1px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                    }}
                    .detailed-report-btn:disabled {{
                        background: #bdc3c7;
                        cursor: not-allowed;
                        transform: none;
                        box-shadow: none;
                    }}
                </style>
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0;">
                    <h2 style="margin: 0 0 10px 0; color: #2c3e50; font-size: 1.3rem; border-left: 4px solid #3498db; padding-left: 12px;">
                        🏆 V2スコアランキング Top10
                    </h2>
                    <p style="margin: 0 0 15px 16px; color: #7f8c8d; font-size: 0.85rem;">{ranking_date_display}</p>
                    <table class="ranking-table">
                        <thead>
                            <tr>
                                <th>順位</th>
                                <th>会社名</th>
                                <th>銘柄コード</th>
                                <th style="text-align: center;">市場</th>
                                <th style="text-align: right;">スコア</th>
                                <th style="text-align: right;">パーセンタイル</th>
                                <th style="text-align: right;">母集団</th>
                                <th style="text-align: center;">Top10入り</th>
                                <th style="text-align: center;">詳細分析</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
                """
                print("  ✅ V2スコアランキングTop10のHTML生成完了")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"⚠️ ランキングデータの取得に失敗しました: {str(e)}")
            print(f"詳細なエラー情報:\n{error_details}")
            ranking_html = f"""
            <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #f39c12; margin: 20px 0;">
                <p style="margin: 0; color: #856404;">⚠️ ランキングデータの取得に失敗しました: {str(e)}</p>
                <pre style="margin: 10px 0 0 0; font-size: 0.8em; color: #666; overflow-x: auto;">{error_details}</pre>
            </div>
            """

    # 総合スコア
    total_score = macro_score_sum + avg_micro_score
    total_class = "positive" if total_score > 0 else ("negative" if total_score < 0 else "")
    macro_class = "positive" if macro_score_sum > 0 else ("negative" if macro_score_sum < 0 else "")
    micro_class = "positive" if avg_micro_score > 0 else ("negative" if avg_micro_score < 0 else "")
    
    # HTMLテンプレート
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>市場スコア ({date_str})</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .score-cards {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .score-card {{
            flex: 1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .total-score {{
            background-color: rgba(240, 240, 240, 0.7);
        }}
        .macro-score {{
            background-color: rgba(230, 240, 255, 0.7);
        }}
        .micro-score {{
            background-color: rgba(230, 255, 230, 0.7);
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .score-label {{
            font-size: 16px;
            color: #666;
        }}
        .micro-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .micro-table th {{
            background-color: #f0f0f0;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }}
        .micro-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .micro-table tr:hover {{
            background-color: #f9f9f9;
        }}
        .macro-indicators table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .macro-indicators th {{
            background-color: #f0f0f0;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }}
        .macro-indicators td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .chart-container {{
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }}
        .stock-chart {{
            width: 100%;
            height: 500px;
            border: 1px solid #ddd;
            margin-bottom: 30px;
        }}
        .chart-error {{
            color: red;
            padding: 20px;
            background-color: #fff8f8;
            border: 1px solid #ffcaca;
            margin-bottom: 20px;
        }}
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 0.8em;
            color: #666;
        }}
        .debug-info {{
            font-family: monospace;
            font-size: 12px;
            background-color: #f9f9f9;
            padding: 10px;
            margin-top: 10px;
            border: 1px solid #eee;
            display: none;
        }}
        .debug-toggle {{
            color: #999;
            cursor: pointer;
            text-decoration: underline;
        }}
        
        /* KPIタイル用スタイル */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .kpi-tile {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            text-decoration: none;
            color: inherit;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .kpi-tile:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            text-decoration: none;
            color: inherit;
        }}
        
        .kpi-tile.positive {{
            border-left: 4px solid #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }}
        
        .kpi-tile.negative {{
            border-left: 4px solid #dc3545;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        }}
        
        .kpi-label {{
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .kpi-value {{
            font-size: 1.6em;
            font-weight: bold;
            color: #212529;
            margin-bottom: 10px;
        }}
        
        .kpi-spark {{
            height: 60px;
            width: 100%;
        }}
        
        /* ヒートマップ用スタイル */
        .quick-heatmaps {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .heatmap-section {{
            flex: 1;
        }}
        
        .heatmap-section h3 {{
            font-size: 1em;
            color: #495057;
            margin-bottom: 10px;
        }}
        
        .mini-chart {{
            height: 60px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }}
        
        /* レスポンシブ対応 */
        @media (max-width: 768px) {{
            .kpi-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
            }}
            
            .quick-heatmaps {{
                flex-direction: column;
                gap: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>市場スコア総合評価 ({date_str})</h1>

        {macro_overview_html}

        {ranking_html}

        {micro_scores_html}

        <div class="footer">
            <p>データは日次で自動更新されます。スコアは相対評価であり、絶対的な投資判断指標ではありません。</p>
            <span class="debug-toggle" onclick="toggleDebug()">デバッグ情報</span>
            <div class="debug-info" id="debug-info">
                デバッグ情報: スパークラインデータのシンボル: {list(sparkline_data.keys() if sparkline_data else [])}
            </div>
        </div>
    </div>
    
    <script>
        // デバッグ情報の表示切替
        function toggleDebug() {{
            var debug = document.getElementById('debug-info');
            if (debug.style.display === 'none' || debug.style.display === '') {{
                debug.style.display = 'block';
            }} else {{
                debug.style.display = 'none';
            }}
        }}

        // 詳細レポート生成関数
        function generateDetailedReport(symbol) {{
            console.log(`🚀 詳細レポート生成開始: ${{symbol}}`);

            const button = event.target;
            const originalText = button.innerHTML;
            button.disabled = true;
            button.innerHTML = '⏳ 生成中...';

            fetch('http://127.0.0.1:5001/api/generate_detailed_report', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    symbol: symbol
                }})
            }})
            .then(response => {{
                if (!response.ok) {{
                    throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
                }}
                return response.text().then(text => {{
                    try {{
                        return JSON.parse(text);
                    }} catch (parseError) {{
                        console.error('❌ JSON解析エラー:', parseError);
                        throw new Error(`Response parsing failed: ${{parseError.message}}`);
                    }}
                }});
            }})
            .then(data => {{
                console.log('📊 API応答:', data);

                if (data && data.success) {{
                    console.log(`✅ ${{symbol}} の詳細レポート生成完了`);

                    button.innerHTML = '📋 レポートを開く';
                    button.disabled = false;
                    button.style.background = 'linear-gradient(135deg, #27ae60 0%, #229954 100%)';

                    showTemporaryMessage(`✅ ${{symbol}} の詳細レポート生成完了！ボタンをクリックして開いてください`, 'success', 5000);

                    button.onclick = function() {{
                        console.log(`🔗 レポートを開きます: ${{data.report_url}}`);
                        window.open(data.report_url, '_blank');

                        setTimeout(() => {{
                            button.innerHTML = originalText;
                            button.style.background = '';
                            button.onclick = () => generateDetailedReport(symbol);
                        }}, 2000);
                    }};

                }} else {{
                    const errorMsg = data && data.error ? data.error : '不明なエラー';
                    console.error('❌ 詳細レポート生成失敗:', errorMsg);
                    showTemporaryMessage(`❌ ${{symbol}} の詳細レポート生成に失敗: ${{errorMsg}}`, 'error');

                    button.innerHTML = originalText;
                    button.disabled = false;
                }}
            }})
            .catch(error => {{
                console.error('🚫 API呼び出しエラー:', error);

                let errorMessage = '❌ 詳細レポート生成でエラーが発生しました';
                if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {{
                    errorMessage = '⚠️ APIサーバーに接続できません。サーバーを起動してください: python start_watchlist_api.py';
                }} else if (error.message.includes('timeout')) {{
                    errorMessage = '⚠️ レポート生成がタイムアウトしました。しばらく待ってから再試行してください';
                }}

                showTemporaryMessage(errorMessage, 'error');

                button.innerHTML = originalText;
                button.disabled = false;
                button.style.background = '';
            }});
        }}

        // 一時的なメッセージを表示する関数
        function showTemporaryMessage(message, type = 'success', duration = null) {{
            const existingMessage = document.querySelector('.watchlist-message');
            if (existingMessage) {{
                existingMessage.remove();
            }}

            const messageDiv = document.createElement('div');
            messageDiv.className = 'watchlist-message';
            messageDiv.textContent = message;
            messageDiv.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 16px;
                border-radius: 6px;
                color: white;
                font-weight: bold;
                z-index: 10000;
                transition: all 0.3s ease;
                max-width: 400px;
                font-size: 14px;
                ${{type === 'success' ? 'background-color: #28a745;' :
                  type === 'error' ? 'background-color: #dc3545;' :
                  type === 'info' ? 'background-color: #17a2b8;' :
                  'background-color: #ffc107; color: #212529;'}}
            `;

            document.body.appendChild(messageDiv);

            const displayTime = duration !== null ? duration : (type === 'info' ? 1000 : 3000);
            setTimeout(() => {{
                if (messageDiv.parentElement) {{
                    messageDiv.remove();
                }}
            }}, displayTime);
        }}

        // チャート描画処理のために確実にDOMContentLoadedイベントを使用
        document.addEventListener('DOMContentLoaded', function() {{
            // チャート描画のためのJavaScriptを実行
            try {{
                console.log("チャート描画処理を開始します");
                {chart_scripts}
                console.log("チャート描画処理が完了しました");
            }} catch(e) {{
                console.error("チャート描画中にエラーが発生しました:", e);
                
                // エラーメッセージを表示
                var charts = document.querySelectorAll('.stock-chart');
                for(var i = 0; i < charts.length; i++) {{
                    charts[i].innerHTML = '<div class="chart-error">チャートの描画中にエラーが発生しました。詳細はコンソールログを確認してください。</div>';
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    return html 