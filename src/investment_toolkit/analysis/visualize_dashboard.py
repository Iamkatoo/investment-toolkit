import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 共通の色定義 - すべてのグラフで使用する
COMMON_COLORS = {
    # 株価指数
    '^GSPC': '#e41a1c',    # S&P 500 - 赤
    '^DJI': '#377eb8',     # Dow Jones - 青
    '^IXIC': '#4daf4a',    # NASDAQ - 緑
    '^N225': '#ff8c00',    # 日経平均 - ダークオレンジ（紫から変更）
    
    # 恐怖指数
    '^VIX': '#9932cc',     # VIX - 紫（オレンジから変更）
    
    # 金・ドル・為替
    'GOLD': '#ffc125',     # 金 - より濃い黄色（見やすく）
    'USDJPY': '#20b2aa',   # ドル円 - ライトシーグリーン（茶色から変更）
    'TWEXBGSMTH': '#f781bf', # ドル指数 - ピンク
    'DXY': '#f781bf',      # 別表記のドル指数
    
    # 通貨ペア
    'EURUSD': '#999999',   # EUR/USD - 灰色
    'GBPUSD': '#e41a1c',   # GBP/USD - 赤
    'USDCAD': '#377eb8',   # USD/CAD - 青
    'AUDUSD': '#4daf4a',   # AUD/USD - 緑
    'USDCHF': '#984ea3',   # USD/CHF - 紫
    
    'EURJPY': '#ff7f00',   # EUR/JPY - オレンジ
    'GBPJPY': '#ffc125',   # GBP/JPY - 濃い黄色（金と同じ）
}

def plot_normalized_indices(df, start_date='2010-01-01'):
    """
    株価指数の推移（正規化）グラフを作成
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, ^GSPC, ^DJI, ^IXIC, ^N225 カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        正規化された株価指数のグラフ
    """
    # 表示期間でフィルタリング
    df_filtered = df[df['date'] >= start_date].copy()
    
    # 正規化（初期値=1として変化率を計算）
    indices = ['^GSPC', '^DJI', '^IXIC', '^N225']
    index_names = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^N225': '日経平均'
    }
    
    fig = go.Figure()
    
    # 最新日付を取得
    latest_date = df_filtered['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')
    
    for idx in indices:
        if idx in df_filtered.columns:
            # 初日を基準に正規化
            first_value = df_filtered[idx].iloc[0]
            normalized = df_filtered[idx] / first_value - 1
            
            # 最新値を取得
            latest_value = df_filtered[df_filtered['date'] == latest_date][idx].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=normalized,
                mode='lines',
                name=index_names.get(idx, idx),
                line=dict(color=COMMON_COLORS.get(idx, '#1f77b4'), width=2)
            ))
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            fig.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=-30 if idx == '^N225' else (-20 if idx == '^IXIC' else 0),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(idx, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(idx, '#1f77b4'))
            )
    
    # 説明テキストを追加
    explanation = """
    <b>株価指数の推移（正規化）</b><br>
    このグラフは主要株価指数の相対的なパフォーマンスを比較しています。全ての指数は期間開始時を基準（0%）として正規化されています。<br>
    各線の最後に表示されている数値は、グラフの最終日における実際の指数値です。<br><br>
    <b>見方:</b><br>
    • <b>相対的強さ:</b> 上に位置する指数ほど相対的なパフォーマンスが良い<br>
    • <b>トレンド変化:</b> 急激な傾きの変化は市場センチメントの転換を示唆<br>
    • <b>分散・相関:</b> 指数間の乖離が大きい場合はセクター/地域間の格差を示す<br><br>
    <b>活用方法:</b><br>
    • 地域間のローテーション（資金移動）を確認<br>
    • アウトパフォームしている指数のセクター構成を分析<br>
    • 指数間の連動性低下は、リスク分散の好機を示唆することがある
    """
    
    fig.update_layout(
        title={
            'text': '株価指数の推移（正規化）',
            'y':0.95,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title='日付',
        yaxis_title='変化率（初期値=0）',
        legend_title='指数',
        hovermode='x unified',
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
        height=600,  # 高さを他のグラフと同じに調整
    )
    
    # Y軸を%表示に設定
    fig.update_yaxes(tickformat='.0%', gridcolor='lightgray')
    
    # X軸のグリッドラインを追加
    fig.update_xaxes(gridcolor='lightgray')
    
    # ゼロラインを強調
    fig.add_hline(y=0, line_dash="solid", line_width=1, line_color="gray")
    
    return fig, explanation

# ------------------------------------------------------------------
# 追加サブグラフ : 直近 n 日を起点 (=0%) に正規化した指数推移
# ------------------------------------------------------------------
def _plot_indices_window(df, window_days: int, title_suffix: str):
    """
    汎用ヘルパー : 直近 window_days 日を起点に正規化して描画
    """
    import plotly.graph_objects as go
    import pandas as pd

    # 日付範囲を計算
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=window_days)
    df_win = df[df['date'] >= start_date].copy()

    indices = ['^GSPC', '^DJI', '^IXIC', '^N225']
    names = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones',
             '^IXIC': 'NASDAQ', '^N225': '日経平均'}

    fig = go.Figure()
    
    # 最新日付を取得
    latest_date = df_win['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')
    
    for idx in indices:
        if idx in df_win.columns:
            base = df_win[idx].iloc[0]
            normalized = df_win[idx] / base - 1
            
            # 最新値を取得
            latest_value = df_win[df_win['date'] == latest_date][idx].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=df_win['date'],
                    y=normalized,
                    mode='lines',
                    name=names.get(idx, idx),
                    line=dict(color=COMMON_COLORS.get(idx, '#1f77b4'), width=2)
                )
            )
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            fig.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.0f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=-30 if idx == '^N225' else (-20 if idx == '^IXIC' else 0),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(idx, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(idx, '#1f77b4'))
            )

    fig.update_layout(
        title=f'株価指数 ({title_suffix}) — 初日を 0 %',
        xaxis_title='日付',
        yaxis_title='変化率',
        hovermode='x unified',
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
        height=600
    )
    fig.update_yaxes(tickformat='.0%')

    explanation = (
        f"<b>{title_suffix} の株価指数推移</b><br>"
        "表示開始日の終値を 0 % とし、その後の変化率を比較しています。<br>"
        "各線の最後に表示されている数値は、グラフの最終日における実際の指数値です。"
    )
    return fig, explanation


def plot_normalized_indices_6w(df):
    """直近 6 週間 (42 日) を基準に正規化"""
    return _plot_indices_window(df, 42, "直近 6 W")

def plot_normalized_indices_3m(df):
    """直近 3 か月 (90 日) を基準に正規化"""
    return _plot_indices_window(df, 90, "直近 3 M")

def plot_vix_vs_sp500(df, start_date='2010-01-01'):
    """
    VIXとS&P500の比較グラフを作成
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, ^VIX, ^GSPC カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        VIXとS&P500の比較グラフ
    """
    # 表示期間でフィルタリング
    df_filtered = df[df['date'] >= start_date].copy()
    
    # VIXとS&P500のデュアル軸グラフを作成
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # S&P500追加（左軸）
    fig.add_trace(
        go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['^GSPC'],
            name='S&P 500',
            line=dict(color=COMMON_COLORS.get('^GSPC', '#1f77b4'), width=2)
        ),
        secondary_y=False
    )
    
    # VIX追加（右軸）
    fig.add_trace(
        go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['^VIX'],
            name='VIX（恐怖指数）',
            line=dict(color=COMMON_COLORS.get('^VIX', '#d62728'), width=2)
        ),
        secondary_y=True
    )
    
    # 主要なVIXレベルに水平ラインを追加
    fig.add_hline(y=20, line_dash="dash", line_width=1, line_color="green", secondary_y=True,
                 annotation_text="低ボラティリティ", annotation_position="bottom right")
    fig.add_hline(y=30, line_dash="dash", line_width=1, line_color="orange", secondary_y=True,
                 annotation_text="高ボラティリティ", annotation_position="bottom right")
    fig.add_hline(y=40, line_dash="dash", line_width=1, line_color="red", secondary_y=True)
    
    # 説明テキストを追加
    explanation = """
    <b>VIX（恐怖指数）とS&P500の関係</b><br>
    VIXは市場の予想変動率（ボラティリティ）を示し、投資家の恐怖心や不確実性のレベルを測る指標です。<br><br>
    <b>VIXの見方:</b><br>
    • <b>20以下:</b> 市場は落ち着いている。過度な楽観主義を示すこともある<br>
    • <b>20-30:</b> 通常の不確実性レベル<br>
    • <b>30以上:</b> 市場の恐怖が高まっている<br>
    • <b>40以上:</b> 極度の不安/パニック<br><br>
    <b>VIXとS&P500の逆相関:</b><br>
    一般的に、VIXが上昇すると株価は下落し、VIXが下落すると株価は上昇する傾向があります。<br>
    急激なVIXの上昇は市場の急落を伴うことが多く、極端に高いVIXは「逆張り」のシグナルになることも。
    """
    
    fig.update_layout(
        title={
            'text': 'VIX（恐怖指数）とS&P500の関係',
            'y':0.95,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title='日付',
        legend_title='指標',
        hovermode='x unified',
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
        height=600,  # 高さを他のグラフと同じに調整
    )
    
    # 左Y軸（S&P500）の設定
    fig.update_yaxes(
        title_text='S&P 500 指数',
        secondary_y=False,
        gridcolor='lightgray',
        tickfont=dict(color=COMMON_COLORS.get('^GSPC', '#1f77b4')),
        title_font=dict(color=COMMON_COLORS.get('^GSPC', '#1f77b4'))
    )
    
    # 右Y軸（VIX）の設定
    fig.update_yaxes(
        title_text='VIX 指数',
        secondary_y=True,
        gridcolor='lightgray',
        tickfont=dict(color=COMMON_COLORS.get('^VIX', '#d62728')),
        title_font=dict(color=COMMON_COLORS.get('^VIX', '#d62728')),
        range=[0, max(df_filtered['^VIX']) * 1.1]  # 少し余裕を持たせる
    )
    
    # X軸のグリッドライン
    fig.update_xaxes(gridcolor='lightgray')
    
    return fig, explanation

def plot_gold_dollar_yen(df, start_date='2010-01-01'):
    """
    金・ドル・為替（正規化）グラフを作成
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, GOLD, USDJPY, TWEXBGSMTH カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        正規化された金・ドル・為替のグラフ
    """
    # 表示期間でフィルタリング
    df_filtered = df[df['date'] >= start_date].copy()
    
    # 正規化（初期値=1として変化率を計算）
    # 長期グラフではGOLDデータを使用（2010年から利用可能）
    # 短期グラフではGCUSDデータを使用（2024年9月から利用可能）
    assets = ['GOLD', 'USDJPY', 'TWEXBGSMTH']
    asset_names = {
        'GOLD': '金価格',
        'USDJPY': 'ドル円',
        'TWEXBGSMTH': 'ドル指数'
    }
    
    fig = go.Figure()
    
    # 最新日付を取得
    latest_date = df_filtered['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')
    
    for asset in assets:
        if asset in df_filtered.columns:
            # 初日を基準に正規化
            first_value = df_filtered[asset].iloc[0]
            normalized = df_filtered[asset] / first_value - 1
            
            # 最新値を取得
            latest_value = df_filtered[df_filtered['date'] == latest_date][asset].iloc[0]
            
            # 金価格は小数点なし、その他は小数点1桁で表示
            format_str = '.0f' if asset == 'GOLD' else '.1f'
            
            fig.add_trace(go.Scatter(
                x=df_filtered['date'],
                y=normalized,
                mode='lines',
                name=asset_names.get(asset, asset),
                line=dict(color=COMMON_COLORS.get(asset, '#1f77b4'), width=2)
            ))
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整（重ならないようにする）
            ay_offset = 0
            if asset == 'GOLD':
                ay_offset = -30
            elif asset == 'USDJPY':
                ay_offset = 0
            else:
                ay_offset = 30
                
            fig.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:{format_str}}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(asset, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(asset, '#1f77b4'))
            )
    
    # 説明テキストを追加
    explanation = """
    <b>金・ドル・為替の関係（正規化）</b><br>
    このグラフは金価格、ドル円レート、ドル指数の相対的な動きを比較しています。全て期間開始時を基準（0%）として正規化されています。<br>
    各線の最後に表示されている数値は、グラフの最終日における実際の価格/レートです。<br><br>
    <b>見方:</b><br>
    • <b>金とドルの関係:</b> 一般的に金価格とドル指数は逆相関の関係にあります（ドル高→金安、ドル安→金高）<br>
    • <b>ドル指数とドル円:</b> ドル指数が上昇するとドル円も上昇する傾向がありますが、日本固有の要因で乖離することもあります<br>
    • <b>リスク回避時:</b> 市場不安時には金とドル（特に対円）が同時に買われることがあります<br><br>
    <b>活用方法:</b><br>
    • <b>マクロ環境分析:</b> ドル指数上昇＋金下落は通常、「タカ派的な金融政策」や「リスク選好」を示唆<br>
    • <b>インフレ予測:</b> 金価格の上昇はインフレ懸念の高まりを示すことがある<br>
    • <b>円の特性:</b> リスクオフ時の円高（ドル円下落）パターン vs. 日銀政策による円安（ドル円上昇）の影響を区別
    """
    
    fig.update_layout(
        title='金・ドル・為替（正規化）',
        xaxis_title='日付',
        yaxis_title='変化率（初期値=0）',
        legend_title='資産',
        hovermode='x unified',
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    
    # Y軸を%表示に設定
    fig.update_yaxes(tickformat='.0%')
    
    return fig, explanation

# --------------------------------------------------------------
# 汎用ヘルパー : 任意 window で再正規化する金・ドル・円グラフ
# --------------------------------------------------------------
def _plot_gold_dollar_yen_window(df, window_days: int, title_suffix: str):
    import plotly.graph_objects as go
    import pandas as pd

    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=window_days)
    df_win = df[df['date'] >= start_date].copy()

    series = ['GCUSD', 'USDJPY', 'TWEXBGSMTH']
    names  = {'GCUSD': '金価格', 'USDJPY': 'ドル円', 'TWEXBGSMTH': 'ドル指数'}

    fig = go.Figure()
    
    # 最新日付を取得
    latest_date = df_win['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')
    
    for s in series:
        if s in df_win.columns:
            base = df_win[s].iloc[0]
            normalized = df_win[s] / base - 1
            
            # 最新値を取得
            latest_value = df_win[df_win['date'] == latest_date][s].iloc[0]
            
            # 金価格は小数点なし、その他は小数点1桁で表示
            format_str = '.0f' if s == 'GCUSD' else '.1f'
            
            fig.add_trace(
                go.Scatter(
                    x=df_win['date'],
                    y=normalized,
                    mode='lines',
                    name=names.get(s, s),
                    line=dict(color=COMMON_COLORS.get(s, '#1f77b4'), width=2)
                )
            )
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整（重ならないようにする）
            ay_offset = 0
            if s == 'GCUSD':
                ay_offset = -30
            elif s == 'USDJPY':
                ay_offset = 0
            else:
                ay_offset = 30
                
            fig.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:{format_str}}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(s, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(s, '#1f77b4'))
            )

    fig.update_layout(
        title=f'金・ドル・円 ({title_suffix}) — 初日を 0 %',
        xaxis_title='日付', yaxis_title='変化率', hovermode='x unified',
        template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    fig.update_yaxes(tickformat='.0%')
    exp = f"<b>{title_suffix} の金 / ドル / 円 推移(正規化)</b><br>"
    exp += "このグラフは金価格、ドル円レート、ドル指数の相対的な動きを比較しています。全て期間開始時を基準（0%）として正規化されています。<br>"
    exp += "各線の最後に表示されている数値は、グラフの最終日における実際の価格/レートです。<br><br>"
    exp += "<b>見方:</b><br>"
    exp += "• <b>金とドルの関係:</b> 一般的に金価格とドル指数は逆相関の関係にあります（ドル高→金安、ドル安→金高）<br>"
    
    return fig, exp

def plot_gold_dollar_yen_6w(df):
    return _plot_gold_dollar_yen_window(df, 42, "直近 6 W")

def plot_gold_dollar_yen_3m(df):
    return _plot_gold_dollar_yen_window(df, 90, "直近 3 M")

def plot_currency_pairs(df, start_date='2010-01-01'):
    """
    通貨ペア（対ドル／クロス円）のトレンドグラフを作成し、
    2 つの図と説明文を返す
    """
    import pandas as pd
    import plotly.graph_objects as go

    # 表示期間フィルタ
    df_filtered = df[df['date'] >= start_date].copy()

    # ---------- 設定 ----------
    usd_pairs = ['EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF']
    usd_names = {
        'EURUSD': 'EUR/USD', 'GBPUSD': 'GBP/USD', 'USDCAD': 'USD/CAD',
        'AUDUSD': 'AUD/USD', 'USDCHF': 'USD/CHF'
    }

    jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY']
    jpy_names = {
        'USDJPY': 'USD/JPY', 'EURJPY': 'EUR/JPY', 'GBPJPY': 'GBP/JPY'
    }
    # ---------------------------

    # 最新日付を取得
    latest_date = df_filtered['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')

    # ① 対ドルペア
    fig_usd = go.Figure()
    for pair in usd_pairs:
        if pair in df_filtered.columns:
            first = df_filtered[pair].iloc[0]
            normalized = df_filtered[pair] / first - 1
            
            # 最新値を取得
            latest_value = df_filtered[df_filtered['date'] == latest_date][pair].iloc[0]
            
            fig_usd.add_trace(
                go.Scatter(
                    x=df_filtered['date'],
                    y=normalized,
                    mode='lines',
                    name=usd_names.get(pair, pair),
                    line=dict(color=COMMON_COLORS.get(pair, '#1f77b4'), width=2)
                )
            )
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整
            ay_offset = 0
            if pair == 'EURUSD':
                ay_offset = -40
            elif pair == 'GBPUSD':
                ay_offset = -20
            elif pair == 'USDCAD':
                ay_offset = 0
            elif pair == 'AUDUSD':
                ay_offset = 20
            else:  # USDCHF
                ay_offset = 40
                
            fig_usd.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.4f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(pair, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(pair, '#1f77b4'))
            )
            
    fig_usd.update_layout(
        title='対ドル通貨ペアの推移（正規化）',
        xaxis_title='日付', yaxis_title='変化率（初期値=0）',
        hovermode='x unified', template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    fig_usd.update_yaxes(tickformat='.0%')

    # ② クロス円ペア
    fig_jpy = go.Figure()
    for pair in jpy_pairs:
        if pair in df_filtered.columns:
            first = df_filtered[pair].iloc[0]
            normalized = df_filtered[pair] / first - 1
            
            # 最新値を取得
            latest_value = df_filtered[df_filtered['date'] == latest_date][pair].iloc[0]
            
            fig_jpy.add_trace(
                go.Scatter(
                    x=df_filtered['date'],
                    y=normalized,
                    mode='lines',
                    name=jpy_names.get(pair, pair),
                    line=dict(color=COMMON_COLORS.get(pair, '#1f77b4'), width=2)
                )
            )
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整
            ay_offset = 0
            if pair == 'USDJPY':
                ay_offset = -30
            elif pair == 'EURJPY':
                ay_offset = 0
            else:  # GBPJPY
                ay_offset = 30
                
            fig_jpy.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(pair, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(pair, '#1f77b4'))
            )
            
    fig_jpy.update_layout(
        title='クロス円通貨ペアの推移（正規化）',
        xaxis_title='日付', yaxis_title='変化率（初期値=0）',
        hovermode='x unified', template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    fig_jpy.update_yaxes(tickformat='.0%')

    # 説明文
    explanation = (
        "<b>対ドル通貨ペアの見方</b>: 上昇＝ドル安／下降＝ドル高<br>"
        "<b>クロス円ペアの見方</b>: 上昇＝円安／下降＝円高<br>"
        "各線の最後に表示されている数値は、グラフの最終日における実際のレートです。<br><br>"
        "EUR=ユーロ<br>"
        "GBP=英ポンド<br>"
        "CAD=カナダドル<br>"
        "AUD=オーストラリアドル<br>"
        "CHF=スイスフラン<br>"
    )

    # ３タプルで返す（generate_reports が 3 要素を期待）
    return fig_usd, fig_jpy, explanation

def _plot_currency_pairs_window(df, window_days: int, title_suffix: str):
    """直近 window_days で再正規化した USD ペア & JPY ペア"""
    import pandas as pd
    import plotly.graph_objects as go

    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=window_days)
    df_win = df[df['date'] >= start_date].copy()

    usd_pairs = ['EURUSD', 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF']
    jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY']

    usd_names = {'EURUSD':'EUR/USD','GBPUSD':'GBP/USD','USDCAD':'USD/CAD',
                 'AUDUSD':'AUD/USD','USDCHF':'USD/CHF'}
    jpy_names = {'USDJPY':'USD/JPY','EURJPY':'EUR/JPY','GBPJPY':'GBP/JPY'}

    # 最新日付を取得
    latest_date = df_win['date'].max()
    latest_date_str = latest_date.strftime('%m/%d')

    # USD グラフ
    fig_usd = go.Figure()
    for p in usd_pairs:
        if p in df_win.columns:
            base = df_win[p].iloc[0]
            normalized = df_win[p]/base - 1
            
            # 最新値を取得
            latest_value = df_win[df_win['date'] == latest_date][p].iloc[0]
            
            fig_usd.add_trace(go.Scatter(
                x=df_win['date'], 
                y=normalized,
                mode='lines', 
                name=usd_names.get(p,p),
                line=dict(color=COMMON_COLORS.get(p, '#1f77b4'), width=2)
            ))
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整
            ay_offset = 0
            if p == 'EURUSD':
                ay_offset = -40
            elif p == 'GBPUSD':
                ay_offset = -20
            elif p == 'USDCAD':
                ay_offset = 0
            elif p == 'AUDUSD':
                ay_offset = 20
            else:  # USDCHF
                ay_offset = 40
                
            fig_usd.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.4f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(p, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(p, '#1f77b4'))
            )
            
    fig_usd.update_layout(
        title=f'対ドル通貨ペア ({title_suffix}) — 初日を 0 %',
        xaxis_title='日付', yaxis_title='変化率', template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    fig_usd.update_yaxes(tickformat='.0%')

    # JPY グラフ
    fig_jpy = go.Figure()
    for p in jpy_pairs:
        if p in df_win.columns:
            base = df_win[p].iloc[0]
            normalized = df_win[p]/base - 1
            
            # 最新値を取得
            latest_value = df_win[df_win['date'] == latest_date][p].iloc[0]
            
            fig_jpy.add_trace(go.Scatter(
                x=df_win['date'], 
                y=normalized,
                mode='lines', 
                name=jpy_names.get(p,p),
                line=dict(color=COMMON_COLORS.get(p, '#1f77b4'), width=2)
            ))
            
            # 最新値に注釈を追加
            latest_normalized = normalized.iloc[-1]
            
            # 注釈の位置を調整
            ay_offset = 0
            if p == 'USDJPY':
                ay_offset = -30
            elif p == 'EURJPY':
                ay_offset = 0
            else:  # GBPJPY
                ay_offset = 30
                
            fig_jpy.add_annotation(
                x=latest_date,
                y=latest_normalized,
                text=f"{latest_date_str} {latest_value:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=COMMON_COLORS.get(p, '#1f77b4'),
                font=dict(color=COMMON_COLORS.get(p, '#1f77b4'))
            )
            
    fig_jpy.update_layout(
        title=f'クロス円通貨ペア ({title_suffix}) — 初日を 0 %',
        xaxis_title='日付', yaxis_title='変化率', template='plotly_white',
        legend={'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        margin=dict(t=100, b=100, r=120),
    )
    fig_jpy.update_yaxes(tickformat='.0%')

    explanation = (
        f"<b>{title_suffix} の通貨ペア推移</b><br>"
        "<b>対ドル通貨ペアの見方</b>: 上昇＝ドル安／下降＝ドル高<br>"
        "<b>クロス円ペアの見方</b>: 上昇＝円安／下降＝円高<br>"
        "各線の最後に表示されている数値は、グラフの最終日における実際のレートです。"
    )

    return fig_usd, fig_jpy, explanation

def plot_currency_pairs_6w(df):
    return _plot_currency_pairs_window(df, 42, "直近 6 W")

def plot_currency_pairs_3m(df):
    return _plot_currency_pairs_window(df, 90, "直近 3 M")

def plot_interest_rates(df, start_date='2010-01-01'):
    """
    金利推移（政策金利 vs 長期金利）グラフを作成
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, FEDFUNDS, DGS10 カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        金利推移の2軸グラフ
    """
    # 表示期間でフィルタリング
    df_filtered = df[df['date'] >= start_date].copy()
    
    # 2軸グラフを作成
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # FF金利（左軸）
    if 'FEDFUNDS' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['FEDFUNDS'],
                name='FF金利（政策金利）',
                line=dict(color='red')
            ),
            secondary_y=False
        )
    
    # 10年債利回り（右軸）
    if 'DGS10' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['DGS10'],
                name='10年債利回り',
                line=dict(color='blue')
            ),
            secondary_y=True
        )
    
    # 説明テキストを追加
    explanation = """
    <b>金利推移（政策金利 vs 長期金利）</b><br>
    このグラフはFF金利（FRBの政策金利、赤線・左軸）と10年債利回り（青線・右軸）の推移を表示しています。<br><br>
    <b>見方:</b><br>
    • <b>イールドスプレッド:</b> 10年債利回り - FF金利の差。正常時はプラス（右上がり）<br>
    • <b>イールドカーブの傾き:</b> スプレッドが縮小 = フラット化、マイナスになるとインバージョン（景気後退のシグナル）<br>
    • <b>金利サイクル:</b> FF金利の引き上げ/引き下げサイクルは経済全体に大きな影響<br><br>
    <b>活用方法:</b><br>
    • <b>景気予測:</b> イールドカーブのインバージョン（逆転）は将来の景気後退を示唆することが多い<br>
    • <b>金融政策の方向性:</b> 長期金利の動きから市場の将来予想を読み取れる<br>
    • <b>投資戦略:</b> 金利上昇局面（FF金利上昇）では債券より株式、金利下降局面では債券のパフォーマンスが優れる傾向
    """
    
    fig.update_layout(
        title='金利推移（政策金利 vs 長期金利）',
        hovermode='x unified',
        template='plotly_white',
        
    )
    
    fig.update_xaxes(title_text='日付')
    fig.update_yaxes(title_text='FF金利（%）', secondary_y=False)
    fig.update_yaxes(title_text='10年債利回り（%）', secondary_y=True)
    
    return fig, explanation

def plot_inflation(df, start_date='2010-01-01'):
    """
    インフレ（CPI, PCE など）グラフを作成
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, CPIAUCSL, CPILFESL, CPILEGSL, PCEPI カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        インフレ指標のグラフ
    """
    import pandas as pd
    import plotly.graph_objects as go

    # --------------------------------------------------
    # 準備
    # --------------------------------------------------
    indicators = ['CPIAUCSL', 'CPILFESL', 'CPILEGSL', 'PCEPI']
    names = {
        'CPIAUCSL': 'CPI総合',
        'CPILFESL': 'コアCPI',
        'CPILEGSL': 'エネルギー除くCPI',
        'PCEPI':    'PCE物価指数'
    }

    # ① 必要列だけ抽出して月次に resample
    df_m = (
        df[['date'] + indicators]
        .copy()
        .set_index('date')
        .resample('ME').last()          # 月末値を採用
        .loc[start_date:]              # 表示期間フィルタ
    )

    # ② YoY(% ) を計算
    for col in indicators:
        df_m[f'{col}_YoY'] = df_m[col].pct_change(12) * 100

    # --------------------------------------------------
    # 図を作成
    # --------------------------------------------------
    fig = go.Figure()
    
    # 最新日付を取得
    latest_date = df_m.index.max()
    latest_date_str = latest_date.strftime('%m/%d')
    
    for col in indicators:
        yoy = f'{col}_YoY'
        if yoy in df_m.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_m.index,
                    y=df_m[yoy],
                    mode='lines',
                    name=names.get(col, col)
                )
            )
            
            # 最新値を取得
            latest_value = df_m[yoy].iloc[-1]
            
            # 注釈の位置を調整
            ay_offset = 0
            if col == 'CPIAUCSL':
                ay_offset = -40
            elif col == 'CPILFESL':
                ay_offset = -20
            elif col == 'CPILEGSL':
                ay_offset = 0
            else:  # PCEPI
                ay_offset = 20
                
            fig.add_annotation(
                x=latest_date,
                y=latest_value,
                text=f"{latest_date_str} {latest_value:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=40,
                ay=ay_offset,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#1f77b4",  # インフレ指標は特定の色が定義されていないため標準色を使用
                font=dict(color="#1f77b4")
            )

    fig.update_layout(
        title='インフレ指標（前年同月比）',
        xaxis_title='日付',
        yaxis_title='YoY (%)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_yaxes(tickformat='.1f')

    # 説明テキストを追加
    explanation = """
    <b>インフレ指標（前年同月比）</b><br>
    このグラフは主要なインフレ指標の前年同月比変化率を表示しています。<br>
    各線の最後に表示されている数値は、グラフの最終月における実際の前年同月比（%）です。<br><br>
    <b>指標の説明:</b><br>
    • <b>CPI総合:</b> 消費者物価指数全体（食品・エネルギー含む）<br>
    • <b>コアCPI:</b> 変動の大きい食品とエネルギーを除いたCPI<br>
    • <b>エネルギー除くCPI:</b> エネルギーのみを除いたCPI<br>
    • <b>PCE物価指数:</b> 個人消費支出物価指数（FRBが重視）<br><br>
    <b>見方:</b><br>
    • <b>FRBのターゲット:</b> PCEで2%前後が望ましいとされる<br>
    • <b>コアと総合の乖離:</b> 食品・エネルギー価格の影響度を示す<br>
    • <b>トレンドの方向性:</b> 上昇/下降トレンドは金融政策の方向性に影響<br><br>
    <b>活用方法:</b><br>
    • <b>金融政策予測:</b> インフレ上昇→引き締め、インフレ低下→緩和の傾向<br>
    • <b>資産配分:</b> インフレ率上昇時は実物資産（株式、コモディティ、不動産）が優位<br>
    • <b>債券投資戦略:</b> インフレ上昇は名目債券に不利、TIPS等のインフレ連動債に有利
    """

    return fig, explanation


def plot_economic_score(df_scored, start_date='2015-01-01'):
    """
    経済スコア推移グラフを作成し、図と説明文を返す

    Parameters
    ----------
    df_scored : pandas.DataFrame
        date, evaluation カラムを含むデータフレーム
    start_date : str, optional
        表示開始日（デフォルト: '2015-01-01'）

    Returns
    -------
    tuple
        (plotly.graph_objects.Figure, str)
    """
    import pandas as pd
    import plotly.graph_objects as go

    # 対象期間抽出
    df_filtered = df_scored[df_scored['date'] >= start_date].copy()

    # 図を作成
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['evaluation'],
            mode='lines',
            name='経済スコア',
            line=dict(color='green', width=2),
        )
    )

    # ゼロライン
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # レイアウト
    fig.update_layout(
        title='経済スコア推移',
        xaxis_title='日付',
        yaxis_title='スコア',
        hovermode='x unified',
        template='plotly_white',
    )

    # 説明文
    explanation = """
    <b>経済スコア推移</b><br>
    このグラフは主要 5 指標（政策金利、イールドカーブスプレッド、長期金利、
    社債スプレッド、ドル指数）を統合した総合スコアを表示しています。<br><br>
    • <b>正のスコア</b>: 景気拡大寄り<br>
    • <b>負のスコア</b>: 景気減速・リスクオフ寄り<br>
    0 付近の反転やトレンドの加速に注目してください。
    """

    # ２要素だけ返す
    return fig, explanation

def plot_economic_subplots(df, start_date='2010-01-01'):
    """
    経済指標のサブプロット可視化
    
    Parameters:
    -----------
    df : pandas.DataFrame
        date, FEDFUNDS, yield_difference, DGS10, BAA10Y, TWEXBGSMTH カラムを含むデータフレーム
    start_date : str
        表示開始日（デフォルト: '2010-01-01'）
        
    Returns:
    --------
    plotly.graph_objects.Figure
        経済指標のサブプロットグラフ
    """
    # 表示期間でフィルタリング
    df_filtered = df[df['date'] >= start_date].copy()
    
    # サブプロット作成（5行1列）
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'FF金利（政策金利）- FEDFUNDS [%]',
            'イールドカーブスプレッド - yield_difference [%]',
            '10年債利回り - DGS10 [%]',
            'Baa社債スプレッド - BAA10Y [%]',
            'ドル指数 - TWEXBGSMTH [指数]'
        )
    )
    
    # 1. FEDFUNDS (政策金利)
    if 'FEDFUNDS' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['FEDFUNDS'],
                mode='lines',
                name='FF金利',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        fig.update_yaxes(title_text='[%]', row=1, col=1)
    
    # 2. yield_difference (イールドカーブスプレッド)
    if 'yield_difference' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['yield_difference'],
                mode='lines',
                name='イールドスプレッド',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text='[%]', row=2, col=1)
        
        # ゼロラインを追加
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 3. DGS10 (10年債利回り)
    if 'DGS10' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['DGS10'],
                mode='lines',
                name='10年債利回り',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        fig.update_yaxes(title_text='[%]', row=3, col=1)
    
    # 4. BAA10Y (社債スプレッド)
    if 'BAA10Y' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['BAA10Y'],
                mode='lines',
                name='Baa社債スプレッド',
                line=dict(color='orange')
            ),
            row=4, col=1
        )
        fig.update_yaxes(title_text='[%]', row=4, col=1)
    
    # 5. TWEXBGSMTH (ドル指数)
    if 'TWEXBGSMTH' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered['date'],
                y=df_filtered['TWEXBGSMTH'],
                mode='lines',
                name='ドル指数',
                line=dict(color='purple')
            ),
            row=5, col=1
        )
        fig.update_yaxes(title_text='[指数]', row=5, col=1)
    
    # 説明テキストを追加
    explanation = """
    <b>経済指標サブプロット</b><br>
    このグラフは経済スコア算出に使用される5つの主要指標を個別に表示しています。<br><br>
    <b>指標と意味:</b><br>
    • <b>FF金利（政策金利）:</b> FRBの政策金利。上昇は引き締め、下降は緩和を示す<br>
    • <b>イールドカーブスプレッド:</b> 長短金利差。正常時は正、ゼロ以下は景気後退シグナル<br>
    • <b>10年債利回り:</b> 長期金利の代表。経済成長とインフレ期待を反映<br>
    • <b>Baa社債スプレッド:</b> 社債と国債の利回り差。拡大は信用リスク上昇を示す<br>
    • <b>ドル指数:</b> 主要通貨に対するドルの強さ。上昇はドル高<br><br>
    <b>スコアリングの考え方:</b><br>
    • FF金利: 低下または安定→高スコア、急上昇→低スコア<br>
    • イールドスプレッド: 大きい→高スコア、フラット/インバート→低スコア<br>
    • 10年債: 低水準で安定→高スコア、高水準または急上昇→低スコア<br>
    • 社債スプレッド: 縮小→高スコア、拡大→低スコア<br>
    • ドル指数: 安定/適度な弱さ→高スコア、急上昇→低スコア<br><br>
    <b>活用方法:</b><br>
    • 総合スコアの構成要素を個別に確認して、どの要因が経済環境に最も影響しているかを理解<br>
    • 各指標の変化方向から、今後の経済環境変化を予測
    """
    
    fig.update_layout(
        height=1200,
        title_text='経済指標サブプロット',
        showlegend=False,
        template='plotly_white',
        
    )
    
    # X軸ラベルは最下段のみ表示
    fig.update_xaxes(title_text='日付', row=5, col=1)
    
    return fig, explanation 