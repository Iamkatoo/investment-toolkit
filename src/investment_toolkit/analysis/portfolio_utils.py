"""
ポートフォリオ（保有株）の情報を PostgreSQL (user_data.trade_journal) から読み込み、
最新終値と突き合わせて損益を計算し、図表を返すユーティリティ
―― 売却フラグ / 複数ロット対応版
"""

from pathlib import Path
from typing import Tuple, List, Optional
from io import StringIO
from datetime import timedelta, datetime
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, text

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# DB接続設定をインポート
from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
def _get_engine() -> Engine:
    """SQLAlchemy Engineを取得"""
    connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string, pool_pre_ping=True)


# --------------------------------------------------------------------------
# 年率換算利益率計算のヘルパー関数
# --------------------------------------------------------------------------
def calculate_annualized_return(purchase_date, current_date, pl_percent):
    """
    年率換算利益率を計算する
    
    Parameters:
    -----------
    purchase_date : datetime
        購入日
    current_date : datetime
        現在日（または売却日）
    pl_percent : float
        利益率（%）
    
    Returns:
    --------
    float
        年率換算利益率（%）
    """
    if pd.isna(purchase_date) or pd.isna(current_date) or pd.isna(pl_percent):
        return 0.0
    
    # 保有期間（日数）を計算
    holding_days = (current_date - purchase_date).days
    
    # 保有期間が0日以下の場合は0を返す
    if holding_days <= 0:
        return 0.0
    
    # 保有期間（年）を計算
    holding_years = holding_days / 365.25
    
    # 年率換算利益率を計算
    # (1 + pl_percent/100)^(1/holding_years) - 1
    if pl_percent <= -100:  # 全損の場合
        return -100.0
    
    try:
        annualized_return = ((1 + pl_percent / 100) ** (1 / holding_years) - 1) * 100
        return round(annualized_return, 2)
    except (ZeroDivisionError, ValueError, OverflowError):
        return 0.0


def calculate_weighted_annualized_return(df_lots):
    """
    複数ロットの加重平均年率換算利益率を計算する
    
    Parameters:
    -----------
    df_lots : DataFrame
        同一銘柄の複数ロットデータ（date, qty, price, close等を含む）
    
    Returns:
    --------
    float
        加重平均年率換算利益率（%）
    """
    if df_lots.empty:
        return 0.0
    
    current_date = datetime.now()
    total_investment = 0.0
    weighted_annualized_sum = 0.0
    
    for _, row in df_lots.iterrows():
        investment = row['qty'] * row['price']
        pl_percent = ((row['close'] / row['price']) - 1) * 100
        
        # 各ロットの年率換算利益率を計算
        annualized_return = calculate_annualized_return(
            pd.to_datetime(row['date']), 
            current_date, 
            pl_percent
        )
        
        # 投資額で加重
        weighted_annualized_sum += annualized_return * investment
        total_investment += investment
    
    if total_investment == 0:
        return 0.0
    
    return round(weighted_annualized_sum / total_investment, 2)


# --------------------------------------------------------------------------
# 1. 取引データのロード
# --------------------------------------------------------------------------
def load_transactions(engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    PostgreSQL (user_data.trade_journal) から保有中ロットを取得
    ・sell_date IS NULL（未売却）のロットだけを抽出
    ・通貨列（.T は JPY、それ以外は USD）を付与

    Parameters:
    -----------
    engine : Engine, optional
        SQLAlchemy Engine。Noneの場合は内部で作成

    Returns:
    --------
    pd.DataFrame
        保有中ロットのデータフレーム
    """
    if engine is None:
        engine = _get_engine()

    query = text("""
        SELECT
            symbol,
            buy_date as date,
            buy_price as price,
            qty,
            buy_reason_text as reason,
            CASE WHEN sell_date IS NOT NULL THEN true ELSE false END as sold,
            sell_date as sold_date,
            sell_price as sold_price,
            sell_reason_text as sold_reason
        FROM user_data.trade_journal
        WHERE sell_date IS NULL
        ORDER BY symbol, buy_date
    """)

    df = pd.read_sql_query(query, engine)

    if df.empty:
        # 空のDataFrameに必要なカラムを追加
        df = pd.DataFrame(columns=['symbol', 'date', 'price', 'qty', 'reason',
                                   'sold', 'sold_date', 'sold_price', 'sold_reason', 'currency'])
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["currency"] = df["symbol"].apply(
        lambda x: "JPY" if x.endswith(".T") else "USD"
    )
    return df


def load_all_transactions(engine: Optional[Engine] = None) -> pd.DataFrame:
    """
    PostgreSQL (user_data.trade_journal) から全ロットを取得
    ・全ロット（保有中・売却済み）を返す
    ・通貨列（.T は JPY、それ以外は USD）を付与

    Parameters:
    -----------
    engine : Engine, optional
        SQLAlchemy Engine。Noneの場合は内部で作成

    Returns:
    --------
    pd.DataFrame
        全ロットのデータフレーム
    """
    if engine is None:
        engine = _get_engine()

    query = text("""
        SELECT
            symbol,
            buy_date as date,
            buy_price as price,
            qty,
            buy_reason_text as reason,
            CASE WHEN sell_date IS NOT NULL THEN true ELSE false END as sold,
            sell_date as sold_date,
            sell_price as sold_price,
            sell_reason_text as sold_reason
        FROM user_data.trade_journal
        ORDER BY symbol, buy_date
    """)

    df = pd.read_sql_query(query, engine)

    if df.empty:
        df = pd.DataFrame(columns=['symbol', 'date', 'price', 'qty', 'reason',
                                   'sold', 'sold_date', 'sold_price', 'sold_reason', 'currency'])
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["currency"] = df["symbol"].apply(lambda x: "JPY" if x.endswith(".T") else "USD")
    return df


# --------------------------------------------------------------------------
# 2. 現在ポジション（加重平均コスト）を作成
# --------------------------------------------------------------------------
def make_current_positions(df_tx: pd.DataFrame) -> pd.DataFrame:
    """
    df_tx: ロット単位の取引 DataFrame  
    return: 現在保有ポジション（qty_total, avg_cost, purchase_date, currency）
    """
    pos = (
        df_tx.groupby("symbol")
        .apply(
            lambda g: pd.Series(
                {
                    "qty_total": g["qty"].sum(),
                    "avg_cost": (g["qty"] * g["price"]).sum() / g["qty"].sum(),
                    "purchase_date": g["date"].min(),  # 最初に買った日を採用
                    "currency": g["currency"].iloc[0],
                }
            )
        )
        .reset_index()
    )
    return pos


# --------------------------------------------------------------------------
# 3. 最新終値を付与して損益計算
# --------------------------------------------------------------------------
def enrich_with_latest_prices(engine: Engine, df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    latest close を取得し market_value / unrealized_pl / pl_% / annualized_return を付与
    """
    symbols = list(df_pos["symbol"])
    query = text(
        """
        WITH latest AS (
          SELECT DISTINCT ON (symbol) symbol, date, close
          FROM fmp_data.daily_prices
          WHERE symbol = ANY(:symbols)
          ORDER BY symbol, date DESC
        )
        SELECT * FROM latest;
        """
    )
    px = pd.read_sql_query(query, engine, params={"symbols": symbols})
    df = df_pos.merge(px, on="symbol", how="left")

    df["market_value"] = df["qty_total"] * df["close"]
    df["unrealized_pl"] = df["qty_total"] * (df["close"] - df["avg_cost"])
    df["pl_percent"] = (df["close"] / df["avg_cost"] - 1) * 100
    
    # 年率換算利益率を計算
    current_date = datetime.now()
    df["annualized_return"] = df.apply(
        lambda row: calculate_annualized_return(
            pd.to_datetime(row["purchase_date"]), 
            current_date, 
            row["pl_percent"]
        ), 
        axis=1
    )
    
    return df


# --------------------------------------------------------------------------
# 4‑A. 含み損益バー（通貨別サブプロット）
# --------------------------------------------------------------------------
def make_pl_bars(df: pd.DataFrame) -> go.Figure:
    currencies = df["currency"].unique()
    fig = make_subplots(
        rows=len(currencies),
        cols=1,
        subplot_titles=[f"{cur} 建て含み損益" for cur in currencies],
        vertical_spacing=0.25,
    )

    for i, cur in enumerate(currencies, 1):
        cur_df = df[df["currency"] == cur]
        colors = ["green" if x >= 0 else "red" for x in cur_df["unrealized_pl"]]

        fig.add_bar(
            x=cur_df["symbol"],
            y=cur_df["unrealized_pl"],
            marker_color=colors,
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text="P/L", row=i, col=1)

    fig.update_layout(
        title="含み損益（加重平均コスト）",
        height=300 * len(currencies),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# --------------------------------------------------------------------------
# 4‑C. ロット別（買付日別）含み損益バー
# --------------------------------------------------------------------------
def make_per_lot_bars(df_lot: pd.DataFrame, engine: Engine = None) -> List[go.Figure]:
    """
    df_lot は lot 単位で
      symbol | date_x | qty | price | close | unrealized_pl | currency などが入るDF
    
    通貨ごとに銘柄の買付日別の損益を積み上げ棒グラフで表示
    戻り値は通貨ごとのFigureオブジェクトのリスト
    """
    # dfがなければ空のグラフを返す
    if df_lot.empty:
        fig = go.Figure()
        fig.update_layout(
            title="ロット別含み損益（データなし）",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return [fig]
    
    df_lot = df_lot.copy()
    
    # デバッグ出力
    print(f"カラム: {df_lot.columns.tolist()}")
    
    # 日付カラムがmergeで重複してdate_xになっているので、それを使用
    if 'date_x' in df_lot.columns:
        df_lot["date"] = pd.to_datetime(df_lot["date_x"])
        df_lot["date_str"] = df_lot["date"].dt.strftime('%Y-%m-%d')
    else:
        # 日付カラムがない場合は "不明" を使用
        print("日付カラムが見つからないため、'不明'を使用します")
        df_lot["date_str"] = "不明"
    
    # 損益の計算
    if 'unrealized_pl' not in df_lot.columns:
        df_lot['unrealized_pl'] = df_lot['qty'] * (df_lot['close'] - df_lot['price'])
    
    # ロット別の年率換算利益率を計算
    current_date = datetime.now()
    df_lot["pl_percent"] = ((df_lot["close"] / df_lot["price"]) - 1) * 100
    
    # date列が存在するかチェックし、適切な列名を使用
    date_col = "date"
    if "date" not in df_lot.columns:
        if "date_x" in df_lot.columns:
            date_col = "date_x"
        elif "date_y" in df_lot.columns:
            date_col = "date_y"
        else:
            # デバッグ用：利用可能な列を表示
            print(f"利用可能な列: {df_lot.columns.tolist()}")
            # 日付列が見つからない場合はスキップ
            df_lot["annualized_return"] = 0.0
            date_col = None
    
    if date_col:
        df_lot["annualized_return"] = df_lot.apply(
            lambda row: calculate_annualized_return(
                pd.to_datetime(row[date_col]), 
                current_date, 
                row["pl_percent"]
            ), 
            axis=1
        )
    
    # 通貨ごとにサブプロット作成
    currencies = sorted(df_lot['currency'].unique())
    figures = []  # 通貨ごとの図を格納するリスト
    
    # 通貨ごとの合計損益を計算
    currency_totals = {}
    for currency in currencies:
        currency_df = df_lot[df_lot['currency'] == currency]
        total_pl = currency_df['unrealized_pl'].sum()
        currency_totals[currency] = total_pl
    
    # 各通貨ごとにグラフを作成
    for currency in currencies:
        currency_data = df_lot[df_lot['currency'] == currency]
        symbols = currency_data['symbol'].unique()
        
        # 通貨のタイトル（合計損益付き）
        total = currency_totals[currency]
        sign = "+" if total >= 0 else ""
        title = f"{currency} 建て含み損益 (合計: {sign}{total:.2f})"
        
        # 新しい図を作成
        fig = go.Figure()
        
        # 各シンボルの処理
        for symbol in symbols:
            symbol_data = currency_data[currency_data['symbol'] == symbol]
            
            # シンボルごとの合計値を計算
            symbol_total = symbol_data['unrealized_pl'].sum()
            sign = "+" if symbol_total >= 0 else ""
            
            # 損益が正のロットと負のロットに分ける
            positive_lots = symbol_data[symbol_data['unrealized_pl'] >= 0]
            negative_lots = symbol_data[symbol_data['unrealized_pl'] < 0]
            
            # 正の損益のロットを追加
            for _, lot in positive_lots.iterrows():
                fig.add_trace(
                    go.Bar(
                        name=f"{lot['symbol']} ({lot['date_str']})",
                        x=[lot['symbol']],
                        y=[lot['unrealized_pl']],
                        text=f"+{lot['unrealized_pl']:.2f}",
                        customdata=[lot['date_str']],
                        hovertemplate="<b>%{x}</b><br>日付: %{customdata}<br>P/L: %{y:.2f}",
                        marker_color='green',
                        opacity=0.7,
                        showlegend=True,
                        legendgroup=lot['symbol']
                    )
                )
            
            # 負の損益のロットを追加
            for _, lot in negative_lots.iterrows():
                fig.add_trace(
                    go.Bar(
                        name=f"{lot['symbol']} ({lot['date_str']})",
                        x=[lot['symbol']],
                        y=[lot['unrealized_pl']],
                        text=f"{lot['unrealized_pl']:.2f}",
                        customdata=[lot['date_str']],
                        hovertemplate="<b>%{x}</b><br>日付: %{customdata}<br>P/L: %{y:.2f}",
                        marker_color='red',
                        opacity=0.7,
                        showlegend=True,
                        legendgroup=lot['symbol']
                    )
                )
        
        # 合計行を追加
        total_unrealized_pl = currency_data["unrealized_pl"].sum()
        total_invested = (currency_data["qty"] * currency_data["price"]).sum()
        if total_invested != 0:
            total_pl_percent = total_unrealized_pl / total_invested * 100
        else:
            total_pl_percent = 0
            
        # 加重平均年率換算利益率を計算
        total_annualized = 0.0
        if total_invested != 0:
            weighted_sum = 0.0
            for _, row in currency_data.iterrows():
                if pd.notna(row.get("annualized_return")) and pd.notna(row["qty"]) and pd.notna(row["price"]):
                    investment = row["qty"] * row["price"]
                    weighted_sum += row["annualized_return"] * investment
            total_annualized = round(weighted_sum / total_invested, 2)
            
        sum_row = {
            "symbol": "<b>合計</b>",
            "qty": currency_data["qty"].sum(),
            "price": "",
            "close": "",
            "unrealized_pl": total_unrealized_pl,
            "pl_percent": round(total_pl_percent, 2),
            "annualized_return": total_annualized,
            "status": ""
        }
        cur_tbl = pd.concat([currency_data, pd.DataFrame([sum_row])], ignore_index=True)
        
        # 全体のレイアウト設定
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title=f"P/L ({currency})",
            xaxis_title="銘柄",
            barmode='relative',  # 積み上げ表示
            height=500,  # 固定高さ
            margin=dict(l=50, r=50, t=80, b=150),  # マージン
            hoverlabel=dict(
                bgcolor="white",
                font_size=12
            ),
            legend=dict(
                orientation="h", 
                y=-0.4,  # 凡例をグラフのさらに下に配置
                xanchor="center",
                x=0.5,
                font=dict(size=10),
                traceorder="grouped"  # 凡例をグループ化
            )
        )
        
        # グラフリストに追加
        figures.append(fig)
    
    return figures


# --------------------------------------------------------------------------
# 4-D. 銘柄別3段テクニカルチャート（旧株価推移機能を統合）
# --------------------------------------------------------------------------
def plot_three_panel_chart(df, ticker, market="JP", purchase_date=None, avg_cost=None) -> go.Figure:
    """
    銘柄の3段テクニカルチャートを生成（Plotly）
    旧株価推移機能を統合し、購入日の注釈と平均取得コスト表示も含む
    
    パラメータ:
        df: DataFrame - 銘柄のヒストリカルデータ
        ticker: str - 銘柄コード（例: "7203.T"）
        market: str - "JP"（日本株）または"US"（米国株）
        purchase_date: datetime - 購入日（注釈表示用）
        avg_cost: float - 平均取得価格（水平線表示用）
    
    戻り値:
        fig: go.Figure - Plotlyチャートオブジェクト
    """
    # 市場に応じたSMAを選択
    if market == "JP":
        sma_short = "sma_20"  # データベース上の実際のカラム名
        sma_long = "sma_40"  # データベース上の実際のカラム名
        sma_short_display = "SMA20"  # 表示用の名前
        sma_long_display = "SMA40"   # 表示用の名前
    else:  # US
        sma_short = "sma_20"  # データベース上の実際のカラム名
        sma_long = "sma_40"  # データベース上の実際のカラム名
        sma_short_display = "SMA20"  # 表示用の名前
        sma_long_display = "SMA40"   # 表示用の名前
    
    # サブプロット作成（3段パネル）
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.02,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}],
               [{}]]
    )
    
    # --- 1段目: 価格 + SMA + ボリューム ---
    # 価格
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["close"],
            mode="lines",
            name="価格",
            line=dict(color="#1f77b4", width=1.5)
        ),
        row=1, col=1, secondary_y=False
    )
    
    # 短期SMA
    if sma_short in df.columns and not df[sma_short].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df[sma_short],
                mode="lines",
                name=sma_short_display,
                line=dict(color="#ff7f0e", width=1, dash="dot")
            ),
            row=1, col=1, secondary_y=False
        )
    
    # 長期SMA
    if sma_long in df.columns and not df[sma_long].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df[sma_long],
                mode="lines",
                name=sma_long_display,
                line=dict(color="#2ca02c", width=1, dash="dot")
            ),
            row=1, col=1, secondary_y=False
        )
    
    # ボリューム
    if "volume" in df.columns and not df["volume"].isna().all():
        fig.add_trace(
            go.Bar(
                x=df["date"], y=df["volume"],
                name="Volume",
                marker=dict(color="#d3d3d3", opacity=0.5)
            ),
            row=1, col=1, secondary_y=True
        )
    
    # 取得価格水平線（旧株価推移機能から統合）
    if avg_cost is not None:
        fig.add_hline(
            y=avg_cost,
            line=dict(color="orange", dash="dot"),
            row=1, col=1
        )
        
        # 取得日アノテーション
        if purchase_date is not None:
            fig.add_annotation(
                x=purchase_date,
                y=avg_cost,
                text=f"Buy: {purchase_date.date()}",
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )
    
    # --- 2段目: RSI + MACD ヒストグラム ---
    # MACDヒストグラム（最初に描画）
    if "macd_hist" in df.columns and not df["macd_hist"].isna().all():
        macd_colors = ["#d62728" if val < 0 else "#2ca02c" for val in df["macd_hist"]]
        fig.add_trace(
            go.Bar(
                x=df["date"], y=df["macd_hist"],
                name="MACD Hist",
                marker=dict(color=macd_colors)
            ),
            row=2, col=1, secondary_y=True
        )
        
        # MACD補助線（0）
        fig.add_shape(
            type="line", x0=df["date"].iloc[0], x1=df["date"].iloc[-1], y0=0, y1=0,
            line=dict(color="#7f7f7f", width=1, dash="dash"),
            row=2, col=1, secondary_y=True
        )
    
    # RSI（最後に描画して前面に表示）
    if "rsi_14" in df.columns and not df["rsi_14"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["rsi_14"],
                mode="lines",
                name="RSI(14)",
                line=dict(color="#9467bd", width=2)  # 線を太くして見やすく
            ),
            row=2, col=1, secondary_y=False
        )
        
        # RSI補助線（30, 70）
        fig.add_shape(
            type="line", x0=df["date"].iloc[0], x1=df["date"].iloc[-1], y0=30, y1=30,
            line=dict(color="#d62728", width=1, dash="dash"),
            row=2, col=1
        )
        fig.add_shape(
            type="line", x0=df["date"].iloc[0], x1=df["date"].iloc[-1], y0=70, y1=70,
            line=dict(color="#d62728", width=1, dash="dash"),
            row=2, col=1
        )
    
    # --- 3段目: ATR ---
    if "atr_14" in df.columns and not df["atr_14"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df["atr_14"],
                mode="lines",
                name="ATR(14)",
                line=dict(color="#8c564b", width=1.5)
            ),
            row=3, col=1
        )
    
    # --- レイアウト設定 ---
    # Y軸タイトル
    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="MACD", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    
    # X軸は最下段のみラベル表示
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # 全体レイアウト
    currency = "JPY" if market == "JP" else "USD"
    fig.update_layout(
        title=f"{ticker} Technical Chart ({currency})",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


# --------------------------------------------------------------------------
# 5. テーブル HTML
# --------------------------------------------------------------------------
def create_portfolio_table(df: pd.DataFrame) -> str:
    """ポートフォリオテーブルHTML生成（見やすいスタイル適用）"""
    tbl_cols = ["symbol", "qty_total", "avg_cost", "close", "unrealized_pl", "pl_percent", "annualized_return"]
    
    # テーブルスタイル
    styles = """
    <style>
    .portfolio-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-family: sans-serif;
        font-size: 15px;
    }
    .portfolio-table th {
        background-color: #3498db;
        color: white;
        padding: 8px 12px;
        text-align: center;
    }
    .portfolio-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #ddd;
        text-align: right;
    }
    .portfolio-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .portfolio-table tr:hover {
        background-color: #e3f2fd;
    }
    .portfolio-table .positive {
        color: green;
        font-weight: bold;
    }
    .portfolio-table .negative {
        color: red;
        font-weight: bold;
    }
    </style>
    """
    
    # データフレームのコピーを作成してカラム名を日本語に変更
    df_display = df[tbl_cols].copy()
    df_display.columns = ["銘柄", "保有数量", "平均取得価格", "現在価格", "含み損益", "利益率(%)", "年率換算(%)"]
    
    # HTMLテーブル生成
    html_table = df_display.round(2).to_html(index=False, classes="portfolio-table")
    
    # 損益を色分け（正の値は緑、負の値は赤）
    html_table = html_table.replace('>-', ' class="negative">-')
    
    # 正規表現を使わない方法で数値を装飾（正規表現だとエラー）
    import re
    pattern = re.compile(r'<td>(\d+\.\d+)</td>')
    html_table = pattern.sub(r'<td class="positive">\1</td>', html_table)
    
    return styles + html_table


def make_realized_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    売却済みロットの損益を計算（売却価格・数量・取得価格から）
    """
    df = df.copy()
    # 売却済みのみ抽出
    sold_df = df[df["sold"] == True].copy()
    # 売却日・売却価格がなければ0や空文字列が入っている可能性があるので補正
    sold_df["sold_price"] = pd.to_numeric(sold_df["sold_price"], errors="coerce").fillna(0)
    sold_df["unrealized_pl"] = (sold_df["sold_price"] - sold_df["price"]) * sold_df["qty"]
    sold_df["close"] = sold_df["sold_price"]  # 売却時点の価格として
    sold_df["pl_percent"] = ((sold_df["sold_price"] / sold_df["price"] - 1) * 100).round(2)
    
    # 年率換算利益率を計算（売却済み）
    sold_df["annualized_return"] = sold_df.apply(
        lambda row: calculate_annualized_return(
            pd.to_datetime(row["date"]), 
            pd.to_datetime(row["sold_date"]) if pd.notna(row.get("sold_date")) else datetime.now(), 
            row["pl_percent"]
        ), 
        axis=1
    )
    
    sold_df["status"] = "売却済み"
    return sold_df


def make_alltime_portfolio_table(df: pd.DataFrame, engine: Engine) -> str:
    """
    保有中＋売却済みの全ロットの損益テーブルHTMLを生成
    """
    # 保有中
    holding_df = df[df["sold"] == False].copy()
    if not holding_df.empty:
        symbols = holding_df["symbol"].unique().tolist()
        query = text(
            """
            WITH latest AS (
              SELECT DISTINCT ON (symbol) symbol, date, close
              FROM fmp_data.daily_prices
              WHERE symbol = ANY(:symbols)
              ORDER BY symbol, date DESC
            )
            SELECT * FROM latest;
            """
        )
        px = pd.read_sql_query(query, engine, params={"symbols": symbols})
        holding_df = holding_df.merge(px, on="symbol", how="left")
        holding_df["unrealized_pl"] = (holding_df["close"] - holding_df["price"]) * holding_df["qty"]
        holding_df["pl_percent"] = ((holding_df["close"] / holding_df["price"] - 1) * 100).round(2)
        
        # 年率換算利益率を計算（保有中）
        current_date = datetime.now()
        holding_df["annualized_return"] = holding_df.apply(
            lambda row: calculate_annualized_return(
                pd.to_datetime(row["date"]), 
                current_date, 
                row["pl_percent"]
            ), 
            axis=1
        )
        
        holding_df["status"] = "Unrealized"
        
    # 売却済み
    sold_df = make_realized_pl(df)
    
    # 結合
    all_df = pd.concat([holding_df, sold_df], ignore_index=True)
    # 並び順
    all_df = all_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # テーブルカラム
    tbl_cols = ["symbol", "qty", "price", "close", "unrealized_pl", "pl_percent", "annualized_return", "status"]
    
    # データフレームのコピーを作成してカラム名を日本語に変更
    df_display = all_df[tbl_cols].copy()
    df_display.columns = ["銘柄", "数量", "取得価格", "現在/売却価格", "損益", "利益率(%)", "年率換算(%)", "ステータス"]
    
    styles = """
    <style>
    .portfolio-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-family: sans-serif; font-size: 15px; }
    .portfolio-table th { background-color: #3498db; color: white; padding: 8px 12px; text-align: center; }
    .portfolio-table td { padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: right; }
    .portfolio-table tr:nth-child(even) { background-color: #f2f2f2; }
    .portfolio-table tr:hover { background-color: #e3f2fd; }
    .portfolio-table .positive { color: green; font-weight: bold; }
    .portfolio-table .negative { color: red; font-weight: bold; }
    </style>
    """
    html_table = df_display.round(2).to_html(index=False, classes="portfolio-table")
    html_table = html_table.replace('>-', ' class="negative">-')
    import re
    pattern = re.compile(r'<td>(\d+\.\d+)</td>')
    html_table = pattern.sub(r'<td class="positive">\1</td>', html_table)
    return styles + html_table


def build_alltime_portfolio_section(engine: Engine) -> str:
    """
    保有中＋売却済みの全ロットの損益実績ページHTML（通貨ごとに分割したグラフ＋テーブル、合計行付き）
    """
    import plotly.graph_objects as go
    import html
    df = load_all_transactions(engine)
    # 保有中
    holding_df = df[df["sold"] == False].copy()
    if not holding_df.empty:
        symbols = holding_df["symbol"].unique().tolist()
        query = text(
            """
            WITH latest AS (
              SELECT DISTINCT ON (symbol) symbol, date, close
              FROM fmp_data.daily_prices
              WHERE symbol = ANY(:symbols)
              ORDER BY symbol, date DESC
            )
            SELECT * FROM latest;
            """
        )
        px = pd.read_sql_query(query, engine, params={"symbols": symbols})
        holding_df = holding_df.merge(px, on="symbol", how="left")
        holding_df["unrealized_pl"] = (holding_df["close"] - holding_df["price"]) * holding_df["qty"]
        holding_df["pl_percent"] = ((holding_df["close"] / holding_df["price"] - 1) * 100).round(2)
        holding_df["status"] = "Unrealized"
    # 売却済み
    sold_df = make_realized_pl(df)
    sold_df["status"] = "Realized"
    # 結合
    all_df = pd.concat([holding_df, sold_df], ignore_index=True)
    all_df = all_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 通貨ごとにグラフ・テーブルを分割
    graph_htmls = []
    table_htmls = []
    for cur in all_df["currency"].unique():
        cur_df = all_df[all_df["currency"] == cur]
        
        # --- グラフ ---
        fig = go.Figure()
        
        # ステータスごとに処理（Unrealized, Realized）
        for status in ["Unrealized", "Realized"]:
            s_df = cur_df[cur_df["status"] == status]
            if s_df.empty:
                continue
                
            # プラスの値とマイナスの値を分ける
            positive_df = s_df[s_df["unrealized_pl"] >= 0]
            negative_df = s_df[s_df["unrealized_pl"] < 0]
            
            # プラスの値のトレース（利益）
            if not positive_df.empty:
                profit_values = positive_df["unrealized_pl"].tolist()
                profit_symbols = positive_df["symbol"].tolist()
                
                color = "#2ecc71" if status == "Unrealized" else "#3498db"  # 緑：保有中利益、青：売却済み利益
                
                # プラス値のホバーテキストを生成
                profit_hover_text = [f"{symbol}<br>+{val:,.0f} {cur}" for symbol, val in zip(profit_symbols, profit_values)]
                
                fig.add_trace(go.Bar(
                    x=[status] * len(profit_values),
                    y=profit_values,
                    name=f"{status} Profit",
                    marker_color=color,
                    text=profit_hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True
                ))
            
            # マイナスの値のトレース（損失）
            if not negative_df.empty:
                loss_values = negative_df["unrealized_pl"].tolist()
                loss_symbols = negative_df["symbol"].tolist()
                
                color = "#e74c3c"  # 赤：損失
                
                # マイナス値のホバーテキストを生成
                loss_hover_text = [f"{symbol}<br>{val:,.0f} {cur}" for symbol, val in zip(loss_symbols, loss_values)]
                
                fig.add_trace(go.Bar(
                    x=[status] * len(loss_values),
                    y=loss_values,
                    name=f"{status} Loss",
                    marker_color=color,
                    text=loss_hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True
                ))
        
        fig.update_layout(
            barmode="relative",  # 相対積み上げで正負が正しく表示される
            title={"text": f"{cur}建て All-time P/L (Stacked)", "x":0.5, "xanchor":"center"},
            yaxis_title=f"Total P/L ({cur})",
            xaxis_title="Status",
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        graph_htmls.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if cur==all_df["currency"].unique()[0] else False))
        
        # --- テーブル ---
        tbl_cols = ["symbol", "qty", "price", "close", "unrealized_pl", "pl_percent", "annualized_return", "status"]
        cur_tbl = cur_df[tbl_cols].round(2)
        
        # 合計行を追加
        total_unrealized_pl = cur_tbl["unrealized_pl"].sum()
        total_invested = (cur_tbl["qty"] * cur_tbl["price"]).sum()
        if total_invested != 0:
            total_pl_percent = total_unrealized_pl / total_invested * 100
        else:
            total_pl_percent = 0
            
        # 加重平均年率換算利益率を計算
        total_annualized = 0.0
        if total_invested != 0:
            weighted_sum = 0.0
            for _, row in cur_tbl.iterrows():
                if pd.notna(row["annualized_return"]) and pd.notna(row["qty"]) and pd.notna(row["price"]):
                    investment = row["qty"] * row["price"]
                    weighted_sum += row["annualized_return"] * investment
            total_annualized = round(weighted_sum / total_invested, 2)
        
        sum_row = {
            "symbol": "<b>合計</b>",
            "qty": cur_tbl["qty"].sum(),
            "price": "",
            "close": "",
            "unrealized_pl": total_unrealized_pl,
            "pl_percent": round(total_pl_percent, 2),
            "annualized_return": total_annualized,
            "status": ""
        }
        cur_tbl = pd.concat([cur_tbl, pd.DataFrame([sum_row])], ignore_index=True)
        
        # データフレームのコピーを作成してカラム名を日本語に変更
        df_display = cur_tbl.copy()
        df_display.columns = ["銘柄", "数量", "取得価格", "現在/売却価格", "損益", "利益率(%)", "年率換算(%)", "ステータス"]
        
        styles = """
        <style>
        .portfolio-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-family: sans-serif; font-size: 15px; }
        .portfolio-table th { background-color: #3498db; color: white; padding: 8px 12px; text-align: center; }
        .portfolio-table td { padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: right; }
        .portfolio-table tr:nth-child(even) { background-color: #f2f2f2; }
        .portfolio-table tr:hover { background-color: #e3f2fd; }
        .portfolio-table .positive { color: green; font-weight: bold; }
        .portfolio-table .negative { color: red; font-weight: bold; }
        </style>
        """
        html_table = df_display.to_html(index=False, classes="portfolio-table", escape=False)
        html_table = html_table.replace('>-', ' class="negative">-')
        import re
        pattern = re.compile(r'<td>(\d+\.\d+)</td>')
        html_table = pattern.sub(r'<td class="positive">\1</td>', html_table)
        table_htmls.append(f"<h3>{cur}建て</h3>" + styles + html_table)
    
    # 年次損益グラフ（Realizedのみ）
    realized_graph_htmls = []
    realized_df = all_df[(all_df["status"] == "Realized") & (all_df["sold"] == True)].copy()
    if not realized_df.empty and "sold_date" in realized_df.columns:
        realized_df["sold_year"] = pd.to_datetime(realized_df["sold_date"], errors="coerce").dt.year
        for i, cur in enumerate(realized_df["currency"].unique()):
            cur_df = realized_df[realized_df["currency"] == cur]
            year_pl = cur_df.groupby("sold_year")["unrealized_pl"].sum().reset_index()
            fig = go.Figure()
            fig.add_bar(x=year_pl["sold_year"], y=year_pl["unrealized_pl"], marker_color="#3498db")
            fig.update_layout(
                title=f"{cur}建て 年次Realized損益（利確のみ）",
                xaxis_title="年",
                yaxis_title=f"Realized損益合計 ({cur})",
                height=350,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            include_js = "cdn" if i == 0 else False
            realized_graph_htmls.append(f'<div style="margin-bottom:20px;">' + fig.to_html(full_html=False, include_plotlyjs=include_js) + '</div>')
    # 日本説明
    explanation = """
    <!DOCTYPE html>
    <html lang=\"ja\">
    <head>
      <meta charset=\"utf-8\">
      <title>全期間の損益実績</title>
    </head>
    <body>
    <h2>全期間の損益実績（保有中＋売却済み）</h2>
    <div style='margin-bottom:10px;'>
      <b>各通貨ごとにグラフ・表を分割表示:</b> <br>
      <span style='color:#2ecc71;'>緑=保有中プラス</span>、<span style='color:#3498db;'>青=売却済みプラス</span>、<span style='color:#e74c3c;'>赤=損失</span>。<br>
      status列は <b>Unrealized=保有中</b>、<b>Realized=売却済み</b> を意味します。<br>
      <b>表の最下段に合計（Sum）を表示しています。</b><br>
      <b>年率換算(%):</b> 保有期間を考慮した年率換算利益率。複数回買い増しした銘柄は投資額で加重平均した値を表示。<br>
      例：50%の利益を2年で達成した場合、年率換算は約22.5%となります。
    </div>
    <h3>年次利確損益グラフ（通貨別）</h3>
    <div style='margin-bottom:10px;'>
      各年に売却（利確）した損益の合計を通貨ごとに棒グラフで表示しています。
    </div>
    """
    # HTML組み立て
    html_out = explanation + "\n".join(realized_graph_htmls) + "\n".join(graph_htmls) + "<br>".join(table_htmls) + "</body></html>"
    return html_out


# --------------------------------------------------------------------------
# 6. エントリポイント（daily_report が呼び出す関数）
# --------------------------------------------------------------------------
def build_portfolio_section(engine: Engine) -> Tuple[str, List[go.Figure]]:
    """
    戻り値
    -------
    explanation_html : str
        <h2> を含むテーブル HTML
    figs : List[go.Figure]
        0: 損益バー、1: テクニカルチャート群、2〜: 通貨ごとのロット別含み損益
    """
    # 取引データ読み込み
    df_tx = load_transactions(engine)
    
    # ポジション集計
    df_pos = make_current_positions(df_tx)
    df_pos = enrich_with_latest_prices(engine, df_pos)

    # テーブルHTML
    html = "<h2>My Portfolio</h2>" + create_portfolio_table(df_pos)

    # ロット単位のデータ作成
    df_lot = df_tx.copy()
    
    # シンボルのリスト
    symbols = df_lot["symbol"].unique().tolist()
    
    # 現在価格を取得
    query = text(
        """
        WITH latest AS (
          SELECT DISTINCT ON (symbol) symbol, date, close
          FROM fmp_data.daily_prices
          WHERE symbol = ANY(:symbols)
          ORDER BY symbol, date DESC
        )
        SELECT * FROM latest;
        """
    )
    df_prices = pd.read_sql_query(query, engine, params={"symbols": symbols})
    
    # ロットデータに現在価格を追加
    df_lot = df_lot.merge(df_prices, on="symbol", how="left")
    
    # ロット単位の損益計算
    df_lot["unrealized_pl"] = df_lot["qty"] * (df_lot["close"] - df_lot["price"])
    
    # ロット別の年率換算利益率を計算
    current_date = datetime.now()
    df_lot["pl_percent"] = ((df_lot["close"] / df_lot["price"]) - 1) * 100
    
    # date列が存在するかチェックし、適切な列名を使用
    date_col = "date"
    if "date" not in df_lot.columns:
        if "date_x" in df_lot.columns:
            date_col = "date_x"
        elif "date_y" in df_lot.columns:
            date_col = "date_y"
        else:
            # デバッグ用：利用可能な列を表示
            print(f"利用可能な列: {df_lot.columns.tolist()}")
            # 日付列が見つからない場合はスキップ
            df_lot["annualized_return"] = 0.0
            date_col = None
    
    if date_col:
        df_lot["annualized_return"] = df_lot.apply(
            lambda row: calculate_annualized_return(
                pd.to_datetime(row[date_col]), 
                current_date, 
                row["pl_percent"]
            ), 
            axis=1
        )
    
    # 図作成
    fig_bars = make_pl_bars(df_pos)
    lot_figures = make_per_lot_bars(df_lot, engine)
    
    # 通貨別のロット図を結合した1つの図を作成
    combined_fig = go.Figure()
    
    # すべての通貨の図から全トレースを結合図に追加
    for fig in lot_figures:
        for trace in fig.data:
            combined_fig.add_trace(trace)
    
    # 結合図のレイアウト設定
    combined_fig.update_layout(
        title={
            'text': "ロット別（買付日別）含み損益（全通貨）",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        barmode='relative',
        height=700,
        margin=dict(l=50, r=50, t=80, b=150),
        legend=dict(
            orientation="h", 
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            traceorder="grouped"
        )
    )
    
    # 3段テクニカルチャートを各銘柄ごとに作成（株価推移の代わりに使用）
    tech_charts = []
    
    for symbol in df_lot["symbol"].unique():
        # シンボルの市場を判定（.Tで終わるなら日本株）
        market = "JP" if symbol.endswith(".T") else "US"
        
        # ポジション情報を取得（購入日と平均取得価格）
        position_info = df_pos[df_pos["symbol"] == symbol].iloc[0]
        purchase_date = pd.to_datetime(position_info["purchase_date"])
        avg_cost = position_info["avg_cost"]
        
        # 購入日の1ヶ月前を計算
        start_date = purchase_date - timedelta(days=30)
        
        # テクニカル指標とボリュームデータを取得
        tech_query = text("""
            WITH price_vol AS (
                SELECT p.symbol, p.date, p.close, p.volume
                FROM fmp_data.daily_prices p
                WHERE p.symbol = :symbol
                  AND p.date >= :start_date
                ORDER BY p.date
            ),
            tech AS (
                SELECT t.symbol, t.date, t.sma_5, t.sma_20, t.sma_25, t.sma_40, t.sma_50, 
                       t.rsi_14, t.macd_hist, t.atr_14
                FROM calculated_metrics.technical_indicators t
                WHERE t.symbol = :symbol
                  AND t.date >= :start_date
                ORDER BY t.date
            )
            SELECT p.symbol, p.date, p.close, p.volume, 
                   t.sma_5, t.sma_20, t.sma_25, t.sma_40, t.sma_50, 
                   t.rsi_14, t.macd_hist, t.atr_14
            FROM price_vol p
            LEFT JOIN tech t ON p.symbol = t.symbol AND p.date = t.date
            ORDER BY p.date;
        """)
        
        df_tech = pd.read_sql_query(tech_query, engine, params={
            "symbol": symbol,
            "start_date": start_date
        })
        
        if not df_tech.empty:
            # 3段チャート生成（購入日と平均取得価格も渡す）
            tech_fig = plot_three_panel_chart(
                df_tech, symbol, market, 
                purchase_date=purchase_date, 
                avg_cost=avg_cost
            )
            tech_charts.append(tech_fig)
    
    # 最終的に返すFigureリスト
    result_figures = [fig_bars]
    result_figures.extend(tech_charts)  # テクニカルチャートを追加
    result_figures.extend(lot_figures)  # ロット別含み損益を追加

    return html, result_figures