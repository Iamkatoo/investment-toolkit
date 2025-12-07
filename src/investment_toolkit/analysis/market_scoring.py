"""
株式市場のスコアリングロジックを実装します。
マイクロスコア(個別銘柄評価)とマクロスコア(市場全体評価)の計算を行います。
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from sqlalchemy.engine import Engine
from pathlib import Path
import json

from investment_analysis.analysis.score_weights import MICRO_SCORE_WEIGHTS, MACRO_SCORE_WEIGHTS


def calc_volume_score(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.Series:
    """
    出来高と価格変化をもとに日次スコアを計算する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        インデックスが日付 (datetime64[ns]) または date カラムがあるDataFrame
        必須カラム: close, volume
    params : dict, optional
        計算パラメータ。None の場合はデフォルトパラメータを使用
        
    Returns
    -------
    pd.Series
        出来高スコアのSeries（インデックスはdfと同じ）
        
    Raises
    ------
    ValueError
        必要なカラムが欠けている場合
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> # サンプルデータ作成
    >>> dates = [datetime.now() - timedelta(days=i) for i in range(10)]
    >>> close = np.random.rand(10) * 100
    >>> volume = np.random.rand(10) * 1000
    >>> df = pd.DataFrame({'close': close, 'volume': volume}, index=dates)
    >>> # 出来高スコア計算
    >>> score = calc_volume_score(df)
    >>> print(score)
    """
    # 必要なカラムの確認
    required_cols = ['close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必要なカラム '{col}' がDataFrameにありません")
    
    # デフォルトパラメータ
    default_params = {
        "ma_window": 5,          # 出来高移動平均日数
        "ret_thresh": 0.01,      # 騰落率±判定境界（+1% / -1%）
        "vol_bins": [0.3, 1.0],  # 出来高比 +30% / +100% 境界
        "scores": {
            "strong_up":  2,  # 価格↑ & 出来高 +100%以上
            "weak_up":    1,  # 価格↑ & 出来高 +30〜100%
            "fade_up":   -1,  # 価格↑ & 出来高 −30%以下
            "strong_dn": -2,  # 価格↓ & 出来高 +100%以上
            "weak_dn":   -1,  # 価格↓ & 出来高 +30〜100%
            "fade_dn":    1   # 価格↓ & 出来高 −30%以下
        }
    }
    
    # 提供されたパラメータまたはデフォルトを使用
    if params is None:
        params = default_params
    else:
        # デフォルトを提供されたパラメータで更新
        for key, value in params.items():
            default_params[key] = value
        params = default_params
    
    # パラメータ抽出
    ma_window = params['ma_window']
    ret_thresh = params['ret_thresh']
    vol_bins = params['vol_bins']
    scores = params['scores']
    
    # 出来高移動平均の計算
    volume_ma = df['volume'].rolling(ma_window).mean()
    
    # 出来高変化率の計算
    volume_change = (df['volume'] / volume_ma) - 1
    
    # 価格変化率の計算
    price_ret = df['close'].pct_change()
    
    # スコアシリーズの初期化
    score = pd.Series(0, index=df.index)
    
    # 条件に基づいてスコアを割り当て
    # Strong up: 価格上昇 >= thresh & 出来高増加 >= vol_bins[1]
    mask_strong_up = (price_ret >= ret_thresh) & (volume_change >= vol_bins[1])
    score[mask_strong_up] = scores['strong_up']
    
    # Weak up: 価格上昇 >= thresh & vol_bins[0] <= 出来高増加 < vol_bins[1]
    mask_weak_up = (price_ret >= ret_thresh) & (volume_change >= vol_bins[0]) & (volume_change < vol_bins[1])
    score[mask_weak_up] = scores['weak_up']
    
    # Fade up: 価格上昇 >= thresh & 出来高減少 <= -vol_bins[0]
    mask_fade_up = (price_ret >= ret_thresh) & (volume_change <= -vol_bins[0])
    score[mask_fade_up] = scores['fade_up']
    
    # Strong down: 価格下落 <= -thresh & 出来高増加 >= vol_bins[1]
    mask_strong_dn = (price_ret <= -ret_thresh) & (volume_change >= vol_bins[1])
    score[mask_strong_dn] = scores['strong_dn']
    
    # Weak down: 価格下落 <= -thresh & vol_bins[0] <= 出来高増加 < vol_bins[1]
    mask_weak_dn = (price_ret <= -ret_thresh) & (volume_change >= vol_bins[0]) & (volume_change < vol_bins[1])
    score[mask_weak_dn] = scores['weak_dn']
    
    # Fade down: 価格下落 <= -thresh & 出来高減少 <= -vol_bins[0]
    mask_fade_dn = (price_ret <= -ret_thresh) & (volume_change <= -vol_bins[0])
    score[mask_fade_dn] = scores['fade_dn']
    
    return score


def test_calc_volume_score():
    """calc_volume_score関数のテスト"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 再現性のためにランダムシードを固定
    np.random.seed(42)
    
    # 30日分のダミーデータを生成
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    close = np.random.rand(30) * 100
    # いくつかの価格変動が閾値を超えるようにする
    close[5:10] = close[5:10] * 1.05  # 5% 上昇
    close[15:20] = close[15:20] * 0.95  # 5% 下落
    
    volume = np.random.rand(30) * 1000
    # いくつかの出来高変化が境界を超えるようにする
    volume[5:10] = volume[5:10] * 2.0  # 100% 増加
    volume[15:20] = volume[15:20] * 0.5  # 50% 減少
    
    df = pd.DataFrame({'close': close, 'volume': volume}, index=dates)
    
    # 出来高スコアを計算
    score = calc_volume_score(df)
    
    # スコアがSeriesとして返されることを確認
    assert isinstance(score, pd.Series)
    
    # スコアの長さがdfの長さと一致することを確認
    assert len(score) == len(df)
    
    # スコア値が期待範囲内であることを確認
    expected_scores = [-2, -1, 0, 1, 2]
    for s in score.dropna():
        assert s in expected_scores
    
    # カスタムパラメータでテスト
    custom_params = {
        "ma_window": 3,
        "ret_thresh": 0.02,
        "vol_bins": [0.2, 0.8],
        "scores": {
            "strong_up": 3,
            "weak_up": 2,
            "fade_up": -2,
            "strong_dn": -3,
            "weak_dn": -2,
            "fade_dn": 2
        }
    }
    
    custom_score = calc_volume_score(df, custom_params)
    
    # カスタムスコアがSeriesとして返されることを確認
    assert isinstance(custom_score, pd.Series)
    
    # デフォルトとカスタムスコアで少なくとも1つの値が異なることを確認
    assert not custom_score.equals(score)
    
    print("calc_volume_score関数のテストが成功しました")
    return True


def calculate_micro_score(engine: Engine, symbols: List[str], date: str = None) -> pd.DataFrame:
    """
    個別銘柄のスコアを計算します。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
    symbols : List[str]
        評価対象の銘柄コードリスト
    date : str, optional
        評価日（指定がない場合は最新日）
        
    Returns
    -------
    pd.DataFrame
        銘柄ごとのスコア詳細と合計スコアを含むデータフレーム
    """
    def _safe_set_value(df: pd.DataFrame, idx: int, col: str, value: float) -> None:
        """DataFrameに安全に値を設定する関数"""
        df.loc[idx, col] = value

    def _safe_update_dataframe(df: pd.DataFrame, mask: pd.Series, col: str, value: float) -> None:
        """DataFrameの値を安全に更新する関数"""
        df.loc[mask, col] = value

    if not symbols:
        return pd.DataFrame()
    
    # 最新の日付（評価日が指定されていない場合）を取得
    if date is None:
        query = """
        SELECT MAX(date) as latest_date 
        FROM fmp_data.daily_prices 
        WHERE symbol = %s
        """
        date = pd.read_sql(query, engine, params=(symbols[0],)).iloc[0]['latest_date']
    
    print(f"計算日: {date}")
    print(f"対象銘柄: {', '.join(symbols)}")
    
    # 各銘柄の価格データを取得
    result_records = []
    
    for symbol in symbols:
        print(f"銘柄 {symbol} のデータを取得中...")
        
        # 価格と出来高の基本データを取得
        basic_query = """
        SELECT symbol, date, close, volume
        FROM fmp_data.daily_prices
        WHERE symbol = %s
        ORDER BY date DESC
        LIMIT 60
        """
        
        try:
            # 基本データを取得
            df_basic = pd.read_sql(basic_query, engine, params=(symbol,))
            
            if df_basic.empty:
                print(f"警告: {symbol} のデータが取得できませんでした")
                continue
            
            # 最新日のデータ
            latest_row = df_basic.iloc[0].copy()
            latest_date = latest_row['date']
            
            # 前日のデータ
            prev_row = df_basic.iloc[1].copy() if len(df_basic) > 1 else None
            
            # テクニカル指標と価格データの取得
            data_query = """
            SELECT 
                ti.date,
                ti.atr_14,
                ti.sma_20,
                ti.sma_40,
                dp.volume,
                dp.close
            FROM calculated_metrics.technical_indicators ti
            JOIN fmp_data.daily_prices dp 
                ON ti.symbol = dp.symbol 
                AND ti.date = dp.date
            WHERE ti.symbol = %s
            AND ti.date = %s
            """
            
            try:
                data_df = pd.read_sql(data_query, engine, params=(symbol, latest_date))
                if not data_df.empty:
                    atr_14 = float(data_df['atr_14'].iloc[0])
                    sma_20 = float(data_df['sma_20'].iloc[0])
                    sma_40 = float(data_df['sma_40'].iloc[0])
                    volume = float(data_df['volume'].iloc[0])
                    close = float(data_df['close'].iloc[0])
                    
                    # ATRレシオの計算を追加
                    atr_ratio = atr_14 / close
                else:
                    atr_14 = latest_row['close'] * 0.02
                    sma_20 = latest_row['close']
                    sma_40 = latest_row['close']
                    volume = latest_row['volume']
                    close = latest_row['close']
                    
                    # デフォルトのATRレシオを設定
                    atr_ratio = 0.02  # 2%をデフォルト値として使用
            except Exception as e:
                print(f"エラー: {symbol} のデータ取得中に問題が発生しました - {e}")
                atr_14 = latest_row['close'] * 0.02
                sma_20 = latest_row['close']
                sma_40 = latest_row['close']
                volume = latest_row['volume']
                close = latest_row['close']
                atr_ratio = 0.02  # デフォルト値

            # 出来高5日平均の計算
            volume_query = """
            SELECT 
                dp.date,
                dp.volume
            FROM fmp_data.daily_prices dp
            WHERE dp.symbol = %s
            AND dp.date <= %s
            ORDER BY dp.date DESC
            LIMIT 5
            """
            
            try:
                volume_df = pd.read_sql(volume_query, engine, params=(symbol, latest_date))
                if len(volume_df) >= 5:
                    volume_ma5 = volume_df['volume'].mean()
                else:
                    volume_ma5 = volume
            except Exception as e:
                print(f"エラー: {symbol} の出来高データ取得中に問題が発生しました - {e}")
                volume_ma5 = volume

            # ゴールデンクロス/デッドクロスの判定
            cross_query = """
            WITH recent_data AS (
                SELECT 
                    date,
                    sma_20,
                    sma_40,
                    LAG(sma_20) OVER (ORDER BY date) as prev_sma_20,
                    LAG(sma_40) OVER (ORDER BY date) as prev_sma_40
                FROM calculated_metrics.technical_indicators
                WHERE symbol = %s
                AND date <= %s
                ORDER BY date DESC
                LIMIT 2
            )
            SELECT 
                CASE 
                    WHEN sma_20 > sma_40 AND prev_sma_20 <= prev_sma_40 THEN 1
                    ELSE 0
                END as golden_cross_day,
                CASE 
                    WHEN sma_20 < sma_40 AND prev_sma_20 >= prev_sma_40 THEN 1
                    ELSE 0
                END as dead_cross_day
            FROM recent_data
            WHERE date = %s
            """
            
            try:
                cross_df = pd.read_sql(cross_query, engine, params=(symbol, latest_date, latest_date))
                if not cross_df.empty:
                    golden_cross_day = int(cross_df['golden_cross_day'].iloc[0])
                    dead_cross_day = int(cross_df['dead_cross_day'].iloc[0])
                else:
                    golden_cross_day = 0
                    dead_cross_day = 0
            except Exception as e:
                print(f"エラー: {symbol} のクロスデータ取得中に問題が発生しました - {e}")
                golden_cross_day = 0
                dead_cross_day = 0

            # 記録を作成
            record = {
                'symbol': symbol,
                'date': latest_date,
                'close': close,
                'prev_close': prev_row['close'] if prev_row is not None else close,
                'volume': volume,
                'volume_ma5': volume_ma5,
                'sma_20': sma_20,
                'sma_40': sma_40,
                'atr_14': atr_14,
                'atr_ratio': atr_ratio,  # ATRレシオを追加
                'golden_cross_day': golden_cross_day,
                'dead_cross_day': dead_cross_day
            }
            
            result_records.append(record)
            print(f"{symbol}: {latest_date} のデータを取得しました")
            
        except Exception as e:
            print(f"エラー: {symbol} のデータ取得中に問題が発生しました - {e}")
    
    # データフレームに変換
    if not result_records:
        print("警告: どの銘柄のデータも取得できませんでした")
        return pd.DataFrame()
    
    df_target = pd.DataFrame(result_records)
    
    # セクターとインダストリーの情報を取得
    query_sectors = """
    WITH sector_data AS (
        SELECT 
            cg.symbol, 
            gs.sector_name, 
            gi.industry_name
        FROM reference.company_gics cg
        LEFT JOIN reference.gics_sector gs ON cg.sector_id = gs.sector_id
        LEFT JOIN reference.gics_industry gi ON cg.industry_id = gi.industry_id
        WHERE cg.symbol IN %s
    )
    SELECT 
        sd.symbol, 
        COALESCE(sd.sector_name, 'Unclassified') as sector, 
        COALESCE(sd.industry_name, 'Unclassified') as industry
    FROM sector_data sd
    """
    df_sectors = pd.read_sql(query_sectors, engine, params=(tuple(symbols),))
    
    # 手動でセクター情報を設定
    manual_sectors = {
        'NVDA': ('Information Technology', 'Semiconductors'),
        'TSM': ('Information Technology', 'Semiconductors'),
        '3826.T': ('Information Technology', 'Application Software'),
        '4847.T': ('Information Technology', 'Application Software'),
        'VTI': ('ETF', 'Broad Market ETF'),
        'SPY': ('ETF', 'S&P 500 ETF'),
        'QQQ': ('ETF', 'NASDAQ ETF'),
        'IWM': ('ETF', 'Russell 2000 ETF')
    }
    
    # 不足しているシンボルに対してセクター情報を手動で設定
    for symbol in symbols:
        missing_or_unclassified = False
        
        # 既存のデータがあるか確認
        existing_row = df_sectors[df_sectors['symbol'] == symbol]
        
        if existing_row.empty:
            missing_or_unclassified = True
            print(f"{symbol}はセクター情報テーブルに存在しません")
        elif existing_row['sector'].values[0] == 'Unclassified':
            missing_or_unclassified = True
            print(f"{symbol}のセクターが未分類です")
        
        # セクターが未分類または欠損している場合、手動データを使用
        if missing_or_unclassified and symbol in manual_sectors:
            sector, industry = manual_sectors[symbol]
            print(f"{symbol}に手動でセクター情報を設定: {sector} / {industry}")
            
            if symbol in df_sectors['symbol'].values:
                # 既存の行を更新
                df_sectors.loc[df_sectors['symbol'] == symbol, 'sector'] = sector
                df_sectors.loc[df_sectors['symbol'] == symbol, 'industry'] = industry
            else:
                # 新しい行を追加
                new_row = pd.DataFrame([{'symbol': symbol, 'sector': sector, 'industry': industry}])
                df_sectors = pd.concat([df_sectors, new_row], ignore_index=True)
    
    # セクター情報にNULLがある場合の処理
    df_sectors['sector'] = df_sectors['sector'].fillna('Unclassified')
    df_sectors['industry'] = df_sectors['industry'].fillna('Unclassified')
    
    print(f"銘柄ごとのセクター情報: {df_sectors.to_dict('records')}")
    
    # 価格データとセクター情報をマージ
    df_target = pd.merge(df_target, df_sectors, on='symbol', how='left')
    
    # 1. 前日比変化率スコア
    df_target['price_change_pct'] = ((df_target['close'] / df_target['prev_close']) - 1) * 100
    df_target['price_change_score'] = 0
    
    # +1%以上で+2点
    pos_mask = df_target['price_change_pct'] >= MICRO_SCORE_WEIGHTS['PRICE_CHANGE']['THRESHOLD_POSITIVE']
    _safe_update_dataframe(df_target, pos_mask, 'price_change_score', MICRO_SCORE_WEIGHTS['PRICE_CHANGE']['SCORE_POSITIVE'])
    
    # -1%以上下落で-2点
    neg_mask = df_target['price_change_pct'] <= MICRO_SCORE_WEIGHTS['PRICE_CHANGE']['THRESHOLD_NEGATIVE']
    _safe_update_dataframe(df_target, neg_mask, 'price_change_score', MICRO_SCORE_WEIGHTS['PRICE_CHANGE']['SCORE_NEGATIVE'])
    
    # 2. セクター平均との乖離率スコア
    # セクターごとの平均変化率を計算
    sector_avg = df_target.groupby('sector')['price_change_pct'].mean().reset_index()
    sector_avg.columns = ['sector', 'sector_avg_change']
    df_target = pd.merge(df_target, sector_avg, on='sector', how='left')
    
    # 乖離を計算
    df_target['sector_deviation'] = df_target['price_change_pct'] - df_target['sector_avg_change']
    df_target['sector_score'] = 0
    
    # セクター平均より2%以上下回る場合は-1点
    neg_sector_mask = (df_target['sector_deviation'] <= -MICRO_SCORE_WEIGHTS['SECTOR_DEVIATION']['THRESHOLD'])
    _safe_update_dataframe(df_target, neg_sector_mask, 'sector_score', MICRO_SCORE_WEIGHTS['SECTOR_DEVIATION']['SCORE'])
    
    # 3. インダストリー平均との乖離率スコア
    # インダストリーごとの平均変化率を計算
    industry_avg = df_target.groupby('industry')['price_change_pct'].mean().reset_index()
    industry_avg.columns = ['industry', 'industry_avg_change']
    df_target = pd.merge(df_target, industry_avg, on='industry', how='left')
    
    # 乖離を計算
    df_target['industry_deviation'] = df_target['price_change_pct'] - df_target['industry_avg_change']
    df_target['industry_score'] = 0
    
    # インダストリー平均より2%以上下回る場合は-1点
    neg_industry_mask = (df_target['industry_deviation'] <= -MICRO_SCORE_WEIGHTS['INDUSTRY_DEVIATION']['THRESHOLD'])
    _safe_update_dataframe(df_target, neg_industry_mask, 'industry_score', MICRO_SCORE_WEIGHTS['INDUSTRY_DEVIATION']['SCORE'])
    
    # 4. 出来高変化率スコア
    # 新しい出来高スコア計算関数を使用
    df_target['volume_score'] = 0  # 初期化
    for symbol in df_target['symbol'].unique():
        symbol_data = df_target[df_target['symbol'] == symbol].copy()
        
        if len(symbol_data) > 0:
            # calc_volume_score関数の入力形式に合わせてデータを準備
            volume_data = pd.DataFrame({
                'close': symbol_data['close'],
                'volume': symbol_data['volume'],
                'volume_ma5': symbol_data['volume_ma5']
            })
            
            print(f"\n{symbol}の出来高スコア計算:")
            print(f"現在の出来高: {volume_data['volume'].iloc[0]:,.0f}")
            print(f"5日平均出来高: {volume_data['volume_ma5'].iloc[0]:,.0f}")
            
            # 5日間の移動平均が計算済みの場合はそれを使用
            if 'volume_ma5' in volume_data.columns and not volume_data['volume_ma5'].isna().all():
                # マニュアルで出来高変化率を計算
                volume_change_pct = ((volume_data['volume'] / volume_data['volume_ma5']) - 1) * 100
                df_target.loc[symbol_data.index, 'volume_change_pct'] = volume_change_pct
                print(f"出来高変化率: {volume_change_pct.iloc[0]:.1f}%")
            else:
                # データが十分でない場合はゼロを設定
                df_target.loc[symbol_data.index, 'volume_change_pct'] = 0
                print("データが不十分のため出来高変化率は0%")
            
            # 出来高スコアを計算
            try:
                # 計算用パラメータ
                volume_params = {
                    "ma_window": 5,
                    "ret_thresh": 0.01,
                    "vol_bins": [0.3, 1.0],
                    "scores": {
                        "strong_up": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_VERY_HIGH'],  # +2
                        "weak_up": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_HIGH'],         # +1
                        "fade_up": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_LOW'],         # -1
                        "strong_dn": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_VERY_LOW'],  # -2
                        "weak_dn": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_LOW'],         # -1
                        "fade_dn": MICRO_SCORE_WEIGHTS['VOLUME_CHANGE']['SCORE_HIGH']         # +1
                    }
                }
                
                # データが十分あれば新しい関数で計算
                if len(volume_data) >= 5:
                    volume_score = calc_volume_score(volume_data, volume_params)
                    if not volume_score.empty:
                        df_target.loc[symbol_data.index, 'volume_score'] = volume_score.iloc[-1]
                        print(f"出来高スコア: {volume_score.iloc[-1]}")
                else:
                    print(f"{symbol}: データが不十分のため出来高スコアは0")
                    df_target.loc[symbol_data.index, 'volume_score'] = 0
            except Exception as e:
                print(f"{symbol}の出来高スコア計算でエラー: {e}")
                # エラー時はスコアを0に設定
                df_target.loc[symbol_data.index, 'volume_score'] = 0

    # 5. ゴールデンクロス/デッドクロス判定
    # ゴールデンクロス/デッドクロスからの日数を計算
    df_target['gc_dc_score'] = 0.0
    df_target['gc_dc_days'] = 0  # クロスからの日数を記録
    
    # ゴールデンクロス/デッドクロススコアの計算
    for idx, row in df_target.iterrows():
        symbol = row['symbol']
        golden_cross_day = row['golden_cross_day']
        dead_cross_day = row['dead_cross_day']
        
        print(f"\n{symbol}のGC/DCスコア計算:")
        print(f"SMA20: {row['sma_20']:.2f}")
        print(f"SMA40: {row['sma_40']:.2f}")
        
        # ゴールデンクロスが発生した日を特定
        if golden_cross_day == 1:
            # ゴールデンクロスが発生した日は+1点
            df_target.loc[idx, 'gc_dc_score'] = 1.0
            df_target.loc[idx, 'gc_dc_days'] = 0  # 発生日なので0日
            print(f"{symbol}: ゴールデンクロス発生日、スコア +1.0")
        
        # デッドクロスが発生した日を特定
        elif dead_cross_day == 1:
            # デッドクロスが発生した日は-1点
            df_target.loc[idx, 'gc_dc_score'] = -1.0
            df_target.loc[idx, 'gc_dc_days'] = 0  # 発生日なので0日
            print(f"{symbol}: デッドクロス発生日、スコア -1.0")
        
        # それ以外の場合は、過去のクロス情報とSMAの位置関係を考慮
        else:
            # 過去のクロス情報を取得
            cross_query = """
            WITH cross_dates AS (
                SELECT 
                    date,
                    sma_20,
                    sma_40,
                    CASE 
                        WHEN sma_20 > sma_40 AND LAG(sma_20) OVER (ORDER BY date) <= LAG(sma_40) OVER (ORDER BY date) THEN 'GC'
                        WHEN sma_20 < sma_40 AND LAG(sma_20) OVER (ORDER BY date) >= LAG(sma_40) OVER (ORDER BY date) THEN 'DC'
                        ELSE NULL
                    END as cross_type
                FROM calculated_metrics.technical_indicators
                WHERE symbol = %s
                AND date <= %s
                ORDER BY date DESC
            )
            SELECT 
                date,
                cross_type,
                sma_20,
                sma_40
            FROM cross_dates
            WHERE cross_type IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """
            
            try:
                cross_df = pd.read_sql(cross_query, engine, params=(symbol, row['date']))
                
                if not cross_df.empty:
                    cross_type = cross_df['cross_type'].iloc[0]
                    cross_date = cross_df['date'].iloc[0]
                    days_since_cross = (row['date'] - cross_date).days
                    
                    print(f"最後のクロス: {cross_type} ({cross_date})")
                    print(f"クロスからの経過日数: {days_since_cross}日")
                    
                    # 最大20日間で減衰
                    max_days = 20
                    
                    if days_since_cross <= max_days:
                        # 指数関数的に減衰: e^(-0.05*days)
                        decay_factor = np.exp(-0.05 * days_since_cross)
                        
                        if cross_type == 'GC':
                            # ゴールデンクロスの場合は正の値
                            score = 1.0 * decay_factor
                            # 0.05刻みに丸める
                            score = np.round(score / 0.05) * 0.05
                            df_target.loc[idx, 'gc_dc_score'] = score
                            df_target.loc[idx, 'gc_dc_days'] = days_since_cross
                            print(f"{symbol}: {cross_type} {days_since_cross}日前、減衰スコア +{score:.2f}")
                        elif cross_type == 'DC':
                            # デッドクロスの場合は負の値
                            score = -1.0 * decay_factor
                            # 0.05刻みに丸める
                            score = np.round(score / 0.05) * 0.05
                            df_target.loc[idx, 'gc_dc_score'] = score
                            df_target.loc[idx, 'gc_dc_days'] = days_since_cross
                            print(f"{symbol}: {cross_type} {days_since_cross}日前、減衰スコア {score:.2f}")
                    else:
                        # 20日以上経過している場合は現在のSMA位置関係を使用
                        if row['sma_20'] > row['sma_40']:
                            df_target.loc[idx, 'gc_dc_score'] = 0.5
                            print(f"{symbol}: SMA20 > SMA40、スコア +0.5")
                        elif row['sma_20'] < row['sma_40']:
                            df_target.loc[idx, 'gc_dc_score'] = -0.5
                            print(f"{symbol}: SMA20 < SMA40、スコア -0.5")
                else:
                    # クロスデータがない場合は現在のSMA位置関係を使用
                    if row['sma_20'] > row['sma_40']:
                        df_target.loc[idx, 'gc_dc_score'] = 0.5
                        print(f"{symbol}: SMA20 > SMA40（クロスデータなし）、スコア +0.5")
                    elif row['sma_20'] < row['sma_40']:
                        df_target.loc[idx, 'gc_dc_score'] = -0.5
                        print(f"{symbol}: SMA20 < SMA40（クロスデータなし）、スコア -0.5")
                
            except Exception as e:
                print(f"{symbol}のクロスデータ取得エラー: {e}")
                # エラーの場合は現在のSMA位置関係を使用
                if row['sma_20'] > row['sma_40']:
                    df_target.loc[idx, 'gc_dc_score'] = 0.5
                    print(f"{symbol}: SMA20 > SMA40（エラー発生）、スコア +0.5")
                elif row['sma_20'] < row['sma_40']:
                    df_target.loc[idx, 'gc_dc_score'] = -0.5
                    print(f"{symbol}: SMA20 < SMA40（エラー発生）、スコア -0.5")
    
    # 6. ATR比による異常値動き判定
    print("\nATRスコア計算:")
    for idx, row in df_target.iterrows():
        symbol = row['symbol']
        print(f"\n{symbol}のATRスコア計算:")
        print(f"ATR比: {row['atr_ratio']:.3f}")
        print(f"前日比: {row['price_change_pct']:.1f}%")
        
        # 上昇時のスコア
        up_mask = row['close'] > row['prev_close']
        
        # ATR比が2.0以上（上昇時）：+2点
        if up_mask and row['atr_ratio'] >= MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_VERY_HIGH']:
            df_target.loc[idx, 'atr_score'] = MICRO_SCORE_WEIGHTS['ATR_RATIO']['SCORE_VERY_HIGH_UP']
            print(f"ATR比 {row['atr_ratio']:.3f} >= 2.0（上昇時）、スコア +2.0")
        
        # ATR比が1.3以上2.0未満（上昇時）：+1点
        elif up_mask and row['atr_ratio'] >= MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_HIGH'] and row['atr_ratio'] < MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_VERY_HIGH']:
            df_target.loc[idx, 'atr_score'] = MICRO_SCORE_WEIGHTS['ATR_RATIO']['SCORE_HIGH_UP']
            print(f"ATR比 {row['atr_ratio']:.3f} >= 1.3（上昇時）、スコア +1.0")
        
        # 下落時のスコア
        elif not up_mask and row['atr_ratio'] >= MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_VERY_HIGH']:
            df_target.loc[idx, 'atr_score'] = MICRO_SCORE_WEIGHTS['ATR_RATIO']['SCORE_VERY_HIGH_DOWN']
            print(f"ATR比 {row['atr_ratio']:.3f} >= 2.0（下落時）、スコア -2.0")
        
        elif not up_mask and row['atr_ratio'] >= MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_HIGH'] and row['atr_ratio'] < MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_VERY_HIGH']:
            df_target.loc[idx, 'atr_score'] = MICRO_SCORE_WEIGHTS['ATR_RATIO']['SCORE_HIGH_DOWN']
            print(f"ATR比 {row['atr_ratio']:.3f} >= 1.3（下落時）、スコア -1.0")
        
        # ATR比が0.7未満（小動き）：スコア0（減点しない）
        elif row['atr_ratio'] < MICRO_SCORE_WEIGHTS['ATR_RATIO']['THRESHOLD_LOW']:
            df_target.loc[idx, 'atr_score'] = 0
            print(f"ATR比 {row['atr_ratio']:.3f} < 0.7、スコア 0.0")
        
        else:
            df_target.loc[idx, 'atr_score'] = 0
            print(f"ATR比 {row['atr_ratio']:.3f}、スコア 0.0")
    
    # 7. 総合スコアを計算
    score_columns = ['price_change_score', 'sector_score', 'industry_score', 
                      'volume_score', 'gc_dc_score', 'atr_score']
    
    # すべてのスコア列を表示
    print("\n各銘柄のスコア内訳:")
    for idx, row in df_target.iterrows():
        scores_detail = ", ".join([f"{col}: {row[col]}" for col in score_columns])
        print(f"{row['symbol']}: {scores_detail}")
    
    # スコア計算の検証（各スコアの合計を表示）
    for idx, row in df_target.iterrows():
        total = 0
        components = []
        for col in score_columns:
            if row[col] != 0:
                total += row[col]
                components.append(f"{col}:{row[col]:.2f}")
        print(f"{row['symbol']} 合計スコア: {total:.2f} ({', '.join(components)})")
    
    # 最終的なスコアを再計算（欠損値や異常値の修正後）
    df_target.loc[:, 'total_score'] = (
        df_target['price_change_score'].fillna(0)
        + df_target['sector_score'].fillna(0)
        + df_target['industry_score'].fillna(0)
        + df_target['volume_score'].fillna(0)
        + df_target['gc_dc_score'].fillna(0)
        + df_target['atr_score'].fillna(0)
    )
    
    # 合計スコアの再確認
    for idx, row in df_target.iterrows():
        print(f"{row['symbol']} 最終スコア: {row['total_score']:.2f}")
    
    # 結果を整理
    result_df = df_target[['symbol', 'date', 'close', 'price_change_pct', 'sector', 'industry',
                    'sector_deviation', 'industry_deviation', 'volume_change_pct',
                    'atr_ratio', 'price_change_score', 'sector_score', 'industry_score',
                    'volume_score', 'gc_dc_score', 'atr_score', 'total_score']]
    
    # 欠損値や異常値のチェックと修正
    for col in ['price_change_pct', 'sector_deviation', 'industry_deviation', 'volume_change_pct', 'atr_ratio', 'price_change_score', 'sector_score', 'industry_score', 'volume_score', 'gc_dc_score', 'atr_score']:
        if col in result_df.columns:
            # 欠損値を0に置き換え
            result_df.loc[:, col] = result_df[col].fillna(0)
            # 異常値（極端な値）をクリップ
            if col == 'price_change_pct':
                result_df.loc[:, col] = result_df[col].clip(-20, 20)  # ±20%に制限
            elif col == 'volume_change_pct':
                result_df.loc[:, col] = result_df[col].clip(-100, 500)  # -100%～+500%に制限
            elif col in ['sector_deviation', 'industry_deviation']:
                result_df.loc[:, col] = result_df[col].clip(-10, 10)  # ±10%に制限
            elif col == 'atr_ratio':
                result_df.loc[:, col] = result_df[col].clip(0, 5)  # 0～5に制限
            # スコア列は-2～+2の範囲に制限
            elif col in ['price_change_score', 'sector_score', 'industry_score', 'volume_score', 'gc_dc_score', 'atr_score']:
                result_df.loc[:, col] = result_df[col].clip(-2, 2)
    
    # 最終的なスコアを再計算（欠損値や異常値の修正後）
    result_df.loc[:, 'total_score'] = (
        result_df['price_change_score'].fillna(0)
        + result_df['sector_score'].fillna(0)
        + result_df['industry_score'].fillna(0)
        + result_df['volume_score'].fillna(0)
        + result_df['gc_dc_score'].fillna(0)
        + result_df['atr_score'].fillna(0)
    )
    
    # 欠けている銘柄があれば追加
    missing_symbols = [s for s in symbols if s not in result_df['symbol'].values]
    if missing_symbols:
        print(f"警告: 以下の銘柄のデータが結果に含まれていません: {', '.join(missing_symbols)}")
        
        # 欠けている銘柄のダミーレコードを追加
        for symbol in missing_symbols:
            sector_info = df_sectors[df_sectors['symbol'] == symbol]
            sector = sector_info['sector'].values[0] if not sector_info.empty else 'Unclassified'
            industry = sector_info['industry'].values[0] if not sector_info.empty else 'Unclassified'
            
            try:
                # 過去の価格データを使用
                past_price_query = """
                SELECT close, volume
                FROM fmp_data.daily_prices
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
                """
                
                past_data = pd.read_sql(past_price_query, engine, params=(symbol,))
                
                if not past_data.empty:
                    close = float(past_data['close'].values[0])
                    volume = float(past_data['volume'].values[0])
                    print(f"{symbol}: 過去データの価格({close})と出来高({volume})を使用")
                else:
                    close = 100.0
                    volume = 10000.0
                    print(f"{symbol}: デフォルト値を使用 - 価格({close})、出来高({volume})")
            except Exception as e:
                print(f"{symbol}の過去データ取得エラー: {e}")
                close = 100.0
                volume = 10000.0
            
            dummy_record = pd.DataFrame({
                'symbol': [symbol],
                'date': [pd.to_datetime(date)],
                'close': [close],
                'price_change_pct': [0.0],
                'sector': [sector],
                'industry': [industry],
                'sector_deviation': [0.0],
                'industry_deviation': [0.0],
                'volume_change_pct': [0.0],
                'atr_ratio': [0.0],
                'price_change_score': [0],
                'sector_score': [0],
                'industry_score': [0],
                'volume_score': [0],
                'gc_dc_score': [0],
                'atr_score': [0],
                'total_score': [0]
            })
            
            result_df = pd.concat([result_df, dummy_record], ignore_index=True)
            print(f"{symbol}のダミーレコードを追加しました")
    
    return result_df


def calculate_single_day_macro_score(df_merged: pd.DataFrame, current_idx: int) -> float:
    """
    特定の日付のマクロスコアを計算します。
    
    Parameters
    ----------
    df_merged : pd.DataFrame
        マージされた経済指標データ
    current_idx : int
        計算対象日のインデックス
        
    Returns
    -------
    float
        計算されたマクロスコア
    """
    score_components = {}
    total_score = 0
    
    # 1週間前、1ヶ月前、3ヶ月前のインデックス
    idx_1w = max(0, current_idx - 5)  # 営業日ベースで約1週間
    idx_1m = max(0, current_idx - 21)  # 営業日ベースで約1ヶ月
    idx_3m = max(0, current_idx - 63)  # 営業日ベースで約3ヶ月
    
    # 1. SP500の3ヶ月変化率
    if 'SPY' in df_merged.columns:
        sp500_3m_change = ((df_merged.iloc[current_idx]['SPY'] / df_merged.iloc[idx_3m]['SPY']) - 1) * 100
        
        if sp500_3m_change <= MACRO_SCORE_WEIGHTS['SP500']['THRESHOLD']:
            score_components['SP500_3M'] = MACRO_SCORE_WEIGHTS['SP500']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['SP500']['SCORE']
        else:
            score_components['SP500_3M'] = 0
    
    # 2. VIXの1週間変化率
    if 'VIX' in df_merged.columns:
        vix_1w_change = ((df_merged.iloc[current_idx]['VIX'] / df_merged.iloc[idx_1w]['VIX']) - 1) * 100
        
        if vix_1w_change >= MACRO_SCORE_WEIGHTS['VIX']['THRESHOLD']:
            score_components['VIX_1W'] = MACRO_SCORE_WEIGHTS['VIX']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['VIX']['SCORE']
        else:
            score_components['VIX_1W'] = 0
    
    # 3. GCUSDの3ヶ月変化率
    if 'GCUSD' in df_merged.columns:
        gold_3m_change = ((df_merged.iloc[current_idx]['GCUSD'] / df_merged.iloc[idx_3m]['GCUSD']) - 1) * 100
        
        if gold_3m_change >= MACRO_SCORE_WEIGHTS['GOLD']['THRESHOLD']:
            score_components['GCUSD_3M'] = MACRO_SCORE_WEIGHTS['GOLD']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['GOLD']['SCORE']
        else:
            score_components['GCUSD_3M'] = 0
    
    # 4. USDJPYの3ヶ月変化率
    if 'USDJPY' in df_merged.columns:
        usdjpy_3m_change = ((df_merged.iloc[current_idx]['USDJPY'] / df_merged.iloc[idx_3m]['USDJPY']) - 1) * 100
        
        if usdjpy_3m_change <= MACRO_SCORE_WEIGHTS['USDJPY']['THRESHOLD']:
            score_components['USDJPY_3M'] = MACRO_SCORE_WEIGHTS['USDJPY']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['USDJPY']['SCORE']
        else:
            score_components['USDJPY_3M'] = 0
    
    # 5. 10Y国債利回りの1ヶ月変化
    if 'DGS10' in df_merged.columns:
        dgs10_1m_change = df_merged.iloc[current_idx]['DGS10'] - df_merged.iloc[idx_1m]['DGS10']
        
        if dgs10_1m_change >= MACRO_SCORE_WEIGHTS['DGS10']['THRESHOLD']:
            score_components['DGS10_1M'] = MACRO_SCORE_WEIGHTS['DGS10']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['DGS10']['SCORE']
        else:
            score_components['DGS10_1M'] = 0
    
    # 6. イールドスプレッド
    if 'yield_difference' in df_merged.columns:
        yield_spread = df_merged.iloc[current_idx]['yield_difference']
        
        if yield_spread <= MACRO_SCORE_WEIGHTS['YIELD_SPREAD']['THRESHOLD']:
            score_components['YIELD_SPREAD'] = MACRO_SCORE_WEIGHTS['YIELD_SPREAD']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['YIELD_SPREAD']['SCORE']
        else:
            score_components['YIELD_SPREAD'] = 0
    
    # 7. CPI前年比の加速度
    if 'CPIAUCSL' in df_merged.columns and current_idx >= 252 and idx_1m >= 252:
        # 最新と前月のYoY変化率
        current_cpi = df_merged.iloc[current_idx]['CPIAUCSL']
        prev_month_cpi = df_merged.iloc[idx_1m]['CPIAUCSL']
        year_ago_cpi = df_merged.iloc[current_idx - 252]['CPIAUCSL']
        year_ago_prev_month_cpi = df_merged.iloc[idx_1m - 252]['CPIAUCSL']
        
        current_yoy = ((current_cpi / year_ago_cpi) - 1) * 100
        prev_month_yoy = ((prev_month_cpi / year_ago_prev_month_cpi) - 1) * 100
        
        cpi_acceleration = current_yoy - prev_month_yoy
        
        if cpi_acceleration >= MACRO_SCORE_WEIGHTS['CPI_YOY']['THRESHOLD']:
            score_components['CPI_YOY'] = MACRO_SCORE_WEIGHTS['CPI_YOY']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['CPI_YOY']['SCORE']
        else:
            score_components['CPI_YOY'] = 0
    
    # 8. DXYドル指数の変化率
    if 'TWEXBGSMTH' in df_merged.columns:
        dxy_change = ((df_merged.iloc[current_idx]['TWEXBGSMTH'] / df_merged.iloc[idx_1m]['TWEXBGSMTH']) - 1) * 100
        
        if dxy_change >= MACRO_SCORE_WEIGHTS['DXY']['THRESHOLD']:
            score_components['DXY'] = MACRO_SCORE_WEIGHTS['DXY']['SCORE']
            total_score += MACRO_SCORE_WEIGHTS['DXY']['SCORE']
        else:
            score_components['DXY'] = 0
    
    return total_score, score_components


def calculate_macro_score(df_merged: pd.DataFrame, as_of_date: str = None) -> Tuple[pd.DataFrame, Dict]:
    """
    マクロ経済状況に基づくスコアを計算します。
    
    Parameters
    ----------
    df_merged : pd.DataFrame
        マージされた経済指標データ
    as_of_date : str, optional
        スコア計算の基準日（指定なしの場合は最新日）
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (スコアを含むデータフレーム, 各指標の寄与度)
    """
    print(f"マクロスコア計算を開始: データサイズ {df_merged.shape}")
    
    # 結果を格納するデータフレーム
    df_score = df_merged.copy()
    
    # 基準日が指定されていない場合は最新日を使用
    if as_of_date is None:
        as_of_date = df_score['date'].max()
        print(f"基準日が指定されていないため、最新日 {as_of_date} を使用します")
    
    # 基準日のインデックス
    try:
        current_idx = df_score[df_score['date'] == as_of_date].index[0]
        print(f"基準日のインデックス: {current_idx}")
    except IndexError:
        print(f"エラー: 指定された基準日 {as_of_date} のデータがありません")
        # 最新の日付を使用
        as_of_date = df_score['date'].max()
        current_idx = df_score[df_score['date'] == as_of_date].index[0]
        print(f"代わりに最新日 {as_of_date} (インデックス: {current_idx}) を使用します")
    
    # マクロスコアカラムを初期化
    df_score['macro_score'] = 0.0
    
    # 直近日のスコアとコンポーネントを計算
    print(f"直近日 {as_of_date} のスコアを計算中...")
    current_score, score_components = calculate_single_day_macro_score(df_score, current_idx)
    df_score.loc[current_idx, 'macro_score'] = current_score
    print(f"直近日のスコア: {current_score}")
    print(f"内訳: {score_components}")
    
    # 過去のスコアを反復的に計算（5日ごと）
    # 無限再帰を避けるために、sample_indicesをリストとして事前に計算
    sample_indices = list(range(0, current_idx, 5))
    print(f"過去のスコア計算のためのサンプルポイント数: {len(sample_indices)}")
    
    # 現在のインデックスがサンプルに含まれていなければ追加
    if sample_indices and sample_indices[-1] != current_idx:
        sample_indices.append(current_idx)
    
    # 各サンプルインデックスに対して1回だけ処理を実行
    for idx in sample_indices:
        # すでに計算済みの場合はスキップ
        if idx == current_idx:
            continue
        
        # 各日付のスコアを個別に計算
        try:
            score, _ = calculate_single_day_macro_score(df_score, idx)
            df_score.loc[idx, 'macro_score'] = score
        except Exception as e:
            print(f"インデックス {idx} のスコア計算中にエラーが発生しました: {e}")
            # エラーの場合はスコアを0に設定（あるいは適切な処理）
            df_score.loc[idx, 'macro_score'] = 0
    
    # 欠損値を線形補間
    try:
        df_score['macro_score'] = df_score['macro_score'].interpolate(method='linear')
        print("線形補間によるスコア補完が完了しました")
    except Exception as e:
        print(f"線形補間中にエラーが発生しました: {e}")
        # 欠損値には0を設定するなどの代替処理
        df_score['macro_score'] = df_score['macro_score'].fillna(0)
    
    return df_score, score_components


def get_portfolio_symbols(engine: Engine) -> List[str]:
    """
    ポートフォリオに含まれる銘柄コードのリストを取得します（user_data.trade_journalから）。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
        
    Returns
    -------
    List[str]
        ポートフォリオに含まれる銘柄コードのリスト
    """
    try:
        from sqlalchemy import text
        
        # 保有中の銘柄を取得（sell_dateがNULLの銘柄）
        portfolio_query = text("""
            SELECT DISTINCT symbol 
            FROM user_data.trade_journal 
            WHERE sell_date IS NULL
            ORDER BY symbol
        """)
        
        with engine.connect() as conn:
            result = conn.execute(portfolio_query)
            symbols = [row[0] for row in result]
            
        if symbols:
            print(f"取引記録から保有銘柄を取得: {', '.join(symbols)}")
            return symbols
        else:
            print("取引記録に保有中の銘柄が見つかりませんでした。代替のサンプル銘柄を使用します。")
            return _get_default_symbols(engine)
            
    except Exception as e:
        print(f"取引記録からの銘柄取得に失敗: {e}")
        print("フォールバック: JSONファイルから取得を試行します。")
        return _get_portfolio_symbols_from_json(engine)


def _get_portfolio_symbols_from_json(engine: Engine) -> List[str]:
    """
    JSONファイルからポートフォリオ銘柄を取得（フォールバック用）
    """
    portfolio_path = Path(__file__).resolve().parent.parent.parent / "config" / "portfolio.json"
    
    try:
        if portfolio_path.exists():
            print(f"フォールバック: ポートフォリオファイルを読み込みます: {portfolio_path}")
            with open(portfolio_path, 'r', encoding='utf-8') as f:
                portfolio_data = json.load(f)
            
            # 現在保有中の銘柄のみを抽出
            active_symbols = []
            for item in portfolio_data:
                if item.get('sold', True) == False:  # soldがFalseの場合のみ
                    symbol = item.get('symbol')
                    if symbol and symbol not in active_symbols:
                        active_symbols.append(symbol)
            
            if active_symbols:
                print(f"JSONから保有銘柄を取得: {', '.join(active_symbols)}")
                return active_symbols
                
        print("JSONからの取得も失敗。代替のサンプル銘柄を使用します。")
        return _get_default_symbols(engine)
        
    except Exception as e:
        print(f"JSONファイルからの取得に失敗: {e}")
        return _get_default_symbols(engine)


def _get_default_symbols(engine: Engine) -> List[str]:
    """
    デフォルトのサンプル銘柄リストを取得します。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
        
    Returns
    -------
    List[str]
        サンプル銘柄コードのリスト
    """
    # 代替手段としてfmp_data.daily_pricesからユニークなシンボルを取得
    try:
        # 代表的な大型株をサンプルとして使用
        default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # これらのシンボルがデータベースに存在するか確認
        alternative_query = """
        SELECT DISTINCT symbol 
        FROM fmp_data.daily_prices 
        WHERE symbol IN %s
        AND date >= CURRENT_DATE - INTERVAL '30 days'
        """
        
        df = pd.read_sql(alternative_query, engine, params=(tuple(default_symbols),))
        
        if not df.empty:
            return df['symbol'].tolist()
        
        # それでも見つからない場合は、任意の銘柄を5つ取得
        backup_query = """
        SELECT DISTINCT symbol 
        FROM fmp_data.daily_prices 
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        LIMIT 5
        """
        
        df = pd.read_sql(backup_query, engine)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"サンプル銘柄の取得にも失敗しました: {e}")
        # 最終的なフォールバック
        return ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']


def get_portfolio_details_from_trade_journal(engine: Engine) -> pd.DataFrame:
    """
    trade_journalテーブルから保有銘柄の詳細情報を取得します。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
        
    Returns
    -------
    pd.DataFrame
        保有銘柄の詳細情報（購入時スコア含む）
    """
    try:
        from sqlalchemy import text
        
        portfolio_detail_query = text("""
            SELECT 
                symbol,
                buy_date,
                buy_price,
                qty,
                buy_reason_text,
                buy_rsi,
                buy_sma20,
                buy_sma40,
                buy_macd_hist,
                stop_loss_price,
                take_profit_price,
                total_score_at_buy,
                value_score_at_buy,
                momentum_score_at_buy,
                quality_score_at_buy,
                macro_sector_score_at_buy,
                growth_score_at_buy,
                per_score_at_buy,
                roic_score_at_buy,
                rsi_score_at_buy,
                macd_hist_score_at_buy,
                created_at,
                updated_at
            FROM user_data.trade_journal 
            WHERE sell_date IS NULL
            ORDER BY buy_date DESC
        """)
        
        with engine.connect() as conn:
            df_portfolio = pd.read_sql_query(portfolio_detail_query, conn)
            
        print(f"取引記録から詳細情報を取得: {len(df_portfolio)}銘柄")
        return df_portfolio
        
    except Exception as e:
        print(f"取引記録詳細情報取得エラー: {e}")
        return pd.DataFrame()


def calculate_combined_score(engine: Engine, df_merged: pd.DataFrame, as_of_date: str = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    マクロスコアとミクロスコアを組み合わせた総合評価を計算します。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
    df_merged : pd.DataFrame
        マージされた経済指標データ
    as_of_date : str, optional
        スコア計算の基準日（指定なしの場合は最新日）
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict, pd.DataFrame]
        (マクロスコアを含むデータフレーム, 各指標の寄与度, ミクロスコアを含むデータフレーム)
    """
    # マクロスコアを計算
    df_macro, macro_components = calculate_macro_score(df_merged, as_of_date)
    
    # マクロスコアコンポーネントの合計値を再確認（-1が2つあるのに合計が-1になる問題の修正）
    latest_date = df_macro['date'].max() if as_of_date is None else as_of_date
    total_macro_score = sum(macro_components.values())
    df_macro.loc[df_macro['date'] == latest_date, 'macro_score'] = total_macro_score
    print(f"マクロスコア再確認: {total_macro_score} (コンポーネント合計: {macro_components})")
    
    # ポートフォリオ銘柄を取得（従来の方法）
    symbols = get_portfolio_symbols(engine)
    
    # trade_journalから詳細情報も取得
    df_portfolio_details = get_portfolio_details_from_trade_journal(engine)
    
    # ミクロスコアを計算
    df_micro = calculate_micro_score(engine, symbols, as_of_date)
    
    # 既存のミクロスコアにtrade_journalの情報を統合
    if not df_portfolio_details.empty and not df_micro.empty:
        # 購入時スコア情報を追加
        df_micro = df_micro.merge(
            df_portfolio_details[['symbol', 'buy_date', 'buy_price', 'qty', 'total_score_at_buy', 
                                'value_score_at_buy', 'momentum_score_at_buy', 'quality_score_at_buy']], 
            on='symbol', 
            how='left'
        )
        print(f"取引記録とミクロスコアを統合: {len(df_micro)}銘柄")
    
    return df_macro, macro_components, df_micro


def get_chart_data(engine: Engine, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    特定の銘柄のチャートデータを取得します。
    
    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        データベース接続用エンジン
    symbol : str
        銘柄コード
    start_date : str
        チャートデータの開始日（YYYY-MM-DD形式）
    end_date : str
        チャートデータの終了日（YYYY-MM-DD形式）
        
    Returns
    -------
    pd.DataFrame
        チャートデータを含むデータフレーム
    """
    # チャートデータ取得のクエリ
    chart_query = """
    SELECT 
        ti.date, 
        ti.sma_20,
        ti.sma_40,
        dp.close,
        dp.volume,
        ti.atr_14
    FROM calculated_metrics.technical_indicators ti
    JOIN fmp_data.daily_prices dp 
        ON ti.symbol = dp.symbol 
        AND ti.date = dp.date
    WHERE ti.symbol = %s
    AND ti.date >= %s
    AND ti.date <= %s
    ORDER BY date
    """

    try:
        chart_df = pd.read_sql(chart_query, engine, params=(symbol, start_date, end_date))
        if not chart_df.empty:
            # データが取得できた場合の処理
            chart_df['date'] = pd.to_datetime(chart_df['date'])
            chart_df.set_index('date', inplace=True)
            
            # 出来高の計算
            chart_df['volume_ma5'] = chart_df['volume'].rolling(window=5).mean()
            chart_df['volume_change'] = (chart_df['volume'] / chart_df['volume_ma5'] - 1) * 100
            
            # 価格変化率の計算
            chart_df['price_change'] = chart_df['close'].pct_change() * 100
            
            return chart_df
        else:
            print(f"{symbol}のチャートデータが見つかりませんでした")
            return None
    except Exception as e:
        print(f"{symbol}のチャートデータ取得中にエラーが発生しました: {e}")
        return None

