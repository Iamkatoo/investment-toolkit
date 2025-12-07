import pandas as pd
import numpy as np

def score_of_policy_interest_rate(df):
    """
    政策金利のスコアリング関数
    FEDFUNDS（FRBの政策金利）を使用
    
    正の値: 金利が下降トレンド（経済にポジティブ）
    負の値: 金利が上昇トレンド（経済にネガティブ）
    """
    if 'FEDFUNDS' not in df.columns:
        raise ValueError("FEDFUNDS column not found in dataframe")
    
    # 前年同月との比較
    df['FEDFUNDS_12m_diff'] = df['FEDFUNDS'] - df['FEDFUNDS'].shift(12)
    
    # 3ヶ月前との比較
    df['FEDFUNDS_3m_diff'] = df['FEDFUNDS'] - df['FEDFUNDS'].shift(3)
    
    # スコアリング: 前年比と3ヶ月比の両方を考慮
    # 金利が下がっていると高スコア（好景気に関連）、上がっていると低スコア
    df['FEDFUNDS_score'] = -1 * (
        df['FEDFUNDS_12m_diff'] * 0.7 +  # 前年比に70%のウェイト
        df['FEDFUNDS_3m_diff'] * 0.3     # 3ヶ月比に30%のウェイト
    )
    
    return df

def score_of_yield_curve_spread(df):
    """
    イールドカーブスプレッドのスコアリング関数
    yield_difference（長期金利 - 短期金利）を使用
    
    正の値: イールドカーブがスティープ（経済にポジティブ）
    負の値: イールドカーブがフラット/インバート（経済にネガティブ）
    """
    if 'yield_difference' not in df.columns:
        raise ValueError("yield_difference column not found in dataframe")
    
    # スコアリング: スプレッドの絶対値と変化量の両方を考慮
    df['yield_diff_score'] = (
        # 現在のスプレッド（正ならポジティブ、負ならネガティブ）
        df['yield_difference'] * 3 +
        # 前年からの変化（上昇ならポジティブ、下降ならネガティブ）
        (df['yield_difference'] - df['yield_difference'].shift(12)) * 2
    )
    
    return df

def score_of_long_term_interest_rate(df):
    """
    長期金利のスコアリング関数
    DGS10（10年債イールド）を使用
    
    正の値: 金利が現実的なレベルかつ安定/下降
    負の値: 金利が過度に高いまたは急上昇
    """
    if 'DGS10' not in df.columns:
        raise ValueError("DGS10 column not found in dataframe")
    
    # 前年同月との変化
    df['DGS10_12m_diff'] = df['DGS10'] - df['DGS10'].shift(12)
    
    # 3ヶ月前との変化
    df['DGS10_3m_diff'] = df['DGS10'] - df['DGS10'].shift(3)
    
    # スコアリング: 
    # 1. 現在のレベル（極端に高いと悪い）
    # 2. 変化率（急上昇は悪い、緩やかな上昇や安定は良い）
    df['DGS10_score'] = (
        # 現在レベルのペナルティ（高金利はネガティブ）
        -1 * np.maximum(0, df['DGS10'] - 4) * 0.5 +
        # 前年比変化（上昇はネガティブ）
        -1 * df['DGS10_12m_diff'] * 2 +
        # 3ヶ月比変化（急上昇はネガティブ）
        -1 * df['DGS10_3m_diff'] * 1
    )
    
    return df

def score_of_corporate_bond_spread(df):
    """
    社債スプレッドのスコアリング関数
    BAA10Y（Moody's Baa社債 - 10年債）を使用
    
    正の値: スプレッドが縮小/安定（経済にポジティブ）
    負の値: スプレッドが拡大（経済にネガティブ）
    """
    if 'BAA10Y' not in df.columns:
        raise ValueError("BAA10Y column not found in dataframe")
    
    # スプレッドの変化（1年前比、3ヶ月前比）
    df['BAA10Y_12m_diff'] = df['BAA10Y'] - df['BAA10Y'].shift(12)
    df['BAA10Y_3m_diff'] = df['BAA10Y'] - df['BAA10Y'].shift(3)
    
    # スコアリング:
    # 1. 現在のスプレッド（広いとネガティブ）
    # 2. スプレッドの変化（拡大するとネガティブ）
    df['BAA10Y_score'] = (
        # 絶対的なスプレッドの大きさ（大きいほどネガティブ）
        -1 * df['BAA10Y'] * 1.5 +
        # 前年比変化（拡大はネガティブ）
        -1 * df['BAA10Y_12m_diff'] * 2 +
        # 3ヶ月比変化（急拡大はネガティブ）
        -1 * df['BAA10Y_3m_diff'] * 1
    )
    
    return df

def score_of_us_doller_index(df):
    """
    米ドル指数のスコアリング関数
    TWEXBGSMTH（貿易加重米ドル指数）を使用
    
    正の値: ドル指数が安定または適度に弱い（グローバル経済にポジティブ）
    負の値: ドル指数が急激に上昇（グローバル経済にネガティブ）
    """
    if 'TWEXBGSMTH' not in df.columns:
        raise ValueError("TWEXBGSMTH column not found in dataframe")
    
    # 変化率算出
    df['TWEXBGSMTH_12m_pct'] = df['TWEXBGSMTH'] / df['TWEXBGSMTH'].shift(12) - 1
    df['TWEXBGSMTH_3m_pct'] = df['TWEXBGSMTH'] / df['TWEXBGSMTH'].shift(3) - 1
    
    # スコアリング:
    # 1. 年間変化率（急上昇はネガティブ、緩やかな上昇または下降は中立的）
    # 2. 3ヶ月変化率（急上昇はより顕著にネガティブ）
    df['TWEXBGSMTH_score'] = (
        # 前年比変化（大幅上昇はネガティブ）
        -3 * np.where(df['TWEXBGSMTH_12m_pct'] > 0.05, df['TWEXBGSMTH_12m_pct'], 0) +
        # 前年比変化（適度な下落はポジティブ）
        1 * np.where(df['TWEXBGSMTH_12m_pct'] < 0, -df['TWEXBGSMTH_12m_pct'], 0) +
        # 3ヶ月比変化（急上昇はネガティブ）
        -4 * np.where(df['TWEXBGSMTH_3m_pct'] > 0.03, df['TWEXBGSMTH_3m_pct'], 0)
    )
    
    return df

def evaluate_economic_indicators(df):
    """
    5つの主要経済指標を統合し、総合スコアを算出する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        FEDFUNDS, yield_difference, DGS10, BAA10Y, TWEXBGSMTH カラムを含むデータフレーム
        
    Returns:
    --------
    pandas.DataFrame
        各指標のスコアと総合評価を追加したデータフレーム
    """
    required_columns = ['FEDFUNDS', 'yield_difference', 'DGS10', 'BAA10Y', 'TWEXBGSMTH']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 各スコア関数を適用
    df = score_of_policy_interest_rate(df)
    df = score_of_yield_curve_spread(df)
    df = score_of_long_term_interest_rate(df)
    df = score_of_corporate_bond_spread(df)
    df = score_of_us_doller_index(df)
    
    # 総合評価スコアの計算
    weights = {
        'FEDFUNDS_score': 0.25,
        'yield_diff_score': 0.25,
        'DGS10_score': 0.2,
        'BAA10Y_score': 0.15,
        'TWEXBGSMTH_score': 0.15
    }
    
    df['evaluation'] = sum(df[score] * weight for score, weight in weights.items())
    
    return df 

# ==============================================================
#   12 M カナリア式スコア  ― オリジナルの計算式に合わせた実装
# ==============================================================

def evaluate_economic_indicators_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    書籍『金利は経済のカナリア』で紹介されたロジックに準拠。
    すべて –2 / 0 / +2 の 3 段階スコアを付与し、単純平均を総合スコアとする。
    """

    df = df.copy().reset_index(drop=True)

    # --------------------------------------------------
    # 個別スコア関数
    # --------------------------------------------------
    def _score_policy(row):
        idx = row.name
        if idx < 13:               # 月次 12 + 当月
            return None
        diff = row['FEDFUNDS'] - df['FEDFUNDS'].iloc[idx - 13]
        return 2 if diff <= 0.25 else -2

    def _score_yield_curve(row):
        v = row['yield_difference']      # ＝10Y-3M
        if v >= 1:
            return 2
        elif v >= 0:
            return 0
        else:
            return -2

    def _score_long_rate(row):
        idx = row.name
        if idx < 13:
            return None
        diff = row['DGS10'] - df['DGS10'].iloc[idx - 13]
        return 2 if diff >= 0 else -2

    def _score_corp_spread(row):
        idx = row.name
        if idx < 260:              # ≒1 年前が無い行
            return None
        one_year_before = row['date'] - pd.DateOffset(years=1)
        closest = (df['date'] - one_year_before).abs().idxmin()
        if closest == idx:
            return None
        diff = row['BAA10Y'] - df['BAA10Y'].iloc[closest]
        return 2 if diff <= 0 else -2

    def _score_dollar(row):
        idx = row.name
        if idx < 13:
            return None
        ratio = row['TWEXBGSMTH'] / df['TWEXBGSMTH'].iloc[idx - 13]
        return 2 if ratio <= 1 else -2

    # --------------------------------------------------
    # スコア列を作成
    # --------------------------------------------------
    df['score_policy']   = df.apply(_score_policy,   axis=1)
    df['score_ycspread'] = df.apply(_score_yield_curve, axis=1)
    df['score_long']     = df.apply(_score_long_rate, axis=1)
    df['score_corp']     = df.apply(_score_corp_spread, axis=1)
    df['score_dxy']      = df.apply(_score_dollar,    axis=1)

      # --------------------------------------------------
    # 総合スコア  (単純 **合計**, -10 〜 +10)
    # --------------------------------------------------
    df['evaluation'] = (
        df[['score_policy', 'score_ycspread', 'score_long',
            'score_corp', 'score_dxy']]
        .sum(axis=1, skipna=True)
    )

    return df[['date', 'evaluation']]