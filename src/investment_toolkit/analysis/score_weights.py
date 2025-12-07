"""
スコアリングの重みと定数値を定義します。
"""

# マイクロスコア（個別銘柄評価）用の定数
MICRO_SCORE_WEIGHTS = {
    # 前日比変化率
    "PRICE_CHANGE": {
        "THRESHOLD_POSITIVE": 1.0,  # +1%以上
        "THRESHOLD_NEGATIVE": -1.0,  # -1%以上下落
        "SCORE_POSITIVE": 2,  # +1%以上で+2点
        "SCORE_NEGATIVE": -2,  # -1%以上下落で-2点
    },
    
    # セクター/インダストリー平均との乖離率
    "SECTOR_DEVIATION": {
        "THRESHOLD": 2.0,  # 2%以上の乖離
        "SCORE": -1,  # ネガティブ側に乖離で-1点
    },
    
    "INDUSTRY_DEVIATION": {
        "THRESHOLD": 2.0,  # 2%以上の乖離
        "SCORE": -1,  # ネガティブ側に乖離で-1点
    },
    
    # 出来高変化率（5日移動平均比）
    "VOLUME_CHANGE": {
        "THRESHOLD_VERY_HIGH": 100.0,  # +100%以上
        "THRESHOLD_HIGH": 30.0,  # +30%以上j
        "THRESHOLD_LOW": -30.0,  # -30%以下
        "THRESHOLD_VERY_LOW": -100.0,  # -100%以下
        "SCORE_VERY_HIGH": 2,  # +100%以上で+2点
        "SCORE_HIGH": 1,  # +30%以上で+1点
        "SCORE_LOW": -1,  # -30%以下で-1点
        "SCORE_VERY_LOW": -2,  # -100%以下で-2点
    },
    
    # ゴールデンクロス/デッドクロス判定
    "GC_DC": {
        "SCORE_GOLDEN_CROSS": 1,  # ゴールデンクロスで+1点
        "SCORE_DEAD_CROSS": -1,  # デッドクロスで-1点
    },
    
    # ATR比（異常値動き判定）
    "ATR_RATIO": {
        "THRESHOLD_VERY_HIGH": 2.0,  # ATR比2.0以上
        "THRESHOLD_HIGH": 1.3,  # ATR比1.3以上
        "THRESHOLD_LOW": 0.7,  # ATR比0.7未満
        "SCORE_VERY_HIGH_UP": 2,  # ATR比2.0以上（上昇時）で+2点
        "SCORE_HIGH_UP": 1,  # ATR比1.3以上（上昇時）で+1点
        "SCORE_VERY_HIGH_DOWN": -2,  # ATR比2.0以上（下落時）で-2点
        "SCORE_HIGH_DOWN": -1,  # ATR比1.3以上（下落時）で-1点
        "SCORE_LOW": -1,  # ATR比0.7未満で-1点（小動き）
    },
}

# マクロスコア（市場全体評価）用の定数
MACRO_SCORE_WEIGHTS = {
    "SP500": {
        "PERIOD": "3M",  # 3ヶ月変化率
        "THRESHOLD": -3.0,  # -3%以上下落
        "SCORE": -2,  # -3%以上下落で-2点
    },
    
    "VIX": {
        "PERIOD": "1W",  # 1週間変化率
        "THRESHOLD": 10.0,  # +10%以上上昇
        "SCORE": -2,  # +10%以上上昇で-2点
    },
    
    "GOLD": {
        "PERIOD": "3M",  # 3ヶ月変化率
        "THRESHOLD": 5.0,  # +5%以上上昇
        "SCORE": -1,  # +5%以上上昇で-1点
    },
    
    "USDJPY": {
        "PERIOD": "3M",  # 3ヶ月変化率
        "THRESHOLD": -2.0,  # -2%以上の円高
        "SCORE": -1,  # -2%以上の円高で-1点
    },
    
    "DGS10": {  # 10Y国債
        "PERIOD": "1M",  # 1ヶ月変化率
        "THRESHOLD": 0.5,  # +0.5%以上上昇
        "SCORE": -1,  # +0.5%以上上昇で-1点
    },
    
    "YIELD_SPREAD": {  # イールドスプレッド
        "THRESHOLD": -0.5,  # -0.5%以上逆転
        "SCORE": -2,  # -0.5%以上逆転で-2点
    },
    
    "CPI_YOY": {  # CPI前年比
        "THRESHOLD": 0.4,  # 前月比+0.4pt以上加速
        "SCORE": -2,  # 前月比+0.4pt以上加速で-2点
    },
    
    "DXY": {  # ドル指数
        "THRESHOLD": 2.0,  # +2%以上上昇
        "SCORE": -1,  # +2%以上上昇で-1点
    },
} 