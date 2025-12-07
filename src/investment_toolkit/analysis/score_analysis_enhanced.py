#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
スコア分析機能強化版（フィルタリング対応）

新機能：
- フィルタリング状態を考慮したスコア分析
- フィルタリング前後の比較機能
- より詳細な統計情報
"""

import pandas as pd
import logging
from sqlalchemy import text, Engine
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_top_scored_stocks_filtered(engine: Engine, target_date: str = None, top_n: int = 10, 
                                  apply_filters: bool = True) -> pd.DataFrame:
    """
    フィルタリング対応版：指定日のスコア上位銘柄を取得
    
    Args:
        engine: データベースエンジン
        target_date: 対象日（Noneの場合は最新日）
        top_n: 上位何銘柄を取得するか
        apply_filters: フィルタリングを適用するかどうか
        
    Returns:
        上位銘柄のスコアデータ
    """
    if target_date is None:
        # 最新日を取得
        date_query = text("SELECT MAX(date) as max_date FROM backtest_results.daily_scores")
        with engine.connect() as conn:
            result = conn.execute(date_query).fetchone()
            target_date = result.max_date.strftime('%Y-%m-%d')
    
    # フィルタリング条件
    filter_condition = ""
    if apply_filters:
        filter_condition = """
        AND is_value_trap_filtered = FALSE 
        AND is_quality_growth_filtered = FALSE
        """
    
    # 上位銘柄のスコアを取得
    query = text(f"""
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
        sector_rotation_score,
        is_value_trap_filtered,
        is_quality_growth_filtered
    FROM backtest_results.daily_scores
    WHERE date = :target_date
    {filter_condition}
    ORDER BY total_score DESC
    LIMIT :top_n
    """)
    
    try:
        df = pd.read_sql(query, engine, params={"target_date": target_date, "top_n": top_n})
        
        filter_status = "フィルタリング適用" if apply_filters else "全銘柄対象"
        logger.info(f"スコア上位銘柄取得完了: {len(df)}件 (日付: {target_date}, {filter_status})")
        
        return df
    except Exception as e:
        logger.error(f"スコアデータ取得エラー: {e}")
        return pd.DataFrame()


def get_filtering_statistics(engine: Engine, target_date: str = None) -> Dict[str, Any]:
    """
    フィルタリング統計情報を取得
    
    Args:
        engine: データベースエンジン
        target_date: 対象日（Noneの場合は最新日）
        
    Returns:
        フィルタリング統計情報
    """
    if target_date is None:
        # 最新日を取得
        date_query = text("SELECT MAX(date) as max_date FROM backtest_results.daily_scores")
        with engine.connect() as conn:
            result = conn.execute(date_query).fetchone()
            target_date = result.max_date.strftime('%Y-%m-%d')
    
    query = text("""
    SELECT 
        COUNT(*) as total_stocks,
        COUNT(CASE WHEN is_value_trap_filtered = FALSE THEN 1 END) as value_trap_passed,
        COUNT(CASE WHEN is_quality_growth_filtered = FALSE THEN 1 END) as quality_growth_passed,
        COUNT(CASE WHEN is_value_trap_filtered = FALSE AND is_quality_growth_filtered = FALSE THEN 1 END) as both_passed,
        AVG(total_score) as avg_total_score,
        AVG(CASE WHEN is_value_trap_filtered = FALSE AND is_quality_growth_filtered = FALSE THEN total_score END) as avg_filtered_score,
        MAX(total_score) as max_total_score,
        MIN(total_score) as min_total_score
    FROM backtest_results.daily_scores
    WHERE date = :target_date
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"target_date": target_date}).fetchone()
        
        stats = {
            'target_date': target_date,
            'total_stocks': result.total_stocks,
            'value_trap_passed': result.value_trap_passed,
            'quality_growth_passed': result.quality_growth_passed,
            'both_filters_passed': result.both_passed,
            'avg_total_score': result.avg_total_score,
            'avg_filtered_score': result.avg_filtered_score,
            'max_total_score': result.max_total_score,
            'min_total_score': result.min_total_score,
        }
        
        # パーセンテージ計算
        if stats['total_stocks'] > 0:
            stats['value_trap_pass_rate'] = (stats['value_trap_passed'] / stats['total_stocks']) * 100
            stats['quality_growth_pass_rate'] = (stats['quality_growth_passed'] / stats['total_stocks']) * 100
            stats['both_filters_pass_rate'] = (stats['both_filters_passed'] / stats['total_stocks']) * 100
        else:
            stats['value_trap_pass_rate'] = 0
            stats['quality_growth_pass_rate'] = 0
            stats['both_filters_pass_rate'] = 0
        
        logger.info(f"フィルタリング統計取得完了 (日付: {target_date})")
        
        return stats
        
    except Exception as e:
        logger.error(f"フィルタリング統計取得エラー: {e}")
        return {}


def compare_filtered_vs_unfiltered(engine: Engine, target_date: str = None, top_n: int = 20) -> Dict[str, pd.DataFrame]:
    """
    フィルタリング前後の上位銘柄を比較
    
    Args:
        engine: データベースエンジン
        target_date: 対象日
        top_n: 比較する上位銘柄数
        
    Returns:
        フィルタリング前後の比較データ
    """
    # フィルタリング適用版
    filtered_df = get_top_scored_stocks_filtered(engine, target_date, top_n, apply_filters=True)
    
    # フィルタリング非適用版
    unfiltered_df = get_top_scored_stocks_filtered(engine, target_date, top_n, apply_filters=False)
    
    # 統計情報
    stats = get_filtering_statistics(engine, target_date)
    
    return {
        'filtered': filtered_df,
        'unfiltered': unfiltered_df,
        'statistics': stats
    }


def print_filtering_summary(stats: Dict[str, Any]) -> None:
    """
    フィルタリング統計情報をコンソールに出力
    
    Args:
        stats: フィルタリング統計情報
    """
    print(f"\n=== フィルタリング統計情報 ({stats['target_date']}) ===")
    print(f"総銘柄数: {stats['total_stocks']:,}")
    print(f"バリュートラップフィルター通過: {stats['value_trap_passed']:,} ({stats['value_trap_pass_rate']:.1f}%)")
    print(f"Quality/Growthフィルター通過: {stats['quality_growth_passed']:,} ({stats['quality_growth_pass_rate']:.1f}%)")
    print(f"両フィルター通過: {stats['both_filters_passed']:,} ({stats['both_filters_pass_rate']:.1f}%)")
    print(f"平均スコア（全体）: {stats['avg_total_score']:.2f}")
    print(f"平均スコア（フィルター後）: {stats['avg_filtered_score']:.2f}")
    print(f"スコア範囲: {stats['min_total_score']:.2f} ～ {stats['max_total_score']:.2f}")


def get_specific_stock_score(engine: Engine, symbol: str, target_date: str = None) -> Dict[str, Any]:
    """
    特定銘柄のスコア詳細を取得（フィルタリング状態含む）
    
    Args:
        engine: データベースエンジン
        symbol: 銘柄シンボル
        target_date: 対象日（Noneの場合は最新日）
        
    Returns:
        銘柄の詳細スコア情報
    """
    if target_date is None:
        # 最新日を取得
        date_query = text("SELECT MAX(date) as max_date FROM backtest_results.daily_scores WHERE symbol = :symbol")
        with engine.connect() as conn:
            result = conn.execute(date_query, {"symbol": symbol}).fetchone()
            if result and result.max_date:
                target_date = result.max_date.strftime('%Y-%m-%d')
            else:
                logger.warning(f"銘柄 {symbol} のデータが見つかりません")
                return {}
    
    query = text("""
    SELECT *
    FROM backtest_results.daily_scores
    WHERE symbol = :symbol AND date = :target_date
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"symbol": symbol, "target_date": target_date}).fetchone()
        
        if result:
            stock_data = dict(result)
            
            # フィルタリング状態の解釈
            stock_data['filter_status'] = "両フィルター通過"
            if stock_data['is_value_trap_filtered']:
                stock_data['filter_status'] = "バリュートラップフィルターで除外"
            elif stock_data['is_quality_growth_filtered']:
                stock_data['filter_status'] = "Quality/Growthフィルターで除外"
            
            logger.info(f"銘柄 {symbol} のスコア取得完了 (日付: {target_date})")
            return stock_data
        else:
            logger.warning(f"銘柄 {symbol} の日付 {target_date} のデータが見つかりません")
            return {}
            
    except Exception as e:
        logger.error(f"銘柄 {symbol} のスコア取得エラー: {e}")
        return {}


if __name__ == "__main__":
    # テスト用コード
    from investment_toolkit.database.db_manager import DatabaseManager
    
    # データベース接続
    db_manager = DatabaseManager()
    engine = db_manager.get_engine()
    
    try:
        # フィルタリング統計
        stats = get_filtering_statistics(engine)
        print_filtering_summary(stats)
        
        # 上位銘柄比較
        comparison = compare_filtered_vs_unfiltered(engine, top_n=10)
        
        print(f"\n=== フィルタリング適用後の上位10銘柄 ===")
        if not comparison['filtered'].empty:
            print(comparison['filtered'][['symbol', 'total_score', 'is_value_trap_filtered', 'is_quality_growth_filtered']].to_string(index=False))
        
        print(f"\n=== 全銘柄対象の上位10銘柄 ===")
        if not comparison['unfiltered'].empty:
            print(comparison['unfiltered'][['symbol', 'total_score', 'is_value_trap_filtered', 'is_quality_growth_filtered']].to_string(index=False))
        
        # 特定銘柄のスコア（COKE）
        coke_score = get_specific_stock_score(engine, "COKE")
        if coke_score:
            print(f"\n=== COKE銘柄のスコア詳細 ===")
            print(f"日付: {coke_score['date']}")
            print(f"総合スコア: {coke_score['total_score']:.2f}")
            print(f"フィルター状態: {coke_score['filter_status']}")
            print(f"Value: {coke_score['value_score']:.2f}, Growth: {coke_score['growth_score']:.2f}")
            print(f"Quality: {coke_score['quality_score']:.2f}, Momentum: {coke_score['momentum_score']:.2f}")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        
    finally:
        db_manager.close()
