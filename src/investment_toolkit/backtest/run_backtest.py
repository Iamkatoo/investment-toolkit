#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
オリジナル総合スコア・バックテストモジュール

Total≥80点の銘柄をポートフォリオ候補として、20営業日後のリターンを計算。
ベンチマーク（S&P500）との比較を行い、結果をデータベースに保存。
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sqlalchemy import create_engine, text

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from investment_analysis.database.db_manager import DatabaseManager

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """バックテストエンジンクラス"""
    
    def __init__(self, config_path: str = "config/score_weights.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_db_engine()
        self.benchmark_symbol = "^GSPC"  # S&P500
        self.holding_period = 20  # 20営業日
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def _create_db_engine(self) -> Any:
        """データベース接続エンジンを作成"""
        db_name = os.environ.get("DB_NAME", "investment")
        db_user = os.environ.get("DB_USER", "postgres") 
        db_password = os.environ.get("DB_PASSWORD", "postgres")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return create_engine(connection_string)
    
    def load_daily_scores(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
        """
        日次スコアデータを読み込み
        
        Args:
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            日次スコアのDataFrame
        """
        logger.info("日次スコアデータを読み込み中...")
        
        query = """
        SELECT * FROM backtest_results.daily_scores
        WHERE date >= %(start_date)s
        """
        params = {'start_date': start_date}
        
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
            
        query += " ORDER BY date, symbol"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(f"スコアデータ読み込み完了: {len(df):,}行")
            return df
        except Exception as e:
            logger.error(f"スコアデータ読み込みエラー: {e}")
            raise
    
    def load_price_data(self, start_date: str = "2015-01-01", end_date: Optional[str] = None) -> pd.DataFrame:
        """
        価格データを読み込み（リターン計算用）
        
        Args:
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            価格データのDataFrame
        """
        logger.info("価格データを読み込み中...")
        
        query = """
        SELECT symbol, date, close, adj_close
        FROM fmp_data.daily_prices
        WHERE date >= %(start_date)s
        """
        params = {'start_date': start_date}
        
        if end_date:
            query += " AND date <= %(end_date)s"
            params['end_date'] = end_date
            
        query += " ORDER BY symbol, date"
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(f"価格データ読み込み完了: {len(df):,}行")
            return df
        except Exception as e:
            logger.error(f"価格データ読み込みエラー: {e}")
            raise
    
    def get_eligible_stocks(self, scores_df: pd.DataFrame, target_date: str) -> List[Dict[str, Any]]:
        """
        指定日の買い候補銘柄を取得
        
        Args:
            scores_df: スコアデータ
            target_date: 対象日（文字列形式 'YYYY-MM-DD'）
            
        Returns:
            買い候補銘柄のリスト
        """
        thresholds = self.config['buy_thresholds']
        
        # 文字列の日付をdateオブジェクトに変換
        from datetime import datetime
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        # その日のスコアをフィルタ（dateオブジェクト同士で比較）
        daily_scores = scores_df[scores_df['date'] == target_date_obj].copy()
        
        logger.debug(f"日付 {target_date}: スコアデータ件数 = {len(daily_scores)}")
        
        if len(daily_scores) == 0:
            return []
        
        # 各条件の該当件数を確認
        total_cond = daily_scores['total_score'] >= thresholds['total_score']
        value_cond = daily_scores['value_score'] >= thresholds['minimum_scores']['value']
        growth_cond = daily_scores['growth_score'] >= thresholds['minimum_scores']['growth']
        quality_cond = daily_scores['quality_score'] >= thresholds['minimum_scores']['quality']
        momentum_cond = daily_scores['momentum_score'] >= thresholds['minimum_scores']['momentum']
        
        logger.debug(f"  総合≥{thresholds['total_score']}: {total_cond.sum()}, Value≥{thresholds['minimum_scores']['value']}: {value_cond.sum()}, Growth≥{thresholds['minimum_scores']['growth']}: {growth_cond.sum()}, Quality≥{thresholds['minimum_scores']['quality']}: {quality_cond.sum()}, Momentum≥{thresholds['minimum_scores']['momentum']}: {momentum_cond.sum()}")
        
        # 買い候補条件をチェック
        eligible = daily_scores[
            (daily_scores['total_score'] >= thresholds['total_score']) &
            (daily_scores['value_score'] >= thresholds['minimum_scores']['value']) &
            (daily_scores['growth_score'] >= thresholds['minimum_scores']['growth']) &
            (daily_scores['quality_score'] >= thresholds['minimum_scores']['quality']) &
            (daily_scores['momentum_score'] >= thresholds['minimum_scores']['momentum'])
        ].copy()
        
        logger.debug(f"  全条件満たす銘柄: {len(eligible)}")
        
        # 総合スコア順でソート
        eligible = eligible.sort_values('total_score', ascending=False)
        
        return eligible.to_dict('records')
    
    def calculate_future_returns(self, price_df: pd.DataFrame, symbol: str, entry_date: str, holding_days: int = 20) -> Optional[Dict[str, Any]]:
        """
        指定銘柄の将来リターンを計算
        
        Args:
            price_df: 価格データ
            symbol: 銘柄コード
            entry_date: エントリー日（文字列形式 'YYYY-MM-DD'）
            holding_days: 保有日数
            
        Returns:
            リターン情報の辞書
        """
        # 文字列の日付をdateオブジェクトに変換
        from datetime import datetime
        entry_date_obj = datetime.strptime(entry_date, '%Y-%m-%d').date()
        
        # 該当銘柄の価格データを取得
        symbol_prices = price_df[price_df['symbol'] == symbol].copy()
        symbol_prices = symbol_prices.sort_values('date')
        
        # エントリー日のデータを取得（dateオブジェクト同士で比較）
        entry_data = symbol_prices[symbol_prices['date'] == entry_date_obj]
        if len(entry_data) == 0:
            return None
        
        # adj_closeがNoneの場合はcloseを使用
        entry_price = entry_data.iloc[0]['adj_close']
        if entry_price is None or pd.isna(entry_price):
            entry_price = entry_data.iloc[0]['close']
        
        entry_idx = symbol_prices[symbol_prices['date'] == entry_date_obj].index[0]
        
        # 保有期間後の日付を探す
        future_prices = symbol_prices[symbol_prices['date'] > entry_date_obj].head(holding_days)
        
        if len(future_prices) == 0:
            return None
        
        # 最後の営業日の価格を取得
        exit_data = future_prices.iloc[-1]
        exit_date = exit_data['date']
        exit_price = exit_data['adj_close']
        if exit_price is None or pd.isna(exit_price):
            exit_price = exit_data['close']
        
        # リターン計算
        return_pct = (exit_price - entry_price) / entry_price * 100
        
        return {
            'entry_date': entry_date_obj,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'holding_period_days': len(future_prices),
            'return_pct': return_pct
        }
    
    def calculate_benchmark_return(self, price_df: pd.DataFrame, entry_date, exit_date) -> float:
        """
        ベンチマーク（S&P500）のリターンを計算
        
        Args:
            price_df: 価格データ
            entry_date: エントリー日（dateオブジェクト）
            exit_date: エグジット日（dateオブジェクト）
            
        Returns:
            ベンチマークリターン（%）
        """
        # S&P500の価格データを取得
        sp500_prices = price_df[price_df['symbol'] == self.benchmark_symbol].copy()
        
        # dateオブジェクト同士で比較
        entry_data = sp500_prices[sp500_prices['date'] == entry_date]
        exit_data = sp500_prices[sp500_prices['date'] == exit_date]
        
        if len(entry_data) == 0 or len(exit_data) == 0:
            return 0.0
        
        # adj_closeがNoneの場合はcloseを使用
        entry_price = entry_data.iloc[0]['adj_close']
        if entry_price is None or pd.isna(entry_price):
            entry_price = entry_data.iloc[0]['close']
            
        exit_price = exit_data.iloc[0]['adj_close']
        if exit_price is None or pd.isna(exit_price):
            exit_price = exit_data.iloc[0]['close']
        
        return (exit_price - entry_price) / entry_price * 100
    
    def run_backtest(self, start_date: str = "2015-01-01", end_date: Optional[str] = None, max_positions: int = 50) -> pd.DataFrame:
        """
        バックテストのメイン処理
        
        Args:
            start_date: 開始日
            end_date: 終了日
            max_positions: 最大ポジション数
            
        Returns:
            バックテスト結果のDataFrame
        """
        logger.info("バックテスト開始...")
        
        # データ読み込み
        scores_df = self.load_daily_scores(start_date, end_date)
        price_df = self.load_price_data(start_date, end_date)
        
        # 営業日リストを取得
        trading_dates = sorted(scores_df['date'].unique())
        
        results = []
        
        logger.info(f"バックテスト実行: {len(trading_dates)}営業日")
        
        for i, trade_date in enumerate(tqdm(trading_dates, desc="バックテスト実行中")):
            trade_date_str = trade_date if isinstance(trade_date, str) else trade_date.strftime('%Y-%m-%d')
            
            # 最後の20営業日はスキップ（将来リターンが計算できないため）
            if i >= len(trading_dates) - self.holding_period:
                continue
            
            # 最初の10日は詳細ログ
            if i < 10:
                logger.info(f"処理中 {i+1}/10: {trade_date_str}")
            
            # その日の買い候補銘柄を取得
            eligible_stocks = self.get_eligible_stocks(scores_df, trade_date_str)
            
            if len(eligible_stocks) == 0:
                if i < 10:
                    logger.info(f"  条件を満たす銘柄なし")
                continue
            
            if i < 10:
                logger.info(f"  条件を満たす銘柄: {len(eligible_stocks)}件")
            
            # 上位N銘柄を選択（等金額投資）
            selected_stocks = eligible_stocks[:max_positions]
            
            for stock in selected_stocks:
                symbol = stock['symbol']
                total_score = stock['total_score']
                
                # 将来リターンを計算
                return_info = self.calculate_future_returns(
                    price_df, symbol, trade_date_str, self.holding_period
                )
                
                if return_info is None:
                    continue
                
                # ベンチマークリターンを計算
                benchmark_return = self.calculate_benchmark_return(
                    price_df, return_info['entry_date'], return_info['exit_date']
                )
                
                # 超過リターンを計算
                excess_return = return_info['return_pct'] - benchmark_return
                
                # 結果を記録
                result = {
                    'backtest_date': trade_date_str,
                    'symbol': symbol,
                    'entry_total_score': total_score,
                    'entry_price': return_info['entry_price'],
                    'exit_price': return_info['exit_price'],
                    'entry_date': return_info['entry_date'],
                    'exit_date': return_info['exit_date'],
                    'holding_period_days': return_info['holding_period_days'],
                    'return_pct': return_info['return_pct'],
                    'benchmark_return_pct': benchmark_return,
                    'excess_return_pct': excess_return,
                    'position_size': 100000 / len(selected_stocks)  # 等金額（仮想$100,000を等分）
                }
                
                results.append(result)
        
        logger.info(f"バックテスト完了: {len(results)}取引")
        return pd.DataFrame(results)
    
    def save_backtest_results(self, results_df: pd.DataFrame) -> None:
        """
        バックテスト結果をデータベースに保存
        
        Args:
            results_df: バックテスト結果データフレーム
        """
        if len(results_df) == 0:
            logger.warning("保存するバックテスト結果がありません")
            return
        
        logger.info(f"バックテスト結果を保存中: {len(results_df)}行")
        
        try:
            # 既存データを削除（再計算の場合）
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM backtest_results.bt_summary"))
                conn.commit()
            
            # 新しいデータを保存
            results_df.to_sql(
                'bt_summary',
                self.engine,
                schema='backtest_results',
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info("バックテスト結果保存完了")
            
        except Exception as e:
            logger.error(f"バックテスト結果保存エラー: {e}")
            raise
    
    def calculate_portfolio_performance(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        ポートフォリオパフォーマンスを計算
        
        Args:
            results_df: バックテスト結果
            
        Returns:
            日次ポートフォリオパフォーマンス
        """
        logger.info("ポートフォリオパフォーマンスを計算中...")
        
        # 日別のリターンを集計
        daily_returns = results_df.groupby('backtest_date').agg({
            'return_pct': 'mean',  # 等金額投資の平均リターン
            'benchmark_return_pct': 'mean',
            'excess_return_pct': 'mean',
            'symbol': 'count'  # ポジション数
        }).reset_index()
        
        daily_returns.columns = ['date', 'daily_return_pct', 'benchmark_daily_return_pct', 'excess_return_pct', 'positions_count']
        
        # 累積リターンを計算
        daily_returns['cumulative_return_pct'] = (1 + daily_returns['daily_return_pct'] / 100).cumprod() - 1
        daily_returns['benchmark_cumulative_return_pct'] = (1 + daily_returns['benchmark_daily_return_pct'] / 100).cumprod() - 1
        
        # ポートフォリオ価値を計算（初期値$100,000）
        initial_value = 100000
        daily_returns['portfolio_value'] = initial_value * (1 + daily_returns['cumulative_return_pct'])
        daily_returns['benchmark_value'] = initial_value * (1 + daily_returns['benchmark_cumulative_return_pct'])
        
        # ドローダウンを計算
        rolling_max = daily_returns['portfolio_value'].expanding().max()
        daily_returns['drawdown_pct'] = (daily_returns['portfolio_value'] - rolling_max) / rolling_max * 100
        
        return daily_returns
    
    def save_portfolio_performance(self, portfolio_df: pd.DataFrame) -> None:
        """
        ポートフォリオパフォーマンスをデータベースに保存
        
        Args:
            portfolio_df: ポートフォリオパフォーマンスデータフレーム
        """
        if len(portfolio_df) == 0:
            logger.warning("保存するポートフォリオパフォーマンスデータがありません")
            return
        
        logger.info(f"ポートフォリオパフォーマンスを保存中: {len(portfolio_df)}行")
        
        try:
            # 既存データを削除
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM backtest_results.portfolio_performance"))
                conn.commit()
            
            # 新しいデータを保存
            portfolio_df.to_sql(
                'portfolio_performance',
                self.engine,
                schema='backtest_results',
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info("ポートフォリオパフォーマンス保存完了")
            
        except Exception as e:
            logger.error(f"ポートフォリオパフォーマンス保存エラー: {e}")
            raise


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="オリジナル総合スコア・バックテスト")
    parser.add_argument("--start-date", default="2015-01-01", help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="終了日 (YYYY-MM-DD)")
    parser.add_argument("--max-positions", type=int, default=50, help="最大ポジション数")
    parser.add_argument("--config", default="config/score_weights.yaml", help="設定ファイルパス")
    
    args = parser.parse_args()
    
    try:
        # バックテストエンジンを初期化
        engine = BacktestEngine(args.config)
        
        # バックテスト実行
        results_df = engine.run_backtest(args.start_date, args.end_date, args.max_positions)
        
        # 結果を保存
        engine.save_backtest_results(results_df)
        
        # ポートフォリオパフォーマンスを計算・保存
        portfolio_df = engine.calculate_portfolio_performance(results_df)
        engine.save_portfolio_performance(portfolio_df)
        
        logger.info("バックテスト処理完了")
        
        # 簡易統計を表示
        if len(results_df) > 0:
            avg_return = results_df['return_pct'].mean()
            avg_benchmark = results_df['benchmark_return_pct'].mean()
            avg_excess = results_df['excess_return_pct'].mean()
            win_rate = (results_df['return_pct'] > 0).mean() * 100
            
            logger.info(f"平均リターン: {avg_return:.2f}%")
            logger.info(f"平均ベンチマーク: {avg_benchmark:.2f}%")
            logger.info(f"平均超過リターン: {avg_excess:.2f}%")
            logger.info(f"勝率: {win_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"処理エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 