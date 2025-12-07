#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ウォッチリスト管理モジュール
銘柄の追加・削除・追跡・分析を行う
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class WatchlistManager:
    """ウォッチリスト管理クラス"""
    
    def __init__(self, engine):
        self.engine = engine
        
    def add_stock_to_watchlist(self, 
                              symbol: str, 
                              analysis_type: str, 
                              analysis_metadata: Dict[str, Any], 
                              analysis_category: str = None,
                              added_reason: str = None,
                              notes: str = None) -> bool:
        """
        銘柄をウォッチリストに追加
        
        Args:
            symbol: 銘柄コード
            analysis_type: 分析タイプ ('top_stocks', 'rsi35_below' など)
            analysis_metadata: 分析メタデータ (スコア、価格、RSI等)
            analysis_category: 分析カテゴリ
            added_reason: 追加理由
            notes: メモ
            
        Returns:
            成功/失敗
        """
        try:
            # 既に追加済みかチェック
            existing_query = text("""
            SELECT id FROM watchlist.tracked_stocks 
            WHERE symbol = :symbol AND analysis_type = :analysis_type AND is_active = true
            """)
            
            with self.engine.connect() as conn:
                existing = conn.execute(existing_query, {
                    'symbol': symbol,
                    'analysis_type': analysis_type
                }).fetchone()
                
                if existing:
                    print(f"銘柄 {symbol} は既に {analysis_type} でウォッチリスト中です")
                    return False
                
                # 新規追加
                insert_query = text("""
                INSERT INTO watchlist.tracked_stocks 
                (symbol, analysis_type, analysis_category, added_reason, analysis_metadata, notes)
                VALUES (:symbol, :analysis_type, :analysis_category, :added_reason, :analysis_metadata, :notes)
                RETURNING id
                """)
                
                result = conn.execute(insert_query, {
                    'symbol': symbol,
                    'analysis_type': analysis_type,
                    'analysis_category': analysis_category,
                    'added_reason': added_reason,
                    'analysis_metadata': json.dumps(analysis_metadata),
                    'notes': notes
                })
                
                conn.commit()
                new_id = result.fetchone()[0]
                print(f"銘柄 {symbol} をウォッチリストに追加しました (ID: {new_id})")
                return True
                
        except Exception as e:
            print(f"ウォッチリスト追加エラー: {e}")
            return False
    
    def add_multiple_stocks(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        複数銘柄を一括でウォッチリストに追加
        
        Args:
            stocks_data: 銘柄データのリスト
            
        Returns:
            追加結果の統計
        """
        success_count = 0
        failure_count = 0
        errors = []
        
        for stock_data in stocks_data:
            try:
                success = self.add_stock_to_watchlist(
                    symbol=stock_data['symbol'],
                    analysis_type=stock_data['analysis_type'],
                    analysis_metadata=stock_data['metadata'],
                    analysis_category=stock_data.get('analysis_category'),
                    added_reason=stock_data.get('added_reason'),
                    notes=stock_data.get('notes')
                )
                
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    
            except Exception as e:
                failure_count += 1
                errors.append(f"{stock_data['symbol']}: {str(e)}")
        
        return {
            'success_count': success_count,
            'failure_count': failure_count,
            'total_count': len(stocks_data),
            'errors': errors
        }
    
    def remove_stock_from_watchlist(self, stock_id: int, removal_reason: str = None) -> bool:
        """
        ウォッチリストから銘柄を削除（論理削除）
        
        Args:
            stock_id: 銘柄ID
            removal_reason: 削除理由
            
        Returns:
            成功/失敗
        """
        try:
            update_query = text("""
            UPDATE watchlist.tracked_stocks 
            SET is_active = false, removed_date = CURRENT_DATE, removal_reason = :removal_reason
            WHERE id = :stock_id AND is_active = true
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(update_query, {
                    'stock_id': stock_id,
                    'removal_reason': removal_reason
                })
                
                conn.commit()
                
                if result.rowcount > 0:
                    print(f"銘柄ID {stock_id} をウォッチリストから削除しました")
                    return True
                else:
                    print(f"銘柄ID {stock_id} が見つかりません")
                    return False
                    
        except Exception as e:
            print(f"ウォッチリスト削除エラー: {e}")
            return False
    
    def get_current_watchlist(self, analysis_type: str = None) -> pd.DataFrame:
        """
        現在のウォッチリストを取得（パフォーマンス最適化版）
        """
        try:
            # Step 1: まず基本的なウォッチリスト情報を軽量クエリで取得
            base_query = """
            SELECT 
                w.id,
                w.symbol,
                w.added_date,
                w.analysis_type,
                w.analysis_category,
                w.analysis_metadata,
                w.notes,
                (CURRENT_DATE - w.added_date) as days_since_added,
                (w.analysis_metadata->>'price')::numeric as initial_price,
                (w.analysis_metadata->>'rsi')::numeric as initial_rsi,
                (w.analysis_metadata->>'score')::numeric as initial_score
            FROM watchlist.tracked_stocks w
            WHERE w.is_active = true
            """
            
            if analysis_type:
                query = text(base_query + " AND w.analysis_type = :analysis_type ORDER BY w.added_date DESC")
                params = {"analysis_type": analysis_type}
            else:
                query = text(base_query + " ORDER BY w.added_date DESC")
                params = {}
            
            with self.engine.connect() as conn:
                watchlist_df = pd.read_sql(query, conn, params=params)
            
            if watchlist_df.empty:
                print("アクティブなウォッチリスト銘柄がありません")
                return pd.DataFrame()
            
            # 銘柄リストを取得
            symbols = tuple(watchlist_df['symbol'].tolist())
            print(f"ウォッチリスト基本情報取得成功: {len(watchlist_df)}件 - {symbols}")
            
            # Step 2: 現在の市場データを一括取得（最新データのみ）
            current_data_query = text("""
            WITH latest_dates AS (
                SELECT symbol, MAX(date) as latest_date
                FROM backtest_results.vw_daily_master
                WHERE symbol IN :symbols
                GROUP BY symbol
            )
            SELECT 
                vm.symbol,
                vm.close as current_price,
                vm.rsi_14 as current_rsi,
                vm.market_cap
            FROM backtest_results.vw_daily_master vm
            JOIN latest_dates ld ON vm.symbol = ld.symbol AND vm.date = ld.latest_date
            WHERE vm.symbol IN :symbols
            """)
            
            # Step 3: 現在のスコアデータを一括取得（最新データのみ）  
            current_score_query = text("""
            WITH latest_score_dates AS (
                SELECT symbol, MAX(date) as latest_date
                FROM backtest_results.daily_scores
                WHERE symbol IN :symbols
                GROUP BY symbol
            )
            SELECT 
                ds.symbol,
                ds.total_score as current_score,
                ds.value_score,
                ds.growth_score,
                ds.quality_score,
                ds.momentum_score,
                ds.macro_sector_score
            FROM backtest_results.daily_scores ds
            JOIN latest_score_dates lsd ON ds.symbol = lsd.symbol AND ds.date = lsd.latest_date
            WHERE ds.symbol IN :symbols
            """)
            
            # Step 4: 会社情報を一括取得
            company_info_query = text("""
            SELECT DISTINCT ON (symbol)
                symbol, 
                company_name,
                industry,
                sector
            FROM fmp_data.company_profile
            WHERE symbol IN :symbols
            ORDER BY symbol
            """)
            
            # 並列でデータ取得
            with self.engine.connect() as conn:
                try:
                    current_data_df = pd.read_sql(current_data_query, conn, params={'symbols': symbols})
                    print(f"  現在価格データ取得: {len(current_data_df)}件")
                except Exception as e:
                    print(f"  価格データ取得エラー (スキップ): {e}")
                    current_data_df = pd.DataFrame()
                
                try:
                    current_score_df = pd.read_sql(current_score_query, conn, params={'symbols': symbols})
                    print(f"  現在スコアデータ取得: {len(current_score_df)}件")
                except Exception as e:
                    print(f"  スコアデータ取得エラー (スキップ): {e}")
                    current_score_df = pd.DataFrame()
                
                try:
                    company_df = pd.read_sql(company_info_query, conn, params={'symbols': symbols})
                    print(f"  会社情報取得: {len(company_df)}件")
                except Exception as e:
                    print(f"  会社情報取得エラー (スキップ): {e}")
                    company_df = pd.DataFrame()
            
            # Step 5: データをマージして変化率を計算
            if not current_data_df.empty:
                watchlist_df = watchlist_df.merge(current_data_df, on='symbol', how='left')
            else:
                watchlist_df['current_price'] = None
                watchlist_df['current_rsi'] = None
                watchlist_df['market_cap'] = None
            
            if not current_score_df.empty:
                watchlist_df = watchlist_df.merge(current_score_df, on='symbol', how='left')
            else:
                watchlist_df['current_score'] = None
                watchlist_df['value_score'] = None
                watchlist_df['growth_score'] = None
                watchlist_df['quality_score'] = None
                watchlist_df['momentum_score'] = None
                watchlist_df['macro_sector_score'] = None
            
            if not company_df.empty:
                watchlist_df = watchlist_df.merge(company_df, on='symbol', how='left')
            else:
                watchlist_df['company_name'] = watchlist_df['symbol']
                watchlist_df['industry'] = 'N/A'
                watchlist_df['sector'] = 'N/A'
            
            # 変化率を計算
            watchlist_df['price_change_pct'] = None
            watchlist_df['rsi_change'] = None
            watchlist_df['score_change'] = None
            
            # 価格変化率
            mask = (watchlist_df['initial_price'].notna()) & (watchlist_df['current_price'].notna()) & (watchlist_df['initial_price'] > 0)
            watchlist_df.loc[mask, 'price_change_pct'] = (
                (watchlist_df.loc[mask, 'current_price'] - watchlist_df.loc[mask, 'initial_price']) / 
                watchlist_df.loc[mask, 'initial_price'] * 100
            ).round(2)
            
            # RSI変化
            mask = (watchlist_df['initial_rsi'].notna()) & (watchlist_df['current_rsi'].notna())
            watchlist_df.loc[mask, 'rsi_change'] = (
                watchlist_df.loc[mask, 'current_rsi'] - watchlist_df.loc[mask, 'initial_rsi']
            ).round(2)
            
            # スコア変化
            mask = (watchlist_df['initial_score'].notna()) & (watchlist_df['current_score'].notna())
            watchlist_df.loc[mask, 'score_change'] = (
                watchlist_df.loc[mask, 'current_score'] - watchlist_df.loc[mask, 'initial_score']
            ).round(2)
            
            # company_nameのフォールバック
            watchlist_df['company_name'] = watchlist_df['company_name'].fillna(watchlist_df['symbol'])
            watchlist_df['industry'] = watchlist_df['industry'].fillna('N/A')
            watchlist_df['sector'] = watchlist_df['sector'].fillna('N/A')
            
            print(f"ウォッチリスト取得完了: {len(watchlist_df)}件（一括クエリ高速化版）")
            return watchlist_df
            
        except Exception as e:
            print(f"ウォッチリスト取得エラー: {e}")
            import traceback
            traceback.print_exc()
            
            # フォールバック: 基本情報のみ取得
            try:
                print("フォールバック: 基本情報のみ取得中...")
                fallback_query = """
                SELECT 
                    w.id, w.symbol, w.added_date, w.analysis_type, w.analysis_category,
                    w.analysis_metadata, w.notes,
                    (CURRENT_DATE - w.added_date) as days_since_added,
                    (w.analysis_metadata->>'price')::numeric as initial_price,
                    (w.analysis_metadata->>'rsi')::numeric as initial_rsi,
                    (w.analysis_metadata->>'score')::numeric as initial_score,
                    NULL as current_price, NULL as current_rsi, NULL as current_score,
                    NULL as price_change_pct, NULL as rsi_change, NULL as score_change,
                    w.symbol as company_name, 'N/A' as industry, 'N/A' as sector
                FROM watchlist.tracked_stocks w
                WHERE w.is_active = true
                """
                
                if analysis_type:
                    fallback_query += " AND w.analysis_type = :analysis_type"
                    params = {"analysis_type": analysis_type}
                else:
                    params = {}
                
                fallback_query += " ORDER BY w.added_date DESC"
                
                with self.engine.connect() as conn:
                    result = pd.read_sql(text(fallback_query), conn, params=params)
                
                print(f"フォールバック成功: {len(result)}件")
                return result
                
            except Exception as fallback_error:
                print(f"フォールバックも失敗: {fallback_error}")
                return pd.DataFrame()
    
    def get_lightweight_watchlist(self, analysis_type: str = None) -> pd.DataFrame:
        """
        軽量版ウォッチリスト取得（基本情報のみ・高速）
        チェックボックス初期化用の最小限データセット
        """
        try:
            print(f"🚀 軽量版ウォッチリスト取得開始 (analysis_type: {analysis_type})")
            
            # 基本的なウォッチリスト情報のみ取得（JOINなし）
            base_query = """
            SELECT 
                w.id,
                w.symbol,
                w.added_date,
                w.analysis_type,
                w.analysis_category,
                w.notes,
                (CURRENT_DATE - w.added_date) as days_since_added,
                -- 会社名はsymbolと同じにしてAPIレスポンス構造を保持
                w.symbol as company_name
            FROM watchlist.tracked_stocks w
            WHERE w.is_active = true
            """
            
            if analysis_type:
                query = text(base_query + " AND w.analysis_type = :analysis_type ORDER BY w.added_date DESC")
                params = {"analysis_type": analysis_type}
            else:
                query = text(base_query + " ORDER BY w.added_date DESC")
                params = {}
            
            with self.engine.connect() as conn:
                watchlist_df = pd.read_sql(query, conn, params=params)
            
            print(f"✅ 軽量版ウォッチリスト取得成功: {len(watchlist_df)}件")
            return watchlist_df
            
        except Exception as e:
            print(f"❌ 軽量版ウォッチリスト取得失敗: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def update_performance_tracking(self, force_update: bool = False) -> bool:
        """
        ウォッチリスト銘柄のパフォーマンス追跡データを更新
        
        Args:
            force_update: 強制更新フラグ
            
        Returns:
            成功/失敗
        """
        try:
            # 今日のデータが既に存在するかチェック
            if not force_update:
                check_query = text("""
                SELECT COUNT(*) as count 
                FROM watchlist.performance_tracking 
                WHERE analysis_date = CURRENT_DATE
                """)
                
                with self.engine.connect() as conn:
                    result = conn.execute(check_query).fetchone()
                    if result.count > 0:
                        print("今日のパフォーマンストラッキングデータは既に存在します")
                        return True
            
            # アクティブなウォッチリスト銘柄を取得
            watchlist_df = self.get_current_watchlist()
            
            if watchlist_df.empty:
                print("アクティブなウォッチリスト銘柄がありません")
                return True
            
            # パフォーマンストラッキングデータを挿入（軽量化版）
            insert_query = text("""
            INSERT INTO watchlist.performance_tracking 
            (tracked_stock_id, analysis_date, price, rsi_14, total_score, 
             value_score, growth_score, quality_score, momentum_score, macro_sector_score,
             price_change_pct, rsi_change, score_change, days_since_added, market_cap)
            SELECT 
                w.id as tracked_stock_id,
                CURRENT_DATE as analysis_date,
                COALESCE(vm.close, (w.analysis_metadata->>'price')::numeric) as price,
                COALESCE(vm.rsi_14, (w.analysis_metadata->>'rsi')::numeric) as rsi_14,
                COALESCE(ds.total_score, (w.analysis_metadata->>'score')::numeric) as total_score,
                COALESCE(ds.value_score, 0) as value_score,
                COALESCE(ds.growth_score, 0) as growth_score,
                COALESCE(ds.quality_score, 0) as quality_score,
                COALESCE(ds.momentum_score, 0) as momentum_score,
                COALESCE(ds.macro_sector_score, 0) as macro_sector_score,
                CASE 
                    WHEN vm.close IS NOT NULL AND (w.analysis_metadata->>'price')::numeric > 0 
                    THEN ROUND((((vm.close::numeric - (w.analysis_metadata->>'price')::numeric) / (w.analysis_metadata->>'price')::numeric * 100)), 2)
                    ELSE 0
                END as price_change_pct,
                CASE 
                    WHEN vm.rsi_14 IS NOT NULL 
                    THEN ROUND(((vm.rsi_14::numeric - (w.analysis_metadata->>'rsi')::numeric)), 2)
                    ELSE 0
                END as rsi_change,
                CASE 
                    WHEN ds.total_score IS NOT NULL 
                    THEN ROUND(((ds.total_score::numeric - (w.analysis_metadata->>'score')::numeric)), 2)
                    ELSE 0
                END as score_change,
                (CURRENT_DATE - w.added_date) as days_since_added,
                COALESCE(vm.market_cap, 0) as market_cap
            FROM watchlist.tracked_stocks w
            LEFT JOIN backtest_results.vw_daily_master vm ON w.symbol = vm.symbol 
                AND vm.date = (SELECT MAX(date) FROM backtest_results.vw_daily_master WHERE symbol = w.symbol)
            LEFT JOIN backtest_results.daily_scores ds ON w.symbol = ds.symbol 
                AND ds.date = (SELECT MAX(date) FROM backtest_results.daily_scores WHERE symbol = w.symbol)
            WHERE w.is_active = true
            AND NOT EXISTS (
                SELECT 1 FROM watchlist.performance_tracking pt 
                WHERE pt.tracked_stock_id = w.id AND pt.analysis_date = CURRENT_DATE
            )
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(insert_query)
                conn.commit()
                
                print(f"パフォーマンストラッキングデータを更新しました: {result.rowcount}件")
                return True
                
        except Exception as e:
            print(f"パフォーマンストラッキング更新エラー: {e}")
            return False
    
    def get_performance_summary(self, analysis_type: str = None, days_back: int = 30) -> Dict[str, Any]:
        """
        ウォッチリストのパフォーマンスサマリーを取得（軽量化版）
        """
        try:
            # 軽量化: 現在のウォッチリストから直接統計を計算
            print("  パフォーマンスサマリー計算中（軽量版）...")
            
            watchlist_df = self.get_current_watchlist(analysis_type)
            
            if watchlist_df.empty:
                return {'message': 'データがありません'}
            
            # 基本統計を直接計算（重いJOINを避ける）
            summary = {}
            
            if analysis_type:
                # 特定の分析タイプのみ
                analysis_types = [analysis_type]
            else:
                # 全分析タイプ
                analysis_types = watchlist_df['analysis_type'].unique().tolist()
            
            for at in analysis_types:
                subset = watchlist_df[watchlist_df['analysis_type'] == at] if analysis_type is None else watchlist_df
                
                if subset.empty:
                    continue
                
                # 基本統計（軽量計算）
                price_changes = subset['price_change_pct'].dropna()
                total_stocks = len(subset)
                winners = (price_changes > 0).sum() if not price_changes.empty else 0
                losers = (price_changes < 0).sum() if not price_changes.empty else 0
                avg_return = price_changes.mean() if not price_changes.empty else 0
                win_rate = (winners / total_stocks * 100) if total_stocks > 0 else 0
                best_return = price_changes.max() if not price_changes.empty else 0
                worst_return = price_changes.min() if not price_changes.empty else 0
                avg_holding_days = subset['days_since_added'].mean()
                
                summary[at] = {
                    'total_stocks': int(total_stocks),
                    'avg_return_pct': round(float(avg_return), 2),
                    'winners': int(winners),
                    'losers': int(losers),
                    'win_rate': round(win_rate, 1),
                    'best_return_pct': round(float(best_return), 2),
                    'worst_return_pct': round(float(worst_return), 2),
                    'avg_holding_days': round(float(avg_holding_days), 1)
                }
            
            print(f"  パフォーマンスサマリー計算完了: {len(summary)}件")
            return summary
            
        except Exception as e:
            print(f"パフォーマンスサマリー取得エラー: {e}")
            return {'error': str(e)}
    
    def get_stock_performance_history(self, symbol: str, analysis_type: str = None) -> pd.DataFrame:
        """
        特定銘柄のパフォーマンス履歴を取得
        
        Args:
            symbol: 銘柄コード
            analysis_type: 分析タイプ
            
        Returns:
            パフォーマンス履歴データ
        """
        try:
            base_query = """
            SELECT 
                pt.*,
                w.symbol,
                w.analysis_type,
                w.added_date,
                w.analysis_metadata
            FROM watchlist.performance_tracking pt
            JOIN watchlist.tracked_stocks w ON pt.tracked_stock_id = w.id
            WHERE w.symbol = :symbol
            """
            
            params = {'symbol': symbol}
            
            if analysis_type:
                query = text(base_query + " AND w.analysis_type = :analysis_type ORDER BY pt.analysis_date")
                params['analysis_type'] = analysis_type
            else:
                query = text(base_query + " ORDER BY pt.analysis_date")
            
            return pd.read_sql(query, self.engine, params=params)
            
        except Exception as e:
            print(f"銘柄パフォーマンス履歴取得エラー: {e}")
            return pd.DataFrame()


# ユーティリティ関数
def format_watchlist_metadata(symbol: str, analysis_type: str, 
                            score: float = None, price: float = None, 
                            rsi: float = None, **kwargs) -> Dict[str, Any]:
    """ウォッチリスト用のメタデータを整形"""
    metadata = {
        'symbol': symbol,
        'analysis_type': analysis_type,
        'added_date': datetime.now().strftime('%Y-%m-%d'),
        'added_time': datetime.now().strftime('%H:%M:%S')
    }
    
    if score is not None:
        metadata['score'] = float(score)
    if price is not None:
        metadata['price'] = float(price)
    if rsi is not None:
        metadata['rsi'] = float(rsi)
    
    # 追加のキーワード引数をメタデータに追加
    for key, value in kwargs.items():
        if value is not None:
            metadata[key] = value
    
    return metadata 