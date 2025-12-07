"""
J-Quants API 日足データ取得・格納モジュール

J-Quants APIから日足データを取得し、j_quants_data.daily_pricesテーブルに格納します。
既存のFMPデータとの整合性を保ちながら、J-Quants固有のデータも保存します。
"""

import logging
import psycopg2
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time

from .client import JQuantsAPIClient
from investment_toolkit.utilities.config import get_connection

logger = logging.getLogger(__name__)

class JQuantsDailyPriceFetcher:
    """J-Quants 日足データ取得・格納クラス"""
    
    def __init__(self):
        self.client = JQuantsAPIClient()
        self.batch_size = 1000  # バッチサイズ
    
    def _calculate_change_metrics(self, current_close: float, previous_close: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        前日比と前日比率を計算
        
        Args:
            current_close: 当日終値
            previous_close: 前日終値
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (前日比, 前日比率)
        """
        if previous_close is None or previous_close == 0:
            return None, None
        
        change = current_close - previous_close
        change_percent = (change / previous_close) * 100
        
        return change, change_percent
    
    def _calculate_vwap(self, turnover_value: float, volume: float) -> Optional[float]:
        """
        VWAPを計算（売買代金 / 出来高）
        
        Args:
            turnover_value: 売買代金
            volume: 出来高
            
        Returns:
            Optional[float]: VWAP
        """
        if volume == 0 or turnover_value == 0:
            return None
        
        return turnover_value / volume
    
    def _convert_jquants_to_db_format(self, jq_data: List[Dict[str, Any]], target_date: date) -> List[Dict[str, Any]]:
        """
        J-QuantsのAPIレスポンスをデータベース格納用フォーマットに変換
        
        Args:
            jq_data: J-Quants APIレスポンス
            target_date: 対象日付
            
        Returns:
            List[Dict[str, Any]]: データベース格納用データ
        """
        converted_data = []
        
        # 前日終値取得のための辞書（後で実装可能）
        # 今回は簡略化してchange系は後で計算
        
        for item in jq_data:
            try:
                # 必須フィールドの検証
                if not all(key in item for key in ['Date', 'Code']):
                    logger.warning(f"必須フィールドが不足しているデータをスキップ: {item}")
                    continue
                
                # 銘柄コード変換（必要に応じて.Tを付加）
                symbol = str(item['Code'])
                if not symbol.endswith('.T') and len(symbol) == 4:
                    symbol += '.T'
                
                # 基本価格データ（調整後を優先使用）
                open_price = float(item.get('AdjustmentOpen', item.get('Open', 0))) if item.get('AdjustmentOpen') or item.get('Open') else None
                high_price = float(item.get('AdjustmentHigh', item.get('High', 0))) if item.get('AdjustmentHigh') or item.get('High') else None
                low_price = float(item.get('AdjustmentLow', item.get('Low', 0))) if item.get('AdjustmentLow') or item.get('Low') else None
                close_price = float(item.get('AdjustmentClose', item.get('Close', 0))) if item.get('AdjustmentClose') or item.get('Close') else None
                
                # 出来高・売買代金
                volume = float(item.get('AdjustmentVolume', item.get('Volume', 0))) if item.get('AdjustmentVolume') or item.get('Volume') else None
                unadjusted_volume = float(item.get('Volume', 0)) if item.get('Volume') else None
                turnover_value = float(item.get('TurnoverValue', 0)) if item.get('TurnoverValue') else None
                
                # VWAP計算
                vwap = self._calculate_vwap(turnover_value or 0, volume or 0)
                
                # その他のJ-Quants固有データ
                upper_limit = float(item.get('UpperLimit', 0)) if item.get('UpperLimit') and item.get('UpperLimit') != '0' else None
                lower_limit = float(item.get('LowerLimit', 0)) if item.get('LowerLimit') and item.get('LowerLimit') != '0' else None
                adjustment_factor = float(item.get('AdjustmentFactor', 1)) if item.get('AdjustmentFactor') else 1.0
                
                # 未調整価格
                unadjusted_open = float(item.get('Open', 0)) if item.get('Open') else None
                unadjusted_high = float(item.get('High', 0)) if item.get('High') else None
                unadjusted_low = float(item.get('Low', 0)) if item.get('Low') else None
                unadjusted_close = float(item.get('Close', 0)) if item.get('Close') else None
                
                record = {
                    'symbol': symbol,
                    'date': target_date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'adj_close': close_price,  # J-Quantsでは調整後がメイン
                    'volume': volume,
                    'unadjusted_volume': unadjusted_volume,
                    'change': None,  # 後で計算
                    'change_percent': None,  # 後で計算
                    'vwap': vwap,
                    'label': target_date.strftime('%Y-%m-%d'),
                    'change_over_time': None,  # 後で計算
                    'turnover_value': turnover_value,
                    'upper_limit': upper_limit,
                    'lower_limit': lower_limit,
                    'adjustment_factor': adjustment_factor,
                    'unadjusted_open': unadjusted_open,
                    'unadjusted_high': unadjusted_high,
                    'unadjusted_low': unadjusted_low,
                    'unadjusted_close': unadjusted_close
                }
                
                converted_data.append(record)
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"データ変換エラー（スキップ）: {item} - {e}")
                continue
        
        logger.info(f"データ変換完了: {len(jq_data)} -> {len(converted_data)} 件")
        return converted_data
    
    def _save_to_database(self, data: List[Dict[str, Any]]) -> int:
        """
        データをデータベースに保存
        
        Args:
            data: 保存するデータ
            
        Returns:
            int: 保存成功件数
        """
        if not data:
            logger.info("保存するデータがありません")
            return 0
        
        insert_sql = """
        INSERT INTO j_quants_data.daily_prices (
            symbol, date, open, high, low, close, adj_close, volume, unadjusted_volume,
            change, change_percent, vwap, label, change_over_time,
            turnover_value, upper_limit, lower_limit, adjustment_factor,
            unadjusted_open, unadjusted_high, unadjusted_low, unadjusted_close
        ) VALUES (
            %(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(adj_close)s,
            %(volume)s, %(unadjusted_volume)s, %(change)s, %(change_percent)s, %(vwap)s,
            %(label)s, %(change_over_time)s, %(turnover_value)s, %(upper_limit)s,
            %(lower_limit)s, %(adjustment_factor)s, %(unadjusted_open)s, %(unadjusted_high)s,
            %(unadjusted_low)s, %(unadjusted_close)s
        )
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume,
            unadjusted_volume = EXCLUDED.unadjusted_volume,
            change = EXCLUDED.change,
            change_percent = EXCLUDED.change_percent,
            vwap = EXCLUDED.vwap,
            label = EXCLUDED.label,
            change_over_time = EXCLUDED.change_over_time,
            turnover_value = EXCLUDED.turnover_value,
            upper_limit = EXCLUDED.upper_limit,
            lower_limit = EXCLUDED.lower_limit,
            adjustment_factor = EXCLUDED.adjustment_factor,
            unadjusted_open = EXCLUDED.unadjusted_open,
            unadjusted_high = EXCLUDED.unadjusted_high,
            unadjusted_low = EXCLUDED.unadjusted_low,
            unadjusted_close = EXCLUDED.unadjusted_close,
            updated_at = CURRENT_TIMESTAMP
        """
        
        success_count = 0
        
        with get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # バッチ処理で挿入
                    for i in range(0, len(data), self.batch_size):
                        batch = data[i:i + self.batch_size]
                        cursor.executemany(insert_sql, batch)
                        success_count += len(batch)
                        logger.debug(f"バッチ処理進行中: {success_count}/{len(data)}")
                    
                    conn.commit()
                    logger.info(f"データベース保存完了: {success_count} 件")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"データベース保存エラー: {e}")
                    raise
        
        return success_count
    
    def fetch_and_save_daily_prices(self, target_date: date, max_retries: int = 3) -> Dict[str, Any]:
        """
        指定日の日足データを取得してデータベースに保存
        
        Args:
            target_date: 取得対象日
            max_retries: 最大リトライ回数
            
        Returns:
            Dict[str, Any]: 実行結果
        """
        logger.info(f"日足データ取得開始: {target_date}")
        
        result = {
            'target_date': target_date,
            'success': False,
            'fetched_count': 0,
            'saved_count': 0,
            'error': None,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            # J-Quants APIからデータ取得
            jq_data = self.client.get_daily_quotes(target_date)
            result['fetched_count'] = len(jq_data)
            
            if not jq_data:
                logger.warning(f"指定日のデータが見つかりません: {target_date}")
                result['error'] = "No data found for the specified date"
                return result
            
            # データ変換
            converted_data = self._convert_jquants_to_db_format(jq_data, target_date)
            
            # データベース保存
            saved_count = self._save_to_database(converted_data)
            result['saved_count'] = saved_count
            result['success'] = saved_count > 0
            
            logger.info(f"日足データ取得完了: {target_date} - 取得: {result['fetched_count']} 件, 保存: {saved_count} 件")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"日足データ取得エラー ({target_date}): {e}")
        
        finally:
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def fetch_date_range(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        期間指定での日足データ取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            List[Dict[str, Any]]: 各日の実行結果
        """
        logger.info(f"期間指定データ取得開始: {start_date} から {end_date}")
        
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            # 平日のみ処理（土日をスキップ）
            if current_date.weekday() < 5:  # 0-4 = 月-金
                result = self.fetch_and_save_daily_prices(current_date)
                results.append(result)
                
                # API制限を考慮した待機時間
                time.sleep(1)
            
            current_date += timedelta(days=1)
        
        # 結果サマリー
        total_fetched = sum(r['fetched_count'] for r in results)
        total_saved = sum(r['saved_count'] for r in results)
        success_days = sum(1 for r in results if r['success'])
        
        logger.info(f"期間指定データ取得完了: {success_days}/{len(results)} 日成功, "
                   f"総取得: {total_fetched} 件, 総保存: {total_saved} 件")
        
        return results


def main():
    """メイン実行関数（スクリプトとして実行された場合）"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='J-Quants 日足データ取得')
    parser.add_argument('--date', type=str, help='取得日（YYYY-MM-DD形式）', default=None)
    parser.add_argument('--start-date', type=str, help='開始日（YYYY-MM-DD形式）')
    parser.add_argument('--end-date', type=str, help='終了日（YYYY-MM-DD形式）')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='ログレベル')
    
    args = parser.parse_args()
    
    # ログレベル設定
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fetcher = JQuantsDailyPriceFetcher()
    
    if args.start_date and args.end_date:
        # 期間指定
        start = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end = datetime.strptime(args.end_date, '%Y-%m-%d').date()
        results = fetcher.fetch_date_range(start, end)
        
    else:
        # 単日指定
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        else:
            target_date = date.today()
        
        result = fetcher.fetch_and_save_daily_prices(target_date)
        print(f"結果: {result}")


if __name__ == '__main__':
    main() 