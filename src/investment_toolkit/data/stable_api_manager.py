"""
FMP Stable API による月次データ更新管理クラス
v3 → stable API 移行対応
"""

import logging
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from sqlalchemy import create_engine, text
import pandas as pd

from investment_analysis.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from investment_analysis.data.fmp_api import FMPAPI
from investment_analysis.utilities.notification import NotificationManager


class StableAPIManager:
    """
    FMP Stable API による月次データ更新を管理するクラス
    - actively-trading-list の取得・保存
    - 銘柄ごとの company profile 取得・保存
    - symbol_status の更新（exchange判定）
    - 従業員数データの差分更新
    - 監視・異常検知・通知
    """
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger('stable_api_manager')
        self.logger.setLevel(logging.INFO)
        
        # データベース接続
        self.db_engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        
        # FMP API インスタンス
        self.api = FMPAPI()
        
        # 通知マネージャー
        self.notifier = NotificationManager()
        
        # 実行ID（ログとメトリクス追跡用）
        self.run_id = self._generate_run_id()
        
        # 統計情報
        self.stats = {
            'actively_trading_fetched': 0,
            'profiles_fetched': 0,
            'profiles_errors': 0,
            'symbol_status_updated': 0,
            'employee_counts_updated': 0,
            'anomalies_detected': 0,
            'api_requests': 0,
            'api_errors': 0
        }
        
        # 異常検知のしきい値設定
        self.anomaly_thresholds = {
            'actively_trading_volume_change_pct': 20.0,  # ±20%
            'profile_missing_rate_pct': 1.0,  # 1%
            'exchange_missing_rate_pct': 1.0,  # 1%
            'api_error_rate_pct': 2.0,  # 2%
            'consecutive_timeouts': 5,  # 5回連続
            'processing_time_multiplier': 1.5  # 150%
        }
        
        # markets.json から設定を読み込み
        self.target_exchanges = self._load_target_exchanges()
        
    def _generate_run_id(self) -> str:
        """実行IDを生成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"monthly_{timestamp}_{unique_id}"
    
    def _load_target_exchanges(self) -> List[str]:
        """markets.json から対象取引所リストを読み込み"""
        try:
            import os
            from pathlib import Path
            
            config_file = Path(__file__).resolve().parent.parent.parent / 'config' / 'markets.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('target_exchanges', [])
        except Exception as e:
            self.logger.error(f"markets.json読み込みエラー: {e}")
            # デフォルト値
            return [
                "NASDAQ Global Select",
                "NASDAQ Stock Exchange", 
                "Tokyo",
                "NASDAQ Capital Market",
                "NASDAQ",
                "Nasdaq Global Select",
                "Nasdaq",
                "NASDAQ Global Market",
                "NASDAQ Stock Market",
                "New York Stock Exchange"
            ]
    
    def _log_step_start(self, step_name: str) -> datetime:
        """ステップ開始をログに記録"""
        start_time = datetime.now()
        self.logger.info(f"=== {step_name} 開始 ===")
        self.logger.info(f"実行ID: {self.run_id}")
        self.logger.info(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        return start_time
    
    def _log_step_end(self, step_name: str, start_time: datetime, success_count: int = 0, error_count: int = 0):
        """ステップ終了をログに記録し、DBに保存"""
        end_time = datetime.now()
        duration = end_time - start_time
        duration_ms = int(duration.total_seconds() * 1000)
        
        self.logger.info(f"=== {step_name} 完了 ===")
        self.logger.info(f"所要時間: {duration.total_seconds():.3f}秒")
        self.logger.info(f"成功: {success_count}件, エラー: {error_count}件")
        
        # DB に実行履歴を保存
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ops.run_history_monthly 
                    (run_id, started_at, finished_at, step, ok_count, error_count, duration_ms, exit_code)
                    VALUES (:run_id, :started_at, :finished_at, :step, :ok_count, :error_count, :duration_ms, :exit_code)
                    ON CONFLICT (run_id, step) DO UPDATE SET
                        finished_at = EXCLUDED.finished_at,
                        ok_count = EXCLUDED.ok_count,
                        error_count = EXCLUDED.error_count,
                        duration_ms = EXCLUDED.duration_ms,
                        exit_code = EXCLUDED.exit_code
                """)
                conn.execute(query, {
                    'run_id': self.run_id,
                    'started_at': start_time,
                    'finished_at': end_time,
                    'step': step_name,
                    'ok_count': success_count,
                    'error_count': error_count,
                    'duration_ms': duration_ms,
                    'exit_code': 0 if error_count == 0 else 1
                })
                conn.commit()
        except Exception as e:
            self.logger.error(f"実行履歴の保存エラー: {e}")
    
    def _detect_and_notify_anomaly(self, anomaly_code: str, observed_value: float, 
                                  baseline: float, threshold: float, message: str, 
                                  severity: str = "WARN"):
        """異常検知と通知"""
        try:
            # 異常をDBに記録
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ops.anomalies 
                    (run_id, anomaly_code, severity, observed_value, baseline, threshold_value, message)
                    VALUES (:run_id, :anomaly_code, :severity, :observed_value, :baseline, :threshold_value, :message)
                """)
                conn.execute(query, {
                    'run_id': self.run_id,
                    'anomaly_code': anomaly_code,
                    'severity': severity,
                    'observed_value': observed_value,
                    'baseline': baseline,
                    'threshold_value': threshold,
                    'message': message
                })
                conn.commit()
            
            # 通知送信
            self.notifier.send_anomaly_notification(
                anomaly_code=anomaly_code,
                severity=severity,
                observed_value=observed_value,
                baseline=baseline,
                message=message,
                run_id=self.run_id
            )
            
            self.stats['anomalies_detected'] += 1
            self.logger.warning(f"異常検知: {anomaly_code} - {message}")
            
        except Exception as e:
            self.logger.error(f"異常検知処理エラー: {e}")
    
    def fetch_and_store_actively_trading_list(self) -> bool:
        """actively-trading-list を取得してDBに保存"""
        step_name = "actively-trading-list取得"
        start_time = self._log_step_start(step_name)
        
        try:
            # API からデータ取得
            self.logger.info("actively-trading-list を取得中...")
            data = self.api.get_actively_trading_list()
            self.stats['api_requests'] += 1
            
            if not data:
                self.logger.error("actively-trading-list が空です")
                self._log_step_end(step_name, start_time, 0, 1)
                return False
            
            current_count = len(data)
            self.stats['actively_trading_fetched'] = current_count
            self.logger.info(f"actively-trading-list: {current_count}件取得")
            
            # 過去3ヶ月の平均件数と比較して異常検知
            baseline_count = self._get_actively_trading_baseline()
            if baseline_count > 0:
                change_pct = abs(current_count - baseline_count) / baseline_count * 100
                threshold = self.anomaly_thresholds['actively_trading_volume_change_pct']
                
                if change_pct > threshold:
                    severity = "ALERT" if change_pct > threshold * 2 else "WARN"
                    self._detect_and_notify_anomaly(
                        anomaly_code="actively_trading_volume_change",
                        observed_value=current_count,
                        baseline=baseline_count,
                        threshold=threshold,
                        message=f"actively-trading-list件数が過去平均から{change_pct:.1f}%変動しました",
                        severity=severity
                    )
            
            # DB に保存（UPSERT）
            with self.db_engine.connect() as conn:
                success_count = 0
                for item in data:
                    symbol = item.get('symbol', '')
                    name = item.get('name', '')
                    
                    if not symbol:
                        continue
                    
                    # UPSERT処理
                    query = text("""
                        INSERT INTO fmp_data.active_trading_symbols (symbol, name, last_seen_at, updated_at)
                        VALUES (:symbol, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (symbol) DO UPDATE SET
                            name = EXCLUDED.name,
                            last_seen_at = EXCLUDED.last_seen_at,
                            updated_at = EXCLUDED.updated_at
                    """)
                    conn.execute(query, {'symbol': symbol, 'name': name})
                    success_count += 1
                
                conn.commit()
                self.logger.info(f"active_trading_symbols テーブルに {success_count}件保存")
            
            self._log_step_end(step_name, start_time, success_count, 0)
            return True
            
        except Exception as e:
            self.logger.error(f"actively-trading-list取得エラー: {e}")
            self.stats['api_errors'] += 1
            self._log_step_end(step_name, start_time, 0, 1)
            return False
    
    def _get_actively_trading_baseline(self) -> float:
        """過去3ヶ月のactively-trading-list平均件数を取得"""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT AVG(daily_count) as avg_count
                    FROM (
                        SELECT DATE(updated_at) as date, COUNT(*) as daily_count
                        FROM fmp_data.active_trading_symbols
                        WHERE updated_at >= CURRENT_DATE - INTERVAL '90 days'
                        GROUP BY DATE(updated_at)
                    ) daily_stats
                """)
                result = conn.execute(query)
                row = result.fetchone()
                return float(row[0]) if row and row[0] else 0.0
        except Exception as e:
            self.logger.error(f"ベースライン計算エラー: {e}")
            return 0.0
    
    def fetch_and_store_company_profiles(self, limit: Optional[int] = None) -> bool:
        """全active_trading_symbolsの会社プロファイルを取得してDBに保存"""
        step_name = "company-profile取得"
        start_time = self._log_step_start(step_name)
        
        try:
            # active_trading_symbols から銘柄リストを取得
            with self.db_engine.connect() as conn:
                query = text("SELECT symbol FROM fmp_data.active_trading_symbols ORDER BY symbol")
                if limit:
                    query = text(f"SELECT symbol FROM fmp_data.active_trading_symbols ORDER BY symbol LIMIT {limit}")
                
                result = conn.execute(query)
                symbols = [row[0] for row in result]
            
            total_symbols = len(symbols)
            self.logger.info(f"company-profile取得対象: {total_symbols}銘柄")
            
            if not symbols:
                self._log_step_end(step_name, start_time, 0, 0)
                return True
            
            success_count = 0
            error_count = 0
            
            for i, symbol in enumerate(symbols):
                if i > 0 and i % 100 == 0:
                    self.logger.info(f"進捗: {i}/{total_symbols} ({i/total_symbols*100:.1f}%)")
                
                try:
                    # company profile 取得
                    profile_data = self.api.get_stable_company_profile(symbol)
                    self.stats['api_requests'] += 1
                    
                    if profile_data:
                        # DB に保存
                        self._save_company_profile(symbol, profile_data)
                        success_count += 1
                        self.stats['profiles_fetched'] += 1
                    else:
                        error_count += 1
                        self.stats['profiles_errors'] += 1
                        self.logger.debug(f"{symbol}: profile取得失敗")
                    
                    # レート制限対策
                    time.sleep(0.2)  # 200ms待機
                    
                except KeyboardInterrupt:
                    self.logger.warning(f"プロセスが中断されました（進捗: {i}/{total_symbols}）")
                    raise
                except Exception as e:
                    error_count += 1
                    self.stats['profiles_errors'] += 1
                    self.stats['api_errors'] += 1
                    self.logger.error(f"{symbol} profile取得エラー: {e}")
                    
                    # 重大なエラーの場合は停止
                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        self.logger.error(f"接続エラーが発生しました。1分後に再試行します。")
                        time.sleep(60)
                        continue
            
            # 異常検知: profile欠損率
            if total_symbols > 0:
                missing_rate = error_count / total_symbols * 100
                threshold = self.anomaly_thresholds['profile_missing_rate_pct']
                
                if missing_rate > threshold:
                    severity = "CRITICAL" if missing_rate > 5.0 else "ALERT"
                    self._detect_and_notify_anomaly(
                        anomaly_code="profile_missing_rate",
                        observed_value=missing_rate,
                        baseline=threshold,
                        threshold=threshold,
                        message=f"company profileの取得失敗率が{missing_rate:.1f}%です",
                        severity=severity
                    )
            
            self._log_step_end(step_name, start_time, success_count, error_count)
            return error_count == 0
            
        except Exception as e:
            self.logger.error(f"company-profile取得処理エラー: {e}")
            self._log_step_end(step_name, start_time, 0, 1)
            return False
    
    def _save_company_profile(self, symbol: str, profile_data: Dict):
        """company profileをDBに保存（当日のスナップショット）"""
        try:
            with self.db_engine.connect() as conn:
                # 当日のスナップショットとして保存
                today = datetime.now().date()
                
                query = text("""
                    INSERT INTO fmp_data.company_profile (
                        symbol, date, price, beta, vol_avg, mkt_cap, last_div, "range",
                        changes, company_name, currency, cik, isin, cusip, exchange,
                        exchange_short_name, industry, website, description, ceo,
                        sector, country, full_time_employees, phone, address, city,
                        state, zip, dcf_diff, dcf, image, ipo_date, default_image,
                        is_etf, is_actively_trading, is_adr, is_fund
                    )
                    VALUES (
                        :symbol, :date, :price, :beta, :vol_avg, :mkt_cap, :last_div, :range_val,
                        :changes, :company_name, :currency, :cik, :isin, :cusip, :exchange,
                        :exchange_short_name, :industry, :website, :description, :ceo,
                        :sector, :country, :full_time_employees, :phone, :address, :city,
                        :state, :zip_val, :dcf_diff, :dcf, :image, :ipo_date, :default_image,
                        :is_etf, :is_actively_trading, :is_adr, :is_fund
                    )
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        price = EXCLUDED.price,
                        beta = EXCLUDED.beta,
                        vol_avg = EXCLUDED.vol_avg,
                        mkt_cap = EXCLUDED.mkt_cap,
                        last_div = EXCLUDED.last_div,
                        "range" = EXCLUDED."range",
                        changes = EXCLUDED.changes,
                        company_name = EXCLUDED.company_name,
                        currency = EXCLUDED.currency,
                        cik = EXCLUDED.cik,
                        isin = EXCLUDED.isin,
                        cusip = EXCLUDED.cusip,
                        exchange = EXCLUDED.exchange,
                        exchange_short_name = EXCLUDED.exchange_short_name,
                        industry = EXCLUDED.industry,
                        website = EXCLUDED.website,
                        description = EXCLUDED.description,
                        ceo = EXCLUDED.ceo,
                        sector = EXCLUDED.sector,
                        country = EXCLUDED.country,
                        full_time_employees = EXCLUDED.full_time_employees,
                        phone = EXCLUDED.phone,
                        address = EXCLUDED.address,
                        city = EXCLUDED.city,
                        state = EXCLUDED.state,
                        zip = EXCLUDED.zip,
                        dcf_diff = EXCLUDED.dcf_diff,
                        dcf = EXCLUDED.dcf,
                        image = EXCLUDED.image,
                        ipo_date = EXCLUDED.ipo_date,
                        default_image = EXCLUDED.default_image,
                        is_etf = EXCLUDED.is_etf,
                        is_actively_trading = EXCLUDED.is_actively_trading,
                        is_adr = EXCLUDED.is_adr,
                        is_fund = EXCLUDED.is_fund
                """)
                
                # データの変換・デフォルト値設定
                params = {
                    'symbol': symbol,
                    'date': today,
                    'price': profile_data.get('price'),
                    'beta': profile_data.get('beta'),
                    'vol_avg': profile_data.get('volAvg'),
                    'mkt_cap': profile_data.get('mktCap'),
                    'last_div': profile_data.get('lastDiv'),
                    'range_val': profile_data.get('range'),
                    'changes': profile_data.get('changes'),
                    'company_name': profile_data.get('companyName'),
                    'currency': profile_data.get('currency'),
                    'cik': profile_data.get('cik'),
                    'isin': profile_data.get('isin'),
                    'cusip': profile_data.get('cusip'),
                    'exchange': profile_data.get('exchange'),
                    'exchange_short_name': profile_data.get('exchangeShortName'),
                    'industry': profile_data.get('industry'),
                    'website': profile_data.get('website'),
                    'description': profile_data.get('description'),
                    'ceo': profile_data.get('ceo'),
                    'sector': profile_data.get('sector'),
                    'country': profile_data.get('country'),
                    'full_time_employees': profile_data.get('fullTimeEmployees'),
                    'phone': profile_data.get('phone'),
                    'address': profile_data.get('address'),
                    'city': profile_data.get('city'),
                    'state': profile_data.get('state'),
                    'zip_val': profile_data.get('zip'),
                    'dcf_diff': profile_data.get('dcfDiff'),
                    'dcf': profile_data.get('dcf'),
                    'image': profile_data.get('image'),
                    'ipo_date': profile_data.get('ipoDate'),
                    'default_image': 1 if profile_data.get('defaultImage') is True else 0 if profile_data.get('defaultImage') is False else None,
                    'is_etf': 1 if profile_data.get('isEtf') is True else 0 if profile_data.get('isEtf') is False else None,
                    'is_actively_trading': 1 if profile_data.get('isActivelyTrading') is True else 0 if profile_data.get('isActivelyTrading') is False else None,
                    'is_adr': 1 if profile_data.get('isAdr') is True else 0 if profile_data.get('isAdr') is False else None,
                    'is_fund': 1 if profile_data.get('isFund') is True else 0 if profile_data.get('isFund') is False else None
                }
                
                conn.execute(query, params)
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"{symbol} company profile保存エラー: {e}")
            raise
    
    def update_symbol_status_from_profiles(self) -> bool:
        """company_profileのexchange情報を基にsymbol_statusを更新"""
        step_name = "symbol_status更新"
        start_time = self._log_step_start(step_name)
        
        try:
            today = datetime.now().date()
            
            # 当日のcompany_profileからexchange情報を取得
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT symbol, exchange, company_name, sector, country, is_actively_trading
                    FROM fmp_data.company_profile
                    WHERE date = :today AND exchange IS NOT NULL
                """)
                result = conn.execute(query, {'today': today})
                profiles = list(result)
            
            if not profiles:
                self.logger.warning("当日のcompany profileデータがありません")
                self._log_step_end(step_name, start_time, 0, 0)
                return False
            
            total_profiles = len(profiles)
            success_count = 0
            
            # 対象exchange（markets.jsonで定義）
            target_exchanges = set(self.target_exchanges)
            
            self.logger.info(f"対象取引所: {', '.join(sorted(target_exchanges))}")
            self.logger.info(f"symbol_status更新対象: {total_profiles}銘柄")
            
            with self.db_engine.connect() as conn:
                for symbol, exchange, company_name, sector, country, is_actively_trading in profiles:
                    try:
                        # exchange正規化
                        normalized_exchange = self._normalize_exchange(exchange)
                        
                        # target_exchangesに含まれているかチェック
                        is_active = (normalized_exchange in target_exchanges and 
                                   is_actively_trading is True)
                        
                        # symbol_statusを更新（手動無効化は尊重）
                        query = text("""
                            INSERT INTO fmp_data.symbol_status 
                            (symbol, name, exchange, type, is_active, manually_deactivated, last_updated)
                            VALUES (:symbol, :name, :exchange, 'stock', :is_active, false, CURRENT_TIMESTAMP)
                            ON CONFLICT (symbol) DO UPDATE SET
                                name = EXCLUDED.name,
                                exchange = EXCLUDED.exchange,
                                is_active = CASE 
                                    WHEN fmp_data.symbol_status.manually_deactivated = true THEN false
                                    ELSE EXCLUDED.is_active
                                END,
                                last_updated = EXCLUDED.last_updated
                        """)
                        
                        conn.execute(query, {
                            'symbol': symbol,
                            'name': company_name or '',
                            'exchange': normalized_exchange,
                            'is_active': is_active
                        })
                        
                        success_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"{symbol} symbol_status更新エラー: {e}")
                
                conn.commit()
            
            self.stats['symbol_status_updated'] = success_count
            
            # 異常検知: exchange欠損率
            missing_exchange_count = total_profiles - len([p for p in profiles if p[1]])  # exchange IS NOT NULL なので0のはず
            if total_profiles > 0:
                missing_rate = missing_exchange_count / total_profiles * 100
                threshold = self.anomaly_thresholds['exchange_missing_rate_pct']
                
                if missing_rate > threshold:
                    self._detect_and_notify_anomaly(
                        anomaly_code="exchange_missing_rate",
                        observed_value=missing_rate,
                        baseline=threshold,
                        threshold=threshold,
                        message=f"exchangeデータの欠損率が{missing_rate:.1f}%です",
                        severity="ALERT"
                    )
            
            self._log_step_end(step_name, start_time, success_count, 0)
            return True
            
        except Exception as e:
            self.logger.error(f"symbol_status更新エラー: {e}")
            self._log_step_end(step_name, start_time, 0, 1)
            return False
    
    def _normalize_exchange(self, exchange: str) -> str:
        """exchange名を正規化"""
        if not exchange:
            return ""
        
        # markets.json の exchange_mapping を使用
        try:
            import os
            from pathlib import Path
            
            config_file = Path(__file__).resolve().parent.parent.parent / 'config' / 'markets.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
                mapping = config.get('exchange_mapping', {})
                return mapping.get(exchange, exchange)
        except Exception:
            # フォールバック: 直接正規化
            return exchange
    
    def update_exchange_restrictions(self) -> bool:
        """取引所制限によるmanually_deactivated状態を更新"""
        step_name = "取引所制限更新"
        start_time = self._log_step_start(step_name)
        
        try:
            # 対象取引所の定義（NASDAQ系、NYSE系、Tokyo系）
            target_exchanges = {
                # NASDAQ系
                'NASDAQ',
                'NASDAQ Global Select',
                'NASDAQ Global Market', 
                'NASDAQ Stock Market',
                'NASDAQ Capital Market',
                'Nasdaq',
                'Nasdaq Global Select',
                'NASDAQ Stock Exchange',
                
                # NYSE系
                'New York Stock Exchange',
                'NYSE',
                'NYSE American',
                'NYSE Arca',
                'NYSE Chicago',
                'NYSE National',
                
                # Tokyo系
                'Tokyo',
                'TSE',
                'Tokyo Stock Exchange',
                'Japan Exchange Group',
                
                # その他の主要米国市場
                'BATS',
                'CBOE',
                'IEX'
            }
            
            self.logger.info(f"対象取引所数: {len(target_exchanges)}")
            self.logger.info(f"対象取引所: {', '.join(sorted(target_exchanges))}")
            
            with self.db_engine.connect() as conn:
                # 除外対象銘柄を特定（対象取引所以外でmanually_deactivated=false）
                result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM fmp_data.symbol_status
                    WHERE exchange IS NOT NULL 
                    AND exchange NOT IN :target_exchanges
                    AND manually_deactivated = false
                """), {'target_exchanges': tuple(target_exchanges)})
                
                symbols_to_deactivate_count = result.scalar()
                
                # 再有効化対象銘柄を特定（対象取引所でmanually_deactivated=true）
                result = conn.execute(text("""
                    SELECT COUNT(*)
                    FROM fmp_data.symbol_status
                    WHERE exchange IS NOT NULL 
                    AND exchange IN :target_exchanges
                    AND manually_deactivated = true
                """), {'target_exchanges': tuple(target_exchanges)})
                
                symbols_to_reactivate_count = result.scalar()
                
                self.logger.info(f"手動除外対象: {symbols_to_deactivate_count}銘柄")
                self.logger.info(f"再有効化対象: {symbols_to_reactivate_count}銘柄")
                
                deactivated_count = 0
                reactivated_count = 0
                
                # 除外対象を手動除外
                if symbols_to_deactivate_count > 0:
                    result = conn.execute(text("""
                        UPDATE fmp_data.symbol_status 
                        SET manually_deactivated = true,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE exchange IS NOT NULL 
                        AND exchange NOT IN :target_exchanges
                        AND manually_deactivated = false
                    """), {'target_exchanges': tuple(target_exchanges)})
                    
                    deactivated_count = result.rowcount
                    self.logger.info(f"手動除外設定完了: {deactivated_count}銘柄")
                
                # 対象取引所の手動除外を解除
                if symbols_to_reactivate_count > 0:
                    result = conn.execute(text("""
                        UPDATE fmp_data.symbol_status 
                        SET manually_deactivated = false,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE exchange IS NOT NULL 
                        AND exchange IN :target_exchanges
                        AND manually_deactivated = true
                    """), {'target_exchanges': tuple(target_exchanges)})
                    
                    reactivated_count = result.rowcount
                    self.logger.info(f"手動除外解除完了: {reactivated_count}銘柄")
                
                conn.commit()
                
                # 統計情報の更新
                self.stats['exchange_restrictions_deactivated'] = deactivated_count
                self.stats['exchange_restrictions_reactivated'] = reactivated_count
                
                self._log_step_end(step_name, start_time, deactivated_count + reactivated_count, 0)
                return True
                
        except Exception as e:
            self.logger.error(f"取引所制限更新エラー: {e}")
            self._log_step_end(step_name, start_time, 0, 1)
            return False

    def update_employee_counts_for_active_symbols(self) -> bool:
        """is_active=trueの銘柄の従業員数データを差分更新"""
        step_name = "従業員数データ更新"
        start_time = self._log_step_start(step_name)
        
        try:
            # is_active=true の銘柄を取得
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT symbol 
                    FROM fmp_data.symbol_status 
                    WHERE is_active = true 
                    ORDER BY symbol
                """)
                result = conn.execute(query)
                active_symbols = [row[0] for row in result]
            
            total_symbols = len(active_symbols)
            self.logger.info(f"従業員数データ更新対象: {total_symbols}銘柄（is_active=true）")
            
            if not active_symbols:
                self._log_step_end(step_name, start_time, 0, 0)
                return True
            
            success_count = 0
            error_count = 0
            
            for i, symbol in enumerate(active_symbols):
                if i > 0 and i % 50 == 0:
                    self.logger.info(f"従業員数データ進捗: {i}/{total_symbols} ({i/total_symbols*100:.1f}%)")
                
                try:
                    # 従業員数データ取得
                    employee_data = self.api.get_stable_employee_count(symbol)
                    self.stats['api_requests'] += 1
                    
                    if employee_data:
                        # DBに保存（差分更新）
                        self._save_employee_count(symbol, employee_data)
                        success_count += 1
                        self.stats['employee_counts_updated'] += 1
                    else:
                        error_count += 1
                        self.logger.debug(f"{symbol}: 従業員数データなし")
                    
                    # レート制限対策
                    time.sleep(0.3)  # 300ms待機
                    
                except Exception as e:
                    error_count += 1
                    self.stats['api_errors'] += 1
                    self.logger.error(f"{symbol} 従業員数データエラー: {e}")
            
            self._log_step_end(step_name, start_time, success_count, error_count)
            return True
            
        except Exception as e:
            self.logger.error(f"従業員数データ更新エラー: {e}")
            self._log_step_end(step_name, start_time, 0, 1)
            return False
    
    def _save_employee_count(self, symbol: str, employee_data: List[Dict]):
        """従業員数データをDBに保存"""
        try:
            with self.db_engine.connect() as conn:
                for item in employee_data:
                    date_str = item.get('date')
                    employee_count = item.get('employeeCount')
                    
                    if not date_str or employee_count is None:
                        continue
                    
                    query = text("""
                        INSERT INTO fmp_data.employee_counts (symbol, date, employee_count)
                        VALUES (:symbol, :date, :employee_count)
                        ON CONFLICT (symbol, date) DO UPDATE SET
                            employee_count = EXCLUDED.employee_count
                    """)
                    
                    conn.execute(query, {
                        'symbol': symbol,
                        'date': date_str,
                        'employee_count': employee_count
                    })
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"{symbol} 従業員数データ保存エラー: {e}")
            raise
    
    def run_monthly_update(self, test_mode: bool = False, limit: Optional[int] = None) -> bool:
        """月次更新の全工程を実行"""
        self.logger.info("=" * 60)
        self.logger.info("FMP Stable API 月次更新開始")
        self.logger.info("=" * 60)
        
        overall_start_time = datetime.now()
        
        # 開始通知
        self.notifier.send_monthly_update_start_notification("FMP Stable API月次更新")
        
        success = True
        
        try:
            # Step 1: actively-trading-list 取得・保存
            if not self.fetch_and_store_actively_trading_list():
                success = False
                if not test_mode:
                    return False
            
            # Step 2: company profile 取得・保存
            if not self.fetch_and_store_company_profiles(limit=limit):
                success = False
                if not test_mode:
                    return False
            
            # Step 3: symbol_status 更新
            if not self.update_symbol_status_from_profiles():
                success = False
                if not test_mode:
                    return False
            
            # Step 4: 取引所制限によるmanually_deactivated更新
            if not self.update_exchange_restrictions():
                success = False
                # 制限は必須ではないため、継続
            
            # Step 5: 従業員数データ更新（is_active=trueのみ）
            if not self.update_employee_counts_for_active_symbols():
                success = False
                # 従業員数データは必須ではないため、継続
            
            return success
            
        except Exception as e:
            self.logger.error(f"月次更新中に予期せぬエラー: {e}")
            return False
        
        finally:
            # 完了通知
            overall_end_time = datetime.now()
            total_duration = overall_end_time - overall_start_time
            
            # サマリー作成
            summary = self._create_summary(overall_start_time, overall_end_time, success)
            api_usage = self.api.get_api_usage_summary()
            
            self.notifier.send_monthly_update_completion_notification(
                "FMP Stable API月次更新", 
                summary, 
                api_usage
            )
            
            self.logger.info("=" * 60)
            self.logger.info("FMP Stable API 月次更新完了")
            self.logger.info("=" * 60)
    
    def _create_summary(self, start_time: datetime, end_time: datetime, success: bool) -> str:
        """実行結果サマリーを作成"""
        duration = end_time - start_time
        
        summary_lines = [
            f"実行ID: {self.run_id}",
            f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"処理時間: {duration}",
            f"全体結果: {'成功' if success else '失敗'}",
            "",
            "【処理結果】",
            f"actively-trading取得: {self.stats['actively_trading_fetched']}件",
            f"company profile取得: {self.stats['profiles_fetched']}件",
            f"company profileエラー: {self.stats['profiles_errors']}件", 
            f"symbol_status更新: {self.stats['symbol_status_updated']}件",
            f"従業員数データ更新: {self.stats['employee_counts_updated']}件",
            "",
            "【API統計】",
            f"APIリクエスト総数: {self.stats['api_requests']}回",
            f"APIエラー総数: {self.stats['api_errors']}回",
            f"異常検知: {self.stats['anomalies_detected']}件"
        ]
        
        return "\n".join(summary_lines)


def main():
    """テスト実行用のメイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FMP Stable API 月次更新")
    parser.add_argument("--test", action="store_true", help="テストモード（エラー時も処理継続）")
    parser.add_argument("--limit", type=int, help="処理銘柄数の制限（テスト用）")
    args = parser.parse_args()
    
    manager = StableAPIManager()
    success = manager.run_monthly_update(test_mode=args.test, limit=args.limit)
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
