#!/usr/bin/env python3
"""
V2 Scoring System - Main Scoring Engine

This is the main orchestration engine for the V2 scoring system that integrates
all feature calculation modules, normalization, pillar scoring, and validation.

Implementation Task 9: Main Scoring Engine
- Integrates all V2 modules (Tasks 1-8)
- Provides unified interface for score calculation
- Handles error management and logging
- Supports CLI and batch interfaces
"""

import argparse
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import V2 scoring modules
from investment_toolkit.scoring.schema_parser import ScoreSchemaParser
from investment_toolkit.scoring.fundamental_features import FundamentalTrendCalculator
from investment_toolkit.scoring.quality_value_features import QualityValueCalculator
from investment_toolkit.scoring.price_momentum_features import PriceMomentumCalculator
from investment_toolkit.scoring.normalization_engine import NormalizationEngine
from investment_toolkit.scoring.pillar_scoring import PillarScoringEngine
from investment_toolkit.scoring.validation import ScoringValidator
from investment_toolkit.utilities.feature_flags import FeatureFlags
from investment_toolkit.scoring.ranking_snapshot import (
    DEFAULT_WEEKLY_METHODS as DEFAULT_RANKING_METHODS,
    SUPPORTED_WEEKLY_METHODS as SUPPORTED_RANKING_METHODS,
    ensure_methods as ensure_ranking_methods,
    generate_score_rankings,
)

# Import database utilities (assuming they exist)
try:
    from investment_toolkit.utilities.database import get_stock_universe, save_scores_to_db
except ImportError:
    # Fallback implementations
    def get_stock_universe(target_date: str, limit: Optional[int] = None) -> List[str]:
        """Get stock universe from database for target date"""
        try:
            from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
            from sqlalchemy import create_engine, text
            import logging

            logger = logging.getLogger(__name__)
            engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

            # より堅牢なクエリで、データ品質チェックを含む
            # Note: vw_daily_masterは2015年以降のみのため、fmp_data.daily_pricesを使用
            query = text("""
                SELECT DISTINCT symbol
                FROM fmp_data.daily_prices
                WHERE date = :target_date
                AND close > 0
                AND volume > 0  -- 取引量があることを確認
                AND symbol NOT LIKE 'TEST%'  -- テストデータを除外
                AND LENGTH(symbol) <= 10  -- 異常に長いシンボルを除外
                ORDER BY symbol
            """ + (" LIMIT :limit" if limit is not None else ""))

            params = {'target_date': target_date}
            if limit is not None:
                params['limit'] = limit

            with engine.connect() as conn:
                result = conn.execute(query, params)
                symbols = [row[0] for row in result.fetchall()]

            logger.info(f"Retrieved {len(symbols)} valid symbols for {target_date}")

            # 全銘柄処理を実行（制限なし）

            return symbols

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error fetching stock universe: {e}")
            print(f"Error fetching stock universe: {e}")
            # フォールバック: 少数の主要銘柄を返す
            return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    def save_scores_to_db(scores: pd.DataFrame, table_name: str) -> bool:
        """Save scores to database table"""
        try:
            from investment_toolkit.utilities.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
            from sqlalchemy import create_engine
            import logging
            import numpy as np

            logger = logging.getLogger(__name__)
            engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

            # Ensure table_name includes schema for V2 tables
            if table_name in ['scores_v2', 'daily_scores_v2'] and '.' not in table_name:
                full_table_name = f'backtest_results.{table_name}'
            else:
                full_table_name = table_name

            logger.info(f"Attempting to save {len(scores)} scores to {full_table_name}")

            # Prepare scores DataFrame with required columns
            scores_to_save = scores.copy()

            # Add date column if missing (required for V2 table)
            if 'date' not in scores_to_save.columns and hasattr(scores_to_save, 'calculation_date'):
                scores_to_save['date'] = scores_to_save['calculation_date']

            # Filter out columns that don't exist in the database schema
            # This prevents errors when V2 calculates columns not in the DB schema
            if table_name in ['scores_v2', 'daily_scores_v2']:
                from sqlalchemy import text
                with engine.connect() as conn:
                    schema_result = conn.execute(text('''
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'backtest_results'
                        AND table_name = :table_name
                    '''), {'table_name': table_name})

                    valid_columns = {col[0] for col in schema_result.fetchall()}

                    # Keep only columns that exist in the database schema
                    available_columns = [col for col in scores_to_save.columns if col in valid_columns]
                    removed_columns = [col for col in scores_to_save.columns if col not in valid_columns]

                    if removed_columns:
                        logger.info(f"Removing {len(removed_columns)} columns not in DB schema: {removed_columns}")
                        logger.debug(f"Removed columns: {removed_columns}")

                    # Ensure required columns are present
                    required_columns = ['symbol', 'date']
                    missing_required = [col for col in required_columns if col not in available_columns]
                    if missing_required:
                        logger.error(f"Missing required columns: {missing_required}")
                        raise ValueError(f"Required columns missing: {missing_required}")

                    scores_to_save = scores_to_save[available_columns]

                    # Log column mapping success
                    logger.info(f"Successfully mapped {len(available_columns)} columns to DB schema")
                    logger.debug(f"Final columns for DB: {available_columns}")

            # Determine if we can safely delete existing rows for the calculation date
            delete_date = None
            if 'date' in scores_to_save.columns:
                unique_dates = scores_to_save['date'].dropna().unique()
                if len(unique_dates) == 1:
                    delete_date = pd.to_datetime(unique_dates[0]).date()
                elif len(unique_dates) > 1:
                    logger.warning(
                        "Scores contain multiple dates; skip pre-delete for existing rows"
                    )

            # Clip numeric values to database precision limits (8,4) = max value 9999.9999
            numeric_columns = scores_to_save.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['symbol', 'date']:  # Skip non-numeric identifier columns
                    scores_to_save[col] = scores_to_save[col].clip(-9999.9999, 9999.9999)

            logger.info(f"Clipped {len(numeric_columns)} numeric columns to database precision")

            # Save to database table with optional pre-delete to avoid unique constraint conflicts
            with engine.begin() as conn:
                if delete_date is not None:
                    try:
                        conn.execute(
                            text(f"DELETE FROM {full_table_name} WHERE date = :date"),
                            {'date': delete_date}
                        )
                        logger.info(
                            f"Deleted existing rows for {full_table_name} on {delete_date}"
                        )
                    except Exception as delete_exc:
                        logger.warning(
                            f"Failed to delete existing rows for {full_table_name}: {delete_exc}"
                        )

                scores_to_save.to_sql(
                    table_name,
                    conn,
                    schema='backtest_results',
                    if_exists='append',
                    index=False,
                    method='multi'
                )

            logger.info(f"Successfully saved {len(scores)} scores to {full_table_name}")
            print(f"Saved {len(scores)} scores to {full_table_name}")
            return True

        except Exception as e:
            # Use proper logging instead of just print
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving scores to database table {table_name}: {e}")
            logger.error(f"Scores DataFrame shape: {scores.shape}")
            logger.error(f"Scores DataFrame columns: {list(scores.columns)}")
            print(f"Error saving scores to database: {e}")
            raise  # Re-raise the exception instead of silently returning False
            return False


class ScoringV2Engine:
    """
    Main V2 Scoring Engine
    
    Orchestrates the entire scoring process from feature extraction
    to final score generation and validation.
    """
    
    def __init__(self, schema_path: str, feature_flags: FeatureFlags, 
                 log_level: str = "INFO"):
        """
        Initialize the V2 Scoring Engine
        
        Args:
            schema_path: Path to score schema YAML file
            feature_flags: FeatureFlags instance for system control
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        self.schema_path = schema_path
        self.feature_flags = feature_flags
        
        # Initialize all V2 modules
        self._initialize_modules()
        
        self.logger.info(f"V2 Scoring Engine initialized with {len(self.feature_flags.get_enabled_flags())} enabled flags")
    
    def setup_logging(self, log_level: str):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/scoring_v2.log', mode='a')
            ]
        )
    
    def _initialize_modules(self):
        """Initialize all V2 scoring modules"""
        try:
            # Core schema parser
            self.schema_parser = ScoreSchemaParser(self.schema_path)
            self.schema_parser.load_schema()
            self.schema = self.schema_parser.schema
            
            # Feature calculators
            self.fundamental_calculator = FundamentalTrendCalculator()
            self.quality_value_calculator = QualityValueCalculator()
            self.momentum_calculator = PriceMomentumCalculator()
            
            # Processing engines
            self.normalization_engine = NormalizationEngine()
            self.pillar_scoring = PillarScoringEngine(self.schema_path)
            
            # Validation system
            self.validator = ScoringValidator()
            
            self.logger.info("All V2 modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize V2 modules: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def calculate_scores(self, target_date: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Main score calculation method
        
        Args:
            target_date: Target date for score calculation (YYYY-MM-DD)
            symbols: Optional list of symbols to process. If None, processes all universe.
            
        Returns:
            DataFrame with calculated scores for all symbols
        """
        self.logger.info(f"Starting V2 score calculation for {target_date}")
        
        try:
            # Step 1: Get stock universe
            if symbols is None:
                symbols = get_stock_universe(target_date)
            self.logger.info(f"Processing {len(symbols)} symbols")
            
            # Step 2: Feature extraction
            features = self._extract_all_features(target_date, symbols)
            self.logger.info(f"Extracted {len(features.columns)} features for {len(features)} symbols")
            
            # Step 3: Normalization
            normalized_features = self._normalize_features(features)
            self.logger.info("Feature normalization completed")
            
            # Step 4: Pillar scoring
            pillar_scores = self._calculate_pillar_scores(normalized_features)
            self.logger.info("Pillar scoring completed")
            
            # Step 5: Validation
            validation_results = self._validate_scores(pillar_scores)
            self.logger.info(f"Validation completed: {validation_results['status']}")
            
            # Step 6: Add metadata
            final_scores = self._add_metadata(pillar_scores, target_date)
            
            self.logger.info(f"V2 score calculation completed successfully for {len(final_scores)} symbols")
            return final_scores
            
        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _extract_all_features(self, target_date: str, symbols: List[str]) -> pd.DataFrame:
        """Extract all features using V2 feature calculators"""
        # Start with symbol DataFrame
        all_features = pd.DataFrame({'symbol': symbols})

        try:
            # Extract fundamental features (Growth)
            if self.feature_flags.is_enabled("ENABLE_FUNDAMENTAL_FEATURES"):
                self.logger.info("Calculating fundamental trend features...")
                try:
                    fundamental_features = self.fundamental_calculator.calculate_all_features(
                        target_date, symbols
                    )
                    if not fundamental_features.empty and 'symbol' in fundamental_features.columns:
                        all_features = all_features.merge(fundamental_features, on='symbol', how='left')
                        self.logger.info(f"Added {len(fundamental_features.columns)-1} fundamental features")  # -1 for symbol
                    else:
                        self.logger.warning("No fundamental features calculated")
                except Exception as e:
                    self.logger.error(f"Fundamental feature calculation failed: {e}")
                    # 基本的なダミーフィーチャーを追加（完全な失敗を防ぐため）
                    all_features['fundamental_feature_dummy'] = 0.0

            # Extract quality/value features
            if self.feature_flags.is_enabled("ENABLE_QUALITY_VALUE_FEATURES"):
                self.logger.info("Calculating quality/value features...")
                try:
                    quality_features = self.quality_value_calculator.calculate_all_features(
                        target_date, symbols
                    )
                    if not quality_features.empty and 'symbol' in quality_features.columns:
                        all_features = all_features.merge(quality_features, on='symbol', how='left')
                        self.logger.info(f"Added {len(quality_features.columns)-1} quality/value features")  # -1 for symbol
                    else:
                        self.logger.warning("No quality/value features calculated")
                except Exception as e:
                    self.logger.error(f"Quality/value feature calculation failed: {e}")
                    # 基本的なダミーフィーチャーを追加
                    all_features['quality_value_feature_dummy'] = 0.0

            # Extract momentum/risk features
            if self.feature_flags.is_enabled("ENABLE_MOMENTUM_FEATURES"):
                self.logger.info("Calculating momentum/risk features...")
                try:
                    momentum_features = self.momentum_calculator.calculate_all_features(
                        target_date, symbols
                    )
                    if not momentum_features.empty and 'symbol' in momentum_features.columns:
                        all_features = all_features.merge(momentum_features, on='symbol', how='left')
                        self.logger.info(f"Added {len(momentum_features.columns)-1} momentum/risk features")  # -1 for symbol
                    else:
                        self.logger.warning("No momentum/risk features calculated")
                except Exception as e:
                    self.logger.error(f"Momentum/risk feature calculation failed: {e}")
                    # 基本的なダミーフィーチャーを追加
                    all_features['momentum_risk_feature_dummy'] = 0.0

            # データ品質チェックとNull値処理
            all_features = self._handle_null_values(all_features)

            return all_features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def _handle_null_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values in feature DataFrame with appropriate strategies

        Args:
            features: DataFrame with potentially null values

        Returns:
            pd.DataFrame: DataFrame with null values handled
        """
        try:
            self.logger.info("Starting null value handling...")

            # Null値統計をログ出力
            null_counts = features.isnull().sum()
            null_percentages = (null_counts / len(features) * 100).round(2)

            for col in features.columns:
                if null_counts[col] > 0:
                    self.logger.info(f"Column '{col}': {null_counts[col]} nulls ({null_percentages[col]}%)")

            # Null値処理戦略
            processed_features = features.copy()

            for column in features.columns:
                if features[column].dtype in ['int64', 'float64']:
                    null_count = features[column].isnull().sum()
                    total_count = len(features)
                    null_percentage = null_count / total_count

                    if null_percentage > 0.8:
                        # 80%以上がNullの場合は0で埋める
                        processed_features[column] = processed_features[column].fillna(0)
                        self.logger.warning(f"Column '{column}' has {null_percentage:.2%} nulls, filling with 0")

                    elif null_percentage > 0.3:
                        # 30-80%がNullの場合は中央値で埋める
                        median_value = processed_features[column].median()
                        processed_features[column] = processed_features[column].fillna(median_value)
                        self.logger.info(f"Column '{column}' has {null_percentage:.2%} nulls, filling with median ({median_value})")

                    elif null_percentage > 0:
                        # 30%未満がNullの場合は平均値で埋める
                        mean_value = processed_features[column].mean()
                        processed_features[column] = processed_features[column].fillna(mean_value)
                        self.logger.info(f"Column '{column}' has {null_percentage:.2%} nulls, filling with mean ({mean_value:.2f})")

            # 最終的な品質チェック
            remaining_nulls = processed_features.isnull().sum().sum()
            if remaining_nulls > 0:
                self.logger.warning(f"Still have {remaining_nulls} null values after processing")
                # 残ったNull値は0で埋める
                processed_features = processed_features.fillna(0)

            self.logger.info(f"Null value handling completed. Total nulls before: {features.isnull().sum().sum()}, after: {processed_features.isnull().sum().sum()}")

            return processed_features

        except Exception as e:
            self.logger.error(f"Null value handling failed: {str(e)}")
            # フォールバック: 全てのNull値を0で埋める
            return features.fillna(0)

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using the normalization engine"""
        try:
            self.logger.info("Starting feature normalization...")

            normalized_df = features.copy()

            # Identify numeric columns that need normalization (exclude symbol, date)
            feature_columns = [col for col in features.columns
                             if col not in ['symbol', 'date']
                             and features[col].dtype in ['int64', 'float64']]

            # Apply normalization to each feature column
            for column in feature_columns:
                if column in normalized_df.columns:
                    try:
                        # Use cross-sectional normalization as default
                        normalized_result = self.normalization_engine.normalize_indicator(
                            normalized_df,
                            indicator_name=column,
                            indicator_column=column,
                            method="cross_sectional"
                        )

                        # Update the column with normalized values
                        if f"{column}_normalized" in normalized_result.columns:
                            normalized_df[column] = normalized_result[f"{column}_normalized"]
                        elif f"{column}_score" in normalized_result.columns:
                            normalized_df[column] = normalized_result[f"{column}_score"]

                    except Exception as col_error:
                        self.logger.warning(f"Failed to normalize {column}: {col_error}")
                        # Keep original values if normalization fails
                        continue

            self.logger.info(f"Normalized {len(feature_columns)} features")
            return normalized_df

        except Exception as e:
            self.logger.error(f"Feature normalization failed: {str(e)}")
            raise
    
    def _calculate_pillar_scores(self, normalized_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate pillar scores using schema-compliant weighted calculations"""
        try:
            self.logger.info("Starting schema-compliant pillar score calculation...")

            # Initialize results DataFrame with symbol
            if 'symbol' in normalized_features.columns:
                results = normalized_features[['symbol']].copy()
            else:
                # symbolがindex場合
                results = pd.DataFrame({'symbol': normalized_features.index})
                results.index = normalized_features.index

            # スキーマから5ピラーの重みを取得
            pillars = self.schema.get('scoring_pillars', {})

            if not pillars:
                self.logger.warning("No pillars found in schema, using default calculation")
                return self._calculate_fallback_scores(normalized_features, results)

            # 各ピラーのスコア計算
            for pillar_name, pillar_config in pillars.items():
                try:
                    pillar_score = self._calculate_single_pillar_score(
                        normalized_features, pillar_name, pillar_config
                    )
                    results[f'{pillar_name}_score'] = pillar_score
                    self.logger.info(f"Calculated {pillar_name} pillar scores")

                except Exception as pillar_error:
                    self.logger.warning(f"Failed to calculate {pillar_name} pillar: {pillar_error}")
                    # フォールバック: 中性的なスコア
                    results[f'{pillar_name}_score'] = 10.0  # 20点満点の半分

            # Total score calculation (weighted sum based on schema)
            total_score = self._calculate_total_score(results, pillars)
            results['total_score'] = total_score

            # Add normalized detailed features to the results for DB storage
            # This ensures all calculated features are preserved in the database
            feature_columns_to_add = [col for col in normalized_features.columns
                                    if col not in results.columns and col not in ['symbol', 'date']]

            for col in feature_columns_to_add:
                results[col] = normalized_features[col]

            self.logger.info(f"Schema-compliant calculation completed for {len(pillars)} pillars")
            self.logger.info(f"Added {len(feature_columns_to_add)} detailed features to results")
            return results

        except Exception as e:
            self.logger.error(f"Pillar scoring failed: {str(e)}")
            # 完全なフォールバック
            return self._calculate_fallback_scores(normalized_features,
                                                 pd.DataFrame({'symbol': normalized_features.index}))

    def _calculate_single_pillar_score(self, features: pd.DataFrame, pillar_name: str, pillar_config: dict) -> pd.Series:
        """
        Calculate score for a single pillar based on schema configuration

        Args:
            features: Normalized features DataFrame
            pillar_name: Name of the pillar (value, growth, quality, momentum, risk)
            pillar_config: Pillar configuration from schema

        Returns:
            pd.Series: Pillar scores for all symbols
        """
        try:
            total_weight = pillar_config.get('total_weight', 20)  # Default to 20 points
            sub_indicators = pillar_config.get('sub_indicators', {})

            pillar_score = pd.Series(0.0, index=features.index)

            if not sub_indicators:
                # No sub-indicators defined, use available features
                numeric_cols = [col for col in features.columns
                              if col not in ['symbol', 'date']
                              and features[col].dtype in ['int64', 'float64']]

                if numeric_cols:
                    # 利用可能な特徴量の平均を使用し、適切な範囲にスケール
                    feature_mean = features[numeric_cols].mean(axis=1)
                    # 0-1範囲を想定して0-total_weight点にスケール
                    pillar_score = feature_mean * total_weight
                else:
                    pillar_score = pd.Series(total_weight / 2, index=features.index)  # 中性的な値

            else:
                # サブ指標に基づく計算
                for indicator_name, indicator_config in sub_indicators.items():
                    weight = indicator_config.get('weight', 1)

                    # 対応する特徴量を探す
                    indicator_score = self._calculate_indicator_score(features, indicator_name, weight)
                    pillar_score += indicator_score

            # スコアを0-total_weight範囲にクリップ
            pillar_score = pillar_score.clip(0, total_weight)

            return pillar_score

        except Exception as e:
            self.logger.warning(f"Error calculating {pillar_name} pillar: {e}")
            # フォールバック: 中性的なスコア
            return pd.Series(pillar_config.get('total_weight', 20) / 2, index=features.index)

    def _calculate_indicator_score(self, features: pd.DataFrame, indicator_name: str, weight: float) -> pd.Series:
        """
        Calculate score for a specific indicator

        Args:
            features: Features DataFrame
            indicator_name: Name of the indicator
            weight: Weight of this indicator

        Returns:
            pd.Series: Indicator scores
        """
        # 指標名に基づいて対応する特徴量を探す
        matching_columns = []

        # 指標名に含まれるキーワードで特徴量を検索
        keywords = indicator_name.lower().replace('_', ' ').split()

        for col in features.columns:
            if col in ['symbol', 'date']:
                continue

            col_lower = col.lower()
            # キーワードマッチング
            if any(keyword in col_lower for keyword in keywords):
                matching_columns.append(col)

        if matching_columns:
            # マッチした特徴量の平均を重み付けしたスコアとして使用
            feature_values = features[matching_columns].mean(axis=1)

            # 特別な処理: valuation_deviation_5yは負の値が良い（割安）なので方向を反転
            if 'valuation_deviation' in indicator_name.lower():
                # 負の乖離は良い（割安）、正の乖離は悪い（割高）
                # -50% → 1.0, 0% → 0.5, +50% → 0.0 のように変換
                feature_values = np.clip((0 - feature_values) / 100 + 0.5, 0, 1)
            else:
                # 通常の処理：0-1範囲にクリップ
                feature_values = np.clip(feature_values, 0, 1)

            # weight点にスケール
            return feature_values * weight
        else:
            # マッチする特徴量が見つからない場合は中性的な値
            return pd.Series(weight / 2, index=features.index)

    def _calculate_total_score(self, pillar_scores: pd.DataFrame, pillars_config: dict) -> pd.Series:
        """
        Calculate total score from pillar scores based on schema weights

        Args:
            pillar_scores: DataFrame with individual pillar scores
            pillars_config: Pillar configuration from schema

        Returns:
            pd.Series: Total scores
        """
        total_score = pd.Series(0.0, index=pillar_scores.index)

        for pillar_name, pillar_config in pillars_config.items():
            pillar_column = f'{pillar_name}_score'
            if pillar_column in pillar_scores.columns:
                # 既に適切な重み付けがされているので合計
                total_score += pillar_scores[pillar_column]
            else:
                self.logger.warning(f"Pillar score column {pillar_column} not found")

        # Total scoreを100点満点にクリップ
        total_score = total_score.clip(0, 100)

        return total_score

    def _calculate_fallback_scores(self, features: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback scoring method when schema-based calculation fails

        Args:
            features: Features DataFrame
            results: Results DataFrame with symbol column

        Returns:
            pd.DataFrame: Basic calculated scores
        """
        try:
            self.logger.info("Using fallback scoring calculation")

            # 基本的な5ピラー構造を想定
            pillar_names = ['value', 'growth', 'quality', 'momentum', 'risk']
            pillar_weights = [20, 20, 20, 20, 10]  # Risk pillar is 10 points

            numeric_cols = [col for col in features.columns
                          if col not in ['symbol', 'date']
                          and features[col].dtype in ['int64', 'float64']]

            if numeric_cols:
                # 利用可能な特徴量を5つのピラーに分散
                features_per_pillar = max(1, len(numeric_cols) // 5)

                for i, (pillar_name, weight) in enumerate(zip(pillar_names, pillar_weights)):
                    start_idx = i * features_per_pillar
                    end_idx = min((i + 1) * features_per_pillar, len(numeric_cols))

                    if start_idx < len(numeric_cols):
                        pillar_features = numeric_cols[start_idx:end_idx]
                        if pillar_features:
                            # 正規化された特徴量（0-1範囲）を想定してweight点にスケール
                            pillar_mean = features[pillar_features].mean(axis=1)
                            results[f'{pillar_name}_score'] = (pillar_mean * weight).clip(0, weight)
                        else:
                            results[f'{pillar_name}_score'] = weight / 2
                    else:
                        results[f'{pillar_name}_score'] = weight / 2

                # Total score
                pillar_score_cols = [f'{name}_score' for name in pillar_names]
                results['total_score'] = results[pillar_score_cols].sum(axis=1).clip(0, 100)
            else:
                # 特徴量がない場合は中性的なスコア
                for pillar_name, weight in zip(pillar_names, pillar_weights):
                    results[f'{pillar_name}_score'] = weight / 2
                results['total_score'] = 50.0  # 100点中50点

            # Add normalized detailed features to fallback results as well
            feature_columns_to_add = [col for col in normalized_features.columns
                                    if col not in results.columns and col not in ['symbol', 'date']]

            for col in feature_columns_to_add:
                results[col] = normalized_features[col]

            return results

        except Exception as e:
            self.logger.error(f"Fallback scoring failed: {e}")
            # 最終フォールバック
            for pillar_name in ['value', 'growth', 'quality', 'momentum', 'risk']:
                results[f'{pillar_name}_score'] = 10.0
            results['total_score'] = 50.0

            # Add normalized detailed features to final fallback results as well
            try:
                feature_columns_to_add = [col for col in normalized_features.columns
                                        if col not in results.columns and col not in ['symbol', 'date']]

                for col in feature_columns_to_add:
                    results[col] = normalized_features[col]
            except:
                pass  # If normalized_features is not available, continue without detailed features

            return results

    def _validate_scores(self, scores: pd.DataFrame) -> Dict:
        """Validate calculated scores with comprehensive data quality checks"""
        try:
            self.logger.info("Starting comprehensive score validation...")

            # Enhanced validation results
            validation_results = {
                'status': 'PASS',
                'total_symbols': len(scores),
                'issues': [],
                'warnings': [],
                'data_quality_metrics': {},
                'timestamp': datetime.now().isoformat()
            }

            # Basic validation checks
            if len(scores) == 0:
                validation_results['status'] = 'FAIL'
                validation_results['issues'].append('No scores calculated')
                return validation_results

            # 1. Check for NaN values in total_score
            if 'total_score' in scores.columns:
                nan_count = scores['total_score'].isna().sum()
                if nan_count > 0:
                    validation_results['issues'].append(f'{nan_count} symbols with NaN total_score')

            # 2. Check score range (should be between 0-100)
            if 'total_score' in scores.columns:
                total_scores = scores['total_score']
                invalid_scores = scores[(total_scores < 0) | (total_scores > 100)]
                if len(invalid_scores) > 0:
                    validation_results['issues'].append(f'{len(invalid_scores)} symbols with invalid score range')

                # 3. Calculate data quality metrics
                validation_results['data_quality_metrics'] = {
                    'score_mean': float(total_scores.mean()),
                    'score_std': float(total_scores.std()),
                    'score_median': float(total_scores.median()),
                    'score_min': float(total_scores.min()),
                    'score_max': float(total_scores.max())
                }

                # 4. Check for suspicious score patterns
                score_mean = total_scores.mean()
                if score_mean > 80:
                    validation_results['warnings'].append(f'Average score suspiciously high: {score_mean:.2f}')
                elif score_mean < 20:
                    validation_results['warnings'].append(f'Average score suspiciously low: {score_mean:.2f}')

            # 5. Pillar score validation
            pillar_columns = [col for col in scores.columns if col.endswith('_score') and col != 'total_score']
            pillar_metrics = {}

            for pillar_col in pillar_columns:
                if pillar_col in scores.columns:
                    pillar_scores = scores[pillar_col]
                    pillar_name = pillar_col.replace('_score', '')

                    # Expected max score
                    expected_max = 10 if pillar_name == 'risk' else 20

                    # Check pillar score ranges
                    invalid_pillar = pillar_scores[(pillar_scores < 0) | (pillar_scores > expected_max)]
                    if len(invalid_pillar) > 0:
                        validation_results['warnings'].append(f'{pillar_name} pillar has {len(invalid_pillar)} invalid scores')

                    # Calculate pillar metrics
                    pillar_metrics[pillar_name] = {
                        'mean': float(pillar_scores.mean()),
                        'null_count': int(pillar_scores.isna().sum()),
                        'completeness_pct': float((pillar_scores.count() / len(scores)) * 100)
                    }

                    # Check completeness
                    completeness = (pillar_scores.count() / len(scores)) * 100
                    if completeness < 50:
                        validation_results['warnings'].append(f'{pillar_name} pillar low completeness: {completeness:.1f}%')

            validation_results['data_quality_metrics']['pillar_metrics'] = pillar_metrics

            # 6. Overall data quality assessment
            total_warnings = len(validation_results['warnings'])
            total_issues = len(validation_results['issues'])

            if total_issues > 0:
                validation_results['status'] = 'FAIL'
            elif total_warnings > 5:
                validation_results['status'] = 'WARNING'
            elif total_warnings > 0:
                validation_results['status'] = 'CAUTION'

            # Log validation summary
            if validation_results['status'] == 'PASS':
                self.logger.info("✅ All validation checks passed")
            elif validation_results['status'] == 'CAUTION':
                self.logger.info(f"⚠️ Validation passed with {total_warnings} warnings")
            elif validation_results['status'] == 'WARNING':
                self.logger.warning(f"⚠️ Validation issues detected: {total_warnings} warnings")
            else:
                self.logger.error(f"❌ Validation failed: {total_issues} issues")

            # Log top issues/warnings
            for issue in validation_results['issues'][:3]:
                self.logger.error(f"  Issue: {issue}")
            for warning in validation_results['warnings'][:3]:
                self.logger.warning(f"  Warning: {warning}")

            return validation_results

        except Exception as e:
            self.logger.error(f"Score validation failed: {str(e)}")
            raise
    
    def _add_metadata(self, scores: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Add metadata to final scores"""
        final_scores = scores.copy()

        # Ensure symbol is a column, not an index
        if 'symbol' not in final_scores.columns:
            final_scores = final_scores.reset_index()
            if 'index' in final_scores.columns:
                final_scores = final_scores.rename(columns={'index': 'symbol'})

        # Add calculation metadata
        final_scores['date'] = target_date  # Required column for V2 table

        # Fix column name mismatches between calculated features and DB schema
        column_mappings = {
            # Growth features
            'ttm_rev_yoy': 'ttm_revenue_yoy',
            'final_eps': 'eps',  # From fundamental_features enhanced query

            # Quality/Value features
            'pe_ratio': 'per_score',  # basic_metrics P/E to scoring column
            'pb_ratio': 'pbr_score',  # basic_metrics P/B to scoring column
            'ev_ebitda': 'ev_ebitda_score',  # composite_valuation_metrics to scoring
            'altman_z': 'altman_z_score',  # composite_valuation_metrics to scoring

            # Momentum/Risk features (from price_trend_features)
            'vol_adj_momentum_composite': 'vol_adjusted_momentum_composite',
            'idio_vol': 'idio_vol_percentile',
            'earnings_surprise_vol': 'earnings_surprise_volatility',
            'max_drawdown_12_24m': 'max_drawdown_risk',
            'turnover': 'liquidity_risk',

            # Use existing calculated values where available
            'market_cap': 'market_cap',  # Keep existing calculated market cap
            'roe': 'roe_score',          # basic_metrics ROE to scoring
            'roa': 'roa_score'           # basic_metrics ROA to scoring (if needed)
        }

        # Apply column name mappings
        for calc_col, db_col in column_mappings.items():
            if calc_col in final_scores.columns:
                final_scores[db_col] = final_scores[calc_col]
                # Remove the original column to avoid DB conflicts, but keep 'date' column
                if calc_col != 'date':
                    final_scores = final_scores.drop(columns=[calc_col])

        # Create missing composite indicators from existing features
        # These composites are expected by the DB schema but not calculated by individual feature modules

        # 1. CAGR 3Y/5Y Composite (weighted average of 3-year and 5-year CAGR)
        if 'cagr_3y_eps' in final_scores.columns and 'cagr_5y_eps' in final_scores.columns and 'cagr_3y_5y_composite' not in final_scores.columns:
            # Weight 3-year more heavily as it's more recent
            final_scores['cagr_3y_5y_composite'] = (
                0.6 * final_scores['cagr_3y_eps'].fillna(0) +
                0.4 * final_scores['cagr_5y_eps'].fillna(0)
            )

        # 2. Trend State Composite (combination of trend indicators)
        trend_columns = ['is_above_200dma', 'is_above_50dma', 'is_gc_50_200', 'trend_strength']
        if all(col in final_scores.columns for col in trend_columns) and 'trend_state_composite' not in final_scores.columns:
            final_scores['trend_state_composite'] = (
                0.3 * final_scores['is_above_200dma'].fillna(0) +
                0.2 * final_scores['is_above_50dma'].fillna(0) +
                0.2 * final_scores['is_gc_50_200'].fillna(0) +
                0.3 * final_scores['trend_strength'].fillna(0)
            )

        # 3. Relative Strength Composite (if individual components exist)
        rs_columns = ['sector_rel_strength', 'sizebucket_rel_strength']
        if all(col in final_scores.columns for col in rs_columns) and 'relative_strength_composite' not in final_scores.columns:
            final_scores['relative_strength_composite'] = (
                0.6 * final_scores['sector_rel_strength'].fillna(0) +
                0.4 * final_scores['sizebucket_rel_strength'].fillna(0)
            )

        # Add missing metadata values to reduce null count
        # These are V2-specific metadata fields expected by the DB schema
        if 'v2_algorithm_version' not in final_scores.columns:
            final_scores['v2_algorithm_version'] = 'v2.0.0'

        # Clean up columns with "_x" and "_y" suffixes from pandas merge operations
        # These suffixes are created when DataFrames with overlapping column names are merged
        columns_to_rename = {}
        columns_to_drop = []

        # First pass: identify all columns that need processing
        for col in final_scores.columns:
            if col.endswith('_x'):
                base_name = col[:-2]  # Remove "_x" suffix
                y_col = f"{base_name}_y"

                # Only rename if the base name doesn't already exist
                if base_name not in final_scores.columns:
                    columns_to_rename[col] = base_name

                # Mark "_y" version for dropping if it exists
                if y_col in final_scores.columns:
                    columns_to_drop.append(y_col)

            elif col.endswith('_y'):
                base_name = col[:-2]  # Remove "_y" suffix
                x_col = f"{base_name}_x"

                # Only rename if there's no "_x" version and base name doesn't exist
                if x_col not in final_scores.columns and base_name not in final_scores.columns:
                    columns_to_rename[col] = base_name

        # Apply renames
        if columns_to_rename:
            final_scores = final_scores.rename(columns=columns_to_rename)
            self.logger.info(f"Renamed {len(columns_to_rename)} columns to remove _x/_y suffixes")

        # Drop the columns marked for removal (only if they actually exist)
        if columns_to_drop:
            existing_columns_to_drop = [col for col in columns_to_drop if col in final_scores.columns]
            if existing_columns_to_drop:
                final_scores = final_scores.drop(columns=existing_columns_to_drop)
                self.logger.info(f"Cleaned up {len(existing_columns_to_drop)} duplicate columns from merge operations")

        # Ensure proper data types for database columns that might have been converted to integers
        # This addresses the timestamp type mismatch error
        if 'as_of_date' in final_scores.columns:
            final_scores['as_of_date'] = pd.to_datetime(final_scores['as_of_date']).dt.date

        # Convert Boolean fields from numeric to boolean for PostgreSQL compatibility
        # PostgreSQL boolean fields reject numeric values (1.0, 0.0) and require True/False
        boolean_fields = ['is_above_200dma', 'is_above_50dma', 'is_gc_50_200']
        for field in boolean_fields:
            if field in final_scores.columns:
                # Convert numeric values to boolean using comparison: >0.5 -> True, otherwise False
                # This handles 1.0, 1, True -> True and 0.0, 0, False, NaN -> False
                final_scores[field] = (final_scores[field].fillna(0) > 0.5).astype(bool)
                self.logger.info(f"Converted {field} from numeric to boolean type for PostgreSQL compatibility")

        # Filter out columns that don't exist in the target database table
        # These are temporary columns that shouldn't be saved to the database
        columns_to_exclude = [
            'as_of_date_x', 'as_of_date_y',
            'last_updated', 'last_updated_x', 'last_updated_y',
            'data_quality_flags', 'data_quality_flags_x', 'data_quality_flags_y'
        ]

        existing_columns_to_exclude = [col for col in columns_to_exclude if col in final_scores.columns]
        if existing_columns_to_exclude:
            final_scores = final_scores.drop(columns=existing_columns_to_exclude)
            self.logger.info(f"Excluded {len(existing_columns_to_exclude)} metadata columns from database insertion")

        return final_scores
    
    def save_scores(self, scores: pd.DataFrame, output_table: str = "scores_v2") -> bool:
        """Save scores to database"""
        try:
            self.logger.info(f"Saving scores to table: {output_table}")
            
            success = save_scores_to_db(scores, output_table)
            
            if success:
                self.logger.info(f"Successfully saved {len(scores)} scores to {output_table}")
            else:
                self.logger.error(f"Failed to save scores to {output_table}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Score saving failed: {str(e)}")
            return False
    
    def generate_reports(self, scores: pd.DataFrame, output_dir: str = "reports/v2"):
        """Generate comprehensive reports"""
        try:
            self.logger.info(f"Generating reports in: {output_dir}")
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate validation report
            validation_report = self.validator.generate_validation_report(scores)
            
            # Save reports
            with open(f"{output_dir}/validation_report.html", 'w') as f:
                f.write(validation_report)
            
            # Generate score distribution analysis
            scores.describe().to_csv(f"{output_dir}/score_distribution.csv")
            
            self.logger.info("Report generation completed")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def health_check(self) -> Dict:
        """Perform system health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'version': 'v2.0.0',
            'modules': {},
            'feature_flags': {},
            'overall_status': 'HEALTHY'
        }
        
        try:
            # Check each module
            health_status['modules']['schema_parser'] = 'OK' if self.schema_parser else 'ERROR'
            health_status['modules']['fundamental_calculator'] = 'OK' if self.fundamental_calculator else 'ERROR'
            health_status['modules']['quality_value_calculator'] = 'OK' if self.quality_value_calculator else 'ERROR'
            health_status['modules']['momentum_calculator'] = 'OK' if self.momentum_calculator else 'ERROR'
            health_status['modules']['normalization_engine'] = 'OK' if self.normalization_engine else 'ERROR'
            health_status['modules']['pillar_scoring'] = 'OK' if self.pillar_scoring else 'ERROR'
            health_status['modules']['validator'] = 'OK' if self.validator else 'ERROR'
            
            # Check feature flags
            health_status['feature_flags'] = {
                'enabled_count': len(self.feature_flags.get_enabled_flags()),
                'enabled_flags': self.feature_flags.get_enabled_flags()
            }
            
            # Overall status
            if 'ERROR' in health_status['modules'].values():
                health_status['overall_status'] = 'DEGRADED'
            
        except Exception as e:
            health_status['overall_status'] = 'ERROR'
            health_status['error'] = str(e)
        
        return health_status


def main():
    """CLI interface for V2 scoring system"""
    parser = argparse.ArgumentParser(
        description="V2 Scoring System - Main Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate scores for specific date
  python scoring_v2.py --target-date 2024-09-14 --config config/score_schema.yaml --feature-flags config/feature_flags.yaml
  
  # Calculate for specific symbols
  python scoring_v2.py --target-date 2024-09-14 --config config/score_schema.yaml --feature-flags config/feature_flags.yaml --symbols AAPL,GOOGL,MSFT
  
  # Health check
  python scoring_v2.py --health-check --config config/score_schema.yaml --feature-flags config/feature_flags.yaml
        """
    )
    
    parser.add_argument("--target-date", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--config", required=True, help="Score schema config file path")
    parser.add_argument("--feature-flags", required=True, help="Feature flags config file path")
    parser.add_argument("--symbols", help="Comma-separated symbols (optional)")
    parser.add_argument("--output-table", default="scores_v2", help="Output database table name")
    parser.add_argument("--output-dir", default="reports/v2", help="Output directory for reports")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--save-to-db", action="store_true", help="Save results to database")
    parser.add_argument("--generate-reports", action="store_true", help="Generate comprehensive reports")
    parser.add_argument("--health-check", action="store_true", help="Perform system health check")
    parser.add_argument("--save-rankings", action="store_true", help="Persist ranking snapshots after saving scores")
    parser.add_argument("--ranking-market", choices=["global", "jp", "us"], default="global", help="Market label for ranking snapshots")
    parser.add_argument("--ranking-weekly-window", type=int, default=5, help="Trading days to include for weekly ranking window")
    parser.add_argument(
        "--ranking-weekly-method",
        dest="ranking_weekly_methods",
        action="append",
        choices=sorted(SUPPORTED_RANKING_METHODS),
        help="Weekly aggregation method for rankings (repeatable)",
    )
    parser.add_argument("--ranking-min-observations", type=int, default=3, help="Minimum observations required for weekly ranking inclusion")
    parser.add_argument("--ranking-dry-run", action="store_true", help="Preview ranking snapshot without database writes")
    
    args = parser.parse_args()
    
    try:
        # Initialize feature flags
        flags = FeatureFlags(args.feature_flags)
        
        # Initialize V2 engine
        engine = ScoringV2Engine(
            schema_path=args.config,
            feature_flags=flags,
            log_level=args.log_level
        )
        
        # Health check mode
        if args.health_check:
            health_status = engine.health_check()
            print(f"V2 Scoring Engine Health Status: {health_status['overall_status']}")
            for module, status in health_status['modules'].items():
                print(f"  {module}: {status}")
            print(f"Enabled feature flags ({health_status['feature_flags']['enabled_count']}): {health_status['feature_flags']['enabled_flags']}")
            return 0 if health_status['overall_status'] != 'ERROR' else 1
        
        # Validate required arguments for scoring
        if not args.target_date:
            print("Error: --target-date is required for scoring operations")
            return 1
        
        # Parse symbols if provided
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        
        # Calculate scores
        print(f"V2 Scoring System - Processing date: {args.target_date}")
        scores = engine.calculate_scores(args.target_date, symbols)
        
        print(f"Successfully calculated scores for {len(scores)} symbols")
        print(f"Score pillars: {[col for col in scores.columns if col.endswith('_pillar_score')]}")
        
        # Save to database if requested
        if args.save_to_db:
            success = engine.save_scores(scores, args.output_table)
            if success:
                print(f"Scores saved to database table: {args.output_table}")
                if args.save_rankings:
                    try:
                        weekly_methods = ensure_ranking_methods(args.ranking_weekly_methods or DEFAULT_RANKING_METHODS)
                        snapshot = generate_score_rankings(
                            target_date=args.target_date,
                            weekly_methods=weekly_methods,
                            weekly_window=args.ranking_weekly_window,
                            min_observations=args.ranking_min_observations,
                            market=args.ranking_market,
                            dry_run=args.ranking_dry_run,
                        )
                        if args.ranking_dry_run and not snapshot.empty:
                            print(snapshot.head())
                        print("Ranking snapshots processed successfully")
                    except Exception as ranking_error:
                        print(f"Ranking snapshot generation failed: {ranking_error}", file=sys.stderr)
                        logging.error("Ranking snapshot generation failed", exc_info=ranking_error)
                        return 1
            else:
                print("Failed to save scores to database")
                return 1
        elif args.save_rankings:
            print("Warning: --save-rankings requires --save-to-db to be set; skipping ranking snapshots")

        # Generate reports if requested
        if args.generate_reports:
            engine.generate_reports(scores, args.output_dir)
            print(f"Reports generated in: {args.output_dir}")
        
        print("V2 scoring completed successfully")
        return 0
        
    except Exception as e:
        print(f"V2 scoring failed: {str(e)}", file=sys.stderr)
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
