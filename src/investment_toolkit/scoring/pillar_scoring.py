#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pillar Scoring Engine - New Scoring System Score Calculation & Constraints

This module implements the pillar-based scoring system for the new 5-pillar approach:
1. Value Pillar (20 points) - Valuation, shareholder yield, dilution
2. Growth Pillar (20 points) - EPS/Revenue growth, acceleration, CAGR, consistency  
3. Quality Pillar (20 points) - ROIC-WACC, profitability, cash quality, leverage
4. Momentum Pillar (20 points) - Price trends, relative strength, momentum persistence
5. Risk Pillar (10 points) - Volatility, liquidity, drawdown, earnings surprise

Key Features:
- Pillar aggregation with weight redistribution for missing indicators
- Soft caps: Quality-based pillar score limitations
- Hard gates: Red flag exclusions and score caps  
- Macro adjustments: Regime-based weight modifications
- Final 100-point score calculation with comprehensive validation
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scoring.schema_parser import ScoreSchemaParser

logger = logging.getLogger(__name__)

class RedFlagSeverity(Enum):
    """Red flag severity levels"""
    CRITICAL = "critical"      # Complete exclusion from ranking
    HIGH = "high"             # Total score cap at 60 points
    MEDIUM = "medium"         # Specific pillar limitations

@dataclass 
class PillarScore:
    """Container for pillar score results"""
    raw_score: float
    adjusted_score: float
    max_possible: float
    used_indicators: int
    total_indicators: int
    soft_caps_applied: List[str]
    metadata: Dict[str, Any]

@dataclass
class RedFlagResult:
    """Container for red flag analysis results"""
    has_critical_flags: bool
    has_high_flags: bool
    critical_reasons: List[str]  
    high_reasons: List[str]
    recommended_action: str
    score_impact: str

@dataclass
class MacroAdjustment:
    """Container for macro adjustment results"""
    regime_state: str
    adjustments: Dict[str, int]  # pillar_name -> adjustment_points
    total_impact: float
    applied_indicators: List[str]

class PillarScoringEngine:
    """
    Core pillar scoring engine implementing 5-pillar score calculation.
    
    This engine aggregates normalized indicators into pillar scores, applies
    soft caps and hard gates, handles macro adjustments, and calculates 
    final 100-point scores with comprehensive validation.
    
    Attributes:
        schema_parser (ScoreSchemaParser): Configuration parser
        pillars (Dict): Pillar definitions and weights
        soft_caps (Dict): Soft cap rules and thresholds
        hard_gates (Dict): Hard gate rules and actions
        macro_rules (Dict): Macro adjustment rules
    """
    
    def __init__(self, schema_path: str = "config/score_schema.yaml"):
        """
        Initialize the pillar scoring engine.
        
        Args:
            schema_path (str): Path to the schema configuration file
        """
        self.schema_parser = ScoreSchemaParser(schema_path)
        
        # Load configuration components
        self.pillars = self.schema_parser.get_pillar_definitions()
        self.soft_caps = self.schema_parser.get_soft_caps()
        self.hard_gates = self.schema_parser.get_hard_gates() 
        self.macro_rules = self.schema_parser.get_macro_adjustment_rules()
        
        logger.info("Pillar scoring engine initialized successfully")

    def calculate_pillar_score(self,
                              pillar_name: str,
                              indicator_values: Dict[str, float],
                              return_details: bool = False) -> Union[float, PillarScore]:
        """
        Calculate score for a specific pillar from indicator values.
        
        Args:
            pillar_name (str): Name of pillar to calculate
            indicator_values (Dict[str, float]): Mapping of indicator_name -> normalized_value
            return_details (bool): Whether to return detailed PillarScore object
            
        Returns:
            Union[float, PillarScore]: Pillar score or detailed result object
        """
        logger.debug(f"Calculating {pillar_name} pillar score")
        
        # Get pillar configuration
        if pillar_name not in self.pillars:
            raise ValueError(f"Unknown pillar: {pillar_name}")
            
        pillar_config = self.pillars[pillar_name]
        max_points = pillar_config.get('total_weight', 20)
        sub_indicators = pillar_config.get('sub_indicators', {})
        
        total_raw_score = 0.0
        total_available_weight = 0.0
        total_used_weight = 0.0
        used_indicators = 0
        
        # Calculate weighted sum of available indicators
        for indicator_name, indicator_config in sub_indicators.items():
            indicator_weight = indicator_config.get('weight', 0)
            total_available_weight += indicator_weight
            
            if indicator_name in indicator_values:
                indicator_value = indicator_values[indicator_name]
                
                if not pd.isna(indicator_value):
                    # Handle direction (higher_is_better vs lower_is_better)
                    direction = indicator_config.get('direction', 'higher_is_better')
                    if direction == 'lower_is_better':
                        # Invert the score for "lower is better" indicators
                        adjusted_value = 1.0 - indicator_value
                    else:
                        adjusted_value = indicator_value
                        
                    # Ensure value is in [0, 1] range
                    adjusted_value = np.clip(adjusted_value, 0.0, 1.0)
                    
                    contribution = adjusted_value * indicator_weight
                    total_raw_score += contribution
                    total_used_weight += indicator_weight
                    used_indicators += 1
        
        # Handle missing indicators via weight redistribution
        if total_used_weight > 0:
            # Redistribute weights to maintain pillar total
            weight_multiplier = total_available_weight / total_used_weight
            adjusted_score = total_raw_score * weight_multiplier
            
            # Scale to pillar max points
            final_score = adjusted_score * (max_points / total_available_weight)
        else:
            # No valid indicators available
            adjusted_score = 0.0
            final_score = 0.0
            
        # Ensure score is within bounds
        final_score = np.clip(final_score, 0.0, max_points)
        
        if return_details:
            return PillarScore(
                raw_score=total_raw_score,
                adjusted_score=final_score,
                max_possible=max_points,
                used_indicators=used_indicators,
                total_indicators=len(sub_indicators),
                soft_caps_applied=[],  # Will be populated by soft caps method
                metadata={
                    'weight_multiplier': weight_multiplier if total_used_weight > 0 else 0,
                    'total_available_weight': total_available_weight,
                    'total_used_weight': total_used_weight
                }
            )
        
        return final_score

    def apply_soft_caps(self, 
                       pillar_scores: Dict[str, float],
                       indicator_values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply soft caps to pillar scores based on quality thresholds.
        
        Soft caps limit specific pillar scores when quality conditions are not met,
        rather than complete exclusion.
        
        Args:
            pillar_scores (Dict[str, float]): Current pillar scores
            indicator_values (Dict[str, float]): Raw indicator values for evaluation
            
        Returns:
            Dict[str, float]: Pillar scores with soft caps applied
        """
        logger.debug("Applying soft caps to pillar scores")
        
        adjusted_scores = pillar_scores.copy()
        applied_caps = []
        
        # Growth Pillar Soft Caps
        
        # 1. Negative ROE -> Growth cap at 50%
        if 'roe' in indicator_values and indicator_values['roe'] < 0:
            original_growth = adjusted_scores.get('growth', 0)
            cap_limit = self.pillars['growth']['total_weight'] * 0.5  # 50% of 20 points = 10
            adjusted_scores['growth'] = min(original_growth, cap_limit)
            if original_growth > cap_limit:
                applied_caps.append('growth_negative_roe_cap')
                logger.debug(f"Applied negative ROE cap to growth: {original_growth:.2f} -> {cap_limit:.2f}")
        
        # 2. Negative revenue growth -> Growth cap at 60%  
        if 'ttm_revenue_yoy' in indicator_values and indicator_values['ttm_revenue_yoy'] < 0:
            original_growth = adjusted_scores.get('growth', 0)
            cap_limit = self.pillars['growth']['total_weight'] * 0.6  # 60% of 20 points = 12
            adjusted_scores['growth'] = min(adjusted_scores['growth'], cap_limit)
            if original_growth > cap_limit:
                applied_caps.append('growth_negative_revenue_cap')
                logger.debug(f"Applied negative revenue growth cap: {original_growth:.2f} -> {cap_limit:.2f}")
        
        # Quality Pillar Soft Caps
        
        # 3. High accruals -> Quality cap at 70%
        if 'operating_accruals' in indicator_values and indicator_values['operating_accruals'] > 0.1:
            original_quality = adjusted_scores.get('quality', 0)
            cap_limit = self.pillars['quality']['total_weight'] * 0.7  # 70% of 20 points = 14
            adjusted_scores['quality'] = min(original_quality, cap_limit)
            if original_quality > cap_limit:
                applied_caps.append('quality_high_accruals_cap')
                logger.debug(f"Applied high accruals cap to quality: {original_quality:.2f} -> {cap_limit:.2f}")
        
        # 4. Low interest coverage -> Quality cap at 60%
        if ('interest_coverage_ttm' in indicator_values and 
            0 < indicator_values['interest_coverage_ttm'] < 2.0):
            original_quality = adjusted_scores.get('quality', 0)  
            cap_limit = self.pillars['quality']['total_weight'] * 0.6  # 60% of 20 points = 12
            adjusted_scores['quality'] = min(adjusted_scores['quality'], cap_limit)
            if original_quality > cap_limit:
                applied_caps.append('quality_low_interest_coverage_cap')
                logger.debug(f"Applied low interest coverage cap: {original_quality:.2f} -> {cap_limit:.2f}")
        
        # Value Pillar Soft Caps
        
        # 5. Extreme P/E ratios -> Value cap at 30%
        if 'pe_ratio' in indicator_values:
            pe = indicator_values['pe_ratio']
            if pe > 100 or pe < 0:
                original_value = adjusted_scores.get('value', 0)
                cap_limit = self.pillars['value']['total_weight'] * 0.3  # 30% of 20 points = 6
                adjusted_scores['value'] = min(original_value, cap_limit)
                if original_value > cap_limit:
                    cap_reason = 'extreme_high_pe' if pe > 100 else 'negative_pe'
                    applied_caps.append(f'value_{cap_reason}_cap')
                    logger.debug(f"Applied extreme PE cap to value: {original_value:.2f} -> {cap_limit:.2f}")
        
        # Momentum Pillar Soft Caps
        
        # 6. High volatility -> Momentum cap at 40%
        if 'idio_vol' in indicator_values and indicator_values['idio_vol'] > 0.8:  # 80% annualized vol
            original_momentum = adjusted_scores.get('momentum', 0)
            cap_limit = self.pillars['momentum']['total_weight'] * 0.4  # 40% of 20 points = 8
            adjusted_scores['momentum'] = min(original_momentum, cap_limit)
            if original_momentum > cap_limit:
                applied_caps.append('momentum_high_volatility_cap')
                logger.debug(f"Applied high volatility cap to momentum: {original_momentum:.2f} -> {cap_limit:.2f}")
        
        if applied_caps:
            logger.info(f"Applied soft caps: {applied_caps}")
            
        return adjusted_scores

    def apply_hard_gates(self,
                        pillar_scores: Dict[str, float], 
                        indicator_values: Dict[str, float],
                        historical_data: Dict[str, List] = None) -> Tuple[Dict[str, float], RedFlagResult]:
        """
        Apply hard gates (red flags) for exclusion and score capping.
        
        Hard gates can completely exclude stocks from rankings or impose
        strict score caps based on fundamental quality issues.
        
        Args:
            pillar_scores (Dict[str, float]): Current pillar scores  
            indicator_values (Dict[str, float]): Raw indicator values
            historical_data (Dict[str, List]): Historical indicator data for multi-period checks
            
        Returns:
            Tuple[Dict[str, float], RedFlagResult]: Adjusted scores and red flag details
        """
        logger.debug("Applying hard gates (red flags)")
        
        critical_flags = []
        high_flags = []
        adjusted_scores = pillar_scores.copy()
        
        # Critical Level Flags (Complete Exclusion)
        
        # 1. ROIC-WACC consecutive negative (4+ quarters)
        if historical_data and 'roic_minus_wacc' in historical_data:
            roic_wacc_history = historical_data['roic_minus_wacc'][-4:]  # Last 4 quarters
            if len(roic_wacc_history) == 4 and all(val < 0 for val in roic_wacc_history if not pd.isna(val)):
                critical_flags.append('roic_wacc_consecutive_negative')
                logger.warning("Critical flag: ROIC-WACC negative for 4+ consecutive quarters")
        
        # 2. Severe financial distress 
        financial_distress_conditions = [
            # High leverage + low interest coverage
            ('net_debt_to_ebitda' in indicator_values and 
             'interest_coverage_ttm' in indicator_values and
             indicator_values['net_debt_to_ebitda'] > 8 and 
             indicator_values['interest_coverage_ttm'] < 1.5),
             
            # Negative operating AND free cash flow  
            ('operating_cash_flow' in indicator_values and
             'free_cash_flow' in indicator_values and
             indicator_values['operating_cash_flow'] < 0 and
             indicator_values['free_cash_flow'] < 0)
        ]
        
        if any(financial_distress_conditions):
            critical_flags.append('financial_distress')
            logger.warning("Critical flag: Severe financial distress detected")
        
        # High Level Flags (Score Cap at 60)
        
        # 3. Severe EPS deceleration
        if ('eps_yoy_slope_4q' in indicator_values and 'ttm_eps_yoy' in indicator_values and
            indicator_values['eps_yoy_slope_4q'] < -0.5 and  # 50% deceleration
            indicator_values['ttm_eps_yoy'] < -0.3):  # 30% decline
            high_flags.append('severe_eps_deceleration')
            logger.warning("High flag: Severe EPS deceleration detected")
        
        # 4. Margin deterioration (if historical data available)
        if (historical_data and 'gross_margin' in historical_data and 'operating_margin' in historical_data):
            current_gm = indicator_values.get('gross_margin')
            current_om = indicator_values.get('operating_margin')
            
            if current_gm is not None and current_om is not None:
                # Compare to year-ago levels (index -4 for quarterly data)
                prev_gm = historical_data['gross_margin'][-4] if len(historical_data['gross_margin']) >= 4 else None
                prev_om = historical_data['operating_margin'][-4] if len(historical_data['operating_margin']) >= 4 else None
                
                if (prev_gm is not None and prev_om is not None and
                    (current_gm - prev_gm) < -0.05 and  # 5% gross margin decline
                    (current_om - prev_om) < -0.03):    # 3% operating margin decline
                    high_flags.append('margin_deterioration')
                    logger.warning("High flag: Significant margin deterioration")
        
        # Apply actions based on flags
        if critical_flags:
            # Complete exclusion - return None/NaN scores
            adjusted_scores = {pillar: np.nan for pillar in adjusted_scores}
            action = "exclude_from_ranking"
            impact = "complete_exclusion"
            logger.warning(f"Stock excluded due to critical flags: {critical_flags}")
            
        elif high_flags:
            # Cap total score at 60 points
            total_current = sum(score for score in adjusted_scores.values() if not pd.isna(score))
            if total_current > 60:
                # Proportionally reduce all pillar scores to sum to 60
                scale_factor = 60.0 / total_current
                adjusted_scores = {pillar: score * scale_factor 
                                 for pillar, score in adjusted_scores.items()}
                action = "cap_total_score_60"
                impact = f"reduced_from_{total_current:.1f}_to_60"
                logger.warning(f"Total score capped at 60 due to high flags: {high_flags}")
            else:
                action = "monitor"
                impact = "no_score_change"
        else:
            action = "none"
            impact = "no_flags_detected"
        
        red_flag_result = RedFlagResult(
            has_critical_flags=bool(critical_flags),
            has_high_flags=bool(high_flags), 
            critical_reasons=critical_flags,
            high_reasons=high_flags,
            recommended_action=action,
            score_impact=impact
        )
        
        return adjusted_scores, red_flag_result

    def apply_macro_adjustments(self,
                               pillar_scores: Dict[str, float],
                               macro_indicators: Dict[str, float] = None) -> Tuple[Dict[str, float], MacroAdjustment]:
        """
        Apply macro regime-based weight adjustments to pillar scores.
        
        Adjusts pillar weights based on macroeconomic conditions like ISM manufacturing,
        credit spreads, and VIX levels within ±2 point limits.
        
        Args:
            pillar_scores (Dict[str, float]): Current pillar scores
            macro_indicators (Dict[str, float]): Current macro indicator values
            
        Returns:
            Tuple[Dict[str, float], MacroAdjustment]: Adjusted scores and adjustment details
        """
        logger.debug("Applying macro adjustments")
        
        if macro_indicators is None:
            # No macro data available, return original scores
            return pillar_scores, MacroAdjustment(
                regime_state="unknown",
                adjustments={},
                total_impact=0.0,
                applied_indicators=[]
            )
        
        total_adjustments = {'value': 0, 'growth': 0, 'quality': 0, 'momentum': 0, 'risk': 0}
        regime_states = []
        applied_indicators = []
        
        # ISM Manufacturing adjustments
        if 'ism_manufacturing' in macro_indicators:
            ism = macro_indicators['ism_manufacturing']
            applied_indicators.append('ism_manufacturing')
            
            if ism > 52:  # Expansion
                total_adjustments['growth'] += 2
                total_adjustments['quality'] -= 1
                total_adjustments['risk'] -= 1
                regime_states.append('ism_expansion')
                
            elif ism < 48:  # Contraction
                total_adjustments['quality'] += 2
                total_adjustments['value'] += 1
                total_adjustments['growth'] -= 2
                total_adjustments['momentum'] -= 1
                regime_states.append('ism_contraction')
            else:
                regime_states.append('ism_neutral')
        
        # High Yield Credit Spread adjustments
        if 'high_yield_oas' in macro_indicators:
            hy_spread = macro_indicators['high_yield_oas']
            applied_indicators.append('high_yield_oas')
            
            if hy_spread < 0.04:  # Tight credit conditions
                total_adjustments['momentum'] += 1
                total_adjustments['risk'] -= 1
                regime_states.append('credit_tight')
                
            elif hy_spread > 0.08:  # Wide credit spreads
                total_adjustments['quality'] += 2
                total_adjustments['value'] += 1
                total_adjustments['momentum'] -= 2
                total_adjustments['risk'] += 1
                regime_states.append('credit_wide')
            else:
                regime_states.append('credit_normal')
        
        # VIX adjustments
        if 'vix_close' in macro_indicators:
            vix = macro_indicators['vix_close']  
            applied_indicators.append('vix_close')
            
            if vix < 18:  # Low volatility
                total_adjustments['growth'] += 1
                total_adjustments['momentum'] += 1
                total_adjustments['risk'] -= 2
                regime_states.append('vix_low')
                
            elif vix > 30:  # High volatility
                total_adjustments['quality'] += 1
                total_adjustments['value'] += 1
                total_adjustments['momentum'] -= 1
                total_adjustments['risk'] += 1
                regime_states.append('vix_high')
            else:
                regime_states.append('vix_normal')
        
        # Apply ±2 point limits
        for pillar in total_adjustments:
            total_adjustments[pillar] = np.clip(total_adjustments[pillar], -2, 2)
        
        # Calculate adjusted scores
        base_weights = {'value': 20, 'growth': 20, 'quality': 20, 'momentum': 20, 'risk': 10}
        adjusted_scores = {}
        total_impact = 0.0
        
        for pillar, score in pillar_scores.items():
            if pd.isna(score):
                adjusted_scores[pillar] = score
                continue
                
            original_weight = base_weights[pillar]
            adjusted_weight = original_weight + total_adjustments[pillar]
            
            # Adjust score proportionally
            weight_multiplier = adjusted_weight / original_weight if original_weight > 0 else 1.0
            adjusted_score = score * weight_multiplier
            
            # Maintain pillar maximum bounds
            max_pillar_score = base_weights[pillar]
            adjusted_scores[pillar] = min(adjusted_score, max_pillar_score)
            
            total_impact += abs(adjusted_scores[pillar] - score)
        
        macro_adjustment = MacroAdjustment(
            regime_state='|'.join(regime_states),
            adjustments=total_adjustments,
            total_impact=total_impact,
            applied_indicators=applied_indicators
        )
        
        if total_impact > 0.1:  # Log only if meaningful adjustment
            logger.info(f"Applied macro adjustments: {total_adjustments}, impact: {total_impact:.2f}")
        
        return adjusted_scores, macro_adjustment

    def calculate_final_score(self,
                             indicator_values: Dict[str, float],
                             historical_data: Dict[str, List] = None,
                             macro_indicators: Dict[str, float] = None,
                             return_breakdown: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Calculate final 100-point score with full pillar processing pipeline.
        
        This is the main entry point that orchestrates the complete scoring process:
        pillar calculation -> soft caps -> hard gates -> macro adjustments -> final score.
        
        Args:
            indicator_values (Dict[str, float]): Normalized indicator values
            historical_data (Dict[str, List]): Historical data for multi-period checks
            macro_indicators (Dict[str, float]): Macro indicators for regime adjustments
            return_breakdown (bool): Whether to return detailed breakdown
            
        Returns:
            Union[float, Dict[str, Any]]: Final score or detailed breakdown
        """
        logger.debug("Calculating final score with full pipeline")
        
        # Step 1: Calculate base pillar scores
        pillar_scores = {}
        pillar_details = {}
        
        for pillar_name in ['value', 'growth', 'quality', 'momentum', 'risk']:
            try:
                if return_breakdown:
                    pillar_details[pillar_name] = self.calculate_pillar_score(
                        pillar_name, indicator_values, return_details=True
                    )
                    pillar_scores[pillar_name] = pillar_details[pillar_name].adjusted_score
                else:
                    pillar_scores[pillar_name] = self.calculate_pillar_score(
                        pillar_name, indicator_values, return_details=False
                    )
            except Exception as e:
                logger.error(f"Error calculating {pillar_name} pillar: {e}")
                pillar_scores[pillar_name] = 0.0
        
        base_total = sum(score for score in pillar_scores.values() if not pd.isna(score))
        
        # Step 2: Apply soft caps
        soft_capped_scores = self.apply_soft_caps(pillar_scores, indicator_values)
        soft_cap_total = sum(score for score in soft_capped_scores.values() if not pd.isna(score))
        
        # Step 3: Apply hard gates
        hard_gated_scores, red_flag_result = self.apply_hard_gates(
            soft_capped_scores, indicator_values, historical_data
        )
        hard_gate_total = sum(score for score in hard_gated_scores.values() 
                             if not pd.isna(score))
        
        # If critical flags exclude the stock, return early
        if red_flag_result.has_critical_flags:
            if return_breakdown:
                return {
                    'final_score': np.nan,
                    'pillar_scores': {k: np.nan for k in pillar_scores},
                    'base_total': base_total,
                    'soft_cap_total': soft_cap_total,
                    'hard_gate_total': np.nan,
                    'macro_adjusted_total': np.nan,
                    'red_flags': red_flag_result,
                    'macro_adjustment': None,
                    'excluded_reason': 'critical_red_flags'
                }
            return np.nan
        
        # Step 4: Apply macro adjustments
        final_scores, macro_adjustment = self.apply_macro_adjustments(
            hard_gated_scores, macro_indicators
        )
        final_total = sum(score for score in final_scores.values() if not pd.isna(score))
        
        # Ensure final score is within [0, 100] bounds
        final_total = np.clip(final_total, 0.0, 100.0)
        
        if return_breakdown:
            return {
                'final_score': final_total,
                'pillar_scores': final_scores,
                'base_total': base_total,
                'soft_cap_total': soft_cap_total,
                'hard_gate_total': hard_gate_total,
                'macro_adjusted_total': final_total,
                'red_flags': red_flag_result,
                'macro_adjustment': macro_adjustment,
                'pillar_details': pillar_details if pillar_details else None,
                'processing_stages': {
                    'base_pillars': pillar_scores,
                    'post_soft_caps': soft_capped_scores,
                    'post_hard_gates': hard_gated_scores,
                    'post_macro_adj': final_scores
                }
            }
        
        return final_total

    def batch_calculate_scores(self,
                              df: pd.DataFrame,
                              indicator_columns: Dict[str, str],
                              historical_data_columns: Dict[str, str] = None,
                              macro_data_columns: Dict[str, str] = None) -> pd.DataFrame:
        """
        Calculate scores for multiple stocks in batch.
        
        Args:
            df (pd.DataFrame): Input dataframe with normalized indicators
            indicator_columns (Dict[str, str]): Mapping of indicator_name -> column_name
            historical_data_columns (Dict[str, str]): Historical data column mappings
            macro_data_columns (Dict[str, str]): Macro data column mappings
            
        Returns:
            pd.DataFrame: DataFrame with calculated pillar and final scores
        """
        logger.info(f"Batch calculating scores for {len(df)} stocks")
        
        result_df = df.copy()
        
        # Add score columns
        score_columns = ['final_score', 'value_score', 'growth_score', 'quality_score', 
                        'momentum_score', 'risk_score', 'red_flag_status', 'macro_regime']
        for col in score_columns:
            result_df[col] = np.nan
        
        # Process each row
        for idx, row in result_df.iterrows():
            try:
                # Extract indicator values
                indicator_values = {}
                for indicator_name, column_name in indicator_columns.items():
                    if column_name in row.index:
                        indicator_values[indicator_name] = row[column_name]
                
                # Extract historical data if available
                historical_data = {}
                if historical_data_columns:
                    for indicator_name, column_name in historical_data_columns.items():
                        if column_name in row.index and pd.notna(row[column_name]):
                            # Assume historical data is stored as comma-separated string
                            try:
                                values = [float(x) for x in str(row[column_name]).split(',')]
                                historical_data[indicator_name] = values
                            except:
                                pass
                
                # Extract macro data if available
                macro_indicators = {}
                if macro_data_columns:
                    for indicator_name, column_name in macro_data_columns.items():
                        if column_name in row.index:
                            macro_indicators[indicator_name] = row[column_name]
                
                # Calculate score with breakdown
                result = self.calculate_final_score(
                    indicator_values, 
                    historical_data if historical_data else None,
                    macro_indicators if macro_indicators else None,
                    return_breakdown=True
                )
                
                # Store results
                result_df.loc[idx, 'final_score'] = result['final_score']
                result_df.loc[idx, 'value_score'] = result['pillar_scores'].get('value', np.nan)
                result_df.loc[idx, 'growth_score'] = result['pillar_scores'].get('growth', np.nan)
                result_df.loc[idx, 'quality_score'] = result['pillar_scores'].get('quality', np.nan)
                result_df.loc[idx, 'momentum_score'] = result['pillar_scores'].get('momentum', np.nan)
                result_df.loc[idx, 'risk_score'] = result['pillar_scores'].get('risk', np.nan)
                
                # Store red flag status
                if result['red_flags'].has_critical_flags:
                    red_flag_status = 'critical'
                elif result['red_flags'].has_high_flags:
                    red_flag_status = 'high'
                else:
                    red_flag_status = 'none'
                result_df.loc[idx, 'red_flag_status'] = red_flag_status
                
                # Store macro regime
                if result['macro_adjustment']:
                    result_df.loc[idx, 'macro_regime'] = result['macro_adjustment'].regime_state
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Leave as NaN values
                continue
        
        logger.info("Batch score calculation complete")
        return result_df