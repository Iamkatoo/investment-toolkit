#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Score Schema Parser - New Scoring System Configuration Interpreter

This module provides the ScoreSchemaParser class that loads and interprets
the new 5-pillar scoring system configuration from YAML files.

Key Features:
- Load and validate YAML schema files
- Extract pillar and sub-indicator configurations
- Validate weights sum to 100 points
- Provide clean configuration access interfaces
- Handle hard gates, soft caps, and macro adjustments
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors"""
    pass


class ScoreSchemaParser:
    """
    Parser for the new scoring system YAML schema configuration.
    
    This class loads and validates the scoring schema, providing clean
    interfaces to access configuration data for the 5-pillar system.
    
    Attributes:
        schema_path (str): Path to the schema YAML file
        schema (Dict): Parsed schema configuration
        pillars (Dict): Pillar configurations
        normalization_config (Dict): Normalization framework settings
    """
    
    def __init__(self, schema_path: str = "config/score_schema.yaml"):
        """
        Initialize the schema parser.
        
        Args:
            schema_path (str): Path to the YAML schema file
            
        Raises:
            FileNotFoundError: If schema file doesn't exist
            SchemaValidationError: If schema validation fails
        """
        self.schema_path = schema_path
        self.schema: Dict[str, Any] = {}
        self.pillars: Dict[str, Any] = {}
        self.normalization_config: Dict[str, Any] = {}
        self.hard_gates: Dict[str, Any] = {}
        self.soft_caps: Dict[str, Any] = {}
        self.macro_adjustments: Dict[str, Any] = {}
        
        # Load and validate schema on initialization
        self.load_schema()
        self.validate_schema()
        
        logger.info(f"Successfully initialized ScoreSchemaParser with {self.schema_path}")
    
    def load_schema(self) -> Dict[str, Any]:
        """
        Load the YAML schema file.
        
        Returns:
            Dict[str, Any]: Parsed schema configuration
            
        Raises:
            FileNotFoundError: If schema file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            schema_file = Path(self.schema_path)
            if not schema_file.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                self.schema = yaml.safe_load(f)
            
            # Extract major configuration sections
            self.pillars = self.schema.get('scoring_pillars', {})
            self.normalization_config = self.schema.get('normalization_framework', {})
            self.hard_gates = self.schema.get('hard_gates', {})
            self.soft_caps = self.schema.get('soft_caps', {})
            self.macro_adjustments = self.schema.get('macro_weight_adjustment', {})
            
            logger.info(f"Successfully loaded schema from {self.schema_path}")
            logger.info(f"Schema version: {self.schema.get('schema_version', 'unknown')}")
            logger.info(f"Found {len(self.pillars)} pillars: {list(self.pillars.keys())}")
            
            return self.schema
            
        except yaml.YAMLError as e:
            error_msg = f"YAML parsing error in {self.schema_path}: {e}"
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)
        except Exception as e:
            error_msg = f"Error loading schema file {self.schema_path}: {e}"
            logger.error(error_msg)
            raise
    
    def validate_schema(self) -> bool:
        """
        Validate the loaded schema for consistency and completeness.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # Check basic schema structure
            required_sections = ['scoring_pillars', 'normalization_framework']
            for section in required_sections:
                if section not in self.schema:
                    raise SchemaValidationError(f"Missing required section: {section}")
            
            # Validate pillars exist and have required structure
            if not self.pillars:
                raise SchemaValidationError("No pillars found in scoring_pillars section")
            
            # Expected pillars in the new system
            expected_pillars = ['value', 'growth', 'quality', 'momentum', 'risk']
            found_pillars = set(self.pillars.keys())
            
            if found_pillars != set(expected_pillars):
                missing = set(expected_pillars) - found_pillars
                extra = found_pillars - set(expected_pillars)
                error_msg = f"Pillar mismatch. Missing: {missing}, Extra: {extra}"
                raise SchemaValidationError(error_msg)
            
            # Validate weights sum to 100
            total_weight = 0
            pillar_weights = {}
            
            for pillar_name, pillar_config in self.pillars.items():
                if 'total_weight' not in pillar_config:
                    raise SchemaValidationError(f"Pillar {pillar_name} missing total_weight")
                
                weight = pillar_config['total_weight']
                if not isinstance(weight, (int, float)) or weight <= 0:
                    raise SchemaValidationError(f"Invalid weight for {pillar_name}: {weight}")
                
                total_weight += weight
                pillar_weights[pillar_name] = weight
                
                # Validate sub-indicators exist
                if 'sub_indicators' not in pillar_config:
                    raise SchemaValidationError(f"Pillar {pillar_name} missing sub_indicators")
                
                # Validate sub-indicator weights sum to pillar total
                sub_total = 0
                for sub_name, sub_config in pillar_config['sub_indicators'].items():
                    if 'weight' not in sub_config:
                        raise SchemaValidationError(
                            f"Sub-indicator {pillar_name}.{sub_name} missing weight"
                        )
                    sub_total += sub_config['weight']
                
                if abs(sub_total - weight) > 0.01:  # Allow small floating point differences
                    raise SchemaValidationError(
                        f"Pillar {pillar_name} sub-indicator weights ({sub_total}) don't sum to pillar weight ({weight})"
                    )
            
            # Validate total weight equals 100 (or 90 if using macro adjustments)
            expected_total = 100
            if self.macro_adjustments:
                # If macro adjustments are present, base weight can be 90
                # with up to ±10 points of macro adjustments
                if abs(total_weight - 90) <= 0.01:
                    expected_total = 90
                    logger.info(f"Schema uses base weight of 90 with macro adjustments")
            
            if abs(total_weight - expected_total) > 0.01:
                if expected_total == 90:
                    raise SchemaValidationError(
                        f"Total weights ({total_weight}) should be 90 (base) or 100 (without macro adjustments)"
                    )
                else:
                    raise SchemaValidationError(f"Total weights ({total_weight}) don't sum to 100")
            
            # Validate normalization framework
            norm_config = self.normalization_config
            if 'cross_sectional' not in norm_config or 'time_series' not in norm_config:
                raise SchemaValidationError("Missing cross_sectional or time_series in normalization_framework")
            
            logger.info(f"Schema validation successful. Pillar weights: {pillar_weights}")
            return True
            
        except SchemaValidationError:
            raise
        except Exception as e:
            error_msg = f"Unexpected error during schema validation: {e}"
            logger.error(error_msg)
            raise SchemaValidationError(error_msg)
    
    def get_pillar_config(self, pillar_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pillar.
        
        Args:
            pillar_name (str): Name of the pillar (value, growth, quality, momentum, risk)
            
        Returns:
            Dict[str, Any]: Pillar configuration
            
        Raises:
            KeyError: If pillar doesn't exist
        """
        if pillar_name not in self.pillars:
            available = list(self.pillars.keys())
            raise KeyError(f"Pillar '{pillar_name}' not found. Available: {available}")
        
        return self.pillars[pillar_name]
    
    def get_sub_indicator_config(self, pillar: str, indicator: str) -> Dict[str, Any]:
        """
        Get configuration for a specific sub-indicator.
        
        Args:
            pillar (str): Pillar name
            indicator (str): Sub-indicator name
            
        Returns:
            Dict[str, Any]: Sub-indicator configuration
            
        Raises:
            KeyError: If pillar or indicator doesn't exist
        """
        pillar_config = self.get_pillar_config(pillar)
        
        if 'sub_indicators' not in pillar_config:
            raise KeyError(f"No sub_indicators found in pillar '{pillar}'")
        
        sub_indicators = pillar_config['sub_indicators']
        if indicator not in sub_indicators:
            available = list(sub_indicators.keys())
            raise KeyError(f"Indicator '{indicator}' not found in pillar '{pillar}'. Available: {available}")
        
        return sub_indicators[indicator]
    
    def get_normalization_config(self, indicator: str) -> Dict[str, Any]:
        """
        Get normalization configuration for a specific indicator.
        
        Args:
            indicator (str): Indicator name to find normalization method
            
        Returns:
            Dict[str, Any]: Normalization configuration
        """
        # Search through all pillars to find the indicator
        for pillar_name, pillar_config in self.pillars.items():
            sub_indicators = pillar_config.get('sub_indicators', {})
            if indicator in sub_indicators:
                sub_config = sub_indicators[indicator]
                norm_method = sub_config.get('normalization_method', 'cross_sectional')
                
                # Return the appropriate normalization config
                if norm_method == 'cross_sectional':
                    return self.normalization_config.get('cross_sectional', {})
                elif norm_method == 'time_series':
                    return self.normalization_config.get('time_series', {})
                elif norm_method == 'binary_score':
                    # Binary scoring doesn't need complex normalization
                    return {'method': 'binary_score'}
                else:
                    logger.warning(f"Unknown normalization method '{norm_method}' for indicator '{indicator}'")
                    return self.normalization_config.get('cross_sectional', {})
        
        # Default to cross-sectional if indicator not found
        logger.warning(f"Indicator '{indicator}' not found, using default cross_sectional normalization")
        return self.normalization_config.get('cross_sectional', {})
    
    def get_hard_gates(self) -> Dict[str, Any]:
        """
        Get hard gate (exclusion) rules.
        
        Returns:
            Dict[str, Any]: Hard gate configurations
        """
        return self.hard_gates
    
    def get_soft_caps(self) -> Dict[str, Any]:
        """
        Get soft cap (conditional score limitation) rules.
        
        Returns:
            Dict[str, Any]: Soft cap configurations
        """
        return self.soft_caps
    
    def get_macro_adjustments(self) -> Dict[str, Any]:
        """
        Get macro weight adjustment rules.
        
        Returns:
            Dict[str, Any]: Macro adjustment configurations
        """
        return self.macro_adjustments
    
    def get_all_sub_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all sub-indicators across all pillars.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping indicator names to their configs
        """
        all_indicators = {}
        
        for pillar_name, pillar_config in self.pillars.items():
            sub_indicators = pillar_config.get('sub_indicators', {})
            for indicator_name, indicator_config in sub_indicators.items():
                # Add pillar context to indicator config
                enriched_config = indicator_config.copy()
                enriched_config['pillar'] = pillar_name
                enriched_config['pillar_weight'] = pillar_config.get('total_weight', 0)
                all_indicators[indicator_name] = enriched_config
        
        return all_indicators
    
    def get_required_columns(self, indicator: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get required database columns for indicators.
        
        Args:
            indicator (Optional[str]): Specific indicator name, or None for all
            
        Returns:
            Union[List[str], Dict[str, List[str]]]: Required columns
        """
        if indicator:
            # Get columns for specific indicator
            all_indicators = self.get_all_sub_indicators()
            if indicator not in all_indicators:
                raise KeyError(f"Indicator '{indicator}' not found")
            return all_indicators[indicator].get('required_columns', [])
        
        # Get all required columns grouped by indicator
        all_columns = {}
        all_indicators = self.get_all_sub_indicators()
        
        for indicator_name, config in all_indicators.items():
            all_columns[indicator_name] = config.get('required_columns', [])
        
        return all_columns
    
    def get_schema_version(self) -> str:
        """
        Get the schema version.
        
        Returns:
            str: Schema version string
        """
        return self.schema.get('schema_version', 'unknown')
    
    def get_schema_metadata(self) -> Dict[str, Any]:
        """
        Get schema metadata including version, creation date, etc.
        
        Returns:
            Dict[str, Any]: Schema metadata
        """
        metadata = {
            'schema_version': self.schema.get('schema_version', 'unknown'),
            'created_date': self.schema.get('created_date', 'unknown'),
            'description': self.schema.get('description', ''),
            'pillar_count': len(self.pillars),
            'total_indicators': sum(len(p.get('sub_indicators', {})) for p in self.pillars.values()),
            'file_path': self.schema_path
        }
        return metadata
    
    def get_pillar_definitions(self) -> Dict[str, Any]:
        """
        Get all pillar definitions.
        
        Returns:
            Dict[str, Any]: Complete pillar configurations
        """
        return self.pillars
    
    def get_indicator_config(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific indicator by name.
        
        Args:
            indicator_name (str): Name of the indicator
            
        Returns:
            Dict[str, Any]: Indicator configuration
            
        Raises:
            KeyError: If indicator not found
        """
        all_indicators = self.get_all_sub_indicators()
        if indicator_name not in all_indicators:
            available = list(all_indicators.keys())
            raise KeyError(f"Indicator '{indicator_name}' not found. Available: {available}")
        
        return all_indicators[indicator_name]
    
    def get_macro_adjustment_rules(self) -> Dict[str, Any]:
        """
        Get macro adjustment rules (alias for get_macro_adjustments).
        
        Returns:
            Dict[str, Any]: Macro adjustment configurations
        """
        return self.get_macro_adjustments()