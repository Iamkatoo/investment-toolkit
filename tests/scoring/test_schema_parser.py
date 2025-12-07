#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for ScoreSchemaParser

These tests verify the schema parser can correctly load, validate, and provide
access to the scoring system configuration.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from investment_toolkit.scoring.schema_parser import ScoreSchemaParser, SchemaValidationError


class TestScoreSchemaParser:
    """Test suite for ScoreSchemaParser class"""
    
    @pytest.fixture
    def valid_schema(self):
        """Create a valid test schema"""
        return {
            'schema_version': '2.0.0',
            'created_date': '2025-09-11',
            'description': 'Test schema',
            'scoring_pillars': {
                'value': {
                    'total_weight': 20,
                    'description': 'Value pillar',
                    'sub_indicators': {
                        'pe_ratio': {
                            'weight': 10,
                            'description': 'P/E ratio',
                            'direction': 'lower_is_better',
                            'normalization_method': 'cross_sectional',
                            'required_columns': ['close_price', 'eps']
                        },
                        'pb_ratio': {
                            'weight': 10,
                            'description': 'P/B ratio',
                            'direction': 'lower_is_better',
                            'normalization_method': 'cross_sectional',
                            'required_columns': ['close_price', 'book_value']
                        }
                    }
                },
                'growth': {
                    'total_weight': 80,  # Intentionally large to test 100 total
                    'description': 'Growth pillar',
                    'sub_indicators': {
                        'eps_growth': {
                            'weight': 80,
                            'description': 'EPS growth',
                            'direction': 'higher_is_better',
                            'normalization_method': 'time_series',
                            'required_columns': ['eps']
                        }
                    }
                }
            },
            'normalization_framework': {
                'cross_sectional': {
                    'method': 'percentile_ranking',
                    'grouping_dimensions': ['gics_sector', 'market_cap_bucket']
                },
                'time_series': {
                    'lookback_period': '5_years',
                    'methods': {
                        'z_score': {
                            'calculation': '(current_value - mean_5y) / std_5y'
                        }
                    }
                }
            },
            'hard_gates': {
                'negative_roe': {
                    'condition': 'roe < 0',
                    'action': 'exclude_from_ranking'
                }
            },
            'soft_caps': {
                'high_pe': {
                    'condition': 'pe_ratio > 100',
                    'action': 'cap_value_score_at_50_percent'
                }
            }
        }
    
    @pytest.fixture
    def temp_schema_file(self, valid_schema):
        """Create a temporary schema file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_schema, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_init_success(self, temp_schema_file):
        """Test successful initialization"""
        parser = ScoreSchemaParser(temp_schema_file)
        assert parser.schema_path == temp_schema_file
        assert parser.schema is not None
        assert len(parser.pillars) == 2
        assert 'value' in parser.pillars
        assert 'growth' in parser.pillars
    
    def test_init_file_not_found(self):
        """Test initialization with non-existent file"""
        with pytest.raises(FileNotFoundError):
            ScoreSchemaParser("nonexistent_file.yaml")
    
    def test_load_schema_success(self, temp_schema_file):
        """Test successful schema loading"""
        parser = ScoreSchemaParser(temp_schema_file)
        schema = parser.load_schema()
        
        assert schema['schema_version'] == '2.0.0'
        assert 'scoring_pillars' in schema
        assert len(parser.pillars) == 2
        assert len(parser.normalization_config) > 0
    
    def test_validate_schema_success(self, temp_schema_file):
        """Test successful schema validation"""
        parser = ScoreSchemaParser(temp_schema_file)
        result = parser.validate_schema()
        assert result is True
    
    def test_validate_schema_missing_sections(self):
        """Test schema validation with missing sections"""
        invalid_schema = {'schema_version': '1.0.0'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_schema, f)
            temp_file = f.name
        
        try:
            with pytest.raises(SchemaValidationError, match="Missing required section"):
                ScoreSchemaParser(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_schema_wrong_pillars(self):
        """Test schema validation with wrong pillar names"""
        invalid_schema = {
            'scoring_pillars': {
                'wrong_pillar': {
                    'total_weight': 100,
                    'sub_indicators': {}
                }
            },
            'normalization_framework': {
                'cross_sectional': {},
                'time_series': {}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_schema, f)
            temp_file = f.name
        
        try:
            with pytest.raises(SchemaValidationError, match="Pillar mismatch"):
                ScoreSchemaParser(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_schema_wrong_total_weight(self, valid_schema):
        """Test schema validation with weights not summing to 100"""
        valid_schema['scoring_pillars']['value']['total_weight'] = 50  # Total will be 130
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_schema, f)
            temp_file = f.name
        
        try:
            with pytest.raises(SchemaValidationError, match="don't sum to 100"):
                ScoreSchemaParser(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_schema_sub_indicator_weights_mismatch(self, valid_schema):
        """Test schema validation with sub-indicator weights not matching pillar weight"""
        # Change pe_ratio weight so sub-indicators don't sum to pillar weight
        valid_schema['scoring_pillars']['value']['sub_indicators']['pe_ratio']['weight'] = 15
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_schema, f)
            temp_file = f.name
        
        try:
            with pytest.raises(SchemaValidationError, match="don't sum to pillar weight"):
                ScoreSchemaParser(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_get_pillar_config_success(self, temp_schema_file):
        """Test getting pillar configuration"""
        parser = ScoreSchemaParser(temp_schema_file)
        value_config = parser.get_pillar_config('value')
        
        assert value_config['total_weight'] == 20
        assert value_config['description'] == 'Value pillar'
        assert 'sub_indicators' in value_config
        assert len(value_config['sub_indicators']) == 2
    
    def test_get_pillar_config_not_found(self, temp_schema_file):
        """Test getting non-existent pillar"""
        parser = ScoreSchemaParser(temp_schema_file)
        
        with pytest.raises(KeyError, match="Pillar 'nonexistent' not found"):
            parser.get_pillar_config('nonexistent')
    
    def test_get_sub_indicator_config_success(self, temp_schema_file):
        """Test getting sub-indicator configuration"""
        parser = ScoreSchemaParser(temp_schema_file)
        pe_config = parser.get_sub_indicator_config('value', 'pe_ratio')
        
        assert pe_config['weight'] == 10
        assert pe_config['direction'] == 'lower_is_better'
        assert pe_config['normalization_method'] == 'cross_sectional'
        assert 'close_price' in pe_config['required_columns']
    
    def test_get_sub_indicator_config_not_found(self, temp_schema_file):
        """Test getting non-existent sub-indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        
        with pytest.raises(KeyError, match="Indicator 'nonexistent' not found"):
            parser.get_sub_indicator_config('value', 'nonexistent')
    
    def test_get_normalization_config_cross_sectional(self, temp_schema_file):
        """Test getting normalization config for cross-sectional indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        norm_config = parser.get_normalization_config('pe_ratio')
        
        assert norm_config['method'] == 'percentile_ranking'
        assert 'grouping_dimensions' in norm_config
    
    def test_get_normalization_config_time_series(self, temp_schema_file):
        """Test getting normalization config for time-series indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        norm_config = parser.get_normalization_config('eps_growth')
        
        assert norm_config['lookback_period'] == '5_years'
        assert 'methods' in norm_config
    
    def test_get_normalization_config_not_found(self, temp_schema_file):
        """Test getting normalization config for non-existent indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        # Should return default cross_sectional config with warning
        norm_config = parser.get_normalization_config('nonexistent_indicator')
        
        assert norm_config['method'] == 'percentile_ranking'
    
    def test_get_all_sub_indicators(self, temp_schema_file):
        """Test getting all sub-indicators"""
        parser = ScoreSchemaParser(temp_schema_file)
        all_indicators = parser.get_all_sub_indicators()
        
        assert len(all_indicators) == 3  # pe_ratio, pb_ratio, eps_growth
        assert 'pe_ratio' in all_indicators
        assert 'pb_ratio' in all_indicators
        assert 'eps_growth' in all_indicators
        
        # Check enriched data
        pe_config = all_indicators['pe_ratio']
        assert pe_config['pillar'] == 'value'
        assert pe_config['pillar_weight'] == 20
    
    def test_get_required_columns_specific_indicator(self, temp_schema_file):
        """Test getting required columns for specific indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        columns = parser.get_required_columns('pe_ratio')
        
        assert isinstance(columns, list)
        assert 'close_price' in columns
        assert 'eps' in columns
    
    def test_get_required_columns_all(self, temp_schema_file):
        """Test getting required columns for all indicators"""
        parser = ScoreSchemaParser(temp_schema_file)
        all_columns = parser.get_required_columns()
        
        assert isinstance(all_columns, dict)
        assert 'pe_ratio' in all_columns
        assert 'pb_ratio' in all_columns
        assert 'eps_growth' in all_columns
    
    def test_get_required_columns_not_found(self, temp_schema_file):
        """Test getting required columns for non-existent indicator"""
        parser = ScoreSchemaParser(temp_schema_file)
        
        with pytest.raises(KeyError, match="Indicator 'nonexistent' not found"):
            parser.get_required_columns('nonexistent')
    
    def test_get_hard_gates(self, temp_schema_file):
        """Test getting hard gates configuration"""
        parser = ScoreSchemaParser(temp_schema_file)
        hard_gates = parser.get_hard_gates()
        
        assert isinstance(hard_gates, dict)
        assert 'negative_roe' in hard_gates
        assert hard_gates['negative_roe']['condition'] == 'roe < 0'
    
    def test_get_soft_caps(self, temp_schema_file):
        """Test getting soft caps configuration"""
        parser = ScoreSchemaParser(temp_schema_file)
        soft_caps = parser.get_soft_caps()
        
        assert isinstance(soft_caps, dict)
        assert 'high_pe' in soft_caps
        assert soft_caps['high_pe']['condition'] == 'pe_ratio > 100'
    
    def test_get_schema_version(self, temp_schema_file):
        """Test getting schema version"""
        parser = ScoreSchemaParser(temp_schema_file)
        version = parser.get_schema_version()
        
        assert version == '2.0.0'
    
    def test_get_schema_metadata(self, temp_schema_file):
        """Test getting schema metadata"""
        parser = ScoreSchemaParser(temp_schema_file)
        metadata = parser.get_schema_metadata()
        
        assert metadata['schema_version'] == '2.0.0'
        assert metadata['created_date'] == '2025-09-11'
        assert metadata['pillar_count'] == 2
        assert metadata['total_indicators'] == 3
        assert metadata['file_path'] == temp_schema_file
    
    def test_malformed_yaml(self):
        """Test handling of malformed YAML"""
        malformed_yaml = "invalid: yaml: content: [unclosed"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_file = f.name
        
        try:
            with pytest.raises(SchemaValidationError, match="YAML parsing error"):
                ScoreSchemaParser(temp_file)
        finally:
            os.unlink(temp_file)


class TestRealSchemaFile:
    """Test with the actual schema file if it exists"""
    
    @pytest.fixture
    def real_schema_path(self):
        """Get path to the real schema file"""
        return "config/score_schema.yaml"
    
    @pytest.mark.skipif(
        not os.path.exists("config/score_schema.yaml"),
        reason="Real schema file not found"
    )
    def test_real_schema_loads(self, real_schema_path):
        """Test that the real schema file loads successfully"""
        parser = ScoreSchemaParser(real_schema_path)
        assert parser.schema is not None
        assert len(parser.pillars) == 5  # Expected 5 pillars in real schema
    
    @pytest.mark.skipif(
        not os.path.exists("config/score_schema.yaml"),
        reason="Real schema file not found"
    )
    def test_real_schema_weights_sum_to_100(self, real_schema_path):
        """Test that real schema weights sum to 100"""
        parser = ScoreSchemaParser(real_schema_path)
        
        total_weight = sum(
            pillar_config['total_weight'] 
            for pillar_config in parser.pillars.values()
        )
        
        assert abs(total_weight - 100) < 0.01  # Allow small floating point differences
    
    @pytest.mark.skipif(
        not os.path.exists("config/score_schema.yaml"),
        reason="Real schema file not found"
    )
    def test_real_schema_has_all_expected_pillars(self, real_schema_path):
        """Test that real schema has all expected pillars"""
        parser = ScoreSchemaParser(real_schema_path)
        expected_pillars = {'value', 'growth', 'quality', 'momentum', 'risk'}
        actual_pillars = set(parser.pillars.keys())
        
        assert actual_pillars == expected_pillars