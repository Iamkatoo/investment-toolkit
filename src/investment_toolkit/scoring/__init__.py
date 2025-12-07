"""
New Scoring System Module

This module implements the new 5-pillar scoring system with dual normalization
(cross-sectional × time-series) for comprehensive stock evaluation.

Pillars:
- Value (20 points)
- Growth (20 points) 
- Quality (20 points)
- Momentum (20 points)
- Risk (10 points)

Total: 100 points
"""

from .schema_parser import ScoreSchemaParser

__version__ = "2.0.0"
__all__ = ["ScoreSchemaParser"]