"""
Market intelligence and learning components for adaptive matching.
"""

from .market_data import MarketIntelligenceEngine
from .feedback_system import MatchingFeedbackSystem
from .trend_analyzer import SkillTrendAnalyzer

__all__ = [
    'MarketIntelligenceEngine',
    'MatchingFeedbackSystem',
    'SkillTrendAnalyzer'
] 