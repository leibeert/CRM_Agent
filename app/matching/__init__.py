"""
Enhanced Candidate Matching System

This module provides advanced candidate matching capabilities including:
- Multi-dimensional scoring
- Semantic skill matching
- Market intelligence integration
- Learning from feedback
"""

from .core.scorer import AdvancedCandidateScorer
from .core.semantic_matcher import SemanticSkillMatcher
from .parsers.job_parser import IntelligentJobParser
from .intelligence.market_data import MarketIntelligenceEngine
from .intelligence.feedback_system import MatchingFeedbackSystem

__version__ = "1.0.0"
__all__ = [
    'AdvancedCandidateScorer',
    'SemanticSkillMatcher', 
    'IntelligentJobParser',
    'MarketIntelligenceEngine',
    'MatchingFeedbackSystem'
] 