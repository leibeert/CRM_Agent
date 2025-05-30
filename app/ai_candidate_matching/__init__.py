"""
AI-Powered Candidate Matching System

This package provides machine learning models and services for intelligent 
candidate-job matching based on skills, experience, and job requirements.

Authors: CRM AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "CRM AI Team"

from .core.feature_extractor import FeatureExtractor
from .core.ml_models import CandidateJobMatcher
from .services.matching_service import MLMatchingService

__all__ = [
    "FeatureExtractor",
    "CandidateJobMatcher", 
    "MLMatchingService"
] 