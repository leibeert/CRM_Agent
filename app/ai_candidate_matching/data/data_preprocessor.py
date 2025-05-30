"""
Data Preprocessor for AI Candidate Matching

This module provides data preprocessing utilities for the ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses data for ML model training and inference.
    """
    
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.is_fitted = False
    
    def clean_candidate_data(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean and standardize candidate data.
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            Cleaned candidate data
        """
        cleaned_candidates = []
        
        for candidate in candidates:
            cleaned = self._clean_single_candidate(candidate)
            if cleaned:
                cleaned_candidates.append(cleaned)
        
        logger.info(f"Cleaned {len(cleaned_candidates)} candidates from {len(candidates)} total")
        return cleaned_candidates
    
    def _clean_single_candidate(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean a single candidate record."""
        try:
            # Ensure required fields exist
            cleaned = {
                'id': candidate.get('id', ''),
                'first_name': str(candidate.get('first_name', '')).strip(),
                'last_name': str(candidate.get('last_name', '')).strip(),
                'email': str(candidate.get('email', '')).strip().lower(),
                'title': str(candidate.get('title', '')).strip(),
                'years_of_experience': self._clean_experience(candidate.get('years_of_experience', 0)),
                'custom_description': str(candidate.get('custom_description', '')).strip(),
                'description': str(candidate.get('description', '')).strip(),
                'address': str(candidate.get('address', '')).strip(),
                'phone': str(candidate.get('phone', '')).strip(),
                'skills': candidate.get('skills', []),
                'experience': candidate.get('experience', []),
                'education': candidate.get('education', [])
            }
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning candidate {candidate.get('id', 'unknown')}: {e}")
            return None
    
    def _clean_experience(self, experience: Any) -> int:
        """Clean experience value."""
        try:
            if isinstance(experience, (int, float)):
                return max(0, int(experience))
            elif isinstance(experience, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'\d+', experience)
                if numbers:
                    return max(0, int(numbers[0]))
            return 0
        except:
            return 0
    
    def validate_data(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate data quality and completeness.
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            Validation report
        """
        report = {
            'total_candidates': len(candidates),
            'complete_profiles': 0,
            'missing_fields': {},
            'data_quality_issues': []
        }
        
        required_fields = ['id', 'first_name', 'last_name', 'email']
        
        for candidate in candidates:
            is_complete = True
            
            for field in required_fields:
                if not candidate.get(field):
                    is_complete = False
                    if field not in report['missing_fields']:
                        report['missing_fields'][field] = 0
                    report['missing_fields'][field] += 1
            
            if is_complete:
                report['complete_profiles'] += 1
        
        # Check data quality
        if report['complete_profiles'] < len(candidates) * 0.8:
            report['data_quality_issues'].append("More than 20% of profiles have missing required fields")
        
        return report 