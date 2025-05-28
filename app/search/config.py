from typing import Dict, Any
from pydantic import BaseSettings

class SearchSettings(BaseSettings):
    """Search configuration settings."""
    # Default pagination settings
    DEFAULT_PAGE_SIZE: int = 10
    MAX_PAGE_SIZE: int = 100
    
    # Search settings
    MIN_MATCH_SCORE: float = 0.0
    MAX_MATCH_SCORE: float = 100.0
    
    # Sorting options
    SORT_OPTIONS: Dict[str, str] = {
        'match_score': 'Match Score',
        'experience': 'Experience',
        'education': 'Education',
        'created_at': 'Date Added'
    }
    
    # Sort orders
    SORT_ORDERS: Dict[str, str] = {
        'asc': 'Ascending',
        'desc': 'Descending'
    }
    
    # Skill levels
    SKILL_LEVELS: Dict[str, str] = {
        'beginner': 'Beginner',
        'intermediate': 'Intermediate',
        'advanced': 'Advanced',
        'expert': 'Expert'
    }
    
    # Degree types
    DEGREE_TYPES: Dict[str, str] = {
        'bachelor': 'Bachelor',
        'master': 'Master',
        'phd': 'PhD',
        'associate': 'Associate',
        'diploma': 'Diploma'
    }
    
    class Config:
        env_prefix = 'SEARCH_'

# Create settings instance
search_settings = SearchSettings() 