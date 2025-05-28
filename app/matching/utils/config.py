"""
Configuration settings for the enhanced matching system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MatchingConfig:
    """Configuration class for matching system settings."""
    
    # Model configurations
    embedding_model: str = 'all-MiniLM-L6-v2'
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # Scoring weights (default values)
    default_skill_weight: float = 0.4
    default_experience_weight: float = 0.3
    default_education_weight: float = 0.2
    default_cultural_weight: float = 0.1
    
    # Cache settings
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', 6379))
    redis_db: int = int(os.getenv('REDIS_DB', 0))
    cache_ttl: int = 3600  # 1 hour
    
    # API keys and external services
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    salary_api_key: str = os.getenv('SALARY_API_KEY', '')
    job_trends_api_key: str = os.getenv('JOB_TRENDS_API_KEY', '')
    
    # Learning parameters
    min_feedback_samples: int = 50
    model_update_frequency: int = 100
    learning_rate: float = 0.01
    
    # Performance settings
    max_candidates_per_batch: int = 100
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # Skill relationship settings
    max_skill_relationships: int = 1000
    relationship_confidence_threshold: float = 0.8
    
    # Market intelligence settings
    market_data_refresh_hours: int = 24
    trend_analysis_days: int = 30
    
    @classmethod
    def from_env(cls) -> 'MatchingConfig':
        """Create configuration from environment variables."""
        return cls(
            embedding_model=os.getenv('EMBEDDING_MODEL', cls.embedding_model),
            similarity_threshold=float(os.getenv('SIMILARITY_THRESHOLD', cls.similarity_threshold)),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', cls.confidence_threshold)),
            default_skill_weight=float(os.getenv('SKILL_WEIGHT', cls.default_skill_weight)),
            default_experience_weight=float(os.getenv('EXPERIENCE_WEIGHT', cls.default_experience_weight)),
            default_education_weight=float(os.getenv('EDUCATION_WEIGHT', cls.default_education_weight)),
            default_cultural_weight=float(os.getenv('CULTURAL_WEIGHT', cls.default_cultural_weight)),
        )
    
    def get_scoring_weights(self) -> Dict[str, float]:
        """Get default scoring weights as dictionary."""
        return {
            'skills': self.default_skill_weight,
            'experience': self.default_experience_weight,
            'education': self.default_education_weight,
            'cultural_fit': self.default_cultural_weight
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Check that weights sum to 1.0
        total_weight = (
            self.default_skill_weight + 
            self.default_experience_weight + 
            self.default_education_weight + 
            self.default_cultural_weight
        )
        
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
        
        # Check threshold ranges
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # Check required API keys
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        return True


# Global configuration instance
config = MatchingConfig.from_env()


def get_config() -> MatchingConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration settings."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    
    config.validate() 