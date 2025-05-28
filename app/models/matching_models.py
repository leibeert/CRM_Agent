"""
Database models for the enhanced candidate matching system.
"""

from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db import AuthBase, ArgoteamBase


class SkillEmbedding(ArgoteamBase):
    """Store skill embeddings for semantic matching."""
    __tablename__ = 'skill_embeddings'
    
    id = Column(Integer, primary_key=True)
    skill_name = Column(String(100), unique=True, index=True)
    embedding_vector = Column(JSON)  # Store as JSON array
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SkillRelationship(ArgoteamBase):
    """Store relationships between skills for better matching."""
    __tablename__ = 'skill_relationships'
    
    id = Column(Integer, primary_key=True)
    parent_skill = Column(String(100), index=True)
    child_skill = Column(String(100), index=True)
    relationship_type = Column(String(50))  # 'similar', 'prerequisite', 'related', 'alternative'
    similarity_score = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MatchingFeedback(AuthBase):
    """Store feedback on matching accuracy for learning."""
    __tablename__ = 'matching_feedback'
    
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, index=True)
    job_description_hash = Column(String(64), index=True)
    predicted_score = Column(Float)
    actual_outcome = Column(String(20))  # 'hired', 'rejected', 'withdrawn'
    performance_rating = Column(Float)  # 1-5 scale
    time_to_hire = Column(Integer)  # days
    recruiter_feedback = Column(Text)
    feedback_date = Column(DateTime, default=datetime.utcnow)
    features = Column(JSON)  # Store feature vector used for prediction
    
    # Relationships
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship('User')


class MarketIntelligence(ArgoteamBase):
    """Store market intelligence data for skills."""
    __tablename__ = 'market_intelligence'
    
    id = Column(Integer, primary_key=True)
    skill_name = Column(String(100), index=True)
    demand_score = Column(Float)  # 0-1 scale
    salary_impact = Column(Float)  # percentage impact on salary
    trend_direction = Column(String(20))  # 'rising', 'stable', 'declining'
    competition_level = Column(String(20))  # 'low', 'medium', 'high'
    availability_score = Column(Float)  # 0-1 scale
    location = Column(String(100))
    data_source = Column(String(50))
    last_updated = Column(DateTime, default=datetime.utcnow)


class CandidateCompetency(ArgoteamBase):
    """Store soft skills and competencies for candidates."""
    __tablename__ = 'candidate_competencies'
    
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('resources.id'), index=True)
    competency_type = Column(String(50))  # 'leadership', 'communication', 'problem_solving', etc.
    proficiency_level = Column(String(20))  # 'beginner', 'intermediate', 'advanced', 'expert'
    evidence_source = Column(String(30))  # 'self_reported', 'reference', 'assessment', 'inferred'
    confidence_score = Column(Float)  # 0-1 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    candidate = relationship('Resource')


class CandidatePortfolio(ArgoteamBase):
    """Store portfolio and work samples for candidates."""
    __tablename__ = 'candidate_portfolio'
    
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey('resources.id'), index=True)
    project_name = Column(String(255))
    project_url = Column(String(500))
    technologies_used = Column(JSON)  # Array of technologies
    project_type = Column(String(30))  # 'personal', 'professional', 'open_source', 'academic'
    impact_description = Column(Text)
    complexity_score = Column(Float)  # 0-1 scale
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    candidate = relationship('Resource')


class ExperienceAchievement(ArgoteamBase):
    """Store achievements and accomplishments for experiences."""
    __tablename__ = 'experience_achievements'
    
    id = Column(Integer, primary_key=True)
    experience_id = Column(Integer, ForeignKey('experiences.id'), index=True)
    achievement_type = Column(String(30))  # 'leadership', 'innovation', 'performance', 'recognition'
    description = Column(Text)
    impact_metrics = Column(JSON)  # Store quantifiable impact data
    verification_status = Column(String(20))  # 'verified', 'unverified', 'disputed'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experience = relationship('Experience')


class JobDescriptionAnalysis(AuthBase):
    """Store parsed job description analysis."""
    __tablename__ = 'job_description_analysis'
    
    id = Column(Integer, primary_key=True)
    job_description_hash = Column(String(64), unique=True, index=True)
    original_text = Column(Text)
    parsed_title = Column(String(255))
    seniority_level = Column(String(20))
    required_skills = Column(JSON)  # Array of skill requirements
    preferred_skills = Column(JSON)  # Array of preferred skills
    responsibilities = Column(JSON)  # Array of responsibilities
    company_culture = Column(JSON)  # Array of culture indicators
    work_environment = Column(String(50))
    team_size_estimate = Column(String(20))
    salary_range = Column(JSON)  # {min: x, max: y, currency: 'USD'}
    location_requirements = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship('User')


class MatchingConfiguration(AuthBase):
    """Store user-specific matching configurations."""
    __tablename__ = 'matching_configurations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, index=True)
    scoring_weights = Column(JSON)  # Custom scoring weights
    similarity_threshold = Column(Float, default=0.7)
    market_intelligence_enabled = Column(Boolean, default=True)
    learning_enabled = Column(Boolean, default=True)
    explanation_detail_level = Column(String(20), default='detailed')  # 'brief', 'detailed', 'comprehensive'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User') 