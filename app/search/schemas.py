from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SkillFilter(BaseModel):
    """Schema for skill-based filtering."""
    skill_name: str
    min_level: Optional[str] = None
    min_duration: Optional[int] = None  # in months

class ExperienceFilter(BaseModel):
    """Schema for experience-based filtering."""
    title: Optional[str] = None
    company: Optional[str] = None
    min_years: Optional[int] = None  # Experience duration in years

class EducationFilter(BaseModel):
    """Schema for education-based filtering."""
    degree_type: Optional[str] = None
    field_of_study: Optional[str] = None
    school: Optional[str] = None

class SearchQuery(BaseModel):
    """Schema for search query parameters."""
    keywords: Optional[str] = None
    skills: List[SkillFilter] = Field(default_factory=list)
    experience: Optional[ExperienceFilter] = None
    education: Optional[EducationFilter] = None
    location: Optional[str] = None
    min_match_score: Optional[float] = 0.0
    sort_by: str = 'match_score'
    sort_order: str = 'desc'
    page: int = 1
    page_size: int = 10

class CandidateResponse(BaseModel):
    """Schema for candidate search results."""
    id: int
    first_name: Optional[str] = None  # Allow NULL values
    last_name: Optional[str] = None   # Allow NULL values
    email: Optional[str] = None       # Allow NULL values
    phone_number: Optional[str] = None
    match_score: float
    skills: List[Dict[str, Any]]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    """Schema for search response."""
    candidates: List[CandidateResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

class SavedSearchCreate(BaseModel):
    """Schema for creating a saved search."""
    name: str
    description: Optional[str] = None
    filters: SearchQuery
    sort_by: str = 'match_score'
    sort_order: str = 'desc'

class SavedSearchResponse(BaseModel):
    """Schema for saved search response."""
    id: int
    name: str
    description: Optional[str]
    filters: SearchQuery
    sort_by: str
    sort_order: str
    created_at: datetime
    updated_at: datetime 