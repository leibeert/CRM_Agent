"""Utility functions for search operations."""

from typing import List, Dict, Any
from sqlalchemy import or_, and_, func
from db import Resource, Skill, ResourceSkill, Experience, Study

def build_skill_filter(skill_name: str, min_level: str = None, min_duration: int = None):
    """Build a filter for skills."""
    filters = [Skill.skill_name.ilike(f"%{skill_name}%")]
    if min_level:
        filters.append(ResourceSkill.level >= min_level)
    if min_duration:
        filters.append(ResourceSkill.duration >= min_duration)
    return and_(*filters)

def build_experience_filter(title: str = None, company: str = None, min_years: int = None):
    """Build a filter for experience."""
    filters = []
    if title:
        filters.append(Experience.title.ilike(f"%{title}%"))
    if company:
        filters.append(Experience.company.ilike(f"%{company}%"))
    if min_years:
        filters.append(
            func.extract('year', func.now()) - func.extract('year', Experience.start_date) >= min_years
        )
    return and_(*filters) if filters else None

def build_education_filter(degree_type: str = None, field_of_study: str = None, min_graduation_year: int = None):
    """Build a filter for education."""
    filters = []
    if degree_type:
        filters.append(Study.degree_type_id == degree_type)
    if field_of_study:
        filters.append(Study.field_of_study.ilike(f"%{field_of_study}%"))
    if min_graduation_year:
        filters.append(
            func.extract('year', Study.end_date) >= min_graduation_year
        )
    return and_(*filters) if filters else None

def format_candidate_response(candidate: Resource, match_score: float) -> Dict[str, Any]:
    """Format a candidate response with all related data."""
    return {
        'id': candidate.id,
        'first_name': candidate.first_name,
        'last_name': candidate.last_name,
        'email': candidate.email,
        'phone_number': candidate.phone_number,
        'match_score': match_score,
        'skills': [{
            'name': rs.skill.skill_name,
            'level': rs.level,
            'duration': rs.duration
        } for rs in candidate.skills],
        'experience': [{
            'title': exp.title,
            'company': exp.company,
            'start_date': exp.start_date,
            'end_date': exp.end_date,
            'description': exp.description
        } for exp in candidate.experiences],
        'education': [{
            'degree_type': study.degree_type.name if study.degree_type else None,
            'field_of_study': study.field_of_study,
            'school': study.school.school_name if study.school else None,
            'start_date': study.start_date,
            'end_date': study.end_date
        } for study in candidate.studies]
    } 