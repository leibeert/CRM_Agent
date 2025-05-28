"""
Advanced candidate scoring system with multi-dimensional analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from .semantic_matcher import SemanticSkillMatcher
from ..utils.config import get_config
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configuration for scoring weights."""
    skills: float = 0.4
    experience: float = 0.3
    education: float = 0.2
    cultural_fit: float = 0.1
    
    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.skills + self.experience + self.education + self.cultural_fit
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class SkillMatch:
    """Represents a skill match between candidate and requirement."""
    required_skill: str
    candidate_skill: str
    similarity_score: float
    proficiency_level: str
    years_experience: int
    is_exact_match: bool
    confidence: float


@dataclass
class ExperienceMatch:
    """Represents experience matching analysis."""
    role_similarity: float
    industry_match: bool
    seniority_match: float
    responsibility_overlap: float
    achievement_relevance: float
    total_years: int
    relevant_years: int


@dataclass
class EducationMatch:
    """Represents education matching analysis."""
    degree_relevance: float
    institution_prestige: float
    field_of_study_match: float
    certification_bonus: float
    continuous_learning: float


@dataclass
class CulturalFitMatch:
    """Represents cultural fit analysis."""
    work_style_match: float
    team_collaboration: float
    communication_style: float
    adaptability: float
    leadership_potential: float


@dataclass
class MatchScore:
    """Complete matching score with detailed breakdown."""
    overall_score: float
    confidence: float
    
    # Component scores
    skills_score: float
    experience_score: float
    education_score: float
    cultural_fit_score: float
    
    # Detailed breakdowns
    skill_matches: List[SkillMatch] = field(default_factory=list)
    experience_match: Optional[ExperienceMatch] = None
    education_match: Optional[EducationMatch] = None
    cultural_fit_match: Optional[CulturalFitMatch] = None
    
    # Explanations
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"


class AdvancedCandidateScorer:
    """Advanced candidate scoring with multi-dimensional analysis."""
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.config = get_config()
        self.cache = get_cache()
        self.semantic_matcher = SemanticSkillMatcher()
        self.weights = weights or ScoringWeights()
        
        # Industry and role mappings
        self.industry_mappings = self._load_industry_mappings()
        self.role_hierarchies = self._load_role_hierarchies()
        
    def score_candidate(self, candidate_data: Dict[str, Any], 
                       job_requirements: Dict[str, Any]) -> MatchScore:
        """Score a candidate against job requirements."""
        
        try:
            # Generate cache key
            candidate_id = candidate_data.get('id', 0)
            job_hash = self._generate_job_hash(job_requirements)
            
            # Check cache first
            cached_score = self.cache.get_candidate_score(candidate_id, job_hash)
            if cached_score:
                return MatchScore(**cached_score)
            
            # Calculate component scores
            skills_score, skill_matches = self._score_skills(candidate_data, job_requirements)
            experience_score, experience_match = self._score_experience(candidate_data, job_requirements)
            education_score, education_match = self._score_education(candidate_data, job_requirements)
            cultural_fit_score, cultural_fit_match = self._score_cultural_fit(candidate_data, job_requirements)
            
            # Calculate overall score
            overall_score = (
                skills_score * self.weights.skills +
                experience_score * self.weights.experience +
                education_score * self.weights.education +
                cultural_fit_score * self.weights.cultural_fit
            )
            
            # Calculate confidence based on data completeness and match quality
            confidence = self._calculate_confidence(
                candidate_data, job_requirements, skill_matches
            )
            
            # Generate explanations
            strengths, weaknesses, recommendations = self._generate_explanations(
                skills_score, experience_score, education_score, cultural_fit_score,
                skill_matches, experience_match, education_match, cultural_fit_match
            )
            
            # Create match score object
            match_score = MatchScore(
                overall_score=overall_score,
                confidence=confidence,
                skills_score=skills_score,
                experience_score=experience_score,
                education_score=education_score,
                cultural_fit_score=cultural_fit_score,
                skill_matches=skill_matches,
                experience_match=experience_match,
                education_match=education_match,
                cultural_fit_match=cultural_fit_match,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations
            )
            
            # Cache the result
            self.cache.cache_candidate_score(candidate_id, job_hash, match_score.__dict__)
            
            return match_score
            
        except Exception as e:
            logger.error(f"Error scoring candidate {candidate_data.get('id', 'unknown')}: {str(e)}")
            # Return minimal score on error
            return MatchScore(
                overall_score=0.0,
                confidence=0.0,
                skills_score=0.0,
                experience_score=0.0,
                education_score=0.0,
                cultural_fit_score=0.0,
                weaknesses=["Error occurred during scoring"],
                recommendations=["Please review candidate data and try again"]
            )
    
    def _score_skills(self, candidate_data: Dict[str, Any], 
                     job_requirements: Dict[str, Any]) -> Tuple[float, List[SkillMatch]]:
        """Score candidate skills against job requirements."""
        
        required_skills = job_requirements.get('required_skills', [])
        preferred_skills = job_requirements.get('preferred_skills', [])
        candidate_skills = self._extract_candidate_skills(candidate_data)
        
        if not required_skills and not preferred_skills:
            return 0.5, []  # Neutral score if no requirements
        
        skill_matches = []
        total_score = 0.0
        total_weight = 0.0
        
        # Score required skills (higher weight)
        for req_skill in required_skills:
            best_match = self._find_best_skill_match(req_skill, candidate_skills)
            if best_match:
                skill_matches.append(best_match)
                total_score += best_match.similarity_score * 1.0  # Full weight for required
                total_weight += 1.0
            else:
                # Missing required skill
                skill_matches.append(SkillMatch(
                    required_skill=req_skill,
                    candidate_skill="",
                    similarity_score=0.0,
                    proficiency_level="none",
                    years_experience=0,
                    is_exact_match=False,
                    confidence=1.0
                ))
                total_weight += 1.0
        
        # Score preferred skills (lower weight)
        for pref_skill in preferred_skills:
            best_match = self._find_best_skill_match(pref_skill, candidate_skills)
            if best_match:
                skill_matches.append(best_match)
                total_score += best_match.similarity_score * 0.5  # Half weight for preferred
                total_weight += 0.5
        
        # Calculate final skills score
        skills_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Bonus for additional relevant skills
        additional_skills_bonus = self._calculate_additional_skills_bonus(
            candidate_skills, required_skills + preferred_skills
        )
        skills_score = min(1.0, skills_score + additional_skills_bonus)
        
        return skills_score, skill_matches
    
    def _score_experience(self, candidate_data: Dict[str, Any], 
                         job_requirements: Dict[str, Any]) -> Tuple[float, ExperienceMatch]:
        """Score candidate experience against job requirements."""
        
        experiences = candidate_data.get('experiences', [])
        required_experience = job_requirements.get('experience_requirements', {})
        
        if not experiences:
            return 0.0, ExperienceMatch(0, False, 0, 0, 0, 0, 0)
        
        # Calculate experience metrics
        total_years = self._calculate_total_experience(experiences)
        relevant_years = self._calculate_relevant_experience(experiences, job_requirements)
        
        # Role similarity analysis
        role_similarity = self._calculate_role_similarity(experiences, job_requirements)
        
        # Industry match
        industry_match = self._check_industry_match(experiences, job_requirements)
        
        # Seniority level match
        seniority_match = self._calculate_seniority_match(experiences, job_requirements)
        
        # Responsibility overlap
        responsibility_overlap = self._calculate_responsibility_overlap(experiences, job_requirements)
        
        # Achievement relevance
        achievement_relevance = self._calculate_achievement_relevance(experiences, job_requirements)
        
        # Calculate overall experience score
        experience_score = (
            role_similarity * 0.3 +
            (1.0 if industry_match else 0.5) * 0.2 +
            seniority_match * 0.2 +
            responsibility_overlap * 0.2 +
            achievement_relevance * 0.1
        )
        
        # Apply experience length modifier
        min_years = required_experience.get('minimum_years', 0)
        if min_years > 0:
            years_modifier = min(1.0, relevant_years / min_years)
            experience_score *= years_modifier
        
        experience_match = ExperienceMatch(
            role_similarity=role_similarity,
            industry_match=industry_match,
            seniority_match=seniority_match,
            responsibility_overlap=responsibility_overlap,
            achievement_relevance=achievement_relevance,
            total_years=total_years,
            relevant_years=relevant_years
        )
        
        return experience_score, experience_match
    
    def _score_education(self, candidate_data: Dict[str, Any], 
                        job_requirements: Dict[str, Any]) -> Tuple[float, EducationMatch]:
        """Score candidate education against job requirements."""
        
        education = candidate_data.get('education', [])
        certifications = candidate_data.get('certifications', [])
        education_requirements = job_requirements.get('education_requirements', {})
        
        if not education and not certifications:
            return 0.3, EducationMatch(0, 0, 0, 0, 0)  # Minimal score for missing education
        
        # Degree relevance
        degree_relevance = self._calculate_degree_relevance(education, education_requirements)
        
        # Institution prestige (simplified)
        institution_prestige = self._calculate_institution_prestige(education)
        
        # Field of study match
        field_match = self._calculate_field_of_study_match(education, job_requirements)
        
        # Certification bonus
        cert_bonus = self._calculate_certification_bonus(certifications, job_requirements)
        
        # Continuous learning indicator
        continuous_learning = self._calculate_continuous_learning(education, certifications)
        
        # Calculate overall education score
        education_score = (
            degree_relevance * 0.4 +
            institution_prestige * 0.2 +
            field_match * 0.3 +
            continuous_learning * 0.1
        )
        
        # Add certification bonus
        education_score = min(1.0, education_score + cert_bonus)
        
        education_match = EducationMatch(
            degree_relevance=degree_relevance,
            institution_prestige=institution_prestige,
            field_of_study_match=field_match,
            certification_bonus=cert_bonus,
            continuous_learning=continuous_learning
        )
        
        return education_score, education_match
    
    def _score_cultural_fit(self, candidate_data: Dict[str, Any], 
                           job_requirements: Dict[str, Any]) -> Tuple[float, CulturalFitMatch]:
        """Score cultural fit based on available data."""
        
        # This is a simplified implementation - in practice, you'd use
        # personality assessments, interview feedback, etc.
        
        competencies = candidate_data.get('competencies', [])
        work_preferences = candidate_data.get('work_preferences', {})
        company_culture = job_requirements.get('company_culture', {})
        
        # Work style match (remote, office, hybrid preferences)
        work_style_match = self._calculate_work_style_match(work_preferences, company_culture)
        
        # Team collaboration indicators
        team_collaboration = self._assess_team_collaboration(competencies)
        
        # Communication style
        communication_style = self._assess_communication_style(competencies)
        
        # Adaptability
        adaptability = self._assess_adaptability(competencies, candidate_data.get('experiences', []))
        
        # Leadership potential
        leadership_potential = self._assess_leadership_potential(competencies, candidate_data.get('experiences', []))
        
        # Calculate overall cultural fit score
        cultural_fit_score = (
            work_style_match * 0.3 +
            team_collaboration * 0.25 +
            communication_style * 0.2 +
            adaptability * 0.15 +
            leadership_potential * 0.1
        )
        
        cultural_fit_match = CulturalFitMatch(
            work_style_match=work_style_match,
            team_collaboration=team_collaboration,
            communication_style=communication_style,
            adaptability=adaptability,
            leadership_potential=leadership_potential
        )
        
        return cultural_fit_score, cultural_fit_match
    
    def _extract_candidate_skills(self, candidate_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and normalize candidate skills from various sources."""
        
        skills = []
        
        # Direct skills
        if 'skills' in candidate_data:
            for skill in candidate_data['skills']:
                if isinstance(skill, str):
                    skills.append({
                        'name': skill,
                        'proficiency': 'intermediate',
                        'years_experience': 1
                    })
                else:
                    skills.append(skill)
        
        # Skills from experiences
        for exp in candidate_data.get('experiences', []):
            for skill in exp.get('technologies_used', []):
                skills.append({
                    'name': skill,
                    'proficiency': 'intermediate',
                    'years_experience': exp.get('duration_years', 1)
                })
        
        # Skills from projects
        for project in candidate_data.get('portfolio', []):
            for tech in project.get('technologies_used', []):
                skills.append({
                    'name': tech,
                    'proficiency': 'intermediate',
                    'years_experience': 1
                })
        
        return skills
    
    def _find_best_skill_match(self, required_skill: str, 
                              candidate_skills: List[Dict[str, Any]]) -> Optional[SkillMatch]:
        """Find the best matching skill from candidate's skills."""
        
        best_match = None
        best_similarity = 0.0
        
        for skill in candidate_skills:
            skill_name = skill.get('name', '')
            similarity = self.semantic_matcher.calculate_similarity(required_skill, skill_name)
            
            if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                best_similarity = similarity
                best_match = SkillMatch(
                    required_skill=required_skill,
                    candidate_skill=skill_name,
                    similarity_score=similarity,
                    proficiency_level=skill.get('proficiency', 'intermediate'),
                    years_experience=skill.get('years_experience', 1),
                    is_exact_match=(similarity >= 0.95),
                    confidence=min(1.0, similarity + 0.1)
                )
        
        return best_match
    
    def _generate_job_hash(self, job_requirements: Dict[str, Any]) -> str:
        """Generate a hash for job requirements for caching."""
        import hashlib
        import json
        
        # Create a normalized representation
        normalized = {
            'required_skills': sorted(job_requirements.get('required_skills', [])),
            'preferred_skills': sorted(job_requirements.get('preferred_skills', [])),
            'experience_requirements': job_requirements.get('experience_requirements', {}),
            'education_requirements': job_requirements.get('education_requirements', {})
        }
        
        job_str = json.dumps(normalized, sort_keys=True)
        return hashlib.md5(job_str.encode()).hexdigest()
    
    def _calculate_confidence(self, candidate_data: Dict[str, Any], 
                            job_requirements: Dict[str, Any], 
                            skill_matches: List[SkillMatch]) -> float:
        """Calculate confidence score based on data completeness and match quality."""
        
        confidence_factors = []
        
        # Data completeness
        has_skills = bool(candidate_data.get('skills') or candidate_data.get('experiences'))
        has_experience = bool(candidate_data.get('experiences'))
        has_education = bool(candidate_data.get('education'))
        
        completeness = (has_skills + has_experience + has_education) / 3.0
        confidence_factors.append(completeness)
        
        # Match quality
        if skill_matches:
            avg_match_confidence = sum(match.confidence for match in skill_matches) / len(skill_matches)
            confidence_factors.append(avg_match_confidence)
        
        # Requirements clarity
        req_clarity = 1.0 if job_requirements.get('required_skills') else 0.5
        confidence_factors.append(req_clarity)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    # Additional helper methods would be implemented here...
    # For brevity, I'm including stubs for the remaining methods
    
    def _calculate_additional_skills_bonus(self, candidate_skills, required_skills):
        """Calculate bonus for additional relevant skills."""
        return 0.0  # Stub implementation
    
    def _calculate_total_experience(self, experiences):
        """Calculate total years of experience."""
        return sum(exp.get('duration_years', 0) for exp in experiences)
    
    def _calculate_relevant_experience(self, experiences, job_requirements):
        """Calculate years of relevant experience."""
        return sum(exp.get('duration_years', 0) for exp in experiences)  # Simplified
    
    def _calculate_role_similarity(self, experiences, job_requirements):
        """Calculate similarity between past roles and target role."""
        return 0.7  # Stub implementation
    
    def _check_industry_match(self, experiences, job_requirements):
        """Check if candidate has industry experience."""
        return True  # Stub implementation
    
    def _calculate_seniority_match(self, experiences, job_requirements):
        """Calculate seniority level match."""
        return 0.8  # Stub implementation
    
    def _calculate_responsibility_overlap(self, experiences, job_requirements):
        """Calculate overlap in responsibilities."""
        return 0.6  # Stub implementation
    
    def _calculate_achievement_relevance(self, experiences, job_requirements):
        """Calculate relevance of achievements."""
        return 0.5  # Stub implementation
    
    def _calculate_degree_relevance(self, education, education_requirements):
        """Calculate relevance of degree to job."""
        return 0.7  # Stub implementation
    
    def _calculate_institution_prestige(self, education):
        """Calculate institution prestige score."""
        return 0.6  # Stub implementation
    
    def _calculate_field_of_study_match(self, education, job_requirements):
        """Calculate field of study match."""
        return 0.8  # Stub implementation
    
    def _calculate_certification_bonus(self, certifications, job_requirements):
        """Calculate bonus for relevant certifications."""
        return 0.1  # Stub implementation
    
    def _calculate_continuous_learning(self, education, certifications):
        """Calculate continuous learning indicator."""
        return 0.7  # Stub implementation
    
    def _calculate_work_style_match(self, work_preferences, company_culture):
        """Calculate work style compatibility."""
        return 0.8  # Stub implementation
    
    def _assess_team_collaboration(self, competencies):
        """Assess team collaboration skills."""
        return 0.7  # Stub implementation
    
    def _assess_communication_style(self, competencies):
        """Assess communication style fit."""
        return 0.8  # Stub implementation
    
    def _assess_adaptability(self, competencies, experiences):
        """Assess adaptability."""
        return 0.6  # Stub implementation
    
    def _assess_leadership_potential(self, competencies, experiences):
        """Assess leadership potential."""
        return 0.5  # Stub implementation
    
    def _load_industry_mappings(self):
        """Load industry mapping data."""
        return {}  # Stub implementation
    
    def _load_role_hierarchies(self):
        """Load role hierarchy data."""
        return {}  # Stub implementation
    
    def _generate_explanations(self, skills_score, experience_score, education_score, 
                             cultural_fit_score, skill_matches, experience_match, 
                             education_match, cultural_fit_match):
        """Generate human-readable explanations."""
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Analyze skills
        if skills_score >= 0.8:
            strengths.append("Strong technical skill alignment with job requirements")
        elif skills_score >= 0.6:
            strengths.append("Good technical skill match with some gaps")
        else:
            weaknesses.append("Significant gaps in required technical skills")
            recommendations.append("Consider additional training in key technical areas")
        
        # Analyze experience
        if experience_score >= 0.8:
            strengths.append("Excellent relevant work experience")
        elif experience_score >= 0.6:
            strengths.append("Good relevant experience with transferable skills")
        else:
            weaknesses.append("Limited relevant work experience")
            recommendations.append("Look for transferable skills and growth potential")
        
        # Analyze education
        if education_score >= 0.7:
            strengths.append("Strong educational background")
        elif education_score < 0.4:
            weaknesses.append("Educational background may not align with requirements")
        
        # Analyze cultural fit
        if cultural_fit_score >= 0.8:
            strengths.append("Excellent cultural fit indicators")
        elif cultural_fit_score < 0.5:
            weaknesses.append("Potential cultural fit concerns")
            recommendations.append("Conduct behavioral interview to assess fit")
        
        return strengths, weaknesses, recommendations 