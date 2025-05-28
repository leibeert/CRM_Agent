import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from .base_model import BaseMLModel
from app.db import Resource, Experience, Skill, ResourceSkill, Study, DegreeType, School

class CandidateMatchingModel(BaseMLModel):
    """ML model for matching candidates to job descriptions."""
    
    def __init__(self, db_session: Session):
        super().__init__()
        self.db = db_session
        self.nlp = spacy.load('en_core_web_md')
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.experience_embeddings = {}
        self.is_trained = False
    
    def train(self, data: Any = None) -> None:
        """Train the model with experience data."""
        # Get all experiences for training
        experiences = self.db.query(Experience).all()
        experience_texts = [
            f"{exp.title} {exp.company} {exp.description or ''}"
            for exp in experiences
        ]
        
        # Fit TF-IDF on experience texts
        self.tfidf.fit(experience_texts)
        
        # Pre-compute experience embeddings
        self.experience_embeddings = {}
        for exp in experiences:
            doc = self.nlp(f"{exp.title} {exp.company} {exp.description or ''}")
            self.experience_embeddings[exp.id] = doc.vector
        
        self.is_trained = True
    
    def predict(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching candidates based on job description and requirements."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        job_description = input_data.get('job_description', '')
        required_skills = input_data.get('required_skills', [])
        required_education = input_data.get('required_education', {})
        min_match_score = input_data.get('min_match_score', 0.5)
        
        # Get all candidates
        candidates = self.db.query(Resource).all()
        matches = []
        
        for candidate in candidates:
            # Calculate experience similarity
            exp_similarity = self._calculate_experience_similarity(job_description, candidate.experiences)
            
            # Calculate skill match
            skill_match = self._calculate_skill_match(required_skills, candidate.skills)
            
            # Calculate education match
            edu_match = self._calculate_education_match(required_education, candidate.studies)
            
            # Calculate overall match score (weighted)
            match_score = (
                0.5 * exp_similarity +  # Experience is most important
                0.3 * skill_match +     # Skills are second most important
                0.2 * edu_match         # Education is least important
            )
            
            if match_score >= min_match_score:
                matches.append({
                    'id': candidate.id,
                    'first_name': candidate.first_name,
                    'last_name': candidate.last_name,
                    'match_score': match_score,
                    'match_details': self._get_match_details(exp_similarity, skill_match, edu_match)
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model's performance."""
        # This would be implemented with actual test data and metrics
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    def _calculate_experience_similarity(self, job_description: str, candidate_experiences: List[Experience]) -> float:
        """Calculate similarity between job description and candidate experiences."""
        if not candidate_experiences:
            return 0.0
        
        # Get job description embedding
        job_doc = self.nlp(job_description)
        job_vector = job_doc.vector
        
        # Calculate similarities with each experience
        similarities = []
        for exp in candidate_experiences:
            exp_vector = self.experience_embeddings.get(exp.id)
            if exp_vector is not None:
                similarity = cosine_similarity([job_vector], [exp_vector])[0][0]
                # Weight by experience duration
                duration = (exp.end_date or datetime.now()) - exp.start_date
                duration_years = duration.days / 365.25
                weighted_similarity = similarity * min(duration_years / 5, 1.0)  # Cap at 5 years
                similarities.append(weighted_similarity)
        
        return max(similarities) if similarities else 0.0
    
    def _calculate_skill_match(self, required_skills: List[Dict[str, Any]], candidate_skills: List[ResourceSkill]) -> float:
        """Calculate skill match score."""
        if not required_skills:
            return 1.0  # No skills required means perfect match
        if not candidate_skills:
            return 0.0  # No skills means no match
        
        # Convert skills to vectors for comparison
        required_skill_vectors = [self.nlp(skill['name']).vector for skill in required_skills]
        candidate_skill_vectors = [self.nlp(rs.skill.skill_name).vector for rs in candidate_skills]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(required_skill_vectors, candidate_skill_vectors)
        
        # Get best matches for each required skill
        best_matches = np.max(similarity_matrix, axis=1)
        
        # Calculate weighted score based on skill levels and durations
        weighted_scores = []
        for i, skill in enumerate(required_skills):
            if best_matches[i] > 0.7:  # Only consider good matches
                # Find the matching candidate skill
                match_idx = np.argmax(similarity_matrix[i])
                candidate_skill = candidate_skills[match_idx]
                
                # Calculate level score
                level_scores = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
                level_score = level_scores.get(candidate_skill.level.lower(), 0.25)
                
                # Calculate duration score (cap at 5 years)
                duration_score = min(candidate_skill.duration / 60, 1.0) if candidate_skill.duration else 0.25
                
                # Combine scores
                weighted_score = best_matches[i] * (0.4 * level_score + 0.6 * duration_score)
                weighted_scores.append(weighted_score)
        
        return sum(weighted_scores) / len(required_skills) if weighted_scores else 0.0
    
    def _calculate_education_match(self, required_education: Dict[str, Any], candidate_education: List[Study]) -> float:
        """Calculate education match score."""
        if not required_education:
            return 1.0  # No education required means perfect match
        if not candidate_education:
            return 0.0  # No education means no match
        
        best_match_score = 0.0
        for study in candidate_education:
            score = 0.0
            
            # Match degree type
            if required_education.get('degree_type'):
                if study.degree_type and study.degree_type.name.lower() == required_education['degree_type'].lower():
                    score += 0.4
            
            # Match field of study
            if required_education.get('field_of_study'):
                if study.field_of_study and required_education['field_of_study'].lower() in study.field_of_study.lower():
                    score += 0.3
            
            # Match school
            if required_education.get('school'):
                if study.school and required_education['school'].lower() in study.school.school_name.lower():
                    score += 0.3
            
            best_match_score = max(best_match_score, score)
        
        return best_match_score
    
    def _get_match_details(self, exp_similarity: float, skill_match: float, edu_match: float) -> List[str]:
        """Get detailed match information."""
        details = []
        
        if exp_similarity > 0:
            details.append(f"Experience match: {exp_similarity:.1%}")
        
        if skill_match > 0:
            details.append(f"Skills match: {skill_match:.1%}")
        
        if edu_match > 0:
            details.append(f"Education match: {edu_match:.1%}")
        
        return details 