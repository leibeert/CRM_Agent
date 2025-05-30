"""
Feature Extractor for ML-based Candidate Matching

This module extracts numerical features from job descriptions and candidate profiles
for machine learning models.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from job descriptions and candidate profiles for ML models.
    """
    
    def __init__(self):
        self.skill_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.description_vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.title_encoder = LabelEncoder()
        
        # Skill categories mapping
        self.skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift', 'kotlin', 'r', 'typescript', 'scala', 'rust'],
            'web_frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'rails'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra', 'sqlite'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy'],
            'devops': ['jenkins', 'git', 'docker', 'kubernetes', 'terraform', 'ansible'],
            'business_skills': ['project management', 'agile', 'scrum', 'business analysis'],
            'finance': ['murex', 'calypso', 'risk management', 'trading', 'derivatives']
        }
        
        # Experience level mapping
        self.experience_levels = {
            'entry': (0, 2),
            'junior': (1, 3),
            'mid': (3, 6),
            'senior': (6, 10),
            'lead': (8, 15),
            'expert': (10, 25)
        }
        
        self.is_fitted = False
        
    def extract_job_features(self, job_description: str, job_title: str = "", 
                           required_experience: int = 0, location: str = "",
                           salary_range: Tuple[int, int] = None) -> np.ndarray:
        """
        Extract features from a job description.
        
        Args:
            job_description: Raw job description text
            job_title: Job title
            required_experience: Required years of experience
            location: Job location
            salary_range: Salary range tuple (min, max)
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # 1. Text-based features from job description
        description_features = self._extract_text_features(job_description)
        features.extend(description_features)
        
        # 2. Skill-based features
        skill_features = self._extract_skill_features(job_description + " " + job_title)
        features.extend(skill_features)
        
        # 3. Experience features
        exp_features = self._extract_experience_features(required_experience)
        features.extend(exp_features)
        
        # 4. Job title features
        title_features = self._extract_title_features(job_title)
        features.extend(title_features)
        
        # 5. Location features
        location_features = self._extract_location_features(location)
        features.extend(location_features)
        
        # 6. Salary features
        salary_features = self._extract_salary_features(salary_range)
        features.extend(salary_features)
        
        # 7. Job complexity features
        complexity_features = self._extract_job_complexity_features(job_description)
        features.extend(complexity_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_candidate_features(self, candidate_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from candidate profile.
        
        Args:
            candidate_data: Dictionary containing candidate information
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # 1. Skills features
        candidate_skills = self._get_candidate_skills(candidate_data)
        skill_text = " ".join(candidate_skills)
        skill_features = self._extract_skill_features(skill_text)
        features.extend(skill_features)
        
        # 2. Experience features
        total_experience = self._calculate_total_experience(candidate_data)
        exp_features = self._extract_experience_features(total_experience)
        features.extend(exp_features)
        
        # 3. Education features
        education_features = self._extract_education_features(candidate_data)
        features.extend(education_features)
        
        # 4. Title/Role features
        current_title = candidate_data.get('title', '')
        title_features = self._extract_title_features(current_title)
        features.extend(title_features)
        
        # 5. Location features
        location = candidate_data.get('address', '')
        location_features = self._extract_location_features(location)
        features.extend(location_features)
        
        # 6. Career progression features
        career_features = self._extract_career_progression_features(candidate_data)
        features.extend(career_features)
        
        # 7. Skill diversity features
        diversity_features = self._extract_skill_diversity_features(candidate_skills)
        features.extend(diversity_features)
        
        # 8. Description-based features
        description = candidate_data.get('custom_description', '') or candidate_data.get('description', '')
        description_features = self._extract_text_features(description)
        features.extend(description_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract TF-IDF features from text."""
        if not text or not text.strip():
            return [0.0] * 300  # Return zeros if no text
            
        try:
            if self.is_fitted:
                tfidf_features = self.description_vectorizer.transform([text]).toarray()[0]
            else:
                # For single text, create a simple feature representation
                words = text.lower().split()
                # Simple word count features
                feature_words = ['python', 'java', 'javascript', 'react', 'angular', 'spring', 'mysql', 'aws', 'docker', 'git']
                tfidf_features = [words.count(word) for word in feature_words]
                tfidf_features.extend([0.0] * (300 - len(tfidf_features)))
                tfidf_features = tfidf_features[:300]
                
            return tfidf_features.tolist()
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return [0.0] * 300
    
    def _extract_skill_features(self, text: str) -> List[float]:
        """Extract skill-based features."""
        text_lower = text.lower()
        features = []
        
        # Count skills by category
        for category, skills in self.skill_categories.items():
            category_count = sum(1 for skill in skills if skill in text_lower)
            category_ratio = category_count / len(skills) if skills else 0
            features.extend([category_count, category_ratio])
        
        # Total unique skills mentioned
        all_skills = [skill for skills in self.skill_categories.values() for skill in skills]
        total_skills = sum(1 for skill in all_skills if skill in text_lower)
        features.append(total_skills)
        
        # Skill diversity (number of categories with skills)
        categories_with_skills = sum(1 for category, skills in self.skill_categories.items() 
                                   if any(skill in text_lower for skill in skills))
        features.append(categories_with_skills)
        
        return features
    
    def _extract_experience_features(self, years_experience: int) -> List[float]:
        """Extract experience-related features."""
        features = []
        
        # Raw experience
        features.append(float(years_experience))
        
        # Experience level encoding
        exp_level = self._get_experience_level(years_experience)
        level_encoding = [0.0] * len(self.experience_levels)
        if exp_level in self.experience_levels:
            level_idx = list(self.experience_levels.keys()).index(exp_level)
            level_encoding[level_idx] = 1.0
        features.extend(level_encoding)
        
        # Experience bins
        features.extend([
            1.0 if years_experience <= 2 else 0.0,  # Entry level
            1.0 if 2 < years_experience <= 5 else 0.0,  # Junior
            1.0 if 5 < years_experience <= 10 else 0.0,  # Mid
            1.0 if years_experience > 10 else 0.0  # Senior
        ])
        
        return features
    
    def _extract_title_features(self, title: str) -> List[float]:
        """Extract features from job title."""
        title_lower = title.lower()
        features = []
        
        # Role type indicators
        role_types = {
            'developer': ['developer', 'engineer', 'programmer'],
            'analyst': ['analyst', 'business analyst'],
            'manager': ['manager', 'lead', 'director'],
            'data': ['data scientist', 'data analyst', 'data engineer'],
            'devops': ['devops', 'sre', 'infrastructure'],
            'frontend': ['frontend', 'front-end', 'ui', 'ux'],
            'backend': ['backend', 'back-end', 'server'],
            'fullstack': ['fullstack', 'full-stack', 'full stack']
        }
        
        for role_type, keywords in role_types.items():
            has_role = any(keyword in title_lower for keyword in keywords)
            features.append(1.0 if has_role else 0.0)
        
        # Seniority indicators
        seniority_keywords = ['senior', 'lead', 'principal', 'staff', 'junior', 'intern']
        for keyword in seniority_keywords:
            features.append(1.0 if keyword in title_lower else 0.0)
        
        return features
    
    def _extract_location_features(self, location: str) -> List[float]:
        """Extract location-based features."""
        location_lower = location.lower()
        features = []
        
        # Major tech hubs
        tech_hubs = ['paris', 'london', 'berlin', 'amsterdam', 'zurich', 'dublin', 'barcelona', 'milan']
        for hub in tech_hubs:
            features.append(1.0 if hub in location_lower else 0.0)
        
        # Remote work indicator
        remote_keywords = ['remote', 'work from home', 'distributed', 'anywhere']
        is_remote = any(keyword in location_lower for keyword in remote_keywords)
        features.append(1.0 if is_remote else 0.0)
        
        # Country indicators
        countries = ['france', 'uk', 'germany', 'netherlands', 'switzerland', 'morocco', 'spain', 'italy']
        for country in countries:
            features.append(1.0 if country in location_lower else 0.0)
        
        return features
    
    def _extract_salary_features(self, salary_range: Optional[Tuple[int, int]]) -> List[float]:
        """Extract salary-related features."""
        if not salary_range:
            return [0.0, 0.0, 0.0, 0.0]
        
        min_salary, max_salary = salary_range
        avg_salary = (min_salary + max_salary) / 2
        salary_spread = max_salary - min_salary
        
        # Normalize salaries (assuming range 20k-200k)
        normalized_min = min(max(min_salary / 200000, 0), 1)
        normalized_max = min(max(max_salary / 200000, 0), 1)
        normalized_avg = min(max(avg_salary / 200000, 0), 1)
        normalized_spread = min(max(salary_spread / 180000, 0), 1)
        
        return [normalized_min, normalized_max, normalized_avg, normalized_spread]
    
    def _extract_job_complexity_features(self, job_description: str) -> List[float]:
        """Extract job complexity indicators."""
        text_lower = job_description.lower()
        features = []
        
        # Technical complexity indicators
        complex_keywords = [
            'architecture', 'design patterns', 'microservices', 'distributed systems',
            'scalability', 'performance optimization', 'security', 'algorithms',
            'machine learning', 'ai', 'big data', 'cloud native'
        ]
        
        complexity_score = sum(1 for keyword in complex_keywords if keyword in text_lower)
        features.append(min(complexity_score / len(complex_keywords), 1.0))
        
        # Leadership requirements
        leadership_keywords = ['lead', 'mentor', 'manage', 'coordinate', 'supervise']
        leadership_score = sum(1 for keyword in leadership_keywords if keyword in text_lower)
        features.append(min(leadership_score / len(leadership_keywords), 1.0))
        
        # Innovation requirements
        innovation_keywords = ['innovative', 'cutting-edge', 'research', 'prototype', 'poc']
        innovation_score = sum(1 for keyword in innovation_keywords if keyword in text_lower)
        features.append(min(innovation_score / len(innovation_keywords), 1.0))
        
        return features
    
    def _get_candidate_skills(self, candidate_data: Dict[str, Any]) -> List[str]:
        """Extract skills from candidate data."""
        skills = []
        
        # From skills field
        if 'skills' in candidate_data and candidate_data['skills']:
            for skill in candidate_data['skills']:
                if isinstance(skill, dict) and 'name' in skill:
                    skills.append(skill['name'].lower())
                elif isinstance(skill, str):
                    skills.append(skill.lower())
        
        # From description
        description = candidate_data.get('custom_description', '') or candidate_data.get('description', '')
        if description:
            # Extract skills mentioned in description
            all_skills = [skill for skills in self.skill_categories.values() for skill in skills]
            for skill in all_skills:
                if skill in description.lower():
                    skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _calculate_total_experience(self, candidate_data: Dict[str, Any]) -> int:
        """Calculate total years of experience."""
        # From years_of_experience field
        if 'years_of_experience' in candidate_data:
            return int(candidate_data['years_of_experience'] or 0)
        
        # Calculate from experience records
        if 'experience' in candidate_data and candidate_data['experience']:
            total_months = 0
            for exp in candidate_data['experience']:
                if isinstance(exp, dict):
                    start_date = exp.get('start_date')
                    end_date = exp.get('end_date') or datetime.now().strftime('%Y-%m-%d')
                    
                    if start_date:
                        try:
                            start = datetime.strptime(start_date[:10], '%Y-%m-%d')
                            end = datetime.strptime(end_date[:10], '%Y-%m-%d')
                            months = (end - start).days / 30.44  # Average days per month
                            total_months += max(months, 0)
                        except:
                            continue
            
            return int(total_months / 12)
        
        return 0
    
    def _extract_education_features(self, candidate_data: Dict[str, Any]) -> List[float]:
        """Extract education-related features."""
        features = []
        
        education = candidate_data.get('education', [])
        if not education:
            return [0.0] * 8  # Return zeros if no education data
        
        # Education levels
        has_bachelor = any('bachelor' in str(edu).lower() or 'licence' in str(edu).lower() 
                          for edu in education if edu)
        has_master = any('master' in str(edu).lower() or 'mba' in str(edu).lower() 
                        for edu in education if edu)
        has_phd = any('phd' in str(edu).lower() or 'doctorate' in str(edu).lower() 
                     for edu in education if edu)
        
        features.extend([
            1.0 if has_bachelor else 0.0,
            1.0 if has_master else 0.0,
            1.0 if has_phd else 0.0
        ])
        
        # Field of study relevance
        relevant_fields = ['computer science', 'engineering', 'mathematics', 'physics', 'business']
        has_relevant_field = any(field in str(education).lower() for field in relevant_fields)
        features.append(1.0 if has_relevant_field else 0.0)
        
        # Number of degrees
        num_degrees = len([edu for edu in education if edu])
        features.append(min(num_degrees / 3.0, 1.0))  # Normalize to max 3 degrees
        
        # Prestigious institutions (simplified)
        prestigious_keywords = ['mit', 'stanford', 'harvard', 'oxford', 'cambridge', 'sorbonne']
        has_prestigious = any(keyword in str(education).lower() for keyword in prestigious_keywords)
        features.append(1.0 if has_prestigious else 0.0)
        
        # Recent education (within last 10 years)
        # This would need date parsing - simplified for now
        features.append(0.5)  # Placeholder
        
        # Continuous learning indicators
        continuous_learning_keywords = ['certification', 'course', 'training', 'bootcamp']
        has_continuous_learning = any(keyword in str(education).lower() 
                                    for keyword in continuous_learning_keywords)
        features.append(1.0 if has_continuous_learning else 0.0)
        
        return features
    
    def _extract_career_progression_features(self, candidate_data: Dict[str, Any]) -> List[float]:
        """Extract career progression indicators."""
        features = []
        
        experience = candidate_data.get('experience', [])
        if not experience or len(experience) < 2:
            return [0.0] * 5
        
        # Sort experiences by start date
        sorted_exp = []
        for exp in experience:
            if isinstance(exp, dict) and exp.get('start_date'):
                try:
                    start_date = datetime.strptime(exp['start_date'][:10], '%Y-%m-%d')
                    sorted_exp.append((start_date, exp))
                except:
                    continue
        
        sorted_exp.sort(key=lambda x: x[0])
        
        if len(sorted_exp) < 2:
            return [0.0] * 5
        
        # Career progression indicators
        titles = [exp[1].get('title', '').lower() for exp in sorted_exp]
        
        # Upward progression (junior -> senior, developer -> lead, etc.)
        progression_score = 0
        for i in range(1, len(titles)):
            prev_title, curr_title = titles[i-1], titles[i]
            if ('senior' in curr_title and 'senior' not in prev_title) or \
               ('lead' in curr_title and 'lead' not in prev_title) or \
               ('manager' in curr_title and 'manager' not in prev_title):
                progression_score += 1
        
        features.append(min(progression_score / max(len(titles) - 1, 1), 1.0))
        
        # Job stability (average tenure)
        tenures = []
        for start_date, exp in sorted_exp:
            end_date_str = exp.get('end_date')
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str[:10], '%Y-%m-%d')
                    tenure_months = (end_date - start_date).days / 30.44
                    tenures.append(max(tenure_months, 0))
                except:
                    continue
        
        avg_tenure = np.mean(tenures) if tenures else 0
        features.append(min(avg_tenure / 36, 1.0))  # Normalize to 3 years
        
        # Number of job changes
        num_jobs = len(sorted_exp)
        features.append(min(num_jobs / 10, 1.0))  # Normalize to max 10 jobs
        
        # Industry consistency
        # Simplified - would need industry classification
        features.append(0.5)  # Placeholder
        
        # Recent activity (last job within 6 months)
        if sorted_exp:
            last_job_end = sorted_exp[-1][1].get('end_date')
            if not last_job_end:  # Currently employed
                features.append(1.0)
            else:
                try:
                    end_date = datetime.strptime(last_job_end[:10], '%Y-%m-%d')
                    months_since = (datetime.now() - end_date).days / 30.44
                    features.append(1.0 if months_since <= 6 else 0.0)
                except:
                    features.append(0.5)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_skill_diversity_features(self, skills: List[str]) -> List[float]:
        """Extract skill diversity metrics."""
        features = []
        
        if not skills:
            return [0.0] * 4
        
        # Skills per category
        category_counts = {}
        for category, category_skills in self.skill_categories.items():
            count = sum(1 for skill in skills if skill in category_skills)
            category_counts[category] = count
        
        # Skill diversity (number of categories represented)
        categories_represented = sum(1 for count in category_counts.values() if count > 0)
        features.append(categories_represented / len(self.skill_categories))
        
        # Skill depth (max skills in any category)
        max_category_skills = max(category_counts.values()) if category_counts else 0
        features.append(min(max_category_skills / 10, 1.0))
        
        # Skill breadth (total unique skills)
        features.append(min(len(skills) / 50, 1.0))
        
        # Balance score (how evenly distributed skills are across categories)
        if categories_represented > 0:
            category_ratios = [count / len(skills) for count in category_counts.values() if count > 0]
            balance_score = 1 - np.std(category_ratios) if len(category_ratios) > 1 else 1.0
            features.append(balance_score)
        else:
            features.append(0.0)
        
        return features
    
    def _get_experience_level(self, years: int) -> str:
        """Determine experience level from years."""
        for level, (min_years, max_years) in self.experience_levels.items():
            if min_years <= years <= max_years:
                return level
        return 'expert' if years > 15 else 'entry'
    
    def fit(self, job_descriptions: List[str], candidate_descriptions: List[str]):
        """Fit the vectorizers on training data."""
        all_texts = job_descriptions + candidate_descriptions
        all_texts = [text for text in all_texts if text and text.strip()]
        
        if all_texts:
            self.description_vectorizer.fit(all_texts)
            
            # Extract skills text for skill vectorizer
            skill_texts = []
            for text in all_texts:
                skills_found = []
                text_lower = text.lower()
                for skills in self.skill_categories.values():
                    for skill in skills:
                        if skill in text_lower:
                            skills_found.append(skill)
                skill_texts.append(" ".join(skills_found))
            
            if skill_texts:
                self.skill_vectorizer.fit(skill_texts)
        
        self.is_fitted = True
        logger.info("FeatureExtractor fitted successfully")
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []
        
        # Text features
        feature_names.extend([f"text_feature_{i}" for i in range(300)])
        
        # Skill features
        for category in self.skill_categories.keys():
            feature_names.extend([f"{category}_count", f"{category}_ratio"])
        feature_names.extend(["total_skills", "skill_categories"])
        
        # Experience features
        feature_names.append("years_experience")
        feature_names.extend([f"exp_level_{level}" for level in self.experience_levels.keys()])
        feature_names.extend(["exp_entry", "exp_junior", "exp_mid", "exp_senior"])
        
        # Title features
        role_types = ['developer', 'analyst', 'manager', 'data', 'devops', 'frontend', 'backend', 'fullstack']
        feature_names.extend([f"role_{role}" for role in role_types])
        seniority_keywords = ['senior', 'lead', 'principal', 'staff', 'junior', 'intern']
        feature_names.extend([f"seniority_{keyword}" for keyword in seniority_keywords])
        
        # Location features
        tech_hubs = ['paris', 'london', 'berlin', 'amsterdam', 'zurich', 'dublin', 'barcelona', 'milan']
        feature_names.extend([f"location_{hub}" for hub in tech_hubs])
        feature_names.append("is_remote")
        countries = ['france', 'uk', 'germany', 'netherlands', 'switzerland', 'morocco', 'spain', 'italy']
        feature_names.extend([f"country_{country}" for country in countries])
        
        # Salary features
        feature_names.extend(["salary_min", "salary_max", "salary_avg", "salary_spread"])
        
        # Job complexity features
        feature_names.extend(["technical_complexity", "leadership_requirements", "innovation_requirements"])
        
        return feature_names 