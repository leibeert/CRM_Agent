"""
Data Loader for AI Candidate Matching

This module loads and processes data from CSV files and database for training ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import os
from pathlib import Path
import sqlite3
import mysql.connector
from sqlalchemy import create_engine
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads data from various sources for ML model training and inference.
    """
    
    def __init__(self, data_path: str = None, db_config: Dict[str, str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to CSV data files
            db_config: Database configuration dictionary
        """
        self.data_path = data_path or "database_tables"
        self.db_config = db_config
        self.engine = None
        
        if db_config:
            self._setup_database_connection()
    
    def _setup_database_connection(self):
        """Setup database connection."""
        try:
            if self.db_config.get('type') == 'mysql':
                connection_string = (
                    f"mysql+pymysql://{self.db_config['user']}:"
                    f"{self.db_config['password']}@{self.db_config['host']}:"
                    f"{self.db_config['port']}/{self.db_config['database']}"
                )
                self.engine = create_engine(connection_string)
                logger.info("MySQL database connection established")
            elif self.db_config.get('type') == 'sqlite':
                connection_string = f"sqlite:///{self.db_config['database']}"
                self.engine = create_engine(connection_string)
                logger.info("SQLite database connection established")
        except Exception as e:
            logger.error(f"Failed to setup database connection: {e}")
            self.engine = None
    
    def load_candidates_data(self, source: str = 'csv') -> pd.DataFrame:
        """
        Load candidates data from CSV or database.
        
        Args:
            source: Data source ('csv' or 'db')
            
        Returns:
            DataFrame with candidate data
        """
        if source == 'csv':
            return self._load_candidates_from_csv()
        elif source == 'db' and self.engine:
            return self._load_candidates_from_db()
        else:
            raise ValueError(f"Invalid source: {source}")
    
    def load_skills_data(self, source: str = 'csv') -> pd.DataFrame:
        """
        Load skills data from CSV or database.
        
        Args:
            source: Data source ('csv' or 'db')
            
        Returns:
            DataFrame with skills data
        """
        if source == 'csv':
            return self._load_skills_from_csv()
        elif source == 'db' and self.engine:
            return self._load_skills_from_db()
        else:
            raise ValueError(f"Invalid source: {source}")
    
    def load_experiences_data(self, source: str = 'csv') -> pd.DataFrame:
        """
        Load experience data from CSV or database.
        
        Args:
            source: Data source ('csv' or 'db')
            
        Returns:
            DataFrame with experience data
        """
        if source == 'csv':
            return self._load_experiences_from_csv()
        elif source == 'db' and self.engine:
            return self._load_experiences_from_db()
        else:
            raise ValueError(f"Invalid source: {source}")
    
    def load_complete_candidate_profiles(self, source: str = 'csv') -> List[Dict[str, Any]]:
        """
        Load complete candidate profiles with all related data.
        
        Args:
            source: Data source ('csv' or 'db')
            
        Returns:
            List of complete candidate profile dictionaries
        """
        logger.info(f"Loading complete candidate profiles from {source}")
        
        # Load base data
        candidates_df = self.load_candidates_data(source)
        skills_df = self.load_skills_data(source)
        experiences_df = self.load_experiences_data(source)
        
        # Load resource-skills mapping
        if source == 'csv':
            resource_skills_df = self._load_resource_skills_from_csv()
        else:
            resource_skills_df = self._load_resource_skills_from_db()
        
        # Load education data if available
        try:
            if source == 'csv':
                education_df = self._load_education_from_csv()
            else:
                education_df = self._load_education_from_db()
        except:
            education_df = pd.DataFrame()
        
        # Merge data to create complete profiles
        complete_profiles = []
        
        for _, candidate in candidates_df.iterrows():
            candidate_id = candidate['id']
            
            profile = {
                'id': candidate_id,
                'first_name': candidate.get('first_name', ''),
                'last_name': candidate.get('last_name', ''),
                'email': candidate.get('email', ''),
                'phone': candidate.get('phone', ''),
                'address': candidate.get('address', ''),
                'title': candidate.get('title', ''),
                'years_of_experience': candidate.get('years_of_experience', 0),
                'custom_description': candidate.get('custom_description', ''),
                'description': candidate.get('description', ''),
                'created_at': candidate.get('created_at', ''),
                'updated_at': candidate.get('updated_at', '')
            }
            
            # Add skills
            candidate_skill_ids = resource_skills_df[
                resource_skills_df['resource_id'] == candidate_id
            ]['skill_id'].tolist()
            
            candidate_skills = []
            for skill_id in candidate_skill_ids:
                skill_row = skills_df[skills_df['id'] == skill_id]
                if not skill_row.empty:
                    skill_info = skill_row.iloc[0]
                    # Get proficiency from resource_skills table
                    proficiency_row = resource_skills_df[
                        (resource_skills_df['resource_id'] == candidate_id) & 
                        (resource_skills_df['skill_id'] == skill_id)
                    ]
                    
                    # Convert numeric level to text proficiency
                    if not proficiency_row.empty:
                        level = proficiency_row.iloc[0]['level']
                        if level == 0:
                            proficiency = 'beginner'
                        elif level == 1:
                            proficiency = 'beginner'
                        elif level == 2:
                            proficiency = 'intermediate'
                        elif level == 3:
                            proficiency = 'advanced'
                        else:
                            proficiency = 'intermediate'
                    else:
                        proficiency = 'intermediate'
                    
                    candidate_skills.append({
                        'id': skill_id,
                        'name': skill_info['name'],
                        'category': skill_info.get('category', ''),
                        'proficiency_level': proficiency
                    })
            
            profile['skills'] = candidate_skills
            
            # Add experiences
            candidate_experiences = experiences_df[
                experiences_df['resource_id'] == candidate_id
            ].to_dict('records')
            
            profile['experience'] = candidate_experiences
            
            # Add education if available
            if not education_df.empty:
                candidate_education = education_df[
                    education_df['resource_id'] == candidate_id
                ].to_dict('records')
                profile['education'] = candidate_education
            else:
                profile['education'] = []
            
            complete_profiles.append(profile)
        
        logger.info(f"Loaded {len(complete_profiles)} complete candidate profiles")
        return complete_profiles
    
    def _load_candidates_from_csv(self) -> pd.DataFrame:
        """Load candidates from CSV file."""
        csv_path = os.path.join(self.data_path, 'resources.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Resources CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} candidates from CSV")
        return df
    
    def _load_candidates_from_db(self) -> pd.DataFrame:
        """Load candidates from database."""
        query = "SELECT * FROM resources"
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} candidates from database")
        return df
    
    def _load_skills_from_csv(self) -> pd.DataFrame:
        """Load skills from CSV file."""
        csv_path = os.path.join(self.data_path, 'skills.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Skills CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} skills from CSV")
        return df
    
    def _load_skills_from_db(self) -> pd.DataFrame:
        """Load skills from database."""
        query = "SELECT * FROM skills"
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} skills from database")
        return df
    
    def _load_experiences_from_csv(self) -> pd.DataFrame:
        """Load experiences from CSV file."""
        csv_path = os.path.join(self.data_path, 'experiences.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Experiences CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} experiences from CSV")
        return df
    
    def _load_experiences_from_db(self) -> pd.DataFrame:
        """Load experiences from database."""
        query = "SELECT * FROM experiences"
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} experiences from database")
        return df
    
    def _load_resource_skills_from_csv(self) -> pd.DataFrame:
        """Load resource-skills mapping from CSV file."""
        csv_path = os.path.join(self.data_path, 'resource_skills.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Resource skills CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} resource-skill mappings from CSV")
        return df
    
    def _load_resource_skills_from_db(self) -> pd.DataFrame:
        """Load resource-skills mapping from database."""
        query = "SELECT * FROM resource_skills"
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df)} resource-skill mappings from database")
        return df
    
    def _load_education_from_csv(self) -> pd.DataFrame:
        """Load education data from CSV file."""
        csv_path = os.path.join(self.data_path, 'education.csv')
        if not os.path.exists(csv_path):
            logger.warning(f"Education CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} education records from CSV")
        return df
    
    def _load_education_from_db(self) -> pd.DataFrame:
        """Load education data from database."""
        try:
            query = "SELECT * FROM education"
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} education records from database")
            return df
        except:
            logger.warning("Education table not found in database")
            return pd.DataFrame()
    
    def create_synthetic_job_data(self, num_jobs: int = 100) -> List[Dict[str, Any]]:
        """
        Create synthetic job data for training purposes.
        
        Args:
            num_jobs: Number of synthetic jobs to create
            
        Returns:
            List of synthetic job dictionaries
        """
        logger.info(f"Creating {num_jobs} synthetic job descriptions")
        
        # Load skills to use in job descriptions
        skills_df = self.load_skills_data()
        all_skills = skills_df['name'].tolist()
        
        # Job templates
        job_templates = [
            {
                'title': 'Senior Python Developer',
                'description': 'We are looking for an experienced Python developer to join our team. You will be responsible for developing scalable web applications using modern frameworks.',
                'required_skills': ['python', 'django', 'flask', 'postgresql', 'git'],
                'preferred_skills': ['docker', 'aws', 'redis'],
                'min_experience': 5,
                'location': 'Paris, France',
                'salary_range': (60000, 80000)
            },
            {
                'title': 'Frontend React Developer',
                'description': 'Join our frontend team to build amazing user interfaces using React and modern JavaScript technologies.',
                'required_skills': ['javascript', 'react', 'html', 'css', 'git'],
                'preferred_skills': ['typescript', 'redux', 'webpack'],
                'min_experience': 3,
                'location': 'London, UK',
                'salary_range': (50000, 70000)
            },
            {
                'title': 'Data Scientist',
                'description': 'We need a data scientist to analyze large datasets and build machine learning models to drive business insights.',
                'required_skills': ['python', 'machine learning', 'pandas', 'numpy', 'scikit-learn'],
                'preferred_skills': ['tensorflow', 'pytorch', 'sql', 'tableau'],
                'min_experience': 4,
                'location': 'Berlin, Germany',
                'salary_range': (65000, 85000)
            },
            {
                'title': 'DevOps Engineer',
                'description': 'Looking for a DevOps engineer to manage our cloud infrastructure and CI/CD pipelines.',
                'required_skills': ['docker', 'kubernetes', 'aws', 'jenkins', 'git'],
                'preferred_skills': ['terraform', 'ansible', 'monitoring'],
                'min_experience': 4,
                'location': 'Amsterdam, Netherlands',
                'salary_range': (70000, 90000)
            },
            {
                'title': 'Full Stack Developer',
                'description': 'We are seeking a full stack developer proficient in both frontend and backend technologies.',
                'required_skills': ['javascript', 'node.js', 'react', 'mongodb', 'express'],
                'preferred_skills': ['typescript', 'graphql', 'docker'],
                'min_experience': 3,
                'location': 'Remote',
                'salary_range': (55000, 75000)
            }
        ]
        
        synthetic_jobs = []
        
        for i in range(num_jobs):
            # Select a random template
            template = np.random.choice(job_templates)
            
            # Add some variation
            job = template.copy()
            job['id'] = f"job_{i+1}"
            
            # Randomly modify some aspects
            if np.random.random() < 0.3:  # 30% chance to modify experience requirement
                job['min_experience'] = max(1, job['min_experience'] + np.random.randint(-2, 3))
            
            if np.random.random() < 0.2:  # 20% chance to add random skills
                additional_skills = np.random.choice(all_skills, size=np.random.randint(1, 3), replace=False)
                job['preferred_skills'].extend(additional_skills.tolist())
            
            # Add some noise to salary
            salary_noise = np.random.randint(-5000, 5001)
            min_sal, max_sal = job['salary_range']
            job['salary_range'] = (max(20000, min_sal + salary_noise), max(30000, max_sal + salary_noise))
            
            synthetic_jobs.append(job)
        
        return synthetic_jobs
    
    def create_training_pairs(self, candidates: List[Dict[str, Any]], 
                            jobs: List[Dict[str, Any]],
                            num_pairs: int = 1000) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Create candidate-job pairs with compatibility scores for training.
        
        Args:
            candidates: List of candidate profiles
            jobs: List of job descriptions
            num_pairs: Number of training pairs to create
            
        Returns:
            List of (candidate, job, compatibility_score) tuples
        """
        logger.info(f"Creating {num_pairs} training pairs")
        
        training_pairs = []
        
        for _ in range(num_pairs):
            # Randomly select candidate and job
            candidate = np.random.choice(candidates)
            job = np.random.choice(jobs)
            
            # Calculate a synthetic compatibility score
            score = self._calculate_synthetic_compatibility(candidate, job)
            
            training_pairs.append((candidate, job, score))
        
        return training_pairs
    
    def _calculate_synthetic_compatibility(self, candidate: Dict[str, Any], 
                                         job: Dict[str, Any]) -> float:
        """
        Calculate a synthetic compatibility score for training data.
        
        Args:
            candidate: Candidate profile
            job: Job description
            
        Returns:
            Compatibility score between 0 and 1
        """
        score = 0.0
        
        # Skill matching (40% of score)
        candidate_skills = [skill['name'].lower() for skill in candidate.get('skills', [])]
        required_skills = [skill.lower() for skill in job.get('required_skills', [])]
        preferred_skills = [skill.lower() for skill in job.get('preferred_skills', [])]
        
        # Required skills match
        required_matches = sum(1 for skill in required_skills if skill in candidate_skills)
        required_score = required_matches / len(required_skills) if required_skills else 1.0
        
        # Preferred skills match
        preferred_matches = sum(1 for skill in preferred_skills if skill in candidate_skills)
        preferred_score = preferred_matches / len(preferred_skills) if preferred_skills else 0.5
        
        skill_score = 0.7 * required_score + 0.3 * preferred_score
        score += 0.4 * skill_score
        
        # Experience matching (30% of score)
        candidate_exp = candidate.get('years_of_experience', 0)
        required_exp = job.get('min_experience', 0)
        
        if candidate_exp >= required_exp:
            exp_score = min(1.0, 1.0 + (candidate_exp - required_exp) * 0.1)  # Bonus for extra experience
        else:
            exp_score = max(0.0, candidate_exp / required_exp)  # Penalty for insufficient experience
        
        score += 0.3 * exp_score
        
        # Title/role matching (20% of score)
        candidate_title = candidate.get('title', '').lower()
        job_title = job.get('title', '').lower()
        
        # Simple keyword matching
        title_keywords = ['senior', 'junior', 'lead', 'developer', 'engineer', 'analyst', 'manager']
        title_score = 0.5  # Default
        
        for keyword in title_keywords:
            if keyword in candidate_title and keyword in job_title:
                title_score += 0.1
        
        title_score = min(1.0, title_score)
        score += 0.2 * title_score
        
        # Random factor (10% of score) to add some noise
        random_factor = np.random.uniform(0.3, 1.0)
        score += 0.1 * random_factor
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def save_training_data(self, training_pairs: List[Tuple], filepath: str):
        """
        Save training data to file.
        
        Args:
            training_pairs: List of training pairs
            filepath: Path to save the data
        """
        training_data = []
        
        for candidate, job, score in training_pairs:
            training_data.append({
                'candidate': candidate,
                'job': job,
                'compatibility_score': score
            })
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(training_data)} training pairs to {filepath}")
    
    def load_training_data(self, filepath: str) -> List[Tuple]:
        """
        Load training data from file.
        
        Args:
            filepath: Path to the training data file
            
        Returns:
            List of training pairs
        """
        with open(filepath, 'r') as f:
            training_data = json.load(f)
        
        training_pairs = []
        for item in training_data:
            training_pairs.append((
                item['candidate'],
                item['job'],
                item['compatibility_score']
            ))
        
        logger.info(f"Loaded {len(training_pairs)} training pairs from {filepath}")
        return training_pairs 