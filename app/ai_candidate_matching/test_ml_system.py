#!/usr/bin/env python3
"""
Simple Test Script for AI Candidate Matching System

This script tests the ML-based candidate matching system using your existing data.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_candidate_data():
    """Load candidate data from CSV files."""
    try:
        # Data paths
        data_path = "../../database_tables"
        
        # Load CSV files
        resources_df = pd.read_csv(os.path.join(data_path, "resources.csv"))
        skills_df = pd.read_csv(os.path.join(data_path, "skills.csv"))
        resource_skills_df = pd.read_csv(os.path.join(data_path, "resource_skills.csv"))
        experiences_df = pd.read_csv(os.path.join(data_path, "experiences.csv"))
        
        logger.info(f"Loaded {len(resources_df)} candidates")
        logger.info(f"Loaded {len(skills_df)} skills")
        logger.info(f"Loaded {len(resource_skills_df)} candidate-skill mappings")
        logger.info(f"Loaded {len(experiences_df)} experience records")
        
        return resources_df, skills_df, resource_skills_df, experiences_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None


def create_candidate_profiles(resources_df, skills_df, resource_skills_df, experiences_df):
    """Create complete candidate profiles."""
    profiles = []
    
    for _, candidate in resources_df.iterrows():
        candidate_id = candidate['id']
        
        # Base profile
        profile = {
            'id': candidate_id,
            'first_name': candidate.get('first_name', ''),
            'last_name': candidate.get('last_name', ''),
            'email': candidate.get('email', ''),
            'title': candidate.get('title', ''),
            'years_of_experience': candidate.get('years_of_experience', 0),
            'custom_description': candidate.get('custom_description', ''),
            'address': candidate.get('address', ''),
        }
        
        # Add skills
        candidate_skill_ids = resource_skills_df[
            resource_skills_df['resource_id'] == candidate_id
        ]['skill_id'].tolist()
        
        candidate_skills = []
        for skill_id in candidate_skill_ids[:10]:  # Limit to first 10 skills
            skill_row = skills_df[skills_df['id'] == skill_id]
            if not skill_row.empty:
                skill_info = skill_row.iloc[0]
                candidate_skills.append({
                    'id': skill_id,
                    'name': skill_info['name'],
                    'category': skill_info.get('category', ''),
                })
        
        profile['skills'] = candidate_skills
        
        # Add experiences (limit to recent ones)
        candidate_experiences = experiences_df[
            experiences_df['resource_id'] == candidate_id
        ].head(5).to_dict('records')
        
        profile['experience'] = candidate_experiences
        profiles.append(profile)
    
    logger.info(f"Created {len(profiles)} complete candidate profiles")
    return profiles


def simple_skill_matching(candidate_skills, job_requirements):
    """Simple skill matching algorithm."""
    candidate_skill_names = [skill['name'].lower() for skill in candidate_skills]
    job_requirements_lower = [req.lower() for req in job_requirements]
    
    matches = 0
    for req in job_requirements_lower:
        for skill in candidate_skill_names:
            if req in skill or skill in req:
                matches += 1
                break
    
    return matches / len(job_requirements) if job_requirements else 0


def calculate_experience_score(candidate_exp, required_exp):
    """Calculate experience matching score."""
    if candidate_exp >= required_exp:
        return min(1.0, 1.0 + (candidate_exp - required_exp) * 0.05)
    else:
        return max(0.0, candidate_exp / required_exp) if required_exp > 0 else 0.5


def test_candidate_matching(profiles):
    """Test candidate matching with sample jobs."""
    
    # Sample job descriptions
    test_jobs = [
        {
            'title': 'Python Developer',
            'required_skills': ['python', 'django', 'flask', 'postgresql', 'git'],
            'preferred_skills': ['docker', 'aws', 'redis'],
            'min_experience': 3,
            'description': 'We need a Python developer with web framework experience.'
        },
        {
            'title': 'Data Scientist',
            'required_skills': ['python', 'machine learning', 'pandas', 'numpy', 'sql'],
            'preferred_skills': ['tensorflow', 'pytorch', 'aws'],
            'min_experience': 4,
            'description': 'Data scientist role requiring ML and Python skills.'
        },
        {
            'title': 'Frontend Developer',
            'required_skills': ['javascript', 'react', 'html', 'css', 'git'],
            'preferred_skills': ['typescript', 'redux', 'webpack'],
            'min_experience': 2,
            'description': 'Frontend developer for React applications.'
        }
    ]
    
    for job in test_jobs:
        logger.info(f"\n=== Testing Job: {job['title']} ===")
        
        candidate_scores = []
        
        for candidate in profiles:
            # Calculate skill matching score
            skill_score = simple_skill_matching(
                candidate['skills'], 
                job['required_skills'] + job['preferred_skills']
            )
            
            # Calculate experience score
            candidate_exp = candidate.get('years_of_experience', 0)
            exp_score = calculate_experience_score(candidate_exp, job['min_experience'])
            
            # Calculate overall score (60% skills, 40% experience)
            overall_score = 0.6 * skill_score + 0.4 * exp_score
            
            candidate_scores.append({
                'candidate': candidate,
                'skill_score': skill_score,
                'experience_score': exp_score,
                'overall_score': overall_score
            })
        
        # Sort by overall score
        candidate_scores.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Show top 5 candidates
        logger.info(f"Top 5 candidates for {job['title']}:")
        for i, result in enumerate(candidate_scores[:5], 1):
            candidate = result['candidate']
            logger.info(f"  {i}. {candidate.get('first_name', '')} {candidate.get('last_name', '')}")
            logger.info(f"     Title: {candidate.get('title', 'N/A')}")
            logger.info(f"     Experience: {candidate.get('years_of_experience', 0)} years")
            logger.info(f"     Skills Score: {result['skill_score']:.3f}")
            logger.info(f"     Experience Score: {result['experience_score']:.3f}")
            logger.info(f"     Overall Score: {result['overall_score']:.3f}")
            
            # Show matching skills
            candidate_skill_names = [skill['name'] for skill in candidate['skills'][:5]]
            logger.info(f"     Top Skills: {', '.join(candidate_skill_names)}")
            logger.info("")


def analyze_data_quality(profiles):
    """Analyze the quality of candidate data."""
    logger.info("\n=== Data Quality Analysis ===")
    
    total_candidates = len(profiles)
    complete_profiles = 0
    
    skill_distribution = {}
    experience_distribution = {'0-2': 0, '3-5': 0, '6-10': 0, '10+': 0}
    
    for candidate in profiles:
        # Check completeness
        required_fields = ['first_name', 'last_name', 'email', 'title']
        is_complete = all(candidate.get(field) for field in required_fields)
        if is_complete:
            complete_profiles += 1
        
        # Analyze skills
        num_skills = len(candidate.get('skills', []))
        skill_range = f"{min(num_skills//5*5, 20)}+"
        skill_distribution[skill_range] = skill_distribution.get(skill_range, 0) + 1
        
        # Analyze experience
        exp = candidate.get('years_of_experience', 0)
        if exp <= 2:
            experience_distribution['0-2'] += 1
        elif exp <= 5:
            experience_distribution['3-5'] += 1
        elif exp <= 10:
            experience_distribution['6-10'] += 1
        else:
            experience_distribution['10+'] += 1
    
    logger.info(f"Total candidates: {total_candidates}")
    logger.info(f"Complete profiles: {complete_profiles} ({complete_profiles/total_candidates*100:.1f}%)")
    logger.info(f"Experience distribution: {experience_distribution}")
    logger.info(f"Skill count distribution: {skill_distribution}")


def main():
    """Main testing function."""
    logger.info("Starting AI Candidate Matching System Test")
    
    # Load data
    logger.info("Loading data from CSV files...")
    resources_df, skills_df, resource_skills_df, experiences_df = load_candidate_data()
    
    if resources_df is None:
        logger.error("Failed to load data. Please check CSV files exist.")
        return
    
    # Create candidate profiles
    logger.info("Creating candidate profiles...")
    profiles = create_candidate_profiles(resources_df, skills_df, resource_skills_df, experiences_df)
    
    if not profiles:
        logger.error("No candidate profiles created.")
        return
    
    # Analyze data quality
    analyze_data_quality(profiles)
    
    # Test candidate matching
    logger.info("Testing candidate matching...")
    test_candidate_matching(profiles)
    
    logger.info("\n=== Test Complete ===")
    logger.info("This is a simplified version of the ML system.")
    logger.info("The full ML version would provide:")
    logger.info("- 400+ advanced features")
    logger.info("- Ensemble ML models")
    logger.info("- Semantic skill matching")
    logger.info("- Confidence scores")
    logger.info("- Detailed explanations")


if __name__ == "__main__":
    main() 