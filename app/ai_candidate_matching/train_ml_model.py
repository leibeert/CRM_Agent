#!/usr/bin/env python3
"""
Training Script for ML-based Candidate Matching System

This script trains the ML models using the available candidate data and creates
synthetic job data for training purposes.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_candidate_matching.services.matching_service import MLMatchingService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("Starting ML Candidate Matching System Training")
    
    # Initialize the ML matching service
    # Point to the database_tables directory for CSV data
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database_tables")
    model_path = os.path.join(os.path.dirname(__file__), "models")
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model path: {model_path}")
    
    # Check if data files exist
    required_files = ['resources.csv', 'skills.csv', 'resource_skills.csv', 'experiences.csv']
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Required data file not found: {file_path}")
            return
    
    # Initialize the service
    ml_service = MLMatchingService(data_path=data_path, model_path=model_path)
    
    # Train the model
    logger.info("Starting model training...")
    try:
        training_results = ml_service.train_model(
            num_synthetic_jobs=150,
            num_training_pairs=1500,
            validation_split=0.2,
            optimize_hyperparameters=False  # Set to True for better performance but longer training
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Training Results:")
        for key, value in training_results.items():
            if key != 'training_metrics':
                logger.info(f"  {key}: {value}")
        
        # Display training metrics
        if 'training_metrics' in training_results:
            logger.info("Training Metrics:")
            for metric, value in training_results['training_metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save training results
        results_file = os.path.join(model_path, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        logger.info(f"Training results saved to: {results_file}")
        
        # Test the trained model
        logger.info("\nTesting the trained model...")
        test_model(ml_service)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


def test_model(ml_service: MLMatchingService):
    """Test the trained model with sample job descriptions."""
    
    # Sample job descriptions for testing
    test_jobs = [
        {
            'title': 'Senior Python Developer',
            'description': '''
            We are looking for an experienced Python developer to join our backend team.
            You will be responsible for developing scalable web applications using Django
            and Flask frameworks. Experience with PostgreSQL, Redis, and AWS is required.
            You should have strong knowledge of RESTful APIs, microservices architecture,
            and containerization with Docker.
            ''',
            'required_experience': 5,
            'location': 'Paris, France',
            'salary_range': (65000, 85000)
        },
        {
            'title': 'Frontend React Developer',
            'description': '''
            Join our frontend team to build modern, responsive user interfaces using React.
            You should be proficient in JavaScript ES6+, HTML5, CSS3, and have experience
            with state management libraries like Redux. Knowledge of TypeScript and testing
            frameworks is a plus.
            ''',
            'required_experience': 3,
            'location': 'London, UK',
            'salary_range': (50000, 70000)
        },
        {
            'title': 'Data Scientist',
            'description': '''
            We need a data scientist to analyze large datasets and build machine learning
            models. You should be proficient in Python, pandas, numpy, scikit-learn, and
            have experience with deep learning frameworks like TensorFlow or PyTorch.
            SQL knowledge and experience with cloud platforms is required.
            ''',
            'required_experience': 4,
            'location': 'Berlin, Germany',
            'salary_range': (70000, 90000)
        }
    ]
    
    for i, job in enumerate(test_jobs, 1):
        logger.info(f"\n--- Test Job {i}: {job['title']} ---")
        
        # Find best candidates for this job
        top_candidates = ml_service.find_best_candidates(
            job_description=job['description'],
            job_title=job['title'],
            required_experience=job['required_experience'],
            location=job['location'],
            salary_range=job['salary_range'],
            top_k=5
        )
        
        logger.info(f"Top 5 candidates for {job['title']}:")
        for j, result in enumerate(top_candidates, 1):
            candidate = result['candidate']
            score = result['compatibility_score']
            confidence = result['confidence']
            
            logger.info(f"  {j}. {candidate.get('first_name', '')} {candidate.get('last_name', '')}")
            logger.info(f"     Score: {score:.3f} (Confidence: {confidence:.3f})")
            logger.info(f"     Title: {candidate.get('title', 'N/A')}")
            logger.info(f"     Experience: {candidate.get('years_of_experience', 0)} years")
            
            # Show top skills
            skills = candidate.get('skills', [])[:5]
            skill_names = [skill.get('name', '') for skill in skills]
            logger.info(f"     Top Skills: {', '.join(skill_names)}")
            
            # Show explanation
            explanation = result.get('explanation', {})
            interpretation = explanation.get('score_interpretation', '')
            logger.info(f"     Interpretation: {interpretation}")
            
            if explanation.get('recommendations'):
                logger.info(f"     Recommendation: {explanation['recommendations'][0]}")
            
            logger.info("")


def demo_single_prediction(ml_service: MLMatchingService):
    """Demo single candidate-job compatibility prediction."""
    
    logger.info("\n--- Single Prediction Demo ---")
    
    # Sample candidate (you can modify this)
    sample_candidate = {
        'id': 'demo_candidate',
        'first_name': 'John',
        'last_name': 'Doe',
        'title': 'Software Developer',
        'years_of_experience': 5,
        'skills': [
            {'name': 'Python', 'proficiency_level': 'advanced'},
            {'name': 'Django', 'proficiency_level': 'intermediate'},
            {'name': 'JavaScript', 'proficiency_level': 'intermediate'},
            {'name': 'PostgreSQL', 'proficiency_level': 'intermediate'},
            {'name': 'Git', 'proficiency_level': 'advanced'}
        ],
        'custom_description': 'Experienced software developer with strong Python and web development skills.',
        'address': 'Paris, France'
    }
    
    # Sample job
    job_description = '''
    We are seeking a talented Python developer to join our growing team.
    The ideal candidate will have experience with Django, REST APIs, and database design.
    Knowledge of frontend technologies and cloud platforms is a plus.
    '''
    
    # Make prediction
    result = ml_service.predict_candidate_job_compatibility(
        candidate_data=sample_candidate,
        job_description=job_description,
        job_title='Python Developer',
        required_experience=3,
        location='Paris, France',
        salary_range=(55000, 75000)
    )
    
    logger.info(f"Candidate: {sample_candidate['first_name']} {sample_candidate['last_name']}")
    logger.info(f"Job: Python Developer")
    logger.info(f"Compatibility Score: {result['compatibility_score']:.3f}")
    logger.info(f"Confidence: {result['confidence']:.3f}")
    logger.info(f"Interpretation: {result['explanation']['score_interpretation']}")
    
    # Show key factors
    key_factors = result['explanation'].get('key_factors', [])
    if key_factors:
        logger.info("Key Factors:")
        for factor in key_factors:
            logger.info(f"  - {factor['factor']}: {factor['details']}")
    
    # Show recommendations
    recommendations = result['explanation'].get('recommendations', [])
    if recommendations:
        logger.info("Recommendations:")
        for rec in recommendations:
            logger.info(f"  - {rec}")


if __name__ == "__main__":
    main() 