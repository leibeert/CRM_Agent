"""
ML Matching Service

This is the main service that orchestrates the ML-based candidate matching pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import os
import json

from ..core.feature_extractor import FeatureExtractor
from ..core.ml_models import CandidateJobMatcher
from ..data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class MLMatchingService:
    """
    Main service for ML-based candidate-job matching.
    """
    
    def __init__(self, data_path: str = None, model_path: str = None):
        """
        Initialize the ML matching service.
        
        Args:
            data_path: Path to data files
            model_path: Path to save/load trained models
        """
        self.data_path = data_path or "database_tables"
        self.model_path = model_path or "app/ai_candidate_matching/models"
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.ml_model = CandidateJobMatcher(model_type='ensemble')
        self.data_loader = DataLoader(data_path=self.data_path)
        
        # Training data cache
        self.candidates_cache = None
        self.jobs_cache = None
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
    
    def train_model(self, num_synthetic_jobs: int = 200, 
                   num_training_pairs: int = 2000,
                   validation_split: float = 0.2,
                   optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the ML model using available data.
        
        Args:
            num_synthetic_jobs: Number of synthetic jobs to create
            num_training_pairs: Number of training pairs to generate
            validation_split: Fraction of data for validation
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting ML model training...")
        
        # Load candidate data
        logger.info("Loading candidate profiles...")
        candidates = self.data_loader.load_complete_candidate_profiles(source='csv')
        self.candidates_cache = candidates
        
        if len(candidates) == 0:
            raise ValueError("No candidate data found. Please ensure CSV files are available.")
        
        logger.info(f"Loaded {len(candidates)} candidate profiles")
        
        # Create synthetic job data
        logger.info("Creating synthetic job data...")
        jobs = self.data_loader.create_synthetic_job_data(num_jobs=num_synthetic_jobs)
        self.jobs_cache = jobs
        
        # Create training pairs
        logger.info("Creating training pairs...")
        training_pairs = self.data_loader.create_training_pairs(
            candidates, jobs, num_pairs=num_training_pairs
        )
        
        # Extract features
        logger.info("Extracting features...")
        X, y = self._extract_features_from_pairs(training_pairs)
        
        # Fit feature extractor
        candidate_descriptions = [
            (candidate.get('custom_description', '') or candidate.get('description', ''))
            for candidate in candidates
        ]
        job_descriptions = [job['description'] for job in jobs]
        
        self.feature_extractor.fit(job_descriptions, candidate_descriptions)
        
        # Train ML model
        logger.info("Training ML model...")
        training_metrics = self.ml_model.train(
            X, y, 
            validation_split=validation_split,
            optimize_hyperparameters=optimize_hyperparameters
        )
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        
        # Prepare results
        results = {
            'training_metrics': training_metrics,
            'num_candidates': len(candidates),
            'num_jobs': len(jobs),
            'num_training_pairs': len(training_pairs),
            'feature_dimensions': X.shape[1],
            'model_type': self.ml_model.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training results: {results}")
        
        return results
    
    def predict_candidate_job_compatibility(self, candidate_data: Dict[str, Any], 
                                          job_description: str,
                                          job_title: str = "",
                                          required_experience: int = 0,
                                          location: str = "",
                                          salary_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Predict compatibility between a candidate and job.
        
        Args:
            candidate_data: Candidate profile dictionary
            job_description: Job description text
            job_title: Job title
            required_experience: Required years of experience
            location: Job location
            salary_range: Salary range tuple
            
        Returns:
            Prediction results with score and explanations
        """
        if not self.is_trained:
            self._load_models()
        
        # Extract features
        candidate_features = self.feature_extractor.extract_candidate_features(candidate_data)
        job_features = self.feature_extractor.extract_job_features(
            job_description, job_title, required_experience, location, salary_range
        )
        
        # Combine features (simple concatenation for now)
        combined_features = np.concatenate([candidate_features, job_features]).reshape(1, -1)
        
        # Make prediction
        score, confidence = self.ml_model.predict_with_confidence(combined_features)
        
        # Get feature importance for explanation
        feature_importance = self.ml_model.get_feature_importance()
        
        # Generate explanation
        explanation = self._generate_explanation(
            candidate_data, job_description, job_title, 
            score[0], confidence[0], feature_importance
        )
        
        return {
            'compatibility_score': float(score[0]),
            'confidence': float(confidence[0]),
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
    
    def find_best_candidates(self, job_description: str,
                           job_title: str = "",
                           required_experience: int = 0,
                           location: str = "",
                           salary_range: Tuple[int, int] = None,
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find the best candidates for a given job.
        
        Args:
            job_description: Job description text
            job_title: Job title
            required_experience: Required years of experience
            location: Job location
            salary_range: Salary range tuple
            top_k: Number of top candidates to return
            
        Returns:
            List of top candidates with scores
        """
        if not self.is_trained:
            self._load_models()
        
        # Load candidates if not cached
        if self.candidates_cache is None:
            self.candidates_cache = self.data_loader.load_complete_candidate_profiles(source='csv')
        
        candidates = self.candidates_cache
        
        if not candidates:
            return []
        
        # Extract job features once
        job_features = self.feature_extractor.extract_job_features(
            job_description, job_title, required_experience, location, salary_range
        )
        
        # Score all candidates
        candidate_scores = []
        
        for candidate in candidates:
            try:
                # Extract candidate features
                candidate_features = self.feature_extractor.extract_candidate_features(candidate)
                
                # Combine features
                combined_features = np.concatenate([candidate_features, job_features]).reshape(1, -1)
                
                # Predict compatibility
                score, confidence = self.ml_model.predict_with_confidence(combined_features)
                
                candidate_scores.append({
                    'candidate': candidate,
                    'compatibility_score': float(score[0]),
                    'confidence': float(confidence[0])
                })
                
            except Exception as e:
                logger.warning(f"Error scoring candidate {candidate.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by compatibility score
        candidate_scores.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        # Return top k candidates
        top_candidates = candidate_scores[:top_k]
        
        # Add explanations for top candidates
        for result in top_candidates:
            explanation = self._generate_explanation(
                result['candidate'], job_description, job_title,
                result['compatibility_score'], result['confidence']
            )
            result['explanation'] = explanation
        
        return top_candidates
    
    def get_model_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics of the trained model.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_trained:
            return {}
        
        metrics = {
            'training_metrics': self.ml_model.training_metrics,
            'model_type': self.ml_model.model_type,
            'feature_importance': self.ml_model.get_feature_importance(),
            'is_trained': self.is_trained
        }
        
        return metrics
    
    def retrain_with_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrain the model with user feedback.
        
        Args:
            feedback_data: List of feedback dictionaries with candidate, job, and actual_score
            
        Returns:
            Retraining results
        """
        logger.info(f"Retraining model with {len(feedback_data)} feedback samples")
        
        # Extract features from feedback data
        X_feedback = []
        y_feedback = []
        
        for feedback in feedback_data:
            candidate = feedback['candidate']
            job = feedback['job']
            actual_score = feedback['actual_score']
            
            # Extract features
            candidate_features = self.feature_extractor.extract_candidate_features(candidate)
            job_features = self.feature_extractor.extract_job_features(
                job.get('description', ''),
                job.get('title', ''),
                job.get('min_experience', 0),
                job.get('location', ''),
                job.get('salary_range')
            )
            
            combined_features = np.concatenate([candidate_features, job_features])
            X_feedback.append(combined_features)
            y_feedback.append(actual_score)
        
        X_feedback = np.array(X_feedback)
        y_feedback = np.array(y_feedback)
        
        # Retrain model (incremental learning would be better, but this is simpler)
        retraining_metrics = self.ml_model.train(X_feedback, y_feedback, validation_split=0.2)
        
        # Save updated model
        self._save_models()
        
        logger.info("Model retrained successfully with feedback")
        
        return {
            'retraining_metrics': retraining_metrics,
            'feedback_samples': len(feedback_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_features_from_pairs(self, training_pairs: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from training pairs.
        
        Args:
            training_pairs: List of (candidate, job, score) tuples
            
        Returns:
            Feature matrix X and target vector y
        """
        X = []
        y = []
        
        for candidate, job, score in training_pairs:
            try:
                # Extract candidate features
                candidate_features = self.feature_extractor.extract_candidate_features(candidate)
                
                # Extract job features
                job_features = self.feature_extractor.extract_job_features(
                    job.get('description', ''),
                    job.get('title', ''),
                    job.get('min_experience', 0),
                    job.get('location', ''),
                    job.get('salary_range')
                )
                
                # Combine features
                combined_features = np.concatenate([candidate_features, job_features])
                
                X.append(combined_features)
                y.append(score)
                
            except Exception as e:
                logger.warning(f"Error extracting features from training pair: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def _generate_explanation(self, candidate: Dict[str, Any], 
                            job_description: str, job_title: str,
                            score: float, confidence: float,
                            feature_importance: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate human-readable explanation for the compatibility score.
        
        Args:
            candidate: Candidate profile
            job_description: Job description
            job_title: Job title
            score: Compatibility score
            confidence: Confidence score
            feature_importance: Feature importance dictionary
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            'overall_score': score,
            'confidence': confidence,
            'score_interpretation': self._interpret_score(score),
            'key_factors': [],
            'recommendations': []
        }
        
        # Analyze skills match
        candidate_skills = [skill['name'].lower() for skill in candidate.get('skills', [])]
        job_text_lower = (job_description + " " + job_title).lower()
        
        # Find matching skills
        matching_skills = []
        for skill in candidate_skills:
            if skill in job_text_lower:
                matching_skills.append(skill)
        
        if matching_skills:
            explanation['key_factors'].append({
                'factor': 'Skill Match',
                'impact': 'positive',
                'details': f"Candidate has {len(matching_skills)} relevant skills: {', '.join(matching_skills[:5])}"
            })
        
        # Analyze experience
        candidate_exp = candidate.get('years_of_experience', 0)
        if candidate_exp > 0:
            explanation['key_factors'].append({
                'factor': 'Experience',
                'impact': 'positive' if candidate_exp >= 3 else 'neutral',
                'details': f"Candidate has {candidate_exp} years of experience"
            })
        
        # Analyze title match
        candidate_title = candidate.get('title', '').lower()
        if any(word in candidate_title for word in job_title.lower().split()):
            explanation['key_factors'].append({
                'factor': 'Role Alignment',
                'impact': 'positive',
                'details': f"Candidate's current role ({candidate.get('title', '')}) aligns with job title"
            })
        
        # Generate recommendations
        if score < 0.6:
            explanation['recommendations'].append("Consider additional skill development or training")
        if score >= 0.8:
            explanation['recommendations'].append("Excellent match - recommend for interview")
        elif score >= 0.6:
            explanation['recommendations'].append("Good match - worth considering")
        
        return explanation
    
    def _interpret_score(self, score: float) -> str:
        """Interpret compatibility score."""
        if score >= 0.9:
            return "Excellent match"
        elif score >= 0.8:
            return "Very good match"
        elif score >= 0.7:
            return "Good match"
        elif score >= 0.6:
            return "Fair match"
        elif score >= 0.4:
            return "Poor match"
        else:
            return "Very poor match"
    
    def _save_models(self):
        """Save trained models to disk."""
        # Save ML model
        ml_model_path = os.path.join(self.model_path, 'ml_model.joblib')
        self.ml_model.save_model(ml_model_path)
        
        # Save feature extractor
        feature_extractor_path = os.path.join(self.model_path, 'feature_extractor.joblib')
        import joblib
        joblib.dump(self.feature_extractor, feature_extractor_path)
        
        logger.info(f"Models saved to {self.model_path}")
    
    def _load_models(self):
        """Load trained models from disk."""
        try:
            # Load ML model
            ml_model_path = os.path.join(self.model_path, 'ml_model.joblib')
            if os.path.exists(ml_model_path):
                self.ml_model.load_model(ml_model_path)
            
            # Load feature extractor
            feature_extractor_path = os.path.join(self.model_path, 'feature_extractor.joblib')
            if os.path.exists(feature_extractor_path):
                import joblib
                self.feature_extractor = joblib.load(feature_extractor_path)
            
            self.is_trained = True
            logger.info(f"Models loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'data_path': self.data_path,
            'candidates_loaded': len(self.candidates_cache) if self.candidates_cache else 0,
            'jobs_loaded': len(self.jobs_cache) if self.jobs_cache else 0
        } 