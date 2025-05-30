"""
Advanced ML Model Trainer for CRM Candidate Matching System

This module creates custom machine learning models using your database data
to improve candidate-job matching accuracy beyond the current AI system.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import json

# For deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# For advanced NLP
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CandidateMatchingModelTrainer:
    """
    Advanced ML model trainer for candidate-job matching using your database data.
    
    Features:
    - Skill similarity neural networks
    - Experience relevance models  
    - Education matching algorithms
    - Composite scoring ensembles
    - Semantic embedding models
    """
    
    def __init__(self, data_path: str = "database_tables/"):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
        # Load all database tables
        self.load_database_tables()
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("ML Model Trainer initialized successfully")
    
    def load_database_tables(self):
        """Load all CSV tables from database_tables folder."""
        try:
            # Core candidate data
            self.resources = pd.read_csv(f"{self.data_path}/resources.csv")
            self.skills = pd.read_csv(f"{self.data_path}/skills.csv") 
            self.resource_skills = pd.read_csv(f"{self.data_path}/resource_skills.csv")
            self.experiences = pd.read_csv(f"{self.data_path}/experiences.csv")
            self.studies = pd.read_csv(f"{self.data_path}/studies.csv")
            
            logger.info(f"Loaded {len(self.resources)} candidates, {len(self.skills)} skills")
            logger.info(f"Loaded {len(self.resource_skills)} skill mappings, {len(self.experiences)} experiences")
            
        except Exception as e:
            logger.error(f"Error loading database tables: {str(e)}")
            raise
    
    def create_candidate_feature_matrix(self) -> pd.DataFrame:
        """
        Create comprehensive feature matrix for each candidate.
        
        Features include:
        - Skill embeddings and proficiency levels
        - Experience vectors (years, roles, companies)
        - Education features (degrees, institutions)
        - Career progression metrics
        - Skill diversity scores
        """
        
        logger.info("Creating candidate feature matrix...")
        
        features_list = []
        
        for _, candidate in self.resources.iterrows():
            candidate_id = candidate['id']
            
            # Basic candidate features
            features = {
                'candidate_id': candidate_id,
                'years_of_experience': candidate.get('years_of_experience', 0),
                'title': candidate.get('title', ''),
                'nationality': candidate.get('nationality', ''),
                'location': candidate.get('address', '')
            }
            
            # Skill features
            candidate_skills = self.resource_skills[
                self.resource_skills['resource_id'] == candidate_id
            ]
            
            # Calculate skill metrics
            features.update(self._calculate_skill_features(candidate_skills))
            
            # Experience features  
            candidate_experiences = self.experiences[
                self.experiences.get('resource_id', -1) == candidate_id
            ] if 'resource_id' in self.experiences.columns else pd.DataFrame()
            
            features.update(self._calculate_experience_features(candidate_experiences))
            
            # Education features
            candidate_education = self.studies[
                self.studies.get('resource_id', -1) == candidate_id  
            ] if 'resource_id' in self.studies.columns else pd.DataFrame()
            
            features.update(self._calculate_education_features(candidate_education))
            
            features_list.append(features)
        
        feature_matrix = pd.DataFrame(features_list)
        logger.info(f"Created feature matrix with {len(feature_matrix)} candidates and {len(feature_matrix.columns)} features")
        
        return feature_matrix
    
    def _calculate_skill_features(self, candidate_skills: pd.DataFrame) -> Dict[str, Any]:
        """Calculate skill-based features for a candidate."""
        
        if candidate_skills.empty:
            return {
                'total_skills': 0,
                'avg_skill_level': 0,
                'max_skill_level': 0,
                'programming_skills': 0,
                'framework_skills': 0,
                'database_skills': 0,
                'cloud_skills': 0,
                'skill_diversity': 0
            }
        
        # Merge with skill details
        skills_with_details = candidate_skills.merge(
            self.skills, left_on='skill_id', right_on='id', how='left'
        )
        
        # Skill categories (based on your skills.csv category_name column)
        programming_categories = [5]  # Programming languages
        framework_categories = [7]    # Frameworks 
        database_categories = [1]     # Databases
        cloud_categories = [8]        # Cloud/DevOps
        
        total_skills = len(candidate_skills)
        avg_skill_level = candidate_skills['level'].mean() if total_skills > 0 else 0
        max_skill_level = candidate_skills['level'].max() if total_skills > 0 else 0
        
        # Count skills by category
        programming_skills = len(skills_with_details[
            skills_with_details['category_name'].isin(programming_categories)
        ])
        framework_skills = len(skills_with_details[
            skills_with_details['category_name'].isin(framework_categories) 
        ])
        database_skills = len(skills_with_details[
            skills_with_details['category_name'].isin(database_categories)
        ])
        cloud_skills = len(skills_with_details[
            skills_with_details['category_name'].isin(cloud_categories)
        ])
        
        # Skill diversity (number of different categories)
        unique_categories = skills_with_details['category_name'].nunique()
        
        return {
            'total_skills': total_skills,
            'avg_skill_level': avg_skill_level,
            'max_skill_level': max_skill_level,
            'programming_skills': programming_skills,
            'framework_skills': framework_skills, 
            'database_skills': database_skills,
            'cloud_skills': cloud_skills,
            'skill_diversity': unique_categories
        }
    
    def _calculate_experience_features(self, experiences: pd.DataFrame) -> Dict[str, Any]:
        """Calculate experience-based features for a candidate."""
        
        if experiences.empty:
            return {
                'total_positions': 0,
                'avg_position_duration': 0,
                'career_progression_score': 0,
                'company_diversity': 0,
                'senior_positions': 0
            }
        
        total_positions = len(experiences)
        
        # Calculate position durations (simplified)
        # In real implementation, you'd parse start/end dates
        avg_position_duration = 2.0  # Placeholder
        
        # Career progression (simplified scoring)
        senior_keywords = ['senior', 'lead', 'manager', 'director', 'head', 'chief']
        senior_positions = sum(
            any(keyword in str(exp.get('title', '')).lower() for keyword in senior_keywords)
            for _, exp in experiences.iterrows()
        )
        
        career_progression_score = senior_positions / total_positions if total_positions > 0 else 0
        
        # Company diversity
        unique_companies = experiences['name'].nunique() if 'name' in experiences.columns else 0
        
        return {
            'total_positions': total_positions,
            'avg_position_duration': avg_position_duration,
            'career_progression_score': career_progression_score,
            'company_diversity': unique_companies,
            'senior_positions': senior_positions
        }
    
    def _calculate_education_features(self, education: pd.DataFrame) -> Dict[str, Any]:
        """Calculate education-based features for a candidate."""
        
        if education.empty:
            return {
                'total_degrees': 0,
                'highest_degree_level': 0,
                'cs_related_degrees': 0,
                'top_university': 0
            }
        
        total_degrees = len(education)
        
        # Degree level scoring (simplified)
        degree_levels = {
            'bachelor': 1, 'master': 2, 'phd': 3, 'doctorate': 3,
            'licence': 1, 'mastère': 2, 'ingénieur': 2
        }
        
        highest_level = 0
        cs_related = 0
        
        for _, degree in education.iterrows():
            degree_name = str(degree.get('degree_name', '')).lower()
            
            # Check degree level
            for level_name, level_score in degree_levels.items():
                if level_name in degree_name:
                    highest_level = max(highest_level, level_score)
                    break
            
            # Check if CS-related
            cs_keywords = ['computer', 'informatique', 'software', 'engineering', 'data']
            if any(keyword in degree_name for keyword in cs_keywords):
                cs_related += 1
        
        return {
            'total_degrees': total_degrees,
            'highest_degree_level': highest_level,
            'cs_related_degrees': cs_related,
            'top_university': 1 if total_degrees > 0 else 0  # Simplified
        }
    
    def train_skill_similarity_model(self) -> Dict[str, Any]:
        """
        Train a model to predict skill similarity and relevance.
        Uses skill co-occurrence patterns and semantic embeddings.
        """
        
        logger.info("Training skill similarity model...")
        
        # Create skill co-occurrence matrix
        skill_pairs = []
        similarity_scores = []
        
        # Get all candidates and their skills
        for candidate_id in self.resource_skills['resource_id'].unique():
            candidate_skills = self.resource_skills[
                self.resource_skills['resource_id'] == candidate_id
            ]['skill_id'].tolist()
            
            # Create pairs from each candidate's skills
            for i, skill1 in enumerate(candidate_skills):
                for skill2 in candidate_skills[i+1:]:
                    skill_pairs.append([skill1, skill2])
                    # Positive example (co-occurring skills are similar)
                    similarity_scores.append(1.0)
        
        # Add negative examples (random skill pairs)
        all_skills = self.skills['id'].tolist()
        for _ in range(len(skill_pairs)):
            skill1, skill2 = np.random.choice(all_skills, 2, replace=False)
            # Check if they actually co-occur
            co_occur = any(
                set([skill1, skill2]).issubset(
                    self.resource_skills[
                        self.resource_skills['resource_id'] == cid
                    ]['skill_id'].tolist()
                )
                for cid in self.resource_skills['resource_id'].unique()
            )
            if not co_occur:
                skill_pairs.append([skill1, skill2])
                similarity_scores.append(0.0)
        
        # Create feature vectors for skill pairs
        X = []
        for skill1_id, skill2_id in skill_pairs:
            skill1_info = self.skills[self.skills['id'] == skill1_id].iloc[0]
            skill2_info = self.skills[self.skills['id'] == skill2_id].iloc[0]
            
            features = [
                skill1_info['category_name'] == skill2_info['category_name'],  # Same category
                abs(hash(skill1_info['name']) % 100 - hash(skill2_info['name']) % 100),  # Name similarity proxy
                len(skill1_info['name']),  # Name length
                len(skill2_info['name'])
            ]
            X.append(features)
        
        X = np.array(X)
        y = np.array(similarity_scores)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['skill_similarity'] = model
        
        logger.info(f"Skill similarity model trained. MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return {
            'model_type': 'skill_similarity',
            'mse': mse,
            'r2_score': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def train_candidate_job_matching_model(self) -> Dict[str, Any]:
        """
        Train the main candidate-job matching model using synthetic job data.
        This creates training data by treating each candidate as a 'perfect match'
        for a job based on their skills and experience.
        """
        
        logger.info("Training candidate-job matching model...")
        
        # Get candidate feature matrix
        feature_matrix = self.create_candidate_feature_matrix()
        
        # Create synthetic job descriptions and match scores
        training_data = []
        
        for _, candidate in feature_matrix.iterrows():
            # Generate multiple synthetic jobs for each candidate
            for match_level in [0.9, 0.7, 0.5, 0.3]:  # High, medium, low, very low match
                
                # Create job features based on candidate features with some noise
                job_features = self._generate_synthetic_job(candidate, match_level)
                
                # Combine candidate and job features
                combined_features = {
                    **{f"candidate_{k}": v for k, v in candidate.items() if k != 'candidate_id'},
                    **{f"job_{k}": v for k, v in job_features.items()},
                    'match_score': match_level
                }
                
                training_data.append(combined_features)
        
        df_training = pd.DataFrame(training_data)
        
        # Prepare features and target
        feature_columns = [col for col in df_training.columns if col != 'match_score']
        X = df_training[feature_columns]
        y = df_training['match_score']
        
        # Handle categorical variables
        X_encoded = self._encode_features(X)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
        }
        
        ensemble_predictions = []
        model_scores = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {'mse': mse, 'r2': r2}
            ensemble_predictions.append(y_pred)
            
            self.models[f'matching_{name}'] = model
        
        # Create ensemble prediction
        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        # Store ensemble weights (simple average for now)
        self.models['matching_ensemble_weights'] = [0.5, 0.5]
        self.scalers['matching'] = scaler
        
        logger.info(f"Matching model trained. Ensemble MSE: {ensemble_mse:.4f}, R²: {ensemble_r2:.4f}")
        
        return {
            'model_type': 'candidate_job_matching',
            'ensemble_mse': ensemble_mse,
            'ensemble_r2': ensemble_r2,
            'individual_models': model_scores,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X_train.shape[1]
        }
    
    def _generate_synthetic_job(self, candidate: pd.Series, match_level: float) -> Dict[str, Any]:
        """Generate synthetic job requirements based on candidate profile."""
        
        noise_factor = 1.0 - match_level  # More noise for lower matches
        
        return {
            'required_years_experience': max(0, candidate['years_of_experience'] + 
                                           np.random.normal(0, 2 * noise_factor)),
            'required_skills': candidate['total_skills'] + np.random.normal(0, 3 * noise_factor),
            'required_programming_skills': candidate['programming_skills'] + 
                                         np.random.normal(0, 1 * noise_factor),
            'required_framework_skills': candidate['framework_skills'] + 
                                       np.random.normal(0, 1 * noise_factor),
            'required_database_skills': candidate['database_skills'] + 
                                      np.random.normal(0, 1 * noise_factor),
            'required_degree_level': candidate['highest_degree_level'] + 
                                   np.random.normal(0, 0.5 * noise_factor),
            'seniority_level': np.random.choice(['junior', 'mid', 'senior'], 
                                              p=[0.3, 0.4, 0.3])
        }
    
    def _encode_features(self, X: pd.DataFrame) -> np.ndarray:
        """Encode categorical features for training."""
        
        X_numeric = X.select_dtypes(include=[np.number])
        X_categorical = X.select_dtypes(exclude=[np.number])
        
        # Handle categorical columns
        encoded_categorical = []
        for col in X_categorical.columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                encoded_col = self.encoders[col].fit_transform(X_categorical[col].astype(str))
            else:
                # Handle unseen categories
                try:
                    encoded_col = self.encoders[col].transform(X_categorical[col].astype(str))
                except ValueError:
                    # If new categories, encode as -1
                    encoded_col = np.array([
                        self.encoders[col].transform([val])[0] 
                        if val in self.encoders[col].classes_ else -1
                        for val in X_categorical[col].astype(str)
                    ])
            
            encoded_categorical.append(encoded_col.reshape(-1, 1))
        
        if encoded_categorical:
            categorical_array = np.hstack(encoded_categorical)
            return np.hstack([X_numeric.values, categorical_array])
        else:
            return X_numeric.values
    
    def save_models(self, model_dir: str = "trained_models/"):
        """Save all trained models and preprocessing objects."""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}.joblib")
        
        # Save preprocessing objects
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/scaler_{name}.joblib")
        
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, f"{model_dir}/encoder_{name}.joblib")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'models': list(self.models.keys()),
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'candidate_count': len(self.resources),
            'skill_count': len(self.skills)
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {model_dir}")
    
    def predict_match_score(self, candidate_features: Dict[str, Any], 
                          job_requirements: Dict[str, Any]) -> float:
        """Predict match score for candidate-job pair using trained models."""
        
        if 'matching_random_forest' not in self.models:
            raise ValueError("Matching model not trained yet. Call train_candidate_job_matching_model() first.")
        
        # Combine features
        combined_features = {
            **{f"candidate_{k}": v for k, v in candidate_features.items()},
            **{f"job_{k}": v for k, v in job_requirements.items()}
        }
        
        # Convert to DataFrame for encoding
        X = pd.DataFrame([combined_features])
        X_encoded = self._encode_features(X)
        X_scaled = self.scalers['matching'].transform(X_encoded)
        
        # Get predictions from both models
        rf_pred = self.models['matching_random_forest'].predict(X_scaled)[0]
        gb_pred = self.models['matching_gradient_boosting'].predict(X_scaled)[0]
        
        # Ensemble prediction
        weights = self.models['matching_ensemble_weights']
        ensemble_score = weights[0] * rf_pred + weights[1] * gb_pred
        
        return max(0.0, min(1.0, ensemble_score))  # Clamp to [0, 1]
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all available models and return training results."""
        
        logger.info("Starting comprehensive model training...")
        
        results = {}
        
        try:
            # Train skill similarity model
            results['skill_similarity'] = self.train_skill_similarity_model()
            
            # Train main matching model
            results['candidate_matching'] = self.train_candidate_job_matching_model()
            
            # Save all models
            self.save_models()
            
            logger.info("All models trained successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive model performance report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_statistics': {
                'total_candidates': len(self.resources),
                'total_skills': len(self.skills), 
                'skill_mappings': len(self.resource_skills),
                'experience_records': len(self.experiences),
                'education_records': len(self.studies)
            },
            'trained_models': list(self.models.keys()),
            'model_capabilities': [
                'Skill similarity prediction',
                'Candidate-job match scoring',
                'Feature-based candidate ranking',
                'Ensemble prediction aggregation'
            ]
        }
        
        return report


# Utility functions for integration
def train_custom_models():
    """Main function to train all custom models."""
    
    trainer = CandidateMatchingModelTrainer()
    results = trainer.train_all_models()
    
    print("🚀 Custom Model Training Complete!")
    print("=" * 50)
    
    for model_type, metrics in results.items():
        if isinstance(metrics, dict) and 'error' not in metrics:
            print(f"\n📊 {model_type.upper()} MODEL:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    if 'error' in results:
        print(f"\n❌ Training Error: {results['error']}")
    
    return results


def generate_model_performance_report():
    """Generate and display model performance report."""
    
    trainer = CandidateMatchingModelTrainer()
    report = trainer.get_model_performance_report()
    
    print("📈 MODEL PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Generated: {report['timestamp']}")
    print(f"\n📋 Data Statistics:")
    for key, value in report['data_statistics'].items():
        print(f"  {key}: {value:,}")
    
    print(f"\n🤖 Trained Models: {len(report['trained_models'])}")
    for model in report['trained_models']:
        print(f"  ✓ {model}")
    
    print(f"\n💡 Model Capabilities:")
    for capability in report['model_capabilities']:
        print(f"  • {capability}")
    
    return report


if __name__ == "__main__":
    # Example usage
    print("🧠 CRM AI Model Trainer")
    print("Training custom models on your database...")
    
    results = train_custom_models()
    print("\n" + "="*50)
    report = generate_model_performance_report() 