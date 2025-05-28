import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from typing import List, Dict, Any

class CandidateRanker:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.model_path = "models/candidate_ranker.joblib"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Try to load existing model
        try:
            self.load_model()
        except:
            pass

    def train(self, candidates: List[Dict[str, Any]], requirements: Dict[str, Any], match_scores: List[float]):
        """
        Train the model on candidate data and their match scores
        """
        # Convert candidates and requirements to feature vectors
        X = self._prepare_features(candidates, requirements)
        y = np.array(match_scores)
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Save the model
        self.save_model()

    def predict(self, candidates: List[Dict[str, Any]], requirements: Dict[str, Any]) -> List[float]:
        """
        Predict match scores for candidates
        """
        if not self.is_trained:
            return [0.8] * len(candidates)  # Default score if model is not trained
            
        # Convert candidates and requirements to feature vectors
        X = self._prepare_features(candidates, requirements)
        
        # Make predictions
        return self.model.predict(X).tolist()

    def save_model(self):
        """
        Save the trained model to disk
        """
        if self.is_trained:
            joblib.dump(self.model, self.model_path)

    def load_model(self):
        """
        Load a trained model from disk
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True

    def _prepare_features(self, candidates: List[Dict[str, Any]], requirements: Dict[str, Any]) -> np.ndarray:
        """
        Convert candidates and requirements to feature vectors
        This is a simple implementation - you might want to enhance this
        """
        features = []
        for candidate in candidates:
            # Extract features from candidate
            candidate_features = [
                len(candidate.get('skills', [])),  # Number of skills
                candidate.get('experience', 0),    # Years of experience
                len(candidate.get('education', '')),  # Education length
            ]
            features.append(candidate_features)
        
        return np.array(features) 