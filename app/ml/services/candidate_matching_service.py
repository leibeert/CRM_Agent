from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..models.candidate_matcher import CandidateMatchingModel
from app.db import ArgoteamSessionLocal

class CandidateMatchingService:
    """Service for handling candidate matching operations."""
    
    def __init__(self):
        self.db = ArgoteamSessionLocal()
        self.model = CandidateMatchingModel(self.db)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize and train the model."""
        try:
            self.model.train()
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
    
    def find_matching_candidates(self, job_description: str, required_skills: List[Dict[str, Any]] = None,
                               required_education: Dict[str, Any] = None, min_match_score: float = 0.5) -> List[Dict[str, Any]]:
        """Find matching candidates based on job description and requirements."""
        try:
            input_data = {
                'job_description': job_description,
                'required_skills': required_skills or [],
                'required_education': required_education or {},
                'min_match_score': min_match_score
            }
            return self.model.predict(input_data)
        except Exception as e:
            print(f"Error finding matching candidates: {str(e)}")
            return []
    
    def update_model(self):
        """Update the model with new data."""
        try:
            self.model.train()
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up database connection."""
        self.db.close() 