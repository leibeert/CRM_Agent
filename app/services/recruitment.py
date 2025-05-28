from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.core.agent import RecruitmentAgent
from config.settings import settings

class RecruitmentService:
    def __init__(self):
        self.agent = RecruitmentAgent()

    async def find_matching_candidates(
        self,
        job_description: str,
        use_ml: bool,  # Keeping this for API compatibility
        db: Session
    ) -> Dict[str, Any]:
        """
        Find matching candidates for a job description using LangChain agent
        """
        # Get candidate matches from the agent
        candidates = self.agent.find_matching_candidates(job_description)
        
        # Get match scores from the agent's analysis
        match_scores = [0.8] * len(candidates)  # Default score
        
        return {
            "candidates": candidates[:5],
            "match_scores": match_scores[:5]
        }

    async def train_model(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]],
        match_scores: List[float],
        db: Session
    ) -> None:
        """
        This method is kept for API compatibility but doesn't do anything
        since we're not using ML model yet
        """
        pass 