from typing import List, Dict, Any, Optional
from datetime import datetime

class CandidateBase:
    def __init__(self, name: str, email: str, skills: List[str], experience: int, education: str):
        self.name = name
        self.email = email
        self.skills = skills
        self.experience = experience
        self.education = education

class CandidateCreate(CandidateBase):
    pass

class Candidate(CandidateBase):
    def __init__(self, id: int, created_at: datetime, updated_at: datetime, **kwargs):
        super().__init__(**kwargs)
        self.id = id
        self.created_at = created_at
        self.updated_at = updated_at

class JobDescription:
    def __init__(self, description: str, use_ml: bool = True):
        self.description = description
        self.use_ml = use_ml

class CandidateResponse:
    def __init__(self, candidates: List[Dict[str, Any]], match_scores: List[float]):
        self.candidates = candidates
        self.match_scores = match_scores 