from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.schemas.candidate import Candidate, CandidateCreate, CandidateResponse, JobDescription
from app.services.recruitment import RecruitmentService

router = APIRouter()
recruitment_service = RecruitmentService()

@router.post("/find-candidates", response_model=CandidateResponse)
async def find_candidates(
    job: JobDescription,
    db: Session = Depends(get_db)
):
    """
    Find the top 5 matching candidates for a given job description.
    """
    try:
        return await recruitment_service.find_matching_candidates(job.description, job.use_ml, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model(
    job: JobDescription,
    candidates: List[dict],
    match_scores: List[float],
    db: Session = Depends(get_db)
):
    """
    Train the ML model with new data.
    """
    try:
        await recruitment_service.train_model(job.description, candidates, match_scores, db)
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 