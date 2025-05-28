import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from datetime import datetime

def calculate_text_similarity(text1: str, text2: str, nlp: spacy.Language) -> float:
    """Calculate semantic similarity between two texts."""
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return cosine_similarity([doc1.vector], [doc2.vector])[0][0]

def calculate_skill_level_score(level: str) -> float:
    """Convert skill level to numerical score."""
    level_scores = {
        'beginner': 0.25,
        'intermediate': 0.5,
        'advanced': 0.75,
        'expert': 1.0
    }
    return level_scores.get(level.lower(), 0.25)

def calculate_duration_score(start_date: datetime, end_date: datetime = None) -> float:
    """Calculate normalized duration score."""
    if not end_date:
        end_date = datetime.now()
    duration = end_date - start_date
    duration_years = duration.days / 365.25
    return min(duration_years / 5, 1.0)  # Cap at 5 years

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

def combine_scores(scores: List[float], weights: List[float] = None) -> float:
    """Combine multiple scores using weighted average."""
    if not scores:
        return 0.0
    if weights is None:
        weights = [1.0] * len(scores)
    if len(scores) != len(weights):
        raise ValueError("Number of scores must match number of weights")
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights) 