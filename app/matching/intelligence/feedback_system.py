"""
Feedback system for learning and improving matching algorithms.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    MATCH_QUALITY = "match_quality"
    SKILL_RELEVANCE = "skill_relevance"
    EXPERIENCE_MATCH = "experience_match"
    CULTURAL_FIT = "cultural_fit"
    OVERALL_SATISFACTION = "overall_satisfaction"


class FeedbackRating(Enum):
    """Rating scale for feedback."""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1


@dataclass
class MatchingFeedback:
    """Represents feedback on a matching result."""
    feedback_id: str
    user_id: int
    candidate_id: int
    job_description_hash: str
    feedback_type: FeedbackType
    rating: FeedbackRating
    comments: Optional[str]
    match_score: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class FeedbackAnalysis:
    """Analysis results from feedback data."""
    total_feedback_count: int
    average_rating: float
    rating_distribution: Dict[int, int]
    common_issues: List[str]
    improvement_suggestions: List[str]
    confidence_score: float


class MatchingFeedbackSystem:
    """System for collecting and analyzing feedback on matching results."""
    
    def __init__(self):
        # In-memory storage for demo - in production, use database
        self.feedback_storage: List[MatchingFeedback] = []
        self.feedback_analytics = {}
        
        # Learning parameters
        self.min_feedback_for_learning = 10
        self.feedback_weight_decay = 0.9  # Newer feedback has more weight
        
    def submit_feedback(self, user_id: int, candidate_id: int, 
                       job_description: str, feedback_type: FeedbackType,
                       rating: FeedbackRating, comments: Optional[str] = None,
                       match_score: float = 0.0, metadata: Dict[str, Any] = None) -> str:
        """Submit feedback for a matching result."""
        
        # Generate feedback ID
        feedback_id = f"fb_{user_id}_{candidate_id}_{int(datetime.now().timestamp())}"
        
        # Create job description hash for grouping
        job_hash = str(hash(job_description.lower().strip()))
        
        feedback = MatchingFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            candidate_id=candidate_id,
            job_description_hash=job_hash,
            feedback_type=feedback_type,
            rating=rating,
            comments=comments,
            match_score=match_score,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.feedback_storage.append(feedback)
        
        # Update analytics
        self._update_analytics(feedback)
        
        logger.info(f"Feedback submitted: {feedback_id} - {feedback_type.value}: {rating.value}")
        
        return feedback_id
    
    def get_feedback_for_candidate(self, candidate_id: int) -> List[MatchingFeedback]:
        """Get all feedback for a specific candidate."""
        
        return [fb for fb in self.feedback_storage if fb.candidate_id == candidate_id]
    
    def get_feedback_for_job(self, job_description: str) -> List[MatchingFeedback]:
        """Get all feedback for a specific job description."""
        
        job_hash = str(hash(job_description.lower().strip()))
        return [fb for fb in self.feedback_storage if fb.job_description_hash == job_hash]
    
    def analyze_feedback(self, feedback_type: Optional[FeedbackType] = None,
                        days_back: int = 30) -> FeedbackAnalysis:
        """Analyze feedback data to identify patterns and issues."""
        
        # Filter feedback by type and date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_feedback = [
            fb for fb in self.feedback_storage
            if fb.created_at >= cutoff_date and
            (feedback_type is None or fb.feedback_type == feedback_type)
        ]
        
        if not filtered_feedback:
            return FeedbackAnalysis(
                total_feedback_count=0,
                average_rating=0.0,
                rating_distribution={},
                common_issues=[],
                improvement_suggestions=[],
                confidence_score=0.0
            )
        
        # Calculate statistics
        ratings = [fb.rating.value for fb in filtered_feedback]
        avg_rating = sum(ratings) / len(ratings)
        
        # Rating distribution
        rating_dist = {}
        for rating in ratings:
            rating_dist[rating] = rating_dist.get(rating, 0) + 1
        
        # Identify common issues from comments
        common_issues = self._extract_common_issues(filtered_feedback)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(filtered_feedback, avg_rating)
        
        # Calculate confidence based on feedback volume and consistency
        confidence = min(1.0, len(filtered_feedback) / 100)  # Max confidence at 100+ feedback
        rating_variance = sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
        confidence *= max(0.1, 1.0 - rating_variance / 4)  # Reduce confidence for high variance
        
        return FeedbackAnalysis(
            total_feedback_count=len(filtered_feedback),
            average_rating=avg_rating,
            rating_distribution=rating_dist,
            common_issues=common_issues,
            improvement_suggestions=improvement_suggestions,
            confidence_score=confidence
        )
    
    def _extract_common_issues(self, feedback_list: List[MatchingFeedback]) -> List[str]:
        """Extract common issues from feedback comments."""
        
        issues = []
        
        # Count negative feedback
        negative_feedback = [fb for fb in feedback_list if fb.rating.value <= 2]
        
        if len(negative_feedback) > len(feedback_list) * 0.3:  # More than 30% negative
            issues.append("High rate of negative feedback")
        
        # Analyze comments for keywords
        all_comments = [fb.comments for fb in feedback_list if fb.comments]
        
        if all_comments:
            comment_text = " ".join(all_comments).lower()
            
            issue_keywords = {
                "skills don't match": ["skill", "mismatch", "irrelevant", "wrong"],
                "experience level wrong": ["experience", "junior", "senior", "level"],
                "poor cultural fit": ["culture", "fit", "personality", "team"],
                "overqualified candidates": ["overqualified", "too experienced", "senior"],
                "underqualified candidates": ["underqualified", "inexperienced", "junior"]
            }
            
            for issue, keywords in issue_keywords.items():
                if any(keyword in comment_text for keyword in keywords):
                    issues.append(issue)
        
        return issues[:5]  # Return top 5 issues
    
    def _generate_improvement_suggestions(self, feedback_list: List[MatchingFeedback], 
                                        avg_rating: float) -> List[str]:
        """Generate suggestions for improving matching based on feedback."""
        
        suggestions = []
        
        if avg_rating < 3.0:
            suggestions.append("Overall matching quality needs significant improvement")
        elif avg_rating < 3.5:
            suggestions.append("Consider adjusting matching algorithm weights")
        
        # Analyze by feedback type
        type_ratings = {}
        for fb in feedback_list:
            if fb.feedback_type not in type_ratings:
                type_ratings[fb.feedback_type] = []
            type_ratings[fb.feedback_type].append(fb.rating.value)
        
        for feedback_type, ratings in type_ratings.items():
            avg_type_rating = sum(ratings) / len(ratings)
            
            if avg_type_rating < 3.0:
                if feedback_type == FeedbackType.SKILL_RELEVANCE:
                    suggestions.append("Improve skill matching algorithm - consider semantic similarity")
                elif feedback_type == FeedbackType.EXPERIENCE_MATCH:
                    suggestions.append("Refine experience level matching criteria")
                elif feedback_type == FeedbackType.CULTURAL_FIT:
                    suggestions.append("Enhance cultural fit assessment methods")
        
        # Check for score vs rating correlation
        scores_and_ratings = [(fb.match_score, fb.rating.value) for fb in feedback_list if fb.match_score > 0]
        
        if len(scores_and_ratings) > 5:
            # Simple correlation check
            high_score_low_rating = [
                (score, rating) for score, rating in scores_and_ratings 
                if score > 0.8 and rating <= 2
            ]
            
            if len(high_score_low_rating) > len(scores_and_ratings) * 0.2:
                suggestions.append("High match scores don't correlate with user satisfaction - review scoring algorithm")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _update_analytics(self, feedback: MatchingFeedback):
        """Update real-time analytics with new feedback."""
        
        # Update overall analytics
        if 'overall' not in self.feedback_analytics:
            self.feedback_analytics['overall'] = {
                'count': 0,
                'total_rating': 0,
                'avg_rating': 0.0
            }
        
        overall = self.feedback_analytics['overall']
        overall['count'] += 1
        overall['total_rating'] += feedback.rating.value
        overall['avg_rating'] = overall['total_rating'] / overall['count']
        
        # Update by feedback type
        type_key = feedback.feedback_type.value
        if type_key not in self.feedback_analytics:
            self.feedback_analytics[type_key] = {
                'count': 0,
                'total_rating': 0,
                'avg_rating': 0.0
            }
        
        type_analytics = self.feedback_analytics[type_key]
        type_analytics['count'] += 1
        type_analytics['total_rating'] += feedback.rating.value
        type_analytics['avg_rating'] = type_analytics['total_rating'] / type_analytics['count']
    
    def get_candidate_feedback_score(self, candidate_id: int) -> Dict[str, Any]:
        """Get aggregated feedback score for a candidate."""
        
        candidate_feedback = self.get_feedback_for_candidate(candidate_id)
        
        if not candidate_feedback:
            return {
                'candidate_id': candidate_id,
                'feedback_count': 0,
                'average_rating': 0.0,
                'recommendation': 'No feedback available'
            }
        
        ratings = [fb.rating.value for fb in candidate_feedback]
        avg_rating = sum(ratings) / len(ratings)
        
        # Generate recommendation based on feedback
        if avg_rating >= 4.0:
            recommendation = 'Highly recommended based on positive feedback'
        elif avg_rating >= 3.5:
            recommendation = 'Good candidate with positive feedback'
        elif avg_rating >= 3.0:
            recommendation = 'Average candidate, review feedback details'
        else:
            recommendation = 'Consider carefully, has received negative feedback'
        
        return {
            'candidate_id': candidate_id,
            'feedback_count': len(candidate_feedback),
            'average_rating': avg_rating,
            'rating_distribution': {
                rating: len([fb for fb in candidate_feedback if fb.rating.value == rating])
                for rating in range(1, 6)
            },
            'recommendation': recommendation,
            'recent_feedback': [
                {
                    'type': fb.feedback_type.value,
                    'rating': fb.rating.value,
                    'comments': fb.comments,
                    'date': fb.created_at.isoformat()
                }
                for fb in sorted(candidate_feedback, key=lambda x: x.created_at, reverse=True)[:3]
            ]
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights for improving the matching algorithm."""
        
        if len(self.feedback_storage) < self.min_feedback_for_learning:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {self.min_feedback_for_learning} feedback entries for learning',
                'current_count': len(self.feedback_storage)
            }
        
        # Analyze patterns in feedback
        analysis = self.analyze_feedback()
        
        # Identify algorithm adjustments
        adjustments = []
        
        if analysis.average_rating < 3.5:
            adjustments.append({
                'component': 'overall_algorithm',
                'adjustment': 'reduce_confidence_threshold',
                'reason': 'Low average user satisfaction'
            })
        
        # Check if high-scoring matches get poor feedback
        high_score_poor_feedback = [
            fb for fb in self.feedback_storage
            if fb.match_score > 0.8 and fb.rating.value <= 2
        ]
        
        if len(high_score_poor_feedback) > len(self.feedback_storage) * 0.15:
            adjustments.append({
                'component': 'scoring_weights',
                'adjustment': 'rebalance_skill_vs_experience',
                'reason': 'High scores not correlating with user satisfaction'
            })
        
        return {
            'status': 'ready_for_learning',
            'feedback_analysis': analysis.__dict__,
            'suggested_adjustments': adjustments,
            'learning_confidence': analysis.confidence_score
        }
    
    def export_feedback_data(self, format: str = 'json') -> Dict[str, Any]:
        """Export feedback data for external analysis."""
        
        if format == 'json':
            return {
                'feedback_entries': [
                    {
                        'feedback_id': fb.feedback_id,
                        'user_id': fb.user_id,
                        'candidate_id': fb.candidate_id,
                        'feedback_type': fb.feedback_type.value,
                        'rating': fb.rating.value,
                        'comments': fb.comments,
                        'match_score': fb.match_score,
                        'created_at': fb.created_at.isoformat(),
                        'metadata': fb.metadata
                    }
                    for fb in self.feedback_storage
                ],
                'analytics': self.feedback_analytics,
                'export_timestamp': datetime.now().isoformat()
            }
        
        return {}
    
    def clear_old_feedback(self, days_to_keep: int = 365):
        """Clear feedback older than specified days."""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        initial_count = len(self.feedback_storage)
        self.feedback_storage = [
            fb for fb in self.feedback_storage
            if fb.created_at >= cutoff_date
        ]
        
        removed_count = initial_count - len(self.feedback_storage)
        
        logger.info(f"Removed {removed_count} old feedback entries")
        
        return removed_count 