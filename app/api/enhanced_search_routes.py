"""
API routes for enhanced candidate search and matching functionality.
"""

from flask import Blueprint, request, jsonify
import logging
from functools import wraps

from auth import token_required
from matching.enhanced_search_service import EnhancedSearchService

logger = logging.getLogger(__name__)

# Create blueprint
enhanced_search_bp = Blueprint('enhanced_search', __name__, url_prefix='/api/enhanced-search')

# Initialize enhanced search service
enhanced_search_service = EnhancedSearchService()


def handle_errors(f):
    """Decorator to handle common errors in API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Invalid input',
                'message': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    return decorated_function


@enhanced_search_bp.route('/candidates', methods=['POST'])
@token_required
@handle_errors
def search_candidates_enhanced(current_user):
    """
    Enhanced candidate search with AI-powered matching.
    
    Expected JSON payload:
    {
        "job_description": "Full job description text",
        "job_title": "Software Engineer (optional)",
        "company_name": "Tech Corp (optional)",
        "custom_weights": {
            "skills": 0.5,
            "experience": 0.3,
            "education": 0.1,
            "cultural_fit": 0.1
        },
        "limit": 50,
        "min_score": 0.5
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    job_description = data.get('job_description')
    if not job_description or not job_description.strip():
        return jsonify({'error': 'job_description is required'}), 400
    
    # Extract optional parameters
    job_title = data.get('job_title', '')
    company_name = data.get('company_name', '')
    custom_weights = data.get('custom_weights')
    limit = data.get('limit', 50)
    min_score = data.get('min_score', 0.5)
    
    # Validate parameters
    if not isinstance(limit, int) or limit < 1 or limit > 200:
        return jsonify({'error': 'limit must be an integer between 1 and 200'}), 400
    
    if not isinstance(min_score, (int, float)) or min_score < 0 or min_score > 1:
        return jsonify({'error': 'min_score must be a number between 0 and 1'}), 400
    
    # Validate custom weights if provided
    if custom_weights:
        required_weight_keys = ['skills', 'experience', 'education', 'cultural_fit']
        if not all(key in custom_weights for key in required_weight_keys):
            return jsonify({
                'error': f'custom_weights must contain all keys: {required_weight_keys}'
            }), 400
        
        weight_sum = sum(custom_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            return jsonify({
                'error': 'custom_weights must sum to 1.0'
            }), 400
    
    logger.info(f"Enhanced search request: job_title='{job_title}', limit={limit}, min_score={min_score}")
    
    # Perform enhanced search
    results = enhanced_search_service.search_candidates_enhanced(
        job_description=job_description,
        job_title=job_title,
        company_name=company_name,
        custom_weights=custom_weights,
        limit=limit,
        min_score=min_score
    )
    
    # Convert MatchScore objects to dictionaries for JSON serialization
    for candidate in results.get('candidates', []):
        if 'match_score' in candidate:
            match_score = candidate['match_score']
            candidate['match_score'] = {
                'overall_score': match_score.overall_score,
                'confidence': match_score.confidence,
                'skills_score': match_score.skills_score,
                'experience_score': match_score.experience_score,
                'education_score': match_score.education_score,
                'cultural_fit_score': match_score.cultural_fit_score,
                'strengths': match_score.strengths,
                'weaknesses': match_score.weaknesses,
                'recommendations': match_score.recommendations,
                'calculated_at': match_score.calculated_at.isoformat() if match_score.calculated_at else None
            }
    
    return jsonify(results)


@enhanced_search_bp.route('/skill-recommendations', methods=['POST'])
@token_required
@handle_errors
def get_skill_recommendations(current_user):
    """
    Get skill recommendations based on job requirements and current skills.
    
    Expected JSON payload:
    {
        "job_description": "Full job description text",
        "current_skills": ["Python", "JavaScript", "React"]
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    job_description = data.get('job_description')
    current_skills = data.get('current_skills', [])
    
    if not job_description or not job_description.strip():
        return jsonify({'error': 'job_description is required'}), 400
    
    if not isinstance(current_skills, list):
        return jsonify({'error': 'current_skills must be a list of strings'}), 400
    
    logger.info(f"Skill recommendations request: {len(current_skills)} current skills")
    
    # Get skill recommendations
    recommendations = enhanced_search_service.get_skill_recommendations(
        job_description=job_description,
        current_skills=current_skills
    )
    
    return jsonify(recommendations)


@enhanced_search_bp.route('/market-analysis', methods=['POST'])
@token_required
@handle_errors
def analyze_market_demand(current_user):
    """
    Analyze market demand for specific skills.
    
    Expected JSON payload:
    {
        "skills": ["Python", "React", "AWS"],
        "location": "global"
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    skills = data.get('skills', [])
    location = data.get('location', 'global')
    
    if not isinstance(skills, list) or not skills:
        return jsonify({'error': 'skills must be a non-empty list of strings'}), 400
    
    if len(skills) > 20:
        return jsonify({'error': 'Maximum 20 skills allowed per request'}), 400
    
    if not isinstance(location, str):
        return jsonify({'error': 'location must be a string'}), 400
    
    logger.info(f"Market analysis request: {len(skills)} skills, location='{location}'")
    
    # Analyze market demand
    analysis = enhanced_search_service.analyze_market_demand(
        skills=skills,
        location=location
    )
    
    return jsonify(analysis)


@enhanced_search_bp.route('/parse-job', methods=['POST'])
@token_required
@handle_errors
def parse_job_description(current_user):
    """
    Parse a job description and extract structured information.
    
    Expected JSON payload:
    {
        "job_description": "Full job description text",
        "job_title": "Software Engineer (optional)",
        "company_name": "Tech Corp (optional)"
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    job_description = data.get('job_description')
    if not job_description or not job_description.strip():
        return jsonify({'error': 'job_description is required'}), 400
    
    job_title = data.get('job_title', '')
    company_name = data.get('company_name', '')
    
    logger.info(f"Job parsing request: job_title='{job_title}', company='{company_name}'")
    
    # Parse job description
    parsed_job = enhanced_search_service.job_parser.parse_job_description(
        job_text=job_description,
        job_title=job_title,
        company_name=company_name
    )
    
    # Convert to dictionary for JSON serialization
    result = parsed_job.__dict__.copy()
    
    # Convert JobRequirement objects to dictionaries
    result['required_skills'] = [
        {
            'skill': req.skill,
            'importance': req.importance,
            'proficiency_level': req.proficiency_level,
            'years_experience': req.years_experience,
            'context': req.context
        }
        for req in parsed_job.required_skills
    ]
    
    result['preferred_skills'] = [
        {
            'skill': req.skill,
            'importance': req.importance,
            'proficiency_level': req.proficiency_level,
            'years_experience': req.years_experience,
            'context': req.context
        }
        for req in parsed_job.preferred_skills
    ]
    
    return jsonify(result)


@enhanced_search_bp.route('/score-candidate', methods=['POST'])
@token_required
@handle_errors
def score_single_candidate(current_user):
    """
    Score a single candidate against job requirements.
    
    Expected JSON payload:
    {
        "candidate_data": {
            "id": 123,
            "skills": ["Python", "React"],
            "experiences": [...],
            "education": [...]
        },
        "job_requirements": {
            "required_skills": ["Python", "JavaScript"],
            "preferred_skills": ["React", "AWS"],
            "experience_requirements": {"minimum_years": 3},
            "education_requirements": {"minimum_degree": "bachelor"}
        },
        "custom_weights": {
            "skills": 0.5,
            "experience": 0.3,
            "education": 0.1,
            "cultural_fit": 0.1
        }
    }
    """
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Validate required fields
    candidate_data = data.get('candidate_data')
    job_requirements = data.get('job_requirements')
    
    if not candidate_data:
        return jsonify({'error': 'candidate_data is required'}), 400
    
    if not job_requirements:
        return jsonify({'error': 'job_requirements is required'}), 400
    
    custom_weights = data.get('custom_weights')
    
    # Validate custom weights if provided
    if custom_weights:
        required_weight_keys = ['skills', 'experience', 'education', 'cultural_fit']
        if not all(key in custom_weights for key in required_weight_keys):
            return jsonify({
                'error': f'custom_weights must contain all keys: {required_weight_keys}'
            }), 400
        
        weight_sum = sum(custom_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            return jsonify({
                'error': 'custom_weights must sum to 1.0'
            }), 400
    
    logger.info(f"Single candidate scoring request: candidate_id={candidate_data.get('id', 'unknown')}")
    
    # Score the candidate
    if custom_weights:
        from matching.core.scorer import ScoringWeights, AdvancedCandidateScorer
        weights = ScoringWeights(**custom_weights)
        scorer = AdvancedCandidateScorer(weights)
    else:
        scorer = enhanced_search_service.scorer
    
    match_score = scorer.score_candidate(candidate_data, job_requirements)
    
    # Convert to dictionary for JSON serialization
    result = {
        'overall_score': match_score.overall_score,
        'confidence': match_score.confidence,
        'skills_score': match_score.skills_score,
        'experience_score': match_score.experience_score,
        'education_score': match_score.education_score,
        'cultural_fit_score': match_score.cultural_fit_score,
        'strengths': match_score.strengths,
        'weaknesses': match_score.weaknesses,
        'recommendations': match_score.recommendations,
        'calculated_at': match_score.calculated_at.isoformat() if match_score.calculated_at else None,
        'model_version': match_score.model_version
    }
    
    # Add detailed breakdowns if available
    if match_score.skill_matches:
        result['skill_matches'] = [
            {
                'required_skill': sm.required_skill,
                'candidate_skill': sm.candidate_skill,
                'similarity_score': sm.similarity_score,
                'proficiency_level': sm.proficiency_level,
                'years_experience': sm.years_experience,
                'is_exact_match': sm.is_exact_match,
                'confidence': sm.confidence
            }
            for sm in match_score.skill_matches
        ]
    
    return jsonify(result)


@enhanced_search_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the enhanced search service."""
    
    try:
        # Test basic functionality
        test_result = enhanced_search_service.semantic_matcher.calculate_similarity("python", "python")
        
        return jsonify({
            'status': 'healthy',
            'service': 'enhanced_search',
            'version': '1.0.0',
            'semantic_matcher': 'operational' if test_result == 1.0 else 'degraded',
            'timestamp': logger.handlers[0].formatter.formatTime(logger.makeRecord(
                'test', 20, '', 0, '', (), None
            )) if logger.handlers else None
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'enhanced_search',
            'error': str(e)
        }), 503


# Error handlers for the blueprint
@enhanced_search_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@enhanced_search_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405


@enhanced_search_bp.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'error': 'Payload too large',
        'message': 'The request payload is too large'
    }), 413 