"""
Enhanced search service that integrates advanced matching capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio

from .core.scorer import AdvancedCandidateScorer, ScoringWeights, MatchScore
from .core.semantic_matcher import SemanticSkillMatcher
from .parsers.job_parser import IntelligentJobParser, ParsedJobDescription
from .utils.config import get_config
from .utils.cache import get_cache

# Import existing search service
from search.service import SearchService

logger = logging.getLogger(__name__)


class EnhancedSearchService:
    """Enhanced search service with advanced matching capabilities."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        
        # Initialize components
        self.scorer = AdvancedCandidateScorer()
        self.semantic_matcher = SemanticSkillMatcher()
        self.job_parser = IntelligentJobParser()
        
        # Initialize existing search service
        self.base_search_service = SearchService()
        
        logger.info("Enhanced search service initialized")
    
    def search_candidates_enhanced(self, 
                                 job_description: str,
                                 job_title: str = "",
                                 company_name: str = "",
                                 custom_weights: Optional[Dict[str, float]] = None,
                                 limit: int = 50,
                                 min_score: float = 0.5) -> Dict[str, Any]:
        """
        Enhanced candidate search with AI-powered matching.
        
        Args:
            job_description: Raw job description text
            job_title: Job title (optional)
            company_name: Company name (optional)
            custom_weights: Custom scoring weights
            limit: Maximum number of candidates to return
            min_score: Minimum matching score threshold
            
        Returns:
            Dictionary with search results and metadata
        """
        
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Parse job description
            logger.info("Parsing job description...")
            parsed_job = self.job_parser.parse_job_description(
                job_description, job_title, company_name
            )
            
            # Step 2: Get initial candidate pool using existing search
            logger.info("Getting initial candidate pool...")
            initial_candidates = self._get_initial_candidate_pool(parsed_job)
            
            # Step 3: Apply advanced scoring
            logger.info(f"Scoring {len(initial_candidates)} candidates...")
            scored_candidates = self._score_candidates_batch(
                initial_candidates, parsed_job, custom_weights
            )
            
            # Step 4: Filter and rank results
            logger.info("Filtering and ranking results...")
            filtered_candidates = [
                candidate for candidate in scored_candidates 
                if candidate['match_score'].overall_score >= min_score
            ]
            
            # Sort by overall score (descending)
            filtered_candidates.sort(
                key=lambda x: x['match_score'].overall_score, 
                reverse=True
            )
            
            # Limit results
            final_candidates = filtered_candidates[:limit]
            
            # Step 5: Generate search insights
            search_insights = self._generate_search_insights(
                parsed_job, initial_candidates, final_candidates
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'candidates': final_candidates,
                'parsed_job': parsed_job.__dict__,
                'search_insights': search_insights,
                'metadata': {
                    'total_candidates_evaluated': len(initial_candidates),
                    'candidates_returned': len(final_candidates),
                    'processing_time_seconds': processing_time,
                    'min_score_threshold': min_score,
                    'parsing_confidence': parsed_job.confidence_score,
                    'search_timestamp': start_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            # Fallback to basic search
            return self._fallback_search(job_description, limit)
    
    def get_skill_recommendations(self, 
                                job_description: str,
                                current_skills: List[str]) -> Dict[str, Any]:
        """
        Get skill recommendations based on job requirements and current skills.
        
        Args:
            job_description: Job description text
            current_skills: List of current candidate skills
            
        Returns:
            Dictionary with skill recommendations
        """
        
        try:
            # Parse job requirements
            parsed_job = self.job_parser.parse_job_description(job_description)
            
            # Extract required and preferred skills
            required_skills = [req.skill for req in parsed_job.required_skills]
            preferred_skills = [req.skill for req in parsed_job.preferred_skills]
            all_job_skills = required_skills + preferred_skills
            
            # Find missing skills
            missing_required = []
            missing_preferred = []
            
            for req in parsed_job.required_skills:
                best_match = self._find_best_skill_match(req.skill, current_skills)
                if not best_match or best_match[1] < self.config.similarity_threshold:
                    missing_required.append({
                        'skill': req.skill,
                        'importance': 'required',
                        'proficiency_level': req.proficiency_level,
                        'years_experience': req.years_experience
                    })
            
            for req in parsed_job.preferred_skills:
                best_match = self._find_best_skill_match(req.skill, current_skills)
                if not best_match or best_match[1] < self.config.similarity_threshold:
                    missing_preferred.append({
                        'skill': req.skill,
                        'importance': 'preferred',
                        'proficiency_level': req.proficiency_level,
                        'years_experience': req.years_experience
                    })
            
            # Find related skills that could be valuable
            related_skills = self._find_related_skills(all_job_skills, current_skills)
            
            # Generate learning path recommendations
            learning_path = self._generate_learning_path(
                missing_required + missing_preferred, current_skills
            )
            
            return {
                'missing_required_skills': missing_required,
                'missing_preferred_skills': missing_preferred,
                'related_skills': related_skills,
                'learning_path': learning_path,
                'skill_gap_analysis': {
                    'total_required_skills': len(required_skills),
                    'matched_required_skills': len(required_skills) - len(missing_required),
                    'skill_match_percentage': ((len(required_skills) - len(missing_required)) / len(required_skills) * 100) if required_skills else 100
                }
            }
            
        except Exception as e:
            logger.error(f"Skill recommendation failed: {str(e)}")
            return {
                'missing_required_skills': [],
                'missing_preferred_skills': [],
                'related_skills': [],
                'learning_path': [],
                'error': str(e)
            }
    
    def analyze_market_demand(self, skills: List[str], 
                            location: str = "global") -> Dict[str, Any]:
        """
        Analyze market demand for specific skills.
        
        Args:
            skills: List of skills to analyze
            location: Geographic location for analysis
            
        Returns:
            Market demand analysis
        """
        
        try:
            skill_analysis = {}
            
            for skill in skills:
                # Get cached market data
                market_data = self.cache.get_market_data(skill, location)
                
                if not market_data:
                    # Generate market analysis (simplified)
                    market_data = self._analyze_skill_market_demand(skill, location)
                    self.cache.cache_market_data(skill, location, market_data)
                
                skill_analysis[skill] = market_data
            
            # Generate overall market insights
            market_insights = self._generate_market_insights(skill_analysis)
            
            return {
                'skill_analysis': skill_analysis,
                'market_insights': market_insights,
                'location': location,
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")
            return {
                'skill_analysis': {},
                'market_insights': {},
                'error': str(e)
            }
    
    def _get_initial_candidate_pool(self, parsed_job: ParsedJobDescription) -> List[Dict[str, Any]]:
        """Get initial candidate pool using existing search service."""
        
        try:
            # Extract skills for search
            all_skills = []
            for req in parsed_job.required_skills + parsed_job.preferred_skills:
                all_skills.append(req.skill)
            
            # Import SearchQuery with absolute import to avoid circular imports
            from search.schemas import SearchQuery, SkillFilter
            
            # Create SearchQuery object with proper structure
            skill_filters = []
            for skill in all_skills[:10]:  # Limit to top 10 skills
                skill_filters.append(SkillFilter(skill_name=skill))
            
            search_query = SearchQuery(
                keywords=" ".join(all_skills[:5]),  # Use top 5 skills as keywords
                skills=skill_filters,
                page_size=200,  # Get larger pool for scoring
                sort_by='match_score',
                sort_order='desc'
            )
            
            # Call existing search
            results = self.base_search_service.search_candidates(search_query)
            
            # Convert CandidateResponse objects to dictionaries
            candidates = []
            for candidate in results.candidates:
                candidate_dict = {
                    'id': candidate.id,
                    'first_name': candidate.first_name,
                    'last_name': candidate.last_name,
                    'email': candidate.email,
                    'phone_number': candidate.phone_number,
                    'match_score': candidate.match_score,
                    'skills': candidate.skills,
                    'experience': candidate.experience,
                    'education': candidate.education
                }
                candidates.append(candidate_dict)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get initial candidate pool: {str(e)}")
            return []
    
    def _score_candidates_batch(self, 
                              candidates: List[Dict[str, Any]], 
                              parsed_job: ParsedJobDescription,
                              custom_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Score a batch of candidates against job requirements."""
        
        # Set up custom weights if provided
        if custom_weights:
            weights = ScoringWeights(**custom_weights)
            scorer = AdvancedCandidateScorer(weights)
        else:
            scorer = self.scorer
        
        # Convert parsed job to requirements format
        job_requirements = self._convert_parsed_job_to_requirements(parsed_job)
        
        scored_candidates = []
        
        for candidate in candidates:
            try:
                # Score the candidate
                match_score = scorer.score_candidate(candidate, job_requirements)
                
                # Add score to candidate data
                candidate_with_score = candidate.copy()
                candidate_with_score['match_score'] = match_score
                
                scored_candidates.append(candidate_with_score)
                
            except Exception as e:
                logger.error(f"Failed to score candidate {candidate.get('id', 'unknown')}: {str(e)}")
                # Add candidate with minimal score
                candidate_with_score = candidate.copy()
                candidate_with_score['match_score'] = MatchScore(
                    overall_score=0.0,
                    confidence=0.0,
                    skills_score=0.0,
                    experience_score=0.0,
                    education_score=0.0,
                    cultural_fit_score=0.0
                )
                scored_candidates.append(candidate_with_score)
        
        return scored_candidates
    
    def _convert_parsed_job_to_requirements(self, parsed_job: ParsedJobDescription) -> Dict[str, Any]:
        """Convert parsed job description to requirements format for scoring."""
        
        return {
            'required_skills': [req.skill for req in parsed_job.required_skills],
            'preferred_skills': [req.skill for req in parsed_job.preferred_skills],
            'experience_requirements': parsed_job.experience_requirements,
            'education_requirements': parsed_job.education_requirements,
            'company_culture': parsed_job.company_culture,
            'seniority_level': parsed_job.seniority_level,
            'industry': parsed_job.industry,
            'department': parsed_job.department
        }
    
    def _generate_search_insights(self, 
                                parsed_job: ParsedJobDescription,
                                initial_candidates: List[Dict[str, Any]],
                                final_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights about the search results."""
        
        insights = {
            'job_analysis': {
                'total_required_skills': len(parsed_job.required_skills),
                'total_preferred_skills': len(parsed_job.preferred_skills),
                'seniority_level': parsed_job.seniority_level,
                'parsing_confidence': parsed_job.confidence_score
            },
            'candidate_pool_analysis': {
                'initial_pool_size': len(initial_candidates),
                'final_pool_size': len(final_candidates),
                'filter_efficiency': len(final_candidates) / len(initial_candidates) if initial_candidates else 0
            },
            'score_distribution': {},
            'skill_coverage': {},
            'recommendations': []
        }
        
        if final_candidates:
            # Analyze score distribution
            scores = [c['match_score'].overall_score for c in final_candidates]
            insights['score_distribution'] = {
                'average_score': sum(scores) / len(scores),
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'candidates_above_80': len([s for s in scores if s >= 0.8]),
                'candidates_above_70': len([s for s in scores if s >= 0.7])
            }
            
            # Analyze skill coverage
            required_skills = [req.skill for req in parsed_job.required_skills]
            skill_coverage = {}
            
            for skill in required_skills:
                candidates_with_skill = 0
                for candidate in final_candidates:
                    if self._candidate_has_skill(candidate, skill):
                        candidates_with_skill += 1
                
                coverage_percentage = (candidates_with_skill / len(final_candidates)) * 100
                skill_coverage[skill] = {
                    'candidates_with_skill': candidates_with_skill,
                    'coverage_percentage': coverage_percentage
                }
            
            insights['skill_coverage'] = skill_coverage
            
            # Generate recommendations
            recommendations = []
            
            if insights['score_distribution']['average_score'] < 0.6:
                recommendations.append("Consider relaxing some requirements or expanding the search criteria")
            
            low_coverage_skills = [
                skill for skill, data in skill_coverage.items() 
                if data['coverage_percentage'] < 30
            ]
            
            if low_coverage_skills:
                recommendations.append(f"Skills with low candidate coverage: {', '.join(low_coverage_skills[:3])}")
            
            insights['recommendations'] = recommendations
        
        return insights
    
    def _find_best_skill_match(self, target_skill: str, 
                             candidate_skills: List[str]) -> Optional[Tuple[str, float]]:
        """Find the best matching skill from candidate's skills."""
        
        best_match = None
        best_similarity = 0.0
        
        for skill in candidate_skills:
            similarity = self.semantic_matcher.calculate_similarity(target_skill, skill)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (skill, similarity)
        
        return best_match if best_similarity >= self.config.similarity_threshold else None
    
    def _find_related_skills(self, job_skills: List[str], 
                           current_skills: List[str]) -> List[Dict[str, Any]]:
        """Find skills related to job requirements that candidate might want to learn."""
        
        related_skills = []
        
        for job_skill in job_skills:
            related = self.semantic_matcher.find_related_skills(
                job_skill, current_skills, threshold=0.5
            )
            
            for skill, similarity in related[:3]:  # Top 3 related skills
                related_skills.append({
                    'skill': skill,
                    'related_to': job_skill,
                    'similarity': similarity,
                    'recommendation': f"Learning {skill} could help with {job_skill} requirements"
                })
        
        return related_skills
    
    def _generate_learning_path(self, missing_skills: List[Dict[str, Any]], 
                              current_skills: List[str]) -> List[Dict[str, Any]]:
        """Generate a learning path for missing skills."""
        
        learning_path = []
        
        # Sort missing skills by importance and difficulty
        sorted_skills = sorted(
            missing_skills, 
            key=lambda x: (x['importance'] == 'required', x.get('years_experience', 1))
        )
        
        for i, skill_info in enumerate(sorted_skills[:5]):  # Top 5 skills
            learning_path.append({
                'order': i + 1,
                'skill': skill_info['skill'],
                'importance': skill_info['importance'],
                'estimated_learning_time': self._estimate_learning_time(skill_info),
                'prerequisites': self._find_prerequisites(skill_info['skill'], current_skills),
                'resources': self._suggest_learning_resources(skill_info['skill'])
            })
        
        return learning_path
    
    def _candidate_has_skill(self, candidate: Dict[str, Any], skill: str) -> bool:
        """Check if a candidate has a specific skill."""
        
        candidate_skills = candidate.get('skills', [])
        skill_lower = skill.lower()
        
        for candidate_skill in candidate_skills:
            skill_name = candidate_skill.get('name', '').lower()
            if skill_lower in skill_name or skill_name in skill_lower:
                return True
        
        return False
    
    def _analyze_skill_market_demand(self, skill: str, location: str) -> Dict[str, Any]:
        """Analyze market demand for a specific skill (simplified implementation)."""
        
        # This is a simplified implementation
        # In practice, you'd integrate with job market APIs
        
        skill_category = self.semantic_matcher.get_skill_category(skill)
        
        # Mock market data based on skill category
        demand_scores = {
            'programming_language': 0.8,
            'framework': 0.7,
            'database': 0.6,
            'cloud_platform': 0.9,
            'soft_skill': 0.5,
            'other': 0.4
        }
        
        return {
            'demand_score': demand_scores.get(skill_category, 0.5),
            'trend_direction': 'rising' if skill_category in ['cloud_platform', 'programming_language'] else 'stable',
            'salary_impact': 15.0 if skill_category == 'cloud_platform' else 10.0,
            'competition_level': 'high' if skill_category == 'programming_language' else 'medium',
            'availability_score': 0.6,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _generate_market_insights(self, skill_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market insights from skill analysis."""
        
        if not skill_analysis:
            return {}
        
        avg_demand = sum(data['demand_score'] for data in skill_analysis.values()) / len(skill_analysis)
        rising_skills = [skill for skill, data in skill_analysis.items() if data['trend_direction'] == 'rising']
        high_impact_skills = [skill for skill, data in skill_analysis.items() if data['salary_impact'] > 12]
        
        return {
            'average_demand_score': avg_demand,
            'market_outlook': 'positive' if avg_demand > 0.7 else 'moderate' if avg_demand > 0.5 else 'challenging',
            'trending_skills': rising_skills,
            'high_salary_impact_skills': high_impact_skills,
            'recommendations': self._generate_market_recommendations(skill_analysis)
        }
    
    def _generate_market_recommendations(self, skill_analysis: Dict[str, Any]) -> List[str]:
        """Generate market-based recommendations."""
        
        recommendations = []
        
        high_demand_skills = [
            skill for skill, data in skill_analysis.items() 
            if data['demand_score'] > 0.8
        ]
        
        if high_demand_skills:
            recommendations.append(f"High demand skills to prioritize: {', '.join(high_demand_skills[:3])}")
        
        rising_skills = [
            skill for skill, data in skill_analysis.items() 
            if data['trend_direction'] == 'rising'
        ]
        
        if rising_skills:
            recommendations.append(f"Trending skills with growth potential: {', '.join(rising_skills[:3])}")
        
        return recommendations
    
    def _estimate_learning_time(self, skill_info: Dict[str, Any]) -> str:
        """Estimate learning time for a skill."""
        
        proficiency = skill_info.get('proficiency_level', 'intermediate')
        years_exp = skill_info.get('years_experience', 1)
        
        if proficiency == 'beginner':
            return "2-4 weeks"
        elif proficiency == 'intermediate':
            return "1-3 months"
        elif proficiency == 'advanced':
            return "3-6 months"
        else:
            return "6+ months"
    
    def _find_prerequisites(self, skill: str, current_skills: List[str]) -> List[str]:
        """Find prerequisites for learning a skill."""
        
        # Simplified prerequisite mapping
        prerequisites_map = {
            'react': ['javascript', 'html', 'css'],
            'angular': ['javascript', 'typescript', 'html', 'css'],
            'django': ['python'],
            'spring': ['java'],
            'kubernetes': ['docker', 'linux'],
            'machine learning': ['python', 'statistics', 'mathematics']
        }
        
        skill_lower = skill.lower()
        prerequisites = prerequisites_map.get(skill_lower, [])
        
        # Filter out prerequisites the candidate already has
        missing_prerequisites = []
        for prereq in prerequisites:
            if not any(self.semantic_matcher.calculate_similarity(prereq, current_skill) >= 0.7 
                      for current_skill in current_skills):
                missing_prerequisites.append(prereq)
        
        return missing_prerequisites
    
    def _suggest_learning_resources(self, skill: str) -> List[Dict[str, str]]:
        """Suggest learning resources for a skill."""
        
        # Simplified resource suggestions
        return [
            {
                'type': 'online_course',
                'name': f"Learn {skill}",
                'provider': 'Online Learning Platform',
                'url': f"https://example.com/learn-{skill.lower().replace(' ', '-')}"
            },
            {
                'type': 'documentation',
                'name': f"Official {skill} Documentation",
                'provider': 'Official',
                'url': f"https://docs.{skill.lower().replace(' ', '')}.com"
            }
        ]
    
    def _fallback_search(self, job_description: str, limit: int) -> Dict[str, Any]:
        """Fallback to basic search when enhanced search fails."""
        
        try:
            # Extract basic skills from job description
            basic_skills = self._extract_basic_skills(job_description)
            
            # Use existing search service
            results = self.base_search_service.search_candidates(
                skills=basic_skills[:5],
                limit=limit
            )
            
            return {
                'candidates': results.get('candidates', []),
                'metadata': {
                    'search_type': 'fallback_basic',
                    'candidates_returned': len(results.get('candidates', [])),
                    'message': 'Enhanced search failed, using basic search'
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {str(e)}")
            return {
                'candidates': [],
                'metadata': {
                    'search_type': 'failed',
                    'error': str(e)
                }
            }
    
    def _extract_basic_skills(self, job_description: str) -> List[str]:
        """Extract basic skills from job description using simple patterns."""
        
        import re
        
        # Common skill patterns
        skill_patterns = [
            r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\breact\b',
            r'\bangular\b', r'\bnode\.?js\b', r'\baws\b', r'\bdocker\b',
            r'\bmysql\b', r'\bpostgresql\b', r'\bgit\b', r'\blinux\b'
        ]
        
        found_skills = []
        job_description_lower = job_description.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, job_description_lower)
            found_skills.extend(matches)
        
        return list(set(found_skills))  # Remove duplicates 