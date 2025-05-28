"""
Market intelligence engine for skill demand analysis and trends.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


@dataclass
class SkillDemand:
    """Represents market demand data for a skill."""
    skill: str
    demand_score: float  # 0-1 scale
    growth_trend: str  # 'increasing', 'stable', 'decreasing'
    salary_impact: float  # Multiplier effect on salary
    job_postings_count: int
    location: str
    last_updated: datetime


@dataclass
class MarketTrend:
    """Represents a market trend."""
    trend_name: str
    description: str
    affected_skills: List[str]
    impact_score: float  # 0-1 scale
    time_horizon: str  # 'short', 'medium', 'long'


class MarketIntelligenceEngine:
    """Provides market intelligence and skill demand analysis."""
    
    def __init__(self):
        # Mock market data - in production, this would come from APIs
        self.skill_demand_data = self._initialize_mock_data()
        self.market_trends = self._initialize_market_trends()
        
    def _initialize_mock_data(self) -> Dict[str, SkillDemand]:
        """Initialize mock market data for demonstration."""
        
        # High-demand skills with realistic data
        high_demand_skills = {
            'python': SkillDemand(
                skill='python',
                demand_score=0.95,
                growth_trend='increasing',
                salary_impact=1.3,
                job_postings_count=15420,
                location='global',
                last_updated=datetime.now()
            ),
            'javascript': SkillDemand(
                skill='javascript',
                demand_score=0.92,
                growth_trend='stable',
                salary_impact=1.25,
                job_postings_count=18750,
                location='global',
                last_updated=datetime.now()
            ),
            'react': SkillDemand(
                skill='react',
                demand_score=0.88,
                growth_trend='increasing',
                salary_impact=1.35,
                job_postings_count=12340,
                location='global',
                last_updated=datetime.now()
            ),
            'aws': SkillDemand(
                skill='aws',
                demand_score=0.90,
                growth_trend='increasing',
                salary_impact=1.4,
                job_postings_count=9870,
                location='global',
                last_updated=datetime.now()
            ),
            'docker': SkillDemand(
                skill='docker',
                demand_score=0.85,
                growth_trend='increasing',
                salary_impact=1.2,
                job_postings_count=7650,
                location='global',
                last_updated=datetime.now()
            ),
            'kubernetes': SkillDemand(
                skill='kubernetes',
                demand_score=0.82,
                growth_trend='increasing',
                salary_impact=1.45,
                job_postings_count=5430,
                location='global',
                last_updated=datetime.now()
            ),
            'machine learning': SkillDemand(
                skill='machine learning',
                demand_score=0.87,
                growth_trend='increasing',
                salary_impact=1.5,
                job_postings_count=6780,
                location='global',
                last_updated=datetime.now()
            ),
            'tensorflow': SkillDemand(
                skill='tensorflow',
                demand_score=0.75,
                growth_trend='stable',
                salary_impact=1.35,
                job_postings_count=3210,
                location='global',
                last_updated=datetime.now()
            ),
            'node.js': SkillDemand(
                skill='node.js',
                demand_score=0.83,
                growth_trend='stable',
                salary_impact=1.25,
                job_postings_count=8940,
                location='global',
                last_updated=datetime.now()
            ),
            'java': SkillDemand(
                skill='java',
                demand_score=0.80,
                growth_trend='stable',
                salary_impact=1.2,
                job_postings_count=11230,
                location='global',
                last_updated=datetime.now()
            )
        }
        
        return high_demand_skills
    
    def _initialize_market_trends(self) -> List[MarketTrend]:
        """Initialize market trends data."""
        
        return [
            MarketTrend(
                trend_name="AI/ML Adoption",
                description="Increasing adoption of AI and machine learning across industries",
                affected_skills=['python', 'tensorflow', 'pytorch', 'machine learning', 'data science'],
                impact_score=0.9,
                time_horizon='medium'
            ),
            MarketTrend(
                trend_name="Cloud Migration",
                description="Continued migration to cloud platforms",
                affected_skills=['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                impact_score=0.85,
                time_horizon='long'
            ),
            MarketTrend(
                trend_name="Remote Work Tools",
                description="Increased demand for collaboration and remote work technologies",
                affected_skills=['javascript', 'react', 'vue', 'node.js', 'websockets'],
                impact_score=0.7,
                time_horizon='short'
            ),
            MarketTrend(
                trend_name="DevOps Automation",
                description="Growing emphasis on DevOps practices and automation",
                affected_skills=['docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
                impact_score=0.8,
                time_horizon='medium'
            ),
            MarketTrend(
                trend_name="Cybersecurity Focus",
                description="Increased focus on security across all applications",
                affected_skills=['security', 'penetration testing', 'encryption', 'oauth'],
                impact_score=0.75,
                time_horizon='long'
            )
        ]
    
    def get_skill_demand(self, skill: str, location: str = 'global') -> Optional[SkillDemand]:
        """Get demand data for a specific skill."""
        
        skill_lower = skill.lower().strip()
        
        # Check if we have exact data
        if skill_lower in self.skill_demand_data:
            return self.skill_demand_data[skill_lower]
        
        # Generate estimated data for unknown skills
        return self._estimate_skill_demand(skill, location)
    
    def _estimate_skill_demand(self, skill: str, location: str) -> SkillDemand:
        """Estimate demand for skills not in our database."""
        
        # Simple heuristics for estimation
        skill_lower = skill.lower()
        
        # Programming languages tend to have higher demand
        programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring']
        cloud_tools = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        
        base_demand = 0.5
        growth_trend = 'stable'
        salary_impact = 1.0
        
        if any(lang in skill_lower for lang in programming_languages):
            base_demand += 0.2
            salary_impact += 0.15
        
        if any(fw in skill_lower for fw in frameworks):
            base_demand += 0.15
            growth_trend = 'increasing'
            salary_impact += 0.1
        
        if any(cloud in skill_lower for cloud in cloud_tools):
            base_demand += 0.25
            growth_trend = 'increasing'
            salary_impact += 0.2
        
        # Add some randomness to make it realistic
        base_demand += random.uniform(-0.1, 0.1)
        base_demand = max(0.1, min(1.0, base_demand))
        
        # Estimate job postings based on demand
        job_postings = int(base_demand * 10000 * random.uniform(0.5, 1.5))
        
        return SkillDemand(
            skill=skill,
            demand_score=base_demand,
            growth_trend=growth_trend,
            salary_impact=salary_impact,
            job_postings_count=job_postings,
            location=location,
            last_updated=datetime.now()
        )
    
    def analyze_skill_portfolio(self, skills: List[str], location: str = 'global') -> Dict[str, Any]:
        """Analyze a portfolio of skills for market value."""
        
        if not skills:
            return {}
        
        skill_analyses = []
        total_demand = 0
        total_salary_impact = 0
        
        for skill in skills:
            demand_data = self.get_skill_demand(skill, location)
            if demand_data:
                skill_analyses.append({
                    'skill': skill,
                    'demand_score': demand_data.demand_score,
                    'growth_trend': demand_data.growth_trend,
                    'salary_impact': demand_data.salary_impact,
                    'job_postings': demand_data.job_postings_count
                })
                total_demand += demand_data.demand_score
                total_salary_impact += demand_data.salary_impact
        
        if not skill_analyses:
            return {}
        
        # Calculate portfolio metrics
        avg_demand = total_demand / len(skill_analyses)
        avg_salary_impact = total_salary_impact / len(skill_analyses)
        
        # Identify top skills
        top_skills = sorted(skill_analyses, key=lambda x: x['demand_score'], reverse=True)[:5]
        
        # Identify growth skills
        growth_skills = [s for s in skill_analyses if s['growth_trend'] == 'increasing']
        
        # Calculate portfolio strength
        portfolio_strength = self._calculate_portfolio_strength(skill_analyses)
        
        return {
            'portfolio_summary': {
                'total_skills': len(skills),
                'average_demand': avg_demand,
                'average_salary_impact': avg_salary_impact,
                'portfolio_strength': portfolio_strength
            },
            'top_skills': top_skills,
            'growth_skills': growth_skills,
            'skill_details': skill_analyses,
            'recommendations': self._generate_portfolio_recommendations(skill_analyses)
        }
    
    def _calculate_portfolio_strength(self, skill_analyses: List[Dict]) -> str:
        """Calculate overall portfolio strength."""
        
        if not skill_analyses:
            return 'weak'
        
        avg_demand = sum(s['demand_score'] for s in skill_analyses) / len(skill_analyses)
        high_demand_count = len([s for s in skill_analyses if s['demand_score'] > 0.8])
        growth_count = len([s for s in skill_analyses if s['growth_trend'] == 'increasing'])
        
        if avg_demand > 0.8 and high_demand_count >= 3:
            return 'excellent'
        elif avg_demand > 0.7 and high_demand_count >= 2:
            return 'strong'
        elif avg_demand > 0.6 or growth_count >= 2:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_portfolio_recommendations(self, skill_analyses: List[Dict]) -> List[str]:
        """Generate recommendations for skill portfolio improvement."""
        
        recommendations = []
        
        if not skill_analyses:
            return ['Consider learning high-demand skills like Python, JavaScript, or cloud technologies']
        
        avg_demand = sum(s['demand_score'] for s in skill_analyses) / len(skill_analyses)
        growth_skills = [s for s in skill_analyses if s['growth_trend'] == 'increasing']
        
        if avg_demand < 0.6:
            recommendations.append('Focus on learning more in-demand skills to improve marketability')
        
        if len(growth_skills) < 2:
            recommendations.append('Consider adding trending technologies like AI/ML or cloud platforms')
        
        # Check for skill gaps in common combinations
        skill_names = [s['skill'].lower() for s in skill_analyses]
        
        if 'python' in skill_names and 'machine learning' not in skill_names:
            recommendations.append('Consider learning machine learning to complement your Python skills')
        
        if 'javascript' in skill_names and 'react' not in skill_names:
            recommendations.append('Consider learning React to enhance your JavaScript expertise')
        
        if any('cloud' in s or s in ['aws', 'azure', 'gcp'] for s in skill_names):
            if 'docker' not in skill_names:
                recommendations.append('Consider learning Docker for containerization skills')
        
        if not recommendations:
            recommendations.append('Your skill portfolio looks strong! Keep updating with latest trends')
        
        return recommendations
    
    def get_market_trends(self, skill_category: Optional[str] = None) -> List[MarketTrend]:
        """Get current market trends, optionally filtered by skill category."""
        
        if not skill_category:
            return self.market_trends
        
        # Filter trends that affect the specified skill category
        relevant_trends = []
        for trend in self.market_trends:
            if any(skill_category.lower() in skill.lower() for skill in trend.affected_skills):
                relevant_trends.append(trend)
        
        return relevant_trends
    
    def predict_skill_demand(self, skill: str, months_ahead: int = 12) -> Dict[str, Any]:
        """Predict future demand for a skill."""
        
        current_demand = self.get_skill_demand(skill)
        if not current_demand:
            return {}
        
        # Simple prediction based on current trend
        trend_multipliers = {
            'increasing': 1.2,
            'stable': 1.0,
            'decreasing': 0.8
        }
        
        multiplier = trend_multipliers.get(current_demand.growth_trend, 1.0)
        
        # Apply time decay - trends slow down over time
        time_factor = 1 + (multiplier - 1) * (months_ahead / 12) * 0.8
        
        predicted_demand = min(1.0, current_demand.demand_score * time_factor)
        
        return {
            'skill': skill,
            'current_demand': current_demand.demand_score,
            'predicted_demand': predicted_demand,
            'prediction_horizon': f'{months_ahead} months',
            'confidence': 0.7,  # Mock confidence score
            'trend': current_demand.growth_trend,
            'factors': [
                f'Current trend: {current_demand.growth_trend}',
                f'Market momentum: {"positive" if multiplier > 1 else "neutral" if multiplier == 1 else "negative"}'
            ]
        }
    
    def compare_skills(self, skills: List[str], location: str = 'global') -> Dict[str, Any]:
        """Compare multiple skills across various metrics."""
        
        if len(skills) < 2:
            return {}
        
        comparisons = []
        for skill in skills:
            demand_data = self.get_skill_demand(skill, location)
            if demand_data:
                comparisons.append({
                    'skill': skill,
                    'demand_score': demand_data.demand_score,
                    'growth_trend': demand_data.growth_trend,
                    'salary_impact': demand_data.salary_impact,
                    'job_postings': demand_data.job_postings_count
                })
        
        if not comparisons:
            return {}
        
        # Sort by demand score
        comparisons.sort(key=lambda x: x['demand_score'], reverse=True)
        
        # Find best in each category
        best_demand = max(comparisons, key=lambda x: x['demand_score'])
        best_salary = max(comparisons, key=lambda x: x['salary_impact'])
        most_jobs = max(comparisons, key=lambda x: x['job_postings'])
        
        return {
            'skill_comparison': comparisons,
            'best_overall_demand': best_demand['skill'],
            'best_salary_impact': best_salary['skill'],
            'most_job_opportunities': most_jobs['skill'],
            'summary': {
                'highest_demand_score': best_demand['demand_score'],
                'highest_salary_impact': best_salary['salary_impact'],
                'most_job_postings': most_jobs['job_postings']
            }
        }
    
    def get_learning_path_recommendations(self, current_skills: List[str], 
                                        target_role: str = None) -> Dict[str, Any]:
        """Get personalized learning path recommendations."""
        
        # Analyze current portfolio
        portfolio_analysis = self.analyze_skill_portfolio(current_skills)
        
        # Identify skill gaps based on market trends
        trending_skills = []
        for trend in self.market_trends:
            if trend.impact_score > 0.7:
                trending_skills.extend(trend.affected_skills)
        
        # Remove skills already possessed
        current_skills_lower = [s.lower() for s in current_skills]
        missing_trending = [s for s in trending_skills if s.lower() not in current_skills_lower]
        
        # Prioritize recommendations
        recommendations = []
        for skill in missing_trending[:5]:  # Top 5 recommendations
            demand_data = self.get_skill_demand(skill)
            if demand_data and demand_data.demand_score > 0.7:
                recommendations.append({
                    'skill': skill,
                    'priority': 'high' if demand_data.demand_score > 0.8 else 'medium',
                    'reason': f'High market demand ({demand_data.demand_score:.2f}) and growing trend',
                    'estimated_learning_time': self._estimate_learning_time(skill),
                    'salary_impact': demand_data.salary_impact
                })
        
        return {
            'current_portfolio_strength': portfolio_analysis.get('portfolio_summary', {}).get('portfolio_strength', 'unknown'),
            'recommended_skills': recommendations,
            'learning_path': self._create_learning_path(recommendations),
            'market_insights': {
                'trending_technologies': trending_skills[:10],
                'high_demand_skills': [s for s in self.skill_demand_data.keys() 
                                     if self.skill_demand_data[s].demand_score > 0.8]
            }
        }
    
    def _estimate_learning_time(self, skill: str) -> str:
        """Estimate time needed to learn a skill."""
        
        skill_lower = skill.lower()
        
        # Simple heuristics for learning time estimation
        if skill_lower in ['python', 'java', 'javascript']:
            return '3-6 months'
        elif skill_lower in ['react', 'angular', 'vue']:
            return '2-4 months'
        elif skill_lower in ['docker', 'kubernetes']:
            return '1-3 months'
        elif skill_lower in ['aws', 'azure', 'gcp']:
            return '4-8 months'
        elif 'machine learning' in skill_lower:
            return '6-12 months'
        else:
            return '2-6 months'
    
    def _create_learning_path(self, recommendations: List[Dict]) -> List[Dict]:
        """Create a structured learning path from recommendations."""
        
        if not recommendations:
            return []
        
        # Sort by priority and dependencies
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        
        learning_path = []
        
        # Add foundational skills first
        foundational = ['python', 'javascript', 'sql']
        for skill in foundational:
            matching_recs = [r for r in high_priority + medium_priority if skill in r['skill'].lower()]
            if matching_recs:
                learning_path.extend(matching_recs)
        
        # Add remaining high priority skills
        remaining_high = [r for r in high_priority if r not in learning_path]
        learning_path.extend(remaining_high)
        
        # Add medium priority skills
        remaining_medium = [r for r in medium_priority if r not in learning_path]
        learning_path.extend(remaining_medium[:3])  # Limit to top 3
        
        return learning_path[:5]  # Return top 5 in learning order 