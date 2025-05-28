"""
Skill trend analyzer for identifying market patterns and future demands.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of skill trend."""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    EMERGING = "emerging"
    OBSOLETE = "obsolete"


@dataclass
class SkillTrend:
    """Represents a skill trend analysis."""
    skill: str
    trend_direction: TrendDirection
    growth_rate: float  # Percentage change
    confidence: float  # 0-1 scale
    time_period: str
    supporting_factors: List[str]
    risk_factors: List[str]
    prediction_horizon: str


@dataclass
class MarketPattern:
    """Represents a detected market pattern."""
    pattern_name: str
    description: str
    affected_skills: List[str]
    pattern_strength: float  # 0-1 scale
    detected_at: datetime
    expected_duration: str


class SkillTrendAnalyzer:
    """Analyzes skill trends and market patterns."""
    
    def __init__(self):
        # Mock historical data for demonstration
        self.historical_data = self._initialize_historical_data()
        self.market_patterns = self._initialize_market_patterns()
        
        # Trend analysis parameters
        self.trend_window_months = 12
        self.confidence_threshold = 0.6
        
    def _initialize_historical_data(self) -> Dict[str, List[Dict]]:
        """Initialize mock historical skill demand data."""
        
        # Simulate 24 months of data for key skills
        skills = [
            'python', 'javascript', 'react', 'aws', 'docker', 'kubernetes',
            'machine learning', 'tensorflow', 'node.js', 'java', 'angular',
            'vue', 'azure', 'gcp', 'mongodb', 'postgresql', 'redis'
        ]
        
        historical_data = {}
        
        for skill in skills:
            skill_data = []
            base_demand = random.uniform(0.4, 0.9)
            
            # Generate trend based on skill type
            if skill in ['python', 'aws', 'docker', 'kubernetes', 'machine learning']:
                trend_factor = 1.02  # Growing skills
            elif skill in ['javascript', 'react', 'node.js']:
                trend_factor = 1.01  # Stable growth
            else:
                trend_factor = 0.999  # Slight decline or stable
            
            current_demand = base_demand
            
            for month in range(24):
                # Add some noise
                noise = random.uniform(-0.05, 0.05)
                current_demand = max(0.1, min(1.0, current_demand * trend_factor + noise))
                
                skill_data.append({
                    'month': month,
                    'demand_score': current_demand,
                    'job_postings': int(current_demand * 10000 * random.uniform(0.8, 1.2)),
                    'salary_trend': current_demand * random.uniform(0.9, 1.1)
                })
            
            historical_data[skill] = skill_data
        
        return historical_data
    
    def _initialize_market_patterns(self) -> List[MarketPattern]:
        """Initialize detected market patterns."""
        
        return [
            MarketPattern(
                pattern_name="AI/ML Surge",
                description="Rapid increase in AI and machine learning skill demand",
                affected_skills=['python', 'tensorflow', 'pytorch', 'machine learning', 'data science'],
                pattern_strength=0.9,
                detected_at=datetime.now() - timedelta(days=90),
                expected_duration="2-3 years"
            ),
            MarketPattern(
                pattern_name="Cloud Migration Wave",
                description="Continued enterprise migration to cloud platforms",
                affected_skills=['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
                pattern_strength=0.85,
                detected_at=datetime.now() - timedelta(days=180),
                expected_duration="3-5 years"
            ),
            MarketPattern(
                pattern_name="Frontend Framework Evolution",
                description="Shift towards modern frontend frameworks",
                affected_skills=['react', 'vue', 'angular', 'typescript'],
                pattern_strength=0.7,
                detected_at=datetime.now() - timedelta(days=120),
                expected_duration="1-2 years"
            ),
            MarketPattern(
                pattern_name="DevOps Automation",
                description="Increased focus on DevOps and automation tools",
                affected_skills=['docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
                pattern_strength=0.8,
                detected_at=datetime.now() - timedelta(days=150),
                expected_duration="2-4 years"
            )
        ]
    
    def analyze_skill_trend(self, skill: str, months_back: int = 12) -> SkillTrend:
        """Analyze trend for a specific skill."""
        
        skill_lower = skill.lower()
        
        if skill_lower not in self.historical_data:
            # Generate estimated trend for unknown skills
            return self._estimate_skill_trend(skill)
        
        # Get historical data
        data = self.historical_data[skill_lower]
        recent_data = data[-months_back:] if len(data) >= months_back else data
        
        if len(recent_data) < 3:
            return self._estimate_skill_trend(skill)
        
        # Calculate trend metrics
        demand_values = [d['demand_score'] for d in recent_data]
        job_posting_values = [d['job_postings'] for d in recent_data]
        
        # Calculate growth rate
        start_demand = sum(demand_values[:3]) / 3  # Average of first 3 months
        end_demand = sum(demand_values[-3:]) / 3   # Average of last 3 months
        
        growth_rate = ((end_demand - start_demand) / start_demand) * 100 if start_demand > 0 else 0
        
        # Determine trend direction
        if growth_rate > 10:
            trend_direction = TrendDirection.RISING
        elif growth_rate > 2:
            trend_direction = TrendDirection.EMERGING if end_demand > 0.7 else TrendDirection.STABLE
        elif growth_rate > -5:
            trend_direction = TrendDirection.STABLE
        elif growth_rate > -15:
            trend_direction = TrendDirection.DECLINING
        else:
            trend_direction = TrendDirection.OBSOLETE
        
        # Calculate confidence based on data consistency
        variance = sum((d - end_demand) ** 2 for d in demand_values[-6:]) / min(6, len(demand_values))
        confidence = max(0.3, 1.0 - variance * 2)
        
        # Identify supporting and risk factors
        supporting_factors = self._identify_supporting_factors(skill, trend_direction, growth_rate)
        risk_factors = self._identify_risk_factors(skill, trend_direction, growth_rate)
        
        return SkillTrend(
            skill=skill,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            confidence=confidence,
            time_period=f"{months_back} months",
            supporting_factors=supporting_factors,
            risk_factors=risk_factors,
            prediction_horizon="6-12 months"
        )
    
    def _estimate_skill_trend(self, skill: str) -> SkillTrend:
        """Estimate trend for skills without historical data."""
        
        skill_lower = skill.lower()
        
        # Categorize skills and assign likely trends
        emerging_tech = ['ai', 'machine learning', 'blockchain', 'quantum', 'edge computing']
        cloud_tech = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        modern_frameworks = ['react', 'vue', 'svelte', 'next.js', 'nuxt']
        stable_languages = ['python', 'javascript', 'java', 'c#']
        declining_tech = ['flash', 'silverlight', 'jquery', 'backbone']
        
        if any(tech in skill_lower for tech in emerging_tech):
            trend_direction = TrendDirection.EMERGING
            growth_rate = random.uniform(15, 30)
            confidence = 0.7
        elif any(tech in skill_lower for tech in cloud_tech):
            trend_direction = TrendDirection.RISING
            growth_rate = random.uniform(8, 20)
            confidence = 0.8
        elif any(tech in skill_lower for tech in modern_frameworks):
            trend_direction = TrendDirection.RISING
            growth_rate = random.uniform(5, 15)
            confidence = 0.75
        elif any(tech in skill_lower for tech in stable_languages):
            trend_direction = TrendDirection.STABLE
            growth_rate = random.uniform(-2, 5)
            confidence = 0.85
        elif any(tech in skill_lower for tech in declining_tech):
            trend_direction = TrendDirection.DECLINING
            growth_rate = random.uniform(-20, -5)
            confidence = 0.9
        else:
            trend_direction = TrendDirection.STABLE
            growth_rate = random.uniform(-5, 5)
            confidence = 0.5
        
        supporting_factors = self._identify_supporting_factors(skill, trend_direction, growth_rate)
        risk_factors = self._identify_risk_factors(skill, trend_direction, growth_rate)
        
        return SkillTrend(
            skill=skill,
            trend_direction=trend_direction,
            growth_rate=growth_rate,
            confidence=confidence,
            time_period="estimated",
            supporting_factors=supporting_factors,
            risk_factors=risk_factors,
            prediction_horizon="6-12 months"
        )
    
    def _identify_supporting_factors(self, skill: str, trend: TrendDirection, growth_rate: float) -> List[str]:
        """Identify factors supporting the skill trend."""
        
        factors = []
        skill_lower = skill.lower()
        
        if trend in [TrendDirection.RISING, TrendDirection.EMERGING]:
            if 'python' in skill_lower:
                factors.extend(['AI/ML adoption', 'Data science growth', 'Automation trends'])
            elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp']):
                factors.extend(['Cloud migration', 'Digital transformation', 'Remote work adoption'])
            elif any(js in skill_lower for js in ['javascript', 'react', 'node']):
                factors.extend(['Web development growth', 'Mobile app demand', 'Real-time applications'])
            elif 'docker' in skill_lower or 'kubernetes' in skill_lower:
                factors.extend(['DevOps adoption', 'Microservices architecture', 'Container orchestration'])
            
            if growth_rate > 15:
                factors.append('Strong market momentum')
            elif growth_rate > 5:
                factors.append('Steady market growth')
        
        elif trend == TrendDirection.STABLE:
            factors.extend(['Established technology', 'Large existing codebase', 'Enterprise adoption'])
        
        # Add general factors
        if not factors:
            factors.append('Market demand analysis')
        
        return factors[:4]  # Limit to top 4 factors
    
    def _identify_risk_factors(self, skill: str, trend: TrendDirection, growth_rate: float) -> List[str]:
        """Identify risk factors for the skill trend."""
        
        risks = []
        skill_lower = skill.lower()
        
        if trend == TrendDirection.DECLINING:
            risks.extend(['Technology obsolescence', 'Better alternatives available', 'Reduced market demand'])
        elif trend == TrendDirection.OBSOLETE:
            risks.extend(['Legacy technology', 'No active development', 'Security concerns'])
        elif trend == TrendDirection.EMERGING:
            risks.extend(['Market uncertainty', 'Rapid technology changes', 'Skills shortage'])
        
        # Technology-specific risks
        if 'javascript' in skill_lower:
            risks.append('Framework fragmentation')
        elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp']):
            risks.append('Vendor lock-in concerns')
        elif 'machine learning' in skill_lower:
            risks.append('Complexity and specialization requirements')
        
        # General risks based on growth rate
        if growth_rate > 20:
            risks.append('Potential market bubble')
        elif growth_rate < -10:
            risks.append('Declining job opportunities')
        
        return risks[:3]  # Limit to top 3 risks
    
    def analyze_portfolio_trends(self, skills: List[str]) -> Dict[str, Any]:
        """Analyze trends for a portfolio of skills."""
        
        if not skills:
            return {}
        
        skill_trends = []
        for skill in skills:
            trend = self.analyze_skill_trend(skill)
            skill_trends.append(trend)
        
        # Portfolio analysis
        rising_skills = [t for t in skill_trends if t.trend_direction in [TrendDirection.RISING, TrendDirection.EMERGING]]
        stable_skills = [t for t in skill_trends if t.trend_direction == TrendDirection.STABLE]
        declining_skills = [t for t in skill_trends if t.trend_direction in [TrendDirection.DECLINING, TrendDirection.OBSOLETE]]
        
        # Calculate portfolio health
        avg_growth_rate = sum(t.growth_rate for t in skill_trends) / len(skill_trends)
        avg_confidence = sum(t.confidence for t in skill_trends) / len(skill_trends)
        
        # Portfolio strength assessment
        if len(rising_skills) >= len(skill_trends) * 0.6:
            portfolio_health = "excellent"
        elif len(rising_skills) >= len(skill_trends) * 0.4:
            portfolio_health = "good"
        elif len(declining_skills) <= len(skill_trends) * 0.3:
            portfolio_health = "moderate"
        else:
            portfolio_health = "needs_improvement"
        
        return {
            'portfolio_summary': {
                'total_skills': len(skills),
                'rising_skills_count': len(rising_skills),
                'stable_skills_count': len(stable_skills),
                'declining_skills_count': len(declining_skills),
                'average_growth_rate': avg_growth_rate,
                'average_confidence': avg_confidence,
                'portfolio_health': portfolio_health
            },
            'skill_trends': [
                {
                    'skill': t.skill,
                    'trend_direction': t.trend_direction.value,
                    'growth_rate': t.growth_rate,
                    'confidence': t.confidence,
                    'supporting_factors': t.supporting_factors,
                    'risk_factors': t.risk_factors
                }
                for t in skill_trends
            ],
            'recommendations': self._generate_portfolio_recommendations(skill_trends, portfolio_health)
        }
    
    def _generate_portfolio_recommendations(self, skill_trends: List[SkillTrend], 
                                          portfolio_health: str) -> List[str]:
        """Generate recommendations for portfolio improvement."""
        
        recommendations = []
        
        declining_skills = [t for t in skill_trends if t.trend_direction in [TrendDirection.DECLINING, TrendDirection.OBSOLETE]]
        rising_skills = [t for t in skill_trends if t.trend_direction in [TrendDirection.RISING, TrendDirection.EMERGING]]
        
        if portfolio_health == "needs_improvement":
            recommendations.append("Consider updating skills portfolio - high proportion of declining technologies")
        
        if len(declining_skills) > 0:
            declining_names = [t.skill for t in declining_skills[:2]]
            recommendations.append(f"Consider transitioning from declining skills: {', '.join(declining_names)}")
        
        if len(rising_skills) < len(skill_trends) * 0.3:
            recommendations.append("Add more emerging/rising technologies to stay competitive")
        
        # Specific technology recommendations
        has_cloud = any('aws' in t.skill.lower() or 'azure' in t.skill.lower() or 'gcp' in t.skill.lower() 
                       for t in skill_trends)
        if not has_cloud:
            recommendations.append("Consider adding cloud platform skills (AWS, Azure, or GCP)")
        
        has_ai_ml = any('machine learning' in t.skill.lower() or 'ai' in t.skill.lower() 
                       for t in skill_trends)
        if not has_ai_ml and len(skill_trends) > 3:
            recommendations.append("Consider adding AI/ML skills for future opportunities")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def predict_future_demand(self, skill: str, months_ahead: int = 12) -> Dict[str, Any]:
        """Predict future demand for a skill."""
        
        current_trend = self.analyze_skill_trend(skill)
        
        # Simple prediction based on current trend
        monthly_growth = current_trend.growth_rate / 12  # Convert annual to monthly
        
        # Apply trend direction modifiers
        if current_trend.trend_direction == TrendDirection.EMERGING:
            monthly_growth *= 1.2  # Accelerated growth for emerging tech
        elif current_trend.trend_direction == TrendDirection.OBSOLETE:
            monthly_growth *= 1.5  # Accelerated decline for obsolete tech
        
        # Calculate predicted demand
        current_demand = 0.7  # Assume current baseline
        predicted_demand = current_demand * (1 + monthly_growth / 100) ** months_ahead
        predicted_demand = max(0.1, min(1.0, predicted_demand))
        
        # Calculate prediction confidence
        prediction_confidence = current_trend.confidence * max(0.3, 1.0 - months_ahead / 24)
        
        return {
            'skill': skill,
            'current_trend': current_trend.trend_direction.value,
            'current_growth_rate': current_trend.growth_rate,
            'predicted_demand': predicted_demand,
            'prediction_confidence': prediction_confidence,
            'prediction_horizon': f"{months_ahead} months",
            'key_factors': current_trend.supporting_factors,
            'risk_factors': current_trend.risk_factors
        }
    
    def get_market_patterns(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get detected market patterns."""
        
        patterns = self.market_patterns
        
        if active_only:
            # Filter to patterns detected in last 6 months
            cutoff_date = datetime.now() - timedelta(days=180)
            patterns = [p for p in patterns if p.detected_at >= cutoff_date]
        
        return [
            {
                'pattern_name': p.pattern_name,
                'description': p.description,
                'affected_skills': p.affected_skills,
                'pattern_strength': p.pattern_strength,
                'detected_at': p.detected_at.isoformat(),
                'expected_duration': p.expected_duration
            }
            for p in patterns
        ]
    
    def identify_skill_gaps(self, current_skills: List[str], 
                           target_trends: List[TrendDirection] = None) -> Dict[str, Any]:
        """Identify skill gaps based on market trends."""
        
        if target_trends is None:
            target_trends = [TrendDirection.RISING, TrendDirection.EMERGING]
        
        # Get all trending skills from patterns
        trending_skills = set()
        for pattern in self.market_patterns:
            if pattern.pattern_strength > 0.7:
                trending_skills.update(pattern.affected_skills)
        
        # Add skills from historical data that are trending
        for skill, data in self.historical_data.items():
            trend = self.analyze_skill_trend(skill)
            if trend.trend_direction in target_trends and trend.confidence > 0.6:
                trending_skills.add(skill)
        
        # Identify gaps
        current_skills_lower = [s.lower() for s in current_skills]
        skill_gaps = [s for s in trending_skills if s.lower() not in current_skills_lower]
        
        # Prioritize gaps
        prioritized_gaps = []
        for skill in skill_gaps[:10]:  # Limit to top 10
            trend = self.analyze_skill_trend(skill)
            priority = "high" if trend.growth_rate > 15 else "medium" if trend.growth_rate > 5 else "low"
            
            prioritized_gaps.append({
                'skill': skill,
                'priority': priority,
                'growth_rate': trend.growth_rate,
                'trend_direction': trend.trend_direction.value,
                'confidence': trend.confidence,
                'supporting_factors': trend.supporting_factors[:2]
            })
        
        # Sort by priority and growth rate
        prioritized_gaps.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['priority']],
            x['growth_rate']
        ), reverse=True)
        
        return {
            'identified_gaps': prioritized_gaps[:8],  # Top 8 gaps
            'gap_analysis': {
                'total_gaps_identified': len(skill_gaps),
                'high_priority_gaps': len([g for g in prioritized_gaps if g['priority'] == 'high']),
                'market_coverage': len(current_skills_lower) / (len(current_skills_lower) + len(skill_gaps)) if skill_gaps else 1.0
            },
            'recommendations': [
                f"Focus on high-priority skills: {', '.join([g['skill'] for g in prioritized_gaps[:3] if g['priority'] == 'high'])}",
                "Consider market patterns when planning skill development",
                "Regular skill portfolio review recommended"
            ]
        } 