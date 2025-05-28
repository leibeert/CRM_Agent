"""
Advanced skill extraction from job descriptions and resumes.
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found, using basic extraction")
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not available, using basic extraction")
    SPACY_AVAILABLE = False
    nlp = None


@dataclass
class ExtractedSkill:
    """Represents an extracted skill with metadata."""
    name: str
    category: str
    confidence: float
    context: str
    proficiency_indicators: List[str]
    years_mentioned: Optional[int] = None


class AdvancedSkillExtractor:
    """Advanced skill extraction using NLP and pattern matching."""
    
    def __init__(self):
        self.programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'rust',
            'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl', 'shell',
            'bash', 'powershell', 'sql', 'html', 'css', 'sass', 'less', 'dart', 'lua',
            'haskell', 'clojure', 'erlang', 'elixir', 'f#', 'vb.net', 'objective-c',
            'assembly', 'cobol', 'fortran', 'pascal', 'delphi', 'vba', 'groovy'
        }
        
        self.frameworks_libraries = {
            'react', 'angular', 'vue', 'svelte', 'ember', 'backbone', 'jquery',
            'express', 'fastapi', 'flask', 'django', 'spring', 'laravel', 'rails',
            'asp.net', 'blazor', 'xamarin', 'flutter', 'react native', 'ionic',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'plotly', 'opencv', 'nltk', 'spacy',
            'bootstrap', 'tailwind', 'material-ui', 'ant design', 'chakra ui'
        }
        
        self.databases = {
            'mysql', 'postgresql', 'sqlite', 'mongodb', 'redis', 'elasticsearch',
            'cassandra', 'dynamodb', 'oracle', 'sql server', 'mariadb', 'couchdb',
            'neo4j', 'influxdb', 'clickhouse', 'snowflake', 'bigquery', 'redshift'
        }
        
        self.cloud_platforms = {
            'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
            'linode', 'vultr', 'cloudflare', 'netlify', 'vercel', 'firebase'
        }
        
        self.tools_technologies = {
            'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
            'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'git', 'svn',
            'jira', 'confluence', 'slack', 'teams', 'zoom', 'figma', 'sketch',
            'photoshop', 'illustrator', 'after effects', 'premiere pro',
            'tableau', 'power bi', 'looker', 'grafana', 'prometheus', 'elk stack'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving',
            'critical thinking', 'creativity', 'adaptability', 'time management',
            'project management', 'agile', 'scrum', 'kanban', 'mentoring',
            'coaching', 'presentation', 'negotiation', 'conflict resolution'
        }
        
        # Combine all skill categories
        self.all_skills = (
            self.programming_languages | self.frameworks_libraries | 
            self.databases | self.cloud_platforms | self.tools_technologies |
            self.soft_skills
        )
        
        # Proficiency indicators
        self.proficiency_patterns = {
            'expert': r'\b(expert|advanced|senior|lead|principal|architect)\b',
            'intermediate': r'\b(intermediate|mid-level|experienced|proficient)\b',
            'beginner': r'\b(beginner|junior|entry-level|basic|learning)\b'
        }
        
        # Years of experience patterns
        self.years_pattern = re.compile(
            r'(\d+)[\s\-]*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)?',
            re.IGNORECASE
        )
    
    def extract_skills(self, text: str) -> List[ExtractedSkill]:
        """Extract skills from text with context and confidence."""
        
        if not text:
            return []
        
        text_lower = text.lower()
        extracted_skills = []
        
        # Use spaCy if available for better context extraction
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
        
        # Extract skills from each sentence for better context
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find skills in this sentence
            for skill in self.all_skills:
                if self._is_skill_mentioned(skill, sentence_lower):
                    # Determine skill category
                    category = self._get_skill_category(skill)
                    
                    # Extract proficiency indicators
                    proficiency = self._extract_proficiency(sentence)
                    
                    # Extract years of experience
                    years = self._extract_years(sentence)
                    
                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(skill, sentence, proficiency, years)
                    
                    extracted_skill = ExtractedSkill(
                        name=skill,
                        category=category,
                        confidence=confidence,
                        context=sentence.strip(),
                        proficiency_indicators=proficiency,
                        years_mentioned=years
                    )
                    
                    extracted_skills.append(extracted_skill)
        
        # Remove duplicates and merge similar skills
        return self._deduplicate_skills(extracted_skills)
    
    def _is_skill_mentioned(self, skill: str, text: str) -> bool:
        """Check if a skill is mentioned in text with word boundaries."""
        
        # Handle multi-word skills
        if ' ' in skill:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
        
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _get_skill_category(self, skill: str) -> str:
        """Determine the category of a skill."""
        
        if skill in self.programming_languages:
            return 'Programming Language'
        elif skill in self.frameworks_libraries:
            return 'Framework/Library'
        elif skill in self.databases:
            return 'Database'
        elif skill in self.cloud_platforms:
            return 'Cloud Platform'
        elif skill in self.tools_technologies:
            return 'Tool/Technology'
        elif skill in self.soft_skills:
            return 'Soft Skill'
        else:
            return 'Other'
    
    def _extract_proficiency(self, text: str) -> List[str]:
        """Extract proficiency indicators from text."""
        
        proficiency_indicators = []
        
        for level, pattern in self.proficiency_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                proficiency_indicators.append(level)
        
        return proficiency_indicators
    
    def _extract_years(self, text: str) -> Optional[int]:
        """Extract years of experience from text."""
        
        match = self.years_pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _calculate_confidence(self, skill: str, context: str, 
                            proficiency: List[str], years: Optional[int]) -> float:
        """Calculate confidence score for skill extraction."""
        
        base_confidence = 0.7
        
        # Boost confidence for exact matches
        if skill in context.lower():
            base_confidence += 0.1
        
        # Boost confidence if proficiency is mentioned
        if proficiency:
            base_confidence += 0.1
        
        # Boost confidence if years are mentioned
        if years:
            base_confidence += 0.1
        
        # Boost confidence for technical skills in technical contexts
        technical_keywords = ['develop', 'build', 'implement', 'design', 'code', 'program']
        if any(keyword in context.lower() for keyword in technical_keywords):
            if self._get_skill_category(skill) in ['Programming Language', 'Framework/Library', 'Database']:
                base_confidence += 0.05
        
        return min(1.0, base_confidence)
    
    def _deduplicate_skills(self, skills: List[ExtractedSkill]) -> List[ExtractedSkill]:
        """Remove duplicate skills and merge similar ones."""
        
        # Group skills by name
        skill_groups = {}
        for skill in skills:
            if skill.name not in skill_groups:
                skill_groups[skill.name] = []
            skill_groups[skill.name].append(skill)
        
        # Merge skills with same name
        merged_skills = []
        for skill_name, skill_list in skill_groups.items():
            if len(skill_list) == 1:
                merged_skills.append(skill_list[0])
            else:
                # Merge multiple mentions of the same skill
                best_skill = max(skill_list, key=lambda s: s.confidence)
                
                # Combine proficiency indicators
                all_proficiency = []
                for s in skill_list:
                    all_proficiency.extend(s.proficiency_indicators)
                best_skill.proficiency_indicators = list(set(all_proficiency))
                
                # Use the highest years mentioned
                all_years = [s.years_mentioned for s in skill_list if s.years_mentioned]
                if all_years:
                    best_skill.years_mentioned = max(all_years)
                
                # Combine contexts
                contexts = [s.context for s in skill_list]
                best_skill.context = ' | '.join(contexts[:3])  # Limit to 3 contexts
                
                merged_skills.append(best_skill)
        
        # Sort by confidence
        merged_skills.sort(key=lambda s: s.confidence, reverse=True)
        
        return merged_skills
    
    def extract_skill_requirements(self, job_text: str) -> Dict[str, List[ExtractedSkill]]:
        """Extract required vs preferred skills from job description."""
        
        # Split text into sections
        required_section = ""
        preferred_section = ""
        
        # Look for required skills section
        required_patterns = [
            r'required\s*(?:skills?|qualifications?|experience)?\s*:?\s*(.*?)(?=preferred|nice|bonus|plus|\n\n|$)',
            r'must\s*have\s*:?\s*(.*?)(?=preferred|nice|bonus|plus|\n\n|$)',
            r'essential\s*:?\s*(.*?)(?=preferred|nice|bonus|plus|\n\n|$)'
        ]
        
        for pattern in required_patterns:
            match = re.search(pattern, job_text, re.IGNORECASE | re.DOTALL)
            if match:
                required_section = match.group(1)
                break
        
        # Look for preferred skills section
        preferred_patterns = [
            r'preferred\s*(?:skills?|qualifications?|experience)?\s*:?\s*(.*?)(?=required|\n\n|$)',
            r'nice\s*to\s*have\s*:?\s*(.*?)(?=required|\n\n|$)',
            r'bonus\s*:?\s*(.*?)(?=required|\n\n|$)',
            r'plus\s*:?\s*(.*?)(?=required|\n\n|$)'
        ]
        
        for pattern in preferred_patterns:
            match = re.search(pattern, job_text, re.IGNORECASE | re.DOTALL)
            if match:
                preferred_section = match.group(1)
                break
        
        # If no clear sections found, extract from entire text
        if not required_section and not preferred_section:
            all_skills = self.extract_skills(job_text)
            return {
                'required': all_skills[:len(all_skills)//2],  # First half as required
                'preferred': all_skills[len(all_skills)//2:]  # Second half as preferred
            }
        
        return {
            'required': self.extract_skills(required_section) if required_section else [],
            'preferred': self.extract_skills(preferred_section) if preferred_section else []
        }
    
    def get_skill_statistics(self, skills: List[ExtractedSkill]) -> Dict[str, any]:
        """Get statistics about extracted skills."""
        
        if not skills:
            return {}
        
        # Count by category
        category_counts = {}
        for skill in skills:
            category = skill.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Average confidence
        avg_confidence = sum(s.confidence for s in skills) / len(skills)
        
        # Skills with years mentioned
        skills_with_years = [s for s in skills if s.years_mentioned]
        
        return {
            'total_skills': len(skills),
            'category_distribution': category_counts,
            'average_confidence': avg_confidence,
            'skills_with_experience_years': len(skills_with_years),
            'max_years_mentioned': max([s.years_mentioned for s in skills_with_years]) if skills_with_years else None,
            'high_confidence_skills': len([s for s in skills if s.confidence > 0.8])
        } 