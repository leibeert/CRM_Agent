"""
Intelligent job description parser using AI to extract structured information.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import json

from ..utils.config import get_config
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI not available, using fallback parsing")
    OPENAI_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not available, using basic NLP")
    SPACY_AVAILABLE = False


@dataclass
class JobRequirement:
    """Represents a single job requirement."""
    skill: str
    importance: str  # 'required', 'preferred', 'nice_to_have'
    proficiency_level: str  # 'beginner', 'intermediate', 'advanced', 'expert'
    years_experience: Optional[int] = None
    context: str = ""  # Original context where this was mentioned


@dataclass
class ParsedJobDescription:
    """Complete parsed job description with structured data."""
    
    # Basic information
    title: str
    company: str
    location: str
    employment_type: str  # 'full_time', 'part_time', 'contract', 'internship'
    remote_policy: str  # 'remote', 'hybrid', 'on_site'
    
    # Requirements
    required_skills: List[JobRequirement]
    preferred_skills: List[JobRequirement]
    experience_requirements: Dict[str, Any]
    education_requirements: Dict[str, Any]
    
    # Job details
    responsibilities: List[str]
    benefits: List[str]
    salary_range: Optional[Dict[str, Any]] = None
    
    # Company and culture
    company_description: str = ""
    team_info: str = ""
    company_culture: List[str] = None
    
    # Metadata
    seniority_level: str = "mid"  # 'entry', 'mid', 'senior', 'lead', 'executive'
    industry: str = ""
    department: str = ""
    
    # Parsing metadata
    confidence_score: float = 0.0
    parsing_method: str = "unknown"
    extracted_entities: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.company_culture is None:
            self.company_culture = []
        if self.extracted_entities is None:
            self.extracted_entities = {}


class IntelligentJobParser:
    """AI-powered job description parser."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        
        # Initialize NLP models
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
        
        # Skill patterns and mappings
        self.skill_patterns = self._load_skill_patterns()
        self.seniority_keywords = self._load_seniority_keywords()
        self.benefit_patterns = self._load_benefit_patterns()
    
    def parse_job_description(self, job_text: str, 
                            job_title: str = "", 
                            company_name: str = "") -> ParsedJobDescription:
        """Parse a job description and extract structured information."""
        
        # Check cache first
        cached_result = self.cache.get_job_analysis(job_text)
        if cached_result:
            return ParsedJobDescription(**cached_result)
        
        try:
            # Try AI-powered parsing first
            if self.openai_client:
                result = self._parse_with_ai(job_text, job_title, company_name)
                if result.confidence_score > 0.7:
                    self.cache.cache_job_analysis(job_text, result.__dict__)
                    return result
            
            # Fallback to rule-based parsing
            result = self._parse_with_rules(job_text, job_title, company_name)
            self.cache.cache_job_analysis(job_text, result.__dict__)
            return result
            
        except Exception as e:
            logger.error(f"Error parsing job description: {str(e)}")
            # Return minimal parsed result
            return ParsedJobDescription(
                title=job_title or "Unknown Position",
                company=company_name or "Unknown Company",
                location="Unknown",
                employment_type="full_time",
                remote_policy="unknown",
                required_skills=[],
                preferred_skills=[],
                experience_requirements={},
                education_requirements={},
                responsibilities=[],
                benefits=[],
                confidence_score=0.1,
                parsing_method="error_fallback"
            )
    
    def _parse_with_ai(self, job_text: str, job_title: str, company_name: str) -> ParsedJobDescription:
        """Parse job description using OpenAI GPT."""
        
        prompt = self._create_parsing_prompt(job_text, job_title, company_name)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst specializing in parsing job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            parsed_data = json.loads(result_text)
            
            # Convert to our data structure
            result = self._convert_ai_result(parsed_data, job_title, company_name)
            result.parsing_method = "openai_gpt"
            result.confidence_score = 0.9
            
            return result
            
        except Exception as e:
            logger.error(f"AI parsing failed: {str(e)}")
            raise
    
    def _parse_with_rules(self, job_text: str, job_title: str, company_name: str) -> ParsedJobDescription:
        """Parse job description using rule-based approach."""
        
        # Clean and normalize text
        cleaned_text = self._clean_text(job_text)
        
        # Extract basic information
        title = job_title or self._extract_title(cleaned_text)
        company = company_name or self._extract_company(cleaned_text)
        location = self._extract_location(cleaned_text)
        employment_type = self._extract_employment_type(cleaned_text)
        remote_policy = self._extract_remote_policy(cleaned_text)
        
        # Extract skills and requirements
        required_skills = self._extract_required_skills(cleaned_text)
        preferred_skills = self._extract_preferred_skills(cleaned_text)
        experience_req = self._extract_experience_requirements(cleaned_text)
        education_req = self._extract_education_requirements(cleaned_text)
        
        # Extract job details
        responsibilities = self._extract_responsibilities(cleaned_text)
        benefits = self._extract_benefits(cleaned_text)
        salary_range = self._extract_salary_range(cleaned_text)
        
        # Extract company and culture info
        company_desc = self._extract_company_description(cleaned_text)
        team_info = self._extract_team_info(cleaned_text)
        culture_keywords = self._extract_culture_keywords(cleaned_text)
        
        # Determine metadata
        seniority_level = self._determine_seniority_level(cleaned_text, title)
        industry = self._determine_industry(cleaned_text, company)
        department = self._determine_department(cleaned_text, title)
        
        # Calculate confidence based on extraction success
        confidence = self._calculate_parsing_confidence(
            required_skills, preferred_skills, responsibilities, benefits
        )
        
        return ParsedJobDescription(
            title=title,
            company=company,
            location=location,
            employment_type=employment_type,
            remote_policy=remote_policy,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_requirements=experience_req,
            education_requirements=education_req,
            responsibilities=responsibilities,
            benefits=benefits,
            salary_range=salary_range,
            company_description=company_desc,
            team_info=team_info,
            company_culture=culture_keywords,
            seniority_level=seniority_level,
            industry=industry,
            department=department,
            confidence_score=confidence,
            parsing_method="rule_based"
        )
    
    def _create_parsing_prompt(self, job_text: str, job_title: str, company_name: str) -> str:
        """Create a prompt for AI-powered parsing."""
        
        return f"""
Parse the following job description and extract structured information. Return the result as a JSON object with the following structure:

{{
    "title": "job title",
    "company": "company name",
    "location": "location",
    "employment_type": "full_time|part_time|contract|internship",
    "remote_policy": "remote|hybrid|on_site",
    "required_skills": [
        {{"skill": "skill name", "importance": "required", "proficiency_level": "beginner|intermediate|advanced|expert", "years_experience": 2}}
    ],
    "preferred_skills": [
        {{"skill": "skill name", "importance": "preferred", "proficiency_level": "intermediate", "years_experience": 1}}
    ],
    "experience_requirements": {{
        "minimum_years": 3,
        "preferred_years": 5,
        "industry_experience": "technology"
    }},
    "education_requirements": {{
        "minimum_degree": "bachelor",
        "preferred_degree": "master",
        "field_of_study": "computer science"
    }},
    "responsibilities": ["responsibility 1", "responsibility 2"],
    "benefits": ["benefit 1", "benefit 2"],
    "salary_range": {{"min": 80000, "max": 120000, "currency": "USD"}},
    "company_description": "brief company description",
    "team_info": "team information",
    "company_culture": ["culture keyword 1", "culture keyword 2"],
    "seniority_level": "entry|mid|senior|lead|executive",
    "industry": "industry name",
    "department": "department name"
}}

Job Title: {job_title}
Company: {company_name}

Job Description:
{job_text}

Extract as much information as possible. If information is not available, use appropriate defaults or empty values.
"""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize job description text."""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\/\&\+\#\$\%]', '', text)
        
        return text.strip()
    
    def _extract_required_skills(self, text: str) -> List[JobRequirement]:
        """Extract required skills from job description."""
        
        required_skills = []
        
        # Look for required skills sections
        required_patterns = [
            r'required\s+(?:skills?|qualifications?|experience)[:\s]+(.*?)(?=preferred|nice|benefits|responsibilities|$)',
            r'must\s+have[:\s]+(.*?)(?=preferred|nice|benefits|responsibilities|$)',
            r'essential\s+(?:skills?|requirements?)[:\s]+(.*?)(?=preferred|nice|benefits|responsibilities|$)'
        ]
        
        for pattern in required_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills_text = match.group(1)
                skills = self._extract_skills_from_text(skills_text, "required")
                required_skills.extend(skills)
        
        # Look for specific skill patterns
        for skill_pattern in self.skill_patterns:
            if re.search(skill_pattern, text, re.IGNORECASE):
                skill_name = self._extract_skill_name_from_pattern(skill_pattern, text)
                if skill_name:
                    required_skills.append(JobRequirement(
                        skill=skill_name,
                        importance="required",
                        proficiency_level="intermediate"
                    ))
        
        return self._deduplicate_skills(required_skills)
    
    def _extract_preferred_skills(self, text: str) -> List[JobRequirement]:
        """Extract preferred/nice-to-have skills."""
        
        preferred_skills = []
        
        # Look for preferred skills sections
        preferred_patterns = [
            r'preferred\s+(?:skills?|qualifications?|experience)[:\s]+(.*?)(?=required|benefits|responsibilities|$)',
            r'nice\s+to\s+have[:\s]+(.*?)(?=required|benefits|responsibilities|$)',
            r'bonus\s+(?:skills?|points?)[:\s]+(.*?)(?=required|benefits|responsibilities|$)'
        ]
        
        for pattern in preferred_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills_text = match.group(1)
                skills = self._extract_skills_from_text(skills_text, "preferred")
                preferred_skills.extend(skills)
        
        return self._deduplicate_skills(preferred_skills)
    
    def _extract_skills_from_text(self, text: str, importance: str) -> List[JobRequirement]:
        """Extract individual skills from a text block."""
        
        skills = []
        
        # Split by common delimiters
        skill_candidates = re.split(r'[,\n\r\•\-\*]', text)
        
        for candidate in skill_candidates:
            candidate = candidate.strip()
            if len(candidate) > 2 and len(candidate) < 50:  # Reasonable skill name length
                
                # Extract years of experience if mentioned
                years_match = re.search(r'(\d+)\s*\+?\s*years?', candidate, re.IGNORECASE)
                years_experience = int(years_match.group(1)) if years_match else None
                
                # Clean skill name
                skill_name = re.sub(r'\d+\s*\+?\s*years?', '', candidate, flags=re.IGNORECASE)
                skill_name = re.sub(r'[^\w\s\-\.\+\#]', '', skill_name).strip()
                
                if skill_name and len(skill_name) > 1:
                    # Determine proficiency level
                    proficiency = self._determine_proficiency_level(candidate)
                    
                    skills.append(JobRequirement(
                        skill=skill_name,
                        importance=importance,
                        proficiency_level=proficiency,
                        years_experience=years_experience,
                        context=candidate
                    ))
        
        return skills
    
    def _determine_proficiency_level(self, text: str) -> str:
        """Determine proficiency level from context."""
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['expert', 'advanced', 'senior', 'lead']):
            return 'expert'
        elif any(word in text_lower for word in ['intermediate', 'mid', 'solid', 'good']):
            return 'intermediate'
        elif any(word in text_lower for word in ['beginner', 'entry', 'basic', 'junior']):
            return 'beginner'
        else:
            return 'intermediate'  # Default
    
    def _load_skill_patterns(self) -> List[str]:
        """Load common skill patterns for recognition."""
        
        return [
            r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\btypescript\b',
            r'\breact\b', r'\bangular\b', r'\bvue\.?js\b', r'\bnode\.?js\b',
            r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bdocker\b', r'\bkubernetes\b',
            r'\bmysql\b', r'\bpostgresql\b', r'\bmongodb\b', r'\bredis\b',
            r'\bgit\b', r'\blinux\b', r'\bagile\b', r'\bscrum\b',
            r'\bmachine\s+learning\b', r'\bdata\s+science\b', r'\bai\b'
        ]
    
    def _load_seniority_keywords(self) -> Dict[str, List[str]]:
        """Load keywords for determining seniority level."""
        
        return {
            'entry': ['entry', 'junior', 'associate', 'trainee', 'intern', 'graduate'],
            'mid': ['mid', 'intermediate', 'regular', 'standard'],
            'senior': ['senior', 'sr', 'experienced', 'lead', 'principal'],
            'lead': ['lead', 'team lead', 'tech lead', 'principal', 'staff'],
            'executive': ['director', 'vp', 'cto', 'ceo', 'head of', 'chief']
        }
    
    def _load_benefit_patterns(self) -> List[str]:
        """Load patterns for recognizing benefits."""
        
        return [
            r'health\s+insurance', r'dental\s+insurance', r'vision\s+insurance',
            r'401\(k\)', r'retirement\s+plan', r'pension',
            r'paid\s+time\s+off', r'pto', r'vacation\s+days',
            r'flexible\s+hours', r'remote\s+work', r'work\s+from\s+home',
            r'stock\s+options', r'equity', r'bonus',
            r'professional\s+development', r'training', r'conference'
        ]
    
    # Additional helper methods (stubs for brevity)
    
    def _extract_title(self, text: str) -> str:
        """Extract job title from text."""
        return "Software Engineer"  # Stub
    
    def _extract_company(self, text: str) -> str:
        """Extract company name from text."""
        return "Unknown Company"  # Stub
    
    def _extract_location(self, text: str) -> str:
        """Extract location from text."""
        return "Remote"  # Stub
    
    def _extract_employment_type(self, text: str) -> str:
        """Extract employment type."""
        return "full_time"  # Stub
    
    def _extract_remote_policy(self, text: str) -> str:
        """Extract remote work policy."""
        return "hybrid"  # Stub
    
    def _extract_experience_requirements(self, text: str) -> Dict[str, Any]:
        """Extract experience requirements."""
        return {"minimum_years": 3}  # Stub
    
    def _extract_education_requirements(self, text: str) -> Dict[str, Any]:
        """Extract education requirements."""
        return {"minimum_degree": "bachelor"}  # Stub
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities."""
        return ["Develop software applications"]  # Stub
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefits."""
        return ["Health insurance", "401k"]  # Stub
    
    def _extract_salary_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract salary range."""
        return None  # Stub
    
    def _extract_company_description(self, text: str) -> str:
        """Extract company description."""
        return ""  # Stub
    
    def _extract_team_info(self, text: str) -> str:
        """Extract team information."""
        return ""  # Stub
    
    def _extract_culture_keywords(self, text: str) -> List[str]:
        """Extract company culture keywords."""
        return []  # Stub
    
    def _determine_seniority_level(self, text: str, title: str) -> str:
        """Determine seniority level."""
        return "mid"  # Stub
    
    def _determine_industry(self, text: str, company: str) -> str:
        """Determine industry."""
        return "technology"  # Stub
    
    def _determine_department(self, text: str, title: str) -> str:
        """Determine department."""
        return "engineering"  # Stub
    
    def _calculate_parsing_confidence(self, required_skills, preferred_skills, 
                                    responsibilities, benefits) -> float:
        """Calculate confidence in parsing results."""
        
        score = 0.0
        
        if required_skills:
            score += 0.3
        if preferred_skills:
            score += 0.2
        if responsibilities:
            score += 0.3
        if benefits:
            score += 0.2
        
        return min(1.0, score)
    
    def _deduplicate_skills(self, skills: List[JobRequirement]) -> List[JobRequirement]:
        """Remove duplicate skills."""
        
        seen_skills = set()
        unique_skills = []
        
        for skill in skills:
            skill_key = skill.skill.lower().strip()
            if skill_key not in seen_skills:
                seen_skills.add(skill_key)
                unique_skills.append(skill)
        
        return unique_skills
    
    def _extract_skill_name_from_pattern(self, pattern: str, text: str) -> Optional[str]:
        """Extract skill name from regex pattern match."""
        
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        return None
    
    def _convert_ai_result(self, parsed_data: Dict[str, Any], 
                          job_title: str, company_name: str) -> ParsedJobDescription:
        """Convert AI parsing result to our data structure."""
        
        # Convert skills to JobRequirement objects
        required_skills = []
        for skill_data in parsed_data.get('required_skills', []):
            required_skills.append(JobRequirement(
                skill=skill_data.get('skill', ''),
                importance='required',
                proficiency_level=skill_data.get('proficiency_level', 'intermediate'),
                years_experience=skill_data.get('years_experience')
            ))
        
        preferred_skills = []
        for skill_data in parsed_data.get('preferred_skills', []):
            preferred_skills.append(JobRequirement(
                skill=skill_data.get('skill', ''),
                importance='preferred',
                proficiency_level=skill_data.get('proficiency_level', 'intermediate'),
                years_experience=skill_data.get('years_experience')
            ))
        
        return ParsedJobDescription(
            title=parsed_data.get('title', job_title),
            company=parsed_data.get('company', company_name),
            location=parsed_data.get('location', 'Unknown'),
            employment_type=parsed_data.get('employment_type', 'full_time'),
            remote_policy=parsed_data.get('remote_policy', 'unknown'),
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_requirements=parsed_data.get('experience_requirements', {}),
            education_requirements=parsed_data.get('education_requirements', {}),
            responsibilities=parsed_data.get('responsibilities', []),
            benefits=parsed_data.get('benefits', []),
            salary_range=parsed_data.get('salary_range'),
            company_description=parsed_data.get('company_description', ''),
            team_info=parsed_data.get('team_info', ''),
            company_culture=parsed_data.get('company_culture', []),
            seniority_level=parsed_data.get('seniority_level', 'mid'),
            industry=parsed_data.get('industry', ''),
            department=parsed_data.get('department', '')
        ) 