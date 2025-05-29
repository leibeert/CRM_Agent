import json
import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging
from typing import Dict, List, Tuple, Any

# Assuming database connection details are loaded from .env in app.py or here
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection for CandidateMatcher (using argoteam DB)
DB_HOST = os.getenv('ARGOTEAM_DB_HOST', 'localhost')
DB_PORT = os.getenv('ARGOTEAM_DB_PORT', '3306')
DB_NAME = os.getenv('ARGOTEAM_DB_NAME', 'argoteam')
DB_USER = os.getenv('ARGOTEAM_DB_USER', 'root')
DB_PASSWORD = os.getenv('ARGOTEAM_DB_PASSWORD', 'admin')

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize LLM for skill analysis and relatedness check
# Ensure OPENAI_API_KEY is set
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5) # Lower temperature for more focused responses
output_parser = StrOutputParser()

# Enhanced prompt for detailed job description analysis
detailed_analysis_prompt = PromptTemplate.from_template(
    """Analyze the following job description and extract detailed requirements in JSON format.

Job Description: {job_description}

Extract and return ONLY a valid JSON object with these fields:
{{
    "required_skills": ["skill1", "skill2", "skill3"],
    "preferred_skills": ["optional_skill1", "optional_skill2"],
    "experience_years": 3,
    "experience_titles": ["Software Developer", "Engineer"],
    "experience_companies": [],
    "education_degrees": ["Bachelor", "Master"],
    "education_fields": ["Computer Science", "Engineering"],
    "industry_keywords": ["fintech", "healthcare", "startup"],
    "seniority_level": "mid"
}}

Focus on technical skills, programming languages, frameworks, tools, and specific requirements.
For experience_years, extract the minimum years mentioned (e.g., "3+ years" -> 3).
For seniority_level, use: "junior", "mid", "senior", or "lead".

JSON:"""
)

# SQL generation prompt
sql_generation_prompt = PromptTemplate.from_template(
    """Based on the job requirements, generate a SQL query to find the best matching candidates.

Requirements: {requirements}

Database Schema:
- resources (id, first_name, last_name, email, phone_number, created_at)
- skills (id, name)
- resource_skills (resource_id, skill_id, level, duration)
- experiences (id, resource_id, title, name, description, start_date, end_date)
- studies (id, resource_id, degree_name, school_id, start_date, end_date)

IMPORTANT MYSQL SYNTAX RULES:
- Use INTERVAL 3 YEAR not INTERVAL '3 years'
- Use proper MySQL date functions
- Keep the query simple and focused on skill matching
- Don't use complex subqueries unless necessary

Generate a SQL query that:
1. Finds candidates with matching skills using INNER JOIN
2. Calculates match_score as: ROUND((COUNT(DISTINCT rs.skill_id) * 100.0 / {total_required_skills}), 1)
3. Uses MySQL-compatible syntax only
4. Groups by candidate ID and basic info
5. Orders by match_score DESC
6. Limits to top 20 candidates
7. Focus on skill matching, keep experience/education optional

Example structure:
SELECT DISTINCT r.id, r.first_name, r.last_name, r.email,
       COUNT(DISTINCT rs.skill_id) as matched_skills,
       ROUND((COUNT(DISTINCT rs.skill_id) * 100.0 / {total_required_skills}), 1) as match_score
FROM resources r
INNER JOIN resource_skills rs ON r.id = rs.resource_id
INNER JOIN skills s ON rs.skill_id = s.id
WHERE s.name IN ('skill1', 'skill2')
GROUP BY r.id, r.first_name, r.last_name, r.email
HAVING matched_skills > 0
ORDER BY match_score DESC, matched_skills DESC
LIMIT 20;

Return ONLY the SQL query, no explanations:"""
)

detailed_analysis_chain = detailed_analysis_prompt | llm | output_parser
sql_generation_chain = sql_generation_prompt | llm | output_parser

class CandidateMatcher:
    def __init__(self):
        # Use Langchain SQLDatabase for querying candidates
        self.db_chain = self._get_db_chain()
        self.llm = llm # Use the initialized LLM
        self.analysis_chain = detailed_analysis_chain
        self.sql_generation_chain = sql_generation_chain # Use the new SQL generation chain

    def _get_db_chain(self):
        db = SQLDatabase.from_uri(
            DATABASE_URL,
            include_tables=[
                'resources',
                'skills',
                'resource_skills',
                'experiences',
                'studies',
                'degree_types',
                'schools'
            ],
            sample_rows_in_table_info=2 # Reduce sample rows to save tokens
        )

        # Simple query chain to get candidate skills
        # This will need to be more sophisticated to query based on job description requirements
        # For now, just getting all candidates and their skills

        # Define a prompt for querying candidate skills based on inferred requirements
        # This is a simplified example; a real application would need a more complex query generation process
        sql_prompt = PromptTemplate.from_template(
            """Based on the user request and the available tables ({table_info}), write a SQL query to find relevant candidates and their skills. \n\nUser Request: {query}\nSQL Query:"""
        )

        # This is just a placeholder chain. The actual querying logic will be in find_matching_candidates
        # where we fetch all candidates and then filter/score them.
        # query_chain = sql_prompt | self.llm | StrOutputParser()

        return db # Returning the SQLDatabase instance for direct querying

    def get_candidate_skills(self):
        """Fetches all candidates and their skills from the database."""
        session = SessionLocal()
        try:
            # Use text() to properly declare the SQL query
            query = text("""
                SELECT r.id, r.first_name, r.last_name, s.name
                FROM resources r
                JOIN resource_skills rs ON r.id = rs.resource_id
                JOIN skills s ON rs.skill_id = s.id
            """
            )
            
            result = session.execute(query).fetchall()

            # Process the result to group skills by candidate
            candidates_with_skills = {}
            for row in result:
                candidate_id, first_name, last_name, skill_name = row
                full_name = f"{first_name} {last_name}"
                if full_name not in candidates_with_skills:
                    candidates_with_skills[full_name] = {
                        'id': candidate_id,
                        'first_name': first_name,
                        'last_name': last_name,
                        'skills': []
                    }
                candidates_with_skills[full_name]['skills'].append(skill_name)

            # Convert to a list of dictionaries
            return list(candidates_with_skills.values())

        except Exception as e:
            logger.error(f"Error fetching candidate skills: {str(e)}")
            return []
        finally:
            session.close()

    def analyze_job_description(self, job_description: str) -> list[str]:
        """Analyzes job description to extract required skills using LLM."""
        try:
            logger.info(f"Analyzing job description: {job_description}")
            required_skills_str = self.analysis_chain.invoke({"job_description": job_description})
            # Split the string by comma and strip whitespace
            required_skills = [skill.strip() for skill in required_skills_str.split(',') if skill.strip()]
            logger.info(f"Required skills extracted: {required_skills}")
            return required_skills
        except Exception as e:
            logger.error(f"Error analyzing job description: {str(e)}")
            return []

    def calculate_skill_match_score(self, required_skills: list[str], candidate_skills: list[str]) -> tuple[float, list[str]]:
        """Calculates a match score based on required and candidate skills.
           Returns score (0-1) and details of matched skills.
           This version only checks for exact matches for performance reasons.
        """
        logger.debug(f"--- Starting skill matching (exact only) for candidate against required skills: {required_skills} ---")
        logger.debug(f"Candidate possesses skills: {candidate_skills}")

        exact_matches = 0
        matched_details = []

        # Normalize skills (lowercase for comparison)
        normalized_required = [s.lower() for s in required_skills]
        normalized_candidate = [s.lower() for s in candidate_skills]

        # Keep track of original case for matched skills
        required_skills_map = {s.lower(): s for s in required_skills}
        candidate_skills_map = {s.lower(): s for s in candidate_skills}

        # Keep track of used candidate skills (by their normalized name) to avoid double counting
        used_candidate_skills_normalized = set()

        logger.debug("Checking for exact matches...")
        for req_skill_normalized in normalized_required:
            if req_skill_normalized in normalized_candidate and req_skill_normalized not in used_candidate_skills_normalized:
                 original_req_skill = required_skills_map[req_skill_normalized]
                 # original_cand_skill = candidate_skills_map[req_skill_normalized] # Not needed for exact match detail
                 matched_details.append(f"{original_req_skill} (Exact match)")
                 exact_matches += 1
                 used_candidate_skills_normalized.add(req_skill_normalized) # Mark as used
                 logger.debug(f"Exact match found: {original_req_skill}")

        # Calculate score
        total_required = len(required_skills)
        if total_required == 0:
            logger.debug("No required skills provided, score is 0.0")
            logger.debug("--- Finished skill matching (exact only) ---")
            return 0.0, matched_details # Avoid division by zero

        # Score is now based only on exact matches
        score = exact_matches / total_required
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        logger.debug(f"Final match score (exact only) for this candidate: {score:.2f}, Details: {matched_details}")
        logger.debug("--- Finished skill matching (exact only) ---")
        # Return 0 for partial matches as this logic is removed
        return score, matched_details

    def find_matching_candidates(self, job_description: str) -> list[dict]:
        """Finds and ranks candidates based on skill match with job description."""
        logger.info("Starting candidate matching process.")
        required_skills = self.analyze_job_description(job_description)
        if not required_skills:
            logger.info("No required skills extracted from job description. Returning empty list.")
            logger.info("--- Finished candidate matching process ---")
            return []

        all_candidates_with_skills = self.get_candidate_skills()
        if not all_candidates_with_skills:
            logger.warning("No candidates found in the database. Returning empty list.")
            logger.info("--- Finished candidate matching process ---")
            return []

        scored_candidates = []
        logger.debug(f"Processing {len(all_candidates_with_skills)} candidates for matching.")
        for candidate in all_candidates_with_skills:
            logger.debug(f"Processing candidate: {candidate.get('first_name', 'N/A')} {candidate.get('last_name', 'N/A')} (ID: {candidate.get('id', 'N/A')})")
            score, match_details = self.calculate_skill_match_score(
                required_skills,
                candidate.get('skills', []) # Ensure skills key exists
            )
            if score > 0:
                scored_candidates.append({
                    'id': candidate['id'],
                    'first_name': candidate['first_name'],
                    'last_name': candidate['last_name'],
                    'match_score': score,
                    'match_details': match_details # Include match details
                })

        # Sort candidates by match score in descending order
        scored_candidates.sort(key=lambda x: x['match_score'], reverse=True)

        logger.info(f"Finished candidate matching process. Found {len(scored_candidates)} candidates with score > 0.")

        return scored_candidates

    def analyze_job_description_detailed(self, job_description: str) -> Dict[str, Any]:
        """Enhanced job description analysis that extracts comprehensive requirements."""
        try:
            logger.info(f"Performing detailed analysis of job description: {job_description[:100]}...")
            
            # Get LLM analysis
            analysis_result = self.analysis_chain.invoke({"job_description": job_description})
            logger.debug(f"Raw LLM analysis result: {analysis_result}")
            
            # Try to parse JSON response
            try:
                # Clean the response - sometimes LLM adds markdown or extra text
                json_start = analysis_result.find('{')
                json_end = analysis_result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    clean_json = analysis_result[json_start:json_end]
                    requirements = json.loads(clean_json)
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                # Fallback to simple extraction
                requirements = self._fallback_extraction(job_description)
            
            # Validate and set defaults
            requirements = self._validate_requirements(requirements)
            logger.info(f"Extracted requirements: {requirements}")
            return requirements
            
        except Exception as e:
            logger.error(f"Error in detailed job description analysis: {str(e)}")
            return self._get_default_requirements()

    def _fallback_extraction(self, job_description: str) -> Dict[str, Any]:
        """Fallback method for extracting requirements when LLM JSON parsing fails."""
        text_lower = job_description.lower()
        
        # Extract skills using common patterns
        skills = []
        skill_patterns = [
            r'\b(python|java|javascript|react|angular|vue|node\.?js|docker|kubernetes|aws|azure|sql|mysql|postgresql|mongodb|git|jenkins|ci\/cd)\b',
            r'\b(\w+(?:\.js|\.py|\.java))\b',  # File extensions
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        # Extract experience years
        experience_years = 0
        years_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?'
        ]
        
        for pattern in years_patterns:
            match = re.search(pattern, text_lower)
            if match:
                experience_years = int(match.group(1))
                break
        
        return {
            "required_skills": list(set(skills[:10])),  # Remove duplicates, limit to 10
            "preferred_skills": [],
            "experience_years": experience_years,
            "experience_titles": [],
            "experience_companies": [],
            "education_degrees": [],
            "education_fields": [],
            "industry_keywords": [],
            "seniority_level": "mid"
        }

    def _validate_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set defaults for extracted requirements."""
        defaults = {
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": 0,
            "experience_titles": [],
            "experience_companies": [],
            "education_degrees": [],
            "education_fields": [],
            "industry_keywords": [],
            "seniority_level": "mid"
        }
        
        validated = defaults.copy()
        for key, default_value in defaults.items():
            if key in requirements:
                validated[key] = requirements[key] if requirements[key] else default_value
            else:
                validated[key] = default_value
        
        return validated

    def _get_default_requirements(self) -> Dict[str, Any]:
        """Get default requirements structure."""
        return {
            "required_skills": [],
            "preferred_skills": [],
            "experience_years": 0,
            "experience_titles": [],
            "experience_companies": [],
            "education_degrees": [],
            "education_fields": [],
            "industry_keywords": [],
            "seniority_level": "mid"
        }

    def generate_sql_query(self, requirements: Dict[str, Any]) -> str:
        """Generate SQL query based on extracted requirements."""
        try:
            logger.info("Generating SQL query from requirements")
            
            required_skills = requirements.get("required_skills", [])
            total_required_skills = len(required_skills) if required_skills else 1
            
            # Use LLM to generate SQL query with skill count
            prompt_vars = {
                "requirements": json.dumps(requirements),
                "total_required_skills": total_required_skills
            }
            sql_query = self.sql_generation_chain.invoke(prompt_vars)
            
            # Clean the SQL query
            sql_query = sql_query.strip()
            if sql_query.startswith('```sql'):
                sql_query = sql_query[6:]
            if sql_query.endswith('```'):
                sql_query = sql_query[:-3]
            sql_query = sql_query.strip()
            
            logger.debug(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            return self._get_fallback_sql(requirements)

    def _get_fallback_sql(self, requirements: Dict[str, Any]) -> str:
        """Generate a fallback SQL query when LLM generation fails."""
        required_skills = requirements.get("required_skills", [])
        experience_years = requirements.get("experience_years", 0)
        
        if not required_skills:
            # Basic query if no skills specified - just return some candidates
            return """
            SELECT DISTINCT r.id, r.first_name, r.last_name, r.email,
                   COUNT(rs.skill_id) as matched_skills,
                   GREATEST(COUNT(rs.skill_id) * 20.0, 10.0) as match_score
            FROM resources r
            LEFT JOIN resource_skills rs ON r.id = rs.resource_id
            GROUP BY r.id, r.first_name, r.last_name, r.email
            ORDER BY matched_skills DESC
            LIMIT 20;
            """
        
        # Create skill conditions with proper escaping for MySQL
        skill_conditions = []
        for skill in required_skills[:5]:
            # Escape single quotes for SQL safety and use exact matches
            escaped_skill = skill.replace("'", "''")
            skill_conditions.append(f"s.name = '{escaped_skill}'")
        
        skill_where = " OR ".join(skill_conditions)
        total_skills = len(required_skills)
        
        # Simple, reliable SQL query focused on skill matching
        query = f"""
        SELECT DISTINCT r.id, r.first_name, r.last_name, r.email,
               COUNT(DISTINCT rs.skill_id) as matched_skills,
               ROUND((COUNT(DISTINCT rs.skill_id) * 100.0 / {total_skills}), 1) as match_score
        FROM resources r
        INNER JOIN resource_skills rs ON r.id = rs.resource_id
        INNER JOIN skills s ON rs.skill_id = s.id
        WHERE ({skill_where})
        GROUP BY r.id, r.first_name, r.last_name, r.email
        HAVING matched_skills > 0
        ORDER BY match_score DESC, matched_skills DESC
        LIMIT 20;
        """
        
        return query

    def execute_candidate_search(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute the generated SQL query to find candidates."""
        session = SessionLocal()
        try:
            logger.info("Executing candidate search query")
            result = session.execute(text(sql_query)).fetchall()
            
            candidates = []
            for row in result:
                # Convert row to dictionary
                candidate = dict(row._mapping) if hasattr(row, '_mapping') else dict(zip(row.keys(), row))
                candidates.append(candidate)
            
            logger.info(f"Found {len(candidates)} candidates from SQL query")
            return candidates
            
        except Exception as e:
            logger.error(f"Error executing candidate search: {str(e)}")
            return []
        finally:
            session.close()

    def enhance_candidate_details(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance candidate results with additional details like skills and experience."""
        session = SessionLocal()
        try:
            enhanced_candidates = []
            
            for candidate in candidates:
                candidate_id = candidate.get('id')
                if not candidate_id:
                    continue
                
                # Get candidate skills
                skills_query = text("""
                    SELECT s.name, rs.level, rs.duration
                    FROM skills s
                    JOIN resource_skills rs ON s.id = rs.skill_id
                    WHERE rs.resource_id = :candidate_id
                    ORDER BY rs.level DESC, s.name
                """)
                skills_result = session.execute(skills_query, {"candidate_id": candidate_id}).fetchall()
                
                # Get candidate experience
                exp_query = text("""
                    SELECT title, name, description, start_date, end_date
                    FROM experiences
                    WHERE resource_id = :candidate_id
                    ORDER BY start_date DESC
                    LIMIT 5
                """)
                exp_result = session.execute(exp_query, {"candidate_id": candidate_id}).fetchall()
                
                # Get candidate education
                edu_query = text("""
                    SELECT degree_name, school_id, end_date
                    FROM studies
                    WHERE resource_id = :candidate_id
                    ORDER BY end_date DESC
                    LIMIT 3
                """)
                edu_result = session.execute(edu_query, {"candidate_id": candidate_id}).fetchall()
                
                # Build enhanced candidate object
                enhanced_candidate = candidate.copy()
                enhanced_candidate['skills'] = [
                    {
                        'name': skill.name,
                        'level': skill.level or 1,
                        'duration': skill.duration or 0
                    } for skill in skills_result
                ]
                enhanced_candidate['experience'] = [
                    {
                        'title': exp.title,
                        'company': exp.name,  # Using 'name' field as company
                        'description': exp.description,
                        'start_date': exp.start_date.isoformat() if exp.start_date else None,
                        'end_date': exp.end_date.isoformat() if exp.end_date else None
                    } for exp in exp_result
                ]
                enhanced_candidate['education'] = [
                    {
                        'degree_name': edu.degree_name,
                        'school_id': edu.school_id,
                        'graduation_date': edu.end_date.isoformat() if edu.end_date else None
                    } for edu in edu_result
                ]
                
                enhanced_candidates.append(enhanced_candidate)
            
            return enhanced_candidates
            
        except Exception as e:
            logger.error(f"Error enhancing candidate details: {str(e)}")
            return candidates  # Return original candidates if enhancement fails
        finally:
            session.close()

    def find_matching_candidates_enhanced(self, job_description: str) -> Dict[str, Any]:
        """Enhanced candidate matching with detailed analysis and SQL generation."""
        logger.info("Starting enhanced candidate matching process")
        
        try:
            # Step 1: Analyze job description
            requirements = self.analyze_job_description_detailed(job_description)
            
            # Step 2: Generate SQL query
            sql_query = self.generate_sql_query(requirements)
            
            # Step 3: Execute query
            candidates = self.execute_candidate_search(sql_query)
            
            # Step 4: Enhance results with detailed information
            enhanced_candidates = self.enhance_candidate_details(candidates)
            
            # Step 5: Prepare response
            response = {
                "requirements_extracted": requirements,
                "sql_query": sql_query,
                "candidates_found": len(enhanced_candidates),
                "candidates": enhanced_candidates[:10],  # Limit to top 10 for display
                "search_summary": self._generate_search_summary(requirements, len(enhanced_candidates))
            }
            
            logger.info(f"Enhanced matching completed. Found {len(enhanced_candidates)} candidates")
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced candidate matching: {str(e)}")
            return {
                "requirements_extracted": {},
                "sql_query": "",
                "candidates_found": 0,
                "candidates": [],
                "search_summary": f"Error occurred during search: {str(e)}"
            }

    def _generate_search_summary(self, requirements: Dict[str, Any], candidate_count: int) -> str:
        """Generate a human-readable summary of the search."""
        skills = requirements.get("required_skills", [])
        experience_years = requirements.get("experience_years", 0)
        seniority = requirements.get("seniority_level", "")
        
        summary_parts = []
        
        if skills:
            summary_parts.append(f"Skills: {', '.join(skills[:5])}")
        
        if experience_years > 0:
            summary_parts.append(f"Experience: {experience_years}+ years")
        
        if seniority:
            summary_parts.append(f"Level: {seniority}")
        
        criteria = " | ".join(summary_parts)
        
        return f"Found {candidate_count} candidates matching criteria: {criteria}"

# Example usage (for testing within this file if needed)
if __name__ == '__main__':
    matcher = CandidateMatcher()
    # Example job descriptions to test enhanced functionality
    
    print("=== Testing Enhanced Candidate Matching ===\n")
    
    test_jobs = [
        "We are looking for a Senior Python Developer with 5+ years of experience in Django, React, and AWS. Must have Bachelor's degree in Computer Science.",
        "Looking for a Full Stack JavaScript Engineer with React, Node.js, and MongoDB experience. 3+ years required.",
        "Need an experienced DevOps Engineer with Docker, Kubernetes, Jenkins, and cloud platforms (AWS/Azure). 4+ years minimum."
    ]
    
    for i, job_desc in enumerate(test_jobs, 1):
        print(f"\n--- Test Job {i} ---")
        print(f"Job Description: {job_desc}")
        print("\n" + "="*50)
        
        try:
            results = matcher.find_matching_candidates_enhanced(job_desc)
            
            print(f"Requirements Extracted: {results['requirements_extracted']}")
            print(f"Candidates Found: {results['candidates_found']}")
            print(f"Search Summary: {results['search_summary']}")
            
            if results['sql_query']:
                print(f"\nGenerated SQL Query:\n{results['sql_query'][:300]}...")
            
            if results['candidates']:
                print(f"\nTop 3 Candidates:")
                for j, candidate in enumerate(results['candidates'][:3], 1):
                    print(f"{j}. {candidate.get('first_name', 'N/A')} {candidate.get('last_name', 'N/A')} - {candidate.get('match_score', 0):.1f}% match")
                    if candidate.get('skills'):
                        skills = [skill['name'] for skill in candidate['skills'][:3]]
                        print(f"   Skills: {', '.join(skills)}")
            
        except Exception as e:
            print(f"Error testing job {i}: {str(e)}")
        
        print("\n" + "="*50) 