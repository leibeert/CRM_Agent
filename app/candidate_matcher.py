import json
import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

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

# Prompt for analyzing job description to extract required skills
analysis_prompt = PromptTemplate.from_template(
    "Analyze the following job description and extract the key required skills and technologies.\nList the skills concisely, separated by commas.\n\nJob Description: {job_description}\n\nRequired Skills:"
)

analysis_chain = analysis_prompt | llm | output_parser

# Prompt for checking if two skills are related
relatedness_prompt = PromptTemplate.from_template(
    "Are the following two skills closely related or represent a similar area of expertise? Answer only YES or NO.\nSkill 1: {skill1}\nSkill 2: {skill2}\nRelated:"
)

relatedness_chain = relatedness_prompt | llm | output_parser

class CandidateMatcher:
    def __init__(self):
        # Use Langchain SQLDatabase for querying candidates
        self.db_chain = self._get_db_chain()
        self.llm = llm # Use the initialized LLM
        self.analysis_chain = analysis_chain
        self.relatedness_chain = relatedness_chain # Use the new relatedness chain

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

# Example usage (for testing within this file if needed)
if __name__ == '__main__':
    matcher = CandidateMatcher()
    # Make sure you have data in your database for this to work
    # Example job description
    job_desc = "We are looking for a Python developer with strong Django and React experience."
    # job_desc = "Need an experienced Go programmer."

    matching_candidates = matcher.find_matching_candidates(job_desc)

    if matching_candidates:
        print(f"\nMatching Candidates for '{job_desc}':")
        for candidate in matching_candidates:
            print(f"- {candidate['first_name']} {candidate['last_name']}: {candidate['match_score']:.1f}% Match")
            if candidate['match_details']:
                print("  Match Details:")
                for detail in candidate['match_details']:
                    print(f"  - {detail}")
    else:
        print("No matching candidates found.") 