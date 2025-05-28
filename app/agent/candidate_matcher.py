import pandas as pd
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pymysql

# Load environment variables
load_dotenv()

class CandidateMatcher:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=os.getenv('MODEL_NAME', 'gpt-4-turbo-preview'),
            temperature=float(os.getenv('TEMPERATURE', 0.7))
        )
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'recruitment_crm')
        }

    def get_candidate_skills(self) -> pd.DataFrame:
        """Get candidate skills directly from the database."""
        try:
            # Connect to database
            conn = pymysql.connect(**self.db_config)
            
            # SQL query to get candidate skills
            query = """
            SELECT 
                r.id as resource_id,
                r.first_name,
                r.last_name,
                rs.id as resource_skill_id,
                rs.level,
                rs.duration,
                s.id as skill_id,
                s.name as skill_name
            FROM resources r
            JOIN resource_skills rs ON r.id = rs.resource_id
            JOIN skills s ON rs.skill_id = s.id
            ORDER BY r.id, s.name
            """
            
            # Read directly into DataFrame
            df = pd.read_sql(query, conn)
            return df
            
        except Exception as e:
            print(f"Error getting candidate skills: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def analyze_job_description(self, job_description: str) -> List[str]:
        """Extract required skills from job description."""
        prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""
            Analyze the following job description and extract the required technical skills.
            Return only a list of skills, one per line, without any additional text.
            
            Job Description:
            {job_description}
            """
        )
        
        response = self.llm.predict(prompt.format(job_description=job_description))
        required_skills = [skill.strip() for skill in response.split('\n') if skill.strip()]
        return required_skills

    def calculate_skill_match_score(self, candidate_skills: pd.DataFrame, required_skills: List[str]) -> float:
        """Calculate how well a candidate's skills match the required skills."""
        # Convert all skills to lowercase for comparison
        candidate_skill_names = candidate_skills['skill_name'].str.lower().tolist()
        required_skills_lower = [skill.lower() for skill in required_skills]
        
        # Calculate exact matches
        exact_matches = sum(1 for skill in required_skills_lower if skill in candidate_skill_names)
        
        # Calculate partial matches (skills that contain required skill as substring)
        partial_matches = 0
        for req_skill in required_skills_lower:
            for cand_skill in candidate_skill_names:
                if req_skill in cand_skill or cand_skill in req_skill:
                    partial_matches += 0.5
                    break
        
        # Calculate total match score
        total_matches = exact_matches + partial_matches
        return total_matches / len(required_skills) if required_skills else 0

    def find_matching_candidates(self, job_description: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Find the top matching candidates for a job description."""
        # Get latest candidate skills from database
        skills_df = self.get_candidate_skills()
        
        # Extract required skills from job description
        required_skills = self.analyze_job_description(job_description)
        print(f"\nRequired skills extracted: {required_skills}")
        
        # Group candidate skills by resource
        candidate_groups = skills_df.groupby('resource_id')
        
        # Calculate match scores for each candidate
        candidate_scores = []
        for resource_id, skills in candidate_groups:
            # Calculate skill match score
            skill_score = self.calculate_skill_match_score(skills, required_skills)
            
            # Get candidate details
            candidate = skills.iloc[0]
            candidate_scores.append({
                'resource_id': resource_id,
                'first_name': candidate['first_name'],
                'last_name': candidate['last_name'],
                'match_score': skill_score,
                'skills': skills[['skill_name', 'level', 'duration']].to_dict('records')
            })
        
        # Sort candidates by match score
        candidate_scores.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Return top N candidates
        return candidate_scores[:top_n] 