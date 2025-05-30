#!/usr/bin/env python3
"""
CRM App Integration Test for AI Candidate Matching

This script demonstrates how to integrate the ML candidate matching system
with your existing CRM Flask app.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
import pandas as pd

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRMCandidateMatchingService:
    """Service class for integrating ML matching with CRM app."""
    
    def __init__(self, data_path="../../database_tables"):
        self.data_path = data_path
        self.candidates_cache = None
        self.skills_cache = None
        self._load_data()
    
    def _load_data(self):
        """Load candidate data from CSV files."""
        try:
            # Load CSV files
            resources_df = pd.read_csv(os.path.join(self.data_path, "resources.csv"))
            skills_df = pd.read_csv(os.path.join(self.data_path, "skills.csv"))
            resource_skills_df = pd.read_csv(os.path.join(self.data_path, "resource_skills.csv"))
            experiences_df = pd.read_csv(os.path.join(self.data_path, "experiences.csv"))
            
            # Create candidate profiles
            self.candidates_cache = self._create_profiles(
                resources_df, skills_df, resource_skills_df, experiences_df
            )
            self.skills_cache = skills_df['name'].tolist()
            
            logger.info(f"Loaded {len(self.candidates_cache)} candidate profiles")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.candidates_cache = []
            self.skills_cache = []
    
    def _create_profiles(self, resources_df, skills_df, resource_skills_df, experiences_df):
        """Create candidate profiles from dataframes."""
        profiles = []
        
        for _, candidate in resources_df.iterrows():
            candidate_id = candidate['id']
            
            # Base profile
            profile = {
                'id': candidate_id,
                'first_name': candidate.get('first_name', ''),
                'last_name': candidate.get('last_name', ''),
                'email': candidate.get('email', ''),
                'title': candidate.get('title', ''),
                'years_of_experience': candidate.get('years_of_experience', 0),
                'custom_description': candidate.get('custom_description', ''),
                'address': candidate.get('address', ''),
            }
            
            # Add skills
            candidate_skill_ids = resource_skills_df[
                resource_skills_df['resource_id'] == candidate_id
            ]['skill_id'].tolist()
            
            candidate_skills = []
            for skill_id in candidate_skill_ids:
                skill_row = skills_df[skills_df['id'] == skill_id]
                if not skill_row.empty:
                    skill_info = skill_row.iloc[0]
                    candidate_skills.append({
                        'id': skill_id,
                        'name': skill_info['name'],
                        'category': skill_info.get('category', ''),
                    })
            
            profile['skills'] = candidate_skills
            
            # Add experiences
            candidate_experiences = experiences_df[
                experiences_df['resource_id'] == candidate_id
            ].to_dict('records')
            
            profile['experience'] = candidate_experiences
            profiles.append(profile)
        
        return profiles
    
    def search_candidates(self, query_params):
        """
        Search candidates based on job requirements.
        
        Args:
            query_params: Dictionary with job requirements
            
        Returns:
            List of matched candidates with scores
        """
        # Extract parameters
        job_title = query_params.get('job_title', '')
        required_skills = query_params.get('required_skills', [])
        preferred_skills = query_params.get('preferred_skills', [])
        min_experience = query_params.get('min_experience', 0)
        location = query_params.get('location', '')
        
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(',') if s.strip()]
        if isinstance(preferred_skills, str):
            preferred_skills = [s.strip() for s in preferred_skills.split(',') if s.strip()]
        
        # Score all candidates
        candidate_scores = []
        
        for candidate in self.candidates_cache:
            score_data = self._calculate_match_score(
                candidate, required_skills, preferred_skills, min_experience, job_title
            )
            candidate_scores.append(score_data)
        
        # Sort by score
        candidate_scores.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return candidate_scores
    
    def _calculate_match_score(self, candidate, required_skills, preferred_skills, min_experience, job_title):
        """Calculate match score for a candidate."""
        candidate_skills = [skill['name'].lower() for skill in candidate.get('skills', [])]
        
        # Skill matching
        required_matches = 0
        for skill in required_skills:
            if any(skill.lower() in cs or cs in skill.lower() for cs in candidate_skills):
                required_matches += 1
        
        preferred_matches = 0
        for skill in preferred_skills:
            if any(skill.lower() in cs or cs in skill.lower() for cs in candidate_skills):
                preferred_matches += 1
        
        # Calculate skill scores
        required_score = required_matches / len(required_skills) if required_skills else 1.0
        preferred_score = preferred_matches / len(preferred_skills) if preferred_skills else 0.5
        skill_score = 0.7 * required_score + 0.3 * preferred_score
        
        # Experience matching
        candidate_exp = candidate.get('years_of_experience', 0)
        if candidate_exp >= min_experience:
            exp_score = min(1.0, 1.0 + (candidate_exp - min_experience) * 0.05)
        else:
            exp_score = max(0.0, candidate_exp / min_experience) if min_experience > 0 else 0.5
        
        # Title matching
        candidate_title = candidate.get('title', '').lower()
        job_title_lower = job_title.lower()
        title_score = 0.5
        
        # Simple keyword matching for title
        if job_title_lower:
            common_words = ['developer', 'engineer', 'analyst', 'manager', 'lead', 'senior', 'junior']
            for word in common_words:
                if word in candidate_title and word in job_title_lower:
                    title_score += 0.1
        
        title_score = min(1.0, title_score)
        
        # Overall score (skills 50%, experience 30%, title 20%)
        overall_score = 0.5 * skill_score + 0.3 * exp_score + 0.2 * title_score
        
        return {
            'candidate': candidate,
            'skill_score': skill_score,
            'experience_score': exp_score,
            'title_score': title_score,
            'overall_score': overall_score,
            'required_skill_matches': required_matches,
            'preferred_skill_matches': preferred_matches,
            'interpretation': self._interpret_score(overall_score)
        }
    
    def _interpret_score(self, score):
        """Interpret compatibility score."""
        if score >= 0.9:
            return "Excellent match"
        elif score >= 0.8:
            return "Very good match"
        elif score >= 0.7:
            return "Good match"
        elif score >= 0.6:
            return "Fair match"
        elif score >= 0.4:
            return "Poor match"
        else:
            return "Very poor match"
    
    def get_candidate_details(self, candidate_id):
        """Get detailed information about a candidate."""
        for candidate in self.candidates_cache:
            if candidate['id'] == candidate_id:
                return candidate
        return None
    
    def get_available_skills(self):
        """Get list of available skills for autocomplete."""
        return self.skills_cache


# Flask app for testing
app = Flask(__name__)
matching_service = CRMCandidateMatchingService()


@app.route('/api/ai-search', methods=['POST'])
def ai_search():
    """AI-powered candidate search endpoint."""
    try:
        query_params = request.get_json()
        
        # Get top candidates
        top_k = query_params.get('top_k', 10)
        results = matching_service.search_candidates(query_params)[:top_k]
        
        # Format response
        response = {
            'status': 'success',
            'total_candidates': len(matching_service.candidates_cache),
            'results': []
        }
        
        for result in results:
            candidate = result['candidate']
            response['results'].append({
                'id': candidate['id'],
                'name': f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".strip(),
                'title': candidate.get('title', ''),
                'experience_years': candidate.get('years_of_experience', 0),
                'email': candidate.get('email', ''),
                'location': candidate.get('address', ''),
                'skills': [skill['name'] for skill in candidate.get('skills', [])[:10]],
                'compatibility_score': round(result['overall_score'], 3),
                'skill_score': round(result['skill_score'], 3),
                'experience_score': round(result['experience_score'], 3),
                'interpretation': result['interpretation'],
                'required_skill_matches': result['required_skill_matches'],
                'preferred_skill_matches': result['preferred_skill_matches']
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/candidate/<int:candidate_id>', methods=['GET'])
def get_candidate(candidate_id):
    """Get detailed candidate information."""
    candidate = matching_service.get_candidate_details(candidate_id)
    
    if candidate:
        return jsonify({
            'status': 'success',
            'candidate': candidate
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Candidate not found'
        }), 404


@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Get available skills for autocomplete."""
    return jsonify({
        'status': 'success',
        'skills': matching_service.get_available_skills()
    })


@app.route('/test-search', methods=['GET'])
def test_search_page():
    """Test page for the AI search functionality."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Candidate Matching Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .form-group { margin: 10px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, button { padding: 8px; margin: 5px 0; }
            input[type="text"], textarea { width: 300px; }
            button { background: #007cba; color: white; border: none; cursor: pointer; }
            button:hover { background: #005a87; }
            .results { margin-top: 20px; }
            .candidate { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .score { color: #007cba; font-weight: bold; }
            .interpretation { font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>AI Candidate Matching System Test</h1>
        
        <div class="form-group">
            <label>Job Title:</label>
            <input type="text" id="jobTitle" placeholder="e.g., Senior Python Developer">
        </div>
        
        <div class="form-group">
            <label>Required Skills (comma-separated):</label>
            <input type="text" id="requiredSkills" placeholder="e.g., Python, Django, PostgreSQL">
        </div>
        
        <div class="form-group">
            <label>Preferred Skills (comma-separated):</label>
            <input type="text" id="preferredSkills" placeholder="e.g., Docker, AWS, Redis">
        </div>
        
        <div class="form-group">
            <label>Minimum Experience (years):</label>
            <input type="number" id="minExperience" value="3" min="0">
        </div>
        
        <div class="form-group">
            <label>Location:</label>
            <input type="text" id="location" placeholder="e.g., Paris, France">
        </div>
        
        <div class="form-group">
            <label>Top Candidates:</label>
            <input type="number" id="topK" value="10" min="1" max="20">
        </div>
        
        <button onclick="searchCandidates()">Search Candidates</button>
        
        <div id="results" class="results"></div>
        
        <script>
        function searchCandidates() {
            const params = {
                job_title: document.getElementById('jobTitle').value,
                required_skills: document.getElementById('requiredSkills').value,
                preferred_skills: document.getElementById('preferredSkills').value,
                min_experience: parseInt(document.getElementById('minExperience').value) || 0,
                location: document.getElementById('location').value,
                top_k: parseInt(document.getElementById('topK').value) || 10
            };
            
            fetch('/api/ai-search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayResults(data);
                } else {
                    document.getElementById('results').innerHTML = 
                        '<p style="color: red;">Error: ' + data.message + '</p>';
                }
            })
            .catch(error => {
                document.getElementById('results').innerHTML = 
                    '<p style="color: red;">Error: ' + error + '</p>';
            });
        }
        
        function displayResults(data) {
            let html = '<h2>Search Results</h2>';
            html += '<p>Found ' + data.results.length + ' candidates (out of ' + data.total_candidates + ' total)</p>';
            
            data.results.forEach((candidate, index) => {
                html += '<div class="candidate">';
                html += '<h3>' + (index + 1) + '. ' + candidate.name + '</h3>';
                html += '<p><strong>Title:</strong> ' + candidate.title + '</p>';
                html += '<p><strong>Experience:</strong> ' + candidate.experience_years + ' years</p>';
                html += '<p><strong>Email:</strong> ' + candidate.email + '</p>';
                html += '<p><strong>Location:</strong> ' + candidate.location + '</p>';
                html += '<p><strong>Skills:</strong> ' + candidate.skills.join(', ') + '</p>';
                html += '<p class="score"><strong>Compatibility Score:</strong> ' + candidate.compatibility_score + '</p>';
                html += '<p class="interpretation">' + candidate.interpretation + '</p>';
                html += '<p><strong>Skill Matches:</strong> Required: ' + candidate.required_skill_matches + 
                        ', Preferred: ' + candidate.preferred_skill_matches + '</p>';
                html += '</div>';
            });
            
            document.getElementById('results').innerHTML = html;
        }
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    print("\n🚀 Starting AI Candidate Matching Test Server")
    print("📊 Data loaded:")
    print(f"   - {len(matching_service.candidates_cache)} candidates")
    print(f"   - {len(matching_service.skills_cache)} skills")
    print("\n🌐 Test the system at: http://localhost:5001/test-search")
    print("📡 API endpoint: http://localhost:5001/api/ai-search")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5001) 