from app.candidate_matcher import CandidateMatcher

def test_candidate_matcher():
    # Initialize the matcher
    matcher = CandidateMatcher()
    
    # Test job descriptions
    job_descriptions = [
        {
            "title": "Python Developer",
            "description": """
            We are looking for a Python developer with experience in web development.
            The ideal candidate should have strong skills in Django, SQL, and API development.
            Experience with React and Docker is a plus.
            """
        },
        {
            "title": "Full Stack Developer",
            "description": """
            Seeking a Full Stack Developer with expertise in both frontend and backend development.
            Must have experience with JavaScript, React, Node.js, and database management.
            Knowledge of cloud platforms (AWS/Azure) is required.
            """
        },
        {
            "title": "Data Scientist",
            "description": """
            Looking for a Data Scientist with strong Python skills and experience in machine learning.
            Must be proficient in data analysis, statistical modeling, and have experience with
            libraries like TensorFlow or PyTorch. SQL knowledge is essential.
            """
        }
    ]
    
    # Test each job description
    for job in job_descriptions:
        print(f"\n{'='*50}")
        print(f"Testing for: {job['title']}")
        print(f"{'='*50}")
        
        # Find matching candidates
        matches = matcher.find_matching_candidates(job['description'])
        
        # Print results
        print("\nTop matching candidates:")
        for i, candidate in enumerate(matches, 1):
            print(f"\n{i}. {candidate['first_name']} {candidate['last_name']}")
            print(f"   Match Score: {candidate['match_score']:.2%}")
            print("   Skills:")
            for skill in candidate['skills']:
                print(f"   - {skill['skill_name']} (Level: {skill['level']}, Duration: {skill['duration']})")

if __name__ == "__main__":
    test_candidate_matcher() 