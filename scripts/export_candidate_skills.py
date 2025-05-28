import pandas as pd
import pymysql
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '3306')),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'recruitment_crm')
}

def export_candidate_skills():
    """Export candidate skills with their levels and duration."""
    try:
        # Connect to database
        conn = pymysql.connect(**DB_CONFIG)
        print("Connected to database successfully!")
        
        # SQL query to join the tables and get the required information
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
        
        # Read the data into a pandas DataFrame
        df = pd.read_sql(query, conn)
        
        # Create output directory
        output_dir = "candidate_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        output_file = os.path.join(output_dir, "candidate_skills.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSuccessfully exported candidate skills to {output_file}")
        
        # Print sample of the data
        print("\nSample of exported data:")
        print(df.head())
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total candidates: {df['resource_id'].nunique()}")
        print(f"Total skills: {df['skill_id'].nunique()}")
        print(f"Total skill records: {len(df)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        if 'conn' in locals():
            conn.close()
            print("\nDatabase connection closed")

if __name__ == "__main__":
    export_candidate_skills() 