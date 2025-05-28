from sqlalchemy import create_engine, String
from sqlalchemy.sql import text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database credentials from environment variables
ARGOTEAM_DB_HOST = os.getenv('ARGOTEAM_DB_HOST', 'localhost')
ARGOTEAM_DB_PORT = os.getenv('ARGOTEAM_DB_PORT', '3306')
ARGOTEAM_DB_NAME = os.getenv('ARGOTEAM_DB_NAME', 'argoteam')
ARGOTEAM_DB_USER = os.getenv('ARGOTEAM_DB_USER', 'root')
ARGOTEAM_DB_PASSWORD = os.getenv('ARGOTEAM_DB_PASSWORD', 'admin')

# Create database URL
DATABASE_URL = f"mysql+pymysql://{ARGOTEAM_DB_USER}:{ARGOTEAM_DB_PASSWORD}@{ARGOTEAM_DB_HOST}:{ARGOTEAM_DB_PORT}/{ARGOTEAM_DB_NAME}"

def add_phone_number_column():
    """Add phone_number column to resources table."""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Add phone_number column
        with engine.connect() as connection:
            # Check if column exists
            result = connection.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.columns 
                WHERE table_schema = :db_name
                AND table_name = 'resources'
                AND column_name = 'phone_number'
            """), {"db_name": ARGOTEAM_DB_NAME})
            
            if result.scalar() == 0:
                # Add column if it doesn't exist
                connection.execute(text("""
                    ALTER TABLE resources
                    ADD COLUMN phone_number VARCHAR(20) NULL
                """))
                print("Successfully added phone_number column to resources table")
            else:
                print("phone_number column already exists in resources table")
            
            connection.commit()
            
    except Exception as e:
        print(f"Error adding phone_number column: {str(e)}")
        raise

if __name__ == "__main__":
    add_phone_number_column() 