from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection details
AUTH_DB_HOST = os.getenv('AUTH_DB_HOST', 'localhost')
AUTH_DB_PORT = os.getenv('AUTH_DB_PORT', '3306')
AUTH_DB_NAME = os.getenv('AUTH_DB_NAME', 'auth_db')
AUTH_DB_USER = os.getenv('AUTH_DB_USER', 'root')
AUTH_DB_PASSWORD = os.getenv('AUTH_DB_PASSWORD', 'admin')

# Create database URL
DATABASE_URL = f"mysql+pymysql://{AUTH_DB_USER}:{AUTH_DB_PASSWORD}@{AUTH_DB_HOST}:{AUTH_DB_PORT}/{AUTH_DB_NAME}"

def add_updated_at_column():
    """Add updated_at column to users table if it doesn't exist."""
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as connection:
            # Check if column exists
            result = connection.execute(text("""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = :db_name
                AND table_name = 'users'
                AND column_name = 'updated_at'
            """), {"db_name": AUTH_DB_NAME})
            
            if result.scalar() == 0:
                # Add the column if it doesn't exist
                connection.execute(text("""
                    ALTER TABLE users
                    ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                """))
                print("Successfully added updated_at column to users table")
            else:
                print("updated_at column already exists in users table")
                
    except Exception as e:
        print(f"Error adding updated_at column: {str(e)}")
        raise
    finally:
        engine.dispose()

if __name__ == "__main__":
    add_updated_at_column() 