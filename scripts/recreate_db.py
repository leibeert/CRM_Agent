from app.db import Base, engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recreate_tables():
    try:
        # Drop all tables
        logger.info("Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        
        # Create all tables
        logger.info("Creating all tables...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables recreated successfully!")
    except Exception as e:
        logger.error(f"Error recreating database tables: {str(e)}")
        raise

if __name__ == "__main__":
    recreate_tables() 