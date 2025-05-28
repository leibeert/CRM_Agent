from app.db import ArgoteamBase, argoteam_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_argoteam_db():
    """Initialize the argoteam database tables."""
    try:
        # Create all tables
        ArgoteamBase.metadata.create_all(bind=argoteam_engine)
        logger.info("Successfully created argoteam database tables")
    except Exception as e:
        logger.error(f"Error creating argoteam database tables: {str(e)}")
        raise

if __name__ == "__main__":
    init_argoteam_db() 