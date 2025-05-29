from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, DateTime, Table, Boolean, JSON
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.sql import func

# Load environment variables
load_dotenv()

# Create SQLAlchemy engine for MySQL using AUTH_DB credentials
AUTH_DB_HOST = os.getenv('AUTH_DB_HOST', 'localhost')
AUTH_DB_PORT = os.getenv('AUTH_DB_PORT', '3306')
AUTH_DB_NAME = os.getenv('AUTH_DB_NAME', 'auth_db')
AUTH_DB_USER = os.getenv('AUTH_DB_USER', 'root')
AUTH_DB_PASSWORD = os.getenv('AUTH_DB_PASSWORD', 'admin')

# Create SQLAlchemy engine for MySQL using ARGOTEAM_DB credentials
ARGOTEAM_DB_HOST = os.getenv('ARGOTEAM_DB_HOST', 'localhost')
ARGOTEAM_DB_PORT = os.getenv('ARGOTEAM_DB_PORT', '3306')
ARGOTEAM_DB_NAME = os.getenv('ARGOTEAM_DB_NAME', 'argoteam')
ARGOTEAM_DB_USER = os.getenv('ARGOTEAM_DB_USER', 'root')
ARGOTEAM_DB_PASSWORD = os.getenv('ARGOTEAM_DB_PASSWORD', 'admin')

# Database URLs
AUTH_DATABASE_URL = f"mysql+pymysql://{AUTH_DB_USER}:{AUTH_DB_PASSWORD}@{AUTH_DB_HOST}:{AUTH_DB_PORT}/{AUTH_DB_NAME}"
ARGOTEAM_DATABASE_URL = f"mysql+pymysql://{ARGOTEAM_DB_USER}:{ARGOTEAM_DB_PASSWORD}@{ARGOTEAM_DB_HOST}:{ARGOTEAM_DB_PORT}/{ARGOTEAM_DB_NAME}"

# Create engines
auth_engine = create_engine(AUTH_DATABASE_URL)
argoteam_engine = create_engine(ARGOTEAM_DATABASE_URL)

# Create session factories
AuthSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=auth_engine)
ArgoteamSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=argoteam_engine)

# Create base classes for models
AuthBase = declarative_base()
ArgoteamBase = declarative_base()

# Association table for conversation participants
conversation_participants = Table('conversation_participants', AuthBase.metadata,
    Column('conversation_id', Integer, ForeignKey('conversations.id')),
    Column('user_id', Integer, ForeignKey('users.id'))
)

# Define models for candidate-related tables (using ArgoteamBase)
class Resource(ArgoteamBase):
    __tablename__ = 'resources'

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    email = Column(String(255), nullable=True)
    phone_number = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    skills = relationship("ResourceSkill", back_populates="resource")
    experiences = relationship("Experience", back_populates="resource")
    studies = relationship("Study", back_populates="resource")

class Skill(ArgoteamBase):
    __tablename__ = 'skills'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    category_name = Column(Integer, nullable=True)
    created_by = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resource_skills = relationship("ResourceSkill", back_populates="skill")

class ResourceSkill(ArgoteamBase):
    __tablename__ = 'resource_skills'

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(Integer, ForeignKey('resources.id'))
    skill_id = Column(Integer, ForeignKey('skills.id'))
    level = Column(Integer, nullable=True)  # Changed from String to Integer to match CSV
    is_level_verified = Column(Boolean, default=False, nullable=True)  # Added missing field
    is_verified = Column(Boolean, default=False, nullable=True)  # Added missing field
    duration = Column(Integer, nullable=True)  # Duration in months
    created_by = Column(Integer, nullable=True)  # Added missing field
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resource = relationship("Resource", back_populates="skills")
    skill = relationship("Skill", back_populates="resource_skills")

class Experience(ArgoteamBase):
    __tablename__ = 'experiences'

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(Integer, ForeignKey('resources.id'))
    name = Column(String(255))  # Changed from company to name to match actual DB
    description = Column(Text, nullable=True)
    title = Column(String(255))
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    experience_type = Column(Integer, nullable=True)  # Added missing fields from CSV
    job_type = Column(Integer, nullable=True)
    job_location = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resource = relationship("Resource", back_populates="experiences")

class DegreeType(ArgoteamBase):
    __tablename__ = 'degree_types'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    studies = relationship("Study", back_populates="degree_type")

class School(ArgoteamBase):
    __tablename__ = 'schools'

    id = Column(Integer, primary_key=True, index=True)
    school_name = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    studies = relationship("Study", back_populates="school")

class Study(ArgoteamBase):
    __tablename__ = 'studies'

    id = Column(Integer, primary_key=True, index=True)
    resource_id = Column(Integer, ForeignKey('resources.id'))
    degree_type_id = Column(Integer, ForeignKey('degree_types.id'), nullable=True)
    school_id = Column(Integer, ForeignKey('schools.id'), nullable=True)
    degree_name = Column(String(255), nullable=True)  # Added missing field from CSV
    description = Column(Text, nullable=True)  # Added missing field from CSV
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    study_type = Column(Integer, nullable=True)  # Added missing field from CSV
    is_verified = Column(Boolean, default=False, nullable=True)  # Added missing field from CSV
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resource = relationship("Resource", back_populates="studies")
    degree_type = relationship("DegreeType", back_populates="studies")
    school = relationship("School", back_populates="studies")

# Auth-related models (using AuthBase)
class User(AuthBase):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255))
    email = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    messages = relationship("Message", back_populates="sender")
    conversations = relationship("Conversation", 
                               secondary=conversation_participants,
                               back_populates="participants")
    saved_searches = relationship('SavedSearch', back_populates='user')

class Conversation(AuthBase):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    participants = relationship("User", 
                              secondary=conversation_participants,
                              back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(AuthBase):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    sender_id = Column(Integer, ForeignKey('users.id'))
    content = Column(Text)  # Use Text for potentially long message content
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_read = Column(Boolean, default=False)
    is_agent = Column(Boolean, default=False)
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    read_at = Column(DateTime)
    edit_history = Column(JSON)  # Store edit history as JSON

    # Relationships
    conversation = relationship('Conversation', back_populates='messages')
    sender = relationship('User', back_populates='messages')
    reactions = relationship('MessageReaction', back_populates='message', cascade='all, delete-orphan')
    attachments = relationship('MessageAttachment', back_populates='message', cascade='all, delete-orphan')

class MessageReaction(AuthBase):
    __tablename__ = 'message_reactions'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('messages.id'))
    user_id = Column(Integer, ForeignKey('users.id'))
    reaction = Column(String(10))  # Store emoji or reaction type with max length of 10 characters
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship('Message', back_populates='reactions')
    user = relationship('User')

class MessageAttachment(AuthBase):
    __tablename__ = 'message_attachments'

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey('messages.id'))
    file_name = Column(String(255))  # Maximum file name length
    file_type = Column(String(100))  # MIME type length
    file_size = Column(Integer)
    file_path = Column(String(500))  # Path length
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship('Message', back_populates='attachments')

def get_auth_db():
    """Get database session for auth database."""
    db = AuthSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_argoteam_db():
    """Get database session for argoteam database."""
    db = ArgoteamSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_langchain_db():
    """Get LangChain SQLDatabase instance for argoteam database."""
    return SQLDatabase.from_uri(
        ARGOTEAM_DATABASE_URL,
        include_tables=[
            'resources',           # Main candidates table
            'skills',             # Skills table
            'resource_skills',    # Resource-Skills relationship
            'experiences',        # Experience records
            'studies',           # Education records
            'degree_types',      # Types of degrees
            'schools'            # Educational institutions
        ],
        sample_rows_in_table_info=3
    ) 