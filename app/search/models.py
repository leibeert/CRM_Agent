from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db import AuthBase

class SavedSearch(AuthBase):
    """Model for storing saved search filters."""
    __tablename__ = 'saved_searches'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(String(1000))
    filters = Column(JSON, nullable=False)
    sort_by = Column(String(50), nullable=False, default='match_score')
    sort_order = Column(String(10), nullable=False, default='desc')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="saved_searches") 