from sqlalchemy import or_, and_, func, desc, asc
from sqlalchemy.orm import joinedload
from typing import List, Optional, Dict, Any
from .models import SavedSearch
from db import ArgoteamSessionLocal, Resource, Skill, ResourceSkill, Experience, Study
from .schemas import SearchQuery, SearchResponse, CandidateResponse, SavedSearchCreate
from db import AuthSessionLocal

class SearchService:
    def __init__(self):
        self.argoteam_db = ArgoteamSessionLocal()
        self.auth_db = AuthSessionLocal()

    def search_candidates(self, query: SearchQuery) -> SearchResponse:
        """Search candidates with intelligent filtering."""
        try:
            # Start with base query
            base_query = self.argoteam_db.query(Resource)
            
            # INTELLIGENT KEYWORD SEARCH
            if query.keywords:
                keywords = query.keywords.lower().split()
                
                # For each keyword, search in multiple places
                keyword_filters = []
                for keyword in keywords:
                    # Create comprehensive search for each keyword
                    keyword_filter = or_(
                        # Search in personal info
                        Resource.first_name.ilike(f'%{keyword}%'),
                        Resource.last_name.ilike(f'%{keyword}%'),
                        Resource.email.ilike(f'%{keyword}%'),
                        
                        # Search in skills (most important!)
                        Resource.id.in_(
                            self.argoteam_db.query(Resource.id)
                            .join(ResourceSkill)
                            .join(Skill)
                            .filter(Skill.name.ilike(f'%{keyword}%'))
                        ),
                        
                        # Search in experience titles and companies
                        Resource.id.in_(
                            self.argoteam_db.query(Resource.id)
                            .join(Experience)
                            .filter(or_(
                                Experience.title.ilike(f'%{keyword}%'),
                                Experience.name.ilike(f'%{keyword}%'),
                                Experience.description.ilike(f'%{keyword}%')
                            ))
                        ),
                        
                        # Search in education/studies
                        Resource.id.in_(
                            self.argoteam_db.query(Resource.id)
                            .join(Study)
                            .filter(Study.degree_name.ilike(f'%{keyword}%'))
                        )
                    )
                    keyword_filters.append(keyword_filter)
                
                # Combine all keyword filters (candidate must match ALL keywords)
                base_query = base_query.filter(and_(*keyword_filters))

            # SPECIFIC SKILLS FILTER (for structured skill search)
            if query.skills:
                for skill_filter in query.skills:
                    skill_query = self.argoteam_db.query(Resource.id).join(
                        ResourceSkill
                    ).join(
                        Skill
                    ).filter(
                        Skill.name.ilike(f'%{skill_filter.skill_name}%')
                    )
                    
                    if skill_filter.min_level:
                        skill_query = skill_query.filter(
                            ResourceSkill.level >= skill_filter.min_level
                        )
                    
                    if skill_filter.min_duration:
                        skill_query = skill_query.filter(
                            ResourceSkill.duration >= skill_filter.min_duration
                        )
                    
                    base_query = base_query.filter(Resource.id.in_(skill_query))

            # EXPERIENCE FILTER (with duration support)
            if query.experience:
                exp_query = self.argoteam_db.query(Resource.id).join(Experience)
                
                if query.experience.title:
                    exp_query = exp_query.filter(
                        Experience.title.ilike(f'%{query.experience.title}%')
                    )
                
                if query.experience.company:
                    exp_query = exp_query.filter(
                        Experience.name.ilike(f'%{query.experience.company}%')
                    )
                
                # NEW: Experience duration filter
                if query.experience.min_years:
                    # Calculate experience duration from start/end dates
                    exp_query = exp_query.filter(
                        func.datediff(
                            func.coalesce(Experience.end_date, func.now()),
                            Experience.start_date
                        ) >= (query.experience.min_years * 365)
                    )
                
                base_query = base_query.filter(Resource.id.in_(exp_query))

            # EDUCATION FILTER
            if query.education:
                edu_query = self.argoteam_db.query(Resource.id).join(Study)
                
                if query.education.degree_type:
                    edu_query = edu_query.filter(
                        Study.degree_type_id == query.education.degree_type
                    )
                
                if query.education.field_of_study:
                    edu_query = edu_query.filter(
                        Study.degree_name.ilike(f'%{query.education.field_of_study}%')
                    )
                
                if query.education.school:
                    edu_query = edu_query.filter(
                        Study.school_id == query.education.school
                    )
                
                base_query = base_query.filter(Resource.id.in_(edu_query))

            # Calculate total count before pagination
            total = base_query.count()

            # Apply sorting with better logic
            if query.sort_by == 'match_score':
                # For match score, we'll sort after calculating scores
                pass
            elif query.sort_by == 'experience':
                # Sort by total experience duration
                base_query = base_query.outerjoin(Experience).group_by(Resource.id).order_by(
                    func.count(Experience.id).desc() if query.sort_order == 'desc' 
                    else func.count(Experience.id).asc()
                )
            elif query.sort_by == 'skills':
                # Sort by number of skills
                base_query = base_query.outerjoin(ResourceSkill).group_by(Resource.id).order_by(
                    func.count(ResourceSkill.id).desc() if query.sort_order == 'desc' 
                    else func.count(ResourceSkill.id).asc()
                )
            else:
                # Default to creation date
                base_query = base_query.order_by(
                    Resource.created_at.desc() if query.sort_order == 'desc' 
                    else Resource.created_at.asc()
                )

            # Apply pagination
            base_query = base_query.offset((query.page - 1) * query.page_size).limit(query.page_size)

            # Eager load relationships
            base_query = base_query.options(
                joinedload(Resource.skills).joinedload(ResourceSkill.skill),
                joinedload(Resource.experiences),
                joinedload(Resource.studies)
            )

            # Execute query
            candidates = base_query.all()

            # Calculate intelligent match scores
            candidate_responses = []
            for candidate in candidates:
                match_score = self._calculate_match_score(candidate, query)
                
                candidate_responses.append(CandidateResponse(
                    id=candidate.id,
                    first_name=candidate.first_name,
                    last_name=candidate.last_name,
                    email=candidate.email,
                    phone_number=candidate.phone_number,
                    match_score=match_score,
                    skills=[{
                        'name': skill.skill.name,
                        'level': skill.level,
                        'duration': skill.duration
                    } for skill in candidate.skills],
                    experience=[{
                        'title': exp.title,
                        'company': exp.name,
                        'description': exp.description,
                        'start_date': exp.start_date.isoformat() if exp.start_date else None,
                        'end_date': exp.end_date.isoformat() if exp.end_date else None
                    } for exp in candidate.experiences],
                    education=[{
                        'degree_type_id': edu.degree_type_id,
                        'degree_name': edu.degree_name,
                        'school_id': edu.school_id,
                        'start_date': edu.start_date.isoformat() if edu.start_date else None,
                        'end_date': edu.end_date.isoformat() if edu.end_date else None
                    } for edu in candidate.studies]
                ))

            # Sort by match score if requested
            if query.sort_by == 'match_score':
                candidate_responses.sort(
                    key=lambda x: x.match_score, 
                    reverse=(query.sort_order == 'desc')
                )

            return SearchResponse(
                candidates=candidate_responses,
                total=total,
                page=query.page,
                page_size=query.page_size,
                total_pages=(total + query.page_size - 1) // query.page_size
            )

        except Exception as e:
            raise Exception(f"Error searching candidates: {str(e)}")

    def _calculate_match_score(self, candidate, query: SearchQuery) -> float:
        """Calculate intelligent match score based on multiple factors."""
        score = 0.0
        max_score = 0.0
        
        # Keyword matching score (40% weight)
        if query.keywords:
            max_score += 40
            keywords = query.keywords.lower().split()
            keyword_matches = 0
            
            for keyword in keywords:
                # Check skills
                skill_match = any(
                    keyword in skill.skill.name.lower() 
                    for skill in candidate.skills
                )
                
                # Check experience
                exp_match = any(
                    keyword in (exp.title or '').lower() or 
                    keyword in (exp.name or '').lower() or
                    keyword in (exp.description or '').lower()
                    for exp in candidate.experiences
                )
                
                # Check personal info
                personal_match = (
                    keyword in (candidate.first_name or '').lower() or
                    keyword in (candidate.last_name or '').lower() or
                    keyword in (candidate.email or '').lower()
                )
                
                # Check education
                edu_match = any(
                    keyword in (edu.degree_name or '').lower()
                    for edu in candidate.studies
                )
                
                if skill_match or exp_match or personal_match or edu_match:
                    keyword_matches += 1
            
            score += (keyword_matches / len(keywords)) * 40

        # Specific skills matching (30% weight)
        if query.skills:
            max_score += 30
            skill_matches = 0
            
            for skill_filter in query.skills:
                candidate_skill = next(
                    (skill for skill in candidate.skills 
                     if skill_filter.skill_name.lower() in skill.skill.name.lower()),
                    None
                )
                
                if candidate_skill:
                    skill_matches += 1
                    # Bonus for level matching
                    if (skill_filter.min_level and 
                        candidate_skill.level >= skill_filter.min_level):
                        skill_matches += 0.5
            
            score += (skill_matches / len(query.skills)) * 30

        # Experience matching (20% weight)
        if query.experience:
            max_score += 20
            exp_score = 0
            
            relevant_experiences = [
                exp for exp in candidate.experiences
                if (not query.experience.title or 
                    query.experience.title.lower() in (exp.title or '').lower()) and
                   (not query.experience.company or 
                    query.experience.company.lower() in (exp.name or '').lower())
            ]
            
            if relevant_experiences:
                exp_score = 20
            
            score += exp_score

        # Education matching (10% weight)
        if query.education:
            max_score += 10
            edu_score = 0
            
            relevant_education = [
                edu for edu in candidate.studies
                if (not query.education.field_of_study or 
                    query.education.field_of_study.lower() in (edu.degree_name or '').lower())
            ]
            
            if relevant_education:
                edu_score = 10
            
            score += edu_score

        # Return percentage score
        return (score / max_score * 100) if max_score > 0 else 0.0

    def save_search(self, user_id: int, search: SavedSearchCreate) -> SavedSearch:
        """Save a search query for future use."""
        try:
            saved_search = SavedSearch(
                user_id=user_id,
                name=search.name,
                description=search.description,
                filters=search.filters.dict(),
                sort_by=search.sort_by,
                sort_order=search.sort_order
            )
            self.auth_db.add(saved_search)
            self.auth_db.commit()
            self.auth_db.refresh(saved_search)
            return saved_search
        except Exception as e:
            self.auth_db.rollback()
            raise Exception(f"Error saving search: {str(e)}")

    def get_saved_searches(self, user_id: int) -> List[SavedSearch]:
        """Get all saved searches for a user."""
        try:
            return self.auth_db.query(SavedSearch).filter(SavedSearch.user_id == user_id).all()
        except Exception as e:
            raise Exception(f"Error getting saved searches: {str(e)}")

    def delete_saved_search(self, search_id: int, user_id: int) -> bool:
        """Delete a saved search."""
        try:
            result = self.auth_db.query(SavedSearch).filter(
                SavedSearch.id == search_id,
                SavedSearch.user_id == user_id
            ).delete()
            self.auth_db.commit()
            return result > 0
        except Exception as e:
            self.auth_db.rollback()
            raise Exception(f"Error deleting saved search: {str(e)}")

    def __del__(self):
        """Cleanup database connection."""
        self.argoteam_db.close()
        self.auth_db.close() 