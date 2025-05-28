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
        """Search candidates with advanced filtering and sorting."""
        try:
            # Start with base query
            base_query = self.argoteam_db.query(Resource)

            # Apply keyword search
            if query.keywords:
                keywords = query.keywords.lower().split()
                keyword_filters = []
                for keyword in keywords:
                    keyword_filters.append(
                        or_(
                            Resource.first_name.ilike(f'%{keyword}%'),
                            Resource.last_name.ilike(f'%{keyword}%'),
                            Resource.email.ilike(f'%{keyword}%')
                        )
                    )
                base_query = base_query.filter(or_(*keyword_filters))

            # Apply skills filter
            if query.skills:
                for skill_filter in query.skills:
                    skill_query = self.argoteam_db.query(Resource).join(
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
                            ResourceSkill.duration_months >= skill_filter.min_duration
                        )
                    
                    base_query = base_query.filter(Resource.id.in_(skill_query.subquery()))

            # Apply experience filter
            if query.experience:
                exp_query = self.argoteam_db.query(Resource).join(Experience)
                
                if query.experience.title:
                    exp_query = exp_query.filter(
                        Experience.title.ilike(f'%{query.experience.title}%')
                    )
                
                if query.experience.company:
                    exp_query = exp_query.filter(
                        Experience.company.ilike(f'%{query.experience.company}%')
                    )
                
                if query.experience.min_years:
                    exp_query = exp_query.filter(
                        Experience.years >= query.experience.min_years
                    )
                
                base_query = base_query.filter(Resource.id.in_(exp_query.subquery()))

            # Apply education filter
            if query.education:
                edu_query = self.argoteam_db.query(Resource).join(Study)
                
                if query.education.degree_type:
                    edu_query = edu_query.filter(
                        Study.degree_type == query.education.degree_type
                    )
                
                if query.education.field_of_study:
                    edu_query = edu_query.filter(
                        Study.field_of_study.ilike(f'%{query.education.field_of_study}%')
                    )
                
                if query.education.school:
                    edu_query = edu_query.filter(
                        Study.school.ilike(f'%{query.education.school}%')
                    )
                
                base_query = base_query.filter(Resource.id.in_(edu_query.subquery()))

            # Calculate total count
            total = base_query.count()

            # Apply sorting
            if query.sort_by == 'match_score':
                base_query = base_query.order_by(
                    func.random() if query.sort_order == 'desc' else func.random().desc()
                )
            elif query.sort_by == 'experience':
                base_query = base_query.join(Experience).order_by(
                    Experience.years.desc() if query.sort_order == 'desc' else Experience.years.asc()
                )
            elif query.sort_by == 'education':
                base_query = base_query.join(Study).order_by(
                    Study.graduation_year.desc() if query.sort_order == 'desc' else Study.graduation_year.asc()
                )
            elif query.sort_by == 'created_at':
                base_query = base_query.order_by(
                    Resource.created_at.desc() if query.sort_order == 'desc' else Resource.created_at.asc()
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

            # Format response
            candidate_responses = []
            for candidate in candidates:
                # Calculate match score (simplified version)
                match_score = 0.0
                if query.skills:
                    skill_matches = len([
                        skill for skill in candidate.skills
                        if any(
                            skill.name.lower() in sf.skill_name.lower()
                            for sf in query.skills
                        )
                    ])
                    match_score += (skill_matches / len(query.skills)) * 40

                if query.experience:
                    exp_matches = len([
                        exp for exp in candidate.experience
                        if (
                            (not query.experience.title or query.experience.title.lower() in exp.title.lower()) and
                            (not query.experience.company or query.experience.company.lower() in exp.company.lower()) and
                            (not query.experience.min_years or exp.years >= query.experience.min_years)
                        )
                    ])
                    match_score += (exp_matches / max(1, len(candidate.experience))) * 30

                if query.education:
                    edu_matches = len([
                        edu for edu in candidate.education
                        if (
                            (not query.education.degree_type or edu.degree_type == query.education.degree_type) and
                            (not query.education.field_of_study or query.education.field_of_study.lower() in edu.field_of_study.lower()) and
                            (not query.education.school or query.education.school.lower() in edu.school.lower())
                        )
                    ])
                    match_score += (edu_matches / max(1, len(candidate.education))) * 30

                candidate_responses.append(CandidateResponse(
                    id=candidate.id,
                    first_name=candidate.first_name,
                    last_name=candidate.last_name,
                    email=candidate.email,
                    phone_number=candidate.phone_number,
                    match_score=match_score,
                    skills=[{
                        'name': skill.name,
                        'level': skill.level,
                        'duration': skill.duration_months
                    } for skill in candidate.skills],
                    experience=[{
                        'title': exp.title,
                        'company': exp.company,
                        'years': exp.years,
                        'start_date': exp.start_date.isoformat(),
                        'end_date': exp.end_date.isoformat() if exp.end_date else None
                    } for exp in candidate.experience],
                    education=[{
                        'degree_type': edu.degree_type,
                        'field_of_study': edu.field_of_study,
                        'school': edu.school,
                        'graduation_year': edu.graduation_year
                    } for edu in candidate.education]
                ))

            return SearchResponse(
                candidates=candidate_responses,
                total=total,
                page=query.page,
                page_size=query.page_size,
                total_pages=(total + query.page_size - 1) // query.page_size
            )

        except Exception as e:
            raise Exception(f"Error searching candidates: {str(e)}")

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