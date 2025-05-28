"""
Semantic skill matching using embeddings and similarity calculations.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from ..utils.config import get_config
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available, using fallback similarity")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticSkillMatcher:
    """Advanced semantic skill matching using embeddings and predefined relationships."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        self.model = None
        self.skill_hierarchy = self._load_skill_hierarchy()
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                self.model = None
        
    def calculate_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between two skills."""
        
        # Normalize skill names
        skill1_norm = skill1.lower().strip()
        skill2_norm = skill2.lower().strip()
        
        # Exact match
        if skill1_norm == skill2_norm:
            return 1.0
        
        # Check cache first
        cached_similarity = self.cache.get_skill_similarity(skill1_norm, skill2_norm)
        if cached_similarity is not None:
            return cached_similarity
        
        similarity = 0.0
        
        # Check predefined skill hierarchy
        hierarchy_similarity = self._check_skill_hierarchy(skill1_norm, skill2_norm)
        if hierarchy_similarity > 0:
            similarity = hierarchy_similarity
        elif self.model:
            # Use embedding-based similarity
            similarity = self._calculate_embedding_similarity(skill1_norm, skill2_norm)
        else:
            # Fallback to string-based similarity
            similarity = self._calculate_string_similarity(skill1_norm, skill2_norm)
        
        # Cache the result
        self.cache.cache_skill_similarity(skill1_norm, skill2_norm, similarity)
        
        return similarity
    
    def find_related_skills(self, target_skill: str, skill_pool: List[str], 
                          threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Find all related skills above threshold from a pool of skills."""
        
        threshold = threshold or self.config.similarity_threshold
        related_skills = []
        
        for skill in skill_pool:
            if skill.lower() != target_skill.lower():
                similarity = self.calculate_similarity(target_skill, skill)
                if similarity >= threshold:
                    related_skills.append((skill, similarity))
        
        # Sort by similarity score (descending)
        related_skills.sort(key=lambda x: x[1], reverse=True)
        
        return related_skills
    
    def get_skill_category(self, skill: str) -> str:
        """Determine the category of a skill."""
        skill_lower = skill.lower()
        
        # Programming languages
        programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 
            'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r'
        }
        
        # Frameworks and libraries
        frameworks = {
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
            'laravel', 'rails', 'tensorflow', 'pytorch', 'scikit-learn'
        }
        
        # Databases
        databases = {
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sql server', 'sqlite', 'cassandra', 'dynamodb'
        }
        
        # Cloud platforms
        cloud_platforms = {
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
            'terraform', 'ansible', 'jenkins'
        }
        
        # Soft skills
        soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'agile', 'scrum'
        }
        
        if any(lang in skill_lower for lang in programming_languages):
            return 'programming_language'
        elif any(fw in skill_lower for fw in frameworks):
            return 'framework'
        elif any(db in skill_lower for db in databases):
            return 'database'
        elif any(cloud in skill_lower for cloud in cloud_platforms):
            return 'cloud_platform'
        elif any(soft in skill_lower for soft in soft_skills):
            return 'soft_skill'
        else:
            return 'other'
    
    def _load_skill_hierarchy(self) -> Dict[str, Dict[str, float]]:
        """Load predefined skill relationships and hierarchies."""
        return {
            # Programming Languages and Frameworks
            'python': {
                'django': 0.9,
                'flask': 0.85,
                'fastapi': 0.8,
                'pandas': 0.8,
                'numpy': 0.8,
                'scikit-learn': 0.85,
                'tensorflow': 0.8,
                'pytorch': 0.8,
                'programming': 0.7
            },
            'javascript': {
                'react': 0.9,
                'vue': 0.85,
                'angular': 0.85,
                'node.js': 0.8,
                'nodejs': 0.8,
                'typescript': 0.9,
                'express': 0.8,
                'jquery': 0.7,
                'programming': 0.7
            },
            'java': {
                'spring': 0.9,
                'hibernate': 0.8,
                'maven': 0.7,
                'gradle': 0.7,
                'programming': 0.7,
                'object-oriented programming': 0.8,
                'oop': 0.8
            },
            
            # Machine Learning and AI
            'machine learning': {
                'deep learning': 0.9,
                'neural networks': 0.85,
                'tensorflow': 0.8,
                'pytorch': 0.8,
                'scikit-learn': 0.75,
                'data science': 0.8,
                'artificial intelligence': 0.9,
                'ai': 0.9
            },
            'data science': {
                'machine learning': 0.8,
                'statistics': 0.8,
                'python': 0.7,
                'r': 0.7,
                'pandas': 0.8,
                'numpy': 0.7,
                'matplotlib': 0.6,
                'seaborn': 0.6
            },
            
            # Web Development
            'web development': {
                'html': 0.8,
                'css': 0.8,
                'javascript': 0.9,
                'react': 0.8,
                'angular': 0.8,
                'vue': 0.8,
                'frontend': 0.9,
                'backend': 0.8
            },
            'frontend': {
                'html': 0.9,
                'css': 0.9,
                'javascript': 0.9,
                'react': 0.8,
                'angular': 0.8,
                'vue': 0.8,
                'ui/ux': 0.7
            },
            'backend': {
                'api': 0.8,
                'database': 0.8,
                'server': 0.8,
                'microservices': 0.7,
                'rest': 0.8,
                'graphql': 0.7
            },
            
            # Cloud and DevOps
            'aws': {
                'cloud': 0.9,
                'ec2': 0.8,
                's3': 0.8,
                'lambda': 0.8,
                'rds': 0.7,
                'cloudformation': 0.7,
                'devops': 0.7
            },
            'devops': {
                'docker': 0.9,
                'kubernetes': 0.8,
                'jenkins': 0.8,
                'terraform': 0.8,
                'ansible': 0.7,
                'ci/cd': 0.9,
                'aws': 0.7,
                'azure': 0.7,
                'gcp': 0.7
            },
            
            # Databases
            'sql': {
                'mysql': 0.8,
                'postgresql': 0.8,
                'oracle': 0.7,
                'sql server': 0.8,
                'database': 0.9
            },
            'nosql': {
                'mongodb': 0.8,
                'cassandra': 0.7,
                'dynamodb': 0.7,
                'redis': 0.7,
                'elasticsearch': 0.7,
                'database': 0.8
            }
        }
    
    def _check_skill_hierarchy(self, skill1: str, skill2: str) -> float:
        """Check predefined skill hierarchy for relationship."""
        
        # Direct relationship
        if skill1 in self.skill_hierarchy:
            if skill2 in self.skill_hierarchy[skill1]:
                return self.skill_hierarchy[skill1][skill2]
        
        # Reverse relationship
        if skill2 in self.skill_hierarchy:
            if skill1 in self.skill_hierarchy[skill2]:
                return self.skill_hierarchy[skill2][skill1]
        
        # Check for common parent skills
        skill1_parents = set()
        skill2_parents = set()
        
        for parent, children in self.skill_hierarchy.items():
            if skill1 in children:
                skill1_parents.add(parent)
            if skill2 in children:
                skill2_parents.add(parent)
        
        # If they share a parent, they're related
        common_parents = skill1_parents.intersection(skill2_parents)
        if common_parents:
            # Return average similarity through common parents
            similarities = []
            for parent in common_parents:
                sim1 = self.skill_hierarchy[parent].get(skill1, 0)
                sim2 = self.skill_hierarchy[parent].get(skill2, 0)
                similarities.append(min(sim1, sim2) * 0.8)  # Reduce for indirect relationship
            return max(similarities) if similarities else 0
        
        return 0
    
    def _calculate_embedding_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity using sentence embeddings."""
        try:
            # Get or compute embeddings
            embedding1 = self._get_skill_embedding(skill1)
            embedding2 = self._get_skill_embedding(skill2)
            
            if embedding1 is not None and embedding2 is not None:
                # Calculate cosine similarity
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return max(0.0, min(1.0, similarity))  # Ensure 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {str(e)}")
        
        return 0.0
    
    def _get_skill_embedding(self, skill: str) -> Optional[np.ndarray]:
        """Get or compute embedding for a skill."""
        
        # Check cache first
        cached_embedding = self.cache.get_skill_embedding(skill)
        if cached_embedding is not None:
            return np.array(cached_embedding)
        
        if self.model:
            try:
                # Compute embedding
                embedding = self.model.encode([skill])[0]
                
                # Cache the embedding
                self.cache.cache_skill_embedding(skill, embedding.tolist())
                
                return embedding
            except Exception as e:
                logger.error(f"Error computing embedding for '{skill}': {str(e)}")
        
        return None
    
    def _calculate_string_similarity(self, skill1: str, skill2: str) -> float:
        """Fallback string-based similarity calculation."""
        
        # Simple character-based similarity
        def levenshtein_similarity(s1: str, s2: str) -> float:
            """Calculate normalized Levenshtein similarity."""
            if len(s1) == 0 or len(s2) == 0:
                return 0.0
            
            # Create matrix
            matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
            
            # Initialize first row and column
            for i in range(len(s1) + 1):
                matrix[i][0] = i
            for j in range(len(s2) + 1):
                matrix[0][j] = j
            
            # Fill matrix
            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    if s1[i-1] == s2[j-1]:
                        matrix[i][j] = matrix[i-1][j-1]
                    else:
                        matrix[i][j] = min(
                            matrix[i-1][j] + 1,      # deletion
                            matrix[i][j-1] + 1,      # insertion
                            matrix[i-1][j-1] + 1     # substitution
                        )
            
            # Calculate similarity
            max_len = max(len(s1), len(s2))
            distance = matrix[len(s1)][len(s2)]
            return 1.0 - (distance / max_len)
        
        # Check for substring relationships
        if skill1 in skill2 or skill2 in skill1:
            return 0.6
        
        # Check for common words
        words1 = set(skill1.split())
        words2 = set(skill2.split())
        common_words = words1.intersection(words2)
        
        if common_words:
            jaccard = len(common_words) / len(words1.union(words2))
            return jaccard * 0.8
        
        # Levenshtein similarity as last resort
        return levenshtein_similarity(skill1, skill2) * 0.5
    
    def batch_calculate_similarities(self, target_skill: str, 
                                   skill_list: List[str]) -> List[Tuple[str, float]]:
        """Calculate similarities for a batch of skills efficiently."""
        
        results = []
        
        # Group by whether we need embeddings or not
        hierarchy_skills = []
        embedding_skills = []
        
        for skill in skill_list:
            if skill.lower() == target_skill.lower():
                results.append((skill, 1.0))
            elif self._check_skill_hierarchy(target_skill.lower(), skill.lower()) > 0:
                hierarchy_skills.append(skill)
            else:
                embedding_skills.append(skill)
        
        # Process hierarchy-based similarities
        for skill in hierarchy_skills:
            similarity = self._check_skill_hierarchy(target_skill.lower(), skill.lower())
            results.append((skill, similarity))
        
        # Process embedding-based similarities in batch if model available
        if embedding_skills and self.model:
            try:
                target_embedding = self._get_skill_embedding(target_skill.lower())
                if target_embedding is not None:
                    skill_embeddings = []
                    valid_skills = []
                    
                    for skill in embedding_skills:
                        embedding = self._get_skill_embedding(skill.lower())
                        if embedding is not None:
                            skill_embeddings.append(embedding)
                            valid_skills.append(skill)
                    
                    if skill_embeddings:
                        # Batch cosine similarity calculation
                        similarities = cosine_similarity([target_embedding], skill_embeddings)[0]
                        for skill, similarity in zip(valid_skills, similarities):
                            results.append((skill, max(0.0, min(1.0, similarity))))
                        
                        # Cache all computed similarities
                        for skill, similarity in zip(valid_skills, similarities):
                            self.cache.cache_skill_similarity(target_skill.lower(), skill.lower(), similarity)
            
            except Exception as e:
                logger.error(f"Error in batch similarity calculation: {str(e)}")
                # Fallback to individual calculations
                for skill in embedding_skills:
                    similarity = self.calculate_similarity(target_skill, skill)
                    results.append((skill, similarity))
        else:
            # Fallback for skills without embeddings
            for skill in embedding_skills:
                similarity = self._calculate_string_similarity(target_skill.lower(), skill.lower())
                results.append((skill, similarity))
        
        return results 