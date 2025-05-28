"""
Embedding management utilities for the matching system.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
from .config import get_config
from .cache import get_cache

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingManager:
    """Manages embedding operations for skills and text."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = get_cache()
        self.model = None
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Loaded embedding model: {self.config.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                self.model = None
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text string."""
        
        if not text or not text.strip():
            return None
        
        text_normalized = text.lower().strip()
        
        # Check cache first
        cached_embedding = self.cache.get_skill_embedding(text_normalized)
        if cached_embedding is not None:
            return np.array(cached_embedding)
        
        if self.model:
            try:
                # Compute embedding
                embedding = self.model.encode([text_normalized])[0]
                
                # Cache the embedding
                self.cache.cache_skill_embedding(text_normalized, embedding.tolist())
                
                return embedding
            except Exception as e:
                logger.error(f"Error computing embedding for '{text}': {str(e)}")
        
        return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of texts efficiently."""
        
        if not texts:
            return []
        
        # Normalize texts
        normalized_texts = [text.lower().strip() for text in texts if text and text.strip()]
        
        if not normalized_texts:
            return [None] * len(texts)
        
        # Check cache for existing embeddings
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        for i, text in enumerate(normalized_texts):
            cached_embedding = self.cache.get_skill_embedding(text)
            if cached_embedding is not None:
                embeddings.append(np.array(cached_embedding))
            else:
                embeddings.append(None)
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Compute missing embeddings in batch
        if texts_to_compute and self.model:
            try:
                computed_embeddings = self.model.encode(texts_to_compute)
                
                # Update results and cache
                for idx, embedding in zip(indices_to_compute, computed_embeddings):
                    embeddings[idx] = embedding
                    # Cache the embedding
                    self.cache.cache_skill_embedding(
                        normalized_texts[idx], 
                        embedding.tolist()
                    )
                    
            except Exception as e:
                logger.error(f"Error computing batch embeddings: {str(e)}")
        
        return embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        
        if not text1 or not text2:
            return 0.0
        
        # Exact match
        if text1.lower().strip() == text2.lower().strip():
            return 1.0
        
        # Check cache first
        cached_similarity = self.cache.get_skill_similarity(text1, text2)
        if cached_similarity is not None:
            return cached_similarity
        
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        if embedding1 is not None and embedding2 is not None:
            try:
                # Calculate cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                similarity = max(0.0, min(1.0, similarity))  # Ensure 0-1 range
                
                # Cache the result
                self.cache.cache_skill_similarity(text1, text2, similarity)
                
                return similarity
            except Exception as e:
                logger.error(f"Error calculating similarity: {str(e)}")
        
        return 0.0
    
    def find_similar_texts(self, target_text: str, 
                          candidate_texts: List[str], 
                          threshold: float = 0.7) -> List[tuple]:
        """Find texts similar to target above threshold."""
        
        if not target_text or not candidate_texts:
            return []
        
        similar_texts = []
        target_embedding = self.get_embedding(target_text)
        
        if target_embedding is not None:
            # Get embeddings for all candidates
            candidate_embeddings = self.get_embeddings_batch(candidate_texts)
            
            # Calculate similarities
            for text, embedding in zip(candidate_texts, candidate_embeddings):
                if embedding is not None:
                    try:
                        from sklearn.metrics.pairwise import cosine_similarity
                        similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                        similarity = max(0.0, min(1.0, similarity))
                        
                        if similarity >= threshold:
                            similar_texts.append((text, similarity))
                    except Exception as e:
                        logger.error(f"Error calculating similarity for '{text}': {str(e)}")
        
        # Sort by similarity (descending)
        similar_texts.sort(key=lambda x: x[1], reverse=True)
        
        return similar_texts
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        
        return {
            'model_name': self.config.embedding_model,
            'model_loaded': self.model is not None,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'cache_available': self.cache is not None
        }
    
    def clear_cache(self) -> int:
        """Clear all cached embeddings."""
        
        if self.cache:
            return self.cache.clear_pattern("*skill_embedding*")
        return 0
    
    def precompute_embeddings(self, texts: List[str]) -> int:
        """Precompute and cache embeddings for a list of texts."""
        
        if not texts:
            return 0
        
        computed_count = 0
        
        # Process in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.get_embeddings_batch(batch)
            computed_count += sum(1 for emb in embeddings if emb is not None)
        
        logger.info(f"Precomputed {computed_count} embeddings")
        return computed_count 