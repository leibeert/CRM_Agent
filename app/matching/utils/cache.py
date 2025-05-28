"""
Caching utilities for the matching system to improve performance.
"""

import redis
import pickle
import json
import hashlib
import logging
from typing import Any, Optional, Union, Dict
from datetime import datetime, timedelta
from .config import get_config

logger = logging.getLogger(__name__)


class MatchingCache:
    """Redis-based caching system for matching operations."""
    
    def __init__(self):
        self.config = get_config()
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False  # We'll handle encoding ourselves
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connection established")
        except redis.ConnectionError:
            logger.warning("Redis not available, using in-memory cache fallback")
            self.redis_client = None
            self._memory_cache = {}
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            else:
                # Fallback to memory cache
                cache_item = self._memory_cache.get(key)
                if cache_item and cache_item['expires'] > datetime.now():
                    return cache_item['value']
                elif cache_item:
                    # Expired item
                    del self._memory_cache[key]
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            ttl = ttl or self.config.cache_ttl
            
            if self.redis_client:
                serialized_value = pickle.dumps(value)
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                # Fallback to memory cache
                expires = datetime.now() + timedelta(seconds=ttl)
                self._memory_cache[key] = {
                    'value': value,
                    'expires': expires
                }
                return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return bool(self._memory_cache.pop(key, None))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            if self.redis_client:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # Simple pattern matching for memory cache
                keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                return len(keys_to_delete)
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {str(e)}")
            return 0
    
    # Specific caching methods for matching system
    
    def cache_skill_embedding(self, skill_name: str, embedding: list) -> bool:
        """Cache skill embedding vector."""
        key = self._generate_key("skill_embedding", skill_name.lower())
        return self.set(key, embedding, ttl=86400)  # 24 hours
    
    def get_skill_embedding(self, skill_name: str) -> Optional[list]:
        """Get cached skill embedding."""
        key = self._generate_key("skill_embedding", skill_name.lower())
        return self.get(key)
    
    def cache_skill_similarity(self, skill1: str, skill2: str, similarity: float) -> bool:
        """Cache skill similarity score."""
        # Ensure consistent ordering for cache key
        skills = sorted([skill1.lower(), skill2.lower()])
        key = self._generate_key("skill_similarity", *skills)
        return self.set(key, similarity, ttl=3600)  # 1 hour
    
    def get_skill_similarity(self, skill1: str, skill2: str) -> Optional[float]:
        """Get cached skill similarity score."""
        skills = sorted([skill1.lower(), skill2.lower()])
        key = self._generate_key("skill_similarity", *skills)
        return self.get(key)
    
    def cache_candidate_score(self, candidate_id: int, job_hash: str, score_data: Dict) -> bool:
        """Cache candidate matching score."""
        key = self._generate_key("candidate_score", candidate_id, job_hash)
        return self.set(key, score_data, ttl=1800)  # 30 minutes
    
    def get_candidate_score(self, candidate_id: int, job_hash: str) -> Optional[Dict]:
        """Get cached candidate score."""
        key = self._generate_key("candidate_score", candidate_id, job_hash)
        return self.get(key)
    
    def cache_job_analysis(self, job_description: str, analysis: Dict) -> bool:
        """Cache job description analysis."""
        job_hash = hashlib.md5(job_description.encode()).hexdigest()
        key = self._generate_key("job_analysis", job_hash)
        return self.set(key, analysis, ttl=7200)  # 2 hours
    
    def get_job_analysis(self, job_description: str) -> Optional[Dict]:
        """Get cached job analysis."""
        job_hash = hashlib.md5(job_description.encode()).hexdigest()
        key = self._generate_key("job_analysis", job_hash)
        return self.get(key)
    
    def cache_market_data(self, skill_name: str, location: str, market_data: Dict) -> bool:
        """Cache market intelligence data."""
        key = self._generate_key("market_data", skill_name.lower(), location.lower())
        return self.set(key, market_data, ttl=86400)  # 24 hours
    
    def get_market_data(self, skill_name: str, location: str = "global") -> Optional[Dict]:
        """Get cached market data."""
        key = self._generate_key("market_data", skill_name.lower(), location.lower())
        return self.get(key)
    
    def invalidate_candidate_cache(self, candidate_id: int) -> int:
        """Invalidate all cache entries for a candidate."""
        pattern = f"*candidate_score:{candidate_id}:*"
        return self.clear_pattern(pattern)
    
    def invalidate_skill_cache(self, skill_name: str) -> int:
        """Invalidate all cache entries for a skill."""
        patterns = [
            f"*skill_embedding:{skill_name.lower()}*",
            f"*skill_similarity:*{skill_name.lower()}*",
            f"*market_data:{skill_name.lower()}*"
        ]
        total_cleared = 0
        for pattern in patterns:
            total_cleared += self.clear_pattern(pattern)
        return total_cleared
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'type': 'redis',
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            else:
                return {
                    'type': 'memory',
                    'total_keys': len(self._memory_cache),
                    'memory_usage': 'unknown'
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'type': 'error', 'message': str(e)}


# Global cache instance
_cache_instance = None


def get_cache() -> MatchingCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MatchingCache()
    return _cache_instance 