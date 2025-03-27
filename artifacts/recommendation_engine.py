# recommendation_engine.py - Core recommendation algorithms and processing logic
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional, Tuple, Union
import logging
import redis
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Product recommendation engine using collaborative filtering and content-based methods."""
    
    def __init__(self, config: Dict):
        """Initialize the recommendation engine with configuration settings."""
        self.config = config
        self.redis_client = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            password=config['redis']['password'],
            db=config['redis']['db']
        )
        
        # Cache expiration times
        self.user_cache_ttl = config.get('user_cache_ttl', 3600)  # 1 hour
        self.similar_products_ttl = config.get('similar_products_ttl', 86400)  # 24 hours
        
        # Load trained models if available
        self.product_embeddings = None
        self.user_embeddings = None
        self.load_models()
        
    def load_models(self) -> None:
        """Load pre-trained recommendation models."""
        try:
            self.product_embeddings = np.load(self.config['model_paths']['product_embeddings'])
            self.user_embeddings = np.load(self.config['model_paths']['user_embeddings'])
            logger.info("Successfully loaded recommendation models")
        except Exception as e:
            logger.error(f"Failed to load recommendation models: {e}")
            # Use fallback recommendation strategy if models can't be loaded
    
    def get_recommendations(self, 
                           user_id: str, 
                           category: Optional[str] = None, 
                           limit: int = 5, 
                           exclude_ids: List[str] = None) -> List[Dict]:
        """
        Get personalized product recommendations for a specific user.
        
        Args:
            user_id: The unique identifier for the user
            category: Optional category to filter recommendations
            limit: Maximum number of recommendations to return
            exclude_ids: Product IDs to exclude from recommendations
            
        Returns:
            A list of recommended product dictionaries
        """
        exclude_ids = exclude_ids or []
        
        # Try to get cached recommendations
        cache_key = f"recommendations:{user_id}:{category or 'all'}"
        cached_recommendations = self._get_cached_recommendations(cache_key)
        
        if cached_recommendations:
            # Filter out excluded products from cached results
            filtered_recommendations = [
                product for product in cached_recommendations 
                if product['id'] not in exclude_ids
            ][:limit]
            
            if len(filtered_recommendations) >= limit:
                return filtered_recommendations
        
        # Fetch user behavior data
        user_history = self._get_user_history(user_id)
        
        if not user_history:
            # New user or no history - use popularity-based recommendations
            recommendations = self._get_popular_products(category, limit, exclude_ids)
        else:
            # Combine collaborative and content-based recommendations
            cf_recommendations = self._collaborative_filtering(user_id, category, limit*2, exclude_ids)
            cb_recommendations = self._content_based(user_id, category, limit*2, exclude_ids)
            
            # Merge recommendations with weighting
            recommendations = self._merge_recommendations(cf_recommendations, cb_recommendations, limit)
            
        # Cache the recommendations
        self._cache_recommendations(cache_key, recommendations)
        
        return recommendations[:limit]
    
    def _get_cached_recommendations(self, cache_key: str) -> List[Dict]:
        """Retrieve cached recommendations if available."""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error retrieving cached recommendations: {e}")
        return []
    
    def _cache_recommendations(self, cache_key: str, recommendations: List[Dict]) -> None:
        """Cache recommendations for future requests."""
        try:
            self.redis_client.setex(
                cache_key,
                self.user_cache_ttl,
                json.dumps(recommendations)
            )
        except Exception as e:
            logger.warning(f"Error caching recommendations: {e}")
    
    def _get_user_history(self, user_id: str) -> Dict:
        """Retrieve user's browsing and purchase history."""
        # Implementation would fetch from database
        # This is a placeholder for the actual implementation
        return {
            'viewed_products': ['product1', 'product2', 'product3'],
            'purchased_products': ['product4'],
            'cart_products': ['product5']
        }
    
    def _collaborative_filtering(self, 
                               user_id: str, 
                               category: Optional[str], 
                               limit: int, 
                               exclude_ids: List[str]) -> List[Dict]:
        """Generate recommendations based on similar users' behavior."""
        # Implementation would use user and product embeddings
        # This is a placeholder for the actual implementation
        return [
            {
                'id': 'product6',
                'name': 'Smartphone X',
                'price': 699.99,
                'rating': 4.5,
                'recommendationReason': 'Popular with similar customers'
            }
        ]
    
    def _content_based(self, 
                     user_id: str, 
                     category: Optional[str], 
                     limit: int, 
                     exclude_ids: List[str]) -> List[Dict]:
        """Generate recommendations based on product attributes."""
        # Implementation would use product features and user preferences
        # This is a placeholder for the actual implementation
        return [
            {
                'id': 'product7',
                'name': 'Wireless Earbuds',
                'price': 129.99,
                'rating': 4.2,
                'recommendationReason': 'Similar to products you viewed'
            }
        ]
    
    def _get_popular_products(self, 
                            category: Optional[str], 
                            limit: int, 
                            exclude_ids: List[str]) -> List[Dict]:
        """Get popular products as fallback recommendations."""
        # Implementation would fetch trending or popular products
        # This is a placeholder for the actual implementation
        return [
            {
                'id': 'product8',
                'name': 'Smart Watch',
                'price': 249.99,
                'rating': 4.7,
                'recommendationReason': 'Trending in this category'
            }
        ]
    
    def _merge_recommendations(self, 
                             cf_recommendations: List[Dict], 
                             cb_recommendations: List[Dict], 
                             limit: int) -> List[Dict]:
        """Merge and rank recommendations from different sources."""
        # Create a score and merge recommendations
        merged = {}
        
        # Add collaborative filtering with higher weight
        for i, product in enumerate(cf_recommendations):
            product_id = product['id']
            if product_id not in merged:
                merged[product_id] = {
                    **product,
                    'score': (len(cf_recommendations) - i) * 1.5  # Higher weight
                }
        
        # Add content-based with lower weight
        for i, product in enumerate(cb_recommendations):
            product_id = product['id']
            if product_id not in merged:
                merged[product_id] = {
                    **product,
                    'score': (len(cb_recommendations) - i)
                }
            else:
                # Boost score if product appears in both lists
                merged[product_id]['score'] += (len(cb_recommendations) - i) * 0.5
                
        # Sort by score and return top results
        results = list(merged.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove score field from output
        for product in results:
            del product['score']
            
        return results[:limit]

# Usage example
if __name__ == "__main__":
    config = {
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'password': '',
            'db': 0
        },
        'model_paths': {
            'product_embeddings': './models/product_embeddings.npy',
            'user_embeddings': './models/user_embeddings.npy'
        }
    }
    
    engine = RecommendationEngine(config)
    recommendations = engine.get_recommendations(
        user_id="user123",
        category="electronics",
        limit=5
    )
    
    print(f"Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} - {rec['recommendationReason']}")