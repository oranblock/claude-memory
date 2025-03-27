// ProductRecommendations.js - React component for displaying personalized product recommendations
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import RecommendationCard from './RecommendationCard';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';

const ProductRecommendations = ({ category, limit = 5, excludeIds = [] }) => {
  const { user } = useAuth();
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Skip recommendation fetch if not logged in
    if (!user || !user.id) {
      setLoading(false);
      return;
    }
    
    async function fetchRecommendations() {
      try {
        setLoading(true);
        setError(null);
        
        const response = await axios.get('/api/recommendations', {
          params: {
            userId: user.id,
            category,
            limit,
            excludeIds: excludeIds.join(',')
          }
        });
        
        setRecommendations(response.data.recommendations || []);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch recommendations:', err);
        setError('Unable to load recommendations at this time');
        setLoading(false);
      }
    }
    
    fetchRecommendations();
  }, [user, category, limit, excludeIds]);
  
  // Handle empty states
  if (!user || !user.id) {
    return (
      <div className="guest-recommendations">
        <h3>Sign in to see personalized recommendations</h3>
      </div>
    );
  }
  
  if (loading) {
    return <LoadingSpinner message="Finding products for you..." />;
  }
  
  if (error) {
    return <ErrorMessage message={error} />;
  }
  
  if (recommendations.length === 0) {
    return (
      <div className="empty-recommendations">
        <h3>No recommendations found</h3>
        <p>Browse more products to get personalized recommendations</p>
      </div>
    );
  }
  
  return (
    <div className="recommendations-container">
      <h2>Recommended for You</h2>
      <div className="recommendations-grid">
        {recommendations.map(product => (
          <RecommendationCard 
            key={product.id} 
            product={product} 
            reason={product.recommendationReason}
          />
        ))}
      </div>
    </div>
  );
};

export default ProductRecommendations;