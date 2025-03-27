pythonCopy# data_processor.py - Module for processing and analyzing user data
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import json
import os

class UserDataProcessor:
    """Process and analyze user interaction data."""
    
    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """Initialize the data processor with paths to data and optional config."""
        self.data_path = data_path
        self.config = self._load_config(config_path) if config_path else {}
        self.raw_data = None
        self.processed_data = None
        self.user_segments = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}
            
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        try:
            # Determine file type and load accordingly
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                self.raw_data = pd.read_json(self.data_path)
            elif self.data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
                
            # Perform basic preprocessing
            self._preprocess_data()
            return self.processed_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _preprocess_data(self) -> None:
        """Clean and prepare the data for analysis."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Make a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Handle missing values
        for col in df.columns:
            missing_pct = df[col].isna().mean()
            if missing_pct > 0.5:
                df.drop(columns=[col], inplace=True)
            elif df[col].dtype == 'object':
                df[col].fillna('unknown', inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Convert date columns
        date_columns = self.config.get('date_columns', [])
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Create derived features
        if 'created_at' in df.columns and 'last_active' in df.columns:
            df['account_age_days'] = (df['last_active'] - df['created_at']).dt.days
        
        self.processed_data = df
    
    def segment_users(self, features: List[str], n_clusters: int = 3) -> Dict[str, List]:
        """
        Segment users based on selected features using K-means clustering.
        
        Args:
            features: List of column names to use for clustering
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call load_data() first.")
        
        # Select and scale features
        X = self.processed_data[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to the data
        self.processed_data['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_df = self.processed_data[self.processed_data['cluster'] == i]
            cluster_stats[f'Cluster {i}'] = {
                'count': len(cluster_df),
                'percentage': len(cluster_df) / len(self.processed_data) * 100,
                'feature_means': {col: cluster_df[col].mean() for col in features}
            }
        
        self.user_segments = {
            'n_clusters': n_clusters,
            'features_used': features,
            'cluster_stats': cluster_stats,
            'centroids': kmeans.cluster_centers_.tolist()
        }
        
        return self.user_segments
    
    def visualize_segments(self, save_path: Optional[str] = None) -> None:
        """Generate visualizations for the user segments."""
        if self.user_segments is None:
            raise ValueError("No user segments available. Call segment_users() first.")
            
        if len(self.user_segments['features_used']) < 2:
            raise ValueError("Need at least 2 features for visualization")
            
        # Create a scatter plot of the first two features
        plt.figure(figsize=(10, 8))
        
        features = self.user_segments['features_used'][:2]  # Use first two features
        
        # Plot each cluster
        for i in range(self.user_segments['n_clusters']):
            cluster_data = self.processed_data[self.processed_data['cluster'] == i]
            plt.scatter(
                cluster_data[features[0]], 
                cluster_data[features[1]],
                label=f'Cluster {i}'
            )
            
        # Plot centroids
        centroids = np.array(self.user_segments['centroids'])
        plt.scatter(
            centroids[:, 0], 
            centroids[:, 1], 
            s=100, 
            c='black', 
            marker='X', 
            label='Centroids'
        )
        
        plt.title('User Segments')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()