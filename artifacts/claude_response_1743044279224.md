Complex Multi-Format Test Artifact (Final Version)
This test contains multiple formats, complex structures, and various code types to thoroughly test the capture functionality with proper file naming and extensions.
1. JavaScript React Application Component
javascriptCopy// UserDashboard.jsx - React component with hooks and context
import React, { useState, useEffect, useContext } from 'react';
import { fetchUserData, updateUserPreferences } from '../api/userService';
import { ThemeContext } from '../contexts/ThemeContext';
import DashboardLayout from '../layouts/DashboardLayout';
import UserStatistics from './UserStatistics';
import PreferencesPanel from './PreferencesPanel';
import { Alert, Button, Spinner } from '../components/ui';

const UserDashboard = ({ userId }) => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  
  useEffect(() => {
    async function loadUserData() {
      try {
        setLoading(true);
        const data = await fetchUserData(userId);
        setUserData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load user data');
        setLoading(false);
        console.error(err);
      }
    }
    
    loadUserData();
  }, [userId]);
  
  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };
  
  const handlePreferenceUpdate = async (preferences) => {
    try {
      await updateUserPreferences(userId, preferences);
      // Update local user data
      setUserData({
        ...userData,
        preferences
      });
    } catch (err) {
      setError('Failed to update preferences');
      console.error(err);
    }
  };
  
  if (loading) return (
    <div className="loading-container">
      <Spinner size="large" />
      <p>Loading user dashboard...</p>
    </div>
  );
  
  if (error) return (
    <Alert type="error" title="Error Loading Data">
      {error}
      <Button onClick={() => window.location.reload()}>
        Try Again
      </Button>
    </Alert>
  );
  
  return (
    <DashboardLayout>
      <div className={`user-dashboard theme-${theme}`}>
        <header className="dashboard-header">
          <div className="header-left">
            <h1>Welcome back, {userData.name}</h1>
            <p className="last-login">Last login: {new Date(userData.lastLogin).toLocaleString()}</p>
          </div>
          <div className="header-right">
            <Button onClick={toggleTheme} variant="outline">
              {theme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </Button>
          </div>
        </header>
        
        <nav className="tab-navigation">
          <Button 
            className={activeTab === 'overview' ? 'active' : ''} 
            onClick={() => handleTabChange('overview')}
          >
            Overview
          </Button>
          <Button 
            className={activeTab === 'statistics' ? 'active' : ''} 
            onClick={() => handleTabChange('statistics')}
          >
            Statistics
          </Button>
          <Button 
            className={activeTab === 'preferences' ? 'active' : ''} 
            onClick={() => handleTabChange('preferences')}
          >
            Preferences
          </Button>
          <Button 
            className={activeTab === 'reports' ? 'active' : ''} 
            onClick={() => handleTabChange('reports')}
          >
            Reports
          </Button>
        </nav>
        
        <main className="dashboard-content">
          {activeTab === 'overview' && (
            <div className="overview-panel">
              <h2>Account Overview</h2>
              <div className="account-info">
                <p><strong>Member since:</strong> {new Date(userData.joinDate).toLocaleDateString()}</p>
                <p><strong>Subscription:</strong> {userData.subscription.plan}</p>
                <p><strong>Status:</strong> {userData.status}</p>
                <p><strong>Next billing date:</strong> {new Date(userData.subscription.nextBillingDate).toLocaleDateString()}</p>
              </div>
              
              <div className="quick-stats">
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.totalLogins}</div>
                  <div className="stat-label">Total Logins</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.projectsCreated}</div>
                  <div className="stat-label">Projects</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.statistics.reportsGenerated}</div>
                  <div className="stat-label">Reports</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{userData.storage.used} / {userData.storage.total} GB</div>
                  <div className="stat-label">Storage</div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'statistics' && (
            <UserStatistics stats={userData.statistics} />
          )}
          
          {activeTab === 'preferences' && (
            <PreferencesPanel 
              preferences={userData.preferences}
              onUpdate={handlePreferenceUpdate}
            />
          )}
          
          {activeTab === 'reports' && (
            <div className="reports-panel">
              <h2>Saved Reports</h2>
              {userData.reports.length === 0 ? (
                <p>No reports saved yet.</p>
              ) : (
                <ul className="reports-list">
                  {userData.reports.map(report => (
                    <li key={report.id} className="report-item">
                      <div className="report-title">{report.title}</div>
                      <div className="report-date">{new Date(report.createdAt).toLocaleDateString()}</div>
                      <div className="report-actions">
                        <Button size="small" variant="text">View</Button>
                        <Button size="small" variant="text">Download</Button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </main>
      </div>
    </DashboardLayout>
  );
};

export default UserDashboard;
2. Python Data Analysis Module
pythonCopy# data_processor.py - Advanced data analysis with pandas and scikit-learn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

class DataAnalyzer:
    """Process, analyze and visualize user behavior and engagement data."""
    
    def __init__(self, 
                 data_path: Union[str, Path], 
                 config_path: Optional[Union[str, Path]] = None,
                 log_level: str = 'INFO'):
        """
        Initialize the data analyzer with paths to data and optional config.
        
        Args:
            data_path: Path to the data file or directory
            config_path: Path to configuration JSON file
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Configure logging
        self._setup_logging(log_level)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing DataAnalyzer with data from {data_path}")
        
        self.data_path = Path(data_path)
        self.config = self._load_config(config_path) if config_path else {}
        self.raw_data = None
        self.processed_data = None
        self.analysis_results = {}
        
        # Analysis settings
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.min_cluster_size = self.config.get('min_cluster_size', 5)
        self.n_components_pca = self.config.get('n_components_pca', 2)
        
    def _setup_logging(self, log_level: str) -> None:
        """Set up logging configuration."""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
            
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
            
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        self.logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Handle different data sources
            if self.data_path.is_dir():
                self.logger.info("Loading data from directory")
                self._load_from_directory()
            else:
                self._load_from_file()
                
            # Perform initial data validation
            self._validate_data()
            
            # Basic preprocessing
            self._preprocess_data()
            
            self.logger.info(f"Successfully loaded data with shape {self.processed_data.shape}")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}", exc_info=True)
            # Return empty DataFrame instead of None for consistency
            return pd.DataFrame()
    
    def _load_from_file(self) -> None:
        """Load data from a single file based on extension."""
        file_extension = self.data_path.suffix.lower()
        
        if file_extension == '.csv':
            self.raw_data = pd.read_csv(self.data_path)
        elif file_extension == '.json':
            self.raw_data = pd.read_json(self.data_path)
        elif file_extension == '.xlsx' or file_extension == '.xls':
            self.raw_data = pd.read_excel(self.data_path)
        elif file_extension == '.parquet':
            self.raw_data = pd.read_parquet(self.data_path)
        elif file_extension == '.feather':
            self.raw_data = pd.read_feather(self.data_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    
    def _load_from_directory(self) -> None:
        """Load and merge multiple data files from a directory."""
        data_frames = []
        file_count = 0
        
        # Process all data files in the directory
        for file_path in self.data_path.glob('**/*.*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.json', '.xlsx', '.xls', '.parquet', '.feather']:
                try:
                    self.logger.debug(f"Loading file: {file_path}")
                    df = self._load_single_file(file_path)
                    
                    if df is not None and not df.empty:
                        # Add source file info for traceability
                        df['_source_file'] = str(file_path.name)
                        data_frames.append(df)
                        file_count += 1
                except Exception as e:
                    self.logger.warning(f"Error loading file {file_path}: {e}")
        
        if not data_frames:
            raise ValueError(f"No valid data files found in {self.data_path}")
        
        self.logger.info(f"Loaded {file_count} files from directory")
        
        # Combine all dataframes - this assumes they have compatible structures
        self.raw_data = pd.concat(data_frames, ignore_index=True, sort=False)
    
    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single file based on its extension."""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                return pd.read_excel(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension == '.feather':
                return pd.read_feather(file_path)
            else:
                self.logger.warning(f"Unsupported file extension: {file_extension}")
                return None
        except Exception as e:
            self.logger.warning(f"Error loading {file_path}: {e}")
            return None
    
    def _validate_data(self) -> None:
        """Perform basic validation on the loaded data."""
        if self.raw_data is None:
            raise ValueError("No data loaded")
            
        if self.raw_data.empty:
            raise ValueError("Loaded data is empty")
            
        # Check for required columns based on analysis type
        required_columns = self.config.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            
        # Check for minimum number of rows
        min_rows = self.config.get('min_rows', 10)
        if len(self.raw_data) < min_rows:
            self.logger.warning(f"Dataset has only {len(self.raw_data)} rows, less than minimum {min_rows}")
    
    def _preprocess_data(self) -> None:
        """Clean and prepare the data for analysis."""
        self.logger.info("Preprocessing data")
        
        # Start with a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Handle missing values based on config or column type
        self._handle_missing_values(df)
        
        # Convert date/datetime columns
        date_columns = self.config.get('date_columns', [])
        for col in df.columns:
            if col in date_columns or df[col].dtype == 'object' and self._looks_like_datetime(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                    self.logger.debug(f"Converted {col} to datetime")
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # Create derived features
        self._create_derived_features(df)
        
        # Remove outliers if configured
        if self.config.get('remove_outliers', False):
            df = self._remove_outliers(df)
        
        self.processed_data = df
        self.logger.info("Data preprocessing complete")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> None:
        """Handle missing values in the DataFrame."""
        # Get missing value handling strategy from config
        missing_value_strategy = self.config.get('missing_value_strategy', 'auto')
        
        if missing_value_strategy == 'drop_rows':
            # Drop rows with any missing values
            initial_rows = len(df)
            df.dropna(inplace=True)
            dropped_rows = initial_rows - len(df)
            self.logger.info(f"Dropped {dropped_rows} rows with missing values")
            
        elif missing_value_strategy == 'drop_columns':
            # Drop columns with too many missing values
            threshold = self.config.get('missing_threshold', 0.5)
            initial_columns = len(df.columns)
            
            # Calculate percent missing in each column
            missing_percentages = df.isnull().mean()
            columns_to_drop = missing_percentages[missing_percentages > threshold].index
            
            df.drop(columns=columns_to_drop, inplace=True)
            self.logger.info(f"Dropped {len(columns_to_drop)} columns with >{threshold*100}% missing values")
            
        else:  # 'auto' or other strategy
            # Handle each column based on its data type
            for col in df.columns:
                missing_pct = df[col].isnull().mean()
                
                # Skip if no missing values
                if missing_pct == 0:
                    continue
                    
                # Log the missing percentage
                self.logger.debug(f"Column {col} has {missing_pct*100:.1f}% missing values")
                
                # For columns with majority missing, drop them
                if missing_pct > 0.5:
                    df.drop(columns=[col], inplace=True)
                    self.logger.info(f"Dropped column {col} with {missing_pct*100:.1f}% missing values")
                    continue
                
                # For categorical/object columns, fill with 'unknown' or mode
                if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                    fill_value = 'unknown'
                    if missing_pct < 0.05:  # For small amounts of missing data, use the mode
                        fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    
                # For numeric columns, use median
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                    
                # For datetime columns, use either the median or a specific value
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    if self.config.get('fill_dates_with_median', True):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # Use a specific date (e.g., dataset start or minimum date)
                        min_date = df[col].min()
                        df[col].fillna(min_date, inplace=True)
                
                # For other types, use None
                else:
                    pass  # Leave as NaN/None
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Check if a string series looks like it contains datetime values."""
        # Sample a few non-null values
        sample = series.dropna().head(5).astype(str)
        if len(sample) == 0:
            return False
            
        # Look for common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if sample.str.contains(pattern).all():
                return True
                
        return False
    
    def _create_derived_features(self, df: pd.DataFrame) -> None:
        """Create derived features based on existing columns."""
        # Check if we have datetime columns to work with
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) >= 2:
            # Look for common datetime column pairs
            time_column_pairs = [
                ('created_at', 'updated_at'),
                ('start_time', 'end_time'),
                ('login_time', 'logout_time'),
                ('first_seen', 'last_seen')
            ]
            
            # For each potential pair, create a duration feature if both exist
            for start_col, end_col in time_column_pairs:
                if start_col in datetime_cols and end_col in datetime_cols:
                    duration_col = f"{start_col.split('_')[0]}_{end_col.split('_')[0]}_duration"
                    df[duration_col] = (df[end_col] - df[start_col]).dt.total_seconds()
                    self.logger.info(f"Created duration feature: {duration_col}")
                    
        # If we have user_id and timestamp, create session features
        if 'user_id' in df.columns and any(col in datetime_cols for col in ['timestamp', 'created_at', 'event_time']):
            timestamp_col = next(col for col in ['timestamp', 'created_at', 'event_time'] if col in datetime_cols)
            
            # Add day of week, hour of day
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['hour_of_day'] = df[timestamp_col].dt.hour
            
            self.logger.info("Created time-based features: day_of_week, hour_of_day")
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from numeric columns."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Skip if we have no numeric columns
        if len(numeric_cols) == 0:
            return df
            
        # Get list of columns to check for outliers
        outlier_columns = self.config.get('outlier_columns', numeric_cols.tolist())
        outlier_columns = [col for col in outlier_columns if col in numeric_cols]
        
        if not outlier_columns:
            return df
            
        # Track original size
        original_size = len(df)
        mask = pd.Series(True, index=df.index)
        
        # For each column, find and filter outliers
        for col in outlier_columns:
            # Skip columns with all zeros or constant values
            if df[col].std() == 0:
                continue
                
            # Calculate z-scores
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            
            # Update mask to exclude outliers
            col_mask = z_scores < self.outlier_threshold
            mask &= col_mask
            
            outliers_count = (~col_mask).sum()
            if outliers_count > 0:
                self.logger.info(f"Found {outliers_count} outliers in column {col}")
                
        # Apply the mask to remove outliers
        filtered_df = df[mask].copy()
        removed_count = original_size - len(filtered_df)
        
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} outlier rows ({removed_count/original_size:.1%} of data)")
            
        return filtered_df
    
    def cluster_analysis(self, 
                         features: List[str], 
                         n_clusters: int = 3,
                         method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform cluster analysis on the dataset.
        
        Args:
            features: List of column names to use for clustering
            n_clusters: Number of clusters for K-means
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Dictionary with cluster assignments and metrics
        """
        self.logger.info(f"Performing {method} cluster analysis with {len(features)} features")
        
        if self.processed_data is None:
            raise ValueError("No processed data available. Call load_data() first.")
            
        # Validate features
        for feature in features:
            if feature not in self.processed_data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset")
                
        # Select and scale features
        X = self.processed_data[features].copy()
        
        # Handle missing values if any
        X = X.fillna(X.median())
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering based on the selected method
        if method.lower() == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(X_scaled)
            
            # Calculate silhouette score if scikit-learn is recent enough
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X_scaled, clusters)
                self.logger.info(f"Silhouette score: {silhouette:.3f}")
            except ImportError:
                silhouette = None
                self.logger.warning("Silhouette score calculation not available")
                
            # Store results including centroids
            clustering_result = {
                'method': 'kmeans',
                'n_clusters': n_clusters,
                'features_used': features,
                'silhouette_score': silhouette,
                'centroids': model.cluster_centers_.tolist(),
                'inertia': model.inertia_
            }
            
        elif method.lower() == 'dbscan':
            # DBSCAN doesn't require specifying number of clusters beforehand
            model = DBSCAN(eps=self.config.get('dbscan_eps', 0.5), 
                          min_samples=self.config.get('dbscan_min_samples', 5))
            clusters = model.fit_predict(X_scaled)
            
            # Count number of actual clusters found (excluding noise points labeled as -1)
            n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_points = np.sum(clusters == -1)
            
            self.logger.info(f"DBSCAN found {n_clusters_found} clusters and {noise_points} noise points")
            
            # Store results
            clustering_result = {
                'method': 'dbscan',
                'n_clusters_found': n_clusters_found,
                'features_used': features,
                'noise_points': int(noise_points),
                'noise_percentage': float(noise_points / len(clusters))
            }
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
            
        # Add clusters to dataframe
        self.processed_data['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in set(clusters):
            if i == -1 and method.lower() == 'dbscan':
                label = 'Noise'
            else:
                label = f'Cluster {i}'
                
            cluster_df = self.processed_data[self.processed_data['cluster'] == i]
            cluster_stats[label] = {
                'count': len(cluster_df),
                'percentage': len(cluster_df) / len(self.processed_data) * 100,
                'feature_means': {col: float(cluster_df[col].mean()) for col in features 
                                 if pd.api.types.is_numeric_dtype(cluster_df[col])}
            }
            
        clustering_result['cluster_stats'] = cluster_stats
        
        # Store in results
        self.analysis_results['clustering'] = clustering_result
        
        return clustering_result
    
    def dimension_reduction(self, 
                            features: List[str], 
                            n_components: int = 2) -> pd.DataFrame:
        """
        Perform PCA dimension reduction for visualization.
        
        Args:
            features: List of features to use
            n_components: Number of PCA components (default 2 for visualization)
            
        Returns:
            DataFrame with PCA components
        """
        self.logger.info(f"Performing PCA with {len(features)} features")
        
        if self.processed_data is None:
            raise ValueError("No processed data available. Call load_data() first.")
            
        # Select and scale features
        X = self.processed_data[features].fillna(0).copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Create component names
        component_names = [f'PC{i+1}' for i in range(n_components)]
        
        # Create DataFrame with components
        pca_df = pd.DataFrame(components, columns=component_names)
        
        # Add cluster information if available
        if 'cluster' in self.processed_data.columns:
            pca_df['cluster'] = self.processed_data['cluster'].values
            
        # Store explained variance
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        
        self.logger.info(f"Explained variance by {n_components} components: "
                        f"{explained_variance.sum():.2f}%")
        
        # Store in results
        self.analysis_results['pca'] = {
            'n_components': n_components,
            'features_used': features,
            'explained_variance': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'component_names': component_names,
            'feature_weights': pca.components_.tolist(),
        }
        
        return pca_df
    
    def visualize_clusters(self, 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
        """
        Visualize clusters using PCA or original features.
        
        Args:
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        if 'cluster' not in self.processed_data.columns:
            raise ValueError("No cluster assignments found. Run cluster_analysis first.")
            
        self.logger.info("Generating cluster visualization")
        
        # Check if we have PCA results
        if 'pca' in self.analysis_results:
            # Use PCA components for visualization
            pca_results = self.analysis_results['pca']
            component_names = pca_results['component_names'][:2]  # Only use first 2 components
            
            # Get PCA data
            if len(component_names) >= 2:
                x_col, y_col = component_names[0], component_names[1]
                
                # Apply PCA again to get the component values
                features = pca_results['features_used']
                pca_df = self.dimension_reduction(features, n_components=2)
                
                # Prepare plot data
                x = pca_df[x_col]
                y = pca_df[y_col]
                
                # Create labels
                x_label = f"{x_col} ({pca_results['explained_variance'][0]:.1f}%)"
                y_label = f"{y_col} ({pca_results['explained_variance'][1]:.1f}%)"
            else:
                raise ValueError("Need at least 2 PCA components for visualization")
                
        else:
            # If no PCA, use the first two numeric features
            numeric_cols = self.processed_data.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric features for visualization")
                
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            x = self.processed_data[x_col]
            y = self.processed_data[y_col]
            
            # Create labels
            x_label = x_col
            y_label = y_col
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Get cluster information
        clusters = self.processed_data['cluster'].values
        unique_clusters = sorted(set(clusters))
        
        # Create color map
        if -1 in unique_clusters:  # DBSCAN with noise points
            # Use a grey color for noise points
            cmap = plt.cm.get_cmap('viridis', len(unique_clusters) - 1)
            colors = {cluster: cmap(i) for i, cluster in enumerate(c for c in unique_clusters if c != -1)}
            colors[-1] = (0.7, 0.7, 0.7, 1.0)  # Grey for noise
        else:
            cmap = plt.cm.get_cmap('viridis', len(unique_clusters))
            colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
        
        # Plot each cluster
        for cluster in unique_clusters:
            mask = clusters == cluster
            label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
            plt.scatter(x[mask], y[mask], c=[colors[cluster]], label=label, alpha=0.7)
        
        # Add centroids if K-means was used
        if 'clustering' in self.analysis_results and self.analysis_results['clustering']['method'] == 'kmeans':
            if 'pca' in self.analysis_results:
                # Transform centroids to PCA space
                centroids = np.array(self.analysis_results['clustering']['centroids'])
                features = self.analysis_results['clustering']['features_used']
                
                # Need to apply same scaling and PCA transformation
                scaler = StandardScaler()
                X = self.processed_data[features].fillna(0).copy()
                scaler.fit(X)
                
                pca = PCA(n_components=2)
                pca.fit(scaler.transform(X))
                
                # Transform centroids
                centroids_scaled = scaler.transform(centroids)
                centroids_pca = pca.transform(centroids_scaled)
                
                # Plot centroids
                plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                           marker='X', s=100, c='black', label='Centroids')
                
        # Add labels and title
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Cluster Visualization')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add cluster statistics as text
        if 'clustering' in self.analysis_results and 'cluster_stats' in self.analysis_results['clustering']:
            stats = self.analysis_results['clustering']['cluster_stats']
            
            # Create text for each cluster
            text_lines = []
            for label, cluster_data in stats.items():
                line = f"{label}: {cluster_data['count']} samples ({cluster_data['percentage']:.1f}%)"
                text_lines.append(line)
                
            # Add text to plot
            plt.figtext(0.02, 0.02, '\n'.join(text_lines), fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    def export_results(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Export analysis results to various formats.
        
        Args:
            output_dir: Directory to save exports
            
        Returns:
            Dictionary mapping result types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting results to {output_dir}")
        
        exported_files = {}
        
        # Export processed data
        if self.processed_data is not None:
            # CSV export
            csv_path = output_dir / 'processed_data.csv'
            self.processed_data.to_csv(csv_path, index=False)
            exported_files['processed_data_csv'] = str(csv_path)
            
            # Excel export with multiple sheets
            excel_path = output_dir / 'analysis_results.xlsx'
            with pd.ExcelWriter(excel_path) as writer:
                # Main data
                self.processed_data.to_excel(writer, sheet_name='ProcessedData', index=False)
                
                # Add clustering results if available
                if 'cluster' in self.processed_data.columns:
                    # Cluster summary
                    if 'clustering' in self.analysis_results and 'cluster_stats' in self.analysis_results['clustering']:
                        stats = self.analysis_results['clustering']['cluster_stats']
                        
                        # Convert to DataFrame
                        summary_rows = []
                        for label, data in stats.items():
                            row = {
                                'Cluster': label,
                                'Count': data['count'],
                                'Percentage': data['percentage']
                            }
                            # Add feature means
                            for feature, value in data['feature_means'].items():
                                row[f'Avg_{feature}'] = value
                                
                            summary_rows.append(row)
                            
                        summary_df = pd.DataFrame(summary_rows)
                        summary_df.to_excel(writer, sheet_name='ClusterSummary', index=False)
                        
                # PCA results if available
                if 'pca' in self.analysis_results:
                    pca_data = self.analysis_results['pca']
                    
                    # Create variance explained table
                    variance_data = {
                        'Component': [f'PC{i+1}' for i in range(len(pca_data['explained_variance']))],
                        'Explained Variance (%)': pca_data['explained_variance'],
                        'Cumulative Variance (%)': pca_data['cumulative_variance']
                    }
                    
                    pd.DataFrame(variance_data).to_excel(writer, sheet_name='PCA_Variance', index=False)
                    
                    # Create feature loading table
                    feature_weights = np.array(pca_data['feature_weights'])
                    features = pca_data['features_used']
                    
                    loading_data = {
                        'Feature': features
                    }
                    
                    for i, component in enumerate(pca_data['component_names']):
                        if i < len(feature_weights):
                            loading_data[component] = feature_weights[i]
                    
                    pd.DataFrame(loading_data).to_excel(writer, sheet_name='PCA_Loadings', index=False)
                    
            exported_files['excel_report'] = str(excel_path)
            
        # Export visualizations
        if 'cluster' in self.processed_data.columns:
            vis_path = output_dir / 'cluster_visualization.png'
            self.visualize_clusters(save_path=str(vis_path), show_plot=False)
            exported_files['cluster_visualization'] = str(vis_path)
            
        # Export JSON report with all analysis results
        if self.analysis_results:
            # Create a copy of analysis results that's JSON serializable
            json_safe_results = self._create_json_safe_results()
            
            json_path = output_dir / 'analysis_results.json'
            with open(json_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)
                
            exported_files['analysis_json'] = str(json_path)
            
        self.logger.info(f"Exported {len(exported_files)} files")
        return exported_files
    
    def _create_json_safe_results(self) -> Dict[str, Any]:
        """Create a JSON-serializable copy of analysis results."""
        # Start with a deep copy
        import copy
        results = copy.deepcopy(self.analysis_results)
        
        # Function to make a value JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, set):
                return [make_serializable(item) for item in obj]
            else:
                return obj
                
        # Process the entire results dictionary
        serializable_results = make_serializable(results)
        
        # Add metadata
        serializable_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'dataset_shape': list(self.processed_data.shape) if self.processed_data is not None else None,
            'version': '1.0.0'
        }
        
        return serializable_results
3. SQL Database Schema for Analytics Platform
sqlCopy-- analytics_platform_schema.sql
-- Comprehensive schema for a data analytics platform

-- Users and Authentication
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    account_status VARCHAR(20) NOT NULL DEFAULT 'active',
    role VARCHAR(20) NOT NULL DEFAULT 'analyst',
    CONSTRAINT chk_account_status CHECK (account_status IN ('active', 'inactive', 'suspended', 'deleted')),
    CONSTRAINT chk_role CHECK (role IN ('admin', 'manager', 'analyst', 'viewer', 'guest'))
);

-- User profiles with additional information
CREATE TABLE user_profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    company VARCHAR(100),
    job_title VARCHAR(100),
    department VARCHAR(100),
    phone VARCHAR(30),
    timezone VARCHAR(50) DEFAULT 'UTC',
    language_preference VARCHAR(10) DEFAULT 'en',
    theme_preference VARCHAR(20) DEFAULT 'light',
    notification_settings JSONB DEFAULT '{"email": true, "in_app": true, "reports": true}',
    bio TEXT,
    avatar_url VARCHAR(255),
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_user_profile UNIQUE (user_id)
);

-- Teams and Organizations
CREATE TABLE organizations (
    org_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    domain VARCHAR(100),
    plan_type VARCHAR(50) NOT NULL DEFAULT 'standard',
    max_users INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    billing_email VARCHAR(100),
    contact_name VARCHAR(100),
    contact_phone VARCHAR(30),
    settings JSONB,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    settings JSONB,
    active BOOLEAN DEFAULT TRUE
);

CREATE TABLE team_members (
    team_id INTEGER NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL DEFAULT 'member',
    joined_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    invited_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (team_id, user_id),
    CONSTRAINT chk_team_role CHECK (role IN ('admin', 'owner', 'member'))
);

-- Data Sources and Connections
CREATE TABLE data_sources (
    source_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    connection_details JSONB NOT NULL,
    credentials_secret_id VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    enabled BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_source_type CHECK (type IN 
        ('database', 'api', 'file_upload', 's3', 'gcs', 'azure_blob', 
         'bigquery', 'snowflake', 'redshift', 'google_sheets', 'excel'))
);

CREATE TABLE data_source_permissions (
    source_id INTEGER NOT NULL REFERENCES data_sources(source_id) ON DELETE CASCADE,
    entity_type VARCHAR(10) NOT NULL,
    entity_id INTEGER NOT NULL,
    permission_level VARCHAR(20) NOT NULL,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (source_id, entity_type, entity_id),
    CONSTRAINT chk_entity_type CHECK (entity_type IN ('user', 'team')),
    CONSTRAINT chk_permission_level CHECK (permission_level IN 
        ('owner', 'editor', 'viewer', 'uploader'))
);

-- Dataset Management
CREATE TABLE datasets (
    dataset_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    source_id INTEGER REFERENCES data_sources(source_id),
    source_query TEXT,
    schema_definition JSONB,
    row_count INTEGER,
    column_count INTEGER,
    file_size_bytes BIGINT,
    file_format VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_updated_at TIMESTAMP,
    last_synced_at TIMESTAMP,
    refresh_schedule VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    tags TEXT[],
    CONSTRAINT chk_file_format CHECK (file_format IN 
        ('csv', 'json', 'parquet', 'avro', 'excel', 'sql', 'unknown'))
);

CREATE TABLE dataset_versions (
    version_id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    storage_location VARCHAR(255) NOT NULL,
    change_description TEXT,
    row_count INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    is_current BOOLEAN DEFAULT FALSE,
    CONSTRAINT unique_dataset_version UNIQUE (dataset_id, version_number)
);

CREATE TABLE dataset_columns (
    column_id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    column_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(100),
    description TEXT,
    data_type VARCHAR(50) NOT NULL,
    is_nullable BOOLEAN DEFAULT TRUE,
    is_unique BOOLEAN DEFAULT FALSE,
    is_indexed BOOLEAN DEFAULT FALSE,
    is_primary_key BOOLEAN DEFAULT FALSE,
    is_foreign_key BOOLEAN DEFAULT FALSE,
    referenced_table VARCHAR(100),
    referenced_column VARCHAR(100),
    position INTEGER NOT NULL,
    example_values TEXT[],
    statistics JSONB,
    CONSTRAINT unique_column_position UNIQUE (dataset_id, position),
    CONSTRAINT unique_column_name UNIQUE (dataset_id, column_name)
);

-- Analysis and Dashboards
CREATE TABLE dashboards (
    dashboard_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    layout JSONB,
    theme VARCHAR(20) DEFAULT 'default',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    is_published BOOLEAN DEFAULT FALSE,
    published_at TIMESTAMP,
    published_by INTEGER REFERENCES users(user_id),
    view_count INTEGER DEFAULT 0,
    tags TEXT[],
    thumbnail_url VARCHAR(255)
);

CREATE TABLE dashboard_permissions (
    dashboard_id INTEGER NOT NULL REFERENCES dashboards(dashboard_id) ON DELETE CASCADE,
    entity_type VARCHAR(10) NOT NULL,
    entity_id INTEGER NOT NULL,
    permission_level VARCHAR(20) NOT NULL,
    granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    granted_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (dashboard_id, entity_type, entity_id),
    CONSTRAINT chk_entity_type CHECK (entity_type IN ('user', 'team')),
    CONSTRAINT chk_permission_level CHECK (permission_level IN 
        ('owner', 'editor', 'viewer'))
);

CREATE TABLE visualizations (
    visualization_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    query_definition JSONB NOT NULL,
    chart_config JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    last_run_at TIMESTAMP,
    run_time_ms INTEGER,
    error_message TEXT,
    thumbnail_url VARCHAR(255),
    CONSTRAINT chk_visualization_type CHECK (type IN 
        ('bar', 'line', 'pie', 'scatter', 'area', 'table', 'pivot', 
         'heatmap', 'map', 'funnel', 'gauge', 'kpi', 'histogram', 'box'))
);

CREATE TABLE dashboard_visualizations (
    dashboard_id INTEGER NOT NULL REFERENCES dashboards(dashboard_id) ON DELETE CASCADE,
    visualization_id INTEGER NOT NULL REFERENCES visualizations(visualization_id) ON DELETE CASCADE,
    position_config JSONB NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (dashboard_id, visualization_id)
);

-- Reports and Schedules
CREATE TABLE reports (
    report_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dashboard_id INTEGER REFERENCES dashboards(dashboard_id),
    format VARCHAR(20) NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_modified_by INTEGER REFERENCES users(user_id),
    is_public BOOLEAN DEFAULT FALSE,
    public_url_token VARCHAR(100),
    CONSTRAINT chk_report_format CHECK (format IN 
        ('pdf', 'excel', 'csv', 'html', 'json'))
);

CREATE TABLE scheduled_tasks (
    task_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    task_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(20) NOT NULL,
    entity_id INTEGER NOT NULL,
    schedule_expression VARCHAR(100) NOT NULL,
    schedule_timezone VARCHAR(50) DEFAULT 'UTC',
    parameters JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    last_status VARCHAR(20),
    last_error TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_task_type CHECK (task_type IN 
        ('report_email', 'dashboard_refresh', 'dataset_sync', 'alert_check')),
    CONSTRAINT chk_entity_type CHECK (entity_type IN 
        ('report', 'dashboard', 'dataset', 'alert'))
);

CREATE TABLE task_recipients (
    task_id INTEGER NOT NULL REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
    recipient_type VARCHAR(10) NOT NULL,
    recipient_value VARCHAR(100) NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (task_id, recipient_type, recipient_value),
    CONSTRAINT chk_recipient_type CHECK (recipient_type IN 
        ('email', 'slack', 'webhook', 'user_id', 'team_id'))
);

-- Alerts and Monitoring
CREATE TABLE alerts (
    alert_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    dataset_id INTEGER REFERENCES datasets(dataset_id),
    query_definition JSONB NOT NULL,
    condition_type VARCHAR(20) NOT NULL,
    condition_value NUMERIC,
    comparison_type VARCHAR(20) NOT NULL,
    time_window INTEGER,
    time_window_unit VARCHAR(10),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id),
    last_modified_at TIMESTAMP,
    last_check_at TIMESTAMP,
    last_triggered_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    severity VARCHAR(10) DEFAULT 'medium',
    CONSTRAINT chk_condition_type CHECK (condition_type IN 
        ('threshold', 'change', 'anomaly', 'absence')),
    CONSTRAINT chk_comparison_type CHECK (comparison_type IN 
        ('greater_than', 'less_than', 'equal_to', 'not_equal_to', 
         'percentage_increase', 'percentage_decrease')),
    CONSTRAINT chk_time_window_unit CHECK (time_window_unit IN 
        ('minute', 'hour', 'day', 'week')),
    CONSTRAINT chk_severity CHECK (severity IN 
        ('low', 'medium', 'high', 'critical'))
);

CREATE TABLE alert_recipients (
    alert_id INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    recipient_type VARCHAR(10) NOT NULL,
    recipient_value VARCHAR(100) NOT NULL,
    added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    added_by INTEGER REFERENCES users(user_id),
    PRIMARY KEY (alert_id, recipient_type, recipient_value),
    CONSTRAINT chk_recipient_type CHECK (recipient_type IN 
        ('email', 'slack', 'webhook', 'user_id', 'team_id'))
);

CREATE TABLE alert_history (
    history_id SERIAL PRIMARY KEY,
    alert_id INTEGER NOT NULL REFERENCES alerts(alert_id) ON DELETE CASCADE,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual_value NUMERIC,
    comparison_value NUMERIC,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_type VARCHAR(20),
    resolution_note TEXT,
    CONSTRAINT chk_resolution_type CHECK (resolution_type IN 
        ('auto', 'manual', 'acknowledged'))
);

-- Audit and Activity Tracking
CREATE TABLE activity_log (
    log_id SERIAL PRIMARY KEY,
    org_id INTEGER NOT NULL REFERENCES organizations(org_id),
    user_id INTEGER REFERENCES users(user_id),
    activity_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id INTEGER,
    action VARCHAR(20) NOT NULL,
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_action CHECK (action IN 
        ('create', 'read', 'update', 'delete', 'login', 'logout', 
         'export', 'share', 'run', 'refresh'))
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status_role ON users(account_status, role);
CREATE INDEX idx_team_members_user ON team_members(user_id);
CREATE INDEX idx_datasets_org ON datasets(org_id);
CREATE INDEX idx_datasets_source ON datasets(source_id);
CREATE INDEX idx_dashboards_org ON dashboards(org_id);
CREATE INDEX idx_dashboards_created_by ON dashboards(created_by);
CREATE INDEX idx_visualizations_dataset ON visualizations(dataset_id);
CREATE INDEX idx_activity_log_timestamp ON activity_log(timestamp);
CREATE INDEX idx_activity_log_user ON activity_log(user_id);
CREATE INDEX idx_activity_log_type_action ON activity_log(activity_type, action);

-- Create useful views for common queries
CREATE VIEW active_users_view AS
    SELECT 
        u.user_id, 
        u.username, 
        u.email, 
        u.first_name, 
        u.last_name, 
        u.role,
        p.company,
        p.job_title,
        u.last_login_at,
        u.created_at,
        COUNT(DISTINCT tm.team_id) AS team_count
    FROM users u
    LEFT JOIN user_profiles p ON u.user_id = p.user_id
    LEFT JOIN team_members tm ON u.user_id = tm.user_id
    WHERE u.account_status = 'active'
    GROUP BY u.user_id, p.profile_id;

CREATE VIEW dashboard_analytics_view AS
    SELECT
        d.dashboard_id,
        d.name,
        d.created_at,
        u_created.username AS created_by_user,
        d.last_modified_at,
        u_modified.username AS modified_by_user,
        d.view_count,
        d.is_published,
        COUNT(dv.visualization_id) AS visualization_count,
        MAX(d.last_modified_at) AS last_update
    FROM dashboards d
    LEFT JOIN users u_created ON d.created_by = u_created.user_id
    LEFT JOIN users u_modified ON d.last_modified_by = u_modified.user_id
    LEFT JOIN dashboard_visualizations dv ON d.dashboard_id = dv.dashboard_id
    GROUP BY 
        d.dashboard_id, 
        d.name, 
        d.created_at, 
        u_created.username, 
        d.last_modified_at, 
        u_modified.username,
        d.view_count,
        d.is_published;

CREATE VIEW dataset_usage_view AS
    SELECT
        ds.dataset_id,
        ds.name,
        ds.created_at,
        ds.row_count,
        ds.last_synced_at,
        COUNT(DISTINCT v.visualization_id) AS visualization_count,
        COUNT(DISTINCT dv.dashboard_id) AS dashboard_count,
        COUNT(DISTINCT a.alert_id) AS alert_count
    FROM datasets ds
    LEFT JOIN visualizations v ON ds.dataset_id = v.dataset_id
    LEFT JOIN dashboard_visualizations dv ON v.visualization_id = dv.visualization_id
    LEFT JOIN alerts a ON ds.dataset_id = a.dataset_id
    GROUP BY
        ds.dataset_id,
        ds.name,
        ds.created_at,
        ds.row_count,
        ds.last_synced_at;
4. CSS Styling for the Dashboard
cssCopy/* dashboard.css - Comprehensive styling for analytics dashboard */

:root {
  /* Color palette */
  --color-primary: #3f51b5;
  --color-primary-light: #757de8;
  --color-primary-dark: #002984;
  --color-secondary: #ff4081;
  --color-secondary-light: #ff79b0;
  --color-secondary-dark: #c60055;
  
  /* Neutral colors */
  --color-text-primary: #212121;
  --color-text-secondary: #616161;
  --color-text-disabled: #9e9e9e;
  --color-background: #f5f5f5;
  --color-surface: #ffffff;
  --color-divider: #e0e0e0;
  
  /* Status colors */
  --color-success: #4caf50;
  --color-warning: #ff9800;
  --color-error: #f44336;
  --color-info: #2196f3;
  
  /* Chart colors */
  --color-chart-1: #3366cc;
  --color-chart-2: #dc3912;
  --color-chart-3: #ff9900;
  --color-chart-4: #109618;
  --color-chart-5: #990099;
  --color-chart-6: #0099c6;
  --color-chart-7: #dd4477;
  --color-chart-8: #66aa00;
  
  /* Typography */
  --font-family-primary: 'Roboto', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  --font-family-secondary: 'Roboto Condensed', 'Arial Narrow', sans-serif;
  --font-family-monospace: 'Roboto Mono', 'Consolas', monospace;
  
  /* Font sizes */
  --font-size-xs: 0.75rem;    /* 12px */
  --font-size-sm: 0.875rem;   /* 14px */
  --font-size-md: 1rem;       /* 16px */
  --font-size-lg: 1.125rem;   /* 18px */
  --font-size-xl: 1.25rem;    /* 20px */
  --font-size-2xl: 1.5rem;    /* 24px */
  --font-size-3xl: 1.875rem;  /* 30px */
  --font-size-4xl: 2.25rem;   /* 36px */
  
  /* Spacing */
  --spacing-xs: 0.25rem;      /* 4px */
  --spacing-sm: 0.5rem;       /* 8px */
  --spacing-md: 1rem;         /* 16px */
  --spacing-lg: 1.5rem;       /* 24px */
  --spacing-xl: 2rem;         /* 32px */
  --spacing-2xl: 2.5rem;      /* 40px */
  --spacing-3xl: 3rem;        /* 48px */
  
  /* Border radius */
  --border-radius-sm: 0.125rem;  /* 2px */
  --border-radius-md: 0.25rem;   /* 4px */
  --border-radius-lg: 0.5rem;    /* 8px */
  --border-radius-xl: 1rem;      /* 16px */
  --border-radius-full: 9999px;  /* Round */
  
  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  
  /* Transitions */
  --transition-fast: 150ms;
  --transition-normal: 250ms;
  --transition-slow: 350ms;
  
  /* Z-index layers */
  --z-index-dropdown: 1000;
  --z-index-sticky: 1100;
  --z-index-fixed: 1200;
  --z-index-modal-backdrop: 1300;
  --z-index-modal: 1400;
  --z-index-popover: 1500;
  --z-index-tooltip: 1600;
}

/* Base styles */
*,
*::before,
*::after {
  box-sizing: border-box;
}

html {
  font-size: 16px;
  scroll-behavior: smooth;
}

body {
  font-family: var(--font-family-primary);
  font-size: var(--font-size-md);
  line-height: 1.5;
  color: var(--color-text-primary);
  background-color: var(--color-background);
  margin: 0;
  padding: 0;
  overflow-x: hidden;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  font-weight: 500;
  line-height: 1.2;
  color: var(--color-text-primary);
}

h1 {
  font-size: var(--font-size-4xl);
}

h2 {
  font-size: var(--font-size-3xl);
}

h3 {
  font-size: var(--font-size-2xl);
}

h4 {
  font-size: var(--font-size-xl);
}

h5 {
  font-size: var(--font-size-lg);
}

h6 {
  font-size: var(--font-size-md);
}

p {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
}

a {
  color: var(--color-primary);
  text-decoration: none;
  transition: color var(--transition-fast) ease;
}

a:hover {
  color: var(--color-primary-dark);
  text-decoration: underline;
}

img {
  max-width: 100%;
  height: auto;
}

input, select, textarea, button {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}

/* Grid & Layout */
.container {
  width: 100%;
  padding-right: var(--spacing-lg);
  padding-left: var(--spacing-lg);
  margin-right: auto;
  margin-left: auto;
  max-width: 1320px;
}

.row {
  display: flex;
  flex-wrap: wrap;
  margin-right: calc(var(--spacing-md) * -1);
  margin-left: calc(var(--spacing-md) * -1);
}

.col {
  flex: 1 0 0%;
  padding-right: var(--spacing-md);
  padding-left: var(--spacing-md);
}

/* For specific column sizes */
.col-1 { flex: 0 0 8.333333%; max-width: 8.333333%; }
.col-2 { flex: 0 0 16.666667%; max-width: 16.666667%; }
.col-3 { flex: 0 0 25%; max-width: 25%; }
.col-4 { flex: 0 0 33.333333%; max-width: 33.333333%; }
.col-5 { flex: 0 0 41.666667%; max-width: 41.666667%; }
.col-6 { flex: 0 0 50%; max-width: 50%; }
.col-7 { flex: 0 0 58.333333%; max-width: 58.333333%; }
.col-8 { flex: 0 0 66.666667%; max-width: 66.666667%; }
.col-9 { flex: 0 0 75%; max-width: 75%; }
.col-10 { flex: 0 0 83.333333%; max-width: 83.333333%; }
.col-11 { flex: 0 0 91.666667%; max-width: 91.666667%; }
.col-12 { flex: 0 0 100%; max-width: 100%; }

/* Dashboard Main Components */
.dashboard {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.dashboard-header {
  background-color: var(--color-surface);
  border-bottom: 1px solid var(--color-divider);
  padding: var(--spacing-md) var(--spacing-lg);
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: sticky;
  top: 0;
  z-index: var(--z-index-sticky);
  box-shadow: var(--shadow-sm);
}

.dashboard-sidebar {
  width: 250px;
  background-color: var(--color-surface);
  border-right: 1px solid var(--color-divider);
  height: calc(100vh - 64px);
  position: fixed;
  top: 64px;
  left: 0;
  z-index: var(--z-index-fixed);
  transition: transform var(--transition-normal) ease;
  overflow-y: auto;
  box-shadow: var(--shadow-sm);
}

.dashboard-content {
  flex: 1;
  margin-left: 250px;
  padding: var(--spacing-lg);
  transition: margin-left var(--transition-normal) ease;
}

.dashboard-collapsed .dashboard-sidebar {
  transform: translateX(-250px);
}

.dashboard-collapsed .dashboard-content {
  margin-left: 0;
}

/* Form Elements */
.form-group {
  margin-bottom: var(--spacing-md);
}

.form-label {
  display: inline-block;
  margin-bottom: var(--spacing-xs);
  font-weight: 500;
}

.form-control {
  display: block;
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: var(--font-size-md);
  line-height: 1.5;
  color: var(--color-text-primary);
  background-color: var(--color-surface);
  background-clip: padding-box;
  border: 1px solid var(--color-divider);
  border-radius: var(--border-radius-md);
  transition: border-color var(--transition-fast) ease, box-shadow var(--transition-fast) ease;
}

.form-control:focus {
  color: var(--color-text-primary);
  background-color: var(--color-surface);
  border-color: var(--color-primary-light);
  outline: 0;
  box-shadow: 0 0 0 0.2rem rgba(63, 81, 181, 0.25);
}

.form-select {
  display: block;
  width: 100%;
  padding: var(--spacing-sm) var(--spacing-lg) var(--spacing-sm) var(--spacing-md);
  font-size: var(--font-size-md);
  line-height: 1.5;
  color: var(--color-text-primary);
  background-color: var(--color-surface);
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3e%3cpath fill='none' stroke='%23343a40' stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M2 5l6 6 6-6'/%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right var(--spacing-md) center;
  background-size: 16px 12px;
  border: 1px solid var(--color-divider);
  border-radius: var(--border-radius-md);
  appearance: none;
}

/* Buttons */
.btn {
  display: inline-block;
  font-weight: 400;
  text-align: center;
  white-space: nowrap;
  vertical-align: middle;
  user-select: none;
  padding: var(--spacing-sm) var(--spacing-lg);
  font-size: var(--font-size-md);
  line-height: 1.5;
  border-radius: var(--border-radius-md);
  transition: color var(--transition-fast) ease, background-color var(--transition-fast) ease, border-color var(--transition-fast) ease, box-shadow var(--transition-fast) ease;
  cursor: pointer;
}

.btn:focus, .btn:hover {
  text-decoration: none;
}

.btn-primary {
  color: white;
  background-color: var(--color-primary);
  border: 1px solid var(--color-primary);
}

.btn-primary:hover {
  background-color: var(--color-primary-dark);
  border-color: var(--color-primary-dark);
}

.btn-secondary {
  color: white;
  background-color: var(--color-secondary);
  border: 1px solid var(--color-secondary);
}

.btn-secondary:hover {
  background-color: var(--color-secondary-dark);
  border-color: var(--color-secondary-dark);
}

.btn-outline-primary {
  color: var(--color-primary);
  background-color: transparent;
  border: 1px solid var(--color-primary);
}

.btn-outline-primary:hover {
  color: white;
  background-color: var(--color-primary);
}

.btn-success {
  color: white;
  background-color: var(--color-success);
  border: 1px solid var(--color-success);
}

.btn-warning {
  color: white;
  background-color: var(--color-warning);
  border: 1px solid var(--color-warning);
}

.btn-danger {
  color: white;
  background-color: var(--color-error);
  border: 1px solid var(--color-error);
}

.btn-sm {
  padding: var(--spacing-xs) var(--spacing-md);
  font-size: var(--font-size-sm);
}

.btn-lg {
  padding: var(--spacing-md) var(--spacing-xl);
  font-size: var(--font-size-lg);
}

/* Cards */
.card {
  background-color: var(--color-surface);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  transition: transform var(--transition-normal) ease, box-shadow var(--transition-normal) ease;
}

.card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.card-header {
  padding: var(--spacing-md) var(--spacing-lg);
  background-color: rgba(0, 0, 0, 0.03);
  border-bottom: 1px solid var(--color-divider);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  margin: 0;
  font-size: var(--font-size-lg);
  font-weight: 500;
}

.card-body {
  padding: var(--spacing-lg);
}

.card-footer {
  padding: var(--spacing-md) var(--spacing-lg);
  background-color: rgba(0, 0, 0, 0.03);
  border-top: 1px solid var(--color-divider);
}

/* Widgets */
.widget-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  grid-gap: var(--spacing-lg);
  margin-bottom: var(--spacing-xl);
}

.widget {
  background-color: var(--color-surface);
  border-radius: var(--border-radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  height: 300px;
  display: flex;
  flex-direction: column;
  transition: transform var(--transition-normal) ease, box-shadow var(--transition-normal) ease;
}

.widget:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}

.widget-header {
  padding: var(--spacing-md);
  border-bottom: 1px solid var(--color-divider);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.widget-title {
  margin: 0;
  font-size: var(--font-size-md);
  font-weight: 500;
}

.widget-content {
  flex: 1;
  padding: var(--spacing-md);
  overflow: auto;
  position: relative;
}

/* Chart styles */
.chart-container {
  width: 100%;
  height: 100%;
  min-height: 200px;
  position: relative;
}

/* Table styles */
.table-responsive {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.table {
  width: 100%;
  margin-bottom: var(--spacing-md);
  color: var(--color-text-primary);
  border-collapse: collapse;
}

.table th,
.table td {
  padding: var(--spacing-sm) var(--spacing-md);
  vertical-align: top;
  border-top: 1px solid var(--color-divider);
}

.table thead th {
  vertical-align: bottom;
  border-bottom: 2px solid var(--color-divider);
  background-color: rgba(0, 0, 0, 0.02);
  font-weight: 500;
  text-align: left;
}

.table tbody + tbody {
  border-top: 2px solid var(--color-divider);
}

.table-sm th,
.table-sm td {
  padding: var(--spacing-xs) var(--spacing-sm);
}

.table-bordered {
  border: 1px solid var(--color-divider);
}

.table-bordered th,
.table-bordered td {
  border: 1px solid var(--color-divider);
}

.table-hover tbody tr:hover {
  background-color: rgba(0, 0, 0, 0.04);
}

.table-striped tbody tr:nth-of-type(odd) {
  background-color: rgba(0, 0, 0, 0.02);
}

/* Navigation */
.nav {
  display: flex;
  flex-wrap: wrap;
  padding-left: 0;
  margin-bottom: 0;
  list-style: none;
}

.nav-link {
  display: block;
  padding: var(--spacing-md) var(--spacing-lg);
  text-decoration: none;
  transition: color var(--transition-fast) ease;
}

.nav-tabs {
  border-bottom: 1px solid var(--color-divider);
}

.nav-tabs .nav-link {
  margin-bottom: -1px;
  border: 1px solid transparent;
  border-top-left-radius: var(--border-radius-md);
  border-top-right-radius: var(--border-radius-md);
}

.nav-tabs .nav-link:hover, .nav-tabs .nav-link:focus {
  border-color: var(--color-divider) var(--color-divider) var(--color-divider);
}

.nav-tabs .nav-link.active,
.nav-tabs .nav-item.show .nav-link {
  color: var(--color-primary);
  background-color: var(--color-surface);
  border-color: var(--color-divider) var(--color-divider) var(--color-surface);
}

/* Sidebar navigation */
.sidebar-nav {
  list-style: none;
  padding: 0;
  margin: 0;
}

.sidebar-nav-item {
  border-bottom: 1px solid var(--color-divider);
}

.sidebar-nav-link {
  display: flex;
  align-items: center;
  padding: var(--spacing-md) var(--spacing-lg);
  color: var(--color-text-primary);
  text-decoration: none;
  transition: background-color var(--transition-fast) ease;
}

.sidebar-nav-link:hover {
  background-color: rgba(0, 0, 0, 0.04);
  text-decoration: none;
}

.sidebar-nav-link.active {
  background-color: rgba(63, 81, 181, 0.1);
  color: var(--color-primary);
  border-left: 3px solid var(--color-primary);
}

.sidebar-nav-icon {
  margin-right: var(--spacing-sm);
  width: 20px;
  height: 20px;
}

/* KPI Cards */
.kpi-cards {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
  flex-wrap: wrap;
}

.kpi-card {
  background-color: var(--color-surface);
  border-radius: var(--border-radius-md);
  padding: var(--spacing-md) var(--spacing-lg);
  box-shadow: var(--shadow-sm);
  flex: 1;
  min-width: 200px;
  transition: transform var(--transition-normal) ease, box-shadow var(--transition-normal) ease;
}

.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}

.kpi-card-title {
  font-size: var(--font-size-sm);
  color: var(--color-text-secondary);
  margin-bottom: var(--spacing-xs);
}

.kpi-card-value {
  font-size: var(--font-size-2xl);
  font-weight: 500;
  margin-bottom: var(--spacing-xs);
}

.kpi-card-change {
  font-size: var(--font-size-sm);
  display: flex;
  align-items: center;
}

.kpi-card-change.positive {
  color: var(--color-success);
}

.kpi-card-change.negative {
  color: var(--color-error);
}

/* Loading Spinners */
.spinner {
  display: inline-block;
  width: 2rem;
  height: 2rem;
  vertical-align: text-bottom;
  border: 0.25em solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spinner-border 0.75s linear infinite;
}

.spinner-sm {
  width: 1rem;
  height: 1rem;
  border-width: 0.2em;
}

@keyframes spinner-border {
  to { transform: rotate(360deg); }
}

/* Alerts */
.alert {
  position: relative;
  padding: var(--spacing-md) var(--spacing-lg);
  margin-bottom: var(--spacing-md);
  border: 1px solid transparent;
  border-radius: var(--border-radius-md);
}

.alert-primary {
  color: #004085;
  background-color: #cce5ff;
  border-color: #b8daff;
}

.alert-secondary {
  color: #383d41;
  background-color: #e2e3e5;
  border-color: #d6d8db;
}

.alert-success {
  color: #155724;
  background-color: #d4edda;
  border-color: #c3e6cb;
}

.alert-danger {
  color: #721c24;
  background-color: #f8d7da;
  border-color: #f5c6cb;
}

.alert-warning {
  color: #856404;
  background-color: #fff3cd;
  border-color: #ffeeba;
}

.alert-info {
  color: #0c5460;
  background-color: #d1ecf1;
  border-color: #bee5eb;
}

/* Badge */
.badge {
  display: inline-block;
  padding: 0.25em 0.4em;
  font-size: 75%;
  font-weight: 700;
  line-height: 1;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: var(--border-radius-full);
}

.badge-primary {
  color: #fff;
  background-color: var(--color-primary);
}

.badge-secondary {
  color: #fff;
  background-color: var(--color-secondary);
}

.badge-success {
  color: #fff;
  background-color: var(--color-success);
}

.badge-danger {
  color: #fff;
  background-color: var(--color-error);
}

.badge-warning {
  color: #212529;
  background-color: var(--color-warning);
}

.badge-info {
  color: #fff;
  background-color: var(--color-info);
}

/* Dark Mode */
.theme-dark {
  --color-text-primary: #e0e0e0;
  --color-text-secondary: #adb5bd;
  --color-text-disabled: #6c757d;
  --color-background: #121212;
  --color-surface: #1e1e1e;
  --color-divider: #2d2d2d;
  
  /* Adjust primary and secondary colors for dark mode */
  --color-primary-light: #9fa8da;
  --color-secondary-light: #ff80ab;
}

.theme-dark .table thead th {
  background-color: rgba(255, 255, 255, 0.05);
}

.theme-dark .table-striped tbody tr:nth-of-type(odd) {
  background-color: rgba(255, 255, 255, 0.05);
}

.theme-dark .table-hover tbody tr:hover {
  background-color: rgba(255, 255, 255, 0.075);
}

.theme-dark .sidebar-nav-link:hover {
  background-color: rgba(255, 255, 255, 0.05);
}

.theme-dark .sidebar-nav-link.active {
  background-color: rgba(63, 81, 181, 0.2);
}

.theme-dark .form-control,
.theme-dark .form-select {
  background-color: #2d2d2d;
  border-color: #444;
  color: var(--color-text-primary);
}

.theme-dark .form-control:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 0.2rem rgba(63, 81, 181, 0.25);
}

/* Responsive Utilities */
@media (max-width: 992px) {
  .dashboard-sidebar {
    transform: translateX(-250px);
  }
  
  .dashboard-content {
    margin-left: 0;
  }
  
  .dashboard-sidebar.show {
    transform: translateX(0);
  }
  
  .widget-grid {
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  }
}

@media (max-width: 768px) {
  .container {
    padding-right: var(--spacing-md);
    padding-left: var(--spacing-md);
  }
  
  .dashboard-header {
    padding: var(--spacing-sm) var(--spacing-md);
  }
  
  .dashboard-content {
    padding: var(--spacing-md);
  }
  
  .widget-grid {
    grid-template-columns: 1fr;
  }
  
  .kpi-cards {
    flex-direction: column;
  }
  
  .kpi-card {
    width: 100%;
  }
  
  .d-none-mobile {
    display: none !important;
  }
}

/* Utility Classes */
.d-flex { display: flex !important; }
.d-inline-flex { display: inline-flex !important; }
.flex-row { flex-direction: row !important; }
.flex-column { flex-direction: column !important; }
.justify-content-start { justify-content: flex-start !important; }
.justify-content-end { justify-content: flex-end !important; }
.justify-content-center { justify-content: center !important; }
.justify-content-between { justify-content: space-between !important; }
.justify-content-around { justify-content: space-around !important; }
.align-items-start { align-items: flex-start !important; }
.align-items-end { align-items: flex-end !important; }
.align-items-center { align-items: center !important; }
.align-items-baseline { align-items: baseline !important; }
.align-items-stretch { align-items: stretch !important; }
.flex-grow-0 { flex-grow: 0 !important; }
.flex-grow-1 { flex-grow: 1 !important; }
.flex-shrink-0 { flex-shrink: 0 !important; }
.flex-shrink-1 { flex-shrink: 1 !important; }
.flex-wrap { flex-wrap: wrap !important; }
.flex-nowrap { flex-wrap: nowrap !important; }

.m-0 { margin: 0 !important; }
.mt-0 { margin-top: 0 !important; }
.mr-0 { margin-right: 0 !important; }
.mb-0 { margin-bottom: 0 !important; }
.ml-0 { margin-left: 0 !important; }
.m-1 { margin: var(--spacing-xs) !important; }
.mt-1 { margin-top: var(--spacing-xs) !important; }
.mr-1 { margin-right: var(--spacing-xs) !important; }
.mb-1 { margin-bottom: var(--spacing-xs) !important; }
.ml-1 { margin-left: var(--spacing-xs) !important; }
.m-2 { margin: var(--spacing-sm) !important; }
.mt-2 { margin-top: var(--spacing-sm) !important; }
.mr-2 { margin-right: var(--spacing-sm) !important; }
.mb-2 { margin-bottom: var(--spacing-sm) !important; }