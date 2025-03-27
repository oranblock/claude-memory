# data_processor.py - Advanced data analysis with pandas and scikit-learn
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