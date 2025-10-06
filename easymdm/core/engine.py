"""
Main MDM Engine for EasyMDM Advanced.
Orchestrates the entire Master Data Management process.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

# Rich for better console output
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
    from rich.table import Table
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .config import MDMConfig
from ..database.connector import DatabaseConnector
from ..similarity.matcher import SimilarityMatcher
from ..clustering.clusterer import RecordClusterer, ClusteringResult
from ..survivorship.resolver import SurvivorshipResolver, SurvivorshipResult
from .blocking import BlockingProcessor

logger = logging.getLogger(__name__)


@dataclass
class MDMResult:
    """Result of MDM processing."""
    golden_records: pd.DataFrame
    processing_stats: Dict[str, Any]
    detailed_results: Dict[str, Any]
    execution_time: float
    output_files: List[str]


class PerformanceMonitor:
    """Monitor performance metrics during MDM processing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        with self.lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.metrics[f"{operation}_duration"] = duration
                del self.start_times[operation]
                return duration
            return 0.0
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a custom metric."""
        with self.lock:
            self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            return self.metrics.copy()


class MDMEngine:
    """
    Main Master Data Management Engine.
    
    This class orchestrates the entire MDM process including:
    - Data loading from various sources
    - Blocking (candidate pair generation)
    - Similarity computation
    - Record clustering
    - Survivorship resolution
    - Golden record generation
    - Results export
    """
    
    def __init__(self, config: MDMConfig):
        """
        Initialize MDM Engine.
        
        Args:
            config: MDM configuration object
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.database_connector = DatabaseConnector.create_connector(config.source)
        self.similarity_matcher = SimilarityMatcher(
            config.similarity,
            n_jobs=config.n_jobs,
            use_cache=True
        )
        self.record_clusterer = RecordClusterer(
            config.thresholds,
            algorithm='network'  # Default algorithm
        )
        self.survivorship_resolver = SurvivorshipResolver(
            config.survivorship_rules,
            config.priority_conditions,
            config.unique_id_columns
        )
        self.blocking_processor = BlockingProcessor(config.blocking)
        
        # Rich console for better output
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        logger.info("MDM Engine initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup handlers
        handlers = []
        
        if RICH_AVAILABLE:
            # Use Rich handler for better console output
            rich_handler = RichHandler(console=Console(), show_time=False)
            rich_handler.setFormatter(formatter)
            handlers.append(rich_handler)
        else:
            # Use standard console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)
        
        # Add file handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process(self) -> MDMResult:
        """
        Execute the complete MDM process.
        
        Returns:
            MDMResult containing golden records and processing statistics
        """
        start_time = time.time()
        
        if self.console:
            self.console.print("[bold blue]Starting MDM Processing...[/bold blue]")
        
        try:
            # Step 1: Load data
            self.performance_monitor.start_timer("data_loading")
            df = self._load_data()
            self.performance_monitor.end_timer("data_loading")
            self.performance_monitor.add_metric("input_records", len(df))
            
            # Step 2: Blocking (generate candidate pairs)
            self.performance_monitor.start_timer("blocking")
            candidate_pairs = self._generate_candidate_pairs(df)
            self.performance_monitor.end_timer("blocking")
            self.performance_monitor.add_metric("candidate_pairs", len(candidate_pairs))
            
            # Step 3: Compute similarities
            self.performance_monitor.start_timer("similarity_computation")
            similarities = self._compute_similarities(df, candidate_pairs)
            self.performance_monitor.end_timer("similarity_computation")
            
            # Step 4: Cluster records
            self.performance_monitor.start_timer("clustering")
            clustering_result = self._cluster_records(similarities)
            self.performance_monitor.end_timer("clustering")
            self.performance_monitor.add_metric("clusters_found", len(clustering_result.clusters))
            
            # Step 5: Resolve survivorship and create golden records
            self.performance_monitor.start_timer("survivorship")
            golden_records = self._resolve_survivorship(df, clustering_result)
            self.performance_monitor.end_timer("survivorship")
            self.performance_monitor.add_metric("golden_records", len(golden_records))
            
            # Step 6: Export results
            self.performance_monitor.start_timer("export")
            output_files = self._export_results(
                golden_records, similarities, clustering_result, df
            )
            self.performance_monitor.end_timer("export")
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Prepare results
            processing_stats = self.performance_monitor.get_metrics()
            processing_stats['total_execution_time'] = total_time
            
            detailed_results = {
                'similarities': similarities,
                'clustering_result': clustering_result,
                'input_data': df
            }
            
            if self.console:
                self._print_summary(processing_stats)
            
            return MDMResult(
                golden_records=golden_records,
                processing_stats=processing_stats,
                detailed_results=detailed_results,
                execution_time=total_time,
                output_files=output_files
            )
            
        except Exception as e:
            logger.error(f"MDM processing failed: {e}")
            raise
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from configured source."""
        logger.info("Loading data from source...")
        
        try:
            self.database_connector.connect()
            df = self.database_connector.load_data()
            
            if df.empty:
                raise ValueError("No data loaded from source")
            
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Validate required columns exist
            missing_columns = []
            for config in self.config.similarity:
                if config.column not in df.columns:
                    missing_columns.append(config.column)
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        finally:
            self.database_connector.disconnect()
    
    def _generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate candidate pairs using blocking."""
        logger.info("Generating candidate pairs...")
        
        candidate_pairs = self.blocking_processor.process_blocking(df)
        
        logger.info(f"Generated {len(candidate_pairs)} candidate pairs")
        return candidate_pairs
    
    def _compute_similarities(self, df: pd.DataFrame, 
                            candidate_pairs: pd.MultiIndex) -> pd.DataFrame:
        """Compute similarities for candidate pairs."""
        logger.info("Computing similarities...")
        
        if len(candidate_pairs) == 0:
            logger.warning("No candidate pairs to process")
            return pd.DataFrame()
        
        similarities = self.similarity_matcher.compute_similarities(df, candidate_pairs)
        
        # Apply thresholds
        similarities = self._apply_thresholds(similarities)
        
        logger.info(f"Computed similarities for {len(similarities)} pairs")
        return similarities
    
    def _apply_thresholds(self, similarities: pd.DataFrame) -> pd.DataFrame:
        """Apply threshold configuration to categorize matches."""
        if 'match_category' in similarities.columns:
            return similarities
        
        similarities = similarities.copy()
        
        def categorize_score(score):
            if score >= self.config.thresholds.auto_merge:
                return 'auto_merge'
            elif score >= self.config.thresholds.review:
                return 'review'
            elif score <= self.config.thresholds.definite_no_match:
                return 'definite_no_match'
            else:
                return 'non_match'
        
        if 'overall_score' in similarities.columns:
            similarities['match_category'] = similarities['overall_score'].apply(categorize_score)
        else:
            similarities['overall_score'] = 0.0
            similarities['match_category'] = 'non_match'
        
        return similarities
    
    def _cluster_records(self, similarities: pd.DataFrame) -> ClusteringResult:
        """Cluster records based on similarities."""
        logger.info("Clustering records...")
        
        if similarities.empty:
            logger.warning("No similarities to cluster")
            return ClusteringResult(
                clusters=[],
                cluster_labels={},
                single_records=set(),
                clustering_stats={},
                processing_time=0.0
            )
        
        clustering_result = self.record_clusterer.cluster_records(similarities)
        
        logger.info(f"Found {len(clustering_result.clusters)} clusters")
        return clustering_result
    
    def _resolve_survivorship(self, df: pd.DataFrame, 
                            clustering_result: ClusteringResult) -> pd.DataFrame:
        """Resolve survivorship and create golden records."""
        logger.info("Resolving survivorship...")
        
        golden_records_list = []
        
        # Process clusters
        for i, cluster_ids in enumerate(clustering_result.clusters):
            cluster_df = df.loc[list(cluster_ids)]
            result = self.survivorship_resolver.resolve_cluster(cluster_df)
            
            # Add cluster metadata
            golden_record = result.golden_record.copy()
            golden_record['Record_ID'] = result.survivor_id or min(cluster_ids)
            golden_record['similar_record_ids'] = '|'.join(map(str, sorted(cluster_ids)))
            golden_record['logic'] = result.resolution_logic
            golden_record['confidence_score'] = result.confidence_score
            
            if result.metadata.get('unique_id'):
                golden_record['unique_id'] = result.metadata['unique_id']
            
            golden_records_list.append(golden_record)
        
        # Process single records
        for record_id in clustering_result.single_records:
            if record_id in df.index:
                single_record = df.loc[record_id].to_dict()
                single_record['Record_ID'] = record_id
                single_record['similar_record_ids'] = ''
                single_record['logic'] = 'single_record'
                single_record['confidence_score'] = 1.0
                
                # Generate unique ID if configured
                if self.config.unique_id_columns:
                    unique_id = self.survivorship_resolver._generate_unique_id(single_record)
                    single_record['unique_id'] = unique_id
                
                golden_records_list.append(single_record)
        
        if golden_records_list:
            golden_df = pd.DataFrame(golden_records_list)
        else:
            golden_df = pd.DataFrame()
        
        logger.info(f"Created {len(golden_df)} golden records")
        return golden_df
    
    def _export_results(self, golden_records: pd.DataFrame, 
                       similarities: pd.DataFrame,
                       clustering_result: ClusteringResult,
                       original_df: pd.DataFrame) -> List[str]:
        """Export results to files."""
        logger.info("Exporting results...")
        
        output_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Export golden records
        golden_records_file = os.path.join(
            self.config.output_path, 
            f"golden_records_{timestamp}.csv"
        )
        golden_records.to_csv(golden_records_file, index=False)
        output_files.append(golden_records_file)
        
        # Export detailed similarities (for review)
        if not similarities.empty:
            review_pairs = similarities[similarities['match_category'] == 'review']
            if not review_pairs.empty:
                review_file = os.path.join(
                    self.config.output_path,
                    f"review_pairs_{timestamp}.csv"
                )
                self._export_review_pairs(review_pairs, original_df, review_file)
                output_files.append(review_file)
        
        # Export processing summary
        summary_file = os.path.join(
            self.config.output_path,
            f"processing_summary_{timestamp}.txt"
        )
        self._export_processing_summary(
            summary_file, golden_records, similarities, clustering_result
        )
        output_files.append(summary_file)
        
        # Export detailed statistics
        stats_file = os.path.join(
            self.config.output_path,
            f"detailed_stats_{timestamp}.json"
        )
        self._export_detailed_stats(stats_file, clustering_result)
        output_files.append(stats_file)
        
        logger.info(f"Exported results to {len(output_files)} files")
        return output_files
    
    def _export_review_pairs(self, review_pairs: pd.DataFrame, 
                           original_df: pd.DataFrame, 
                           file_path: str) -> None:
        """Export pairs that require manual review."""
        review_data = []
        
        for idx, row in review_pairs.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                id1, id2 = idx
                
                # Get original records
                record1 = original_df.loc[id1].to_dict()
                record2 = original_df.loc[id2].to_dict()
                
                # Create review record
                review_record = {
                    'pair_id': f"{id1}_{id2}",
                    'record_id_1': id1,
                    'record_id_2': id2,
                    'similarity_score': row.get('overall_score', 0.0),
                    'match_category': row.get('match_category', 'review'),
                }
                
                # Add individual similarity scores
                for col in review_pairs.columns:
                    if col.endswith(('_sim', '_match')) and col != 'overall_score':
                        review_record[col] = row[col]
                
                # Add original record data with prefixes
                for key, value in record1.items():
                    review_record[f"record1_{key}"] = value
                
                for key, value in record2.items():
                    review_record[f"record2_{key}"] = value
                
                review_data.append(review_record)
        
        if review_data:
            review_df = pd.DataFrame(review_data)
            review_df.to_csv(file_path, index=False)
    
    def _export_processing_summary(self, file_path: str,
                                 golden_records: pd.DataFrame,
                                 similarities: pd.DataFrame,
                                 clustering_result: ClusteringResult) -> None:
        """Export processing summary to text file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("MDM PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config.source.type} source\n\n")
            
            # Performance metrics
            metrics = self.performance_monitor.get_metrics()
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            for metric, value in metrics.items():
                if metric.endswith('_duration'):
                    f.write(f"{metric}: {value:.2f} seconds\n")
                else:
                    f.write(f"{metric}: {value}\n")
            f.write("\n")
            
            # Clustering statistics
            f.write("CLUSTERING RESULTS:\n")
            f.write("-" * 20 + "\n")
            for key, value in clustering_result.clustering_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Match categories
            if not similarities.empty and 'match_category' in similarities.columns:
                f.write("MATCH CATEGORIES:\n")
                f.write("-" * 20 + "\n")
                category_counts = similarities['match_category'].value_counts()
                for category, count in category_counts.items():
                    f.write(f"{category}: {count}\n")
                f.write("\n")
            
            # Golden records summary
            f.write("GOLDEN RECORDS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total golden records: {len(golden_records)}\n")
            
            if 'logic' in golden_records.columns:
                logic_counts = golden_records['logic'].value_counts()
                f.write("\nResolution logic distribution:\n")
                for logic, count in logic_counts.items():
                    f.write(f"  {logic}: {count}\n")
    
    def _export_detailed_stats(self, file_path: str, 
                             clustering_result: ClusteringResult) -> None:
        """Export detailed statistics to JSON file."""
        import json
        
        stats = {
            'performance_metrics': self.performance_monitor.get_metrics(),
            'clustering_stats': clustering_result.clustering_stats,
            'configuration': {
                'source_type': self.config.source.type,
                'similarity_methods': [s.method for s in self.config.similarity],
                'thresholds': {
                    'review': self.config.thresholds.review,
                    'auto_merge': self.config.thresholds.auto_merge,
                    'definite_no_match': self.config.thresholds.definite_no_match,
                },
                'survivorship_strategies': [r.strategy for r in self.config.survivorship_rules],
                'priority_conditions': len(self.config.priority_conditions),
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print processing summary using Rich if available."""
        if not self.console:
            return
        
        # Create summary table
        table = Table(title="MDM Processing Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        for key, value in stats.items():
            if key.endswith('_duration'):
                table.add_row(key.replace('_', ' ').title(), f"{value:.2f}s")
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def test_configuration(self) -> Dict[str, bool]:
        """Test the current configuration."""
        logger.info("Testing MDM configuration...")
        
        test_results = {}
        
        # Test database connection
        try:
            test_results['database_connection'] = self.database_connector.test_connection()
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            test_results['database_connection'] = False
        
        # Test similarity methods
        try:
            # Create sample data for testing
            sample_data = pd.DataFrame({
                'test_col': ['test1', 'test2']
            })
            
            for config in self.config.similarity:
                method_name = f"similarity_{config.method}"
                try:
                    score = self.similarity_matcher.compute_single_similarity(
                        'test1', 'test2', config.method
                    )
                    test_results[method_name] = True
                except Exception as e:
                    logger.error(f"Similarity method {config.method} test failed: {e}")
                    test_results[method_name] = False
        
        except Exception as e:
            logger.error(f"Similarity testing failed: {e}")
            test_results['similarity_methods'] = False
        
        # Test output directory
        try:
            os.makedirs(self.config.output_path, exist_ok=True)
            test_results['output_directory'] = os.path.exists(self.config.output_path)
        except Exception as e:
            logger.error(f"Output directory test failed: {e}")
            test_results['output_directory'] = False
        
        logger.info(f"Configuration test completed: {sum(test_results.values())}/{len(test_results)} passed")
        return test_results
    
    def get_data_profile(self) -> Dict[str, Any]:
        """Get a profile of the input data."""
        logger.info("Profiling input data...")
        
        try:
            self.database_connector.connect()
            df = self.database_connector.load_data()
            
            profile = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'null_counts': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
            }
            
            # Add column statistics
            column_stats = {}
            for column in df.columns:
                stats = {
                    'unique_values': df[column].nunique(),
                    'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
                }
                
                if df[column].dtype in ['int64', 'float64']:
                    stats.update({
                        'min': df[column].min(),
                        'max': df[column].max(),
                        'mean': df[column].mean(),
                    })
                elif df[column].dtype == 'object':
                    stats.update({
                        'avg_length': df[column].astype(str).str.len().mean(),
                        'max_length': df[column].astype(str).str.len().max(),
                    })
                
                column_stats[column] = stats
            
            profile['column_statistics'] = column_stats
            
            return profile
            
        except Exception as e:
            logger.error(f"Data profiling failed: {e}")
            return {}
        finally:
            self.database_connector.disconnect()


def create_mdm_engine(config: MDMConfig) -> MDMEngine:
    """Factory function to create an MDM Engine instance."""
    return MDMEngine(config)