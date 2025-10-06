"""
Advanced clustering for EasyMDM.
Implements multiple clustering algorithms for record linkage.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Set, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.config import ThresholdConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    clusters: List[Set[int]]
    cluster_labels: Dict[int, int]
    single_records: Set[int]
    clustering_stats: Dict[str, Any]
    processing_time: float


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    def __init__(self, threshold_config: ThresholdConfig, **kwargs):
        self.threshold_config = threshold_config
        self.options = kwargs
        
    @abstractmethod
    def cluster(self, similarities: pd.DataFrame, 
                record_ids: Optional[Set[int]] = None) -> ClusteringResult:
        """Perform clustering on similarity data."""
        pass
    
    @abstractmethod
    def get_cluster_stats(self, result: ClusteringResult) -> Dict[str, Any]:
        """Get clustering statistics."""
        pass


class NetworkBasedClusterer(BaseClusterer):
    """
    Network-based clustering using graph connectivity.
    This is similar to your original approach but with enhancements.
    """
    
    def cluster(self, similarities: pd.DataFrame, 
                record_ids: Optional[Set[int]] = None) -> ClusteringResult:
        """
        Cluster records using network connectivity.
        
        Args:
            similarities: DataFrame with similarity scores and match categories
            record_ids: Optional set of record IDs to cluster
            
        Returns:
            ClusteringResult object
        """
        start_time = time.time()
        
        # Filter for auto-merge pairs
        auto_merge_pairs = similarities[
            similarities['match_category'] == 'auto_merge'
        ]
        
        if auto_merge_pairs.empty:
            # No clusters, all records are single
            if record_ids is None:
                # Extract all unique record IDs from similarities index
                all_ids = set()
                for idx in similarities.index:
                    all_ids.add(idx[0])
                    all_ids.add(idx[1])
                record_ids = all_ids
            
            return ClusteringResult(
                clusters=[],
                cluster_labels={},
                single_records=record_ids,
                clustering_stats={'num_clusters': 0, 'num_single_records': len(record_ids)},
                processing_time=time.time() - start_time
            )
        
        # Create graph from auto-merge pairs
        G = nx.Graph()
        
        for idx, row in auto_merge_pairs.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                first, second = idx
            else:
                # Handle MultiIndex
                first, second = idx[0], idx[1] if hasattr(idx, '__len__') else (idx, idx)
            
            # Add edge with weight as similarity score
            weight = row.get('overall_score', 1.0)
            G.add_edge(first, second, weight=weight)
        
        # Find connected components (clusters)
        connected_components = list(nx.connected_components(G))
        
        # Create cluster labels
        cluster_labels = {}
        clusters = []
        
        for cluster_id, component in enumerate(connected_components):
            cluster_set = set(component)
            clusters.append(cluster_set)
            
            for record_id in component:
                cluster_labels[record_id] = cluster_id
        
        # Identify single records
        clustered_record_ids = set(cluster_labels.keys())
        
        if record_ids is None:
            # Extract all record IDs from review pairs and auto-merge pairs
            all_ids = set()
            for idx in similarities.index:
                if isinstance(idx, tuple) and len(idx) == 2:
                    all_ids.add(idx[0])
                    all_ids.add(idx[1])
                else:
                    all_ids.add(idx[0] if hasattr(idx, '__len__') else idx)
                    if hasattr(idx, '__len__') and len(idx) > 1:
                        all_ids.add(idx[1])
            record_ids = all_ids
        
        single_records = record_ids - clustered_record_ids
        
        # Calculate statistics
        stats = {
            'num_clusters': len(clusters),
            'num_clustered_records': len(clustered_record_ids),
            'num_single_records': len(single_records),
            'average_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
            'max_cluster_size': max([len(c) for c in clusters]) if clusters else 0,
            'min_cluster_size': min([len(c) for c in clusters]) if clusters else 0,
        }
        
        processing_time = time.time() - start_time
        
        logger.info(f"Network clustering completed: {stats['num_clusters']} clusters, "
                   f"{stats['num_single_records']} single records in {processing_time:.2f}s")
        
        return ClusteringResult(
            clusters=clusters,
            cluster_labels=cluster_labels,
            single_records=single_records,
            clustering_stats=stats,
            processing_time=processing_time
        )
    
    def get_cluster_stats(self, result: ClusteringResult) -> Dict[str, Any]:
        """Get detailed clustering statistics."""
        stats = result.clustering_stats.copy()
        
        if result.clusters:
            cluster_sizes = [len(c) for c in result.clusters]
            stats.update({
                'cluster_size_distribution': {
                    'mean': np.mean(cluster_sizes),
                    'median': np.median(cluster_sizes),
                    'std': np.std(cluster_sizes),
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes),
                },
                'cluster_size_histogram': dict(zip(*np.unique(cluster_sizes, return_counts=True)))
            })
        
        return stats


class HierarchicalClusterer(BaseClusterer):
    """
    Hierarchical clustering using similarity scores.
    Requires scikit-learn.
    """
    
    def __init__(self, threshold_config: ThresholdConfig, **kwargs):
        super().__init__(threshold_config, **kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for hierarchical clustering")
        
        self.linkage = kwargs.get('linkage', 'ward')
        self.distance_threshold = kwargs.get('distance_threshold', 1 - threshold_config.auto_merge)
    
    def cluster(self, similarities: pd.DataFrame, 
                record_ids: Optional[Set[int]] = None) -> ClusteringResult:
        """
        Perform hierarchical clustering.
        
        Args:
            similarities: DataFrame with similarity scores
            record_ids: Optional set of record IDs to cluster
            
        Returns:
            ClusteringResult object
        """
        start_time = time.time()
        
        # Create distance matrix from similarities
        distance_matrix, id_mapping = self._create_distance_matrix(similarities, record_ids)
        
        if distance_matrix.shape[0] < 2:
            # Too few records to cluster
            return ClusteringResult(
                clusters=[],
                cluster_labels={},
                single_records=set(id_mapping.values()) if record_ids is None else record_ids,
                clustering_stats={'num_clusters': 0},
                processing_time=time.time() - start_time
            )
        
        # Perform hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric='precomputed'
        )
        
        cluster_assignments = clusterer.fit_predict(distance_matrix)
        
        # Convert back to original record IDs
        clusters = defaultdict(set)
        cluster_labels = {}
        
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        for matrix_idx, cluster_id in enumerate(cluster_assignments):
            original_id = reverse_mapping[matrix_idx]
            clusters[cluster_id].add(original_id)
            cluster_labels[original_id] = cluster_id
        
        # Convert to list of sets
        final_clusters = list(clusters.values())
        
        # Identify single records
        clustered_records = set(cluster_labels.keys())
        all_records = set(id_mapping.keys()) if record_ids is None else record_ids
        single_records = all_records - clustered_records
        
        # Calculate statistics
        stats = {
            'num_clusters': len(final_clusters),
            'num_clustered_records': len(clustered_records),
            'num_single_records': len(single_records),
            'distance_threshold': self.distance_threshold,
            'linkage': self.linkage,
        }
        
        processing_time = time.time() - start_time
        
        logger.info(f"Hierarchical clustering completed: {stats['num_clusters']} clusters "
                   f"in {processing_time:.2f}s")
        
        return ClusteringResult(
            clusters=final_clusters,
            cluster_labels=cluster_labels,
            single_records=single_records,
            clustering_stats=stats,
            processing_time=processing_time
        )
    
    def _create_distance_matrix(self, similarities: pd.DataFrame, 
                               record_ids: Optional[Set[int]]) -> Tuple[np.ndarray, Dict[int, int]]:
        """Create distance matrix from similarities."""
        # Get all unique record IDs
        all_ids = set()
        for idx in similarities.index:
            if isinstance(idx, tuple) and len(idx) == 2:
                all_ids.add(idx[0])
                all_ids.add(idx[1])
        
        if record_ids is not None:
            all_ids = all_ids.intersection(record_ids)
        
        id_list = sorted(list(all_ids))
        id_mapping = {record_id: i for i, record_id in enumerate(id_list)}
        
        n = len(id_list)
        distance_matrix = np.ones((n, n))  # Initialize with maximum distance
        
        # Fill diagonal with zeros
        np.fill_diagonal(distance_matrix, 0)
        
        # Fill matrix with similarity-based distances
        for idx, row in similarities.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                id1, id2 = idx
            else:
                continue
            
            if id1 in id_mapping and id2 in id_mapping:
                i, j = id_mapping[id1], id_mapping[id2]
                similarity = row.get('overall_score', 0.0)
                distance = 1.0 - similarity  # Convert similarity to distance
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix
        
        return distance_matrix, id_mapping
    
    def get_cluster_stats(self, result: ClusteringResult) -> Dict[str, Any]:
        """Get detailed clustering statistics."""
        return result.clustering_stats


class DBSCANClusterer(BaseClusterer):
    """
    DBSCAN clustering using similarity scores.
    Requires scikit-learn.
    """
    
    def __init__(self, threshold_config: ThresholdConfig, **kwargs):
        super().__init__(threshold_config, **kwargs)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for DBSCAN clustering")
        
        self.eps = kwargs.get('eps', 1 - threshold_config.auto_merge)
        self.min_samples = kwargs.get('min_samples', 2)
    
    def cluster(self, similarities: pd.DataFrame, 
                record_ids: Optional[Set[int]] = None) -> ClusteringResult:
        """
        Perform DBSCAN clustering.
        
        Args:
            similarities: DataFrame with similarity scores
            record_ids: Optional set of record IDs to cluster
            
        Returns:
            ClusteringResult object
        """
        start_time = time.time()
        
        # Create distance matrix
        distance_matrix, id_mapping = self._create_distance_matrix(similarities, record_ids)
        
        if distance_matrix.shape[0] < 2:
            # Too few records to cluster
            return ClusteringResult(
                clusters=[],
                cluster_labels={},
                single_records=set(id_mapping.values()) if record_ids is None else record_ids,
                clustering_stats={'num_clusters': 0},
                processing_time=time.time() - start_time
            )
        
        # Perform DBSCAN clustering
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        cluster_assignments = clusterer.fit_predict(distance_matrix)
        
        # Convert back to original record IDs
        clusters = defaultdict(set)
        cluster_labels = {}
        noise_points = set()
        
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        for matrix_idx, cluster_id in enumerate(cluster_assignments):
            original_id = reverse_mapping[matrix_idx]
            
            if cluster_id == -1:  # Noise point
                noise_points.add(original_id)
            else:
                clusters[cluster_id].add(original_id)
                cluster_labels[original_id] = cluster_id
        
        # Convert to list of sets
        final_clusters = list(clusters.values())
        
        # Single records include noise points and unclustered records
        clustered_records = set(cluster_labels.keys())
        all_records = set(id_mapping.keys()) if record_ids is None else record_ids
        single_records = (all_records - clustered_records) | noise_points
        
        # Calculate statistics
        stats = {
            'num_clusters': len(final_clusters),
            'num_clustered_records': len(clustered_records),
            'num_single_records': len(single_records),
            'num_noise_points': len(noise_points),
            'eps': self.eps,
            'min_samples': self.min_samples,
        }
        
        processing_time = time.time() - start_time
        
        logger.info(f"DBSCAN clustering completed: {stats['num_clusters']} clusters "
                   f"in {processing_time:.2f}s")
        
        return ClusteringResult(
            clusters=final_clusters,
            cluster_labels=cluster_labels,
            single_records=single_records,
            clustering_stats=stats,
            processing_time=processing_time
        )
    
    def _create_distance_matrix(self, similarities: pd.DataFrame, 
                               record_ids: Optional[Set[int]]) -> Tuple[np.ndarray, Dict[int, int]]:
        """Create distance matrix from similarities (same as HierarchicalClusterer)."""
        # Get all unique record IDs
        all_ids = set()
        for idx in similarities.index:
            if isinstance(idx, tuple) and len(idx) == 2:
                all_ids.add(idx[0])
                all_ids.add(idx[1])
        
        if record_ids is not None:
            all_ids = all_ids.intersection(record_ids)
        
        id_list = sorted(list(all_ids))
        id_mapping = {record_id: i for i, record_id in enumerate(id_list)}
        
        n = len(id_list)
        distance_matrix = np.ones((n, n))
        
        # Fill diagonal with zeros
        np.fill_diagonal(distance_matrix, 0)
        
        # Fill matrix with distances
        for idx, row in similarities.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                id1, id2 = idx
            else:
                continue
            
            if id1 in id_mapping and id2 in id_mapping:
                i, j = id_mapping[id1], id_mapping[id2]
                similarity = row.get('overall_score', 0.0)
                distance = 1.0 - similarity
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix, id_mapping
    
    def get_cluster_stats(self, result: ClusteringResult) -> Dict[str, Any]:
        """Get detailed clustering statistics."""
        return result.clustering_stats


class RecordClusterer:
    """
    Main record clustering class with multiple algorithm support.
    """
    
    CLUSTERING_ALGORITHMS = {
        'network': NetworkBasedClusterer,
        'hierarchical': HierarchicalClusterer,
        'dbscan': DBSCANClusterer,
    }
    
    def __init__(self, threshold_config: ThresholdConfig, 
                 algorithm: str = 'network', **kwargs):
        """
        Initialize record clusterer.
        
        Args:
            threshold_config: Threshold configuration
            algorithm: Clustering algorithm to use
            **kwargs: Additional options for the clustering algorithm
        """
        self.threshold_config = threshold_config
        self.algorithm = algorithm.lower()
        self.options = kwargs
        
        if self.algorithm not in self.CLUSTERING_ALGORITHMS:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Initialize clusterer
        clusterer_class = self.CLUSTERING_ALGORITHMS[self.algorithm]
        self.clusterer = clusterer_class(threshold_config, **kwargs)
        
        logger.info(f"Initialized RecordClusterer with algorithm: {self.algorithm}")
    
    def cluster_records(self, similarities: pd.DataFrame, 
                       record_ids: Optional[Set[int]] = None) -> ClusteringResult:
        """
        Cluster records based on similarity scores.
        
        Args:
            similarities: DataFrame with similarity scores and match categories
            record_ids: Optional set of record IDs to consider for clustering
            
        Returns:
            ClusteringResult object
        """
        logger.info(f"Starting record clustering using {self.algorithm} algorithm")
        
        # Apply thresholds to categorize matches
        similarities = self._apply_thresholds(similarities)
        
        # Perform clustering
        result = self.clusterer.cluster(similarities, record_ids)
        
        logger.info(f"Clustering completed: {result.clustering_stats}")
        
        return result
    
    def _apply_thresholds(self, similarities: pd.DataFrame) -> pd.DataFrame:
        """Apply threshold configuration to categorize matches."""
        if 'match_category' in similarities.columns:
            # Categories already applied
            return similarities
        
        similarities = similarities.copy()
        
        # Apply thresholds
        def categorize_score(score):
            if score >= self.threshold_config.auto_merge:
                return 'auto_merge'
            elif score >= self.threshold_config.review:
                return 'review'
            elif score <= self.threshold_config.definite_no_match:
                return 'definite_no_match'
            else:
                return 'non_match'
        
        if 'overall_score' in similarities.columns:
            similarities['match_category'] = similarities['overall_score'].apply(categorize_score)
        else:
            # Use average of all similarity columns
            similarity_columns = [col for col in similarities.columns 
                                if col.endswith(('_sim', '_match')) and col != 'overall_score']
            
            if similarity_columns:
                similarities['overall_score'] = similarities[similarity_columns].mean(axis=1)
                similarities['match_category'] = similarities['overall_score'].apply(categorize_score)
            else:
                # No similarity columns found
                similarities['overall_score'] = 0.0
                similarities['match_category'] = 'non_match'
        
        return similarities
    
    def get_clustering_stats(self, result: ClusteringResult) -> Dict[str, Any]:
        """Get detailed clustering statistics."""
        base_stats = self.clusterer.get_cluster_stats(result)
        
        # Add algorithm-specific information
        base_stats['algorithm'] = self.algorithm
        base_stats['processing_time'] = result.processing_time
        
        return base_stats
    
    def analyze_cluster_quality(self, result: ClusteringResult, 
                               similarities: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality of clustering results.
        
        Args:
            result: ClusteringResult object
            similarities: Original similarity DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        if not result.clusters:
            quality_metrics['average_intra_cluster_similarity'] = 0.0
            quality_metrics['average_inter_cluster_similarity'] = 0.0
            quality_metrics['silhouette_score'] = 0.0
            return quality_metrics
        
        # Calculate intra-cluster similarity (similarity within clusters)
        intra_similarities = []
        
        for cluster in result.clusters:
            cluster_list = list(cluster)
            cluster_similarities = []
            
            for i, id1 in enumerate(cluster_list):
                for id2 in cluster_list[i+1:]:
                    # Find similarity score for this pair
                    pair_sim = self._find_pair_similarity(similarities, id1, id2)
                    if pair_sim is not None:
                        cluster_similarities.append(pair_sim)
            
            if cluster_similarities:
                intra_similarities.extend(cluster_similarities)
        
        # Calculate inter-cluster similarity (similarity between clusters)
        inter_similarities = []
        
        for i, cluster1 in enumerate(result.clusters):
            for cluster2 in result.clusters[i+1:]:
                for id1 in cluster1:
                    for id2 in cluster2:
                        pair_sim = self._find_pair_similarity(similarities, id1, id2)
                        if pair_sim is not None:
                            inter_similarities.append(pair_sim)
        
        quality_metrics['average_intra_cluster_similarity'] = np.mean(intra_similarities) if intra_similarities else 0.0
        quality_metrics['average_inter_cluster_similarity'] = np.mean(inter_similarities) if inter_similarities else 0.0
        
        # Simple silhouette-like score
        if intra_similarities and inter_similarities:
            quality_metrics['silhouette_score'] = (
                quality_metrics['average_intra_cluster_similarity'] - 
                quality_metrics['average_inter_cluster_similarity']
            )
        else:
            quality_metrics['silhouette_score'] = 0.0
        
        return quality_metrics
    
    def _find_pair_similarity(self, similarities: pd.DataFrame, 
                            id1: int, id2: int) -> Optional[float]:
        """Find similarity score for a specific pair of IDs."""
        for idx, row in similarities.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                pair_id1, pair_id2 = idx
                
                if (pair_id1 == id1 and pair_id2 == id2) or \
                   (pair_id1 == id2 and pair_id2 == id1):
                    return row.get('overall_score', 0.0)
        
        return None
    
    def compare_algorithms(self, similarities: pd.DataFrame, 
                          record_ids: Optional[Set[int]] = None,
                          algorithms: Optional[List[str]] = None) -> Dict[str, ClusteringResult]:
        """
        Compare different clustering algorithms on the same data.
        
        Args:
            similarities: DataFrame with similarity scores
            record_ids: Optional set of record IDs to cluster
            algorithms: List of algorithms to compare (default: all available)
            
        Returns:
            Dictionary mapping algorithm names to their results
        """
        if algorithms is None:
            algorithms = list(self.CLUSTERING_ALGORITHMS.keys())
        
        results = {}
        
        for algorithm in algorithms:
            try:
                logger.info(f"Comparing algorithm: {algorithm}")
                
                # Create clusterer for this algorithm
                clusterer_class = self.CLUSTERING_ALGORITHMS[algorithm]
                clusterer = clusterer_class(self.threshold_config, **self.options)
                
                # Apply thresholds
                similarities_with_categories = self._apply_thresholds(similarities)
                
                # Perform clustering
                result = clusterer.cluster(similarities_with_categories, record_ids)
                results[algorithm] = result
                
                logger.info(f"{algorithm}: {result.clustering_stats}")
                
            except Exception as e:
                logger.error(f"Failed to run algorithm {algorithm}: {e}")
                continue
        
        return results


def create_record_clusterer(threshold_config: ThresholdConfig, 
                          algorithm: str = 'network', 
                          **kwargs) -> RecordClusterer:
    """Factory function to create a RecordClusterer instance."""
    return RecordClusterer(threshold_config, algorithm, **kwargs)