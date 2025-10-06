"""
Blocking processor for EasyMDM Advanced.
Implements various blocking strategies for candidate pair generation.
"""

import pandas as pd
import numpy as np
from typing import List, Set, Tuple, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
import time
from itertools import combinations
from collections import defaultdict

# RecordLinkage for standard blocking
try:
    import recordlinkage
    RECORDLINKAGE_AVAILABLE = True
except ImportError:
    RECORDLINKAGE_AVAILABLE = False

# Fuzzy matching libraries
try:
    from fuzzywuzzy import fuzz
    import jellyfish
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

from ..core.config import BlockingConfig

logger = logging.getLogger(__name__)


class BaseBlockingStrategy(ABC):
    """Abstract base class for blocking strategies."""
    
    def __init__(self, config: BlockingConfig):
        self.config = config
        self.columns = config.columns
        self.threshold = config.threshold
        self.options = config.options or {}
        
    @abstractmethod
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate candidate pairs for comparison."""
        pass
    
    def create_blocking_key(self, df: pd.DataFrame) -> pd.Series:
        """Create blocking key by concatenating specified columns."""
        blocking_parts = []
        
        for column in self.columns:
            if column in df.columns:
                # Clean and normalize the column values
                cleaned = df[column].astype(str).str.lower().str.strip()
                # Remove extra whitespace
                cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
                blocking_parts.append(cleaned)
            else:
                logger.warning(f"Blocking column '{column}' not found in DataFrame")
                # Add empty strings for missing columns
                blocking_parts.append(pd.Series([''] * len(df), index=df.index))
        
        # Concatenate with separator
        separator = self.options.get('separator', ' ')
        blocking_key = blocking_parts[0] if len(blocking_parts) == 1 else \
                      blocking_parts[0].str.cat(blocking_parts[1:], sep=separator)
        
        return blocking_key


class ExactBlockingStrategy(BaseBlockingStrategy):
    """Exact match blocking strategy."""
    
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate pairs using exact blocking keys."""
        logger.info("Generating candidate pairs using exact blocking...")
        
        # Create blocking key
        blocking_key = self.create_blocking_key(df)
        
        # Group records by blocking key
        grouped = df.groupby(blocking_key).groups
        
        # Generate pairs within each block
        candidate_pairs = []
        
        for block_key, record_indices in grouped.items():
            if len(record_indices) > 1:
                # Generate all combinations within the block
                for pair in combinations(record_indices, 2):
                    candidate_pairs.append(tuple(sorted(pair)))
        
        # Remove duplicates and create MultiIndex
        unique_pairs = list(set(candidate_pairs))
        
        if unique_pairs:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                unique_pairs, names=['first', 'second']
            )
        else:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                [], names=['first', 'second']
            )
        
        logger.info(f"Generated {len(pairs_multiindex)} candidate pairs using exact blocking")
        return pairs_multiindex


class FuzzyBlockingStrategy(BaseBlockingStrategy):
    """Fuzzy blocking strategy using similarity thresholds."""
    
    def __init__(self, config: BlockingConfig):
        super().__init__(config)
        
        if not FUZZY_AVAILABLE:
            raise ImportError("fuzzywuzzy and jellyfish are required for fuzzy blocking")
        
        self.similarity_method = self.options.get('similarity_method', 'jaro_winkler')
        
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate pairs using fuzzy similarity."""
        logger.info("Generating candidate pairs using fuzzy blocking...")
        
        # Create blocking keys
        blocking_keys = self.create_blocking_key(df)
        
        # Remove empty keys
        valid_mask = blocking_keys.str.len() > 0
        valid_keys = blocking_keys[valid_mask]
        valid_indices = valid_keys.index.tolist()
        
        if len(valid_keys) < 2:
            logger.warning("Not enough valid blocking keys for fuzzy blocking")
            return pd.MultiIndex.from_tuples([], names=['first', 'second'])
        
        # Generate candidate pairs based on similarity
        candidate_pairs = []
        
        # Use optimized comparison for large datasets
        if len(valid_keys) > 10000:
            candidate_pairs = self._generate_pairs_optimized(valid_keys, valid_indices)
        else:
            candidate_pairs = self._generate_pairs_standard(valid_keys, valid_indices)
        
        # Create MultiIndex
        if candidate_pairs:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                candidate_pairs, names=['first', 'second']
            )
        else:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                [], names=['first', 'second']
            )
        
        logger.info(f"Generated {len(pairs_multiindex)} candidate pairs using fuzzy blocking")
        return pairs_multiindex
    
    def _generate_pairs_standard(self, blocking_keys: pd.Series, 
                               valid_indices: List[int]) -> List[Tuple[int, int]]:
        """Standard pairwise comparison for smaller datasets."""
        candidate_pairs = []
        
        for i, idx1 in enumerate(valid_indices):
            key1 = blocking_keys.iloc[i]
            
            for j in range(i + 1, len(valid_indices)):
                idx2 = valid_indices[j]
                key2 = blocking_keys.iloc[j]
                
                # Compute similarity
                similarity = self._compute_similarity(key1, key2)
                
                if similarity >= self.threshold:
                    pair = tuple(sorted([idx1, idx2]))
                    candidate_pairs.append(pair)
        
        return candidate_pairs
    
    def _generate_pairs_optimized(self, blocking_keys: pd.Series, 
                                valid_indices: List[int]) -> List[Tuple[int, int]]:
        """Optimized comparison using pre-filtering for larger datasets."""
        candidate_pairs = []
        
        # Create a dictionary for faster lookups
        key_to_indices = defaultdict(list)
        
        for i, idx in enumerate(valid_indices):
            key = blocking_keys.iloc[i]
            key_to_indices[key].append(idx)
        
        # First pass: exact matches
        for key, indices in key_to_indices.items():
            if len(indices) > 1:
                for pair in combinations(indices, 2):
                    candidate_pairs.append(tuple(sorted(pair)))
        
        # Second pass: fuzzy matches with optimization
        processed_keys = set()
        
        for i, idx1 in enumerate(valid_indices):
            key1 = blocking_keys.iloc[i]
            
            if key1 in processed_keys:
                continue
            
            # Only compare with keys that haven't been processed
            for j in range(i + 1, len(valid_indices)):
                idx2 = valid_indices[j]
                key2 = blocking_keys.iloc[j]
                
                if key2 in processed_keys:
                    continue
                
                # Skip if already exact match
                if key1 == key2:
                    continue
                
                # Quick pre-filter based on length difference
                if abs(len(key1) - len(key2)) / max(len(key1), len(key2)) > (1 - self.threshold):
                    continue
                
                # Compute similarity
                similarity = self._compute_similarity(key1, key2)
                
                if similarity >= self.threshold:
                    pair = tuple(sorted([idx1, idx2]))
                    candidate_pairs.append(pair)
            
            processed_keys.add(key1)
        
        return candidate_pairs
    
    def _compute_similarity(self, str1: str, str2: str) -> float:
        """Compute similarity between two strings."""
        if not str1 or not str2:
            return 0.0
        
        if str1 == str2:
            return 1.0
        
        try:
            if self.similarity_method == 'jaro_winkler':
                return jellyfish.jaro_winkler_similarity(str1, str2)
            elif self.similarity_method == 'jaro':
                return jellyfish.jaro_similarity(str1, str2)
            elif self.similarity_method == 'levenshtein':
                distance = jellyfish.levenshtein_distance(str1, str2)
                max_len = max(len(str1), len(str2))
                return 1.0 - (distance / max_len) if max_len > 0 else 0.0
            elif self.similarity_method == 'fuzzy_ratio':
                return fuzz.ratio(str1, str2) / 100.0
            elif self.similarity_method == 'fuzzy_token_sort':
                return fuzz.token_sort_ratio(str1, str2) / 100.0
            else:
                logger.warning(f"Unknown similarity method: {self.similarity_method}")
                return jellyfish.jaro_winkler_similarity(str1, str2)
                
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0


class SortedNeighborhoodStrategy(BaseBlockingStrategy):
    """Sorted Neighborhood blocking strategy."""
    
    def __init__(self, config: BlockingConfig):
        super().__init__(config)
        self.window_size = config.window_size or self.options.get('window_size', 10)
    
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate pairs using sorted neighborhood method."""
        logger.info("Generating candidate pairs using sorted neighborhood...")
        
        # Create blocking key
        blocking_key = self.create_blocking_key(df)
        
        # Create DataFrame with blocking keys for sorting
        temp_df = pd.DataFrame({
            'index': df.index,
            'blocking_key': blocking_key
        })
        
        # Remove empty keys
        temp_df = temp_df[temp_df['blocking_key'].str.len() > 0]
        
        if len(temp_df) < 2:
            logger.warning("Not enough valid records for sorted neighborhood")
            return pd.MultiIndex.from_tuples([], names=['first', 'second'])
        
        # Sort by blocking key
        temp_df_sorted = temp_df.sort_values('blocking_key').reset_index(drop=True)
        
        # Generate pairs within sliding window
        candidate_pairs = []
        
        for i in range(len(temp_df_sorted)):
            # Define window bounds
            window_start = i
            window_end = min(i + self.window_size, len(temp_df_sorted))
            
            # Generate pairs within the window
            for j in range(window_start + 1, window_end):
                idx1 = temp_df_sorted.iloc[i]['index']
                idx2 = temp_df_sorted.iloc[j]['index']
                
                pair = tuple(sorted([idx1, idx2]))
                candidate_pairs.append(pair)
        
        # Remove duplicates and create MultiIndex
        unique_pairs = list(set(candidate_pairs))
        
        if unique_pairs:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                unique_pairs, names=['first', 'second']
            )
        else:
            pairs_multiindex = pd.MultiIndex.from_tuples(
                [], names=['first', 'second']
            )
        
        logger.info(f"Generated {len(pairs_multiindex)} candidate pairs using sorted neighborhood")
        return pairs_multiindex


class RecordLinkageBlockingStrategy(BaseBlockingStrategy):
    """RecordLinkage library-based blocking strategy."""
    
    def __init__(self, config: BlockingConfig):
        super().__init__(config)
        
        if not RECORDLINKAGE_AVAILABLE:
            raise ImportError("recordlinkage library is required for this blocking strategy")
        
        self.blocking_method = self.options.get('method', 'block')
    
    def generate_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Generate pairs using recordlinkage indexers."""
        logger.info("Generating candidate pairs using recordlinkage...")
        
        try:
            # Initialize indexer
            indexer = recordlinkage.Index()
            
            if self.blocking_method == 'block':
                # Standard blocking on specified columns
                if len(self.columns) == 1:
                    indexer.block(self.columns[0])
                else:
                    # Multiple column blocking
                    for column in self.columns:
                        if column in df.columns:
                            indexer.block(column)
                        else:
                            logger.warning(f"Column '{column}' not found for blocking")
            
            elif self.blocking_method == 'sortedneighbourhood':
                # Sorted neighborhood indexing
                window_size = self.options.get('window_size', 10)
                if len(self.columns) >= 1 and self.columns[0] in df.columns:
                    indexer.sortedneighbourhood(self.columns[0], window=window_size)
                else:
                    logger.error("No valid columns for sorted neighbourhood blocking")
                    return pd.MultiIndex.from_tuples([], names=['first', 'second'])
            
            elif self.blocking_method == 'random':
                # Random sampling
                sample_size = self.options.get('sample_size', min(10000, len(df) * (len(df) - 1) // 2))
                indexer.random(sample_size)
            
            else:
                logger.warning(f"Unknown blocking method: {self.blocking_method}, using standard block")
                if self.columns and self.columns[0] in df.columns:
                    indexer.block(self.columns[0])
                else:
                    # Fallback to random
                    indexer.random(10000)
            
            # Generate candidate pairs
            candidate_pairs = indexer.index(df)
            
            logger.info(f"Generated {len(candidate_pairs)} candidate pairs using recordlinkage")
            return candidate_pairs
            
        except Exception as e:
            logger.error(f"RecordLinkage blocking failed: {e}")
            # Fallback to exact blocking
            logger.info("Falling back to exact blocking...")
            exact_strategy = ExactBlockingStrategy(self.config)
            return exact_strategy.generate_candidate_pairs(df)


class BlockingProcessor:
    """
    Main blocking processor with multiple strategy support.
    """
    
    BLOCKING_STRATEGIES = {
        'exact': ExactBlockingStrategy,
        'fuzzy': FuzzyBlockingStrategy,
        'sorted_neighbourhood': SortedNeighborhoodStrategy,
        'recordlinkage': RecordLinkageBlockingStrategy,
    }
    
    def __init__(self, config: BlockingConfig):
        """
        Initialize blocking processor.
        
        Args:
            config: Blocking configuration
        """
        self.config = config
        self.method = config.method.lower()
        
        if self.method not in self.BLOCKING_STRATEGIES:
            logger.warning(f"Unknown blocking method: {self.method}, using 'exact'")
            self.method = 'exact'
        
        # Initialize strategy
        strategy_class = self.BLOCKING_STRATEGIES[self.method]
        self.strategy = strategy_class(config)
        
        logger.info(f"Initialized BlockingProcessor with method: {self.method}")
    
    def process_blocking(self, df: pd.DataFrame) -> pd.MultiIndex:
        """
        Process blocking to generate candidate pairs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            MultiIndex of candidate pairs
        """
        start_time = time.time()
        
        logger.info(f"Starting blocking process with method: {self.method}")
        
        # Validate input
        if df.empty:
            logger.warning("Empty DataFrame provided for blocking")
            return pd.MultiIndex.from_tuples([], names=['first', 'second'])
        
        # Check if required columns exist
        missing_columns = [col for col in self.config.columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for blocking: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Generate candidate pairs
        try:
            candidate_pairs = self.strategy.generate_candidate_pairs(df)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Blocking completed in {processing_time:.2f} seconds")
            logger.info(f"Generated {len(candidate_pairs)} candidate pairs")
            
            # Log blocking statistics
            self._log_blocking_stats(df, candidate_pairs)
            
            return candidate_pairs
            
        except Exception as e:
            logger.error(f"Blocking process failed: {e}")
            raise
    
    def _log_blocking_stats(self, df: pd.DataFrame, candidate_pairs: pd.MultiIndex) -> None:
        """Log blocking statistics."""
        total_records = len(df)
        total_possible_pairs = total_records * (total_records - 1) // 2
        generated_pairs = len(candidate_pairs)
        
        if total_possible_pairs > 0:
            reduction_ratio = 1.0 - (generated_pairs / total_possible_pairs)
            logger.info(f"Blocking reduction ratio: {reduction_ratio:.4f}")
            logger.info(f"Pairs reduced from {total_possible_pairs:,} to {generated_pairs:,}")
        
        if generated_pairs > 0:
            # Calculate average pairs per record
            record_pair_counts = defaultdict(int)
            
            for pair in candidate_pairs:
                if isinstance(pair, tuple) and len(pair) == 2:
                    record_pair_counts[pair[0]] += 1
                    record_pair_counts[pair[1]] += 1
            
            if record_pair_counts:
                avg_pairs_per_record = np.mean(list(record_pair_counts.values()))
                max_pairs_per_record = max(record_pair_counts.values())
                logger.info(f"Average pairs per record: {avg_pairs_per_record:.2f}")
                logger.info(f"Maximum pairs for single record: {max_pairs_per_record}")
    
    def estimate_blocking_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate blocking performance without actually generating pairs.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with performance estimates
        """
        logger.info("Estimating blocking performance...")
        
        estimates = {
            'total_records': len(df),
            'total_possible_pairs': len(df) * (len(df) - 1) // 2,
        }
        
        # Create blocking key for estimation
        blocking_key = self.strategy.create_blocking_key(df)
        
        # Analyze blocking key distribution
        key_counts = blocking_key.value_counts()
        
        estimates.update({
            'unique_blocking_keys': len(key_counts),
            'average_records_per_key': key_counts.mean(),
            'max_records_per_key': key_counts.max(),
            'blocking_key_distribution': {
                'mean': key_counts.mean(),
                'median': key_counts.median(),
                'std': key_counts.std(),
                'min': key_counts.min(),
                'max': key_counts.max(),
            }
        })
        
        # Estimate pairs based on blocking method
        if self.method == 'exact':
            # For exact blocking, pairs = sum of C(n,2) for each block
            estimated_pairs = sum(n * (n - 1) // 2 for n in key_counts if n > 1)
        elif self.method == 'sorted_neighbourhood':
            # For sorted neighborhood, approximate based on window size
            window_size = self.config.window_size or 10
            estimated_pairs = min(len(df) * window_size // 2, estimates['total_possible_pairs'])
        else:
            # For fuzzy blocking, rough estimate based on threshold
            # This is a very rough estimate
            estimated_pairs = int(estimates['total_possible_pairs'] * (1 - self.config.threshold))
        
        estimates['estimated_candidate_pairs'] = estimated_pairs
        
        if estimates['total_possible_pairs'] > 0:
            estimates['estimated_reduction_ratio'] = 1.0 - (estimated_pairs / estimates['total_possible_pairs'])
        
        return estimates
    
    def compare_blocking_methods(self, df: pd.DataFrame, 
                               methods: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different blocking methods on the same data.
        
        Args:
            df: Input DataFrame
            methods: List of methods to compare (default: all available)
            
        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = list(self.BLOCKING_STRATEGIES.keys())
        
        results = {}
        
        for method in methods:
            try:
                logger.info(f"Comparing blocking method: {method}")
                
                # Create temporary config for this method
                temp_config = BlockingConfig(
                    columns=self.config.columns,
                    method=method,
                    threshold=self.config.threshold,
                    window_size=self.config.window_size,
                    options=self.config.options
                )
                
                # Create strategy
                strategy_class = self.BLOCKING_STRATEGIES[method]
                strategy = strategy_class(temp_config)
                
                # Measure performance
                start_time = time.time()
                candidate_pairs = strategy.generate_candidate_pairs(df)
                processing_time = time.time() - start_time
                
                # Calculate statistics
                total_possible_pairs = len(df) * (len(df) - 1) // 2
                generated_pairs = len(candidate_pairs)
                reduction_ratio = 1.0 - (generated_pairs / total_possible_pairs) if total_possible_pairs > 0 else 0.0
                
                results[method] = {
                    'candidate_pairs': generated_pairs,
                    'processing_time': processing_time,
                    'reduction_ratio': reduction_ratio,
                    'pairs_per_second': generated_pairs / processing_time if processing_time > 0 else 0,
                }
                
                logger.info(f"{method}: {generated_pairs} pairs in {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                results[method] = {
                    'error': str(e),
                    'candidate_pairs': 0,
                    'processing_time': 0,
                    'reduction_ratio': 0,
                }
        
        return results


def create_blocking_processor(config: BlockingConfig) -> BlockingProcessor:
    """Factory function to create a BlockingProcessor instance."""
    return BlockingProcessor(config)