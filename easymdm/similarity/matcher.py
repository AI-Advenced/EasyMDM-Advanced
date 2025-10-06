"""
Advanced similarity matching for EasyMDM.
Implements multiple similarity algorithms with optimization and caching.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from joblib import Parallel, delayed

# Similarity algorithms
import recordlinkage
from fuzzywuzzy import fuzz
import jellyfish
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# Numba for performance optimization
try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..core.config import SimilarityConfig

logger = logging.getLogger(__name__)


class BaseSimilarityFunction(ABC):
    """Abstract base class for similarity functions."""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.name = config.method
        self.threshold = config.threshold
        self.weight = config.weight
        self.options = config.options or {}
        
    @abstractmethod
    def compute(self, str1: str, str2: str) -> float:
        """Compute similarity between two strings."""
        pass
    
    def compute_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Compute similarity for pandas Series (vectorized operation)."""
        # Default implementation using apply
        return pd.Series([
            self.compute(str(s1), str(s2)) 
            for s1, s2 in zip(series1, series2)
        ])
    
    def preprocess_string(self, text: str) -> str:
        """Preprocess string before similarity computation."""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Apply preprocessing options
        if self.options.get('lowercase', True):
            text = text.lower()
            
        if self.options.get('remove_punctuation', False):
            import string
            text = text.translate(str.maketrans('', '', string.punctuation))
            
        if self.options.get('remove_extra_spaces', True):
            text = ' '.join(text.split())
            
        return text


class JaroWinklerSimilarity(BaseSimilarityFunction):
    """Jaro-Winkler similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        try:
            return jellyfish.jaro_winkler_similarity(str1, str2)
        except Exception:
            return 0.0
    
    def compute_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Optimized vectorized computation for Jaro-Winkler."""
        if NUMBA_AVAILABLE and len(series1) > 1000:
            return self._compute_numba_vectorized(series1, series2)
        else:
            return super().compute_vectorized(series1, series2)
    
    def _compute_numba_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Numba-optimized computation."""
        # Convert to numpy arrays of strings
        arr1 = series1.astype(str).values
        arr2 = series2.astype(str).values
        
        @jit(nopython=False)  # nopython=False because of string operations
        def jaro_winkler_numba(s1_arr, s2_arr):
            results = np.zeros(len(s1_arr))
            for i in range(len(s1_arr)):
                results[i] = jellyfish.jaro_winkler_similarity(
                    str(s1_arr[i]).lower(), str(s2_arr[i]).lower()
                )
            return results
        
        if NUMBA_AVAILABLE:
            result_array = jaro_winkler_numba(arr1, arr2)
        else:
            result_array = np.array([
                jellyfish.jaro_winkler_similarity(str(s1).lower(), str(s2).lower())
                for s1, s2 in zip(arr1, arr2)
            ])
            
        return pd.Series(result_array)


class LevenshteinSimilarity(BaseSimilarityFunction):
    """Levenshtein distance similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
            
        try:
            distance = Levenshtein.distance(str1, str2)
            return 1.0 - (distance / max_len)
        except Exception:
            return 0.0
    
    def compute_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Optimized vectorized computation for Levenshtein."""
        def compute_batch(s1_batch, s2_batch):
            return [self.compute(s1, s2) for s1, s2 in zip(s1_batch, s2_batch)]
        
        # Use parallel processing for large datasets
        if len(series1) > 5000:
            n_jobs = min(mp.cpu_count(), 4)
            batch_size = len(series1) // n_jobs
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_batch)(
                    series1.iloc[i:i+batch_size].values,
                    series2.iloc[i:i+batch_size].values
                )
                for i in range(0, len(series1), batch_size)
            )
            
            # Flatten results
            flat_results = [item for sublist in results for item in sublist]
            return pd.Series(flat_results)
        else:
            return super().compute_vectorized(series1, series2)


class CosineSimilarity(BaseSimilarityFunction):
    """Cosine similarity implementation using TF-IDF."""
    
    def __init__(self, config: SimilarityConfig):
        super().__init__(config)
        self.vectorizer = None
        self._cache = {}
        
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        try:
            # Create TF-IDF vectors
            corpus = [str1, str2]
            vectorizer = TfidfVectorizer(
                ngram_range=(1, self.options.get('ngram_range', 2)),
                analyzer='char_wb'
            )
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity[0][0])
        except Exception:
            return 0.0
    
    def compute_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Optimized vectorized computation for Cosine similarity."""
        # Preprocess all strings
        processed_s1 = series1.apply(self.preprocess_string)
        processed_s2 = series2.apply(self.preprocess_string)
        
        # Create combined corpus
        all_texts = list(processed_s1) + list(processed_s2)
        unique_texts = list(set(text for text in all_texts if text))
        
        if not unique_texts:
            return pd.Series([0.0] * len(series1))
        
        try:
            # Fit vectorizer on all unique texts
            vectorizer = TfidfVectorizer(
                ngram_range=(1, self.options.get('ngram_range', 2)),
                analyzer='char_wb'
            )
            
            vectorizer.fit(unique_texts)
            
            # Transform series
            tfidf_s1 = vectorizer.transform(processed_s1.fillna(''))
            tfidf_s2 = vectorizer.transform(processed_s2.fillna(''))
            
            # Compute similarities
            similarities = []
            for i in range(len(series1)):
                if processed_s1.iloc[i] and processed_s2.iloc[i]:
                    sim = cosine_similarity(tfidf_s1[i:i+1], tfidf_s2[i:i+1])
                    similarities.append(float(sim[0][0]))
                elif not processed_s1.iloc[i] and not processed_s2.iloc[i]:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
            
            return pd.Series(similarities)
            
        except Exception as e:
            logger.warning(f"Cosine similarity vectorized computation failed: {e}")
            return super().compute_vectorized(series1, series2)


class JaccardSimilarity(BaseSimilarityFunction):
    """Jaccard similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        try:
            # Create n-grams
            n = self.options.get('n', 2)
            set1 = set(str1[i:i+n] for i in range(len(str1)-n+1))
            set2 = set(str2[i:i+n] for i in range(len(str2)-n+1))
            
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0


class ExactSimilarity(BaseSimilarityFunction):
    """Exact match similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if str1 == str2:
            return 1.0
        else:
            return 0.0
    
    def compute_vectorized(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Optimized vectorized computation for exact matching."""
        processed_s1 = series1.apply(self.preprocess_string)
        processed_s2 = series2.apply(self.preprocess_string)
        
        return (processed_s1 == processed_s2).astype(float)


class FuzzyWuzzyRatioSimilarity(BaseSimilarityFunction):
    """FuzzyWuzzy ratio similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        try:
            return fuzz.ratio(str1, str2) / 100.0
        except Exception:
            return 0.0


class FuzzyWuzzyTokenSortSimilarity(BaseSimilarityFunction):
    """FuzzyWuzzy token sort similarity implementation."""
    
    def compute(self, str1: str, str2: str) -> float:
        str1 = self.preprocess_string(str1)
        str2 = self.preprocess_string(str2)
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        try:
            return fuzz.token_sort_ratio(str1, str2) / 100.0
        except Exception:
            return 0.0


class SimilarityMatcher:
    """Main similarity matcher class with advanced features."""
    
    SIMILARITY_FUNCTIONS = {
        'jarowinkler': JaroWinklerSimilarity,
        'levenshtein': LevenshteinSimilarity,
        'cosine': CosineSimilarity,
        'jaccard': JaccardSimilarity,
        'exact': ExactSimilarity,
        'fuzzy_ratio': FuzzyWuzzyRatioSimilarity,
        'fuzzy_token_sort': FuzzyWuzzyTokenSortSimilarity,
    }
    
    def __init__(self, similarity_configs: List[SimilarityConfig], 
                 use_cache: bool = True, cache_size: int = 10000,
                 n_jobs: int = -1):
        """
        Initialize similarity matcher.
        
        Args:
            similarity_configs: List of similarity configurations
            use_cache: Whether to use caching for similarity computations
            cache_size: Maximum cache size
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.similarity_configs = similarity_configs
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Initialize similarity functions
        self.similarity_functions = {}
        for config in similarity_configs:
            if config.method not in self.SIMILARITY_FUNCTIONS:
                raise ValueError(f"Unknown similarity method: {config.method}")
            
            func_class = self.SIMILARITY_FUNCTIONS[config.method]
            self.similarity_functions[config.column] = func_class(config)
        
        # Cache for similarity computations
        if self.use_cache:
            self._similarity_cache = {}
        
        logger.info(f"Initialized SimilarityMatcher with {len(self.similarity_functions)} functions")
    
    def compute_similarities(self, df: pd.DataFrame, 
                           candidate_pairs: pd.MultiIndex) -> pd.DataFrame:
        """
        Compute similarities for candidate pairs.
        
        Args:
            df: Input DataFrame
            candidate_pairs: MultiIndex of candidate pairs
            
        Returns:
            DataFrame with similarity scores
        """
        start_time = time.time()
        logger.info(f"Computing similarities for {len(candidate_pairs)} candidate pairs")
        
        # Initialize results DataFrame
        results = pd.DataFrame(index=candidate_pairs)
        
        # Compute similarities for each configured column
        for config in self.similarity_configs:
            column = config.column
            
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame")
                continue
            
            similarity_func = self.similarity_functions[column]
            
            # Extract pairs data
            pairs_df = pd.DataFrame(candidate_pairs.tolist(), columns=['first', 'second'])
            series1 = df.loc[pairs_df['first'], column]
            series2 = df.loc[pairs_df['second'], column]
            
            # Reset indices to align
            series1.reset_index(drop=True, inplace=True)
            series2.reset_index(drop=True, inplace=True)
            
            # Compute similarities
            column_name = f"{column}_{config.method}"
            
            try:
                if len(candidate_pairs) > 1000:
                    # Use vectorized computation for large datasets
                    similarities = similarity_func.compute_vectorized(series1, series2)
                else:
                    # Use regular computation for small datasets
                    similarities = pd.Series([
                        similarity_func.compute(str(s1), str(s2))
                        for s1, s2 in zip(series1, series2)
                    ])
                
                results[column_name] = similarities.values
                
            except Exception as e:
                logger.error(f"Failed to compute similarities for column '{column}': {e}")
                results[column_name] = 0.0
        
        # Compute overall similarity score
        results = self._compute_overall_similarity(results)
        
        # Apply thresholds and categorize
        results = self._categorize_matches(results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Similarity computation completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _compute_overall_similarity(self, results: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted overall similarity score."""
        similarity_columns = [col for col in results.columns if col != 'overall_score']
        
        if not similarity_columns:
            results['overall_score'] = 0.0
            return results
        
        # Get weights for each column
        weights = {}
        for config in self.similarity_configs:
            column_name = f"{config.column}_{config.method}"
            if column_name in similarity_columns:
                weights[column_name] = config.weight
        
        # Compute weighted average
        weighted_scores = []
        total_weights = sum(weights.values())
        
        for idx, row in results.iterrows():
            score = 0.0
            for col in similarity_columns:
                if col in weights:
                    score += row[col] * weights[col]
            
            if total_weights > 0:
                score /= total_weights
            
            weighted_scores.append(score)
        
        results['overall_score'] = weighted_scores
        return results
    
    def _categorize_matches(self, results: pd.DataFrame) -> pd.DataFrame:
        """Categorize matches based on thresholds."""
        # For now, use default thresholds - these will be updated by the main engine
        # based on the configuration
        results['match_category'] = 'non_match'  # Default category
        return results
    
    def compute_single_similarity(self, str1: str, str2: str, 
                                method: str, **options) -> float:
        """
        Compute similarity between two strings using specified method.
        
        Args:
            str1: First string
            str2: Second string
            method: Similarity method to use
            **options: Additional options for the method
            
        Returns:
            Similarity score between 0 and 1
        """
        if method not in self.SIMILARITY_FUNCTIONS:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Create temporary config
        from ..core.config import SimilarityConfig
        temp_config = SimilarityConfig(column='temp', method=method, options=options)
        
        # Create similarity function
        similarity_func = self.SIMILARITY_FUNCTIONS[method](temp_config)
        
        return similarity_func.compute(str1, str2)
    
    def benchmark_methods(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, float]:
        """
        Benchmark different similarity methods on a sample of data.
        
        Args:
            df: Input DataFrame
            sample_size: Number of pairs to test
            
        Returns:
            Dictionary with method names and their average computation times
        """
        logger.info(f"Benchmarking similarity methods on {sample_size} pairs")
        
        # Create sample pairs
        sample_df = df.sample(min(sample_size, len(df)))
        pairs = []
        
        for i in range(0, min(sample_size, len(sample_df) - 1)):
            for j in range(i + 1, min(i + 10, len(sample_df))):  # Limit pairs per record
                pairs.append((sample_df.index[i], sample_df.index[j]))
                if len(pairs) >= sample_size:
                    break
            if len(pairs) >= sample_size:
                break
        
        # Test each method
        results = {}
        
        for method_name, method_class in self.SIMILARITY_FUNCTIONS.items():
            try:
                # Create temporary config
                from ..core.config import SimilarityConfig
                temp_config = SimilarityConfig(column='temp', method=method_name)
                method_func = method_class(temp_config)
                
                # Time the method
                start_time = time.time()
                
                for pair in pairs[:100]:  # Test on first 100 pairs
                    val1 = str(df.loc[pair[0]].iloc[0])  # Use first column
                    val2 = str(df.loc[pair[1]].iloc[0])
                    method_func.compute(val1, val2)
                
                elapsed_time = time.time() - start_time
                results[method_name] = elapsed_time / 100  # Average time per computation
                
            except Exception as e:
                logger.warning(f"Benchmark failed for method '{method_name}': {e}")
                results[method_name] = float('inf')
        
        return results
    
    def get_method_recommendations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get method recommendations based on data characteristics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with column types and recommended methods
        """
        recommendations = {}
        
        for column in df.columns:
            column_type = self._analyze_column_type(df[column])
            recommendations[column] = self._get_recommended_methods(column_type)
        
        return recommendations
    
    def _analyze_column_type(self, series: pd.Series) -> str:
        """Analyze column type and characteristics."""
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'empty'
        
        # Convert to strings for analysis
        str_series = non_null_series.astype(str)
        
        # Check for numeric patterns
        numeric_count = sum(s.replace('.', '').replace('-', '').isdigit() for s in str_series)
        numeric_ratio = numeric_count / len(str_series)
        
        if numeric_ratio > 0.8:
            return 'numeric'
        
        # Check average length
        avg_length = str_series.str.len().mean()
        
        if avg_length < 5:
            return 'short_text'
        elif avg_length < 20:
            return 'medium_text'
        else:
            return 'long_text'
    
    def _get_recommended_methods(self, column_type: str) -> List[str]:
        """Get recommended similarity methods for a column type."""
        recommendations = {
            'numeric': ['exact', 'levenshtein'],
            'short_text': ['exact', 'jarowinkler', 'levenshtein'],
            'medium_text': ['jarowinkler', 'fuzzy_token_sort', 'cosine'],
            'long_text': ['cosine', 'jaccard', 'fuzzy_token_sort'],
            'empty': ['exact']
        }
        
        return recommendations.get(column_type, ['jarowinkler', 'levenshtein'])


def create_similarity_matcher(similarity_configs: List[SimilarityConfig],
                            **kwargs) -> SimilarityMatcher:
    """Factory function to create a SimilarityMatcher instance."""
    return SimilarityMatcher(similarity_configs, **kwargs)


# Utility functions for similarity computation
def compute_string_similarity(str1: str, str2: str, method: str = 'jarowinkler') -> float:
    """
    Utility function to compute similarity between two strings.
    
    Args:
        str1: First string
        str2: Second string
        method: Similarity method to use
        
    Returns:
        Similarity score between 0 and 1
    """
    from ..core.config import SimilarityConfig
    
    config = SimilarityConfig(column='temp', method=method)
    
    if method not in SimilarityMatcher.SIMILARITY_FUNCTIONS:
        raise ValueError(f"Unknown similarity method: {method}")
    
    func_class = SimilarityMatcher.SIMILARITY_FUNCTIONS[method]
    func = func_class(config)
    
    return func.compute(str1, str2)


def find_best_similarity_method(str1: str, str2: str, 
                               methods: Optional[List[str]] = None) -> Tuple[str, float]:
    """
    Find the best similarity method for two given strings.
    
    Args:
        str1: First string
        str2: Second string
        methods: List of methods to test (default: all available)
        
    Returns:
        Tuple of (best_method, best_score)
    """
    if methods is None:
        methods = list(SimilarityMatcher.SIMILARITY_FUNCTIONS.keys())
    
    best_method = methods[0]
    best_score = 0.0
    
    for method in methods:
        try:
            score = compute_string_similarity(str1, str2, method)
            if score > best_score:
                best_score = score
                best_method = method
        except Exception:
            continue
    
    return best_method, best_score