"""
Advanced survivorship resolution for EasyMDM.
Implements multiple survivorship strategies with enhanced logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime, date
import hashlib
from dataclasses import dataclass
import re
from collections import defaultdict

from ..core.config import SurvivorshipRule, PriorityCondition

logger = logging.getLogger(__name__)


@dataclass
class SurvivorshipResult:
    """Result of survivorship resolution."""
    golden_record: Dict[str, Any]
    survivor_id: Optional[int]
    resolution_logic: str
    confidence_score: float
    metadata: Dict[str, Any]


class BaseSurvivorshipStrategy(ABC):
    """Abstract base class for survivorship strategies."""
    
    def __init__(self, rule: SurvivorshipRule):
        self.rule = rule
        self.column = rule.column
        self.options = rule.options or {}
        
    @abstractmethod
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """
        Resolve survivorship for a column.
        
        Returns:
            Tuple of (resolved_value, logic_explanation, confidence_score)
        """
        pass
    
    def preprocess_value(self, value: Any) -> Any:
        """Preprocess value before resolution."""
        if pd.isna(value):
            return None
        return value
    
    def get_valid_values(self, records: pd.DataFrame, column: str) -> pd.Series:
        """Get non-null values from records."""
        if column not in records.columns:
            return pd.Series([], dtype=object)
        
        valid_mask = records[column].notna()
        return records[column][valid_mask]


class MostRecentStrategy(BaseSurvivorshipStrategy):
    """Most recent date/timestamp survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the most recent date/timestamp."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        try:
            # Try to parse dates
            date_series = pd.to_datetime(valid_values, errors='coerce', 
                                       infer_datetime_format=True)
            
            # Remove NaT (Not a Time) values
            valid_dates = date_series.dropna()
            
            if valid_dates.empty:
                # Fallback to first non-null value
                return valid_values.iloc[0], "fallback_first_value", 0.5
            
            # Find the most recent date
            max_date = valid_dates.max()
            max_date_mask = date_series == max_date
            
            # Get the original value corresponding to the max date
            original_values = valid_values[max_date_mask]
            
            if len(original_values) == 1:
                return original_values.iloc[0], f"most_recent_date_{max_date}", 1.0
            else:
                # Multiple records with same max date, take first
                return original_values.iloc[0], f"most_recent_date_tie_{max_date}", 0.8
                
        except Exception as e:
            logger.warning(f"Date parsing failed for column {column}: {e}")
            return valid_values.iloc[0], "date_parsing_failed", 0.3


class SourcePriorityStrategy(BaseSurvivorshipStrategy):
    """Source priority survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by source priority order."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        source_order = self.rule.source_order
        if not source_order:
            # No source order specified, take first value
            return valid_values.iloc[0], "no_source_order", 0.5
        
        # Check if we have a source column to compare against
        source_column = self.options.get('source_column', 'source')
        
        if source_column not in records.columns:
            logger.warning(f"Source column '{source_column}' not found")
            return valid_values.iloc[0], "source_column_not_found", 0.3
        
        # Create priority mapping
        priority_map = {source: i for i, source in enumerate(source_order)}
        
        # Filter records with valid values
        valid_records = records[records[column].notna()].copy()
        
        # Add priority scores
        valid_records['priority_score'] = valid_records[source_column].map(
            lambda x: priority_map.get(x, len(source_order) + 1)
        )
        
        # Find minimum priority (highest priority)
        min_priority = valid_records['priority_score'].min()
        best_records = valid_records[valid_records['priority_score'] == min_priority]
        
        if len(best_records) == 1:
            best_value = best_records[column].iloc[0]
            best_source = best_records[source_column].iloc[0]
            return best_value, f"source_priority_{best_source}", 1.0
        else:
            # Multiple records with same priority, take first
            best_value = best_records[column].iloc[0]
            best_source = best_records[source_column].iloc[0]
            return best_value, f"source_priority_tie_{best_source}", 0.8


class LongestStringStrategy(BaseSurvivorshipStrategy):
    """Longest string survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the longest string."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        # Convert to strings and calculate lengths
        string_values = valid_values.astype(str)
        lengths = string_values.str.len()
        
        max_length = lengths.max()
        max_length_mask = lengths == max_length
        longest_values = valid_values[max_length_mask]
        
        if len(longest_values) == 1:
            return longest_values.iloc[0], f"longest_string_length_{max_length}", 1.0
        else:
            # Multiple strings with same max length, take first
            return longest_values.iloc[0], f"longest_string_tie_length_{max_length}", 0.8


class HighestValueStrategy(BaseSurvivorshipStrategy):
    """Highest numeric value survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the highest numeric value."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        try:
            # Convert to numeric
            numeric_values = pd.to_numeric(valid_values, errors='coerce')
            numeric_values = numeric_values.dropna()
            
            if numeric_values.empty:
                # No numeric values, fallback to first
                return valid_values.iloc[0], "no_numeric_values", 0.3
            
            max_value = numeric_values.max()
            max_mask = numeric_values == max_value
            max_values = valid_values[numeric_values.index[max_mask]]
            
            if len(max_values) == 1:
                return max_values.iloc[0], f"highest_value_{max_value}", 1.0
            else:
                return max_values.iloc[0], f"highest_value_tie_{max_value}", 0.8
                
        except Exception as e:
            logger.warning(f"Numeric conversion failed for column {column}: {e}")
            return valid_values.iloc[0], "numeric_conversion_failed", 0.3


class LowestValueStrategy(BaseSurvivorshipStrategy):
    """Lowest numeric value survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the lowest numeric value."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        try:
            # Convert to numeric
            numeric_values = pd.to_numeric(valid_values, errors='coerce')
            numeric_values = numeric_values.dropna()
            
            if numeric_values.empty:
                # No numeric values, fallback to first
                return valid_values.iloc[0], "no_numeric_values", 0.3
            
            min_value = numeric_values.min()
            min_mask = numeric_values == min_value
            min_values = valid_values[numeric_values.index[min_mask]]
            
            if len(min_values) == 1:
                return min_values.iloc[0], f"lowest_value_{min_value}", 1.0
            else:
                return min_values.iloc[0], f"lowest_value_tie_{min_value}", 0.8
                
        except Exception as e:
            logger.warning(f"Numeric conversion failed for column {column}: {e}")
            return valid_values.iloc[0], "numeric_conversion_failed", 0.3


class ThresholdStrategy(BaseSurvivorshipStrategy):
    """Threshold-based survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by threshold comparison."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        threshold = self.rule.threshold
        if threshold is None:
            logger.warning(f"No threshold specified for column {column}")
            return valid_values.iloc[0], "no_threshold", 0.3
        
        comparison = self.options.get('comparison', 'greater')
        
        try:
            # Convert to numeric
            numeric_values = pd.to_numeric(valid_values, errors='coerce')
            numeric_values = numeric_values.dropna()
            
            if numeric_values.empty:
                return valid_values.iloc[0], "no_numeric_values", 0.3
            
            if comparison == 'greater':
                filtered_values = numeric_values[numeric_values > threshold]
                logic = f"greater_than_{threshold}"
            elif comparison == 'less':
                filtered_values = numeric_values[numeric_values < threshold]
                logic = f"less_than_{threshold}"
            elif comparison == 'equal':
                filtered_values = numeric_values[numeric_values == threshold]
                logic = f"equal_to_{threshold}"
            else:
                filtered_values = numeric_values
                logic = f"unknown_comparison_{comparison}"
            
            if filtered_values.empty:
                # No values meet threshold, take first valid
                return valid_values.iloc[0], f"no_threshold_match_{logic}", 0.5
            
            # Take first value that meets threshold
            first_match_idx = filtered_values.index[0]
            return valid_values.loc[first_match_idx], logic, 0.9
            
        except Exception as e:
            logger.warning(f"Threshold comparison failed for column {column}: {e}")
            return valid_values.iloc[0], "threshold_comparison_failed", 0.3


class MostFrequentStrategy(BaseSurvivorshipStrategy):
    """Most frequent value survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the most frequent value."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        if len(valid_values) == 1:
            return valid_values.iloc[0], "single_value", 1.0
        
        # Count value frequencies
        value_counts = valid_values.value_counts()
        
        if len(value_counts) == 1:
            # All values are the same
            return value_counts.index[0], "all_values_same", 1.0
        
        max_count = value_counts.max()
        most_frequent = value_counts[value_counts == max_count]
        
        if len(most_frequent) == 1:
            return most_frequent.index[0], f"most_frequent_count_{max_count}", 1.0
        else:
            # Multiple values with same frequency, take first
            return most_frequent.index[0], f"most_frequent_tie_count_{max_count}", 0.8


class FirstValueStrategy(BaseSurvivorshipStrategy):
    """First non-null value survivorship strategy."""
    
    def resolve(self, records: pd.DataFrame, column: str) -> Tuple[Any, str, float]:
        """Resolve by selecting the first non-null value."""
        valid_values = self.get_valid_values(records, column)
        
        if valid_values.empty:
            return None, "no_valid_values", 0.0
        
        return valid_values.iloc[0], "first_value", 0.7


class SurvivorshipResolver:
    """
    Main survivorship resolver with priority and survivorship rules.
    """
    
    SURVIVORSHIP_STRATEGIES = {
        'most_recent': MostRecentStrategy,
        'source_priority': SourcePriorityStrategy,
        'longest_string': LongestStringStrategy,
        'highest_value': HighestValueStrategy,
        'lowest_value': LowestValueStrategy,
        'greater_than_threshold': ThresholdStrategy,
        'less_than_threshold': ThresholdStrategy,
        'most_frequent': MostFrequentStrategy,
        'first_value': FirstValueStrategy,
    }
    
    def __init__(self, survivorship_rules: List[SurvivorshipRule],
                 priority_conditions: List[PriorityCondition],
                 unique_id_columns: Optional[List[str]] = None):
        """
        Initialize survivorship resolver.
        
        Args:
            survivorship_rules: List of survivorship rules
            priority_conditions: List of priority conditions
            unique_id_columns: Columns to use for unique ID generation
        """
        self.survivorship_rules = survivorship_rules
        self.priority_conditions = sorted(priority_conditions, key=lambda x: x.priority)
        self.unique_id_columns = unique_id_columns or []
        
        # Initialize strategy instances
        self.strategies = {}
        for rule in survivorship_rules:
            if rule.strategy not in self.SURVIVORSHIP_STRATEGIES:
                logger.warning(f"Unknown survivorship strategy: {rule.strategy}")
                continue
            
            strategy_class = self.SURVIVORSHIP_STRATEGIES[rule.strategy]
            self.strategies[rule.column] = strategy_class(rule)
        
        logger.info(f"Initialized SurvivorshipResolver with {len(self.strategies)} strategies")
    
    def resolve_cluster(self, records: pd.DataFrame) -> SurvivorshipResult:
        """
        Resolve survivorship for a cluster of records.
        
        Args:
            records: DataFrame containing records in the cluster
            
        Returns:
            SurvivorshipResult object
        """
        if records.empty:
            return SurvivorshipResult(
                golden_record={},
                survivor_id=None,
                resolution_logic="empty_cluster",
                confidence_score=0.0,
                metadata={}
            )
        
        if len(records) == 1:
            # Single record, just return it
            record = records.iloc[0].to_dict()
            record_id = records.index[0]
            unique_id = self._generate_unique_id(record)
            
            return SurvivorshipResult(
                golden_record=record,
                survivor_id=record_id,
                resolution_logic="single_record",
                confidence_score=1.0,
                metadata={
                    'unique_id': unique_id,
                    'source_records': [record_id],
                    'cluster_size': 1
                }
            )
        
        # Try priority rules first
        priority_result = self._apply_priority_rules(records)
        if priority_result is not None:
            return priority_result
        
        # Apply survivorship rules
        return self._apply_survivorship_rules(records)
    
    def _apply_priority_rules(self, records: pd.DataFrame) -> Optional[SurvivorshipResult]:
        """Apply priority rules to select a survivor."""
        current_records = records.copy()
        applied_conditions = []
        
        for condition in self.priority_conditions:
            column = condition.column
            expected_value = condition.value
            
            if column not in current_records.columns:
                continue
            
            # Convert column values to match expected value type if possible
            try:
                if isinstance(expected_value, (int, float)):
                    # Try numeric conversion
                    current_records[column] = pd.to_numeric(
                        current_records[column], errors='coerce'
                    )
                elif isinstance(expected_value, bool):
                    # Try boolean conversion
                    current_records[column] = current_records[column].map({
                        True: True, False: False, 1: True, 0: False,
                        '1': True, '0': False, 'true': True, 'false': False,
                        'True': True, 'False': False, 'yes': True, 'no': False,
                        'Yes': True, 'No': False, 'Y': True, 'N': False
                    })
            except Exception:
                pass
            
            # Find matching records
            matches = current_records[current_records[column] == expected_value]
            
            if len(matches) == 1:
                # Exactly one match, this is our survivor
                survivor_record = matches.iloc[0]
                survivor_id = matches.index[0]
                
                logic = f"priority_{column}_{expected_value}"
                if applied_conditions:
                    logic = f"priority_chain_{'_'.join(applied_conditions)}_{column}_{expected_value}"
                
                golden_record = survivor_record.to_dict()
                unique_id = self._generate_unique_id(golden_record)
                
                return SurvivorshipResult(
                    golden_record=golden_record,
                    survivor_id=survivor_id,
                    resolution_logic=logic,
                    confidence_score=1.0,
                    metadata={
                        'unique_id': unique_id,
                        'source_records': records.index.tolist(),
                        'cluster_size': len(records),
                        'priority_conditions_applied': applied_conditions + [f"{column}={expected_value}"]
                    }
                )
            elif len(matches) > 1:
                # Multiple matches, continue with next condition
                current_records = matches
                applied_conditions.append(f"{column}={expected_value}")
                continue
            else:
                # No matches for this condition, skip to next
                continue
        
        # If we get here, priority rules didn't result in a single survivor
        return None
    
    def _apply_survivorship_rules(self, records: pd.DataFrame) -> SurvivorshipResult:
        """Apply survivorship rules to create golden record."""
        golden_record = {}
        resolution_logic_parts = []
        confidence_scores = []
        
        # Process each column
        for column in records.columns:
            if column in self.strategies:
                # Use defined strategy
                strategy = self.strategies[column]
                value, logic, confidence = strategy.resolve(records, column)
            else:
                # Default strategy: first non-null value
                valid_values = records[column].dropna()
                if not valid_values.empty:
                    value = valid_values.iloc[0]
                    logic = "first_non_null"
                    confidence = 0.5
                else:
                    value = None
                    logic = "no_valid_values"
                    confidence = 0.0
            
            golden_record[column] = value
            resolution_logic_parts.append(f"{column}:{logic}")
            confidence_scores.append(confidence)
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Generate resolution logic string
        resolution_logic = f"survivorship_{'|'.join(resolution_logic_parts[:3])}"  # Limit length
        
        # Generate unique ID
        unique_id = self._generate_unique_id(golden_record)
        
        return SurvivorshipResult(
            golden_record=golden_record,
            survivor_id=None,  # No single survivor, it's a merged record
            resolution_logic=resolution_logic,
            confidence_score=overall_confidence,
            metadata={
                'unique_id': unique_id,
                'source_records': records.index.tolist(),
                'cluster_size': len(records),
                'survivorship_applied': True,
                'resolution_details': resolution_logic_parts
            }
        )
    
    def _generate_unique_id(self, record: Dict[str, Any]) -> str:
        """Generate unique ID for a record based on configured columns."""
        if not self.unique_id_columns:
            return ""
        
        # Extract values for unique ID columns
        id_values = []
        for column in self.unique_id_columns:
            value = record.get(column, "")
            if pd.isna(value):
                value = ""
            id_values.append(str(value))
        
        # Create MD5 hash
        id_string = "|".join(id_values)
        return hashlib.md5(id_string.encode('utf-8')).hexdigest()
    
    def resolve_multiple_clusters(self, clusters: List[pd.DataFrame]) -> List[SurvivorshipResult]:
        """
        Resolve survivorship for multiple clusters.
        
        Args:
            clusters: List of DataFrames, each representing a cluster
            
        Returns:
            List of SurvivorshipResult objects
        """
        results = []
        
        for i, cluster_df in enumerate(clusters):
            try:
                result = self.resolve_cluster(cluster_df)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to resolve cluster {i}: {e}")
                # Create empty result for failed cluster
                empty_result = SurvivorshipResult(
                    golden_record={},
                    survivor_id=None,
                    resolution_logic=f"error_{str(e)[:50]}",
                    confidence_score=0.0,
                    metadata={'cluster_id': i, 'error': str(e)}
                )
                results.append(empty_result)
        
        return results
    
    def get_strategy_recommendations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get strategy recommendations based on data analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping columns to recommended strategies
        """
        recommendations = {}
        
        for column in df.columns:
            column_recommendations = []
            
            # Analyze column characteristics
            non_null_values = df[column].dropna()
            
            if non_null_values.empty:
                column_recommendations.append('first_value')
            else:
                # Check for date patterns
                if self._looks_like_date(non_null_values):
                    column_recommendations.append('most_recent')
                
                # Check for numeric patterns
                if self._looks_like_numeric(non_null_values):
                    column_recommendations.extend(['highest_value', 'lowest_value'])
                
                # Check for string patterns
                if self._looks_like_string(non_null_values):
                    column_recommendations.append('longest_string')
                
                # Check for categorical patterns
                if self._looks_like_categorical(non_null_values):
                    column_recommendations.extend(['most_frequent', 'source_priority'])
                
                # Always include fallback strategies
                column_recommendations.append('first_value')
            
            recommendations[column] = list(set(column_recommendations))  # Remove duplicates
        
        return recommendations
    
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if series contains date-like values."""
        try:
            parsed = pd.to_datetime(series.head(10), errors='coerce')
            return parsed.notna().sum() > len(parsed) * 0.7  # 70% success rate
        except Exception:
            return False
    
    def _looks_like_numeric(self, series: pd.Series) -> bool:
        """Check if series contains numeric values."""
        try:
            numeric = pd.to_numeric(series.head(10), errors='coerce')
            return numeric.notna().sum() > len(numeric) * 0.7  # 70% success rate
        except Exception:
            return False
    
    def _looks_like_string(self, series: pd.Series) -> bool:
        """Check if series contains string values."""
        str_series = series.astype(str)
        avg_length = str_series.str.len().mean()
        return avg_length > 5  # Average length > 5 characters
    
    def _looks_like_categorical(self, series: pd.Series) -> bool:
        """Check if series contains categorical values."""
        unique_count = series.nunique()
        total_count = len(series)
        return unique_count < total_count * 0.3  # Less than 30% unique values


def create_survivorship_resolver(survivorship_rules: List[SurvivorshipRule],
                               priority_conditions: List[PriorityCondition],
                               unique_id_columns: Optional[List[str]] = None) -> SurvivorshipResolver:
    """Factory function to create a SurvivorshipResolver instance."""
    return SurvivorshipResolver(survivorship_rules, priority_conditions, unique_id_columns)