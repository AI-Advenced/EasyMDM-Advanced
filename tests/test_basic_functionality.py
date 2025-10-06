"""
Basic functionality tests for EasyMDM Advanced.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from easymdm import MDMEngine, MDMConfig
from easymdm.core.config import (
    DatabaseConfig, BlockingConfig, SimilarityConfig, 
    ThresholdConfig, SurvivorshipRule, PriorityCondition
)
from easymdm.similarity.matcher import compute_string_similarity
from easymdm.database.connector import DatabaseConnector


class TestBasicFunctionality:
    """Test basic EasyMDM functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        return pd.DataFrame({
            'first_name': ['John', 'Jon', 'Jane', 'John'],
            'last_name': ['Smith', 'Smith', 'Doe', 'Smyth'],
            'address': ['123 Main St', '123 Main St', '456 Oak Ave', '123 Main Street'],
            'city': ['New York', 'New York', 'Boston', 'New York'],
            'phone': ['555-1234', '555-1234', '555-5678', '(555) 1234'],
            'source': ['CRM', 'Import', 'Manual', 'CRM'],
            'last_updated': ['2023-01-15', '2023-01-10', '2023-02-20', '2023-01-20']
        })
    
    @pytest.fixture
    def csv_file(self, sample_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def basic_config(self, csv_file):
        """Create basic MDM configuration."""
        return MDMConfig(
            source=DatabaseConfig(type='csv', file_path=csv_file),
            blocking=BlockingConfig(columns=['first_name', 'last_name'], method='exact'),
            similarity=[
                SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
                SimilarityConfig(column='address', method='levenshtein', weight=1.0),
            ],
            thresholds=ThresholdConfig(review=0.7, auto_merge=0.9, definite_no_match=0.3),
            survivorship_rules=[
                SurvivorshipRule(column='last_updated', strategy='most_recent'),
            ],
            output_path=tempfile.gettempdir()
        )
    
    def test_similarity_functions(self):
        """Test similarity computation functions."""
        # Test basic similarity methods
        assert compute_string_similarity("John", "Jon", "jarowinkler") > 0.8
        assert compute_string_similarity("Smith", "Smith", "exact") == 1.0
        assert compute_string_similarity("123 Main St", "123 Main Street", "levenshtein") > 0.7
        
        # Test edge cases
        assert compute_string_similarity("", "", "jarowinkler") == 1.0
        assert compute_string_similarity("test", "", "jarowinkler") == 0.0
    
    def test_csv_connector(self, csv_file, sample_data):
        """Test CSV database connector."""
        config = DatabaseConfig(type='csv', file_path=csv_file)
        connector = DatabaseConnector.create_connector(config)
        
        assert connector.test_connection()
        
        connector.connect()
        df = connector.load_data()
        connector.disconnect()
        
        assert len(df) == len(sample_data)
        assert list(df.columns) == list(sample_data.columns)
    
    def test_configuration_creation(self):
        """Test configuration creation and validation."""
        config = MDMConfig(
            source=DatabaseConfig(type='csv', file_path='test.csv'),
            blocking=BlockingConfig(columns=['name'], method='exact'),
            similarity=[SimilarityConfig(column='name', method='jarowinkler', weight=1.0)],
            thresholds=ThresholdConfig()
        )
        
        assert config.source.type == 'csv'
        assert len(config.similarity) == 1
        assert config.thresholds.auto_merge == 0.9  # Default value
    
    def test_mdm_engine_initialization(self, basic_config):
        """Test MDM engine initialization."""
        engine = MDMEngine(basic_config)
        
        assert engine.config == basic_config
        assert engine.similarity_matcher is not None
        assert engine.record_clusterer is not None
        assert engine.survivorship_resolver is not None
    
    def test_configuration_test(self, basic_config):
        """Test configuration testing functionality."""
        engine = MDMEngine(basic_config)
        test_results = engine.test_configuration()
        
        assert isinstance(test_results, dict)
        assert 'database_connection' in test_results
        # CSV connection should work
        assert test_results['database_connection'] is True
    
    def test_data_profiling(self, basic_config):
        """Test data profiling functionality."""
        engine = MDMEngine(basic_config)
        profile = engine.get_data_profile()
        
        assert isinstance(profile, dict)
        if profile:  # If profiling succeeded
            assert 'total_records' in profile
            assert 'total_columns' in profile
            assert profile['total_records'] > 0
    
    def test_end_to_end_processing(self, basic_config):
        """Test complete end-to-end MDM processing."""
        engine = MDMEngine(basic_config)
        
        # Run processing
        result = engine.process()
        
        # Verify results
        assert result is not None
        assert hasattr(result, 'golden_records')
        assert hasattr(result, 'processing_stats')
        assert hasattr(result, 'execution_time')
        
        # Should have some golden records
        assert len(result.golden_records) > 0
        
        # Should have processing statistics
        assert 'input_records' in result.processing_stats
        assert result.processing_stats['input_records'] > 0
        
        # Execution time should be reasonable
        assert 0 < result.execution_time < 60  # Less than 60 seconds
    
    def test_available_connectors(self):
        """Test available database connectors."""
        available = DatabaseConnector.get_available_connectors()
        
        assert isinstance(available, list)
        assert 'csv' in available  # CSV should always be available
        assert 'sqlite' in available  # SQLite should always be available


class TestSimilarityMethods:
    """Test similarity methods in detail."""
    
    @pytest.mark.parametrize("method,str1,str2,expected_min", [
        ("jarowinkler", "John", "Jon", 0.8),
        ("levenshtein", "Smith", "Smyth", 0.6),
        ("exact", "test", "test", 1.0),
        ("exact", "test", "Test", 0.0),  # Case sensitive by default
    ])
    def test_similarity_methods(self, method, str1, str2, expected_min):
        """Test specific similarity methods."""
        score = compute_string_similarity(str1, str2, method)
        assert score >= expected_min
        assert 0 <= score <= 1
    
    def test_similarity_edge_cases(self):
        """Test similarity methods with edge cases."""
        methods = ["jarowinkler", "levenshtein", "exact"]
        
        for method in methods:
            # Empty strings
            assert compute_string_similarity("", "", method) >= 0
            
            # One empty string
            assert compute_string_similarity("test", "", method) == 0
            
            # Identical strings
            if method != "exact":  # exact might be case sensitive
                assert compute_string_similarity("test", "test", method) == 1.0


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_config_serialization(self):
        """Test configuration serialization to/from YAML."""
        config = MDMConfig(
            source=DatabaseConfig(type='csv', file_path='test.csv'),
            blocking=BlockingConfig(columns=['name'], method='exact'),
            similarity=[SimilarityConfig(column='name', method='jarowinkler', weight=1.0)],
            thresholds=ThresholdConfig()
        )
        
        # Test conversion to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['source']['type'] == 'csv'
        
        # Test creation from dict
        config2 = MDMConfig.from_dict(config_dict)
        assert config2.source.type == config.source.type
        assert len(config2.similarity) == len(config.similarity)
    
    def test_config_yaml_operations(self):
        """Test YAML save/load operations."""
        config = MDMConfig(
            source=DatabaseConfig(type='csv', file_path='test.csv'),
            blocking=BlockingConfig(columns=['name'], method='exact'),
            similarity=[SimilarityConfig(column='name', method='jarowinkler', weight=1.0)],
            thresholds=ThresholdConfig()
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.save_yaml(f.name)
            
            # Load back
            config2 = MDMConfig.from_yaml(f.name)
            
            assert config2.source.type == config.source.type
            assert config2.blocking.method == config.blocking.method
            
            os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__])