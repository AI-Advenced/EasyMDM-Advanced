"""
Configuration management for EasyMDM Advanced.
Handles loading, validation, and management of MDM configurations.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import jsonschema
from cerberus import Validator

import logging
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    type: str  # 'sqlite', 'postgresql', 'sqlserver', 'duckdb', 'csv'
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    schema: Optional[str] = None
    table: Optional[str] = None
    file_path: Optional[str] = None
    connection_string: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockingConfig:
    """Blocking configuration for record linkage."""
    columns: List[str]
    method: str = "fuzzy"  # 'fuzzy', 'exact', 'sorted_neighbourhood'
    threshold: float = 0.8
    window_size: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarityConfig:
    """Similarity calculation configuration."""
    column: str
    method: str  # 'jarowinkler', 'levenshtein', 'cosine', 'jaccard', 'exact'
    threshold: Optional[float] = None
    weight: float = 1.0
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    """Threshold configuration for match classification."""
    review: float = 0.7
    auto_merge: float = 0.9
    definite_no_match: float = 0.3


@dataclass
class SurvivorshipRule:
    """Survivorship rule configuration."""
    column: str
    strategy: str  # 'most_recent', 'longest_string', 'source_priority', etc.
    source_order: Optional[List[str]] = None
    threshold: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorityCondition:
    """Priority condition for survivorship."""
    column: str
    value: Any
    priority: int = 1


@dataclass
class MDMConfig:
    """Main MDM configuration class."""
    
    # Data source configuration
    source: DatabaseConfig
    
    # Blocking configuration
    blocking: BlockingConfig
    
    # Similarity configurations
    similarity: List[SimilarityConfig]
    
    # Threshold configuration
    thresholds: ThresholdConfig
    
    # Survivorship configuration
    survivorship_rules: List[SurvivorshipRule] = field(default_factory=list)
    priority_conditions: List[PriorityCondition] = field(default_factory=list)
    
    # Unique ID configuration
    unique_id_columns: List[str] = field(default_factory=list)
    
    # Output configuration
    output_path: str = "./output"
    
    # Performance configuration
    batch_size: int = 10000
    n_jobs: int = -1
    use_multiprocessing: bool = True
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'MDMConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            logger.info(f"Loading configuration from {config_path}")
            return cls.from_dict(config_data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> 'MDMConfig':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            logger.info(f"Loading configuration from {config_path}")
            return cls.from_dict(config_data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'MDMConfig':
        """Create configuration from dictionary."""
        try:
            # Validate configuration
            cls._validate_config(config_data)
            
            # Parse source configuration
            source_data = config_data.get('source', {})
            source = DatabaseConfig(**source_data)
            
            # Parse blocking configuration
            blocking_data = config_data.get('blocking', {})
            blocking = BlockingConfig(**blocking_data)
            
            # Parse similarity configurations
            similarity_list = config_data.get('similarity', [])
            similarity = [SimilarityConfig(**sim) for sim in similarity_list]
            
            # Parse threshold configuration
            threshold_data = config_data.get('thresholds', {})
            thresholds = ThresholdConfig(**threshold_data)
            
            # Parse survivorship rules
            survivorship_data = config_data.get('survivorship', {}).get('rules', [])
            survivorship_rules = [SurvivorshipRule(**rule) for rule in survivorship_data]
            
            # Parse priority conditions
            priority_data = config_data.get('priority_rule', {}).get('conditions', [])
            priority_conditions = [PriorityCondition(**cond) for cond in priority_data]
            
            # Parse unique ID columns
            unique_id_data = config_data.get('unique_id', {})
            unique_id_columns = unique_id_data.get('columns', [])
            
            # Other configurations
            output_path = config_data.get('output_path', './output')
            batch_size = config_data.get('batch_size', 10000)
            n_jobs = config_data.get('n_jobs', -1)
            use_multiprocessing = config_data.get('use_multiprocessing', True)
            log_level = config_data.get('log_level', 'INFO')
            log_file = config_data.get('log_file')
            options = config_data.get('options', {})
            
            return cls(
                source=source,
                blocking=blocking,
                similarity=similarity,
                thresholds=thresholds,
                survivorship_rules=survivorship_rules,
                priority_conditions=priority_conditions,
                unique_id_columns=unique_id_columns,
                output_path=output_path,
                batch_size=batch_size,
                n_jobs=n_jobs,
                use_multiprocessing=use_multiprocessing,
                log_level=log_level,
                log_file=log_file,
                options=options
            )
            
        except Exception as e:
            logger.error(f"Failed to create configuration from dict: {e}")
            raise
    
    @staticmethod
    def _validate_config(config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        schema = {
            'source': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'type': {'type': 'string', 'required': True, 
                           'allowed': ['sqlite', 'postgresql', 'sqlserver', 'duckdb', 'csv']},
                    'host': {'type': 'string'},
                    'port': {'type': 'integer'},
                    'database': {'type': 'string'},
                    'username': {'type': 'string'},
                    'password': {'type': 'string'},
                    'schema': {'type': 'string'},
                    'table': {'type': 'string'},
                    'file_path': {'type': 'string'},
                    'connection_string': {'type': 'string'},
                }
            },
            'blocking': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'columns': {'type': 'list', 'required': True, 'minlength': 1},
                    'method': {'type': 'string', 'allowed': ['fuzzy', 'exact', 'sorted_neighbourhood']},
                    'threshold': {'type': 'float', 'min': 0, 'max': 1},
                }
            },
            'similarity': {
                'type': 'list',
                'required': True,
                'minlength': 1,
                'schema': {
                    'type': 'dict',
                    'schema': {
                        'column': {'type': 'string', 'required': True},
                        'method': {'type': 'string', 'required': True,
                                 'allowed': ['jarowinkler', 'levenshtein', 'cosine', 'jaccard', 'exact']},
                        'threshold': {'type': 'float', 'min': 0, 'max': 1},
                        'weight': {'type': 'float', 'min': 0},
                    }
                }
            },
            'thresholds': {
                'type': 'dict',
                'schema': {
                    'review': {'type': 'float', 'min': 0, 'max': 1},
                    'auto_merge': {'type': 'float', 'min': 0, 'max': 1},
                    'definite_no_match': {'type': 'float', 'min': 0, 'max': 1},
                }
            }
        }
        
        validator = Validator(schema)
        if not validator.validate(config_data):
            raise ValueError(f"Configuration validation failed: {validator.errors}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'source': {
                'type': self.source.type,
                'host': self.source.host,
                'port': self.source.port,
                'database': self.source.database,
                'username': self.source.username,
                'password': self.source.password,
                'schema': self.source.schema,
                'table': self.source.table,
                'file_path': self.source.file_path,
                'connection_string': self.source.connection_string,
                'options': self.source.options,
            },
            'blocking': {
                'columns': self.blocking.columns,
                'method': self.blocking.method,
                'threshold': self.blocking.threshold,
                'window_size': self.blocking.window_size,
                'options': self.blocking.options,
            },
            'similarity': [
                {
                    'column': sim.column,
                    'method': sim.method,
                    'threshold': sim.threshold,
                    'weight': sim.weight,
                    'options': sim.options,
                }
                for sim in self.similarity
            ],
            'thresholds': {
                'review': self.thresholds.review,
                'auto_merge': self.thresholds.auto_merge,
                'definite_no_match': self.thresholds.definite_no_match,
            },
            'survivorship': {
                'rules': [
                    {
                        'column': rule.column,
                        'strategy': rule.strategy,
                        'source_order': rule.source_order,
                        'threshold': rule.threshold,
                        'options': rule.options,
                    }
                    for rule in self.survivorship_rules
                ]
            },
            'priority_rule': {
                'conditions': [
                    {
                        'column': cond.column,
                        'value': cond.value,
                        'priority': cond.priority,
                    }
                    for cond in self.priority_conditions
                ]
            },
            'unique_id': {
                'columns': self.unique_id_columns,
            },
            'output_path': self.output_path,
            'batch_size': self.batch_size,
            'n_jobs': self.n_jobs,
            'use_multiprocessing': self.use_multiprocessing,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'options': self.options,
        }
    
    def save_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        try:
            config_dict = self.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
    
    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        try:
            config_dict = self.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise


def create_sample_config() -> MDMConfig:
    """Create a sample configuration for testing."""
    source = DatabaseConfig(
        type="csv",
        file_path="sample_data.csv"
    )
    
    blocking = BlockingConfig(
        columns=["firstname", "lastname"],
        method="fuzzy",
        threshold=0.8
    )
    
    similarity = [
        SimilarityConfig(column="firstname", method="jarowinkler", weight=2.0),
        SimilarityConfig(column="lastname", method="jarowinkler", weight=2.0),
        SimilarityConfig(column="address", method="levenshtein", weight=1.0),
        SimilarityConfig(column="city", method="exact", weight=1.0),
        SimilarityConfig(column="zip", method="exact", weight=1.0),
    ]
    
    thresholds = ThresholdConfig(
        review=0.7,
        auto_merge=0.9,
        definite_no_match=0.3
    )
    
    survivorship_rules = [
        SurvivorshipRule(column="last_updated", strategy="most_recent"),
        SurvivorshipRule(column="source", strategy="source_priority", 
                        source_order=["system1", "system2", "manual"]),
        SurvivorshipRule(column="address", strategy="longest_string"),
    ]
    
    priority_conditions = [
        PriorityCondition(column="is_verified", value=True, priority=1),
        PriorityCondition(column="confidence_score", value=100, priority=2),
    ]
    
    return MDMConfig(
        source=source,
        blocking=blocking,
        similarity=similarity,
        thresholds=thresholds,
        survivorship_rules=survivorship_rules,
        priority_conditions=priority_conditions,
        unique_id_columns=["firstname", "lastname", "birthdate"],
        output_path="./mdm_output"
    )