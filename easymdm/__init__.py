"""
EasyMDM Advanced - Enhanced Master Data Management Package

A comprehensive solution for data deduplication, record linkage, and master data management
with support for multiple databases and advanced similarity algorithms.
"""

__version__ = "2.0.0"
__author__ = "MDM Team"
__email__ = "mdm@example.com"

from .core.engine import MDMEngine
from .core.config import MDMConfig
from .database.connector import DatabaseConnector
from .similarity.matcher import SimilarityMatcher
from .clustering.clusterer import RecordClusterer
from .survivorship.resolver import SurvivorshipResolver

# Main components export
__all__ = [
    "MDMEngine",
    "MDMConfig", 
    "DatabaseConnector",
    "SimilarityMatcher",
    "RecordClusterer",
    "SurvivorshipResolver",
]

# Version info
VERSION_INFO = {
    "major": 2,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Get the current version string."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "easymdm-advanced",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Enhanced Master Data Management Package",
        "supported_databases": ["SQLite", "PostgreSQL", "SQL Server", "DuckDB", "CSV"],
        "supported_algorithms": ["Jaro-Winkler", "Levenshtein", "Cosine", "Jaccard", "Exact"],
    }

# Configuration
import logging
import warnings

# Configure default logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')