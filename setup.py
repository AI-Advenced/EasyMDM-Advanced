"""
Setup configuration for EasyMDM Advanced - Enhanced Master Data Management Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="easymdm-advanced",
    version="2.0.0",
    author="MDM Team",
    author_email="mdm@example.com",
    description="Advanced Master Data Management package with multi-database support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/easymdm-advanced",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "networkx>=2.8",
        
        # Record linkage and similarity
        "recordlinkage>=0.15.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.12.2",
        "jellyfish>=0.9.0",
        "textdistance>=4.5.0",
        
        # Database connectors
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL
        "pyodbc>=4.0.30",          # SQL Server
        "duckdb>=0.8.0",           # DuckDB
        
        # Performance and utilities
        "tqdm>=4.64.0",
        "joblib>=1.2.0",
        "numba>=0.56.0",
        "scikit-learn>=1.1.0",
        
        # Data validation
        "cerberus>=1.3.4",
        "jsonschema>=4.17.0",
        
        # Logging and monitoring
        "colorlog>=6.7.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "easymdm=easymdm.cli:main",
            "easymdm-config=easymdm.config_generator:main",
            "easymdm-monitor=easymdm.monitor:main",
        ],
    },
    include_package_data=True,
    package_data={
        "easymdm": [
            "templates/*.yaml",
            "schemas/*.json",
            "static/*",
        ],
    },
)