# example_usage.py
"""
EasyMDM Advanced - Complete Usage Examples

This file demonstrates various usage patterns and advanced features
of EasyMDM Advanced for different scenarios and data sources.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# EasyMDM Advanced imports
from easymdm import MDMEngine, MDMConfig
from easymdm.core.config import (
    DatabaseConfig, BlockingConfig, SimilarityConfig, 
    ThresholdConfig, SurvivorshipRule, PriorityCondition
)
from easymdm.database.connector import DatabaseConnector
from easymdm.similarity.matcher import SimilarityMatcher, compute_string_similarity
from easymdm.clustering.clusterer import RecordClusterer
from easymdm.survivorship.resolver import SurvivorshipResolver

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_csv_basic_usage():
    """Example 1: Basic CSV processing with minimal configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic CSV Processing")
    print("="*80)
    
    # Create sample CSV data
    sample_data = {
        'first_name': ['John', 'Jon', 'Jane', 'John', 'Janet'],
        'last_name': ['Smith', 'Smith', 'Doe', 'Smyth', 'Doe'],
        'address': ['123 Main St', '123 Main St', '456 Oak Ave', '123 Main Street', '456 Oak Avenue'],
        'city': ['New York', 'New York', 'Boston', 'New York', 'Boston'],
        'phone': ['555-1234', '555-1234', '555-5678', '(555) 1234', '555.5678'],
        'email': ['john@email.com', 'john@email.com', 'jane@email.com', 'john.smith@email.com', 'jane.doe@email.com'],
        'source': ['CRM', 'Import', 'Manual', 'CRM', 'Manual'],
        'last_updated': ['2023-01-15', '2023-01-10', '2023-02-20', '2023-01-20', '2023-02-25']
    }
    
    df = pd.DataFrame(sample_data)
    csv_file = 'sample_customers.csv'
    df.to_csv(csv_file, index=False)
    print(f"Created sample CSV file: {csv_file}")
    
    # Create configuration programmatically
    config = MDMConfig(
        source=DatabaseConfig(
            type='csv',
            file_path=csv_file
        ),
        blocking=BlockingConfig(
            columns=['first_name', 'last_name'],
            method='exact'
        ),
        similarity=[
            SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
            SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
            SimilarityConfig(column='address', method='levenshtein', weight=1.5),
            SimilarityConfig(column='phone', method='levenshtein', weight=1.0),
            SimilarityConfig(column='email', method='exact', weight=1.5),
        ],
        thresholds=ThresholdConfig(
            review=0.7,
            auto_merge=0.85,
            definite_no_match=0.3
        ),
        survivorship_rules=[
            SurvivorshipRule(column='last_updated', strategy='most_recent'),
            SurvivorshipRule(column='source', strategy='source_priority', 
                           source_order=['CRM', 'Manual', 'Import']),
            SurvivorshipRule(column='address', strategy='longest_string'),
        ],
        priority_conditions=[],
        unique_id_columns=['first_name', 'last_name'],
        output_path='./output_example1'
    )
    
    # Run MDM processing
    engine = MDMEngine(config)
    result = engine.process()
    
    print(f"\n✅ Processing completed!")
    print(f"   Input records: {len(df)}")
    print(f"   Golden records: {len(result.golden_records)}")
    print(f"   Processing time: {result.execution_time:.2f} seconds")
    print(f"   Output files: {len(result.output_files)}")
    
    # Display results
    print("\n📊 Golden Records:")
    if not result.golden_records.empty:
        display_cols = ['first_name', 'last_name', 'address', 'similar_record_ids', 'logic']
        available_cols = [col for col in display_cols if col in result.golden_records.columns]
        print(result.golden_records[available_cols].to_string(index=False))
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)


def example_2_advanced_similarity():
    """Example 2: Advanced similarity configuration and testing."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Advanced Similarity Configuration")
    print("="*80)
    
    # Test different similarity methods
    test_pairs = [
        ("John Smith", "Jon Smith"),
        ("123 Main Street", "123 Main St"),
        ("john@email.com", "john@gmail.com"),
        ("New York", "NY"),
        ("555-1234", "(555) 123-4567")
    ]
    
    methods = ['jarowinkler', 'levenshtein', 'cosine', 'exact', 'fuzzy_ratio']
    
    print("\n🔍 Similarity Method Comparison:")
    print("-" * 60)
    for str1, str2 in test_pairs:
        print(f"\nComparing: '{str1}' vs '{str2}'")
        for method in methods:
            try:
                score = compute_string_similarity(str1, str2, method)
                print(f"  {method:15}: {score:.3f}")
            except Exception as e:
                print(f"  {method:15}: ERROR - {e}")
    
    # Create advanced similarity configuration
    similarity_configs = [
        SimilarityConfig(
            column='name', 
            method='jarowinkler', 
            weight=2.0,
            options={'lowercase': True, 'remove_extra_spaces': True}
        ),
        SimilarityConfig(
            column='address', 
            method='cosine', 
            weight=1.5,
            options={'ngram_range': 2, 'lowercase': True}
        ),
        SimilarityConfig(
            column='phone', 
            method='levenshtein', 
            weight=1.0,
            options={'remove_punctuation': True}
        )
    ]
    
    # Initialize similarity matcher
    matcher = SimilarityMatcher(similarity_configs)
    
    # Get method recommendations for sample data
    sample_df = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Bob Johnson'],
        'address': ['123 Main St, New York', '456 Oak Ave, Boston', '789 Pine Rd, Chicago'],
        'phone': ['555-1234', '(555) 567-8901', '555.234.5678'],
        'email': ['john@email.com', 'jane@company.com', 'bob@work.org']
    })
    
    recommendations = matcher.get_method_recommendations(sample_df)
    
    print(f"\n💡 Method Recommendations by Column Type:")
    print("-" * 50)
    for column, methods in recommendations.items():
        print(f"{column:15}: {', '.join(methods)}")


def example_3_postgresql_advanced():
    """Example 3: PostgreSQL with advanced configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 3: PostgreSQL Advanced Configuration")
    print("="*80)
    
    # Note: This example shows configuration only
    # Actual execution requires a PostgreSQL database
    
    config = MDMConfig(
        source=DatabaseConfig(
            type='postgresql',
            host='localhost',
            port=5432,
            database='customer_db',
            username='mdm_user',
            password='secure_password',
            schema='public',
            table='customers',
            options={
                'connect_timeout': 30,
                'application_name': 'EasyMDM_Processing'
            }
        ),
        blocking=BlockingConfig(
            columns=['first_name', 'last_name'],
            method='fuzzy',
            threshold=0.8,
            options={
                'similarity_method': 'jaro_winkler',
                'separator': ' '
            }
        ),
        similarity=[
            SimilarityConfig(
                column='first_name', 
                method='jarowinkler', 
                weight=2.0,
                threshold=0.7,
                options={'lowercase': True}
            ),
            SimilarityConfig(
                column='last_name', 
                method='jarowinkler', 
                weight=2.0,
                threshold=0.7,
                options={'lowercase': True}
            ),
            SimilarityConfig(
                column='address', 
                method='cosine', 
                weight=1.5,
                threshold=0.6,
                options={'ngram_range': 2}
            ),
            SimilarityConfig(
                column='phone', 
                method='levenshtein', 
                weight=1.0,
                threshold=0.8,
                options={'remove_punctuation': True}
            )
        ],
        thresholds=ThresholdConfig(
            review=0.75,
            auto_merge=0.9,
            definite_no_match=0.3
        ),
        survivorship_rules=[
            SurvivorshipRule(
                column='updated_at', 
                strategy='most_recent'
            ),
            SurvivorshipRule(
                column='data_source', 
                strategy='source_priority',
                source_order=['MASTER_SYSTEM', 'CRM', 'IMPORT', 'MANUAL']
            ),
            SurvivorshipRule(
                column='address', 
                strategy='longest_string'
            ),
            SurvivorshipRule(
                column='confidence_score', 
                strategy='highest_value'
            )
        ],
        priority_conditions=[
            PriorityCondition(column='is_verified', value=True, priority=1),
            PriorityCondition(column='quality_score', value=100, priority=2),
            PriorityCondition(column='data_source', value='MASTER_SYSTEM', priority=3)
        ],
        unique_id_columns=['first_name', 'last_name', 'birth_date'],
        output_path='./postgresql_output',
        batch_size=10000,
        n_jobs=-1,
        use_multiprocessing=True,
        log_level='INFO'
    )
    
    # Save configuration to file
    config_file = 'postgresql_config.yaml'
    config.save_yaml(config_file)
    print(f"✅ Created PostgreSQL configuration: {config_file}")
    
    # Test configuration (without actual database connection)
    print("\n🔍 Configuration Validation:")
    print(f"   Source Type: {config.source.type}")
    print(f"   Blocking Method: {config.blocking.method}")
    print(f"   Similarity Configs: {len(config.similarity)}")
    print(f"   Survivorship Rules: {len(config.survivorship_rules)}")
    print(f"   Priority Conditions: {len(config.priority_conditions)}")
    
    # Show available database connectors
    available = DatabaseConnector.get_available_connectors()
    print(f"\n📊 Available Database Connectors: {', '.join(available)}")
    
    # Cleanup
    if os.path.exists(config_file):
        os.remove(config_file)


def example_4_performance_optimization():
    """Example 4: Performance optimization for large datasets."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Performance Optimization")
    print("="*80)
    
    # Create larger sample dataset
    import random
    import string
    import warnings
    from numba.core.errors import NumbaWarning
    
    def generate_name():
        first_names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
        return random.choice(first_names), random.choice(last_names)
    
    # Generate sample data with duplicates
    data = []
    for i in range(1000):  # Moderate size for example
        first, last = generate_name()
        
        # Add some variations to create duplicates
        if random.random() < 0.3:  # 30% chance of variation
            if random.random() < 0.5:
                first = first + random.choice(['', 'ny', 'nie'])  # Name variations
            else:
                last = last + random.choice(['', 'son', 'sen'])  # Surname variations
        
        data.append({
            'id': i + 1,
            'first_name': first,
            'last_name': last,
            'address': f"{random.randint(100, 999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} {random.choice(['St', 'Ave', 'Rd'])}",
            'city': random.choice(['New York', 'Boston', 'Chicago', 'Los Angeles']),
            'phone': f"555-{random.randint(1000, 9999)}",
            'score': random.randint(60, 100),
            'source': random.choice(['CRM', 'Import', 'Manual']),
            'created_at': f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
    
    df = pd.DataFrame(data)
    csv_file = 'large_sample.csv'
    df.to_csv(csv_file, index=False)
    
    # ------------------ تنظيف البيانات وتجاهل تحذيرات Numba ------------------
    warnings.simplefilter('ignore', category=NumbaWarning)
    df['first_name'] = df['first_name'].astype(str).str.lower().str.strip()
    df['last_name'] = df['last_name'].astype(str).str.lower().str.strip()
    # -------------------------------------------------------------------------
    
    print(f"📊 Generated sample dataset: {len(df):,} records")
    
    # Performance-optimized configuration
    config = MDMConfig(
        source=DatabaseConfig(
            type='csv',
            file_path=csv_file,
            options={
                'encoding': 'utf-8'
            }
        ),
        blocking=BlockingConfig(
            columns=['first_name', 'last_name'],
            method='exact',  # Faster than fuzzy for large datasets
            options={'separator': ' '}
        ),
        similarity=[
            SimilarityConfig(column='first_name', method='jarowinkler', weight=2.0),
            SimilarityConfig(column='last_name', method='jarowinkler', weight=2.0),
            SimilarityConfig(column='address', method='levenshtein', weight=1.0),
        ],
        thresholds=ThresholdConfig(
            review=0.8,
            auto_merge=0.9,
            definite_no_match=0.4
        ),
        survivorship_rules=[
            SurvivorshipRule(column='created_at', strategy='most_recent'),
            SurvivorshipRule(column='score', strategy='highest_value'),
        ],
        output_path='./performance_output',
        batch_size=5000,      # Smaller batches for memory efficiency
        n_jobs=4,             # Limit parallel jobs
        use_multiprocessing=True,
        options={
            'use_chunking': True,
            'chunk_size': 10000,
            'show_progress': True
        }
    )
    
    # Run with performance monitoring
    import time
    
    print("\n🚀 Starting performance-optimized processing...")
    start_time = time.time()
    
    engine = MDMEngine(config)
    result = engine.process()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n⚡ Performance Results:")
    print(f"   Input Records: {len(df):,}")
    print(f"   Golden Records: {len(result.golden_records):,}")
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"   Records/Second: {len(df)/processing_time:.1f}")
    print(f"   Memory Usage: {result.processing_stats.get('memory_usage', 'N/A')}")
    
    # Performance metrics
    if 'candidate_pairs' in result.processing_stats:
        pairs = result.processing_stats['candidate_pairs']
        print(f"   Candidate Pairs: {pairs:,}")
        if pairs > 0:
            print(f"   Pairs/Second: {pairs/processing_time:.1f}")
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)


def example_5_custom_survivorship():
    """Example 5: Custom survivorship strategies and priority rules."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Survivorship Strategies")
    print("="*80)
    
    # Create sample data with quality indicators
    sample_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'first_name': ['John', 'John', 'Jane', 'Jane', 'Robert'],
        'last_name': ['Smith', 'Smith', 'Doe', 'Doe', 'Johnson'],
        'email': ['john@email.com', 'j.smith@company.com', 'jane@work.com', 'jane.doe@email.com', 'rob@email.com'],
        'phone': ['555-1234', '555-1234', '555-5678', '(555) 567-8901', '555-9999'],
        'address': ['123 Main St', '123 Main Street Apt 2', '456 Oak Ave', '456 Oak Avenue Unit 1', '789 Pine Rd'],
        'data_source': ['CRM', 'ERP', 'Manual', 'Import', 'CRM'],
        'quality_score': [95, 85, 90, 75, 98],
        'is_verified': [True, False, True, False, True],
        'confidence_level': ['high', 'medium', 'high', 'low', 'high'],
        'last_updated': ['2023-03-15', '2023-02-10', '2023-03-20', '2023-01-05', '2023-03-25'],
        'verification_date': ['2023-03-10', None, '2023-03-18', None, '2023-03-20']
    }
    
    df = pd.DataFrame(sample_data)
    csv_file = 'survivorship_sample.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"📊 Sample data for survivorship testing:")
    print(df[['customer_id', 'first_name', 'last_name', 'data_source', 'quality_score', 'is_verified']].to_string(index=False))
    
    # Advanced survivorship configuration
    config = MDMConfig(
        source=DatabaseConfig(
            type='csv',
            file_path=csv_file
        ),
        blocking=BlockingConfig(
            columns=['first_name', 'last_name'],
            method='exact'
        ),
        similarity=[
            SimilarityConfig(column='first_name', method='exact', weight=1.0),
            SimilarityConfig(column='last_name', method='exact', weight=1.0),
        ],
        thresholds=ThresholdConfig(
            review=0.5,
            auto_merge=0.8,
            definite_no_match=0.2
        ),
        survivorship_rules=[
            # Most recent timestamp
            SurvivorshipRule(
                column='last_updated', 
                strategy='most_recent'
            ),
            # Source priority with specific ordering
            SurvivorshipRule(
                column='data_source', 
                strategy='source_priority',
                source_order=['CRM', 'ERP', 'Manual', 'Import']
            ),
            # Longest address (more complete information)
            SurvivorshipRule(
                column='address', 
                strategy='longest_string'
            ),
            # Highest quality score
            SurvivorshipRule(
                column='quality_score', 
                strategy='highest_value'
            ),
            # Most frequent confidence level
            SurvivorshipRule(
                column='confidence_level', 
                strategy='most_frequent'
            ),
            # Most recent verification date
            SurvivorshipRule(
                column='verification_date', 
                strategy='most_recent'
            )
        ],
        priority_conditions=[
            # Verified records have highest priority
            PriorityCondition(column='is_verified', value=True, priority=1),
            # High quality scores have second priority
            PriorityCondition(column='quality_score', value=95, priority=2),
            # CRM system has third priority
            PriorityCondition(column='data_source', value='CRM', priority=3),
        ],
        unique_id_columns=['first_name', 'last_name'],
        output_path='./survivorship_output'
    )
    
    # Test survivorship resolver directly
    from easymdm.survivorship.resolver import SurvivorshipResolver
    
    resolver = SurvivorshipResolver(
        config.survivorship_rules,
        config.priority_conditions,
        config.unique_id_columns
    )
    
    # Test with sample cluster
    test_cluster = df[df['first_name'] == 'John'].copy()  # John Smith records
    
    print(f"\n🔍 Testing Survivorship on John Smith cluster:")
    print(test_cluster[['customer_id', 'data_source', 'quality_score', 'is_verified', 'last_updated']].to_string(index=False))
    
    result = resolver.resolve_cluster(test_cluster)
    
    print(f"\n✅ Survivorship Result:")
    print(f"   Survivor ID: {result.survivor_id}")
    print(f"   Resolution Logic: {result.resolution_logic}")
    print(f"   Confidence Score: {result.confidence_score:.2f}")
    
    if result.golden_record:
        print(f"\n📋 Golden Record Fields:")
        for key, value in result.golden_record.items():
            if key in ['first_name', 'last_name', 'data_source', 'quality_score', 'is_verified']:
                print(f"   {key}: {value}")
    
    # Run full MDM process
    print(f"\n🚀 Running full MDM process...")
    engine = MDMEngine(config)
    mdm_result = engine.process()
    
    print(f"\n📊 Final Results:")
    print(f"   Input records: {len(df)}")
    print(f"   Golden records: {len(mdm_result.golden_records)}")
    
    if not mdm_result.golden_records.empty:
        print(f"\n🏆 Golden Records Summary:")
        summary_cols = ['first_name', 'last_name', 'data_source', 'quality_score', 'logic']
        available_cols = [col for col in summary_cols if col in mdm_result.golden_records.columns]
        print(mdm_result.golden_records[available_cols].to_string(index=False))
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)


def main():
    """Run all examples."""
    print("🚀 EasyMDM Advanced - Complete Usage Examples")
    print("=" * 80)
    
    try:
        # Create output directories
        os.makedirs('./output_example1', exist_ok=True)
        os.makedirs('./postgresql_output', exist_ok=True)
        os.makedirs('./performance_output', exist_ok=True)
        os.makedirs('./survivorship_output', exist_ok=True)
        
        # Run examples
        example_1_csv_basic_usage()
        example_2_advanced_similarity()
        example_3_postgresql_advanced()
        example_4_performance_optimization()
        example_5_custom_survivorship()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("📁 Check the output directories for generated files:")
        print("   • ./output_example1/")
        print("   • ./performance_output/")
        print("   • ./survivorship_output/")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()