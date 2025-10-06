#!/usr/bin/env python3
"""
EasyMDM Advanced - Interactive Demo

This script demonstrates the enhanced capabilities of EasyMDM Advanced
with a practical customer data deduplication example.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data():
    """Create realistic customer data with duplicates for demonstration."""
    logger.info("Creating demo customer dataset...")
    
    # Base customer profiles
    base_customers = [
        {
            'first_name': 'John', 'last_name': 'Smith', 
            'address': '123 Main Street', 'city': 'New York', 'state': 'NY', 'zip': '10001',
            'phone': '555-0123', 'email': 'john.smith@email.com'
        },
        {
            'first_name': 'Jane', 'last_name': 'Johnson', 
            'address': '456 Oak Avenue', 'city': 'Boston', 'state': 'MA', 'zip': '02101',
            'phone': '555-0456', 'email': 'jane.johnson@company.com'
        },
        {
            'first_name': 'Robert', 'last_name': 'Williams', 
            'address': '789 Pine Road', 'city': 'Chicago', 'state': 'IL', 'zip': '60601',
            'phone': '555-0789', 'email': 'rob.williams@work.org'
        },
        {
            'first_name': 'Mary', 'last_name': 'Davis', 
            'address': '321 Elm Street', 'city': 'Los Angeles', 'state': 'CA', 'zip': '90001',
            'phone': '555-0321', 'email': 'mary.davis@gmail.com'
        },
        {
            'first_name': 'Michael', 'last_name': 'Brown', 
            'address': '654 Maple Lane', 'city': 'Houston', 'state': 'TX', 'zip': '77001',
            'phone': '555-0654', 'email': 'michael.brown@yahoo.com'
        }
    ]
    
    # Generate variations and duplicates
    all_records = []
    record_id = 1
    
    for base_customer in base_customers:
        # Original record
        record = base_customer.copy()
        record.update({
            'customer_id': record_id,
            'source': 'MASTER_SYSTEM',
            'quality_score': random.randint(90, 100),
            'is_verified': True,
            'confidence_level': 'high',
            'created_date': '2023-01-15',
            'last_updated': '2023-03-15'
        })
        all_records.append(record)
        record_id += 1
        
        # Create variations (duplicates with slight differences)
        num_variations = random.randint(1, 3)
        
        for i in range(num_variations):
            variation = base_customer.copy()
            
            # Add variations to create realistic duplicates
            if random.random() < 0.3:  # Name variations
                if random.random() < 0.5:
                    # Nickname variations
                    name_variants = {
                        'John': 'Jon', 'Jane': 'Janey', 'Robert': 'Bob', 
                        'Mary': 'Marie', 'Michael': 'Mike'
                    }
                    if variation['first_name'] in name_variants:
                        variation['first_name'] = name_variants[variation['first_name']]
                else:
                    # Spelling variations
                    variation['last_name'] = variation['last_name'] + random.choice(['', 'son', 's'])
            
            if random.random() < 0.4:  # Address variations
                addr_variants = {
                    'Street': 'St', 'Avenue': 'Ave', 'Road': 'Rd', 
                    'Lane': 'Ln', '123': '123A'
                }
                for old, new in addr_variants.items():
                    if old in variation['address']:
                        variation['address'] = variation['address'].replace(old, new)
                        break
            
            if random.random() < 0.5:  # Phone format variations
                phone = variation['phone'].replace('555-', '(555) ').replace('-', '-')
                if random.random() < 0.5:
                    phone = phone.replace('(555) ', '555.').replace('-', '.')
                variation['phone'] = phone
            
            if random.random() < 0.3:  # Email variations
                email_parts = variation['email'].split('@')
                if random.random() < 0.5:
                    # Different domain
                    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'company.com']
                    email_parts[1] = random.choice(domains)
                else:
                    # Add numbers or dots
                    email_parts[0] = email_parts[0].replace('.', '') + str(random.randint(1, 99))
                variation['email'] = '@'.join(email_parts)
            
            # Add metadata for variation
            sources = ['CRM_SYSTEM', 'ERP_SYSTEM', 'IMPORT_BATCH', 'MANUAL_ENTRY']
            variation.update({
                'customer_id': record_id,
                'source': random.choice(sources),
                'quality_score': random.randint(60, 95),
                'is_verified': random.choice([True, False]),
                'confidence_level': random.choice(['high', 'medium', 'low']),
                'created_date': (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'last_updated': (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
            })
            
            all_records.append(variation)
            record_id += 1
    
    # Add some additional unique records
    additional_customers = [
        {
            'customer_id': record_id + i,
            'first_name': name[0], 'last_name': name[1],
            'address': f"{random.randint(100, 999)} {random.choice(['First', 'Second', 'Third'])} {random.choice(['Street', 'Avenue', 'Road'])}",
            'city': random.choice(['Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas']),
            'state': random.choice(['AZ', 'PA', 'TX', 'CA', 'TX']),
            'zip': f"{random.randint(10000, 99999)}",
            'phone': f"555-{random.randint(1000, 9999)}",
            'email': f"{name[0].lower()}.{name[1].lower()}@{random.choice(['gmail.com', 'yahoo.com', 'email.com'])}",
            'source': random.choice(['CRM_SYSTEM', 'IMPORT_BATCH']),
            'quality_score': random.randint(70, 95),
            'is_verified': random.choice([True, False]),
            'confidence_level': random.choice(['high', 'medium']),
            'created_date': (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'last_updated': (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        }
        for i, name in enumerate([
            ('Sarah', 'Wilson'), ('David', 'Taylor'), ('Lisa', 'Anderson'), 
            ('James', 'Thomas'), ('Jennifer', 'Jackson')
        ])
    ]
    
    all_records.extend(additional_customers)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Shuffle records to simulate real-world data
    df = df.sample(frac=1).reset_index(drop=True)
    
    logger.info(f"Created demo dataset with {len(df)} records")
    logger.info(f"Expected duplicates: ~{len(base_customers)} groups")
    
    return df


def run_demo():
    """Run the complete EasyMDM Advanced demo."""
    print("🚀 EasyMDM Advanced - Interactive Demo")
    print("=" * 80)
    
    # Create demo data
    df = create_demo_data()
    csv_file = 'demo_customers.csv'
    df.to_csv(csv_file, index=False)
    
    # Display sample data
    print("\n📊 Sample Input Data (first 10 records):")
    print("-" * 80)
    display_cols = ['customer_id', 'first_name', 'last_name', 'address', 'source', 'quality_score', 'is_verified']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Show data statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Unique Names: {df[['first_name', 'last_name']].drop_duplicates().shape[0]}")
    print(f"   Data Sources: {', '.join(df['source'].unique())}")
    print(f"   Verified Records: {df['is_verified'].sum()}/{len(df)}")
    print(f"   Avg Quality Score: {df['quality_score'].mean():.1f}")
    
    print(f"\n🔍 Potential Duplicates Analysis:")
    # Analyze potential duplicates by grouping similar names
    name_groups = df.groupby(['first_name', 'last_name']).size()
    duplicates = name_groups[name_groups > 1]
    print(f"   Exact Name Matches: {len(duplicates)} groups")
    print(f"   Records in Duplicate Groups: {duplicates.sum()}")
    
    if len(duplicates) > 0:
        print(f"\n   Top Duplicate Groups:")
        for (first, last), count in duplicates.head().items():
            print(f"      {first} {last}: {count} records")
    
    # Import EasyMDM Advanced
    try:
        from easymdm import MDMEngine, MDMConfig
        from easymdm.core.config import (
            DatabaseConfig, BlockingConfig, SimilarityConfig, 
            ThresholdConfig, SurvivorshipRule, PriorityCondition
        )
        print(f"\n✅ EasyMDM Advanced imported successfully")
    except ImportError as e:
        print(f"\n❌ Failed to import EasyMDM Advanced: {e}")
        print("   Please ensure the package is installed: pip install -e .")
        return
    
    # Create advanced configuration
    print(f"\n⚙️ Creating Advanced MDM Configuration...")
    
    config = MDMConfig(
        source=DatabaseConfig(
            type='csv',
            file_path=csv_file
        ),
        blocking=BlockingConfig(
            columns=['first_name', 'last_name'],
            method='fuzzy',  # Use fuzzy blocking to catch variations
            threshold=0.8,
            options={'similarity_method': 'jaro_winkler'}
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
                options={'ngram_range': 2, 'lowercase': True}
            ),
            SimilarityConfig(
                column='phone', 
                method='levenshtein', 
                weight=1.0,
                threshold=0.8,
                options={'remove_punctuation': True}
            ),
            SimilarityConfig(
                column='email', 
                method='levenshtein', 
                weight=1.5,
                threshold=0.7,
                options={'lowercase': True}
            )
        ],
        thresholds=ThresholdConfig(
            review=0.75,     # Pairs needing manual review
            auto_merge=0.9,  # Automatic merge threshold
            definite_no_match=0.3
        ),
        survivorship_rules=[
            SurvivorshipRule(
                column='last_updated',
                strategy='most_recent'
            ),
            SurvivorshipRule(
                column='source',
                strategy='source_priority',
                source_order=['MASTER_SYSTEM', 'CRM_SYSTEM', 'ERP_SYSTEM', 'MANUAL_ENTRY', 'IMPORT_BATCH']
            ),
            SurvivorshipRule(
                column='address',
                strategy='longest_string'
            ),
            SurvivorshipRule(
                column='quality_score',
                strategy='highest_value'
            ),
            SurvivorshipRule(
                column='email',
                strategy='source_priority',
                source_order=['MASTER_SYSTEM', 'CRM_SYSTEM', 'MANUAL_ENTRY']
            )
        ],
        priority_conditions=[
            PriorityCondition(column='is_verified', value=True, priority=1),
            PriorityCondition(column='quality_score', value=95, priority=2),
            PriorityCondition(column='source', value='MASTER_SYSTEM', priority=3)
        ],
        unique_id_columns=['first_name', 'last_name'],
        output_path='./demo_output',
        batch_size=1000,
        n_jobs=-1,
        log_level='INFO'
    )
    
    print(f"   ✓ Configured fuzzy blocking on names")
    print(f"   ✓ Configured 5 similarity algorithms")
    print(f"   ✓ Configured 4 survivorship rules")
    print(f"   ✓ Configured 3 priority conditions")
    
    # Initialize and test MDM engine
    print(f"\n🔧 Initializing MDM Engine...")
    
    engine = MDMEngine(config)
    
    # Test configuration
    print(f"   Testing configuration...")
    test_results = engine.test_configuration()
    
    if all(test_results.values()):
        print(f"   ✅ Configuration test passed")
    else:
        print(f"   ⚠️ Configuration test warnings:")
        for component, result in test_results.items():
            if not result:
                print(f"      - {component}: FAILED")
    
    # Profile data
    print(f"\n📊 Profiling Input Data...")
    profile = engine.get_data_profile()
    
    if profile:
        print(f"   Records: {profile.get('total_records', 0):,}")
        print(f"   Columns: {profile.get('total_columns', 0)}")
        print(f"   Memory Usage: {profile.get('memory_usage', 0)/1024/1024:.1f} MB")
        
        # Show null percentages for key columns
        null_stats = profile.get('column_statistics', {})
        key_columns = ['first_name', 'last_name', 'address', 'phone', 'email']
        print(f"   Data Completeness:")
        for col in key_columns:
            if col in null_stats:
                null_pct = null_stats[col].get('null_percentage', 0)
                print(f"      {col}: {100-null_pct:.1f}% complete")
    
    # Run MDM processing
    print(f"\n🚀 Executing MDM Processing...")
    print(f"   This may take a few seconds...")
    
    import time
    start_time = time.time()
    
    try:
        result = engine.process()
        processing_time = time.time() - start_time
        
        print(f"\n✅ MDM Processing Completed Successfully!")
        print(f"   Processing Time: {processing_time:.2f} seconds")
        print(f"   Input Records: {len(df):,}")
        print(f"   Golden Records: {len(result.golden_records):,}")
        print(f"   Reduction: {len(df) - len(result.golden_records):,} duplicates removed")
        print(f"   Efficiency: {(1 - len(result.golden_records)/len(df))*100:.1f}% reduction")
        
        # Show processing statistics
        stats = result.processing_stats
        print(f"\n📈 Processing Statistics:")
        if 'candidate_pairs' in stats:
            print(f"   Candidate Pairs Generated: {stats['candidate_pairs']:,}")
        if 'clusters_found' in stats:
            print(f"   Clusters Found: {stats['clusters_found']:,}")
        
        # Display sample golden records
        print(f"\n🏆 Sample Golden Records:")
        print("-" * 80)
        if not result.golden_records.empty:
            display_cols = ['first_name', 'last_name', 'address', 'source', 'similar_record_ids', 'logic']
            available_cols = [col for col in display_cols if col in result.golden_records.columns]
            sample_size = min(8, len(result.golden_records))
            print(result.golden_records[available_cols].head(sample_size).to_string(index=False))
        
        # Show output files
        if result.output_files:
            print(f"\n📁 Output Files Generated:")
            for file_path in result.output_files:
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   ✓ {os.path.basename(file_path)} ({size:,} bytes)")
                else:
                    print(f"   ✗ {os.path.basename(file_path)} (not found)")
        
        # Analyze results
        print(f"\n🔍 Results Analysis:")
        
        # Count records by resolution logic
        if 'logic' in result.golden_records.columns:
            logic_counts = result.golden_records['logic'].value_counts()
            print(f"   Resolution Methods:")
            for logic_type, count in logic_counts.items():
                print(f"      {logic_type}: {count} records")
        
        # Count records by source system
        if 'source' in result.golden_records.columns:
            source_counts = result.golden_records['source'].value_counts()
            print(f"   Records by Source System:")
            for source, count in source_counts.items():
                print(f"      {source}: {count} records")
        
        # Show duplicate groups found
        if 'similar_record_ids' in result.golden_records.columns:
            merged_records = result.golden_records[result.golden_records['similar_record_ids'] != '']
            if not merged_records.empty:
                print(f"\n🔗 Duplicate Groups Merged:")
                for idx, row in merged_records.head(5).iterrows():
                    name = f"{row.get('first_name', '')} {row.get('last_name', '')}"
                    similar_ids = row['similar_record_ids']
                    record_count = len(similar_ids.split('|')) if similar_ids else 1
                    print(f"      {name}: {record_count} records merged")
        
        print(f"\n🎯 Demo Summary:")
        print(f"   ✅ Successfully processed {len(df):,} customer records")
        print(f"   ✅ Identified and merged {len(df) - len(result.golden_records):,} duplicate records")
        print(f"   ✅ Generated {len(result.golden_records):,} clean golden records")
        print(f"   ✅ Used advanced similarity algorithms and survivorship rules")
        print(f"   ✅ Processing completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ MDM Processing Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"\n🧹 Cleaned up demo files")
    
    print(f"\n" + "=" * 80)
    print("✨ EasyMDM Advanced Demo Completed!")
    print("   Check the ./demo_output/ directory for detailed results.")
    print("   Try running with your own data using: easymdm process --config config.yaml")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()