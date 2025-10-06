# EasyMDM Advanced

**Enhanced Master Data Management (MDM) Package with Multi-Database Support**

EasyMDM Advanced is a comprehensive Python package for Master Data Management (MDM), record linkage, and data deduplication. It provides advanced algorithms for similarity matching, flexible survivorship rules, and support for multiple database systems.



<img width="1216" height="864" alt="image" src="https://github.com/user-attachments/assets/b894ebb2-ce88-4a6a-bb81-0f52231dda29" />

---

## 🚀 Key Features

### Multi-Database Support

* **PostgreSQL** – Full support with connection pooling and optimization
* **SQL Server** – Native support via ODBC drivers
* **SQLite** – Lightweight embedded database support
* **DuckDB** – High-performance analytical database
* **CSV Files** – Direct file processing with optimized performance

### Advanced Similarity Algorithms

* **Jaro-Winkler** – Optimized for names and short strings
* **Levenshtein** – Edit distance with normalization
* **Cosine Similarity** – TF-IDF based for longer text
* **Jaccard** – Set-based similarity with n-grams
* **Exact Match** – Optimized exact comparisons
* **FuzzyWuzzy** – Multiple fuzzy matching variants

### Intelligent Blocking Strategies

* **Exact Blocking** – Traditional exact key matching
* **Fuzzy Blocking** – Similarity-based candidate generation
* **Sorted Neighborhood** – Sliding window approach
* **RecordLinkage Integration** – Standard blocking methods

### Flexible Clustering Algorithms

* **Network-based** – Graph connectivity clustering (default)
* **Hierarchical** – Distance-based hierarchical clustering
* **DBSCAN** – Density-based clustering with noise detection

### Advanced Survivorship Resolution

* **Priority Rules** – Condition-based record selection
* **Most Recent** – Date/timestamp-based resolution
* **Source Priority** – Trust-based source ordering
* **Longest String** – Length-based text selection
* **Value-based** – Highest/lowest numeric values
* **Threshold-based** – Conditional value selection
* **Most Frequent** – Frequency-based resolution

### Performance Optimization

* **Parallel Processing** – Multi-core similarity computation
* **Vectorized Operations** – NumPy/Pandas optimization
* **Caching** – Intelligent similarity caching
* **Batch Processing** – Memory-efficient large dataset handling
* **Numba Integration** – JIT compilation for critical paths

---

## 📦 Installation

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy pyyaml networkx

# Similarity and record linkage
pip install recordlinkage fuzzywuzzy python-Levenshtein jellyfish textdistance

# Database connectors
pip install sqlalchemy psycopg2-binary pyodbc duckdb

# Performance and ML
pip install scikit-learn numba joblib

# Optional: Rich UI and visualization
pip install rich matplotlib seaborn plotly
```

### Install EasyMDM Advanced

```bash
# From source
git clone https://github.com/yourusername/easymdm-advanced.git
cd easymdm-advanced
pip install -e .

# Or from PyPI (when available)
pip install easymdm-advanced
```

---

## 🎯 Quick Start

### 1. Create Configuration

```bash
# PostgreSQL example
easymdm create-config --output config.yaml --database-type postgresql

# CSV example
easymdm create-config --output config.yaml --database-type csv --file-path data.csv
```

### 2. Test Connection

```bash
easymdm test-connection --config config.yaml --sample-size 10
easymdm validate-config --config config.yaml
```

### 3. Run MDM Processing

```bash
easymdm process --config config.yaml --output ./results --profile --test-config
```

### 4. Python API Usage

```python
from easymdm import MDMEngine, MDMConfig

# Load configuration
config = MDMConfig.from_yaml('config.yaml')

# Create and run MDM engine
engine = MDMEngine(config)

# Test configuration
test_results = engine.test_configuration()
print("Configuration test:", all(test_results.values()))

# Profile input data
profile = engine.get_data_profile()
print(f"Input records: {profile['total_records']:,}")

# Execute MDM processing
result = engine.process()
print(f"Golden records created: {len(result.golden_records):,}")
print(f"Processing time: {result.execution_time:.2f} seconds")
print(f"Output files: {result.output_files}")
```

---

## ⚙️ Configuration Examples

### Database Sources

**PostgreSQL**

```yaml
source:
  type: postgresql
  host: localhost
  port: 5432
  database: mydb
  username: user
  password: password
  schema: public
  table: customers
```

**SQL Server**

```yaml
source:
  type: sqlserver
  host: localhost
  port: 1433
  database: CustomerDB
  username: user
  password: password
  schema: dbo
  table: Customers
  options:
    driver: "ODBC Driver 17 for SQL Server"
```

**CSV Files**

```yaml
source:
  type: csv
  file_path: ./data/customers.csv
  options:
    encoding: utf-8
    delimiter: ","
    na_values: ["", "NULL", "N/A"]
```

---

### Similarity Configuration

```yaml
similarity:
  - column: first_name
    method: jarowinkler
    weight: 2.0
    threshold: 0.7
    options:
      lowercase: true
```

### Survivorship Rules

```yaml
survivorship:
  rules:
    - column: last_updated
      strategy: most_recent
```

### Priority Conditions

```yaml
priority_rule:
  conditions:
    - column: is_verified
      value: true
      priority: 1
```

---

## 🔧 Advanced Usage

* **Custom Similarity Functions** – define your own matcher class
* **Batch Processing** – handle large datasets efficiently with multiprocessing
* **Performance Benchmarking** – test similarity methods and blocking strategies

---

## 📊 Output Files

1. `golden_records_TIMESTAMP.csv` – Deduplicated golden records
2. `review_pairs_TIMESTAMP.csv` – Pairs for manual review
3. `processing_summary_TIMESTAMP.txt` – Human-readable summary
4. `detailed_stats_TIMESTAMP.json` – Machine-readable statistics

---

## 🚀 Optimization

* **Memory**: Chunked processing, vectorized operations, caching
* **CPU**: Parallel processing, Numba JIT, batch operations
* **I/O**: Connection pooling, bulk read/write, compression

---

## 🔍 Troubleshooting

* Database connection errors → test connection and check drivers
* Memory issues → reduce batch size, enable chunking
* Slow similarity → benchmark methods, optimize blocking

---

## 📈 Comparison to Original EasyMDM

| Feature      | Original            | Advanced                                    |
| ------------ | ------------------- | ------------------------------------------- |
| Databases    | CSV, SQLite, DuckDB | + PostgreSQL, SQL Server                    |
| Similarity   | Basic               | + Cosine, Jaccard, Fuzzy variants           |
| Blocking     | Fuzzy only          | + Exact, Sorted Neighborhood, RecordLinkage |
| Clustering   | Network only        | + Hierarchical, DBSCAN                      |
| Survivorship | Basic               | + 8 advanced strategies                     |
| Performance  | Single-thread       | Multi-core, Numba JIT, vectorized           |
| CLI          | Basic               | Rich UI, comprehensive commands             |
| Output       | CSV only            | + Review pairs, stats, multiple formats     |
| Memory       | Load all            | + Chunking, streaming, optimization         |

---

## 🤝 Contributing

Fork, create a feature branch, commit, push, and open a Pull Request. Use development mode (`pip install -e ".[dev]"`) and run tests with `pytest`.

---

## 📄 License

MIT License – see LICENSE file for details.

## 🆘 Support

* Documentation: [ReadTheDocs](https://easymdm-advanced.readthedocs.io)
* GitHub Issues & Discussions
* Email: [support@easymdm-advanced.com](mailto:support@easymdm-advanced.com)

---

**EasyMDM Advanced** – Making Master Data Management simple and powerful 🚀

---
