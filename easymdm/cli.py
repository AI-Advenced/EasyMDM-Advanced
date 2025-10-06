"""
Enhanced Command Line Interface for EasyMDM Advanced.
Provides comprehensive CLI commands for MDM operations.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .core.config import MDMConfig, create_sample_config
from .core.engine import MDMEngine
from .database.connector import DatabaseConnector

# Setup logging
logger = logging.getLogger(__name__)


class EasyMDMCLI:
    """Enhanced CLI for EasyMDM operations."""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for CLI."""
        if RICH_AVAILABLE:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                datefmt="[%X]",
                handlers=[RichHandler(console=self.console, show_time=False)]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def print_message(self, message: str, style: str = ""):
        """Print message with optional Rich styling."""
        if self.console:
            self.console.print(message, style=style)
        else:
            print(message)
    
    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"❌ Error: {message}", style="bold red")
        else:
            print(f"Error: {message}")
    
    def print_success(self, message: str):
        """Print success message."""
        if self.console:
            self.console.print(f"✅ {message}", style="bold green")
        else:
            print(f"Success: {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"⚠️ Warning: {message}", style="bold yellow")
        else:
            print(f"Warning: {message}")


def cmd_process(args):
    """Execute MDM processing."""
    cli = EasyMDMCLI()
    
    try:
        cli.print_message("🚀 Starting EasyMDM Advanced Processing", "bold blue")
        
        # Load configuration
        if not os.path.exists(args.config):
            cli.print_error(f"Configuration file not found: {args.config}")
            return 1
        
        config = MDMConfig.from_yaml(args.config)
        
        # Override output path if specified
        if args.output:
            config.output_path = args.output
        
        # Create MDM engine
        engine = MDMEngine(config)
        
        # Test configuration if requested
        if args.test_config:
            cli.print_message("🔍 Testing configuration...", "yellow")
            test_results = engine.test_configuration()
            
            if RICH_AVAILABLE:
                table = Table(title="Configuration Test Results")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="magenta")
                
                for component, passed in test_results.items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    table.add_row(component.replace('_', ' ').title(), status)
                
                cli.console.print(table)
            else:
                for component, passed in test_results.items():
                    status = "PASS" if passed else "FAIL"
                    print(f"{component}: {status}")
            
            if not all(test_results.values()):
                cli.print_error("Configuration test failed")
                return 1
            
            cli.print_success("Configuration test passed")
        
        # Profile data if requested
        if args.profile:
            cli.print_message("📊 Profiling input data...", "yellow")
            profile = engine.get_data_profile()
            
            if profile:
                cli.print_message(f"📈 Data Profile:")
                cli.print_message(f"   Records: {profile.get('total_records', 0):,}")
                cli.print_message(f"   Columns: {profile.get('total_columns', 0)}")
                cli.print_message(f"   Memory Usage: {profile.get('memory_usage', 0):,} bytes")
                
                if args.verbose:
                    # Show detailed column statistics
                    if RICH_AVAILABLE and 'column_statistics' in profile:
                        table = Table(title="Column Statistics")
                        table.add_column("Column", style="cyan")
                        table.add_column("Type", style="green")
                        table.add_column("Unique", style="yellow")
                        table.add_column("Null %", style="red")
                        
                        for col, stats in profile['column_statistics'].items():
                            null_pct = f"{stats.get('null_percentage', 0):.1f}%"
                            table.add_row(
                                col[:30],  # Truncate long column names
                                str(profile.get('data_types', {}).get(col, 'unknown')),
                                str(stats.get('unique_values', 0)),
                                null_pct
                            )
                        
                        cli.console.print(table)
        
        # Execute processing
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
            ) as progress:
                task = progress.add_task("Processing MDM...", total=None)
                result = engine.process()
                progress.update(task, completed=True)
        else:
            cli.print_message("Processing MDM...")
            result = engine.process()
        
        # Display results
        cli.print_success(f"Processing completed successfully!")
        cli.print_message(f"📊 Results:")
        cli.print_message(f"   Golden Records: {len(result.golden_records):,}")
        cli.print_message(f"   Execution Time: {result.execution_time:.2f} seconds")
        cli.print_message(f"   Output Files: {len(result.output_files)}")
        
        if args.verbose:
            cli.print_message(f"\n📁 Output Files:")
            for file_path in result.output_files:
                cli.print_message(f"   • {file_path}")
        
        return 0
        
    except Exception as e:
        cli.print_error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_create_config(args):
    """Create a sample configuration file."""
    cli = EasyMDMCLI()
    
    try:
        if os.path.exists(args.output) and not args.force:
            cli.print_error(f"Configuration file already exists: {args.output}")
            cli.print_message("Use --force to overwrite", "yellow")
            return 1
        
        # Create sample configuration
        config = create_sample_config()
        
        # Customize based on arguments
        if args.database_type:
            config.source.type = args.database_type
            
            if args.database_type == 'postgresql':
                config.source.host = args.host or 'localhost'
                config.source.port = args.port or 5432
                config.source.database = args.database or 'mydb'
                config.source.username = args.username
                config.source.password = args.password
                config.source.schema = args.schema or 'public'
                config.source.table = args.table or 'records'
                
            elif args.database_type == 'sqlserver':
                config.source.host = args.host or 'localhost'
                config.source.port = args.port or 1433
                config.source.database = args.database or 'mydb'
                config.source.username = args.username
                config.source.password = args.password
                config.source.schema = args.schema or 'dbo'
                config.source.table = args.table or 'records'
                
            elif args.database_type == 'sqlite':
                config.source.file_path = args.file_path or 'database.db'
                config.source.table = args.table or 'records'
                
            elif args.database_type == 'csv':
                config.source.file_path = args.file_path or 'data.csv'
        
        # Save configuration
        config.save_yaml(args.output)
        
        cli.print_success(f"Sample configuration created: {args.output}")
        cli.print_message("Edit the configuration file to match your data source and requirements", "yellow")
        
        return 0
        
    except Exception as e:
        cli.print_error(str(e))
        return 1


def cmd_test_connection(args):
    """Test database connection."""
    cli = EasyMDMCLI()
    
    try:
        # Load configuration
        config = MDMConfig.from_yaml(args.config)
        
        cli.print_message("🔍 Testing database connection...", "yellow")
        
        # Test connection
        connector = DatabaseConnector.create_connector(config.source)
        success = connector.test_connection()
        
        if success:
            cli.print_success("Database connection successful!")
            
            # Try to load a sample of data
            if args.sample_size > 0:
                try:
                    connector.connect()
                    df = connector.load_data()
                    
                    if not df.empty:
                        sample_size = min(args.sample_size, len(df))
                        sample_df = df.head(sample_size)
                        
                        cli.print_message(f"\n📊 Sample Data ({sample_size} records):")
                        cli.print_message(f"   Total Records: {len(df):,}")
                        cli.print_message(f"   Columns: {list(df.columns)}")
                        
                        if args.verbose and RICH_AVAILABLE:
                            # Display sample data in table
                            table = Table(title=f"Sample Data (First {sample_size} rows)")
                            
                            # Add columns (limit to avoid overflow)
                            display_columns = list(sample_df.columns)[:10]
                            for col in display_columns:
                                table.add_column(col[:20], overflow="fold")  # Truncate long column names
                            
                            # Add rows
                            for _, row in sample_df.iterrows():
                                row_values = [str(row[col])[:50] for col in display_columns]  # Truncate values
                                table.add_row(*row_values)
                            
                            cli.console.print(table)
                    else:
                        cli.print_warning("No data found in the source")
                        
                except Exception as e:
                    cli.print_warning(f"Could not load sample data: {e}")
                finally:
                    connector.disconnect()
        else:
            cli.print_error("Database connection failed!")
            return 1
        
        return 0
        
    except Exception as e:
        cli.print_error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_validate_config(args):
    """Validate configuration file."""
    cli = EasyMDMCLI()
    
    try:
        cli.print_message("🔍 Validating configuration...", "yellow")
        
        # Try to load configuration
        config = MDMConfig.from_yaml(args.config)
        
        # Perform validation checks
        validation_results = []
        
        # Check source configuration
        if config.source.type in ['postgresql', 'sqlserver']:
            if not config.source.host:
                validation_results.append(("Source", "Missing host", False))
            if not config.source.database:
                validation_results.append(("Source", "Missing database", False))
            if not config.source.username:
                validation_results.append(("Source", "Missing username", False))
            else:
                validation_results.append(("Source", "Database configuration", True))
        
        elif config.source.type in ['sqlite', 'csv']:
            if not config.source.file_path:
                validation_results.append(("Source", "Missing file path", False))
            else:
                validation_results.append(("Source", "File configuration", True))
        
        # Check blocking configuration
        if not config.blocking.columns:
            validation_results.append(("Blocking", "No blocking columns specified", False))
        else:
            validation_results.append(("Blocking", f"{len(config.blocking.columns)} columns", True))
        
        # Check similarity configuration
        if not config.similarity:
            validation_results.append(("Similarity", "No similarity configurations", False))
        else:
            validation_results.append(("Similarity", f"{len(config.similarity)} configurations", True))
        
        # Check thresholds
        if config.thresholds.review >= config.thresholds.auto_merge:
            validation_results.append(("Thresholds", "Review >= Auto-merge", False))
        else:
            validation_results.append(("Thresholds", "Valid threshold ordering", True))
        
        # Display results
        if RICH_AVAILABLE:
            table = Table(title="Configuration Validation Results")
            table.add_column("Component", style="cyan")
            table.add_column("Check", style="white")
            table.add_column("Status", style="magenta")
            
            for component, check, passed in validation_results:
                status = "✅ PASS" if passed else "❌ FAIL"
                table.add_row(component, check, status)
            
            cli.console.print(table)
        else:
            for component, check, passed in validation_results:
                status = "PASS" if passed else "FAIL"
                print(f"{component} - {check}: {status}")
        
        # Overall result
        all_passed = all(result[2] for result in validation_results)
        
        if all_passed:
            cli.print_success("Configuration validation passed!")
        else:
            cli.print_error("Configuration validation failed!")
            return 1
        
        return 0
        
    except Exception as e:
        cli.print_error(f"Configuration validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_benchmark(args):
    """Benchmark similarity methods or blocking strategies."""
    cli = EasyMDMCLI()
    
    try:
        # Load configuration
        config = MDMConfig.from_yaml(args.config)
        
        # Load data
        connector = DatabaseConnector.create_connector(config.source)
        connector.connect()
        df = connector.load_data()
        connector.disconnect()
        
        if df.empty:
            cli.print_error("No data to benchmark")
            return 1
        
        # Limit data size for benchmarking
        sample_size = min(args.sample_size, len(df))
        df_sample = df.head(sample_size)
        
        cli.print_message(f"🏃 Benchmarking on {len(df_sample)} records...", "yellow")
        
        if args.type == 'similarity':
            from .similarity.matcher import SimilarityMatcher
            
            # Benchmark similarity methods
            matcher = SimilarityMatcher(config.similarity)
            results = matcher.benchmark_methods(df_sample, sample_size=1000)
            
            if RICH_AVAILABLE:
                table = Table(title="Similarity Method Benchmark")
                table.add_column("Method", style="cyan")
                table.add_column("Avg Time (ms)", style="yellow")
                table.add_column("Performance", style="green")
                
                for method, avg_time in sorted(results.items(), key=lambda x: x[1]):
                    time_ms = avg_time * 1000
                    if time_ms < 1:
                        perf = "Excellent"
                    elif time_ms < 10:
                        perf = "Good"
                    elif time_ms < 100:
                        perf = "Fair"
                    else:
                        perf = "Slow"
                    
                    table.add_row(method, f"{time_ms:.3f}", perf)
                
                cli.console.print(table)
            else:
                print("Similarity Method Benchmark Results:")
                for method, avg_time in sorted(results.items(), key=lambda x: x[1]):
                    print(f"  {method}: {avg_time*1000:.3f} ms")
        
        elif args.type == 'blocking':
            from .core.blocking import BlockingProcessor
            
            # Benchmark blocking methods
            processor = BlockingProcessor(config.blocking)
            results = processor.compare_blocking_methods(df_sample)
            
            if RICH_AVAILABLE:
                table = Table(title="Blocking Method Benchmark")
                table.add_column("Method", style="cyan")
                table.add_column("Pairs", style="yellow")
                table.add_column("Time (s)", style="green")
                table.add_column("Reduction %", style="blue")
                
                for method, stats in results.items():
                    if 'error' not in stats:
                        reduction_pct = f"{stats['reduction_ratio']*100:.1f}%"
                        table.add_row(
                            method,
                            f"{stats['candidate_pairs']:,}",
                            f"{stats['processing_time']:.2f}",
                            reduction_pct
                        )
                    else:
                        table.add_row(method, "ERROR", "-", "-")
                
                cli.console.print(table)
            else:
                print("Blocking Method Benchmark Results:")
                for method, stats in results.items():
                    if 'error' not in stats:
                        print(f"  {method}: {stats['candidate_pairs']:,} pairs in {stats['processing_time']:.2f}s")
                    else:
                        print(f"  {method}: ERROR")
        
        return 0
        
    except Exception as e:
        cli.print_error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_info(args):
    """Display package information."""
    cli = EasyMDMCLI()
    
    from . import get_info, get_version
    
    info = get_info()
    
    if RICH_AVAILABLE:
        table = Table(title=f"EasyMDM Advanced v{get_version()}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in info.items():
            if isinstance(value, list):
                value = ", ".join(value)
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        cli.console.print(table)
    else:
        print(f"EasyMDM Advanced v{get_version()}")
        for key, value in info.items():
            if isinstance(value, list):
                value = ", ".join(value)
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check available connectors
    available_connectors = DatabaseConnector.get_available_connectors()
    
    cli.print_message(f"\n📊 Available Database Connectors:")
    for connector in available_connectors:
        cli.print_message(f"   • {connector}", "green")
    
    return 0


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="EasyMDM Advanced - Enhanced Master Data Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MDM with configuration file
  easymdm process --config config.yaml --output results/

  # Create sample configuration
  easymdm create-config --output config.yaml --database-type postgresql

  # Test database connection
  easymdm test-connection --config config.yaml --sample-size 100

  # Validate configuration
  easymdm validate-config --config config.yaml

  # Benchmark similarity methods
  easymdm benchmark --config config.yaml --type similarity --sample-size 1000

  # Show package information
  easymdm info
        """
    )
    
    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Execute MDM processing')
    process_parser.add_argument('--config', required=True, help='Configuration file path')
    process_parser.add_argument('--output', help='Output directory (overrides config)')
    process_parser.add_argument('--test-config', action='store_true', help='Test configuration before processing')
    process_parser.add_argument('--profile', action='store_true', help='Profile input data')
    process_parser.set_defaults(func=cmd_process)
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create sample configuration file')
    config_parser.add_argument('--output', default='config.yaml', help='Output configuration file')
    config_parser.add_argument('--force', action='store_true', help='Overwrite existing file')
    config_parser.add_argument('--database-type', choices=['csv', 'sqlite', 'postgresql', 'sqlserver'], 
                              help='Database type')
    config_parser.add_argument('--host', help='Database host')
    config_parser.add_argument('--port', type=int, help='Database port')
    config_parser.add_argument('--database', help='Database name')
    config_parser.add_argument('--username', help='Database username')
    config_parser.add_argument('--password', help='Database password')
    config_parser.add_argument('--schema', help='Database schema')
    config_parser.add_argument('--table', help='Table name')
    config_parser.add_argument('--file-path', help='File path (for CSV/SQLite)')
    config_parser.set_defaults(func=cmd_create_config)
    
    # Test connection command
    test_parser = subparsers.add_parser('test-connection', help='Test database connection')
    test_parser.add_argument('--config', required=True, help='Configuration file path')
    test_parser.add_argument('--sample-size', type=int, default=10, help='Number of sample records to display')
    test_parser.set_defaults(func=cmd_test_connection)
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate configuration file')
    validate_parser.add_argument('--config', required=True, help='Configuration file path')
    validate_parser.set_defaults(func=cmd_validate_config)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark methods')
    benchmark_parser.add_argument('--config', required=True, help='Configuration file path')
    benchmark_parser.add_argument('--type', choices=['similarity', 'blocking'], 
                                 default='similarity', help='Benchmark type')
    benchmark_parser.add_argument('--sample-size', type=int, default=1000, 
                                 help='Sample size for benchmarking')
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    info_parser.set_defaults(func=cmd_info)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console = Console()
            console.print("\n❌ Operation cancelled by user", style="bold red")
        else:
            print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"\n❌ Unexpected error: {e}", style="bold red")
        else:
            print(f"\nUnexpected error: {e}")
        
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == '__main__':
    sys.exit(main())