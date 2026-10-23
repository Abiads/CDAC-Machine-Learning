"""
================================================================================
MODULE: Stage 2 Telco Master Multi-Channel Pipeline Orchestrator
FILE  : src/O09_orchestrator_all.py
PURPOSE:
  Serves as the comprehensive execution runner for Stage 2 integrating all 
  multi-source extraction channels for the IBM Telco Churn ecosystem:
  1. Flat Files (CSV, Parquet) from Landing Zone
  2. Relational Database Paginated Chunks (SQLite / PostgreSQL)
  3. REST API CRM Telemetry Scraper with Exponential Backoff
  4. Real-Time Event Streaming via Apache Kafka Consumer
  5. Schema Contract Enforcement, Quarantine DLQ & RAM Optimization
================================================================================
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add current directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from extractors.O04_db_extractor import DatabaseExtractor
from extractors.O05_api_scraper import APIScraper
from extractors.O06_kafka_consumer import KafkaTelemetryConsumer
from hygiene.O07_schema_enforcer import SchemaEnforcer
from optimization.O08_memory_optimizer import MemoryOptimizer

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoMultiSourceOrchestrator")


# ==============================================================================
# MULTI-SOURCE EXTRACTION CHANNELS
# ==============================================================================

def get_file_batch_data(raw_dir: Path, filename: str = "Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Extraction Channel 1: Flat file batch extraction from landing zone."""
    extractor = BatchFileExtractor(raw_data_dir=raw_dir)
    return extractor.extract_file(filename, file_format="csv")


def get_db_data() -> pd.DataFrame:
    """Extraction Channel 2: Relational database connection pooling & chunking."""
    db_extractor = DatabaseExtractor(connection_string="sqlite:///telco_enterprise.db")
    batch_generator = db_extractor.extract_in_batches(query="SELECT * FROM telco_customers", chunk_size=1000)
    return next(batch_generator)


def get_api_data() -> pd.DataFrame:
    """Extraction Channel 3: REST API customer CRM scraper."""
    scraper = APIScraper(base_url="https://api.telecom-crm.example.com")
    return scraper.fetch_telco_batch(endpoint="customers/recent", limit=10)


def get_kafka_data() -> pd.DataFrame:
    """Extraction Channel 4: Real-time event streaming via Kafka consumer."""
    consumer = KafkaTelemetryConsumer(
        bootstrap_servers="localhost:9092",
        topic="telco-customer-events",
        group_id="telco-ml-group"
    )
    return consumer.poll_messages(max_messages=5)


# ==============================================================================
# MASTER MULTI-CHANNEL PIPELINE RUNNER
# ==============================================================================

def run_stage2_multi_source_pipeline():
    """Executes multi-source extractions, schema validation, and memory optimization."""
    logger.info("====================================================================")
    logger.info(" STARTING STAGE 2: TELCO MULTI-SOURCE INGESTION & HYGIENE PIPELINE")
    logger.info("====================================================================")

    project_root = current_dir.parent
    raw_dir = project_root / "data" / "raw"
    quarantine_dir = project_root / "data" / "quarantine"
    contract_path = project_root / "config" / "O01_telco_schema_contract.yaml"

    target_file = raw_dir / "Telco-Customer-Churn.csv"

    # Step 1: Integrity Check on Main Landing File
    logger.info("\n--- STEP 1: Verifying Landing Zone File Integrity ---")
    if not validate_file_integrity(target_file):
        logger.critical("Primary landing zone file failed verification.")
        return

    # Step 2: Multi-Source Extractions
    logger.info("\n--- STEP 2: Ingesting Across Multi-Source Channels ---")
    df_file = get_file_batch_data(raw_dir)
    logger.info(f"  [Channel 1 - Batch File] Ingested: {len(df_file):,} records")

    df_db = get_db_data()
    logger.info(f"  [Channel 2 - Database  ] Ingested: {len(df_db):,} records")

    df_api = get_api_data()
    logger.info(f"  [Channel 3 - REST API  ] Ingested: {len(df_api):,} records")

    df_kafka = get_kafka_data()
    logger.info(f"  [Channel 4 - Kafka     ] Ingested: {len(df_kafka):,} records")

    # Merge / Union Ingested Sources
    combined_df = pd.concat([df_file, df_db, df_api, df_kafka], ignore_index=True)
    logger.info(f"\nTotal Multi-Source Combined Records: {len(combined_df):,}")

    # Step 3: Schema Contract Enforcement & DLQ Quarantine Routing
    logger.info("\n--- STEP 3: Enforcing Schema Contract Across Combined Stream ---")
    enforcer = SchemaEnforcer(contract_path=contract_path, quarantine_dir=quarantine_dir)
    clean_df = enforcer.validate_and_quarantine(combined_df, batch_id="telco_multisource_batch_001")

    # Step 4: Low-Level RAM Optimization
    logger.info("\n--- STEP 4: Aggressive Memory Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)

    logger.info("====================================================================")
    logger.info(" MULTI-SOURCE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f" Ingested Clean Records Available for Stage 3: {len(optimized_df):,}")
    logger.info("====================================================================")

    print("\n[MULTI-SOURCE PIPELINE SUMMARY]")
    print(f"Total Sources Ingested     : 4 (File, DB, API, Kafka)")
    print(f"Total Records Processed    : {len(combined_df):,}")
    print(f"Passed Hygiene & Contracts : {len(optimized_df):,}")
    print(f"RAM Footprint              : {MemoryOptimizer.get_memory_usage_mb(optimized_df):.3f} MB")
    return optimized_df


if __name__ == "__main__":
    run_stage2_multi_source_pipeline()
