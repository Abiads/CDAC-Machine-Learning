"""
================================================================================
MODULE: Stage 2 Telco Master Pipeline Orchestrator
FILE  : src/O09_orchestrator.py
PURPOSE:
  Serves as the chronological execution runner for Stage 2 on the IBM Telco 
  Customer Churn dataset. Integrates all modular hygiene and ingestion components:
  1. Checksum & zero-byte integrity validation
  2. Raw landing zone batch extraction (Telco-Customer-Churn.csv)
  3. Schema contract enforcement with automated quarantine DLQ routing
  4. Memory-optimized numerical downcasting
  5. Final data readiness inspection for Stage 3 downstream modeling
================================================================================
"""

import logging
import sys
from pathlib import Path

# Add current directory to sys.path for direct script execution
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from hygiene.O02_checksum_validator import validate_file_integrity
from extractors.O03_file_batch_extractor import BatchFileExtractor
from hygiene.O07_schema_enforcer import SchemaEnforcer
from optimization.O08_memory_optimizer import MemoryOptimizer

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoStage2Orchestrator")


def run_stage2_telco_pipeline():
    """Executes the complete end-to-end Data Ingestion and Hygiene pipeline for Telco Churn."""
    logger.info("====================================================================")
    logger.info(" STARTING STAGE 2: TELCO DATA INGESTION & HYGIENE PIPELINE")
    logger.info("====================================================================")

    project_root = current_dir.parent
    raw_dir = project_root / "data" / "raw"
    quarantine_dir = project_root / "data" / "quarantine"
    contract_path = project_root / "config" / "O01_telco_schema_contract.yaml"

    raw_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    target_file = raw_dir / "Telco-Customer-Churn.csv"
    if not target_file.exists():
        # Fallback to repository root dataset if missing
        repo_root_file = project_root.parent.parent / "Telco-Customer-Churn.csv"
        if repo_root_file.exists():
            import shutil
            shutil.copy(repo_root_file, target_file)
            logger.info(f"Copied Telco dataset from {repo_root_file} to {target_file}")
        else:
            raise FileNotFoundError(f"Target Telco dataset not found at {target_file}")

    # STEP A: Checksum & File Integrity Validation
    logger.info("\n--- STEP A: File Integrity & Zero-Byte Verification ---")
    is_valid_file = validate_file_integrity(target_file)
    if not is_valid_file:
        logger.critical("Pipeline halted: File failed integrity verification.")
        return None

    # STEP B: Landing Zone Batch Extraction
    logger.info("\n--- STEP B: Landing Zone Batch Extraction ---")
    extractor = BatchFileExtractor(raw_data_dir=raw_dir)
    raw_df = extractor.extract_file("Telco-Customer-Churn.csv", file_format="csv")
    logger.info(f"Successfully extracted raw DataFrame: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")

    # STEP C: Schema Contract Enforcement & Quarantine Routing
    logger.info("\n--- STEP C: Schema Enforcement & Quarantine DLQ Routing ---")
    enforcer = SchemaEnforcer(contract_path=contract_path, quarantine_dir=quarantine_dir)
    clean_df = enforcer.validate_and_quarantine(raw_df, batch_id="telco_production_batch_001")

    # STEP D: Low-Level Memory Optimization (Type Downcasting)
    logger.info("\n--- STEP D: Memory Optimization & Type Downcasting ---")
    optimized_df = MemoryOptimizer.downcast_numeric_types(clean_df)

    # STEP E: Pipeline Summary & Ingestion Health Sign-off
    logger.info("====================================================================")
    logger.info(" STAGE 2 TELCO PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f" Clean Records Ready for Stage 3 EDA & Modeling: {len(optimized_df):,}")
    logger.info("====================================================================")

    print("\n" + "=" * 60)
    print(" [STAGE 2 PIPELINE SUMMARY - TELCO CUSTOMER CHURN]")
    print("=" * 60)
    print(f"Total Raw Rows Ingested     : {len(raw_df):,}")
    print(f"Total Valid Rows Verified   : {len(optimized_df):,}")
    print(f"Total Columns Enforced      : {optimized_df.shape[1]}")
    print(f"Overall Churn Rate          : {(optimized_df['Churn'] == 'Yes').mean() * 100:.2f}%")
    print(f"Memory Footprint            : {MemoryOptimizer.get_memory_usage_mb(optimized_df):.3f} MB")
    print("=" * 60)
    print("\nSample Ingested Rows:")
    print(optimized_df[["customerID", "gender", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"]].head(5))
    
    return optimized_df


if __name__ == "__main__":
    run_stage2_telco_pipeline()
