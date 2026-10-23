"""
================================================================================
MODULE: Telco Schema Enforcer & Quarantine Dead-Letter Queue (DLQ) Router
FILE  : src/hygiene/O07_schema_enforcer.py
PURPOSE:
  Loads the Telco YAML schema contract and validates incoming DataFrames against 
  strict column presence, data types, nullability rules, categorical domains, 
  and numerical ranges. Isolates corrupted or anomalous records into a 
  Dead-Letter Queue (data/quarantine/) with audit reasons, preventing pipeline failures.
================================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoSchemaEnforcer")


class SchemaEnforcer:
    """Enforces data contracts and routes invalid payloads to quarantine DLQ."""

    def __init__(self, contract_path: Path, quarantine_dir: Path):
        self.contract_path = Path(contract_path)
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.contract = self._load_contract()

    def _load_contract(self) -> Dict[str, Any]:
        """Loads and parses the YAML schema contract specification."""
        if not self.contract_path.exists():
            raise FileNotFoundError(f"[ERROR] Schema contract file not found: {self.contract_path}")

        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)

        logger.info(
            f"Loaded schema contract: '{contract_data['dataset_info']['name']}' "
            f"(v{contract_data['dataset_info']['version']})"
        )
        return contract_data

    def validate_and_quarantine(self, df: pd.DataFrame, batch_id: str = "telco_batch_001") -> pd.DataFrame:
        """
        Validates the incoming Telco DataFrame against expected columns, types,
        nullability, and value boundaries. Splits into valid and quarantined records.

        Args:
            df (pd.DataFrame): Incoming raw Telco DataFrame.
            batch_id (str): Identifier for tracing and naming quarantine files.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing only verified valid records.
        """
        df_work = df.copy()
        expected_cols = self.contract["expected_columns"]

        # 1. Check for Missing Columns (Schema Drift)
        missing_columns = [col for col in expected_cols.keys() if col not in df_work.columns]
        if missing_columns:
            logger.critical(f"[SCHEMA DRIFT] Missing expected columns in batch '{batch_id}': {missing_columns}")
            drift_file = self.quarantine_dir / f"quarantined_{batch_id}_schema_drift.csv"
            df_work.to_csv(drift_file, index=False)
            raise ValueError(f"Schema drift detected! Missing columns: {missing_columns}. Quarantined to {drift_file}")

        # Normalize TotalCharges: In raw data, new customers (tenure=0) may have whitespace ' '
        # Coerce space string to NaN or 0.0 before numeric boundary validation
        if "TotalCharges" in df_work.columns:
            whitespace_mask = df_work["TotalCharges"].astype(str).str.strip() == ""
            if whitespace_mask.sum() > 0:
                logger.info(f"Detected {whitespace_mask.sum()} whitespace values in 'TotalCharges'. Coercing to 0.0.")
                df_work.loc[whitespace_mask, "TotalCharges"] = 0.0
            df_work["TotalCharges"] = pd.to_numeric(df_work["TotalCharges"], errors="coerce")

        total_records = len(df_work)
        logger.info(f"Validating {total_records:,} Telco records against {len(expected_cols)} contract specifications...")

        invalid_mask = pd.Series(False, index=df_work.index)
        violation_reasons = {idx: [] for idx in df_work.index}

        for col, rules in expected_cols.items():
            series = df_work[col]

            # A. Nullability Check
            if not rules.get("nullable", True):
                null_mask = series.isnull()
                if null_mask.any():
                    for idx in df_work[null_mask].index:
                        violation_reasons[idx].append(f"Column '{col}' cannot be null")
                    invalid_mask = invalid_mask | null_mask

            # B. Allowed Categorical Values
            if "allowed_values" in rules:
                allowed_set = set(rules["allowed_values"])
                non_null_mask = ~series.isnull()
                disallowed_mask = non_null_mask & (~series.isin(allowed_set))
                if disallowed_mask.any():
                    for idx in df_work[disallowed_mask].index:
                        violation_reasons[idx].append(
                            f"Column '{col}' value '{series[idx]}' not in allowed set {rules['allowed_values']}"
                        )
                    invalid_mask = invalid_mask | disallowed_mask

            # C. Numerical Range Checks (Min / Max)
            if "min_value" in rules:
                min_val = rules["min_value"]
                non_null_mask = ~series.isnull()
                underflow_mask = non_null_mask & (series < min_val)
                if underflow_mask.any():
                    for idx in df_work[underflow_mask].index:
                        violation_reasons[idx].append(f"Column '{col}' ({series[idx]}) below min {min_val}")
                    invalid_mask = invalid_mask | underflow_mask

            if "max_value" in rules:
                max_val = rules["max_value"]
                non_null_mask = ~series.isnull()
                overflow_mask = non_null_mask & (series > max_val)
                if overflow_mask.any():
                    for idx in df_work[overflow_mask].index:
                        violation_reasons[idx].append(f"Column '{col}' ({series[idx]}) above max {max_val}")
                    invalid_mask = invalid_mask | overflow_mask

        num_invalid = invalid_mask.sum()
        num_valid = total_records - num_invalid

        # Route invalid records to Quarantine DLQ
        if num_invalid > 0:
            quarantine_df = df.loc[invalid_mask].copy()
            quarantine_df["quarantine_reason"] = [
                " | ".join(violation_reasons[idx]) for idx in quarantine_df.index
            ]
            quarantine_df["quarantined_at"] = datetime.utcnow().isoformat()
            
            quarantine_file = self.quarantine_dir / f"quarantined_{batch_id}.csv"
            quarantine_df.to_csv(quarantine_file, index=False)
            logger.warning(
                f"[QUARANTINE DLQ] Isolated {num_invalid:,} corrupt/invalid records into: {quarantine_file}"
            )
            
            # Write audit metadata JSON
            audit_file = self.quarantine_dir / f"quarantined_{batch_id}_audit.json"
            audit_meta = {
                "batch_id": batch_id,
                "timestamp": datetime.utcnow().isoformat(),
                "total_records": total_records,
                "valid_records": int(num_valid),
                "quarantined_records": int(num_invalid),
                "violations_sample": list(quarantine_df["quarantine_reason"].head(10))
            }
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump(audit_meta, f, indent=2)
        else:
            logger.info("Zero contract violations detected! All records meet contract integrity.")

        clean_df = df_work.loc[~invalid_mask].reset_index(drop=True)
        logger.info(f"Validation complete. Passed: {len(clean_df):,} | Quarantined: {num_invalid:,}")
        return clean_df
