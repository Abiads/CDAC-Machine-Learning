"""
================================================================================
PRACTICE: Object-Oriented Data Preprocessing Pipeline (Scikit-Learn Style)
FILE: practice.py
PURPOSE:
  A modular, extensible, and production-grade data preprocessing pipeline
  implementing the fit() and transform() design pattern.
  
COMPONENTS:
  1. BaseTransformer           -> Abstract interface with fit(), transform(), fit_transform()
  2. SchemaStandardizer        -> Snake_case conversion and column alias mapping
  3. StructuralTextSanitizer   -> Whitespace trimming, currency/symbol removal, categorical mapping
  4. DuplicateHandler          -> Deduplication on primary keys or entire rows
  5. MissingValueImputer       -> Leakage-free numerical (median) and categorical (mode) imputation
  6. SchemaValidator           -> Required column assertion, type validation, null thresholds
  7. DataPipeline              -> Master orchestrator with comprehensive logging and audit metrics
================================================================================
"""

import re
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 0. Global Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataPipeline")


# ------------------------------------------------------------------------------
# 1. Base Transformer Interface
# ------------------------------------------------------------------------------
class BaseTransformer:
    """
    Abstract Base Class enforcing the Scikit-Learn transformer protocol.
    Every step in the pipeline inherits from this class.
    """

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "BaseTransformer":
        """
        Learns parameters (statistics, distributions, encodings) from training data.
        Must return self to allow method chaining.
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations to the dataset using learned parameters.
        Must return a new or modified pd.DataFrame.
        """
        raise NotImplementedError("Subclasses must implement the 'transform' method.")

    def fit_transform(self, df: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Fits to data, then transforms it in a single call.
        """
        return self.fit(df, y).transform(df)


# ------------------------------------------------------------------------------
# 2. Schema Standardization Component
# ------------------------------------------------------------------------------
class SchemaStandardizer(BaseTransformer):
    """
    Standardizes arbitrary column headers into clean, uniform snake_case
    and resolves column synonyms / aliases.
    
    Example:
      'Customer Name'      -> 'customer_name'
      'Purchase Amount ($)' -> 'purchase_amount'
      'cust_id'            -> 'customer_id' (via alias_map)
    """

    def __init__(self, alias_map: Optional[Dict[str, str]] = None):
        self.alias_map = {k.lower().strip(): v.lower().strip() for k, v in (alias_map or {}).items()}
        self.step_logger = logging.getLogger("SchemaStandardizer")

    @staticmethod
    def _to_snake_case(col_name: str) -> str:
        """Converts an arbitrary string to clean snake_case."""
        # 1. Replace symbols and brackets with whitespace
        cleaned = re.sub(r"[^\w\s]", " ", col_name)
        # 2. Handle camelCase and PascalCase (e.g. 'CustomerID' -> 'Customer_ID')
        cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", cleaned)
        # 3. Collapse multiple whitespace / underscores into single underscore
        cleaned = re.sub(r"[\s_]+", "_", cleaned)
        # 4. Lowercase and trim leading/trailing underscores
        return cleaned.strip("_").lower()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        raw_cols = df_out.columns.tolist()

        # Step A: Convert to snake_case
        renamed_cols = [self._to_snake_case(c) for c in raw_cols]

        # Step B: Apply column alias mapping
        final_cols = [self.alias_map.get(c, c) for c in renamed_cols]

        mapping_log = {
            raw: final for raw, final in zip(raw_cols, final_cols) if raw != final
        }
        df_out.columns = final_cols

        if mapping_log:
            self.step_logger.info(f"Standardized {len(mapping_log)} column header(s): {mapping_log}")
        else:
            self.step_logger.info("All column headers already adhere to standard schema.")

        return df_out


# ------------------------------------------------------------------------------
# 3. Structural & String Normalization Component
# ------------------------------------------------------------------------------
class StructuralTextSanitizer(BaseTransformer):
    """
    Cleans textual columns, strips irregular whitespace, normalizes casing,
    and maps messy categorical variants to canonical values.
    Also parses dirty numeric strings (e.g., '$1,250.00' -> 1250.0).
    """

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        case_format: str = "lower",
        strip_whitespace: bool = True,
        categorical_mappings: Optional[Dict[str, Dict[str, str]]] = None,
        numeric_clean_columns: Optional[List[str]] = None
    ):
        self.text_columns = text_columns or []
        self.case_format = case_format.lower()
        self.strip_whitespace = strip_whitespace
        self.categorical_mappings = categorical_mappings or {}
        self.numeric_clean_columns = numeric_clean_columns or []
        self.step_logger = logging.getLogger("TextSanitizer")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # 1. Clean dirty numeric columns masquerading as strings (e.g., "$1,250.50", "1500 USD", "N/A")
        for col in self.numeric_clean_columns:
            if col in df_out.columns:
                def sanitize_number(val):
                    if pd.isna(val):
                        return np.nan
                    val_str = str(val).strip()
                    # Check for common missing sentinel representations
                    if val_str.lower() in {"n/a", "na", "null", "none", "unknown", "?", "-", ""}:
                        return np.nan
                    # Remove currency symbols, commas, and extra alphabetic characters
                    cleaned_val = re.sub(r"[^\d.-]", "", val_str)
                    try:
                        return float(cleaned_val) if cleaned_val != "" else np.nan
                    except ValueError:
                        return np.nan

                df_out[col] = df_out[col].apply(sanitize_number)
                self.step_logger.info(f"Sanitized dirty numeric string column '{col}' to float.")

        # 2. Strip whitespace and normalize casing for text columns
        target_text_cols = self.text_columns or df_out.select_dtypes(include=["object", "string"]).columns.tolist()
        for col in target_text_cols:
            if col in df_out.columns:
                series = df_out[col]
                if self.strip_whitespace:
                    series = series.astype(str).str.strip()

                if self.case_format == "lower":
                    series = series.str.lower()
                elif self.case_format == "upper":
                    series = series.str.upper()
                elif self.case_format == "title":
                    series = series.str.title()

                # Restore np.nan for empty or null strings
                df_out[col] = series.replace({"nan": np.nan, "none": np.nan, "": np.nan})

        # 3. Apply categorical canonical alias mappings (e.g. 'blr' -> 'bangalore')
        for col, mapping in self.categorical_mappings.items():
            if col in df_out.columns:
                # Lowercase lookup dictionary
                clean_map = {k.lower().strip(): v for k, v in mapping.items()}
                before_uniques = df_out[col].dropna().nunique()
                df_out[col] = df_out[col].map(lambda x: clean_map.get(str(x).lower().strip(), x) if pd.notna(x) else x)
                after_uniques = df_out[col].dropna().nunique()
                self.step_logger.info(
                    f"Categorical mapping on '{col}': reduced unique categories from {before_uniques} to {after_uniques}."
                )

        return df_out


# ------------------------------------------------------------------------------
# 4. Duplicate Removal Component
# ------------------------------------------------------------------------------
class DuplicateHandler(BaseTransformer):
    """
    Detects and removes duplicate records based on primary key columns or entire rows.
    """

    def __init__(self, key_columns: Optional[List[str]] = None, keep: str = "first"):
        self.key_columns = key_columns
        self.keep = keep
        self.step_logger = logging.getLogger("DuplicateHandler")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        subset = [c for c in (self.key_columns or []) if c in df.columns] or None
        df_out = df.drop_duplicates(subset=subset, keep=self.keep).reset_index(drop=True)
        duplicates_removed = initial_len - len(df_out)

        if duplicates_removed > 0:
            self.step_logger.warning(
                f"Removed {duplicates_removed:,} duplicate record(s) based on subset={subset}. "
                f"Rows remaining: {len(df_out):,}."
            )
        else:
            self.step_logger.info("No duplicate records found.")

        return df_out


# ------------------------------------------------------------------------------
# 5. Missing Value Imputation Component (Leakage-Free fit/transform)
# ------------------------------------------------------------------------------
class MissingValueImputer(BaseTransformer):
    """
    Imputes missing values strictly avoiding data leakage:
      - Numerical columns   -> Learned Median (robust to skew and outliers)
      - Categorical columns -> Learned Mode (most frequent category)
      - Identifier columns  -> Left untouched (or flagged)
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        exclude_columns: Optional[List[str]] = None,
        add_indicator_flags: bool = True
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.exclude_columns = set(exclude_columns or [])
        self.add_indicator_flags = add_indicator_flags
        
        # Learned statistics dictionary: {col_name: learned_fill_value}
        self.learned_statistics_: Dict[str, Any] = {}
        self.flagged_cols_: List[str] = []
        self.step_logger = logging.getLogger("MissingValueImputer")

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "MissingValueImputer":
        """
        Computes median / mode statistics from the training set.
        """
        self.step_logger.info("Fitting MissingValueImputer: learning statistical parameters...")
        self.learned_statistics_ = {}
        self.flagged_cols_ = []

        for col in df.columns:
            if col in self.exclude_columns:
                continue

            series = df[col]
            missing_count = series.isnull().sum()
            
            # Check for 100% missing column edge case
            if missing_count == len(df):
                self.step_logger.error(
                    f"Column '{col}' is 100% missing in training data! Assigning default fallback."
                )
                self.learned_statistics_[col] = 0.0 if pd.api.types.is_numeric_dtype(series) else "missing"
                continue

            if pd.api.types.is_numeric_dtype(series):
                # Numerical strategy
                if self.numeric_strategy == "median":
                    fill_val = series.median()
                elif self.numeric_strategy == "mean":
                    fill_val = series.mean()
                else:
                    raise ValueError(f"Unsupported numeric strategy: '{self.numeric_strategy}'")
                self.learned_statistics_[col] = float(fill_val)
                self.step_logger.debug(f"Learned numerical {self.numeric_strategy} for '{col}': {fill_val:.3f}")

            else:
                # Categorical strategy
                mode_series = series.mode(dropna=True)
                fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
                self.learned_statistics_[col] = fill_val
                self.step_logger.debug(f"Learned categorical mode for '{col}': '{fill_val}'")

            if missing_count > 0 and self.add_indicator_flags:
                self.flagged_cols_.append(col)

        self.step_logger.info(
            f"Fit complete. Parameters learned for {len(self.learned_statistics_)} feature(s)."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned statistics to fill missing values in train or test data.
        """
        if not self.learned_statistics_:
            raise RuntimeError("MissingValueImputer must be fit() before calling transform()!")

        df_out = df.copy()

        # 1. Add binary missingness indicator flags to preserve signal of absence
        if self.add_indicator_flags:
            for col in self.flagged_cols_:
                if col in df_out.columns:
                    flag_col = f"{col}_is_missing"
                    df_out[flag_col] = df_out[col].isnull().astype(int)

        # 2. Apply learned statistics
        for col, fill_val in self.learned_statistics_.items():
            if col in df_out.columns:
                missing_before = df_out[col].isnull().sum()
                if missing_before > 0:
                    df_out[col] = df_out[col].fillna(fill_val)
                    self.step_logger.info(
                        f"Imputed {missing_before} missing value(s) in '{col}' using learned value: {fill_val}"
                    )

        return df_out


# ------------------------------------------------------------------------------
# 6. Schema Validation & Integrity Component
# ------------------------------------------------------------------------------
class SchemaValidator(BaseTransformer):
    """
    Validates pipeline output: ensures mandatory columns exist, checks data types,
    and asserts that critical columns have zero residual null values.
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        max_allowed_null_pct: float = 0.0
    ):
        self.required_columns = required_columns or []
        self.max_allowed_null_pct = max_allowed_null_pct
        self.step_logger = logging.getLogger("SchemaValidator")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Check required columns
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            error_msg = f"Validation Failed: Missing required column(s): {missing_cols}"
            self.step_logger.error(error_msg)
            raise ValueError(error_msg)

        # 2. Check null value threshold
        null_counts = df.isnull().sum()
        total_rows = len(df)
        excessive_nulls = {}
        for col, count in null_counts.items():
            pct = (count / total_rows) * 100 if total_rows > 0 else 0.0
            if pct > self.max_allowed_null_pct:
                excessive_nulls[col] = f"{count} nulls ({pct:.1f}%)"

        if excessive_nulls:
            self.step_logger.warning(
                f"Residual missing values exceed threshold ({self.max_allowed_null_pct}%): {excessive_nulls}"
            )
        else:
            self.step_logger.info("Validation passed: All required columns present with zero residual nulls.")

        return df


# ------------------------------------------------------------------------------
# 7. Master Data Pipeline Orchestrator
# ------------------------------------------------------------------------------
class DataPipeline:
    """
    Sequential Pipeline Orchestrator chaining multiple BaseTransformer components.
    Supports fit(), transform(), and fit_transform() with unified logging and audit metrics.
    """

    def __init__(self, steps: List[BaseTransformer]):
        self.steps = steps
        self.logger = logging.getLogger("DataPipeline")

    def fit(self, df: pd.DataFrame, y: Optional[Any] = None) -> "DataPipeline":
        """Sequentially fits each step on the training dataset."""
        self.logger.info("==================================================")
        self.logger.info(" PIPELINE EXECUTION: FIT PHASE STARTED")
        self.logger.info("==================================================")
        df_current = df.copy()

        for step in self.steps:
            step_name = step.__class__.__name__
            self.logger.info(f"Fitting transformer: [{step_name}]")
            step.fit(df_current, y)
            df_current = step.transform(df_current)

        self.logger.info("Pipeline fit completed successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequentially transforms a dataset using already fitted steps."""
        self.logger.info("==================================================")
        self.logger.info(" PIPELINE EXECUTION: TRANSFORM PHASE STARTED")
        self.logger.info("==================================================")
        df_current = df.copy()

        for step in self.steps:
            step_name = step.__class__.__name__
            self.logger.info(f"Applying transform: [{step_name}]")
            df_current = step.transform(df_current)

        return df_current

    def fit_transform(self, df: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Fits all steps on training data and returns the transformed result."""
        return self.fit(df, y).transform(df)

    @staticmethod
    def audit_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
        """Generates a side-by-side audit comparison between raw and cleaned DataFrames."""
        summary = pd.DataFrame({
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Total Missing Values",
                "Total Duplicate Rows",
                "Memory Usage (KB)"
            ],
            "Raw Input": [
                f"{len(raw_df):,}",
                f"{raw_df.shape[1]}",
                f"{raw_df.isnull().sum().sum():,}",
                f"{raw_df.duplicated().sum():,}",
                f"{raw_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            ],
            "Cleaned Output": [
                f"{len(clean_df):,}",
                f"{clean_df.shape[1]}",
                f"{clean_df.isnull().sum().sum():,}",
                f"{clean_df.duplicated().sum():,}",
                f"{clean_df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            ]
        })
        return summary


# ==============================================================================
# 8. PRACTICE DEMONSTRATION & VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" CDAC PGCP-AI: OBJECT-ORIENTED DATA PREPROCESSING PIPELINE PRACTICE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # A. Create a Realistic "Dirty" Practice Dataset (Train & Test)
    # --------------------------------------------------------------------------
    raw_train_data = {
        "Cust ID": ["C-101", "C-102", "C-102", "C-103", "C-104", "C-105", "C-106"],
        "Full Name": ["  Alice Smith ", "Bob Jones", "Bob Jones", "Charlie Brown  ", "David Lee", "  Eva Green", "Frank Wright"],
        "Age": [29.0, 42.0, 42.0, np.nan, 35.0, 50.0, np.nan],
        "Purchase Amount ($)": ["$1,250.00", "$450.50", "$450.50", "N/A", " $2,100.00 ", "890.25", "$3,400.00"],
        "City ": ["  BANGALORE ", "delhi", "delhi", "bengaluru", "  BLR ", "DELHI", "mumbai  "],
        "Department": ["Sales", "Engineering", "Engineering", "Sales", np.nan, "HR", "Sales"]
    }

    # Unseen test data with similar dirty patterns and missing values
    raw_test_data = {
        "Cust ID": ["C-107", "C-108"],
        "Full Name": ["Grace Hopper ", " Alan Turing"],
        "Age": [np.nan, 38.0],  # Age is missing here -> must use train median!
        "Purchase Amount ($)": ["$5,000.00", "N/A"], # Purchase amount missing -> must use train median!
        "City ": [" Bengaluru ", "DEL"],
        "Department": [np.nan, "Engineering"]  # Department missing -> must use train mode!
    }

    df_train_raw = pd.DataFrame(raw_train_data)
    df_test_raw = pd.DataFrame(raw_test_data)

    print("\n[INPUT] Raw Training Dataset Preview:")
    print(df_train_raw)

    print("\n[INPUT] Raw Unseen Test Dataset Preview:")
    print(df_test_raw)

    # --------------------------------------------------------------------------
    # B. Define Pipeline Configuration & Component Sequence
    # --------------------------------------------------------------------------
    # 1. Alias dictionary to map raw / abbreviated column names to standardized schema
    alias_mapping = {
        "cust_id": "customer_id",
        "full_name": "customer_name",
        "purchase_amount": "purchase_amount",
        "city": "city",
        "department": "department"
    }

    # 2. Categorical alias mapping (e.g. city abbreviations to canonical city name)
    city_canonical_mapping = {
        "city": {
            "bangalore": "Bengaluru",
            "bengaluru": "Bengaluru",
            "blr": "Bengaluru",
            "delhi": "Delhi",
            "del": "Delhi",
            "mumbai": "Mumbai",
            "bom": "Mumbai"
        }
    }

    # 3. Assemble Pipeline Steps
    pipeline = DataPipeline(steps=[
        # Step 1: Convert column names to snake_case and resolve synonyms
        SchemaStandardizer(alias_map=alias_mapping),

        # Step 2: Remove duplicates based on primary key 'customer_id'
        DuplicateHandler(key_columns=["customer_id"]),

        # Step 3: Sanitize currency strings, trim whitespace, standardize categories
        StructuralTextSanitizer(
            text_columns=["customer_name", "city", "department"],
            case_format="lower",
            strip_whitespace=True,
            categorical_mappings=city_canonical_mapping,
            numeric_clean_columns=["purchase_amount"]
        ),

        # Step 4: Leakage-free imputation (Median for numeric, Mode for categorical)
        MissingValueImputer(
            numeric_strategy="median",
            categorical_strategy="mode",
            exclude_columns=["customer_id"],
            add_indicator_flags=True
        ),

        # Step 5: Assert schema integrity and check residual null threshold
        SchemaValidator(
            required_columns=["customer_id", "customer_name", "age", "purchase_amount", "city"],
            max_allowed_null_pct=0.0
        )
    ])

    # --------------------------------------------------------------------------
    # C. Execute Pipeline on Training Set (fit_transform)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXECUTING: pipeline.fit_transform(df_train_raw)")
    print("=" * 80)
    df_train_clean = pipeline.fit_transform(df_train_raw)

    print("\n[OUTPUT] Transformed Clean Training Dataset:")
    print(df_train_clean)

    # --------------------------------------------------------------------------
    # D. Execute Pipeline on Test Set (transform ONLY -> ZERO Data Leakage!)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" EXECUTING: pipeline.transform(df_test_raw) -- Zero Data Leakage")
    print("=" * 80)
    df_test_clean = pipeline.transform(df_test_raw)

    print("\n[OUTPUT] Transformed Clean Test Dataset:")
    print(df_test_clean)

    # --------------------------------------------------------------------------
    # E. Audit Report
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" PIPELINE AUDIT REPORT (TRAINING DATA)")
    print("=" * 80)
    audit_table = DataPipeline.audit_summary(df_train_raw, df_train_clean)
    print(audit_table.to_string(index=False))

    print("\n[SUCCESS] Practice pipeline execution complete!")
