"""
O07_orchestrator.py
Main master runner chaining feature engineering, transformation, selection, and artifact serialization.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import joblib

# Ensure local imports work regardless of execution working directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from O01_domain_feature_builder import DomainFeatureBuilder
from O02_categorical_encoder import CategoricalEncoder
from O03_numerical_transformer import NumericalTransformer
from O04_cardinality_dimensionality_reducer import CardinalityDimensionalityReducer
from O05_feature_selector import FeatureSelector
from O06_data_leakage_validator import DataLeakageValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage4Orchestrator")


def run_stage4_pipeline():
    """Chains feature engineering, encoding, transformation, selection, and artifact serialization."""
    logger.info("==================================================")
    logger.info(" STARTING STAGE 4: FEATURE ENGINEERING & SELECTION")
    logger.info("==================================================")

    # 1. Resolve Paths
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    config_path = project_root / "config/O01_feature_rules.yaml"
    models_dir = project_root / "models"
    processed_dir = project_root / "data/processed"

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_file = project_root / "data/raw/input.csv"
    if input_file.exists():
        logger.info(f"Loading input dataset from: {input_file}")
        df = pd.read_csv(input_file)
    else:
        logger.info(f"Input file not found at {input_file}. Generating synthetic benchmark dataset for Stage 4...")
        df = pd.DataFrame({
            "customer_id": [f"CUST-{i:03d}" for i in range(1, 101)],
            "tenure_months": [12, 24, 6, 48, 3, 18, 36, 12, 60, 2] * 10,
            "monthly_charges": [29.95, 75.0, 19.0, 105.0, 45.0, 80.0, 95.0, 30.0, 110.0, 20.0] * 10,
            "total_charges": [359.4, 1800.0, 114.0, 5040.0, 135.0, 1440.0, 3420.0, 360.0, 6600.0, 40.0] * 10,
            "contract_type": ["month-to-month", "one-year", "month-to-month", "two-year", "month-to-month"] * 20,
            "payment_method": ["electronic_check", "mailed_check", "credit_card", "bank_transfer", "mailed_check"] * 20,
            "churn_status": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 10
        })

    target_col = "churn_status"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from input dataset.")

    # STEP 1: Strict Train/Test Split (Preventing Global Data Leakage)
    logger.info("\n--- STEP 1: Stratified Train/Test Partitioning ---")
    leakage_validator = DataLeakageValidator(test_size=0.2, random_state=42)
    train_df, test_df = leakage_validator.stratified_train_test_split(df, target_col)

    # Instantiate Feature Engineering Engines
    domain_builder = DomainFeatureBuilder(config_path=config_path)
    cardinality_reducer = CardinalityDimensionalityReducer(config_path=config_path)
    cat_encoder = CategoricalEncoder(config_path=config_path)
    num_transformer = NumericalTransformer(config_path=config_path)
    feature_selector = FeatureSelector(config_path=config_path)

    # STEP 2: Domain Feature Building
    logger.info("\n--- STEP 2: Domain Feature Construction ---")
    train_df = domain_builder.build_features(train_df)
    test_df = domain_builder.build_features(test_df)

    # STEP 3: Cardinality & Rare Level Grouping
    logger.info("\n--- STEP 3: Rare-Label Cardinality Reduction ---")
    train_df = cardinality_reducer.group_rare_categories(train_df, is_train=True)
    test_df = cardinality_reducer.group_rare_categories(test_df, is_train=False)

    # STEP 4: Categorical Encoding
    logger.info("\n--- STEP 4: Categorical Encoding (One-Hot & Target) ---")
    train_df = cat_encoder.fit_transform_low_cardinality(train_df, is_train=True)
    test_df = cat_encoder.fit_transform_low_cardinality(test_df, is_train=False)
    train_df = cat_encoder.fit_transform_target_encoding(train_df, target_col=target_col, is_train=True)
    test_df = cat_encoder.fit_transform_target_encoding(test_df, target_col=target_col, is_train=False)

    # STEP 5: Numerical Power Transformation & Scaling
    logger.info("\n--- STEP 5: Numerical Power Transformation & Scaling ---")
    id_cols = [c for c in ["customer_id", target_col] if c in train_df.columns]
    train_df = num_transformer.fit_transform_features(train_df, exclude_cols=id_cols, is_train=True)
    test_df = num_transformer.fit_transform_features(test_df, exclude_cols=id_cols, is_train=False)

    # STEP 6: Multi-Stage Feature Selection
    logger.info("\n--- STEP 6: Multi-Stage Feature Selection ---")
    train_df = feature_selector.apply_variance_threshold(train_df, exclude_cols=id_cols, is_train=True)
    test_df = feature_selector.apply_variance_threshold(test_df, exclude_cols=id_cols, is_train=False)

    train_df = feature_selector.apply_correlation_filter(train_df, exclude_cols=id_cols)
    test_df = feature_selector.apply_correlation_filter(test_df, exclude_cols=id_cols)

    train_df = feature_selector.apply_statistical_scoring(train_df, target_col=target_col, exclude_cols=["customer_id"], is_train=True)
    test_df = feature_selector.apply_statistical_scoring(test_df, target_col=target_col, exclude_cols=["customer_id"], is_train=False)

    # STEP 7: Pipeline Artifact Serialization & Export
    logger.info("\n--- STEP 7: Pipeline Artifact Serialization & Export ---")
    train_output_path = processed_dir / "train_features.csv"
    test_output_path = processed_dir / "test_features.csv"
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Saved processed train features -> {train_output_path} ({train_df.shape})")
    logger.info(f"Saved processed test features  -> {test_output_path} ({test_df.shape})")

    pipeline_artifacts = {
        "domain_builder": domain_builder,
        "cardinality_reducer": cardinality_reducer,
        "cat_encoder": cat_encoder,
        "num_transformer": num_transformer,
        "feature_selector": feature_selector,
    }
    model_artifact_path = models_dir / "stage4_feature_pipeline.joblib"
    joblib.dump(pipeline_artifacts, model_artifact_path)
    logger.info(f"Saved fitted transformer artifacts -> {model_artifact_path}")
    logger.info("==================================================")
    logger.info(" STAGE 4 FEATURE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("==================================================")


if __name__ == "__main__":
    run_stage4_pipeline()