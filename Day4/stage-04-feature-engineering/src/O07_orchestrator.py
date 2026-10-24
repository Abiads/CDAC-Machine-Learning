"""
O07_orchestrator.py
Main master runner chaining feature engineering, transformation, selection, and artifact serialization.
"""

import logging
from pathlib import Path
import pandas as pd
import joblib
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
    # 1. Resolve Paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    config_path = project_root / "config/O01_feature_rules.yaml"
    models_dir = project_root / "models"
    processed_dir = project_root / "data/processed"

    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_file = project_root / "data/raw/input.csv"
    if not input_file.exists():
        return
    else:
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


        # 2 step 1 : Strict Train/Test Split (Preventing Global Data Leakage)
        leakage_validator = DataLeakageValidator(test_size=0.2, random_state=42)
        train_df, test_df = leakage_validator.stratified_train_test_split(df, target_col)

         # 3. Instantiate Feature Engineering Engines
        domain_builder = DomainFeatureBuilder(config_path=config_path)
        cat_encoder = CategoricalEncoder(config_path=config_path)
        num_transformer = NumericalTransformer(config_path=config_path)
        cardinality_reducer = CardinalityDimensionalityReducer(config_path=config_path)
        feature_selector = FeatureSelector(config_path=config_path)

        # 4. STEP 2: Domain Feature Building (Applied independently to train and test)
        logger.info("\n--- STEP 2: Domain Feature Construction ---")
        train_df = domain_builder.build_features(train_df)
        test_df = domain_builder.build_features(test_df)


if __name__ == "__main__":
    run_stage4_pipeline()