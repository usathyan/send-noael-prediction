import pandas as pd
import numpy as np
import logging
from typing import Optional, Any, List
import joblib # For saving/loading models
import xgboost as xgb # Example model
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration --- #
# Path where a pre-trained model would be saved/loaded from
MODEL_DIR = Path("python/model/saved_models")
MODEL_FILENAME = "noael_xgboost_model.joblib" # Example filename
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Define the exact features the model expects ---
# IMPORTANT: This list MUST match the output of feature_extractor.preprocess_for_ml
EXPECTED_FEATURES = [
    'MAX_DOSE', 'SEX_encoded', 'EXTRT_encoded', 'EXROUTE_encoded', 'DOSE_UNIT_encoded',
    'LB_mean__ALB', 'LB_mean__ALBGLOB', 'LB_mean__ALP', 'LB_mean__ALT', 'LB_mean__ANISO',
    'LB_mean__APTT', 'LB_mean__AST', 'LB_mean__BACT', 'LB_mean__BASO', 'LB_mean__BASOLE',
    'LB_mean__BILDIR', 'LB_mean__BILI', 'LB_mean__CA', 'LB_mean__CASTS', 'LB_mean__CHOL',
    'LB_mean__CK', 'LB_mean__CL', 'LB_mean__CLARITY', 'LB_mean__COLOR', 'LB_mean__CREAT',
    'LB_mean__CRYSTALS', 'LB_mean__CSUNCLA', 'LB_mean__CYUNCLA', 'LB_mean__EOS', 'LB_mean__EOSLE',
    'LB_mean__EPIC', 'LB_mean__GGT', 'LB_mean__GLOBUL', 'LB_mean__GLUC', 'LB_mean__HCT',
    'LB_mean__HGB', 'LB_mean__HPOCROM', 'LB_mean__K', 'LB_mean__KETONES', 'LB_mean__LGLUCLE',
    'LB_mean__LGUNSCE', 'LB_mean__LYM', 'LB_mean__LYMLE', 'LB_mean__MCH', 'LB_mean__MCHC',
    'LB_mean__MCV', 'LB_mean__MONO', 'LB_mean__MONOLE', 'LB_mean__NEUT', 'LB_mean__NEUTLE',
    'LB_mean__OCCBLD', 'LB_mean__OTHR', 'LB_mean__PH', 'LB_mean__PHOS', 'LB_mean__PLAT',
    'LB_mean__POIKILO', 'LB_mean__POLYCHR', 'LB_mean__PROT', 'LB_mean__PT', 'LB_mean__RBC',
    'LB_mean__RETI', 'LB_mean__RETIRBC', 'LB_mean__SODIUM', 'LB_mean__SPGRAV', 'LB_mean__TOXGRAN',
    'LB_mean__TRIG', 'LB_mean__UREAN', 'LB_mean__UROBIL', 'LB_mean__VOLUME', 'LB_mean__WBC',
    'LB_max__ALB', 'LB_max__ALBGLOB', 'LB_max__ALP', 'LB_max__ALT', 'LB_max__ANISO',
    'LB_max__APTT', 'LB_max__AST', 'LB_max__BACT', 'LB_max__BASO', 'LB_max__BASOLE',
    'LB_max__BILDIR', 'LB_max__BILI', 'LB_max__CA', 'LB_max__CASTS', 'LB_max__CHOL',
    'LB_max__CK', 'LB_max__CL', 'LB_max__CLARITY', 'LB_max__COLOR', 'LB_max__CREAT',
    'LB_max__CRYSTALS', 'LB_max__CSUNCLA', 'LB_max__CYUNCLA', 'LB_max__EOS', 'LB_max__EOSLE',
    'LB_max__EPIC', 'LB_max__GGT', 'LB_max__GLOBUL', 'LB_max__GLUC', 'LB_max__HCT',
    'LB_max__HGB', 'LB_max__HPOCROM', 'LB_max__K', 'LB_max__KETONES', 'LB_max__LGLUCLE',
    'LB_max__LGUNSCE', 'LB_max__LYM', 'LB_max__LYMLE', 'LB_max__MCH', 'LB_max__MCHC',
    'LB_max__MCV', 'LB_max__MONO', 'LB_max__MONOLE', 'LB_max__NEUT', 'LB_max__NEUTLE',
    'LB_max__OCCBLD', 'LB_max__OTHR', 'LB_max__PH', 'LB_max__PHOS', 'LB_max__PLAT',
    'LB_max__POIKILO', 'LB_max__POLYCHR', 'LB_max__PROT', 'LB_max__PT', 'LB_max__RBC',
    'LB_max__RETI', 'LB_max__RETIRBC', 'LB_max__SODIUM', 'LB_max__SPGRAV', 'LB_max__TOXGRAN',
    'LB_max__TRIG', 'LB_max__UREAN', 'LB_max__UROBIL', 'LB_max__VOLUME', 'LB_max__WBC',
    'TS_PLANNED_DURATION'
    # NOTE: Add LB min/std and BW features here if they were included during actual model training
]

# --- Model Loading --- #
def load_pretrained_model(model_path: Path = MODEL_PATH) -> Optional[Any]:
    """Loads a pre-trained ML model from a file using joblib."""
    if not model_path.exists():
        logger.warning(f"Pre-trained model not found at {model_path}.")
        logger.warning("Returning a dummy predictor. Predictions will not be meaningful.")
        # Return a dummy object or a simple baseline model if needed
        # For now, returning None to indicate no real model loaded
        return None 
        
    try:
        logger.info(f"Loading pre-trained model from {model_path}...")
        model = joblib.load(model_path)
        logger.info("Pre-trained model loaded successfully.")
        # Basic check if it looks like an XGBoost model (adjust if using others)
        if not isinstance(model, xgb.XGBRegressor):
             logger.warning("Loaded object might not be the expected XGBoost model type.")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None

# --- Prediction --- #
def predict_noael_ml(features_df: pd.DataFrame, model: Optional[Any]) -> Optional[float]:
    """Predicts NOAEL using the loaded ML model.

    Args:
        features_df: The DataFrame of numerical features from feature_extractor.
        model: The loaded pre-trained ML model (e.g., XGBoost regressor).
               If None, returns a dummy prediction.

    Returns:
        The predicted NOAEL value (float), or None if prediction fails.
    """
    if model is None:
        logger.warning("No model provided, returning dummy NOAEL prediction (e.g., 0.0).")
        return 0.0 # Or np.nan, or None depending on desired handling
        
    if features_df is None or features_df.empty:
        logger.error("Cannot predict: Input features DataFrame is empty or None.")
        return None
        
    logger.info(f"Predicting NOAEL using loaded ML model on features with shape {features_df.shape}...")
    
    try:
        # Ensure feature names/order match what the model was trained on
        # This might require loading expected feature names alongside the model
        # Or ensuring the feature extractor always produces columns in the same order.
        # Assuming for now the input features_df columns match the training columns.
        
        # XGBoost can sometimes handle DataFrames directly, or might need NumPy array
        # prediction = model.predict(features_df.to_numpy()) 
        prediction = model.predict(features_df)
        
        # Prediction might be an array (e.g., if multiple rows were passed)
        # Assuming features_df has only one row (study-level features)
        predicted_value = float(prediction[0]) 
        logger.info(f"Predicted NOAEL value: {predicted_value}")
        return predicted_value
        
    except Exception as e:
        logger.error(f"Error during ML model prediction: {e}", exc_info=True)
        logger.error(f"Feature columns: {features_df.columns.tolist()}") # Log columns on error
        return None

# --- Dummy Model Creation --- #
def create_and_save_dummy_model(model_path: Path = MODEL_PATH, feature_names: List[str] = EXPECTED_FEATURES):
    """Creates a simple dummy XGBoost model and saves it."""
    logger.info("--- Creating and saving a dummy XGBoost model (using predefined features) --- ")
    # Create dummy data with the expected feature names
    dummy_data = {feature: np.random.rand(5) * 100 for feature in feature_names}
    X_dummy = pd.DataFrame(dummy_data)
    y_dummy = np.random.randint(0, 5, size=5) # Dummy target values

    # Simple XGBoost classifier
    dummy_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    dummy_model.fit(X_dummy, y_dummy)

    # Save the dummy model
    try:
        joblib.dump(dummy_model, model_path)
        logger.info(f"Dummy model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving dummy model to {model_path}: {e}")

# --- Main Execution (for testing/dummy model creation) ---
if __name__ == "__main__":
    # 1. Create dummy features matching the EXPECTED structure for internal testing
    test_dummy_data = {feature: [np.random.rand() * 10] for feature in EXPECTED_FEATURES}
    dummy_features_df = pd.DataFrame(test_dummy_data)
    print("--- Dummy Features for Test Prediction ---")
    print(dummy_features_df.head())

    # 2. Try loading the *existing* model (if any)
    loaded_model = load_pretrained_model()
    print(f"\nModel Loaded: {loaded_model is not None}")

    # 3. Attempt prediction with the loaded model (might fail if features mismatch)
    if loaded_model:
        prediction = predict_noael_ml(dummy_features_df, loaded_model)
        print(f"\n--- Predicted NOAEL (using loaded model): {prediction} ---")
    else:
        print("\n--- Skipping prediction test (no model loaded) ---")

    # 4. Create and save a NEW dummy model with the CORRECT feature names
    create_and_save_dummy_model(MODEL_PATH, EXPECTED_FEATURES) 