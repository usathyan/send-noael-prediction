import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Path as FastApiPath
from pydantic import BaseModel, Field
import uvicorn
# Add CORS middleware import
from fastapi.middleware.cors import CORSMiddleware

# Import functions from our modules
# Need to adjust Python path if running from root or handle packaging
# Assuming execution from the root directory for now
from python.data_processing.send_loader import load_send_study, validate_send_domains
from python.data_processing.domain_parser import parse_study_data
from python.data_processing.feature_extractor import extract_features
# Remove TxGemma related imports
# from python.model.txgemma_wrapper import predict_with_txgemma
# from python.model.noael_calculator import calculate_noael_from_prediction
# from python.model.confidence_scorer import generate_confidence_score
# Import the new ML predictor
from python.model.ml_predictor import load_pretrained_model, predict_noael_ml
# Import the TxGemma NOAEL demo function
from python.txgemma_demos.noael_demo import run_noael_determination_demo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'
)

# --- Configuration --- #
# No longer need TxGemma model name
# TXGEMMA_MODEL_NAME = "google/txgemma-2b-predict"
# Directory to store uploaded and unzipped studies (can be configurable)
UPLOAD_DIR = Path("./uploaded_studies")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load the ML model on startup (or handle None if it doesn't exist)
PRELOADED_ML_MODEL = load_pretrained_model()
if PRELOADED_ML_MODEL is None:
    logging.warning("ML Model not found on startup. /predict endpoint will return dummy values.")

# --- FastAPI App --- #
app = FastAPI(
    title="SEND NOAEL Prediction API (Traditional ML)",
    description="API to upload SEND studies and predict NOAEL using a pre-trained ML model.",
    version="0.2.0" # Version bump
)

# --- CORS Configuration --- #
# Define allowed origins (adjust if your frontend runs on a different port)
origins = [
    "http://localhost",         # Allow requests from base localhost
    "http://localhost:3000",    # Default Next.js dev port
    # Add any other origins if needed (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allow GET and POST requests
    allow_headers=["*"],         # Allow all headers
)

# --- Pydantic Models (Updated for ML approach) ---
class UploadResponse(BaseModel):
    message: str
    study_id: str
    study_path: str

class NoaelResultML(BaseModel):
    predicted_noael: Optional[float] = Field(..., example=50.0)
    units: str = Field(default="mg/kg/day", example="mg/kg/day") # Assume units for now
    model_used: str = Field(default="XGBoost", example="XGBoost") # Example
    status: str = Field(..., example="Prediction successful") # Indicate success/failure/dummy

class PredictionResponseML(BaseModel):
    study_id: str
    noael_result: NoaelResultML
    # Confidence scores might need separate calculation or model property
    confidence: Optional[float] = Field(default=None, example=0.85) 
    error: Optional[str] = None

class DemoNoaelResult(BaseModel): # Can be nested or kept separate
    overall_noael: Optional[float] = Field(None, description="Overall NOAEL determined by stats analysis (most sensitive endpoint)")
    dose_units: str
    analysis_summary: Optional[Dict] = Field(None, description="Summary of statistical analysis results") # Placeholder
    per_endpoint_noael: Optional[Dict] = Field(None, description="NOAEL/LOAEL determined per endpoint") # Placeholder
    summary_prompt: Optional[str] = Field(None, description="Generated text prompt summarizing findings for LLM")
    simulated_response: Optional[str] = Field(None, description="Simulated textual response based on analysis")

class DemoResponse(BaseModel):
    study_id: str
    demo_name: str
    results: Optional[DemoNoaelResult] = None # Using specific model
    raw_results: Optional[Dict] = Field(None, description="Full raw output dictionary from the demo function") # For debugging
    error: Optional[str] = None

# --- Helper Functions ---
def get_study_path(study_id: str) -> Path:
    """Gets the path to the unzipped study directory."""
    return UPLOAD_DIR / study_id

# --- API Endpoints ---
@app.post("/upload/", response_model=UploadResponse, status_code=201)
async def upload_send_study(file: UploadFile = File(..., description="Zip file containing SEND study XPT files")):
    """Uploads a zip archive of a SEND study.

    - Saves the zip file.
    - Extracts its contents into a unique directory under UPLOAD_DIR.
    - Returns the study ID (directory name).
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a Zip file.")

    # Create a unique directory for the study (e.g., using filename without extension)
    study_id = Path(file.filename).stem
    study_path = get_study_path(study_id)

    if study_path.exists():
        # Basic handling for existing study - could overwrite or raise error
        logging.warning(f"Study directory '{study_id}' already exists. Overwriting.")
        # Consider removing existing contents if overwriting
    else:
        study_path.mkdir(parents=True)

    try:
        # Save the zip file temporarily
        temp_zip_path = study_path / file.filename
        with open(temp_zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logging.info(f"Saved uploaded zip file to: {temp_zip_path}")

        # Extract the zip file
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(study_path)
        logging.info(f"Extracted zip file contents to: {study_path}")

        # Clean up the zip file after extraction
        temp_zip_path.unlink()

        # Basic check: See if any .xpt files were extracted
        if not list(study_path.glob('*.xpt')):
            # Clean up directory if no XPT files found
            # shutil.rmtree(study_path) # Consider cleanup
            raise HTTPException(status_code=400, detail="Zip file did not contain any .xpt files.")

        return UploadResponse(
            message="Study uploaded and extracted successfully.",
            study_id=study_id,
            study_path=str(study_path)
        )

    except zipfile.BadZipFile:
        # Clean up directory on error
        # shutil.rmtree(study_path) # Consider cleanup
        raise HTTPException(status_code=400, detail="Invalid or corrupted zip file.")
    except Exception as e:
        # Clean up directory on error
        # shutil.rmtree(study_path) # Consider cleanup
        logging.error(f"Error during file upload/extraction for {study_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {e}")
    finally:
         await file.close()


@app.post("/predict/{study_id}", response_model=PredictionResponseML)
async def run_noael_prediction_ml(study_id: str = FastApiPath(..., description="The ID (directory name) of the uploaded study")):
    """Runs the NOAEL prediction pipeline using a pre-trained ML model."""
    study_path = get_study_path(study_id)
    if not study_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Study ID '{study_id}' not found.")

    try:
        logging.info(f"--- Starting ML Prediction Pipeline for Study: {study_id} --- ")

        # 1. Load Data
        logging.info("Step 1: Loading SEND domains...")
        loaded_data = load_send_study(study_path)
        if not loaded_data:
            raise HTTPException(status_code=400, detail="Failed to load SEND domains.")

        # 2. Validate Data
        logging.info("Step 2: Validating loaded domains...")
        if not validate_send_domains(loaded_data):
            raise HTTPException(status_code=400, detail="SEND domain validation failed.")

        # 3. Parse Data
        logging.info("Step 3: Parsing domain data...")
        parsed_data = parse_study_data(loaded_data)

        # 4. Extract ML Features
        logging.info("Step 4: Extracting features for ML model...")
        features_df = extract_features(parsed_data)
        if features_df is None or features_df.empty:
             logging.error(f"Feature extraction failed or produced empty features for study {study_id}.")
             raise HTTPException(status_code=500, detail="Feature extraction failed or produced no features.")

        # 5. Predict using ML Model
        logging.info("Step 5: Running prediction using pre-loaded ML model...")
        predicted_value = predict_noael_ml(features_df, PRELOADED_ML_MODEL)

        status_msg = "Prediction successful"
        if PRELOADED_ML_MODEL is None:
            status_msg = "Dummy prediction (No model loaded)"
        elif predicted_value is None:
            status_msg = "Prediction failed (Error during prediction)"
        
        # Assume units for now - this should ideally come from model metadata or config
        noael_units = "mg/kg/day"

        logging.info(f"--- ML Prediction Pipeline Finished for Study: {study_id} --- ")

        # Create response object
        result = NoaelResultML(
            predicted_noael=predicted_value,
            units=noael_units,
            status=status_msg
        )

        return PredictionResponseML(
            study_id=study_id,
            noael_result=result,
            confidence=None # Placeholder - add confidence if model provides it
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error during prediction pipeline for study {study_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

@app.post("/predict/{study_id}/txgemma_demos/noael_determination", response_model=DemoResponse, tags=["TxGemma Demos"])
async def run_txgemma_noael_demo(
    study_id: str = FastApiPath(..., description="The ID (directory name) of the uploaded study")
):
    """Runs the TxGemma NOAEL Determination Demo.

    This demo performs statistical analysis on study endpoints (LB, BW) 
    to determine per-endpoint NOAELs and an overall NOAEL based on the most sensitive finding.
    It then generates a text summary and simulates a potential LLM response based on this analysis.
    **Note:** The final response is *simulated* and does not involve a live LLM call.
    """
    study_path = get_study_path(study_id)
    if not study_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Study ID '{study_id}' not found.")

    try:
        logging.info(f"--- Starting TxGemma NOAEL Demo for Study: {study_id} --- ")

        # 1. Load Data (Reusing existing functions)
        logging.info("Step 1: Loading SEND domains...")
        loaded_data = load_send_study(study_path)
        if not loaded_data:
            raise HTTPException(status_code=400, detail="Failed to load SEND domains.")

        # 2. Validate Data (Reusing existing functions)
        logging.info("Step 2: Validating loaded domains...")
        if not validate_send_domains(loaded_data):
            raise HTTPException(status_code=400, detail="SEND domain validation failed.")

        # 3. Parse Data (Reusing existing functions)
        logging.info("Step 3: Parsing domain data...")
        parsed_data = parse_study_data(loaded_data)

        # 4. Run the Demo Logic
        logging.info("Step 4: Running NOAEL determination demo logic...")
        demo_results = run_noael_determination_demo(parsed_data)

        logging.info(f"--- TxGemma NOAEL Demo Finished for Study: {study_id} --- ")
        
        if demo_results.get("error"):
            # Return error from the demo pipeline itself
             return DemoResponse(
                study_id=study_id, 
                demo_name="Automated NOAEL Determination (Simulated)",
                results=None,
                raw_results=demo_results, # Include raw results for debugging
                error=demo_results["error"]
             )

        # Structure the successful response
        structured_result = DemoNoaelResult(
            overall_noael=demo_results.get('overall_noael'),
            dose_units=demo_results.get('dose_units', 'mg/kg/day'),
            # Optionally include summaries if needed for UI, keep raw_results for full detail
            # analysis_summary=demo_results.get('analysis_results'), 
            # per_endpoint_noael=demo_results.get('per_endpoint_noael'),
            summary_prompt=demo_results.get('summary_prompt'),
            simulated_response=demo_results.get('simulated_response')
        )

        return DemoResponse(
            study_id=study_id,
            demo_name="Automated NOAEL Determination (Simulated)",
            results=structured_result,
            raw_results=demo_results, # Include raw results for debugging
            error=None
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error during TxGemma NOAEL demo endpoint for study {study_id}: {e}", exc_info=True)
        # Return generic error response
        return DemoResponse(
            study_id=study_id,
            demo_name="Automated NOAEL Determination (Simulated)",
            results=None,
            error=f"Internal server error during demo execution: {e}"
        )

# --- Root Endpoint --- #
@app.get("/")
async def root():
    return {"message": "Welcome to the SEND NOAEL Prediction API (Traditional ML). Use /docs for documentation."}

# --- Uvicorn Runner (for direct execution) ---
if __name__ == "__main__":
    print("Starting Uvicorn server for ML prediction API...")
    print("Access the API at http://127.0.0.1:8000")
    print("API Documentation available at http://127.0.0.1:8000/docs")
    # Run using the specific command for consistency:
    # .venv/bin/python -m uvicorn python.api.main:app --reload --host 127.0.0.1 --port 8000
    # Or let uvicorn handle it, assuming environment is activated:
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, workers=1) 