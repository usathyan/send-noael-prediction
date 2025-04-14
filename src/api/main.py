import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Import necessary components from our modules (using src/)
from src.data_processing.send_loader import load_send_study
from src.data_processing.domain_parser import parse_domains

# Import processors
from src.processing.noael_processor import process_study_for_txgemma
from src.processing.enhanced_processor import enhanced_process_for_txgemma

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

# Ensure upload directory exists
UPLOAD_DIR = Path("uploaded_studies")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="SEND NOAEL Analysis API (TxGemma Demo)",
    description="API to analyze SEND studies using TxGemma for NOAEL assessment.",
)


# --- Helper Functions ---
def _clean_study_id(study_id: str) -> str:
    """Removes potentially problematic characters for directory names."""
    return "".join(c for c in study_id if c.isalnum() or c in ("-", "_")).rstrip()


def _get_study_dir(study_id: str) -> Path:
    """Gets the directory path for a given study ID."""
    return UPLOAD_DIR / _clean_study_id(study_id)


# Modified _extract_zip to accept a path and handle cleanup
async def _extract_zip_from_path(temp_zip_path: Path, extract_to: Path):
    """Extracts zip file from a given path and cleans up the zip file."""
    try:
        logger.info(f"Starting extraction from {temp_zip_path} to {extract_to}")
        # Ensure target directory exists (should already be created by endpoint)
        extract_to.mkdir(parents=True, exist_ok=True)

        # Using shutil which might be synchronous
        # Replace with async zip library if performance critical and shutil blocks event loop
        shutil.unpack_archive(temp_zip_path, extract_to)
        logger.info(f"Successfully extracted {temp_zip_path.name} to {extract_to}")
    except Exception as e:
        logger.error(
            f"Failed to extract zip file {temp_zip_path.name}: {e}", exc_info=True
        )
        # Optionally, could try to clean up the extract_to directory on failure
        # if extract_to.exists(): shutil.rmtree(extract_to)
    finally:
        # Clean up the temporary zip file regardless of success/failure
        if temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
                logger.info(f"Cleaned up temporary file: {temp_zip_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary zip file {temp_zip_path}: {e}")


# --- API Endpoints ---


@app.post("/upload/", status_code=201)
async def upload_send_study_zip(
    study_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ..., description="Zip archive containing SEND study (.xpt files)"
    ),
):
    """
    Uploads a SEND study as a Zip archive, saves it temporarily,
    and schedules background extraction.
    Replaces existing study data if the study_id already exists.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a Zip archive."
        )

    cleaned_study_id = _clean_study_id(study_id)
    study_dir = _get_study_dir(cleaned_study_id)
    study_dir.mkdir(parents=True, exist_ok=True)  # Ensure base dir exists

    # Define path for the temporary saved zip file within the study dir
    # Using a temporary name helps avoid conflicts if upload is interrupted/retried
    temp_zip_path = study_dir / f"_temp_upload_{file.filename}"

    # --- Save Uploaded File Synchronously ---
    try:
        logger.info(f"Saving uploaded file temporarily to: {temp_zip_path}")
        with open(temp_zip_path, "wb") as buffer:
            # Read file content in chunks to handle large files
            while content := await file.read(1024 * 1024):  # Read 1MB chunks
                buffer.write(content)
        logger.info(f"Successfully saved temporary file: {temp_zip_path}")
    except Exception as e:
        logger.error(
            f"Failed to save uploaded file {file.filename} to {temp_zip_path}: {e}"
        )
        # Clean up partially saved file if it exists
        if temp_zip_path.exists():
            temp_zip_path.unlink()
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {e}"
        )
    finally:
        await file.close()  # Close the upload file handle

    # --- Clear existing extracted data BEFORE scheduling extraction ---
    # Iterate through study_dir and remove files/subdirs EXCEPT the temp zip
    for item in study_dir.iterdir():
        if item.is_file() and item.name != temp_zip_path.name:
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    logger.info(f"Cleared existing extracted data (if any) in {study_dir}")

    # Add extraction from the saved temp file to background tasks
    background_tasks.add_task(_extract_zip_from_path, temp_zip_path, study_dir)

    logger.info(
        f"Accepted upload for study_id: {cleaned_study_id}. Extraction scheduled from {temp_zip_path}"
    )
    return {
        "message": f"Upload accepted for study '{cleaned_study_id}'. Extraction scheduled.",
        "study_id": cleaned_study_id,
    }


@app.post("/analyze_noael/{study_id}")
async def analyze_study_with_txgemma(study_id: str):
    """
    Analyzes a previously uploaded SEND study to generate a NOAEL assessment
    using TxGemma (or configured LLM).
    Focuses on specific findings (e.g., Body Weight) for the prompt.
    """
    logger.info(f"--- Starting TxGemma Analysis Pipeline for Study: {study_id} ---")
    cleaned_study_id = _clean_study_id(study_id)
    study_dir = _get_study_dir(cleaned_study_id)

    if not study_dir.exists() or not any(study_dir.iterdir()):
        logger.error(
            f"Study data directory not found or empty for study_id: {cleaned_study_id} at {study_dir}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Study '{cleaned_study_id}' not found or data is missing. Please upload first.",
        )

    try:
        # 1. Load SEND domains
        logger.info("Step 1: Loading SEND domains...")
        # Remove required_domains/optional_domains arguments
        # load_send_study will attempt to load all *.xpt files found
        loaded_domains = load_send_study(study_dir)
        # Check can happen later after parsing
        # required_domains = ['dm', 'ex', 'ts', 'bw']
        # optional_domains = ['tx'] # Example
        if not loaded_domains:
            # This case likely means the directory was empty or unreadable
            raise HTTPException(
                status_code=500,
                detail="Failed to load any SEND domains. Directory might be empty or inaccessible.",
            )

        # 2. Parse domains
        logger.info("Step 2: Parsing domain data...")
        # Parse all loaded domains
        parsed_data = parse_domains(loaded_domains)
        if not parsed_data or "dm" not in parsed_data or parsed_data["dm"].empty:
            raise ValueError("Parsing failed or essential DM domain missing.")

        # 3. Process for TxGemma (using the real processor now)
        logger.info("Step 3: Processing data and calling LLM for NOAEL assessment...")
        # Call the actual function (no longer async) from the processor module
        result = process_study_for_txgemma(parsed_data, cleaned_study_id)

        logger.info(f"--- TxGemma Analysis Pipeline Finished for Study: {study_id} ---")
        # Return the result dictionary directly as JSON
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        logger.error(f"Data loading error for study {study_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not find necessary files for study {study_id}. Ensure study is uploaded correctly.",
        )
    except ValueError as e:
        logger.error(f"Data processing error for study {study_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing data for study {study_id}: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during TxGemma analysis pipeline for study {study_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/analyze_enhanced/{study_id}")
async def analyze_study_with_enhanced_txgemma(study_id: str):
    """
    Analyzes a previously uploaded SEND study to generate a comprehensive NOAEL assessment
    using TxGemma with enhanced prompting that leverages multiple SEND domains.
    Provides more detailed extraction and structured analysis of findings.
    """
    logger.info(
        f"--- Starting Enhanced TxGemma Analysis Pipeline for Study: {study_id} ---"
    )
    cleaned_study_id = _clean_study_id(study_id)
    study_dir = _get_study_dir(cleaned_study_id)

    if not study_dir.exists() or not any(study_dir.iterdir()):
        logger.error(
            f"Study data directory not found or empty for study_id: {cleaned_study_id} at {study_dir}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Study '{cleaned_study_id}' not found or data is missing. Please upload first.",
        )

    try:
        # 1. Load SEND domains
        logger.info("Step 1: Loading SEND domains...")
        loaded_domains = load_send_study(study_dir)
        if not loaded_domains:
            raise HTTPException(
                status_code=500,
                detail="Failed to load any SEND domains. Directory might be empty or inaccessible.",
            )

        # 2. Parse domains
        logger.info("Step 2: Parsing domain data...")
        parsed_data = parse_domains(loaded_domains)
        if not parsed_data or "dm" not in parsed_data or parsed_data["dm"].empty:
            raise ValueError("Parsing failed or essential DM domain missing.")

        # 3. Process with Enhanced TxGemma processor
        logger.info(
            "Step 3: Processing data with enhanced processor and calling LLM for comprehensive assessment..."
        )
        result = enhanced_process_for_txgemma(parsed_data, cleaned_study_id)

        logger.info(
            f"--- Enhanced TxGemma Analysis Pipeline Finished for Study: {study_id} ---"
        )
        # Return the result dictionary directly as JSON
        return JSONResponse(content=result)

    except FileNotFoundError as e:
        logger.error(f"Data loading error for study {study_id}: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Could not find necessary files for study {study_id}. Ensure study is uploaded correctly.",
        )
    except ValueError as e:
        logger.error(f"Data processing error for study {study_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error processing data for study {study_id}: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error during Enhanced TxGemma analysis pipeline for study {study_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.get("/")
async def read_root():
    """Provides a simple welcome message for the API root."""
    return {"message": "Welcome to the SEND NOAEL Analysis API (TxGemma Demo)"}
