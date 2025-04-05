# SEND NOAEL Prediction API (Traditional ML)

# My Notes

This project was insipred by a Hackathon, to predict NOAEL from SEND datasets. 
I took it as an opportunity to see if we can completely develop this using AI tools available. 

Here is How I did this:
1. [Manus](https://manus.im/share/CDv3aokpoeyRxSpYVeDa7V?replay=1), helped me with creating a plan. See [details](docs/manus/comprehensive_report.md). There are more documents in the docs/manus folder.
2. I then fed the entire documents to [Cursor](https://www.cursor.com/) to develop it.
3. After several iterations, and tweaks, and some handholding, here is the result: It is a working version, and [screenshots](docs/screenshots.md).
4. Also take a look at [Architecture](Architecture.md), which provides details on why TxGemma did not work.

This project provides a FastAPI backend to predict the No Observed Adverse Effect Level (NOAEL) from preclinical toxicology studies submitted in the Standard for Exchange of Nonclinical Data (SEND) format, using a traditional Machine Learning (ML) model.

## Overview

The system allows users to upload SEND datasets (as zip archives containing `.xpt` files). The backend processes these datasets, extracts relevant features, and uses a pre-trained ML model (e.g., XGBoost) to predict the NOAEL.

**Previous Approach (Deprecated):** This project initially used the TxGemma language model for prediction but pivoted to a traditional ML approach due to limitations in the model's ability to perform quantitative predictions directly from text summaries.

## Features

*   Upload SEND study data via a Zip archive.
*   Validate basic SEND domain presence.
*   Parse key domains (Demographics, Exposure, Findings - LB, TS).
*   Extract numerical features suitable for ML modeling.
*   Predict NOAEL using a pre-trained ML model (currently uses a dummy model).
*   FastAPI backend with Swagger UI documentation (`/docs`).

## Project Structure

```
SEND_NOAEL_Prediction/
├── .venv/                  # Virtual environment (created by uv)
├── python/
│   ├── api/
│   │   └── main.py         # FastAPI application, endpoints
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── send_loader.py  # Loads .xpt files from study dir
│   │   ├── domain_parser.py# Parses data from specific domains
│   │   └── feature_extractor.py # Extracts numerical features for ML
│   └── model/
│       ├── __init__.py
│       ├── saved_models/   # Directory for trained model files
│       │   └── noael_xgboost_model.joblib # (Dummy) Pre-trained model
│       └── ml_predictor.py # Loads model and performs prediction
├── uploaded_studies/       # Default location for uploaded/extracted studies
├── .gitignore
├── Architecture.md         # Detailed architecture description (Needs Update)
├── Makefile                # Convenience commands (install, run)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SEND_NOAEL_Prediction
    ```

2.  **Initialize Git:**
    ```bash
    git init
    ```

3.  **Install Prerequisites:**
    *   **Python 3.10+:** Ensure you have a compatible Python version.
    *   **uv:** Install the `uv` package manager if you don't have it (`pip install uv`).
    *   **(macOS) Homebrew:** Needed to install `libomp`.
    *   **(macOS) libomp:** Install the OpenMP runtime required by XGBoost:
        ```bash
        brew install libomp
        ```

4.  **Create Virtual Environment and Install Dependencies:**
    Use the Makefile target (recommended) or run manually:
    *   **Using Makefile:**
        ```bash
        make install
        ```
    *   **Manual:**
        ```bash
        uv venv --python python3 # Or specify your python3.x version 
        source .venv/bin/activate # Or .venv\Scripts\activate on Windows
        uv pip install -r requirements.txt
        ```

5.  **Generate Dummy Model (if needed):**
    The `requirements.txt` install should include `xgboost` and `scikit-learn`. A dummy model file (`noael_xgboost_model.joblib`) is needed for the API to run. If it doesn't exist, generate it:
    ```bash
    .venv/bin/python3 python/model/ml_predictor.py
    ```
    *(This script needs to be run only once to create the file if it's missing)*

## Running the API Server (Backend)

Use the Makefile target (recommended) or run manually:

*   **Using Makefile (from project root):**
    ```bash
    make run-backend
    ```
*   **Manual (from project root):**
    ```bash
    source .venv/bin/activate # If not already active
    uvicorn python.api.main:app --reload --host 127.0.0.1 --port 8000
    ```

The API will be available at `http://127.0.0.1:8000`.
API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

Keep this server running while using the frontend.

## Frontend (Next.js)

This project includes a basic Next.js frontend in the `frontend/` directory to interact with the backend API.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:** (Use `npm` or `yarn` based on your project setup)
    ```bash
    npm install 
    # or
    # yarn install
    ```

### Running the Frontend Development Server

1.  **Prerequisite:** Ensure the backend server is already running (see Backend section above).
2.  From the `frontend/` directory, start the Next.js development server:
    ```bash
    npm run dev
    # or
    # yarn dev
    ```
3.  Open your browser to `http://localhost:3000` (or the port indicated in the terminal).

### Frontend Configuration

The frontend needs to know the backend API URL. It currently makes requests directly to `http://127.0.0.1:8000` (as seen in `frontend/src/app/import/page.tsx`). If you change the backend port or deploy it elsewhere, you'll need to update these URLs. Using environment variables (`.env.local`) for this is recommended for more complex setups.

## Running Backend and Frontend Together

1.  Open your first terminal window, navigate to the project root (`SEND_NOAEL_Prediction/`), and start the backend:
    ```bash
    make run-backend
    ```
2.  Open a second terminal window, navigate to the frontend directory (`SEND_NOAEL_Prediction/frontend/`), and start the frontend:
    ```bash
    npm run dev 
    # or yarn dev
    ```
3.  Access the application in your browser at `http://localhost:3000`.

## Example Prediction

After uploading a study (e.g., `Vaccine-Study-1.zip`) via the `/upload/` endpoint (using the frontend's "Import Data" page or the backend's `/docs` UI), a successful call to `/predict/Vaccine-Study-1` using the current backend (with the dummy ML model) will return a response similar to this:

```json
{
  "study_id": "Vaccine-Study-1",
  "noael_result": {
    "predicted_noael": 51.5804443359375,
    "units": "mg/kg/day",
    "model_used": "XGBoost",
    "status": "Prediction successful"
  },
  "confidence": null,
  "error": null
}
```

**Note:** The `predicted_noael` value is from the *dummy* model and is not pharmacologically meaningful until a real model is trained.

## Future Work / Improvements

*   **Train a real ML Model:** Collect a labeled dataset (SEND studies + NOAELs) and train the XGBoost (or other) model.
*   **Refine Feature Engineering:** Improve the feature extraction process in `feature_extractor.py` based on domain knowledge and model performance.
*   **Model Evaluation:** Implement proper model evaluation metrics.
*   **Confidence Scores:** Develop a method to estimate confidence in the ML predictions.
*   **Error Handling:** Enhance error handling and validation.
*   **Frontend:** Develop a user interface (e.g., using Next.js) for easier interaction.
*   **Configuration:** Make model paths, upload directories, etc., configurable.
*   **Testing:** Add unit and integration tests. 