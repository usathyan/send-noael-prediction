# Project Architecture: SEND NOAEL Prediction (Traditional ML)

This document details the architecture of the SEND NOAEL Prediction tool, which utilizes a traditional Machine Learning (ML) approach to predict the No Observed Adverse Effect Level (NOAEL) from preclinical toxicology studies submitted in the SEND format.

## 1. Overview

The primary goal of this system is to automate the prediction of NOAEL values based on the complex datasets provided in SEND format. Users upload study data as a Zip archive, and the backend API processes this data through a pipeline involving loading, validation, parsing, feature engineering, and finally, prediction using a pre-trained ML model.

## 2. Architectural Pivot: From TxGemma to Traditional ML

This project initially employed the TxGemma Large Language Model (`google/txgemma-2b-predict`) with the hypothesis that it could infer the NOAEL from a text-based summary of the study findings. The workflow involved:

1.  Parsing SEND domains (DM, EX, LB, TS).
2.  Generating a structured text prompt summarizing demographics, exposure, key lab findings (e.g., changes in ALT, AST), and study design.
3.  Sending this prompt to the TxGemma model via the `transformers` library.
4.  Attempting to parse a predicted NOAEL value (numerical value and units) from the model's free-text response.

**Challenges and Rationale for Pivot:**

*   **Inconsistent Output:** The TxGemma model frequently failed to return a parsable numerical NOAEL value, often outputting single letters, SMILES strings (in sanity checks), or generic text unrelated to the quantitative task.
*   **Model Suitability:** Further investigation into the TxGemma model's documentation and examples revealed its primary focus on text generation, classification, and information extraction related to clinical trial text, rather than direct quantitative regression from structured or semi-structured data summaries.
*   **Prompt Engineering Complexity:** Reliably prompting the LLM to perform the desired quantitative prediction proved difficult and sensitive to minor variations in the input summary.

Consequently, the architecture was pivoted to a more conventional and reliable approach using **supervised machine learning**, where explicit numerical features are engineered from the SEND data to train a model (like XGBoost) specifically for the regression task of predicting the NOAEL value.

## 3. Current ML Architecture Components

The system is built around a Python backend using the FastAPI framework.

*   **`python.api.main` (FastAPI Application):**
    *   Defines API endpoints (`/upload`, `/predict/{study_id}`).
    *   Handles HTTP requests and responses.
    *   Orchestrates the data processing and prediction pipeline.
    *   Uses Pydantic models (`UploadResponse`, `PredictionResponseML`, `NoaelResultML`) for request/response validation and serialization.
    *   Loads the pre-trained ML model into memory (`PRELOADED_ML_MODEL`) on startup for efficiency, falling back gracefully if the model file is not found.

*   **`python.data_processing.send_loader`:**
    *   `load_send_study()`: Reads `.xpt` files (using `pandas.read_sas`) for specified SEND domains (DM, EX, LB, TS) from a given study directory into pandas DataFrames.
    *   `validate_send_domains()`: Performs basic checks to ensure essential domains were loaded.

*   **`python.data_processing.domain_parser`:**
    *   `parse_study_data()`: Takes the dictionary of raw domain DataFrames.
    *   Processes each domain DataFrame to extract relevant information, potentially cleaning or structuring it slightly (e.g., converting types, filtering rows). Currently focuses on DM, EX, LB, TS.
    *   Returns a dictionary containing processed data, often still as DataFrames, ready for feature engineering.

*   **`python.data_processing.feature_extractor`:**
    *   `extract_features()`: The core function that bridges raw data and the ML model.
    *   Calls `preprocess_for_ml()`, which performs the heavy lifting:
        *   **Merging:** Combines data from different domains (e.g., DM and EX) based on subject IDs (`USUBJID`).
        *   **Encoding:** Converts categorical features (e.g., `SEX`, `ARM`, `EXTRT`, `EXROUTE`, `LBTESTCD`) into numerical representations using `sklearn.preprocessing.LabelEncoder`. Stores mappings for potential inverse transformation or interpretation.
        *   **Aggregation:** Aggregates findings data (e.g., LB domain) per study or potentially per dose group/subject if needed later. Calculates summary statistics (mean, max, min, std) for key parameters like ALT, AST, potentially filtering by time points or visit numbers.
        *   **Feature Creation:** Derives new features, such as calculating `MAX_DOSE` from the EX domain, extracting planned duration (`TSVAL` where `TSPARAMCD == 'PLANDUR'`) from TS.
        *   **Missing Value Imputation:** Handles missing numerical values using `sklearn.impute.SimpleImputer` (e.g., with mean or median strategy).
        *   **Output:** Returns a single-row pandas DataFrame where columns represent the engineered numerical features for the *entire study*. (Note: For a more granular model, this might return features per dose group).

*   **`python.model.ml_predictor`:**
    *   **Model Storage:** Defines the location (`MODEL_DIR`, `MODEL_PATH`) for storing the serialized, pre-trained model.
    *   `load_pretrained_model()`: Uses `joblib.load()` to deserialize and load the trained model object (e.g., an `xgboost.XGBRegressor` instance) from the `.joblib` file specified by `MODEL_PATH`. Includes error handling if the file is missing.
    *   `predict_noael_ml()`: Takes the feature DataFrame (output from `extract_features`) and the loaded model object.
        *   Ensures the columns in the input DataFrame match the features the model was trained on (implicitly handled by XGBoost/Scikit-learn if column names are consistent).
        *   Calls the model's `predict()` method on the feature data.
        *   Returns the single predicted NOAEL value (as a float).
        *   Includes fallback logic to return a dummy value (e.g., 0.0) if no model was loaded.
    *   **(Training - Not part of API runtime):** Includes an `if __name__ == '__main__':` block demonstrating how to train a *dummy* XGBoost model (`xgboost.XGBRegressor`) on sample data and save it using `joblib.dump()`. This is used to create the initial model file for the API to load.

## 4. Data Flow (Prediction Endpoint)

A typical request to `/predict/{study_id}` follows these steps:

1.  **Request:** FastAPI receives a POST request to `/predict/{study_id}`.
2.  **Path Resolution:** The `study_id` is used to locate the study directory in `uploaded_studies/`.
3.  **Data Loading:** `load_send_study()` reads XPT files into DataFrames.
4.  **Validation:** `validate_send_domains()` checks if required domains loaded.
5.  **Parsing:** `parse_study_data()` processes the raw DataFrames.
6.  **Feature Engineering:** `extract_features()` converts the parsed data into a single row DataFrame of numerical features.
7.  **Prediction:** `predict_noael_ml()` is called with the feature DataFrame and the `PRELOADED_ML_MODEL`.
    *   The `.predict()` method of the loaded XGBoost model is invoked on the feature data.
8.  **Response Formatting:** The predicted numerical value is packaged into the `NoaelResultML` Pydantic model.
9.  **Response:** FastAPI sends the `PredictionResponseML` (containing `NoaelResultML`) back to the client as a JSON response.

## 5. Key Technologies

*   **Python 3.10+**
*   **FastAPI:** Web framework for the API.
*   **Uvicorn:** ASGI server to run FastAPI.
*   **pandas:** Core library for data manipulation (reading XPT, DataFrame operations).
*   **scikit-learn:** Used for data preprocessing (LabelEncoder, SimpleImputer).
*   **xgboost:** Machine learning library providing the Gradient Boosting model used for prediction.
*   **joblib:** For serializing/deserializing the trained Python model objects.
*   **pyreadstat:** (Dependency of pandas SAS reader) For reading `.xpt` files.
*   **uv:** Package manager for installing and managing dependencies in the virtual environment.
*   **(macOS)** **libomp:** Required runtime dependency for XGBoost. 