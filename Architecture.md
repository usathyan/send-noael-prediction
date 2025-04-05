# Project Architecture: SEND NOAEL Prediction (Traditional ML)

This document details the architecture of the SEND NOAEL Prediction tool, which utilizes a traditional Machine Learning (ML) approach to predict the No Observed Adverse Effect Level (NOAEL) from preclinical toxicology studies submitted in the SEND format.

## 1. Overview

The primary goal of this system is to automate the prediction of NOAEL values based on the complex datasets provided in SEND format. Users upload study data as a Zip archive, and the backend API processes this data through a pipeline involving loading, validation, parsing, feature engineering, and finally, prediction using a pre-trained ML model.

## 2. Architectural Pivot: From TxGemma to Traditional ML

This project initially employed the TxGemma Large Language Model (`google/txgemma-2b-predict`) with the hypothesis that it could infer the NOAEL from a text-based summary of the study findings. The initial workflow involved:

1.  Parsing SEND domains (DM, EX, LB, TS).
2.  Generating a structured text prompt summarizing demographics, exposure, key lab findings (e.g., changes in ALT, AST), and study design.
3.  Sending this prompt to the TxGemma model via the `transformers` library.
4.  Attempting to parse a predicted NOAEL value (numerical value and units) from the model's free-text response.

**Detailed Challenges and Rationale for Pivot:**

The attempt to use TxGemma for direct NOAEL prediction encountered significant hurdles:

*   **Task Mismatch (Text Generation vs. Quantitative Regression):** The fundamental issue was using a model optimized for natural language understanding and generation for a quantitative regression task. `txgemma-2b-predict`, while knowledgeable in the biomedical domain, is not inherently designed to perform precise numerical calculations or regressions based on summarized text inputs. Its primary function is text prediction and generation.
*   **Inconsistent and Unparsable Output:** The model rarely produced the desired output format (e.g., a specific number and unit like "50 mg/kg/day"). Instead, outputs were often:
    *   **Non-numerical:** Single letters (e.g., "A"), generic text fragments.
    *   **Irrelevant but Contextual:** SMILES strings during sanity checks (indicating some context understanding but not task execution).
    *   **Unstructured:** Free text that might mention study details but lacked a clearly identifiable or parsable NOAEL value.
    This inconsistency made reliable extraction of a numerical result impossible, leading to frequent "Parsing Failed" states.
*   **Input Representation Difficulty:** Translating structured, tabular SEND data into a textual summary for the LLM loses some numerical precision and relational information inherent in the original tables. Asking the LLM to reason quantitatively based on this secondary textual representation is less direct and potentially more error-prone than using the structured data directly.
*   **Prompt Engineering Complexity:** LLM outputs are highly sensitive to input prompts. Crafting a prompt that could reliably instruct the model to not only understand the summarized study but also perform the specific calculation and return it in the desired format proved extremely difficult and likely would have required extensive experimentation.
*   **Lack of Specific Fine-Tuning:** The base `txgemma-2b-predict` model, while pre-trained on a vast corpus, was not specifically fine-tuned for the task of NOAEL prediction from SEND data summaries. Achieving reliable performance on specialized tasks like quantitative prediction often necessitates fine-tuning on a targeted dataset with examples of the desired input-output behavior.

Consequently, the architecture was pivoted to a more conventional and reliable approach using **supervised machine learning**. This allows explicit numerical and categorical features engineered directly from the structured SEND data to be fed into a model (like XGBoost) specifically designed and trained for the regression task of predicting the numerical NOAEL value.

**Potential LLM Approaches (Alternative Considerations):**

While we pivoted away, making an LLM viable for this task *might* involve:

1.  **Fine-Tuning:** Training a base LLM (like TxGemma or others) on a large dataset of (SEND Summary, NOAEL Value) pairs. This would teach the model the specific task and desired output format.
2.  **Structured Output Prompting:** Using advanced prompting techniques that strongly guide the model to generate output in a specific format (e.g., JSON), although this can still be unreliable for base models not explicitly trained for structured output.
3.  **LLM as Feature Enhancer:** Using the LLM to generate *textual summaries* or *risk assessments* based on the data, which could then be used as *additional features* alongside the numerical data fed into a traditional ML model.
4.  **LLM Function Calling/Agents:** Designing a system where the LLM analyzes the request and potentially calls external tools or functions (which might include traditional ML models or calculators) to determine the NOAEL, rather than calculating it directly. This often requires more sophisticated LLMs with built-in agentic capabilities.

These approaches involve significant effort in data collection, fine-tuning, or complex system design, reinforcing the decision to utilize a traditional ML approach for its directness and reliability for this specific quantitative prediction goal.

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