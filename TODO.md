# TODO - ML Path Refinements for SEND NOAEL Prediction

This file outlines potential next steps and areas for improvement in the traditional Machine Learning (ML) path for predicting NOAEL from SEND datasets.

## 1. Feature Engineering (`python/data_processing/feature_extractor.py`)

This is likely the area with the highest impact on model performance.

-   [ ] **Incorporate Body Weight (BW) Data:**
    -   Calculate features representing changes over time (e.g., relative to baseline or day 0).
    -   Calculate features comparing treated groups to control (e.g., mean/max % difference at study end).
    -   Consider normalization by sex if significant differences exist.
-   [ ] **Incorporate Clinical Observations (CL) Data:**
    -   Requires mapping findings (CLTERM) to severity/relevance.
    -   Could generate features like: count of specific adverse observations per dose group, max severity score per dose group.
-   [ ] **Incorporate Microscopic Findings (MI) Data:**
    -   Requires mapping findings (MITERM, MISPEC, MISEV) and potentially linking to organ weights.
    -   Could generate features like: count/incidence of specific lesions per dose group, max severity score for key organs, number of affected organs.
-   [ ] **Incorporate Organ Measurement (OM) Data:**
    -   Calculate relative organ weights (e.g., % of body weight).
    -   Compare relative organ weights between treated and control groups (% change).
-   [ ] **Refine Lab Test (LB) Features:**
    -   Calculate change from baseline for each subject/group.
    -   Calculate % or fold-change relative to control group means.
    -   Feature for % of subjects exceeding a threshold (e.g., >3x ULN for ALT/AST).
    -   Consider different aggregation methods beyond mean/max/std (e.g., median, specific quantiles).
-   [ ] **Time-Based Features:**
    -   For longitudinal data (BW, LB), calculate slopes or analyze specific time windows (e.g., last week of study).
-   [ ] **Feature Selection/Importance:**
    -   After training an initial model, use feature importance scores (e.g., from XGBoost/Random Forest) to identify and potentially remove less predictive features.
    -   Explore automated feature selection techniques.
-   [ ] **Study vs. Dose Group Features:**
    -   Evaluate feasibility/benefit of generating features per *dose group* instead of aggregating everything to a single *study* level vector. This would require changes to model training and prediction logic.

## 2. Target Variable (NOAEL)

-   [ ] **Define Source:** Determine how the actual NOAEL values (the `y` variable for training) will be obtained for the training studies. This requires a curated dataset with known, reliable NOAELs.
-   [ ] **Data Format:** Ensure the training data includes a column with the target NOAEL value linked to the study/subject identifiers used in feature generation.

## 3. Model Training and Evaluation

-   [ ] **Create Training Script/Notebook:** Develop code to:
    -   Load pre-processed features and target NOAELs.
    -   Split data into training and validation/test sets.
    -   Implement cross-validation.
    -   Train the chosen ML model (e.g., XGBoost).
-   [ ] **Model Selection:** Experiment with different regression models (Random Forest, Support Vector Regression, Neural Networks) beyond XGBoost.
-   [ ] **Hyperparameter Tuning:** Optimize model parameters using techniques like GridSearchCV or RandomizedSearchCV.
-   [ ] **Evaluation Metrics:** Define and track relevant metrics (RMSE, MAE, R², potentially custom metrics based on acceptable error ranges in toxicology, e.g., prediction within the correct dose group interval).
-   [ ] **Save Trained Model:** Implement logic to save the best-performing trained model (e.g., using `joblib`) to replace the current dummy model.

## 4. Data Handling

-   [ ] **Missing Value Imputation:** Explore alternatives to simple mean imputation (median, KNNImputer, IterativeImputer) in `feature_extractor.py`.
-   [ ] **Data Scaling/Normalization:** Apply feature scaling (e.g., StandardScaler, MinMaxScaler) as part of the ML pipeline (typically *after* data splitting, often within the training script).
-   [ ] **Data Acquisition:** Gather a sufficiently large and diverse set of SEND studies with known NOAELs for robust model training.

## 5. Code and Pipeline

-   [ ] **Configuration:** Make paths (model, data) configurable (e.g., via environment variables or config files).
-   [ ] **Testing:** Add unit tests for feature engineering functions and integration tests for the prediction pipeline.

## 6. Integrate Live LLM Call into TxGemma NOAEL Demo (`python/txgemma_demos/noael_demo.py`)

Replace the current *simulated* LLM response with a *live call* to an external LLM API (e.g., Google Generative AI / Gemini, OpenAI GPT) using the generated `summary_prompt`.

**Plan:**

1.  **Preparation & Configuration:**
    -   [ ] **Choose LLM Client:** Select the appropriate Python library (e.g., `google-generativeai`, `openai`).
    -   [ ] **Configure Environment:** Set up required environment variables (e.g., `LLM_API_KEY`, `LLM_MODEL_NAME`) and decide on loading mechanism (e.g., `.env` file + `python-dotenv`, direct environment variables).
    -   [ ] **Update Dependencies:** Add chosen client library and `python-dotenv` (if used) to `requirements.txt` and reinstall (`make install`).

2.  **Modify `noael_demo.py` (`run_noael_determination_demo` function):**
    -   [ ] **Import Modules:** Import client library, `os`, etc.
    -   [ ] **Load Configuration:** Read API key/model name from environment variables within the function, include error checks.
    -   [ ] **Instantiate Client:** Create an instance of the LLM client.
    -   [ ] **Make API Call:** Send `summary_prompt` to the LLM API, wrapped in `try...except` for error handling.
    -   [ ] **Store Response:** Assign the received text to `actual_llm_response` (or similar), setting an error message on failure.

3.  **Update Return Dictionary (`noael_demo.py`):**
    -   [ ] Replace `simulated_response` key/value with `actual_llm_response` (or add the new key alongside).

4.  **Modify API Endpoint (`main.py`):**
    -   [ ] Adjust the `POST /predict/{study_id}/txgemma_demos/noael_determination` endpoint if needed to handle the new key name (`actual_llm_response`) in the final JSON output.

5.  **Testing & Documentation:**
    -   [ ] **Set Environment:** Ensure API keys are correctly set before running the backend.
    -   [ ] **Test Endpoint:** Call the demo endpoint and verify a live LLM response (or error) is received.
    -   [ ] **Update README:** Modify the TxGemma Demo section to explain the live call and configuration requirements.

**Considerations:**
-   *Async:* Explore using `async` endpoints and client libraries for better performance.
-   *Client Scope:* Consider client instantiation scope (per-call vs. app startup).
-   *Cost/Limits:* Be aware of API costs and rate limits.
-   *Prompting:* Refine `summary_prompt` for optimal real LLM results. 