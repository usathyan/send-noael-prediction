# SEND NOAEL Analysis API (LLM Demo)

## Overview

This project provides a focused backend API demo showcasing the capabilities of Large Language Models (LLMs) like Google's Gemini family (accessed via OpenRouter) for analyzing preclinical toxicology studies (SEND format) to assist in NOAEL (No Observed Adverse Effect Level) assessment.

**Goal:** Demonstrate how an LLM can process summarized findings from a SEND study (currently focusing on Body Weight changes) and provide both a potential NOAEL assessment and the reasoning behind it, aligning with the goal of explainable AI in therapeutic development.

**Note:** This repository represents the final state of a project that evolved significantly. See the "Project History" section below for details on previous approaches involving direct TxGemma inference and traditional Machine Learning models.

## Features

*   **Upload SEND Study:** Accepts a Zip archive containing study `.xpt` files via a REST API endpoint (`/upload/`).
*   **Data Parsing:** Parses key SEND domains required for the analysis (currently DM, EX, TS, BW).
*   **Finding Summarization:** Analyzes specific findings (currently Body Weight changes over time compared to controls) and generates a structured natural language prompt.
*   **LLM Interaction (via OpenRouter):** Sends the generated prompt to a configured LLM (e.g., `google/gemini-2.5-pro-exp-03-25:free`) using the OpenAI SDK configured for the OpenRouter API.
*   **Reasoned Output:** Returns the LLM's response, which ideally includes a NOAEL assessment and the reasoning based on the provided summary (`/analyze_noael/{study_id}`).
*   **FastAPI Backend:** Simple and efficient API built with FastAPI.

## Project History & Evolution

This project underwent several iterations, exploring different approaches to NOAEL prediction from SEND data:

1.  **Initial TxGemma Attempt (Local Inference):** The project initially aimed to use the TxGemma model (e.g., `txgemma-2b`) directly via the Hugging Face `transformers` library. The goal was for the LLM to infer the NOAEL from a generated text summary. This faced challenges related to:
    *   The model's primary capability being text generation, not quantitative regression.
    *   Inconsistent and often unparsable output formats.
    *   The complexity of representing structured SEND data effectively in a text prompt for reliable numerical prediction.
    *   Significant compute resource requirements for local inference.

2.  **Pivot to Traditional Machine Learning:** Due to the difficulties with direct LLM prediction, the project pivoted to a traditional ML approach. This involved:
    *   Extensive feature engineering (`feature_extractor.py`) to create numerical/categorical vectors from SEND domains (DM, EX, LB, BW, etc.).
    *   Training an XGBoost model (`ml_predictor.py`) on these features (using a placeholder model trained on random data for pipeline testing).
    *   Building API endpoints to serve predictions from this ML model.

3.  **Refocus on LLM Reasoning (API Demo - Current State):** Recognizing the strengths of LLMs in natural language understanding and reasoning, the project was refocused again. The goal shifted from *direct prediction* by the LLM to *demonstrating its ability to assist in NOAEL assessment by reasoning over summarized findings*. This led to the current architecture:
    *   Removal of the traditional ML pipeline and frontend.
    *   Simplification of data processing to focus on generating a concise, informative text summary of key findings (starting with Body Weight).
    *   Using an external LLM API (via OpenRouter) for accessibility and access to powerful models without local hardware constraints.
    *   The output now emphasizes the LLM's textual response and reasoning, rather than just a single numerical prediction.

This evolution highlights the different ways AI models can be applied to scientific problems and the importance of matching the model's capabilities (text generation vs. numerical prediction) to the specific task.

## Project Structure

```
SEND_NOAEL_Prediction/
├── .venv/                  # Virtual environment (created by uv)
├── src/                    # Main Python source code
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py         # FastAPI application, endpoints
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── send_loader.py  # Loads .xpt files from study dir
│   │   └── domain_parser.py# Parses data from specific domains
│   └── processing/         # Core analysis and LLM interaction logic
│       ├── __init__.py
│       └── noael_processor.py # Performs analysis, generates prompt, calls LLM
├── data/                   # Optional: Location for sample SEND zip files (see Setup)
│   └── external/phuse-scripts/data/send/
│       └── Vaccine-Study-1.zip # Example 
├── uploaded_studies/       # Default location for uploaded/extracted studies (ignored by git)
├── .env.example            # Example environment variables file
├── .gitignore
├── Architecture.md         # System architecture description
├── Makefile                # Convenience commands (install, run, lint)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SEND_NOAEL_Prediction
    ```

2.  **Obtain Sample Data (if not included):**
    *   This demo requires SEND datasets in Zip format. Example datasets (like `Vaccine-Study-1.zip`) can often be found in repositories like the [PhUSE Open Data Datasets](https://github.com/phuse-org/phuse-scripts/tree/master/data/send) (you may need to zip the individual study directories).
    *   Place the Zip file(s) in a location accessible for uploading (e.g., `data/external/phuse-scripts/data/send/`).

3.  **Install Prerequisites:**
    *   **Python 3.10+:** Ensure you have a compatible Python version.
    *   **uv:** Install the `uv` package manager (`pip install uv`).

4.  **Configure LLM Access (OpenRouter):**
    *   Create an OpenRouter account and generate an API key: [https://openrouter.ai/keys](https://openrouter.ai/keys)
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your OpenRouter API key:
        ```dotenv
        # .env
        OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY_HERE
        # Optional: Specify model - check OpenRouter for availability/pricing
        LLM_MODEL_NAME=google/gemini-2.5-pro-exp-03-25:free 
        # Optional: For OpenRouter ranking/identification
        # OPENROUTER_SITE_URL=https://your-site-url.com 
        # OPENROUTER_SITE_NAME=Your Cool App Name
        ```
    *   **Security:** Ensure the `.env` file is included in your `.gitignore` (it should be by default) and **never commit your API key** to version control.

5.  **Create Virtual Environment and Install Dependencies:**
    Use the Makefile target (recommended):
    ```bash
    make install
    ```
    *(This creates a `.venv` directory and installs packages from `requirements.txt` using `uv`)*

## Running the API Server

Use the Makefile target:

```bash
make run
```

The API will be available at `http://127.0.0.1:8000`.
API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

## Usage Flow

1.  **Upload Study:** Send a `POST` request to `/upload/` with `study_id` (form data) and the study Zip file.
    *   Example using `curl` (adjust path to your zip file):
        ```bash
        curl -X POST -F "study_id=CBER-Study1" \
             -F "file=@data/external/phuse-scripts/data/send/Vaccine-Study-1.zip" \
             http://127.0.0.1:8000/upload/
        ```
2.  **Analyze Study:** Send a `POST` request to `/analyze_noael/{study_id}` (using the same `study_id` provided during upload).
    *   Example using `curl`:
        ```bash
        curl -X POST http://127.0.0.1:8000/analyze_noael/CBER-Study1
        ```
3.  **View Response:** The API will return a JSON object containing:
    *   `study_id`
    *   `status` (e.g., "Analysis Successful", "Error")
    *   `analysis_type` (e.g., "Body Weight")
    *   `llm_prompt` (The actual text sent to the LLM)
    *   `llm_response` (The response received from the LLM)
    *   `error` (Details if an error occurred)

## Current Analysis Strategy (Body Weight)

The current implementation in `noael_processor.py` focuses on:
1.  Loading DM, EX, TS, BW domains.
2.  Identifying dose groups from EX.
3.  Calculating percentage body weight change from baseline for each subject.
4.  Aggregating mean changes per dose group at key time points (e.g., study end).
5.  Generating a structured text prompt summarizing study metadata and these BW findings.
6.  Sending the prompt to the configured LLM via OpenRouter.

## Future Work / Improvements

*   Implement analysis for other key endpoints (Lab Findings, Microscopic Pathology, Clinical Observations).
*   Refine prompt engineering for better LLM reasoning.
*   Add more sophisticated statistical analysis to the finding summaries.
*   Improve error handling and reporting.
*   Add unit and integration tests.
*   Add docstrings to functions/modules.

