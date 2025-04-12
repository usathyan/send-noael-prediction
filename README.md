# SEND NOAEL Analysis API Service

## Overview

[TxGemma](https://arxiv.org/abs/2504.06196v1), a new therapeutic Model from Google, provides explainability of predictions and conversational capabilities in natural langauge, abstracting the nuances of interpretations related to predictions, making the predictions accessible and trustworthy (due to transparency of interpretations, see [TDC](https://tdcommons.ai/)) for details.

This is a sample project for demonstrating how to use it for predicting toxicity using SEND datasets.

### Here is what you will see (swagger ui) when you run this demo:

Swagger UI showing Upload and Analyze Endpoints:
![Swagger UI](images/swagger_ui_demo.png)

### Current Analysis Strategy (Body Weight)

The current implementation in `noael_processor.py` focuses on:
1.  Loading DM, EX, TS, BW domains.
2.  Identifying dose groups from EX.
3.  Calculating percentage body weight change from baseline for each subject.
4.  Aggregating mean changes per dose group at key time points (e.g., study end).
5.  Generating a structured text prompt summarizing study metadata and these BW findings.
6.  Sending the prompt to the configured LLM via Friendli API.


**Goal:** Demonstrate how an LLM can process summarized findings from a SEND study (currently focusing on Body Weight changes) and provide both a potential NOAEL assessment and the reasoning behind it, aligning with the goal of explainable AI in therapeutic development. The previous versions and sample python files in the manus folder should give you more representative examples on other predictions to try out.

**Note:** This repository represents the final state of a project that evolved significantly. See the "Project History" section below for details on previous approaches involving direct TxGemma inference and traditional Machine Learning models.

## Features

*   **Upload SEND Study:** Accepts a Zip archive containing study `.xpt` files via a REST API endpoint (`/upload/`). I have a sample selection of sample SEND datasets downloaded from [phuse](https://github.com/phuse-org/SEND-Coding-Bootcamp/tree/main/data/mock_SEND_data), in the data folder. 
*   **Data Parsing:** Parses key SEND domains required for the analysis (currently DM, EX, TS, BW).
*   **Finding Summarization:** Analyzes specific findings (currently Body Weight changes over time compared to controls) and generates a structured natural language prompt.
*   **LLM Interaction (via Friendli):** Sends the generated prompt to your self-hosted Friendli endpoint using the `requests` library.
*   **Reasoned Output:** Returns the LLM's response, which ideally includes a NOAEL assessment and the reasoning based on the provided summary (`/analyze_noael/{study_id}`).
*   **FastAPI Backend:** Simple and efficient API built with FastAPI.

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

4.  **Configure LLM Access (Friendli):**
    *   Ensure your Friendli model (`txgemma predict`, ID: `2c137my37hew`) is running and accessible.
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your Friendli token:
        ```dotenv
        # .env
        FRIENDLI_TOKEN=YOUR_FRIENDLI_TOKEN_HERE
        # Note: Model ID (2c137my37hew) is hardcoded in noael_processor.py
        # Note: Friendli API URL is hardcoded in noael_processor.py
        ```
    *   **Security:** Ensure the `.env` file is included in your `.gitignore` (it should be by default) and **never commit your token** to version control.

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
    *   `status` (e.g., "Analysis Successful", "Analysis Failed")
    *   `analysis_type` (e.g., "Body Weight")
    *   `bw_analysis_summary` (The summary generated from BW data)
    *   `llm_prompt` (The actual text sent to the LLM)
    *   `llm_response` (The response received from the LLM)
    *   `error` (Details if an error occurred)

        **Example Successful Response (TxGemma response based on Sample data provided):**

    *   **study_id:** `study-a`
    *   **status:** `Analysis Successful`
    *   **analysis_type:** `Comprehensive`
    *   **comprehensive_findings_summary:** (Summary generated from BW, CL, LB, etc. - example below)
        > - Control Group (20.00 ug/dose): Mean terminal BW change: 1.82%
        > 
        > --- Clinical Observations (CL) Summary ---
        > Relevant columns (e.g., CLTERM) not found.
        > 
        > --- Laboratory Tests (LB) Summary ---
        > Mean values for key tests:
        > - Control Group (20.00 ug/dose): ALT: 41.12 U/L; AST: 39.00 U/L
    *   **llm_prompt:** (The full prompt text sent to the LLM, including metadata and comprehensive summary)
        > Analyze the following preclinical toxicology study data to help assess the No Observed Adverse Effect Level (NOAEL):
        > 
        > Study Metadata:
        > - Species: Not specified
        > - Sexes Tested: F
        > - Planned Duration: Not specified
        > - Route of Administration: INTRAMUSCULAR
        > - Test Article: Hepatitis B Vaccine
        > 
        > Comprehensive Findings Summary:
        > - Control Group (20.00 ug/dose): Mean terminal BW change: 1.82%
        > 
        > --- Clinical Observations (CL) Summary ---
        > Relevant columns (e.g., CLTERM) not found.
        > 
        > --- Laboratory Tests (LB) Summary ---
        > Mean values for key tests:
        > - Control Group (20.00 ug/dose): ALT: 41.12 U/L; AST: 39.00 U/L
        > 
        > Based on the provided study metadata and comprehensive findings summary:
        > 1. Identify the key toxicology findings suggested by the data.
        > 2. Provide an overall toxicological assessment based on these findings.
        > 3. Assess the characteristics relevant to determining the No Observed Adverse Effect Level (NOAEL).
        > 4. Discuss any limitations in the provided data for making a definitive NOAEL determination.
        > 
        > Please ensure your response specifies the dose units (ug/dose).
    *   **llm_response:** (The full response text from the LLM)
        > ## Analysis of Preclinical Toxicology Study Data
        > 
        > **1. Key Toxicology Findings:**
        > The provided data suggests the following key toxicological findings:
        > 
        > - **Body Weight Change:**  The control group (receiving 20.00 ug/dose) exhibited a mean terminal body weight change of 1.82%. This suggests a potential for weight alteration with the test article. 
        > - **Liver Function Tests:** The control group displayed ALT and AST levels of 41.12 U/L and 39.00 U/L, respectively. These values should be compared to species-specific normal ranges to assess potential liver toxicity.
        > 
        > **2. Overall Toxicological Assessment:**
        > Based on the limited data, it is difficult to provide a definitive overall toxicological assessment.  The observed body weight change warrants further investigation, as it could indicate general toxicity or specific effects related to the target organ (liver?).  The ALT and AST values are within the normal range for the control group, but comparison with species-specific norms is necessary to rule out any subtle liver toxicity.
        > 
        > **3. Characteristics Relevant to NOAEL Determination:**
        > - **Dose Levels:** The data only presents one dose level (20.00 ug/dose). To determine the NOAEL, a range of doses needs to be tested, including doses below and above the observed effect.
        > - **Control Group:**  The control group data serves as a baseline for comparison. However, the absence of "Clinical Observations (CL)" and detailed "Laboratory Tests (LB)" summaries limits the ability to assess the full spectrum of potential adverse effects.
        > 
        > **4. Limitations for NOAEL Determination:**
        > - **Lack of Dose Range:**  The absence of multiple doses makes it impossible to establish a dose-response relationship and pinpoint the NOAEL.
        > - **Incomplete Clinical Data:** The lack of clinical observations and a comprehensive list of laboratory tests hinders a thorough evaluation of potential toxic effects beyond body weight and liver function.
        > - **Species Not Specified:**  Physiological responses to the test article can vary significantly between species. The lack of species information makes it difficult to interpret the data in a broader context.
        > - **Planned Duration Not Specified:** The study duration is unknown, which makes it challenging to assess the potential for delayed or chronic toxicity.
        > 
        > **Conclusion:**
        > The provided data offers limited insight into the toxicological profile of the Hepatitis B Vaccine. To determine the NOAEL, a more comprehensive study is required, including:
        > 
        > * **Multiple dose levels** to establish a dose-response relationship.
        > * **Detailed clinical observations** to monitor for any adverse effects.
        > * **A comprehensive battery of laboratory tests** to assess various organ systems.
        > * **Specification of the test species.**
        > * **Definition of the study duration.** 
        > 
        > Only with this information can a reliable NOAEL determination be made.
    *   **error:** `null`

    *(Note: In this example, the LLM correctly identifies that it cannot determine the NOAEL with only control group data.)*

## API Documentation UI (Swagger)

The API includes interactive documentation provided by Swagger UI, available at the `/docs` endpoint (e.g., `http://127.0.0.1:8000/docs`) when the server is running.

## TxGemma Hosting

Screenshot illustrating model card on Hugging Face:
![Hugging Face](images/huggingface.png)

Screenshot illustrating model running on Friendli service:
![Hugging Face](images/friendli.png)

## Project History & Evolution

This project underwent several iterations, exploring different approaches to NOAEL prediction from SEND data. These are available in previous versions of this git repo:

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
    *   Using your self-hosted Friendli API for the LLM interaction.
    *   The output now emphasizes the LLM's textual response and reasoning, rather than just a single numerical prediction.

This evolution highlights the different ways AI models can be applied to scientific problems and the importance of matching the model's capabilities (text generation vs. numerical prediction) to the specific task.
