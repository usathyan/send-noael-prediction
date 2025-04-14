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
*   **Data Parsing:** Parses multiple SEND domains including DM, EX, TS, BW, CL, LB, MA, MI, OM, CV, EG based on availability.
*   **Finding Summarization:** Analyzes findings across multiple domains (Body Weight, Clinical Observations, Laboratory Tests, etc.) and generates structured natural language prompts.
*   **Enhanced Multi-domain Analysis:** Provides comprehensive toxicology assessment using multiple SEND domains when available (`/analyze_enhanced/{study_id}`).
*   **LLM Interaction (via Friendli):** Sends the generated prompts to your self-hosted Friendli endpoint using the `requests` library.
*   **Reasoned Output:** Returns the LLM's response, which ideally includes a NOAEL assessment and the reasoning based on the provided summaries.
*   **Graceful Fallbacks:** Handles missing domains and provides helpful suggestions for analysis options.
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
│       ├── noael_processor.py # Basic analysis focusing on Body Weight
│       └── enhanced_processor.py # Enhanced multi-domain analysis
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
2.  **Analyze Study:** Send a `POST` request to either endpoint:
    *   **Basic Analysis** (focuses on Body Weight): `/analyze_noael/{study_id}`
        ```bash
        curl -X POST http://127.0.0.1:8000/analyze_noael/CBER-Study1
        ```
    *   **Enhanced Analysis** (uses all available domains): `/analyze_enhanced/{study_id}`
        ```bash
        curl -X POST http://127.0.0.1:8000/analyze_enhanced/CBER-Study1
        ```
3.  **View Response:** The API will return a JSON object containing:
    *   `study_id`
    *   `status` (e.g., "Analysis Successful", "Analysis Failed")
    *   `analysis_type` (e.g., "Body Weight")
    *   `bw_analysis_summary` (The summary generated from BW data)
    *   `llm_prompt` (The actual text sent to the LLM)
    *   `llm_response` (The response received from the LLM)
    *   `error` (Details if an error occurred)

## Sample Responses
<details>
<summary>Example Response Details (`/analyze_noael`)</summary>

- **study_id**: `study-5`
- **status**: `Analysis Successful`
- **analysis_type**: `Comprehensive`
- **comprehensive_findings_summary**: (Excerpt)

- Control Group (0.00 mg): Mean terminal BW change: 1.60%
- Dose Group (20.00 mg): Mean terminal BW change: -3.05%
- Dose Group (50.00 mg): Mean terminal BW change: 0.00%
        
- **llm_prompt**: (Excerpt)

    Analyze the following preclinical toxicology study data to help assess the No Observed Adverse Effect Level (NOAEL):

    Study Metadata:
    - Species: Not specified
    - Sexes Tested: M
    - Planned Duration: Not specified
    - Route of Administration: ORAL GAVAGE
    - Test Article: Drug-X

    Comprehensive Findings Summary:
    - Control Group (0.00 mg): Mean terminal BW change: 1.60%
    - Dose Group (20.00 mg): Mean terminal BW change: -3.05%
    - Dose Group (50.00 mg): Mean terminal BW change: 0.00%
    - Dose Group (150.00 mg): Mean terminal BW change: 6.13%
            
    - **llm_response**: (Excerpt)

        ## Analysis of Preclinical Toxicology Study Data for Drug-X

        **1. Key Toxicological Findings:**

        * **Body Weight Change:** The two control groups (0.00 mg and 20.00 mg) show comparable mean terminal body weight (BW) changes (1.60% and -3.05% respectively), suggesting no significant impact of vehicle or low doses of Drug-X on BW. The 50.00 mg group exhibits a negligible BW change (0.00%), while the highest dose (150.00 mg) reveals a significant positive change of 6.13%. This suggests potential dose-dependent effects of Drug-X on BW, with the highest dose exhibiting a notable increase.

        **2. Overall Toxicological Assessment:**

        Based on the limited available data, the 50.00 mg dose of Drug-X appears to be the most plausible candidate for the NOAEL. This dose shows no statistically significant difference in BW change compared to the control group, while higher doses exhibit statistically significant deviations. However, a definitive conclusion cannot be drawn without additional data and statistical analysis.
            
    - **error**: `null`

</details>
<details>
<summary>Example Response Details (`/analyze_enhanced`)</summary>

- **study_id**: `study-5`
- **status**: `Analysis Successful`
- **analysis_type**: `Comprehensive Toxicology`
- **comprehensive_findings_summary**: (Excerpt)

    ## Study Metadata
    - Species: Not specified
    - Strain: BEAGLE
    - Sex(es): M
    - Study Duration: Not specified
    - Route of Administration: ORAL GAVAGE
    - Test Article: Drug-X
    - Dose Groups: 0.00 mg, 20.00 mg, 50.00 mg, 150.00 mg

    ## Body Weight Analysis
    Mean Body Weights (g) by Time Point:
    | Dose Group | Day -7.0 | Day 10.0 | Day 21.0 | Day 35.0 | Day 44.0 |
    |------------|----------|----------|----------|----------|----------|
    | Control (0.00 mg) | 9.8 | 9.6 | 9.6 | 9.9 | 9.7 |
    | 20.00 mg | 8.6 | 8.2 | 8.6 | 8.2 | 8.0 |
    | 50.00 mg | 8.0 | 7.8 | 7.8 | 8.0 | 7.8 |
    | 150.00 mg | 10.0 | 9.8 | 10.0 | 10.1 | 10.2 |

    Body Weight Change (Terminal vs. Baseline):
    - Control (0.00 mg): Mean change: -1.0% (n=2)
    - 20.00 mg: Mean change: -7.0% (n=1)
    - 50.00 mg: Mean change: -1.3% (n=1)
    - 150.00 mg: Mean change: 1.8% (n=2)
                
    - **llm_response**: (Excerpt)

    ## Toxicological Analysis of SEND Dataset 

    **1. Key Toxicological Findings:**

    **Control (0.00 mg):**
    * **Body Weight:** No significant weight change observed.

    **20.00 mg:**
    * **Body Weight:** Significant weight loss observed (17.8% compared to control). 
    * **Body Weight Change:** Significant negative change (-7.0%).

    **50.00 mg:**
    * **Body Weight:**  Significant weight loss observed (18.9% compared to control).
    * **Body Weight Change:** Significant negative change (-1.3%).

    **150.00 mg:**
    * **Body Weight:** No significant weight loss or gain observed (5.2% compared to control).
    * **Body Weight Change:** Mild positive change (1.8%).

    **4. NOAEL Determination:**
    Based on the available data, the **NOAEL is 10.0 mg/kg**. This dose group exhibited no statistically significant body weight loss compared to the control group.
            
- **error**: `null`
</details>

## Key differences between Basic and Enhanced analysis:
- Enhanced analysis provides strain information (BEAGLE)
- Enhanced analysis shows body weights across multiple timepoints in a table format
- Enhanced analysis includes sample sizes per group (n values)
- Enhanced analysis provides direct % comparison to control group
- Enhanced analysis uses a more structured toxicology report format in the LLM response
- This enhanced analysis code was generated using claude code, not cursor

## API Documentation UI (Swagger)

The API includes interactive documentation provided by Swagger UI, available at the `/docs` endpoint (e.g., `http://127.0.0.1:8000/docs`) when the server is running.

## TxGemma Hosting

Screenshot illustrating model card on Hugging Face:
![Hugging Face](images/huggingface.png)

Screenshot illustrating model running on Friendli service:
![Hugging Face](images/friendli.png)

<details>
<summary>Project History & Evolution</summary>
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
</detail>