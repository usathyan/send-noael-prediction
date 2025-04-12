# Architecture: SEND NOAEL Analysis API (LLM Demo)

## 1. Overview

This document describes the architecture of the refocused SEND NOAEL Analysis API. The primary goal is to provide a streamlined backend service demonstrating the use of a Large Language Model (LLM), such as TxGemma (or compatible Gemini models), to analyze summarized findings from SEND studies and assist in NOAEL assessment.

This version intentionally removes the complexities of a traditional ML prediction pipeline and a dedicated frontend UI to focus on the core LLM interaction workflow.

## 2. Core Components

The system consists of the following main Python components:

1.  **FastAPI Application (`src/api/main.py`):**
    *   Provides the main REST API interface.
    *   Handles incoming HTTP requests.
    *   Defines endpoints for study upload (`/upload/`) and analysis (`/analyze_noael/{study_id}`).
    *   Orchestrates the overall workflow by calling data loading, parsing, and processing modules.
    *   Manages basic configuration (logging, environment variable loading via `python-dotenv`).

2.  **Data Loading (`src/data_processing/send_loader.py`):**
    *   Responsible for locating and reading SEND dataset files (`.xpt`) from a specified study directory.
    *   Uses the `pyreadstat` library to parse `.xpt` files into pandas DataFrames.
    *   Handles basic error checking during file loading.

3.  **Domain Parsing (`src/data_processing/domain_parser.py`):**
    *   Takes raw domain DataFrames (from `send_loader`) as input.
    *   Performs initial cleaning, validation, and extracts key information relevant to the downstream analysis (e.g., standardizing column names, basic type conversions).
    *   Currently focuses on domains needed for the target analysis (DM, EX, TS, BW).

4.  **NOAEL Processor (`src/processing/noael_processor.py`):**
    *   **This is the core logic module for the demo.**
    *   Receives parsed domain data (DM, EX, TS, BW, CL, LB, MA, MI, OM).
    *   **Analysis:** Implements the specific analysis strategy for relevant domains (currently focusing on Body Weight changes, basic summaries for CL, LB, MA, MI, OM).
    *   **Summarization:** Creates a `comprehensive_findings_summary` string by combining summaries from BW and other available domains.
    *   **Prompt Generation:** Constructs a detailed, structured natural language prompt including study metadata and the `comprehensive_findings_summary`.
    *   **LLM Interaction:** 
        *   Retrieves Friendli API token (`FRIENDLI_TOKEN`) from environment variables.
        *   Uses the `requests` library to make a POST request to the Friendli API endpoint (`https://api.friendli.ai/dedicated/v1/chat/completions`).
        *   Sends the generated prompt and necessary parameters (model ID `2c137my37hew`, `max_tokens`, `stream`, etc.) in the payload.
        *   Handles the streamed response from the LLM, including potential errors.
    *   Returns the analysis results, the prompt sent, and the LLM response.

5.  **Configuration (`.env` file):**
    *   Stores sensitive information like the Friendli API token (`FRIENDLI_TOKEN`).
    *   Note: The target LLM model ID and Friendli API URL are currently hardcoded in `src/processing/noael_processor.py`.
    *   Loaded at application startup using `python-dotenv`.

## 3. Data Flow (Analyze Endpoint)

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as main.py
    participant Loader as send_loader.py
    participant Parser as domain_parser.py
    participant Processor as noael_processor.py
    participant FriendliAPI

    Client->>+FastAPI: POST /analyze_noael/{study_id}
    FastAPI->>FastAPI: Validate study_id, locate study dir
    FastAPI->>+Loader: load_send_study(study_dir)
    Loader-->>-FastAPI: domain_dataframes (DM, EX, TS, BW, CL, LB, etc.)
    FastAPI->>+Parser: parse_domains(domain_dataframes)
    Parser-->>-FastAPI: parsed_data
    FastAPI->>+Processor: process_study_for_txgemma(parsed_data, study_id)
    Processor->>Processor: Analyze BW, Summarize other domains (CL, LB, etc.)
    Processor->>Processor: Generate comprehensive_findings_summary
    Processor->>Processor: Generate llm_prompt
    Processor->>Processor: Retrieve FRIENDLI_TOKEN from env
    Processor->>+FriendliAPI: POST /dedicated/v1/chat/completions (prompt, token)
    FriendliAPI-->>-Processor: Streamed LLM Response
    Processor->>Processor: Concatenate stream, handle errors
    Processor-->>-FastAPI: result_dict (summary, prompt, response, status)
    FastAPI->>FastAPI: Format JSON response
    FastAPI-->>-Client: JSON Response
```

## 4. Key Technologies

*   **Python:** Core programming language.
*   **FastAPI:** Web framework for building the API.
*   **Uvicorn:** ASGI server to run the FastAPI application.
*   **Pandas:** Data manipulation and analysis (for handling SEND domain data).
*   **Pyreadstat:** Reading SAS `.xpt` files.
*   **requests:** Used for making HTTP calls to the Friendli API.
*   **python-dotenv:** Loading environment variables for configuration.
*   **uv:** Package and environment management.

## 5. Simplifications from Previous Architecture

*   **No Frontend:** The dedicated Next.js frontend has been removed.
*   **No Traditional ML Model:** The XGBoost model, associated feature extraction (`feature_extractor.py`), model loading (`ml_predictor.py`), and placeholder model generation have been removed.
*   **No Complex Numerical Features:** The focus shifts from generating a large vector of numerical features to generating a textual summary for the LLM.
*   **Simplified API:** Fewer endpoints, focused solely on upload and LLM-based analysis.
*   **Consolidated Demo Logic:** The separate `txgemma_demos` structure is removed; the core analysis/LLM logic resides in the new `processing` module. 