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

A typical request to `/analyze_noael/{study_id}` follows this flow:

1.  Request received by FastAPI (`src/api/main.py`).
2.  `main.py` validates the `study_id` and locates the study directory in `uploaded_studies/`.
3.  `main.py` calls `src.data_processing.send_loader.load_send_study` to read required `.xpt` files (DM, EX, TS, BW, and attempts CL, LB, MA, MI, OM) into pandas DataFrames.
4.  `main.py` calls `src.data_processing.domain_parser.parse_domains` to preprocess these DataFrames.
5.  `main.py` calls `src.processing.noael_processor.process_study_for_txgemma`, passing the parsed data.
6.  `src/processing/noael_processor.py` performs Body Weight analysis and basic summarization for other available domains (CL, LB, MA, MI, OM).
7.  `src/processing/noael_processor.py` generates the `comprehensive_findings_summary`.
8.  `src/processing/noael_processor.py` generates the `llm_prompt` string using the comprehensive summary.
9.  `src/processing/noael_processor.py` retrieves Friendli token from environment variables.
10. `src/processing/noael_processor.py` uses the `requests` library to send the prompt to the Friendli LLM API.
11. The Friendli LLM API processes the prompt and returns a streamed response.
12. `src/processing/noael_processor.py` receives and concatenates the `llm_response` text from the stream (or handles errors).
13. `src/processing/noael_processor.py` packages the `comprehensive_findings_summary`, prompt, response, and status into a result dictionary.
14. `main.py` receives the result dictionary.
15. `main.py` formats the result as a JSON response and sends it back to the client.

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