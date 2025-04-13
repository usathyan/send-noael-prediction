## 1. Challenges with TxGemma Predict Models

The initial strategy was to use TxGemma for direct NOAEL prediction, but this approach encountered significant hurdles:

*   **Task Mismatch (Text Generation vs. Quantitative Regression):**
    *   The fundamental issue was using a model optimized for natural language understanding and generation (`txgemma-2b-predict`) for a quantitative regression task.
    *   It's not inherently designed for precise numerical calculations or regressions based on summarized text inputs.
*   **Inconsistent and Unparsable Output:**
    *   The model rarely produced the desired output format (e.g., "50 mg/kg/day").
    *   Outputs were often non-numerical, irrelevant (but contextual like SMILES strings), or unstructured free text, making reliable extraction impossible.
*   **Input Representation Difficulty:**
    *   Translating structured SEND data into a textual summary loses numerical precision and relational information.
    *   Asking the LLM to reason quantitatively from this secondary text representation is challenging.
*   **Prompt Engineering Complexity:**
    *   Crafting a prompt to reliably instruct the model to perform the specific calculation and return it in the desired format proved extremely difficult.
*   **Lack of Specific Fine-Tuning:**
    *   The base `txgemma-2b-predict` model was not specifically fine-tuned for NOAEL prediction from SEND data summaries.

### Initial Pivot to Traditional ML

Consequently, the project architecture was *initially* pivoted to a more conventional supervised machine learning approach. This explored:

*   Extensive feature engineering to create numerical/categorical vectors from SEND domains (DM, EX, LB, BW, etc.).
*   Training an XGBoost model on these features (using a placeholder model for pipeline testing).
*   Building API endpoints to serve predictions from this ML model.

*(Note: This traditional ML approach was subsequently replaced by the current LLM-based reasoning approach using an external API, as detailed below and implemented in the main codebase.)*

## 2. Core Components of the Final LLM Demo API

The final implemented system consists of the following main Python components focused on demonstrating LLM reasoning:

### FastAPI Application (`src/api/main.py`)
*   Provides the main REST API interface.
*   Handles incoming HTTP requests.
*   Defines endpoints for study upload (`/upload/`) and analysis (`/analyze_noael/{study_id}`).
*   Orchestrates the overall workflow by calling data loading, parsing, and processing modules.
*   Manages basic configuration (logging, environment variable loading via `python-dotenv`).

### Data Loading (`src/data_processing/send_loader.py`)
*   Responsible for locating and reading SEND dataset files (`.xpt`) from a specified study directory.
*   Uses `pyreadstat` to parse `.xpt` files into pandas DataFrames.
*   Handles basic error checking during file loading.

### Domain Parsing (`src/data_processing/domain_parser.py`)
*   Takes raw domain DataFrames as input.
*   Performs initial cleaning, validation, and extracts key information relevant to the downstream analysis (e.g., standardizing column names, basic type conversions).
*   Focuses on domains needed for the analysis (DM, EX, TS, BW, CL, LB, MA, MI, OM).

### NOAEL Processor (`src/processing/noael_processor.py`)
*   **This is the core logic module for the demo.**
*   Receives parsed domain data.
*   **Analysis & Summarization:** Implements analysis for relevant domains (BW, basic summaries for CL, LB, MA, MI, OM) and creates a `comprehensive_findings_summary`.
*   **Prompt Generation:** Constructs a detailed, structured natural language prompt including study metadata and the comprehensive summary.
*   **LLM Interaction:** 
    *   Retrieves Friendli API token (`FRIENDLI_TOKEN`) from environment variables.
    *   Uses `requests` to call the Friendli API endpoint.
    *   Sends the generated prompt to the configured LLM API.
    *   Handles the streamed response from the LLM, including potential errors.
*   Returns the analysis results (summary, prompt, response, status).

*(The final response structure is shown in the main README.md)*

